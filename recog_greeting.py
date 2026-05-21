#!/usr/bin/env python3
"""Task: recognize a face and greet (or introduce Pella to an unknown).

This is ONE task in Pella's repertoire — fully self-contained, so other
task types (follow_person, fetch, navigate, etc.) can live in their own
modules without sharing state with this one.

It owns:
  * its own frame intake + motion/face/person detection
  * its own recognition worker thread (started on construction)
  * the per-person greeting cooldown (last_greeted dict)
  * the full state machine:
        IDLE -> SEEKING -> RECOGNIZING ->
            (GREETING | INTRODUCING) -> COOLDOWN -> IDLE
  * the enrollment lifecycle (driven by submit_transcript)
  * recovery action queueing
  * zoom-view requests

pella_main creates one of these once and calls tick() each iteration.
The task reports back to pella_main only via TickResult:
  * latest frame + faces for live annotation
  * an optional zoom_request when a greet/introduce fires
  * an optional status event ("known_person", "new_person", "failure",
    "successful") that pella_main can log or surface elsewhere.

pella_main never needs to know what INTRODUCING means; it just forwards
transcripts via submit_transcript() and renders what the task asks.
"""

import os
import re
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Optional

import cv2
import numpy as np

import actions
import face_recognizer
from vision import (
    FACE_DETECT_EVERY, GREET_COOLDOWN, MOTION_COOLDOWN, SEEK_TIMEOUT,
    INTRODUCE_COOLDOWN, ENROLL_TIMEOUT, FACE_IDS_DIR, SHARPNESS_THRESHOLD,
    detect_faces, detect_motion, detect_person, is_face_at_edge,
    sharpness, zoom_crop, recognition_worker,
)


# ── Captured-face display ────────────────────────────────────────────────────
# How long pella_main should pin the greeting/introduction zoom image on
# screen before reverting to the live annotated feed. The task tells the
# shell "show this image for N seconds" — pella_main has no opinion on
# what the image means.
CAPTURE_DISPLAY_SEC = 3.0


# ── Task lifecycle phases ────────────────────────────────────────────────────
IDLE         = "idle"          # waiting for a trigger
SEEKING      = "seeking"       # adjusting pose to find a face
RECOGNIZING  = "recognizing"   # complete face visible; voting on identity
GREETING     = "greeting"      # known person — recovery + say + wiggle
INTRODUCING  = "introducing"   # unknown — ask name; enrollment runs
COOLDOWN     = "cooldown"      # grace period before re-engaging


# ── Status events surfaced via TickResult ────────────────────────────────────
# Reported once per state transition so pella_main (or a future
# task_manager) can react without having to inspect task internals.
STATUS_KNOWN_PERSON   = "known_person"    # identified as a known person
                                          # (whether or not we greeted)
STATUS_NEW_PERSON     = "new_person"      # identified as a new person
                                          # (whether or not we introduced)
STATUS_NOT_RECOGNIZED = "not_recognized"  # face seen but identity inconclusive
                                          # (ambiguous votes, too blurry, …)
STATUS_FAILURE        = "failure"         # physical / sensing failure — never
                                          # got a face to evaluate at all
STATUS_SUCCESSFUL     = "successful"      # greet / intro completed cleanly


# ── Recognition debounce / capture timing ────────────────────────────────────
# At ~6 detection cycles per second, 3 observations yields a decision in ~0.5 s.
RECOG_VOTES_REQUIRED      = 3
RECOG_AGREE_MIN           = 2
RECOG_TIMEOUT_SEC         = 6.0
# look_level is a quick body-tilt back to level. stand_up from sit is a full
# rise (leg unfold + balance) and runs noticeably longer, so its stabilization
# waits out the whole motion before any face is sampled. With the seek pose
# held throughout RECOGNIZING (recovery queued at exit), these durations
# target the time before sampling begins — letting the body settle in the
# seek pose first.
RECOG_STABILIZE_LOOK_SEC  = 0.5
RECOG_STABILIZE_SIT_SEC   = 1.5
# Face position jitters during pose motion; brief detection misses shouldn't
# abort the task.
FACE_LOST_TIMEOUT_SEC     = 3.0
# Total time spent in GREETING before transitioning to COOLDOWN. The actions
# queued by the greeting (recovery + wiggle) usually finish well within this.
GREETING_DURATION_SEC     = 6.0
# Grace period after any interaction before the task re-engages.
INTERACTION_COOLDOWN_SEC  = 5.0


# ── Name parsing for the introducing path ────────────────────────────────────

_NON_NAME_WORDS = {
    # Pronouns
    "i", "me", "my", "mine", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers",
    "it", "its", "we", "us", "our", "they", "them", "their",
    "this", "that", "these", "those",
    # Prepositions / particles
    "for", "to", "from", "with", "of", "in", "on", "at", "by", "about",
    # Auxiliary verbs
    "is", "am", "are", "was", "were", "be",
    "do", "does", "did", "have", "has", "had",
    # Connectives / articles
    "and", "or", "but", "the", "a", "an", "not",
    # Question words
    "what", "where", "when", "who", "why", "how", "which",
    # Interjections / common responses
    "yes", "no", "ok", "okay", "yeah", "nah", "uh", "um", "oh",
    "hi", "hey", "hello", "bye",
    # Polite / imperative words seen in mis-captures
    "please", "thanks", "thank", "sorry",
    "stop", "hold", "wait", "excuse", "tell", "give", "take",
}


def _parse_name(text: str) -> str:
    """Extract a name from a casual reply, ignoring courtesy phrases.

    Returns an empty string when the input doesn't look like a name — e.g.
    contains common non-name words like 'excuse me' or 'hold it for me'.
    """
    t = text.strip()
    t = re.split(r"[.!?]", t, maxsplit=1)[0].strip()
    m = re.search(r"(?:i am|i'm|my name is)\s+(.+)", t, re.IGNORECASE)
    if m:
        t = m.group(1).strip()
    courtesy = (
        r",|\s+and\b|"
        r"\s+(?:nice|glad|pleased|happy)\s+(?:to\s+)?(?:meet|meeting)\b"
    )
    t = re.split(courtesy, t, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    words = re.findall(r"[A-Za-z]+", t)
    if not words:
        return ""
    if any(w.lower() in _NON_NAME_WORDS for w in words):
        return ""
    return " ".join(words)


def _tally_recognition(obs):
    """Return ``(verdict, kind)`` from a sequence of recognition observations.

    Each observation is a name string (known) or None (unknown). Returns
    (name, "known"), (None, "unknown"), or (None, "ambiguous") depending on
    whether RECOG_AGREE_MIN of the observations agree.
    """
    if not obs:
        return None, "ambiguous"
    val, count = Counter(obs).most_common(1)[0]
    if count < RECOG_AGREE_MIN:
        return None, "ambiguous"
    return val, ("known" if val else "unknown")


# ── tick result type ─────────────────────────────────────────────────────────

@dataclass
class DisplayRequest:
    """A generic "pin this image on screen for N seconds" request.

    The task hands pella_main a ready-to-display BGR image (it's already
    drawn whatever overlays it wants — face crop, name label, etc.) and a
    duration. pella_main has no knowledge of what the image represents.
    """
    image: np.ndarray
    duration_sec: float


@dataclass
class TickResult:
    """What pella_main needs after one tick of the task.

    latest_frame / latest_faces let pella_main render the live annotated
    feed (since the task owns frame_queue consumption now).

    display_request fires ONCE on a transition tick where the task wants
    pella_main to show a specific image for a while — e.g. a zoomed face
    with a name label after a successful greeting.

    status_event fires ONCE when the task transitions through a meaningful
    state — pella_main can log it or route it elsewhere (e.g. into a future
    task_manager).
    """
    latest_frame: Optional[np.ndarray]            = None
    latest_faces: list                            = field(default_factory=list)
    display_request: Optional[DisplayRequest]     = None
    status_event: Optional[str]                   = None


# ── The task ─────────────────────────────────────────────────────────────────

class RecogGreetingTask:
    """One persistent recognize-and-greet task.

    Constructed once by pella_main. Owns its own recognition worker thread
    and detection state. Each call to tick() pulls the latest frame, runs
    detection, advances the state machine, and returns a TickResult.
    """

    def __init__(self, frame_queue, action_queue, say_queue, recognizer,
                 stop_event):
        """Wire dependencies and start the recognition worker thread.

        frame_queue / action_queue / say_queue: shared queues filled or drained
                                                 by other organs (eye / limbs /
                                                 mouth).
        recognizer:  the ArcFace recognizer, or None if disabled.
        stop_event:  pella_main's shutdown flag; the recognition worker
                     thread exits when this is set.
        """
        self._frame_queue  = frame_queue
        self._action_queue = action_queue
        self._say_queue    = say_queue
        self._recognizer   = recognizer
        self._stop_event   = stop_event

        # Recognition worker thread (owned by this task).
        self._rec_in:  Queue = Queue(maxsize=1)
        self._rec_out: Queue = Queue(maxsize=1)
        self._rec_thread = None
        if recognizer is not None:
            self._rec_thread = threading.Thread(
                target=recognition_worker,
                args=(recognizer, self._rec_in, self._rec_out, stop_event),
                daemon=True,
            )
            self._rec_thread.start()

        # ── Per-frame sense state (lives across ticks) ───────────────────
        self._last_faces           = []
        self._last_complete_faces  = []
        self._last_rec_faces       = []   # bundle from worker
        self._last_rec_names       = []
        self._detect_counter       = 0
        self._prev_frame           = None
        self._frame_w              = 0
        self._frame_h              = 0
        self._last_motion_time     = 0.0   # debounces motion trigger

        # ── State machine ────────────────────────────────────────────────
        self._phase                = IDLE
        self._phase_entered        = 0.0
        # Physical pose we asked for (the body lags this by ~1 s).
        self._camera_pose          = "level"
        self._seek_start           = 0.0
        self._last_introduced      = 0.0
        self._last_complete_seen   = 0.0

        # Recognition state for the current RECOGNIZING phase.
        self._recog_obs              = deque(maxlen=RECOG_VOTES_REQUIRED)
        self._recog_best_face        = None
        self._recog_stabilize_until  = 0.0

        # Per-name cooldown — persists for the life of the task so the same
        # person doesn't get re-greeted in rapid succession across cycles.
        self._last_greeted: dict = {}

        # Enrollment state (driven by submit_transcript / timeout).
        self._enroll_state = {
            "active": False, "frame": None, "landmarks": None,
            "bbox":   None,  "asked_at": 0.0, "best_sharpness": 0.0,
        }

    # ── TTS pre-warm ─────────────────────────────────────────────────────

    def get_warm_phrases(self) -> list:
        """Phrases pella_main should pre-cache at startup to eliminate the
        first-time gen+upload latency for this task's likely utterances.

        Static prompts (always relevant) + a per-name "Hi, X" / "Nice to
        meet you, X" pair for each currently-enrolled person.
        """
        phrases = [
            "Hello, I am Pella. What is your name?",
            "Sorry, I didn't catch your name.",
        ]
        if self._recognizer is not None:
            for name in getattr(self._recognizer, "known", {}):
                display = name.replace("_", " ").title()
                phrases.append(f"Hi, {display}")
                phrases.append(f"Nice to meet you, {display}!")
        return phrases

    # ── Per-iteration tick ───────────────────────────────────────────────

    def tick(self, now) -> TickResult:
        """Advance one display-loop iteration.

        Returns a TickResult with whatever pella_main needs to display and
        log this iteration.
        """
        # ── SENSE ────────────────────────────────────────────────────────
        img = self._pull_latest_frame()
        motion_seen = self._sense_motion_and_faces(now, img)
        rec_got_new = self._pull_latest_recognition()

        # Refresh "have we seen a complete face recently" tracker.
        if self._last_complete_faces:
            self._last_complete_seen = now

        # Update recognition vote buffer + best-face capture if applicable.
        if self._phase == RECOGNIZING:
            if rec_got_new and self._last_rec_faces \
                    and now >= self._recog_stabilize_until:
                self._accumulate_vote()
            if img is not None and self._last_complete_faces \
                    and now >= self._recog_stabilize_until:
                self._update_best_face(img)

        # Enrollment best-frame upgrade (independent of phase).
        if self._enroll_state["active"] and img is not None \
                and self._last_complete_faces:
            self._update_enroll_best(img)

        # Cancel enrollment if the person didn't respond in time.
        if self._enroll_state["active"] \
                and now - self._enroll_state["asked_at"] >= ENROLL_TIMEOUT:
            self._enroll_state["active"] = False
            print("Task[recog_greeting]: enrollment timeout, no name received",
                  flush=True)

        # ── RESPOND: advance the state machine ───────────────────────────
        result = TickResult(latest_frame=img, latest_faces=self._last_faces)
        if self._phase == IDLE:
            self._tick_idle(now, motion_seen, img)
        elif self._phase == SEEKING:
            result = self._tick_seeking(now, img, result)
        elif self._phase == RECOGNIZING:
            result = self._tick_recognizing(now, img, result)
        elif self._phase == GREETING:
            result = self._tick_greeting(now, result)
        elif self._phase == INTRODUCING:
            result = self._tick_introducing(now, result)
        elif self._phase == COOLDOWN:
            result = self._tick_cooldown(now, result)
        return result

    # ── Transcript handoff (called from pella_main's transcript handler) ─

    def submit_transcript(self, now, text) -> bool:
        """Drive enrollment if we're INTRODUCING and waiting for a name.

        Returns True if the task consumed the transcript (pella_main can
        treat it as handled). False otherwise.
        """
        if not self._enroll_state["active"]:
            return False

        name_raw = _parse_name(text)
        if not name_raw:
            self._enroll_state["active"] = False
            try:
                self._say_queue.put_nowait("Sorry, I didn't catch your name.")
            except Exception:
                pass
            print(f"Task[recog_greeting]: enrollment skipped, "
                  f"'{text}' didn't parse as a name", flush=True)
            return True

        dir_name     = name_raw.lower().replace(" ", "_")
        display_name = name_raw.title()
        save_dir     = os.path.join(FACE_IDS_DIR, dir_name)
        if self._recognizer:
            enrolled = self._recognizer.enroll_new(
                dir_name,
                self._enroll_state["frame"],
                self._enroll_state["landmarks"],
                self._enroll_state.get("bbox"),
                save_dir,
            )
        else:
            # No recognizer loaded — fall back to saving the captured frame
            # only (no embedding). Use the same NNN.jpg numbering scheme so
            # repeat enrollments for the same person don't overwrite each other.
            os.makedirs(save_dir, exist_ok=True)
            idx = face_recognizer.next_image_index(save_dir)
            cv2.imwrite(os.path.join(save_dir, f"{idx:03d}.jpg"),
                        self._enroll_state["frame"])
            enrolled = True
        self._enroll_state["active"] = False

        try:
            if enrolled:
                self._say_queue.put_nowait(
                    f"Nice to meet you, {display_name}!")
            else:
                self._say_queue.put_nowait(
                    f"You don't look like the {display_name} I know. "
                    f"Sorry about that.")
            print(f"Task[recog_greeting]: "
                  f"{'enrolled' if enrolled else 'rejected'}: {display_name}",
                  flush=True)
        except Exception as e:
            print(f"Task[recog_greeting]: "
                  f"{'enrolled' if enrolled else 'rejected'}: {display_name} "
                  f"(say_queue full? {e})", flush=True)
        return True

    # ── SENSE helpers ────────────────────────────────────────────────────

    def _pull_latest_frame(self) -> Optional[np.ndarray]:
        """Drain frame_queue, keeping only the newest frame."""
        img = None
        try:
            while True:
                img = self._frame_queue.get_nowait()
        except Empty:
            pass
        return img

    def _sense_motion_and_faces(self, now, img) -> bool:
        """Run motion+person detection and the periodic face detector.

        Returns True if motion+person was seen on this tick — used as the
        IDLE -> SEEKING trigger.
        """
        if img is None:
            return False

        motion_seen = False
        if (now - self._last_motion_time >= MOTION_COOLDOWN
                and detect_motion(img, self._prev_frame)
                and detect_person(img)):
            motion_seen = True
            self._last_motion_time = now
        self._prev_frame = img

        # Periodic face detection — every N frames.
        self._detect_counter += 1
        if self._detect_counter >= FACE_DETECT_EVERY:
            self._detect_counter = 0
            self._last_faces = detect_faces(img)
            img_h, img_w = img.shape[:2]
            self._frame_w, self._frame_h = img_w, img_h
            self._last_complete_faces = [
                f for f in self._last_faces
                if not is_face_at_edge(f, img_w, img_h)
            ]
            if self._recognizer and self._last_faces:
                try:
                    self._rec_in.put_nowait((img.copy(), self._last_faces))
                except Exception:
                    pass
        return motion_seen

    def _pull_latest_recognition(self) -> bool:
        """Drain rec_out, keeping only the newest bundle. Returns True if a
        new bundle arrived this tick."""
        got = False
        try:
            while True:
                self._last_rec_faces, self._last_rec_names = self._rec_out.get_nowait()
                got = True
        except Empty:
            pass
        return got

    def _accumulate_vote(self):
        """Add one recognition vote for the biggest fully-framed face."""
        complete_rec = ([f for f in self._last_rec_faces
                         if not is_face_at_edge(f, self._frame_w, self._frame_h)]
                        if self._frame_w and self._frame_h else [])
        if not complete_rec:
            return
        biggest = max(complete_rec, key=lambda f: f[2] * f[3])
        idx = self._last_rec_faces.index(biggest)
        rec_name = (self._last_rec_names[idx]
                    if idx < len(self._last_rec_names) else None)
        self._recog_obs.append(rec_name)

    def _update_best_face(self, img: np.ndarray):
        """Track the sharpest complete face this RECOGNIZING session has seen."""
        biggest = max(self._last_complete_faces, key=lambda f: f[2] * f[3])
        bx, by, bw, bh = biggest[:4]
        region = img[by:by + bh, bx:bx + bw]
        if region.size == 0:
            return
        s = sharpness(region)
        if self._recog_best_face is None or s > self._recog_best_face["sharpness"]:
            self._recog_best_face = {
                "img":       img.copy(),
                "bbox":      (bx, by, bw, bh),
                "landmarks": biggest[4] if len(biggest) > 4 else None,
                "sharpness": s,
            }

    def _update_enroll_best(self, img: np.ndarray):
        """During enrollment, upgrade enroll_state with the sharpest face."""
        biggest = max(self._last_complete_faces, key=lambda f: f[2] * f[3])
        bx, by, bw, bh = biggest[:4]
        region = img[by:by + bh, bx:bx + bw]
        if region.size == 0:
            return
        s = sharpness(region)
        if s > self._enroll_state["best_sharpness"]:
            self._enroll_state.update({
                "frame":          img.copy(),
                "landmarks":      biggest[4] if len(biggest) > 4 else None,
                "bbox":           (bx, by, bw, bh),
                "best_sharpness": s,
            })
            print(f"Task[recog_greeting]: sharper enroll face captured "
                  f"(sharpness={s:.1f})", flush=True)

    # ── Phase advance helpers ────────────────────────────────────────────

    def _tick_idle(self, now, motion_seen, img):
        """Wait for a trigger — either a face already visible at level (skip
        seeking) or motion+person (look up to find a face)."""
        if img is not None and self._last_complete_faces:
            self._phase                  = RECOGNIZING
            self._phase_entered          = now
            self._recog_obs.clear()
            self._recog_best_face        = None
            self._recog_stabilize_until  = 0.0
            self._last_complete_seen     = now
            self._camera_pose            = "level"
            print("Task[recog_greeting]: -> recognizing "
                  "(face visible at level)", flush=True)
        elif motion_seen:
            self._phase          = SEEKING
            self._phase_entered  = now
            self._camera_pose    = "seeking"
            self._seek_start     = now
            try:
                self._action_queue.put_nowait("look_up")
            except Exception:
                pass
            print("Task[recog_greeting]: -> seeking (motion + person)",
                  flush=True)

    def _tick_seeking(self, now, img, result: TickResult) -> TickResult:
        if self._last_complete_faces:
            # Found a face — transition to RECOGNIZING while STAYING in the
            # current seek pose. Recovery is queued only when we leave
            # RECOGNIZING, so the body holds steady through vote accumulation.
            if self._camera_pose == "seeking_sit":
                stabilize_sec = RECOG_STABILIZE_SIT_SEC
            elif self._camera_pose == "seeking":
                stabilize_sec = RECOG_STABILIZE_LOOK_SEC
            else:
                stabilize_sec = 0.0
            self._phase                 = RECOGNIZING
            self._phase_entered         = now
            self._recog_obs.clear()
            self._recog_best_face       = None
            self._recog_stabilize_until = (now + stabilize_sec
                                           if stabilize_sec > 0 else 0.0)
            self._last_complete_seen    = now
            print(f"Task[recog_greeting]: -> recognizing "
                  f"(holding {self._camera_pose} pose"
                  + (f", stabilizing for {stabilize_sec:.1f}s)"
                     if stabilize_sec > 0 else ")"),
                  flush=True)
            return result

        if (self._camera_pose == "seeking"
                and now - self._seek_start >= SEEK_TIMEOUT
                and not actions.queue_contains(self._action_queue, "look_up")):
            self._camera_pose = "seeking_sit"
            self._seek_start  = now
            try:
                self._action_queue.put_nowait("sit_look_up")
            except Exception:
                pass
            print("Task[recog_greeting]: seek timeout, escalating to sit_look_up",
                  flush=True)
            return result

        if (self._camera_pose == "seeking_sit"
                and now - self._seek_start >= SEEK_TIMEOUT):
            # Sit also failed — stand back up and cool down.
            try:
                self._action_queue.put_nowait("stand_up")
            except Exception:
                pass
            self._camera_pose      = "level"
            self._last_motion_time = now
            self._phase            = COOLDOWN
            self._phase_entered    = now
            print("Task[recog_greeting]: seek give-up -> cooldown", flush=True)
            result.status_event = STATUS_FAILURE
        return result

    def _tick_recognizing(self, now, img, result: TickResult) -> TickResult:
        decide_now = (len(self._recog_obs) >= RECOG_VOTES_REQUIRED
                      or now - self._phase_entered >= RECOG_TIMEOUT_SEC)
        face_lost  = now - self._last_complete_seen > FACE_LOST_TIMEOUT_SEC

        if decide_now:
            return self._dispatch_verdict(now, img, result)
        if face_lost:
            # No TTS — recovery can happen immediately.
            self._queue_recovery(now, after_tts=False)
            self._phase         = COOLDOWN
            self._phase_entered = now
            print("Task[recog_greeting]: face lost before recognition completed "
                  "-> cooldown", flush=True)
            result.status_event = STATUS_FAILURE
        return result

    def _queue_recovery(self, now, after_tts: bool):
        """Queue the recovery action (stand_up / look_level) if needed.

        When after_tts=True, prepend wait_for_tts so the action consumer
        sleeps a bit before issuing the motor command. This lets a just-
        queued say() utterance play through the speaker cleanly before
        motor activity competes with audio on the WebRTC channel.
        """
        if self._camera_pose == "level":
            return
        recovery = ("stand_up" if self._camera_pose == "seeking_sit"
                    else "look_level")
        actions.drain_seek_actions(self._action_queue)
        try:
            if after_tts:
                self._action_queue.put_nowait("wait_for_tts")
            self._action_queue.put_nowait(recovery)
        except Exception:
            pass
        self._camera_pose       = "level"
        self._last_motion_time  = now
        print(f"Task[recog_greeting]: leaving recognizing, queueing "
              f"{'wait_for_tts -> ' if after_tts else ''}{recovery}",
              flush=True)

    def _dispatch_verdict(self, now, img, result: TickResult) -> TickResult:
        verdict, kind = _tally_recognition(self._recog_obs)
        if kind == "known":
            return self._enter_greeting(now, img, verdict, result)
        if kind == "unknown":
            return self._enter_introducing(now, result)
        # ambiguous — no TTS, recovery can fire immediately.
        self._queue_recovery(now, after_tts=False)
        self._phase         = COOLDOWN
        self._phase_entered = now
        print("Task[recog_greeting]: recognition ambiguous -> cooldown",
              flush=True)
        result.status_event = STATUS_NOT_RECOGNIZED
        return result

    def _enter_greeting(self, now, img, verdict, result: TickResult) -> TickResult:
        display_name = verdict.replace("_", " ").title()
        if (now - self._last_greeted.get(verdict, 0.0)) < GREET_COOLDOWN:
            # Recovery still needed; no TTS so no wait.
            self._queue_recovery(now, after_tts=False)
            self._phase         = COOLDOWN
            self._phase_entered = now
            print(f"Task[recog_greeting]: already greeted {display_name} recently "
                  f"-> cooldown", flush=True)
            result.status_event = STATUS_KNOWN_PERSON
            return result

        # Say first so TTS uploads/plays as soon as possible, then queue
        # wait_for_tts + recovery so motor activity holds off until audio
        # has had time to come through the speaker. wiggle lands after
        # recovery as the final gesture.
        try:
            self._say_queue.put_nowait(f"Hi, {display_name}")
        except Exception:
            pass
        self._queue_recovery(now, after_tts=True)
        try:
            self._action_queue.put_nowait("wiggle")
        except Exception:
            pass
        self._last_greeted[verdict] = now
        self._last_introduced       = now   # also suppress introduce
        self._last_motion_time      = now

        # Prefer the live frame for the zoom label, fall back to the captured
        # snapshot if the person has already moved out of frame.
        if self._last_complete_faces and img is not None:
            biggest = max(self._last_complete_faces, key=lambda f: f[2] * f[3])
            crop = zoom_crop(img, biggest)
        elif self._recog_best_face is not None:
            crop = zoom_crop(self._recog_best_face["img"],
                             self._recog_best_face["bbox"])
        else:
            crop = None

        if crop is not None:
            zoom_bgr = crop.copy()
            cv2.putText(zoom_bgr, verdict, (20, zoom_bgr.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 0), 2, cv2.LINE_AA)
            result.display_request = DisplayRequest(
                image=zoom_bgr, duration_sec=CAPTURE_DISPLAY_SEC)

        self._phase          = GREETING
        self._phase_entered  = now
        result.status_event  = STATUS_KNOWN_PERSON
        print(f"Task[recog_greeting]: greeting {display_name}", flush=True)
        return result

    def _enter_introducing(self, now, result: TickResult) -> TickResult:
        if self._enroll_state["active"]:
            self._queue_recovery(now, after_tts=False)
            self._phase         = COOLDOWN
            self._phase_entered = now
            print("Task[recog_greeting]: unknown but enrollment already active "
                  "-> cooldown", flush=True)
            result.status_event = STATUS_NEW_PERSON
            return result
        if now - self._last_introduced < INTRODUCE_COOLDOWN:
            self._queue_recovery(now, after_tts=False)
            self._phase         = COOLDOWN
            self._phase_entered = now
            print(f"Task[recog_greeting]: unknown but introduce on cooldown "
                  f"({INTRODUCE_COOLDOWN - (now - self._last_introduced):.1f}s "
                  f"remaining) -> cooldown", flush=True)
            result.status_event = STATUS_NEW_PERSON
            return result
        if (self._recog_best_face is None
                or self._recog_best_face["sharpness"] < SHARPNESS_THRESHOLD):
            self._queue_recovery(now, after_tts=False)
            self._phase         = COOLDOWN
            self._phase_entered = now
            sharp_str = (f"{self._recog_best_face['sharpness']:.1f}"
                         if self._recog_best_face else "none")
            print(f"Task[recog_greeting]: unknown face not sharp enough "
                  f"(best sharpness={sharp_str}, threshold={SHARPNESS_THRESHOLD}) "
                  f"-> cooldown", flush=True)
            result.status_event = STATUS_NOT_RECOGNIZED
            return result

        # Say first, then queue wait_for_tts + recovery so the "Hello, I am
        # Pella…" prompt plays through before stand_up kicks the motors.
        try:
            self._say_queue.put_nowait(
                "Hello, I am Pella. What is your name?")
        except Exception as e:
            print(f"Task[recog_greeting]: WARN say_queue full, "
                  f"name-ask not queued: {e}", flush=True)
        self._queue_recovery(now, after_tts=True)
        self._enroll_state.update({
            "active":         True,
            "frame":          self._recog_best_face["img"],
            "landmarks":      self._recog_best_face["landmarks"],
            "bbox":           self._recog_best_face["bbox"],
            "asked_at":       now,
            "best_sharpness": self._recog_best_face["sharpness"],
        })
        self._last_introduced  = now
        self._last_motion_time = now

        crop = zoom_crop(self._recog_best_face["img"],
                         self._recog_best_face["bbox"])
        result.display_request = DisplayRequest(
            image=crop, duration_sec=CAPTURE_DISPLAY_SEC)

        self._phase          = INTRODUCING
        self._phase_entered  = now
        result.status_event  = STATUS_NEW_PERSON
        print(f"Task[recog_greeting]: introducing, asking for name "
              f"(sharpness={self._recog_best_face['sharpness']:.1f})",
              flush=True)
        return result

    def _tick_greeting(self, now, result: TickResult) -> TickResult:
        if now - self._phase_entered >= GREETING_DURATION_SEC:
            self._phase         = COOLDOWN
            self._phase_entered = now
            print("Task[recog_greeting]: greeting complete -> cooldown",
                  flush=True)
            result.status_event = STATUS_SUCCESSFUL
        return result

    def _tick_introducing(self, now, result: TickResult) -> TickResult:
        # Driven by submit_transcript / the enrollment timeout check above;
        # both flip enroll_state["active"] back to False when done.
        if not self._enroll_state["active"]:
            self._phase         = COOLDOWN
            self._phase_entered = now
            print("Task[recog_greeting]: introducing complete -> cooldown",
                  flush=True)
            result.status_event = STATUS_SUCCESSFUL
        return result

    def _tick_cooldown(self, now, result: TickResult) -> TickResult:
        if now - self._phase_entered >= INTERACTION_COOLDOWN_SEC:
            self._phase         = IDLE
            self._phase_entered = now
            print("Task[recog_greeting]: cooldown done -> idle", flush=True)
        return result
