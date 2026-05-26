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
    INTRODUCE_COOLDOWN, ENROLL_LISTEN_WINDOW, ENROLL_LOOKBACK_SEC,
    ENROLL_TIMEOUT, CORRECTION_WINDOW, FACE_IDS_DIR, SHARPNESS_THRESHOLD,
    ENROLL_BUFFER_SIZE,
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
    # Common state replies after "I'm X" — would otherwise be mis-parsed
    # as a name correction by the intro-phrase branch.
    "fine", "good", "great", "well", "alright", "tired", "busy",
    "happy", "sad", "lost", "here", "back", "home", "ready",
}


def _parse_name(text: str, *, require_intro_phrase: bool = False) -> str:
    """Extract a name from a casual reply, ignoring courtesy phrases.

    Returns an empty string when the input doesn't look like a name — e.g.
    contains common non-name words like 'excuse me' or 'hold it for me'.

    When `require_intro_phrase=True`, only matches with an explicit
    "my name is X" / "I am X" / "call me X" / etc. — bare names are
    rejected. Used for *corrections* to a stored name: a casual reply
    like "I'm fine, thanks" right after a greeting shouldn't accidentally
    rename the person.
    """
    t = text.strip()
    t = re.split(r"[.!?]", t, maxsplit=1)[0].strip()
    # Handles the most common name-intro phrasings. The apostrophe-s
    # contractions ("my name's", "name's") show up frequently in
    # Whisper transcripts even when the speaker said the full "my name is".
    m = re.search(
        r"(?:i am|i'm|i'?m\s+called|my\s+name\s+is|my\s+name'?s|name'?s|"
        r"call\s+me|this\s+is)\s+(.+)",
        t, re.IGNORECASE,
    )
    if m:
        t = m.group(1).strip()
    elif require_intro_phrase:
        return ""
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

    def __init__(self, frame_queue, action_queue, say_queue, prep_queue,
                 recognizer, stop_event):
        """Wire dependencies and start the recognition worker thread.

        frame_queue / action_queue / say_queue: shared queues filled or drained
                                                 by other organs (eye / limbs /
                                                 mouth).
        prep_queue:  TTS pre-cache channel. Pushing a text here uploads the
                     WAV to the audiohub without playing it, so a subsequent
                     say of the same text fires immediately. Used for
                     dynamic name-dependent phrases that the startup warmup
                     can't anticipate.
        recognizer:  the ArcFace recognizer, or None if disabled.
        stop_event:  pella_main's shutdown flag; the recognition worker
                     thread exits when this is set.
        """
        self._frame_queue  = frame_queue
        self._action_queue = action_queue
        self._say_queue    = say_queue
        self._prep_queue   = prep_queue
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
        self._enroll_state = {"active": False, "asked_at": 0.0}

        # Ring buffer of candidate face captures collected during the
        # introducing window. Scored and picked at enrollment time so
        # we deliberate over many frames instead of greedily committing
        # to the first sharp one in real time (which often turned out
        # to be a poor-pose capture).
        self._enroll_candidates: deque = deque(maxlen=ENROLL_BUFFER_SIZE)

        # Correction state: after Pella greets/introduces someone, the user
        # has CORRECTION_WINDOW seconds to say "my name is X" to rename them.
        # `opened_at` is the monotonic time the window was opened (which is
        # also the earliest accepted capture_t, so corrections shouted before
        # the greeting played don't count).
        self._correction_state = {
            "active": False,
            "dir_name": None,        # canonical key under face_ids/
            "display_name": None,    # human-readable form Pella said
            "opened_at": 0.0,
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
            "Sorry, I cannot see you clearly.",
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

        # Append face captures to the candidate buffer during enrollment.
        # The best one is picked from the buffer when submit_transcript
        # fires successful enrollment.
        if self._enroll_state["active"] and img is not None \
                and self._last_complete_faces:
            self._capture_enroll_candidate(img)

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

    def submit_transcript(self, now, text, capture_t) -> bool:
        """Drive enrollment if we're INTRODUCING and waiting for a name,
        or rename the just-addressed person if we're in a correction
        window after a greeting/enrollment.

        `capture_t` is the monotonic time the user actually started
        speaking (stamped by stt.py at VAD-speech-start), not the time
        the transcript arrived. We accept transcripts whose capture_t
        falls inside [asked_at, asked_at + ENROLL_LISTEN_WINDOW], so a
        slow Whisper transcription doesn't kill an enrollment whose
        speech was timely.

        Returns True if the task consumed the transcript (pella_main can
        treat it as handled). False otherwise.
        """
        # Correction takes precedence over enrollment: if Pella just said
        # "Nice to meet you, Willie!" and the user replies "My name is
        # William.", we want to rename, not start a fresh enrollment.
        if self._correction_state["active"]:
            opened = self._correction_state["opened_at"]
            if capture_t < opened:
                pass  # speech was before the greeting; fall through
            elif capture_t > opened + CORRECTION_WINDOW:
                # Window expired — close it and fall through to normal
                # handling (which will reject the transcript at the
                # enrollment check below, since enroll is not active).
                self._correction_state["active"] = False
            else:
                new_name = _parse_name(text, require_intro_phrase=True)
                if new_name:
                    old_dir = self._correction_state["dir_name"]
                    new_dir = new_name.lower().replace(" ", "_")
                    if new_dir != old_dir:
                        if self._apply_correction(new_name):
                            self._correction_state["active"] = False
                            return True
                    # Same name — user confirmed; just close the window.
                    self._correction_state["active"] = False
                    return True
                # No intro phrase parsed inside the window — leave the
                # window open in case another transcript arrives soon
                # (e.g. user clarifies). Don't consume; fall through.

        if not self._enroll_state["active"]:
            return False

        asked_at = self._enroll_state["asked_at"]
        if capture_t < asked_at - ENROLL_LOOKBACK_SEC:
            # Speech was captured well before we asked — leftover from a
            # previous window. Ignore without consuming.
            print(f"Task[recog_greeting]: ignoring transcript "
                  f"(captured {asked_at - capture_t:.1f}s before intro, "
                  f"outside {ENROLL_LOOKBACK_SEC:.0f}s lookback): "
                  f"{repr(text)}", flush=True)
            return False
        if capture_t > asked_at + ENROLL_LISTEN_WINDOW:
            # User started speaking after the listening window closed.
            # Don't enroll them, but consume the transcript so chat
            # doesn't also try to answer it.
            print(f"Task[recog_greeting]: ignoring transcript "
                  f"(captured {capture_t - asked_at:.1f}s after intro, "
                  f"outside {ENROLL_LISTEN_WINDOW:.0f}s window): "
                  f"{repr(text)}", flush=True)
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

        # Kick off TTS pre-cache for whichever phrase we'll say next, RIGHT
        # NOW — generation + audiohub upload (~10-20 s) will overlap with
        # enrollment (~1-2 s) and any time the user spent talking. By the
        # time we actually put the phrase on say_queue, the cache should
        # be partially or fully warm and playback fires sooner.
        for prep_text in (
            f"Nice to meet you, {display_name}!",
            f"You don't look like the {display_name} I know. "
            f"Sorry about that.",
        ):
            try:
                self._prep_queue.put_nowait(prep_text)
            except Exception:
                pass

        best = self._pick_enroll_best()
        if best is None:
            # Should never happen — _enter_introducing seeds the buffer
            # with at least the recog_best_face. Defensive.
            print("Task[recog_greeting]: no enrollment candidates in "
                  "buffer — skipping", flush=True)
            enrolled = False
        elif self._recognizer:
            # trust_name=True: the user verbally said this name a moment
            # ago, so add the embedding to their set regardless of how
            # similar it looks to the existing embeddings. The verbal
            # claim is authoritative; rejecting the embedding on
            # similarity grounds creates a circular failure when the
            # original enrollment captured a poor-angle/poor-lighting
            # frame ("you can't be William because William doesn't look
            # like you, even though you just said you are").
            enrolled = self._recognizer.enroll_new(
                dir_name,
                best["frame"],
                best["landmarks"],
                best["bbox"],
                save_dir,
                trust_name=True,
            )
        else:
            # No recognizer loaded — fall back to saving the captured frame
            # only (no embedding). Use the same NNN.jpg numbering scheme so
            # repeat enrollments for the same person don't overwrite each other.
            os.makedirs(save_dir, exist_ok=True)
            idx = face_recognizer.next_image_index(save_dir)
            cv2.imwrite(os.path.join(save_dir, f"{idx:03d}.jpg"),
                        best["frame"])
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
        if enrolled:
            # Open the correction window so a follow-up "My name is X"
            # can rename a mishear (e.g. "Willie" -> "William").
            self._open_correction_window(now, dir_name, display_name)
        return True

    # ── Correction window ────────────────────────────────────────────────

    def _open_correction_window(self, now, dir_name, display_name):
        """Open a CORRECTION_WINDOW-second window after Pella addressed
        someone, during which an explicit name intro renames them.

        Called after both:
          * "Hi, X." (greeting of a recognised person)
          * "Nice to meet you, X!" (successful enrollment)
        """
        self._correction_state.update({
            "active":       True,
            "dir_name":     dir_name,
            "display_name": display_name,
            "opened_at":    now,
        })

    def _apply_correction(self, new_name_raw: str):
        """Rename the just-addressed person to `new_name_raw` and say so.

        Handles both the rename on disk + in the recognizer, and updates
        the in-memory per-name cooldown so the rename doesn't make the
        person eligible for an immediate re-greeting under the new name.
        """
        old_dir  = self._correction_state["dir_name"]
        new_dir  = new_name_raw.lower().replace(" ", "_")
        new_disp = new_name_raw.title()
        if old_dir == new_dir:
            return False

        if self._recognizer is not None:
            self._recognizer.rename(old_dir, new_dir)
        else:
            # Fallback: directory move without an embedding map.
            import shutil
            src = os.path.join(FACE_IDS_DIR, old_dir)
            dst = os.path.join(FACE_IDS_DIR, new_dir)
            if os.path.isdir(src) and not os.path.isdir(dst):
                os.rename(src, dst)
            elif os.path.isdir(src) and os.path.isdir(dst):
                # Merge by best-effort copy then remove
                for fname in os.listdir(src):
                    shutil.move(os.path.join(src, fname), dst)
                try:
                    os.rmdir(src)
                except OSError:
                    pass

        # Carry the per-name cooldown over so we don't re-greet under
        # the new name in the next tick.
        if old_dir in self._last_greeted:
            self._last_greeted[new_dir] = self._last_greeted.pop(old_dir)

        # Pre-warm TTS for the new name (most likely the next greeting).
        for prep_text in (f"Hi, {new_disp}", f"Nice to meet you, {new_disp}!"):
            try:
                self._prep_queue.put_nowait(prep_text)
            except Exception:
                pass

        try:
            self._say_queue.put_nowait(f"Sorry, {new_disp}. Got it.")
        except Exception:
            pass

        print(f"Task[recog_greeting]: corrected '{old_dir}' -> '{new_dir}'",
              flush=True)
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

    def _capture_enroll_candidate(self, img: np.ndarray):
        """Append the current largest detected face to the candidate buffer.

        Replaces the older greedy "keep only the sharpest" approach: by
        retaining many candidates we can score them on more than just
        sharpness at enrollment time (see _pick_enroll_best).
        """
        biggest = max(self._last_complete_faces, key=lambda f: f[2] * f[3])
        bx, by, bw, bh = biggest[:4]
        region = img[by:by + bh, bx:bx + bw]
        if region.size == 0:
            return
        s = sharpness(region)
        self._enroll_candidates.append({
            "frame":     img.copy(),
            "landmarks": biggest[4] if len(biggest) > 4 else None,
            "bbox":      (bx, by, bw, bh),
            "sharpness": s,
        })

    def _pick_enroll_best(self) -> Optional[dict]:
        """Score the buffered candidates and return the highest scorer.

        Returns the candidate dict (frame / landmarks / bbox / sharpness)
        or None if no candidates are available. Picks on
            sharpness * (0.5 + 0.5 * frontality)
        — sharpness alone misses pose: a 375-sharp face at a steep yaw is
        worse for recognition than a 200-sharp frontal one. The 0.5 floor
        keeps a candidate without landmark info from being zeroed out.
        """
        if not self._enroll_candidates:
            return None
        scored = [(self._enroll_candidate_score(c), c)
                  for c in self._enroll_candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best = scored[0]
        # Also report the next-best for visibility of what was traded off.
        if len(scored) > 1:
            runner_score, runner = scored[1]
            print(f"Task[recog_greeting]: picked enroll face from "
                  f"{len(self._enroll_candidates)} candidates "
                  f"(score={best_score:.1f} sharp={best['sharpness']:.1f}; "
                  f"runner-up score={runner_score:.1f} "
                  f"sharp={runner['sharpness']:.1f})", flush=True)
        else:
            print(f"Task[recog_greeting]: picked enroll face "
                  f"(only candidate; score={best_score:.1f} "
                  f"sharp={best['sharpness']:.1f})", flush=True)
        return best

    @staticmethod
    def _enroll_candidate_score(c: dict) -> float:
        """Combined quality score: sharpness * (0.5 + 0.5 * frontality).

        Frontality is derived from YuNet 5-point landmarks
        (eyes + nose + mouth corners). Yaw is the dominant pose problem
        for recognition, estimated as |nose_x - eye_midpoint_x| /
        inter-eye distance. A perfectly frontal face has yaw_offset ~ 0;
        we map yaw_offset / 0.3 -> bad. Without landmarks we fall back
        to sharpness only (frontality = 1 in expectation, no penalty).
        """
        s = c["sharpness"]
        lm = c["landmarks"]
        if lm is None or len(lm) < 3:
            return s * 0.5
        left_eye, right_eye, nose = lm[0], lm[1], lm[2]
        eye_mid_x  = (left_eye[0] + right_eye[0]) * 0.5
        eye_width  = abs(right_eye[0] - left_eye[0]) + 1e-6
        yaw_offset = abs(nose[0] - eye_mid_x) / eye_width
        yaw_q      = max(0.0, 1.0 - yaw_offset / 0.3)
        return s * (0.5 + 0.5 * yaw_q)

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
        self._open_correction_window(now, verdict, display_name)
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
            # Let the person know why we're not engaging. Voice the apology
            # first so the wait_for_tts before recovery covers playback.
            try:
                self._say_queue.put_nowait("Sorry, I cannot see you clearly.")
            except Exception:
                pass
            self._queue_recovery(now, after_tts=True)
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
        # Seed the candidate buffer with the recognition-phase best face
        # (the one that triggered introducing). More candidates accumulate
        # via _capture_enroll_candidate across the introducing window.
        self._enroll_candidates.clear()
        self._enroll_candidates.append({
            "frame":     self._recog_best_face["img"],
            "landmarks": self._recog_best_face["landmarks"],
            "bbox":      self._recog_best_face["bbox"],
            "sharpness": self._recog_best_face["sharpness"],
        })
        self._enroll_state.update({"active": True, "asked_at": now})
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
