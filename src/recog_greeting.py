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
from dataclasses import dataclass, field
from queue import Empty
from typing import Optional

import numpy as np

import actions
from perception import Perception
from vision import (
    GREET_COOLDOWN, MOTION_COOLDOWN, SEEK_TIMEOUT,
    INTRODUCE_COOLDOWN, SEE_COMPLAINT_COOLDOWN,
    ENROLL_LISTEN_WINDOW, ENROLL_LOOKBACK_SEC,
    ENROLL_TIMEOUT, ENROLL_MAX_ATTEMPTS, CONFIRM_TIMEOUT_SEC, CORRECTION_WINDOW,
    FACE_IDS_DIR, SHARPNESS_THRESHOLD,
    STITCH_GAP_SEC,
    label_face_zoom, zoom_crop,
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
    "hmm", "huh", "ugh", "eh", "mm", "mmm", "hm",
    "hi", "hey", "hello", "bye",
    # Polite / imperative words seen in mis-captures
    "please", "thanks", "thank", "sorry",
    "stop", "hold", "wait", "excuse", "tell", "give", "take",
    # Common state replies after "I'm X" / "It's X" / "That's X" — would
    # otherwise be mis-parsed as a name correction by the intro-phrase
    # branch. Expanded to cover state words that follow the weaker
    # "it's"/"that's" pointers added to intro_re.
    "fine", "good", "great", "well", "alright", "tired", "busy",
    "happy", "sad", "lost", "here", "back", "home", "ready",
    "cool", "nice", "hot", "warm", "cold", "bad", "right", "wrong",
    "true", "false", "late", "early", "easy", "hard", "fun", "old",
    "new", "big", "small", "first", "last",
}


# Hoisted to module scope so _has_intro_phrase() can reuse it without
# recompiling the regex per call. Same pattern as previously defined inside
# _parse_name — see that function's docstring for what each alternative
# matches and why.
_INTRO_RE = re.compile(
    r"(?:i am|i'm|i'?m\s+called|my\s+name\s+is|my\s+name'?s|name'?s|"
    r"call\s+me|this\s+is|it'?s|it\s+is|that'?s|that\s+is)\s+(.+)",
    re.IGNORECASE,
)

# Yes / no patterns for the "Did you say {name}?" confirmation reply.
# Matched against the lowercased, punctuation-stripped transcript.
# Anchored to start because we want the user's leading word to be the
# affirmation/rejection — "yes, joy" / "no, joy" — not a buried token.
_YES_RE = re.compile(
    r"^(?:yes|yeah|yep|yup|sure|right|correct|that'?s\s+right|"
    r"that\s+is\s+right|ok|okay)\b",
    re.IGNORECASE,
)
_NO_RE = re.compile(
    r"^(?:no|nope|nah|wrong|incorrect|not\s+quite|that'?s\s+wrong)\b(.*)",
    re.IGNORECASE,
)


def _has_intro_phrase(text: str) -> bool:
    """True iff the transcript contains an explicit name-intro phrase.

    Used to decide whether a parsed name needs confirmation. A bare-word
    transcript ("Joy" -> "Enjoy" via Whisper) goes through "Did you say X?"
    because it's the hallucination failure mode; "My name is Joy" goes
    straight to enrollment because it carries enough acoustic context for
    Whisper to commit confidently.
    """
    return _INTRO_RE.search(text) is not None


def _parse_confirmation(text: str):
    """Parse a yes / no / correction reply to "Did you say {name}?".

    Returns a tuple (verdict, new_name):
      verdict   : "yes" | "no" | "unclear"
      new_name  : a name parsed out of a "no, X" / "it's Y" / "my name
                  is Z" reply, or None.

    "yes" → commit the originally-parsed name.
    "no" with new_name → commit the new name instead.
    "no" without new_name → re-prompt for the name (consumes a retry slot).
    "unclear" → ignore and keep waiting; tick() handles the timeout.
    """
    t = re.sub(r"[.,!?]+", " ", text.strip())
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ("unclear", None)

    # Check "no" first, because "no, joy" should NOT be confused with a
    # bare-name parse of "joy" alone — the leading "no" is the signal.
    no_match = _NO_RE.match(t)
    if no_match:
        rest = no_match.group(1).strip(" ,.;:-")
        if rest:
            new_name = _parse_name(rest)
            if new_name:
                return ("no", new_name)
        return ("no", None)

    if _YES_RE.match(t):
        return ("yes", None)

    # No explicit yes/no leading word — treat any successfully-parsed name
    # as an implicit correction ("Joy." / "It's Joy." / "My name is Joy.").
    new_name = _parse_name(t)
    if new_name:
        return ("no", new_name)

    return ("unclear", None)


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
    # Collapse ellipses ("..." or longer) to a single space before splitting
    # on sentence punctuation. Whisper emits "..." for trailing-off speech
    # ("My name is...") and the bare first-sentence split would otherwise
    # cut the utterance at the first dot — losing the actual name that
    # arrives after a stitch from the next clip.
    t = re.sub(r"\.{2,}", " ", t)
    t = re.split(r"[.!?]", t, maxsplit=1)[0].strip()
    intro_re = _INTRO_RE
    # Loop because a stitched transcript can look like "my name is ... my
    # name is Joy" — peeling off only the first intro phrase would leave
    # "my name is Joy" and trip the non-name-word filter on "my". Strip
    # all intro phrases iteratively until none remain.
    matched_any = False
    while True:
        m = intro_re.search(t)
        if not m:
            break
        matched_any = True
        t = m.group(1).strip()
    if not matched_any and require_intro_phrase:
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

        # All cv2 / image / recognition-worker state lives behind this.
        self._perception = Perception(recognizer, stop_event)

        # ── State machine ────────────────────────────────────────────────
        self._phase                = IDLE
        self._phase_entered        = 0.0
        # Physical pose we asked for (the body lags this by ~1 s).
        self._camera_pose          = "level"
        self._seek_start           = 0.0
        self._last_introduced      = 0.0
        # Suppresses chain-repeats of "Sorry, I cannot see you clearly"
        # — see SEE_COMPLAINT_COOLDOWN in vision.py.
        self._last_see_complaint_at = 0.0

        # RECOGNIZING-phase timing (task-side gate on when to start voting).
        self._recog_stabilize_until  = 0.0

        # Per-name cooldown — persists for the life of the task so the same
        # person doesn't get re-greeted in rapid succession across cycles.
        self._last_greeted: dict = {}

        # Enrollment state (driven by submit_transcript / timeout).
        # `attempts` counts attempts made so far for the current
        # introducing window: 1 = the initial intro is open, 2..N = a
        # retry window is open after Pella apologised. When `attempts`
        # exceeds ENROLL_MAX_ATTEMPTS, enrollment closes silently.
        # Reset in _enter_introducing.
        self._enroll_state = {"active": False, "asked_at": 0.0,
                              "attempts": 0}

        # Sentence-stitching: when a transcript arrives that doesn't parse
        # as a name (e.g. VAD split "My name is William" on mid-utterance
        # prosody, giving us just "My name is..."), hold it for up to
        # STITCH_GAP_SEC in case the continuation arrives in the next
        # transcript. Cleared on _enter_introducing and after a successful
        # stitch + parse.
        self._enroll_pending = {
            "text":         None,
            "speech_end_t": 0.0,    # end of audible speech in the held clip
            "expires_at":   0.0,    # monotonic wall-clock deadline
        }

        # Confirmation state: a bare-name parse from the introducing window
        # (e.g. Whisper transcribed "Enjoy" for the user's slow "My name is
        # Joy") routes here instead of enrolling immediately. Pella asks
        # "Did you say {display_name}?" and listens for a yes / no /
        # correction reply, deciding whether to commit the original name,
        # commit a corrected one, or re-prompt. Multi-word intro-phrase
        # transcripts skip this — they have enough acoustic context that
        # Whisper rarely substitutes a wrong name.
        self._confirm_state = {
            "active":       False,
            "dir_name":     None,    # candidate canonical key
            "display_name": None,    # candidate human form
            "asked_at":     0.0,     # monotonic time we asked the question
        }

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
        for name in self._perception.known_names():
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
        # Drain the frame queue, keep the newest. All downstream perception
        # runs on this single frame.
        img = None
        try:
            while True:
                img = self._frame_queue.get_nowait()
        except Empty:
            pass

        motion_seen = self._perception.sense_motion_and_faces(
            now, img, MOTION_COOLDOWN)
        rec_got_new = self._perception.pull_latest_recognition()

        # Update recognition vote buffer + best-face capture if applicable.
        if self._phase == RECOGNIZING:
            if rec_got_new and self._perception.last_rec_faces \
                    and now >= self._recog_stabilize_until:
                self._perception.accumulate_vote()
            if img is not None and self._perception.last_complete_faces \
                    and now >= self._recog_stabilize_until:
                self._perception.update_best_face(img)

        # Append face captures to the candidate buffer during enrollment.
        # The best one is picked from the buffer when submit_transcript
        # fires successful enrollment.
        if self._enroll_state["active"] and img is not None \
                and self._perception.last_complete_faces:
            self._perception.capture_enroll_candidate(img)

        # Pending held transcript expired with no continuation: apologise
        # (first time) or close silently (second time, post-retry).
        if self._enroll_pending["text"] is not None \
                and now >= self._enroll_pending["expires_at"]:
            held = self._enroll_pending["text"]
            self._apologise_and_arm_retry_or_close(
                now, f"held transcript {held!r} never completed")

        # Cancel enrollment if the person didn't respond in time.
        # Same retry semantics: first timeout apologises and gives one
        # more ENROLL_LISTEN_WINDOW; second timeout closes silently.
        if self._enroll_state["active"] \
                and now - self._enroll_state["asked_at"] >= ENROLL_TIMEOUT:
            self._apologise_and_arm_retry_or_close(
                now, "enrollment timeout, no name received")

        # Confirmation timeout: "Did you say X?" went unanswered. Default
        # to "yes" — assume the original parse was correct. Better to
        # enroll a possibly-wrong name (recoverable via the correction
        # window) than to drop the interaction entirely.
        if self._confirm_state["active"] \
                and now - self._confirm_state["asked_at"] >= CONFIRM_TIMEOUT_SEC:
            display_name = self._confirm_state["display_name"]
            print(f"Task[recog_greeting]: confirmation timeout for "
                  f"'{display_name}' — assuming yes", flush=True)
            self._commit_enrollment(now, display_name)

        # ── RESPOND: advance the state machine ───────────────────────────
        result = TickResult(latest_frame=img,
                            latest_faces=self._perception.last_faces)
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

    # ── Enrollment failure: apologise once, then close on second strike ──

    def _apologise_and_arm_retry_or_close(self, now: float, reason: str):
        """Common path when an enrollment attempt failed to receive or
        parse a name.

        Bumps `attempts`. While attempts remain in the budget
        (≤ ENROLL_MAX_ATTEMPTS), say "Sorry, I didn't catch your name",
        reset asked_at to now (giving the user a fresh
        ENROLL_LISTEN_WINDOW), and stay active. The user can repeat
        their name as "My name is X" or just "X".

        When attempts exhausts the budget, close silently — no further
        apologies, no chain of "Sorry"s.

        Pending stitch buffer is always cleared so a fragment from the
        previous attempt doesn't accidentally stitch with the retry
        speech.
        """
        self._enroll_pending = {"text": None, "speech_end_t": 0.0,
                                "expires_at": 0.0}
        if not self._enroll_state["active"]:
            return

        attempts = self._enroll_state.get("attempts", 1)
        if attempts < ENROLL_MAX_ATTEMPTS:
            try:
                self._say_queue.put_nowait(
                    "Sorry, I didn't catch your name.")
            except Exception:
                pass
            self._enroll_state["attempts"] = attempts + 1
            self._enroll_state["asked_at"] = now
            print(f"Task[recog_greeting]: {reason} — apologising "
                  f"(attempt {attempts}/{ENROLL_MAX_ATTEMPTS} failed, "
                  f"retry window open)", flush=True)
        else:
            self._enroll_state["active"] = False
            print(f"Task[recog_greeting]: {reason} — giving up after "
                  f"{attempts} attempts", flush=True)

    # ── Transcript handoff (called from pella_main's transcript handler) ─

    def submit_transcript(self, now, text, capture_t, end_t) -> bool:
        """Drive enrollment if we're INTRODUCING and waiting for a name,
        or rename the just-addressed person if we're in a correction
        window after a greeting/enrollment.

        `capture_t` is the monotonic time the user actually started
        speaking (stamped by stt.py at VAD-speech-start), and `end_t`
        is the corresponding end of audible speech. We accept transcripts
        whose capture_t falls inside [asked_at - ENROLL_LOOKBACK_SEC,
        asked_at + ENROLL_LISTEN_WINDOW], so a slow Whisper transcription
        doesn't kill an enrollment whose speech was timely.

        If a transcript inside that window doesn't parse as a name, it's
        held for STITCH_GAP_SEC and concatenated with the next transcript
        whose speech_start_t is within STITCH_GAP_SEC of the held one's
        end. This forgives a VAD split mid-utterance (e.g. "My name is..."
        / "William.") without needing to extend the silence trailer.

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

        # Confirmation window takes precedence over fresh enrollment too:
        # if Pella just asked "Did you say X?", a reply during that window
        # is a yes/no/correction — not a new name to parse.
        if self._confirm_state["active"]:
            if self._handle_confirmation_reply(now, text, capture_t):
                return True
            # Out-of-window transcript — fall through to normal handling
            # (which will likely drop it because enroll_state is inactive).

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

        # Sentence stitching: if a previous transcript was held because
        # it didn't parse on its own AND this transcript's speech-start
        # is within STITCH_GAP_SEC of the held one's speech-end, treat
        # them as one logical utterance and concatenate before parsing.
        combined_text = text
        pending_text  = self._enroll_pending["text"]
        if pending_text is not None:
            gap = capture_t - self._enroll_pending["speech_end_t"]
            if gap < STITCH_GAP_SEC:
                combined_text = f"{pending_text} {text}"
                print(f"Task[recog_greeting]: stitched "
                      f"({gap:.1f}s gap) '{pending_text}' + '{text}' "
                      f"-> '{combined_text}'", flush=True)
                self._enroll_pending = {"text": None, "speech_end_t": 0.0,
                                        "expires_at": 0.0}
            else:
                # Gap too wide — the held transcript is stale. Drop it
                # and treat the new transcript on its own.
                print(f"Task[recog_greeting]: discarding stale held "
                      f"transcript '{pending_text}' ({gap:.1f}s gap)",
                      flush=True)
                self._enroll_pending = {"text": None, "speech_end_t": 0.0,
                                        "expires_at": 0.0}

        name_raw = _parse_name(combined_text)
        if not name_raw:
            # Don't apologise yet — hold this transcript for STITCH_GAP_SEC
            # in case the rest of the utterance arrives in the next
            # transcript (VAD often splits "My name is William" on prosody).
            self._enroll_pending = {
                "text":         combined_text,
                "speech_end_t": end_t,
                "expires_at":   now + STITCH_GAP_SEC,
            }
            print(f"Task[recog_greeting]: '{combined_text}' didn't parse "
                  f"yet — holding {STITCH_GAP_SEC:.1f}s for a possible "
                  f"continuation", flush=True)
            return True

        # If the user said only a bare name ("Joy") rather than a full
        # intro phrase ("My name is Joy"), Whisper is prone to single-
        # word substitutions ("Joy" -> "Enjoy" / "Destroy"). Route through
        # an explicit confirmation step before we persist anything to
        # disk. Multi-word intro-phrase transcripts skip this — they
        # carry enough acoustic context that Whisper rarely substitutes.
        if not _has_intro_phrase(combined_text):
            self._enter_confirming(now, name_raw)
            return True

        return self._commit_enrollment(now, name_raw)

    # ── Enrollment commit (shared by direct + post-confirmation paths) ───

    def _commit_enrollment(self, now: float, name_raw: str) -> bool:
        """Persist the top-K face captures under the name, queue the
        "Nice to meet you" TTS, and open the correction window.

        Called from two places:
          * submit_transcript() directly when an intro-phrase transcript
            ("My name is X") parses — high-confidence path, no confirm.
          * after a bare-name transcript has been confirmed ("Did you
            say X?" → "Yes" / timeout / "No, Y") in
            _handle_confirmation_reply / the tick timeout.
        """
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

        enrolled = self._perception.enroll_person(dir_name, save_dir)
        self._enroll_state["active"] = False
        self._confirm_state["active"] = False

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

    # ── Confirmation: "Did you say {name}?" ──────────────────────────────

    def _enter_confirming(self, now: float, name_raw: str) -> None:
        """Pause enrollment and ask the user to confirm a bare-name parse.

        Sets _enroll_state inactive (so the enrollment timeout stops
        ticking) and _confirm_state active (so the confirmation timeout
        starts). The phase stays INTRODUCING — _tick_introducing won't
        exit to COOLDOWN while either of these is active.
        """
        dir_name     = name_raw.lower().replace(" ", "_")
        display_name = name_raw.title()
        try:
            self._say_queue.put_nowait(f"Did you say {display_name}?")
        except Exception as e:
            print(f"Task[recog_greeting]: WARN say_queue full, "
                  f"confirmation not queued: {e}", flush=True)
        # Stale stitch buffer would otherwise glue onto the yes/no reply.
        self._enroll_pending = {"text": None, "speech_end_t": 0.0,
                                "expires_at": 0.0}
        # Pause the enrollment-side timeout; confirmation has its own.
        self._enroll_state["active"] = False
        self._confirm_state.update({
            "active":       True,
            "dir_name":     dir_name,
            "display_name": display_name,
            "asked_at":     now,
        })
        print(f"Task[recog_greeting]: bare-name parse '{display_name}' "
              f"-> confirming ('Did you say {display_name}?')", flush=True)

    def _handle_confirmation_reply(self, now: float, text: str,
                                   capture_t: float) -> bool:
        """Route a transcript that arrived during the confirmation window.

        Returns True iff the transcript was consumed (taken to mean a
        yes/no/correction reply), False to let the caller fall through.

        "yes" / timeout (handled in tick)  → commit the original name.
        "no" with new_name                  → commit the new name.
        "no" alone                          → re-prompt for the name
                                             (consumes an attempt).
        "unclear"                           → don't consume, keep waiting.
        """
        asked_at = self._confirm_state["asked_at"]
        # Lookback small (1 s) — the user can't have replied to the prompt
        # before it started playing, but VAD-stamped capture_t can sit
        # slightly earlier than the TTS-end if the user spoke during the
        # tail. Drop anything older than that.
        if capture_t < asked_at - 1.0:
            return False
        if capture_t > asked_at + CONFIRM_TIMEOUT_SEC:
            # Tick should have already fired the timeout; defensive.
            return False

        verdict, new_name = _parse_confirmation(text)
        display_name = self._confirm_state["display_name"]
        if verdict == "yes":
            print(f"Task[recog_greeting]: confirmation 'yes' for "
                  f"'{display_name}' (heard {text!r})", flush=True)
            self._commit_enrollment(now, display_name)
            return True
        if verdict == "no":
            if new_name:
                print(f"Task[recog_greeting]: confirmation 'no' for "
                      f"'{display_name}' with correction '{new_name}' "
                      f"(heard {text!r})", flush=True)
                # New name is itself from a fresh transcript; if THAT
                # came in via a bare-name path inside the "no, X" form,
                # we still commit directly — the user already gave us
                # one round of explicit confirmation context.
                self._commit_enrollment(now, new_name)
                return True
            # "no" alone — reopen enrollment listening and apologise.
            # Reactivate enroll_state so _apologise_and_arm_retry_or_close
            # treats this as a fresh attempt-failure and re-asks.
            print(f"Task[recog_greeting]: confirmation 'no' for "
                  f"'{display_name}' without correction (heard {text!r}) "
                  f"-> re-asking", flush=True)
            self._confirm_state["active"] = False
            self._enroll_state["active"]  = True
            self._enroll_state["asked_at"] = now
            self._apologise_and_arm_retry_or_close(
                now, "user rejected confirmed name")
            return True
        # Unclear — keep waiting for a clearer reply or the timeout.
        print(f"Task[recog_greeting]: confirmation reply unclear "
              f"(heard {text!r}) — keep waiting", flush=True)
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

        self._perception.rename_person(old_dir, new_dir)

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

    # ── Phase advance helpers ────────────────────────────────────────────

    def _tick_idle(self, now, motion_seen, img):
        """Wait for a trigger — either a face already visible at level (skip
        seeking) or motion+person (look up to find a face)."""
        if img is not None and self._perception.last_complete_faces:
            self._phase                  = RECOGNIZING
            self._phase_entered          = now
            self._perception.reset_recognizing()
            self._recog_stabilize_until  = 0.0
            self._perception.last_complete_seen     = now
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
        if self._perception.last_complete_faces:
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
            self._perception.reset_recognizing()
            self._recog_stabilize_until = (now + stabilize_sec
                                           if stabilize_sec > 0 else 0.0)
            self._perception.last_complete_seen    = now
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
            self._perception.last_motion_time = now
            self._phase            = COOLDOWN
            self._phase_entered    = now
            print("Task[recog_greeting]: seek give-up -> cooldown", flush=True)
            result.status_event = STATUS_FAILURE
        return result

    def _tick_recognizing(self, now, img, result: TickResult) -> TickResult:
        decide_now = (len(self._perception.recog_obs) >= RECOG_VOTES_REQUIRED
                      or now - self._phase_entered >= RECOG_TIMEOUT_SEC)
        face_lost  = now - self._perception.last_complete_seen > FACE_LOST_TIMEOUT_SEC

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
        self._perception.last_motion_time  = now
        print(f"Task[recog_greeting]: leaving recognizing, queueing "
              f"{'wait_for_tts -> ' if after_tts else ''}{recovery}",
              flush=True)

    def _dispatch_verdict(self, now, img, result: TickResult) -> TickResult:
        verdict, kind = self._perception.tally()
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
        self._perception.last_motion_time      = now

        # Prefer the live frame for the zoom label, fall back to the captured
        # snapshot if the person has already moved out of frame.
        if self._perception.last_complete_faces and img is not None:
            biggest = max(self._perception.last_complete_faces, key=lambda f: f[2] * f[3])
            crop = zoom_crop(img, biggest)
        elif self._perception.recog_best_face is not None:
            crop = zoom_crop(self._perception.recog_best_face["img"],
                             self._perception.recog_best_face["bbox"])
        else:
            crop = None

        if crop is not None:
            result.display_request = DisplayRequest(
                image=label_face_zoom(crop, verdict),
                duration_sec=CAPTURE_DISPLAY_SEC)

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
        if (self._perception.recog_best_face is None
                or self._perception.recog_best_face["sharpness"] < SHARPNESS_THRESHOLD):
            # Let the person know why we're not engaging — but only if we
            # haven't said it recently. Without this cooldown, the same
            # too-blurry face standing in front of Pella triggers the
            # apology every ~5-10 s.
            apologise = (now - self._last_see_complaint_at
                         >= SEE_COMPLAINT_COOLDOWN)
            if apologise:
                try:
                    self._say_queue.put_nowait(
                        "Sorry, I cannot see you clearly.")
                    self._last_see_complaint_at = now
                except Exception:
                    apologise = False
            # `after_tts` only matters when we actually queued TTS —
            # otherwise the recovery action can fire immediately.
            self._queue_recovery(now, after_tts=apologise)
            self._phase         = COOLDOWN
            self._phase_entered = now
            sharp_str = (f"{self._perception.recog_best_face['sharpness']:.1f}"
                         if self._perception.recog_best_face else "none")
            silent = "" if apologise else \
                f" (silent — within {SEE_COMPLAINT_COOLDOWN:.0f}s cooldown)"
            print(f"Task[recog_greeting]: unknown face not sharp enough "
                  f"(best sharpness={sharp_str}, threshold={SHARPNESS_THRESHOLD})"
                  f"{silent} -> cooldown", flush=True)
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
        # via perception.capture_enroll_candidate across the introducing window.
        self._perception.seed_enroll_candidates({
            "frame":     self._perception.recog_best_face["img"],
            "landmarks": self._perception.recog_best_face["landmarks"],
            "bbox":      self._perception.recog_best_face["bbox"],
            "sharpness": self._perception.recog_best_face["sharpness"],
        })
        # Reset any leftover stitch buffer from a prior attempt.
        self._enroll_pending = {"text": None, "speech_end_t": 0.0,
                                "expires_at": 0.0}
        self._enroll_state.update({"active": True, "asked_at": now,
                                   "attempts": 1})
        # Any leftover confirmation state from a prior introducing
        # session would otherwise gate cooldown forever.
        self._confirm_state["active"] = False
        self._last_introduced  = now
        self._perception.last_motion_time = now

        crop = zoom_crop(self._perception.recog_best_face["img"],
                         self._perception.recog_best_face["bbox"])
        result.display_request = DisplayRequest(
            image=crop, duration_sec=CAPTURE_DISPLAY_SEC)

        self._phase          = INTRODUCING
        self._phase_entered  = now
        result.status_event  = STATUS_NEW_PERSON
        print(f"Task[recog_greeting]: introducing, asking for name "
              f"(sharpness={self._perception.recog_best_face['sharpness']:.1f})",
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
        # Driven by submit_transcript / the enrollment + confirmation
        # timeout checks above; the commit and apology paths flip both
        # _enroll_state["active"] and _confirm_state["active"] back to
        # False. Stay in INTRODUCING while either is still live so a
        # bare-name parse waiting on "Did you say X?" doesn't leak into
        # COOLDOWN before the user has a chance to reply.
        if not self._enroll_state["active"] \
                and not self._confirm_state["active"]:
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
