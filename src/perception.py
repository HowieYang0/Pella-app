#!/usr/bin/env python3
"""Perception layer for the recog_greeting task.

Owns the recognition worker thread and every per-frame sensing detail
tick() consults each iteration:

  * face + motion detection (via vision.py primitives)
  * hand-off to the ArcFace recognition worker + result draining
  * RECOGNIZING-phase state: rolling vote buffer + sharpest-face tracker
  * INTRODUCING-phase candidate ring buffer + ISO/IEC 29794-5-style
    quality-gated top-K selection for multi-template enrollment
  * enrollment + rename wrappers around the FaceRecognizer instance

Everything cv2 / numpy / image-related lives here or in vision.py. The
state machine in recog_greeting.py consumes Perception via method calls
and public read attributes; it never touches cv2 directly.
"""

import os
import threading
from collections import Counter, deque
from queue import Queue, Empty
from typing import Optional

import cv2
import numpy as np

import face_recognizer
from vision import (
    FACE_DETECT_EVERY, FACE_IDS_DIR,
    ENROLL_BUFFER_SIZE, ENROLL_TOP_K,
    ENROLL_MAX_YAW, ENROLL_MAX_ROLL, ENROLL_MIN_IOD,
    ENROLL_BRIGHT_LO, ENROLL_BRIGHT_HI, ENROLL_BRIGHT_MIN_STD,
    detect_faces, detect_motion, detect_person, is_face_at_edge,
    sharpness, recognition_worker,
)


# ── Recognition vote thresholds ─────────────────────────────────────────────
# How many observations to gather in RECOGNIZING before committing to a
# verdict, and how many must agree. Kept here (not in vision.py) because
# they describe how this module aggregates votes, not the vision primitives.
RECOG_VOTES_REQUIRED = 3
RECOG_AGREE_MIN      = 2


# ── Enrollment candidate scoring (was static on RecogGreetingTask) ──────────

def _enroll_candidate_score(c: dict) -> tuple:
    """Return (score, reason). score is -inf when a hard gate fails.

    Hard gates (ISO/IEC 29794-5 inspired):
      * yaw  > ENROLL_MAX_YAW       — face turned too far sideways
      * roll > ENROLL_MAX_ROLL      — head tilted too much
      * IOD  < ENROLL_MIN_IOD       — face too small / far away
      * brightness mean outside [LO, HI] or std too low (washed-out)

    Survivors get a weighted sum in [0, 1] of normalised components:
        0.4 * sharpness + 0.3 * pose + 0.2 * IOD + 0.1 * brightness
    Weights follow the FIQA literature's empirical findings
    (sharpness + pose dominate, IOD is a strong tiebreaker, brightness
    small but real).
    """
    s   = c["sharpness"]
    lm  = c["landmarks"]
    if lm is None or len(lm) < 3:
        return float("-inf"), "no_landmarks"

    left_eye, right_eye, nose = lm[0], lm[1], lm[2]
    eye_mid_x = (left_eye[0] + right_eye[0]) * 0.5
    iod       = float(((right_eye[0] - left_eye[0]) ** 2
                       + (right_eye[1] - left_eye[1]) ** 2) ** 0.5)
    if iod < 1e-3:
        return float("-inf"), "iod_zero"

    yaw_offset = abs(nose[0] - eye_mid_x) / iod
    roll_norm  = abs(right_eye[1] - left_eye[1]) / iod

    if iod < ENROLL_MIN_IOD:
        return float("-inf"), f"iod={iod:.0f}<{ENROLL_MIN_IOD:.0f}"
    if yaw_offset > ENROLL_MAX_YAW:
        return float("-inf"), f"yaw={yaw_offset:.2f}>{ENROLL_MAX_YAW:.2f}"
    if roll_norm > ENROLL_MAX_ROLL:
        return float("-inf"), f"roll={roll_norm:.2f}>{ENROLL_MAX_ROLL:.2f}"

    # Brightness check on the bbox region (cheap, no extra detection).
    bx, by, bw, bh = c["bbox"]
    frame = c["frame"]
    face_region = frame[by:by + bh, bx:bx + bw]
    if face_region.size == 0:
        return float("-inf"), "bbox_empty"
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    bright_mean = float(gray.mean())
    bright_std  = float(gray.std())
    if bright_mean < ENROLL_BRIGHT_LO:
        return float("-inf"), f"dark(mean={bright_mean:.0f})"
    if bright_mean > ENROLL_BRIGHT_HI:
        return float("-inf"), f"bright(mean={bright_mean:.0f})"
    if bright_std < ENROLL_BRIGHT_MIN_STD:
        return float("-inf"), f"flat(std={bright_std:.0f})"

    # Normalise each component into [0, 1].
    sharp_n  = min(1.0, s / 200.0)        # 200 ~ "good" sharpness
    pose_n   = ((1.0 - yaw_offset / ENROLL_MAX_YAW)
                * (1.0 - roll_norm / ENROLL_MAX_ROLL))
    iod_n    = min(1.0, iod / 120.0)      # 120 px ~ "good" face size at 720p
    bright_n = max(0.0, 1.0 - abs(bright_mean - 128.0) / 64.0)

    score = (0.4 * sharp_n
             + 0.3 * pose_n
             + 0.2 * iod_n
             + 0.1 * bright_n)
    return score, "ok"


def _enroll_candidate_relaxed_score(c: dict) -> float:
    """Gates-off fallback used only when every candidate is gated out.

    Same intuition as the old single-metric scorer: sharpness times a
    soft frontality factor (yaw only), so the least-bad candidate is
    picked. Kept tiny because it should almost never fire.
    """
    s  = c["sharpness"]
    lm = c["landmarks"]
    if lm is None or len(lm) < 3:
        return s * 0.5
    eye_mid_x  = (lm[0][0] + lm[1][0]) * 0.5
    eye_width  = abs(lm[1][0] - lm[0][0]) + 1e-6
    yaw_offset = abs(lm[2][0] - eye_mid_x) / eye_width
    yaw_q      = max(0.0, 1.0 - yaw_offset / 0.3)
    return s * (0.5 + 0.5 * yaw_q)


# ── Perception class ────────────────────────────────────────────────────────

class Perception:
    """One tick's worth of perceptual state for the recog_greeting task.

    Constructed once by RecogGreetingTask. Starts the recognition worker
    thread if a recognizer was supplied. All state persists across ticks;
    the task calls the mutation methods (``sense_*``, ``pull_*``,
    ``update_*``, ``capture_*``) and reads the resulting attributes.

    Attributes exposed to the task:
      * last_faces / last_complete_faces — most recent face-detector
        output, and the subset that isn't clipped by a frame edge.
      * last_rec_faces / last_rec_names — most recent bundle from the
        recognition worker (name per face, or None for unknown).
      * frame_w / frame_h — dimensions of the frame we last sensed.
      * last_motion_time — monotonic-time of the last motion+person
        trigger. The task also writes to this to throttle motion
        during active phases.
      * last_complete_seen — monotonic-time of the last tick that saw
        a complete (non-edge) face. Used for face-lost timeouts.
      * recog_obs — rolling vote buffer for the current RECOGNIZING
        session (name string or None per vote).
      * recog_best_face — dict with the sharpest complete face seen
        during the current RECOGNIZING session, or None. Keys: img,
        bbox, landmarks, sharpness.
      * enroll_candidates — ring buffer of candidate face captures
        accumulated during the INTRODUCING window.
    """

    def __init__(self, recognizer, stop_event):
        """Wire dependencies and start the recognition worker thread.

        recognizer:  ArcFace recognizer instance, or None to disable
                     recognition (pull_latest_recognition becomes a no-op,
                     enroll_person falls back to saving raw frames).
        stop_event:  pella_main's shutdown flag; the recognition worker
                     thread exits when this is set.
        """
        self._recognizer = recognizer
        self._stop_event = stop_event

        # Recognition worker thread — owns its own I/O channels.
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

        # Per-frame sense state — read by the task each tick.
        self.last_faces           = []
        self.last_complete_faces  = []
        self.last_rec_faces       = []
        self.last_rec_names       = []
        self.frame_w              = 0
        self.frame_h              = 0
        self.last_motion_time     = 0.0
        self.last_complete_seen   = 0.0

        # RECOGNIZING-phase rolling state.
        self.recog_obs            = deque(maxlen=RECOG_VOTES_REQUIRED)
        self.recog_best_face      = None

        # INTRODUCING-phase candidate buffer.
        self.enroll_candidates: deque = deque(maxlen=ENROLL_BUFFER_SIZE)

        # Internal — face-detection cadence + last frame for motion diff.
        self._detect_counter      = 0
        self._prev_frame          = None

    # ── Sensing ──────────────────────────────────────────────────────────

    def sense_motion_and_faces(self, now: float, img: Optional[np.ndarray],
                               motion_cooldown: float) -> bool:
        """Run motion+person detection and the periodic face detector.

        Returns True if motion+person was seen on this tick — the task
        uses this as the IDLE -> SEEKING trigger. motion_cooldown is
        supplied by the task so the debounce policy stays task-side.

        Updates public attributes: last_faces, last_complete_faces,
        frame_w, frame_h, last_motion_time, last_complete_seen.
        """
        if img is None:
            return False

        motion_seen = False
        if (now - self.last_motion_time >= motion_cooldown
                and detect_motion(img, self._prev_frame)
                and detect_person(img)):
            motion_seen = True
            self.last_motion_time = now
        self._prev_frame = img

        # Periodic face detection — every N frames.
        self._detect_counter += 1
        if self._detect_counter >= FACE_DETECT_EVERY:
            self._detect_counter = 0
            self.last_faces = detect_faces(img)
            img_h, img_w = img.shape[:2]
            self.frame_w, self.frame_h = img_w, img_h
            self.last_complete_faces = [
                f for f in self.last_faces
                if not is_face_at_edge(f, img_w, img_h)
            ]
            if self._recognizer and self.last_faces:
                try:
                    self._rec_in.put_nowait((img.copy(), self.last_faces))
                except Exception:
                    pass

        # Fold the "have I seen a complete face this tick?" refresh in
        # here so the task doesn't have to poll last_complete_faces
        # itself every iteration.
        if self.last_complete_faces:
            self.last_complete_seen = now

        return motion_seen

    # ── Recognition worker draining + voting ─────────────────────────────

    def pull_latest_recognition(self) -> bool:
        """Drain rec_out, keeping only the newest bundle. Returns True if
        a new bundle arrived this tick."""
        got = False
        try:
            while True:
                self.last_rec_faces, self.last_rec_names = self._rec_out.get_nowait()
                got = True
        except Empty:
            pass
        return got

    def accumulate_vote(self):
        """Add one recognition vote for the biggest fully-framed face."""
        complete_rec = ([f for f in self.last_rec_faces
                         if not is_face_at_edge(f, self.frame_w, self.frame_h)]
                        if self.frame_w and self.frame_h else [])
        if not complete_rec:
            return
        biggest = max(complete_rec, key=lambda f: f[2] * f[3])
        idx = self.last_rec_faces.index(biggest)
        rec_name = (self.last_rec_names[idx]
                    if idx < len(self.last_rec_names) else None)
        self.recog_obs.append(rec_name)

    def tally(self):
        """Return ``(verdict, kind)`` from the current vote buffer.

        Each observation is a name string (known) or None (unknown).
        Returns (name, "known"), (None, "unknown"), or (None, "ambiguous")
        depending on whether RECOG_AGREE_MIN of the observations agree.
        """
        obs = self.recog_obs
        if not obs:
            return None, "ambiguous"
        val, count = Counter(obs).most_common(1)[0]
        if count < RECOG_AGREE_MIN:
            return None, "ambiguous"
        return val, ("known" if val else "unknown")

    # ── Sharpest-face tracking during RECOGNIZING ────────────────────────

    def update_best_face(self, img: np.ndarray):
        """Track the sharpest complete face this RECOGNIZING session has seen."""
        biggest = max(self.last_complete_faces, key=lambda f: f[2] * f[3])
        bx, by, bw, bh = biggest[:4]
        region = img[by:by + bh, bx:bx + bw]
        if region.size == 0:
            return
        s = sharpness(region)
        if self.recog_best_face is None or s > self.recog_best_face["sharpness"]:
            self.recog_best_face = {
                "img":       img.copy(),
                "bbox":      (bx, by, bw, bh),
                "landmarks": biggest[4] if len(biggest) > 4 else None,
                "sharpness": s,
            }

    def reset_recognizing(self):
        """Clear the vote buffer and best-face tracker.

        Called by the task when entering a fresh RECOGNIZING phase so a
        prior session's votes/captures don't leak into the new attempt.
        """
        self.recog_obs.clear()
        self.recog_best_face = None

    # ── Enrollment candidate buffer ──────────────────────────────────────

    def capture_enroll_candidate(self, img: np.ndarray):
        """Append the current largest detected face to the candidate buffer.

        Replaces the older greedy "keep only the sharpest" approach: by
        retaining many candidates we can score them on more than just
        sharpness at enrollment time (see pick_enroll_top_k).
        """
        biggest = max(self.last_complete_faces, key=lambda f: f[2] * f[3])
        bx, by, bw, bh = biggest[:4]
        region = img[by:by + bh, bx:bx + bw]
        if region.size == 0:
            return
        s = sharpness(region)
        self.enroll_candidates.append({
            "frame":     img.copy(),
            "landmarks": biggest[4] if len(biggest) > 4 else None,
            "bbox":      (bx, by, bw, bh),
            "sharpness": s,
        })

    def seed_enroll_candidates(self, entry: dict):
        """Insert an initial candidate (typically the RECOGNIZING best-face).

        The task calls this at INTRODUCING entry so the buffer isn't
        empty if the user speaks their name before another face frame
        arrives.
        """
        self.enroll_candidates.clear()
        self.enroll_candidates.append(entry)

    def pick_enroll_top_k(self, k: int = ENROLL_TOP_K) -> list:
        """Score the buffered candidates and return up to ``k`` survivors.

        Each candidate passes through hard ISO/IEC 29794-5-style gates
        (yaw, roll, inter-ocular distance, face-region brightness) before
        being ranked by a weighted sum of normalised sharpness + pose +
        IOD + brightness. Top-K survivors are returned in descending
        score order — they will all be enrolled as distinct embeddings
        under the same identity (multi-template enrollment, the dominant
        production pattern per NIST FRVT).

        If every candidate fails the gates, the top-K by relaxed score
        are returned anyway so a user who said their name doesn't get
        silently dropped; the journal log notes the fallback.
        """
        if not self.enroll_candidates:
            return []

        scored = []
        reject_reasons: Counter = Counter()
        for c in self.enroll_candidates:
            score, reason = _enroll_candidate_score(c)
            scored.append((score, reason, c))
            if score == float("-inf"):
                gate = reason.split("=")[0].split("(")[0]
                reject_reasons[gate] += 1

        survivors = [(s, r, c) for s, r, c in scored if s > float("-inf")]
        rejected  = len(scored) - len(survivors)
        gate_summary = (", ".join(f"{g}={n}"
                                  for g, n in reject_reasons.most_common())
                        or "none")

        if survivors:
            survivors.sort(key=lambda x: x[0], reverse=True)
            top = survivors[:k]
            print(f"Task[recog_greeting]: enrolling top {len(top)} of "
                  f"{len(self.enroll_candidates)} candidates "
                  f"({rejected} rejected by gates: {gate_summary})",
                  flush=True)
        else:
            relaxed = [(_enroll_candidate_relaxed_score(c), c)
                       for c in self.enroll_candidates]
            relaxed.sort(key=lambda x: x[0], reverse=True)
            top = [(score, "fallback", c) for score, c in relaxed[:k]]
            print(f"Task[recog_greeting]: WARN all "
                  f"{len(self.enroll_candidates)} candidates failed "
                  f"quality gates ({gate_summary}); falling back to "
                  f"relaxed score (top {len(top)})", flush=True)

        for i, (score, reason, c) in enumerate(top, 1):
            print(f"  #{i}: score={score:.2f} sharp={c['sharpness']:.0f} "
                  f"({reason})", flush=True)
        return [c for _, _, c in top]

    # ── Recognizer wrappers (task never touches _recognizer directly) ────

    def enroll_person(self, dir_name: str, save_dir: str) -> bool:
        """Persist the top-K candidates as ``dir_name``.

        Returns True if at least one embedding was written (or, when no
        recognizer is loaded, if the fallback frame-save succeeded).
        Falls back to writing raw .jpg frames without embeddings when the
        recognizer is None, so the greeting flow still records a face
        crop for later manual enrollment.
        """
        top_k = self.pick_enroll_top_k()
        if not top_k:
            # Should never happen — the task seeds the buffer with the
            # recog_best_face at INTRODUCING entry. Defensive.
            print("Task[recog_greeting]: no enrollment candidates in "
                  "buffer — skipping", flush=True)
            return False
        if self._recognizer:
            # Multi-template enrollment: save each of the top-K as a
            # distinct embedding under the same identity. Recognition
            # later matches by max cosine across the set, so different
            # poses captured during this single window all contribute.
            #
            # trust_name=True on every call: the user verbally said this
            # name a moment ago, so similarity-vs-existing-embeddings
            # checks would create a circular failure.
            enrolled = False
            for best in top_k:
                ok = self._recognizer.enroll_new(
                    dir_name,
                    best["frame"],
                    best["landmarks"],
                    best["bbox"],
                    save_dir,
                    trust_name=True,
                )
                enrolled = enrolled or ok
            return enrolled
        # No recognizer loaded — save frames only. next_image_index gives
        # each a distinct NNN.jpg so repeat enrollments don't overwrite.
        os.makedirs(save_dir, exist_ok=True)
        for best in top_k:
            idx = face_recognizer.next_image_index(save_dir)
            cv2.imwrite(os.path.join(save_dir, f"{idx:03d}.jpg"),
                        best["frame"])
        return True

    def known_names(self):
        """Return the list of currently-enrolled directory names.

        Empty when no recognizer is loaded. Used by the task at startup to
        build the "Hi, X" / "Nice to meet you, X!" TTS pre-warm list.
        """
        if self._recognizer is None:
            return []
        return list(getattr(self._recognizer, "known", {}))

    def rename_person(self, old_dir: str, new_dir: str) -> None:
        """Rename an enrolled person's directory + embeddings.

        Delegates to the recognizer's rename() when one is loaded (which
        also updates its in-memory embedding map). Falls back to a
        filesystem-only rename otherwise, keeping the same merge-if-target-
        exists behaviour the task used to perform inline.
        """
        if self._recognizer is not None:
            self._recognizer.rename(old_dir, new_dir)
            return
        import shutil
        src = os.path.join(FACE_IDS_DIR, old_dir)
        dst = os.path.join(FACE_IDS_DIR, new_dir)
        if os.path.isdir(src) and not os.path.isdir(dst):
            os.rename(src, dst)
        elif os.path.isdir(src) and os.path.isdir(dst):
            for fname in os.listdir(src):
                shutil.move(os.path.join(src, fname), dst)
            try:
                os.rmdir(src)
            except OSError:
                pass
