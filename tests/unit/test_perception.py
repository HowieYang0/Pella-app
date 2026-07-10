"""Tests for stateful methods on Perception.

Covers the perception surface the task actually consumes:
  - tally() — vote-buffer aggregation
  - reset_recognizing() — RECOGNIZING-phase teardown
  - seed_enroll_candidates() — INTRODUCING-phase seeding
  - capture_enroll_candidate() — the sharpness gate at buffer-append time
  - known_names() — pass-through into the recognizer
  - accumulate_vote() — the biggest-face-in-bundle picking logic
  - pull_latest_recognition() — draining the worker output queue

The recognition worker thread and all sensing (frame intake, motion,
face detection) are integration concerns — they need real video frames
or heavy mocking, so we exercise them on the dock, not in unit tests.
"""

import threading
from unittest.mock import Mock

import numpy as np
import pytest

from perception import (
    Perception, RECOG_VOTES_REQUIRED, RECOG_AGREE_MIN, CAPTURE_SHARPNESS_MIN,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def stop_event():
    return threading.Event()


@pytest.fixture
def perc_no_recognizer(stop_event):
    """A Perception with no recognizer — worker thread not started, so
    tests never touch the queues in a way that races with a live
    thread."""
    return Perception(recognizer=None, stop_event=stop_event)


@pytest.fixture
def perc_with_fake_recognizer(stop_event):
    """A Perception with a Mock recognizer. The .known attribute stands in
    for the real recognizer's dir-name-keyed embedding map."""
    recognizer = Mock()
    recognizer.known = {"daddy": None, "joy": None, "william": None}
    # Turn off the worker-thread startup by pretending recognizer is None
    # for that one branch. The cleanest way: install after Perception
    # constructs. We do the reverse — construct with recognizer=None,
    # then attach — so no thread starts.
    p = Perception(recognizer=None, stop_event=stop_event)
    p._recognizer = recognizer
    return p


# ── tally() ─────────────────────────────────────────────────────────────────

def test_tally_empty_is_ambiguous(perc_no_recognizer):
    """No observations at all — nothing to commit to."""
    verdict, kind = perc_no_recognizer.tally()
    assert verdict is None
    assert kind == "ambiguous"


def test_tally_single_below_threshold(perc_no_recognizer):
    """One vote for 'daddy' is not enough to pass RECOG_AGREE_MIN=2."""
    perc_no_recognizer.recog_obs.append("daddy")
    verdict, kind = perc_no_recognizer.tally()
    assert verdict is None
    assert kind == "ambiguous"


def test_tally_majority_known(perc_no_recognizer):
    """Two agreeing 'daddy' votes -> known verdict."""
    perc_no_recognizer.recog_obs.extend(["daddy", "daddy"])
    verdict, kind = perc_no_recognizer.tally()
    assert verdict == "daddy"
    assert kind == "known"


def test_tally_majority_unknown(perc_no_recognizer):
    """Two 'None' votes agreeing -> unknown verdict."""
    perc_no_recognizer.recog_obs.extend([None, None])
    verdict, kind = perc_no_recognizer.tally()
    assert verdict is None
    assert kind == "unknown"


def test_tally_split_no_majority(perc_no_recognizer):
    """Three-way split with no majority -> ambiguous."""
    # RECOG_VOTES_REQUIRED = 3, RECOG_AGREE_MIN = 2. Three different
    # verdicts each at count 1, so no name reaches the agree threshold.
    perc_no_recognizer.recog_obs.extend(["daddy", "joy", None])
    verdict, kind = perc_no_recognizer.tally()
    assert verdict is None
    assert kind == "ambiguous"


def test_tally_known_beats_unknown(perc_no_recognizer):
    """Two 'daddy' + one None -> the known name wins, no dilution."""
    perc_no_recognizer.recog_obs.extend(["daddy", "daddy", None])
    verdict, kind = perc_no_recognizer.tally()
    assert verdict == "daddy"
    assert kind == "known"


# ── reset_recognizing() ─────────────────────────────────────────────────────

def test_reset_recognizing_clears_state(perc_no_recognizer):
    """Both the vote buffer and the best-face snapshot are wiped."""
    p = perc_no_recognizer
    p.recog_obs.extend(["daddy", "daddy"])
    p.recog_best_face = {"img": None, "bbox": (0, 0, 10, 10),
                         "landmarks": None, "sharpness": 42.0}

    p.reset_recognizing()

    assert len(p.recog_obs) == 0
    assert p.recog_best_face is None


def test_reset_recognizing_no_op_when_already_clean(perc_no_recognizer):
    """Idempotent — resetting an empty session doesn't blow up."""
    p = perc_no_recognizer
    p.reset_recognizing()
    p.reset_recognizing()   # again
    assert len(p.recog_obs) == 0
    assert p.recog_best_face is None


# ── seed_enroll_candidates() ────────────────────────────────────────────────

def test_seed_enroll_candidates_replaces_buffer(perc_no_recognizer):
    """Seeding wipes any prior candidates and drops in the new entry.

    (The alternative — appending — would leak the previous session's
    candidates into the new INTRODUCING window.)
    """
    p = perc_no_recognizer
    # Leftover from a prior session
    p.enroll_candidates.append({"stale": True})
    p.enroll_candidates.append({"stale": True})

    seed = {"frame": None, "landmarks": None,
            "bbox": (0, 0, 10, 10), "sharpness": 100.0}
    p.seed_enroll_candidates(seed)

    assert len(p.enroll_candidates) == 1
    assert p.enroll_candidates[0] is seed


# ── known_names() ───────────────────────────────────────────────────────────

def test_known_names_empty_without_recognizer(perc_no_recognizer):
    """No recognizer means no known names — get_warm_phrases skips
    the per-name TTS pre-cache entirely."""
    assert perc_no_recognizer.known_names() == []


def test_known_names_returns_dir_names(perc_with_fake_recognizer):
    """The names are the recognizer's 'known' dict keys (dir_name strings)."""
    names = perc_with_fake_recognizer.known_names()
    assert set(names) == {"daddy", "joy", "william"}


def test_known_names_returns_new_list_each_call(perc_with_fake_recognizer):
    """Returning a fresh list means callers can safely iterate + mutate
    the recognizer's known map between calls without invalidating an
    iterator."""
    n1 = perc_with_fake_recognizer.known_names()
    n2 = perc_with_fake_recognizer.known_names()
    assert n1 is not n2   # different list objects
    assert n1 == n2       # same contents


# ── accumulate_vote() ───────────────────────────────────────────────────────

def _mk_face(x, y, w, h):
    """Tuple shape that vision.detect_faces returns."""
    return (x, y, w, h)


def test_accumulate_vote_no_data_is_noop(perc_no_recognizer):
    """Without last_rec_faces / frame_w / frame_h the vote is skipped."""
    p = perc_no_recognizer
    p.accumulate_vote()
    assert len(p.recog_obs) == 0


def test_accumulate_vote_picks_biggest_face_name(perc_no_recognizer):
    """When multiple faces are in the bundle, vote for the biggest one's
    identity — the biggest is the person likely engaging with Pella."""
    p = perc_no_recognizer
    p.frame_w = 640
    p.frame_h = 480
    p.last_rec_faces = [
        _mk_face(50,  50,  40, 40),    # small
        _mk_face(200, 100, 200, 200),  # BIGGEST
        _mk_face(400, 300, 60, 60),    # medium
    ]
    p.last_rec_names = ["someone_else", "daddy", "unknown"]

    p.accumulate_vote()

    assert list(p.recog_obs) == ["daddy"]


def test_accumulate_vote_ignores_edge_faces(perc_no_recognizer):
    """Faces clipped by the frame edge don't get votes cast for them —
    they'd give unreliable embeddings anyway."""
    p = perc_no_recognizer
    p.frame_w = 640
    p.frame_h = 480
    # Only "face" is against the left edge — is_face_at_edge should
    # filter it out and leave nothing to vote on.
    p.last_rec_faces = [_mk_face(0, 100, 60, 60)]
    p.last_rec_names = ["daddy"]

    p.accumulate_vote()

    assert len(p.recog_obs) == 0


def test_accumulate_vote_handles_none_name(perc_no_recognizer):
    """An unknown-identity biggest face casts a None vote — the tally
    then reads it as evidence for 'unknown'."""
    p = perc_no_recognizer
    p.frame_w = 640
    p.frame_h = 480
    p.last_rec_faces = [_mk_face(200, 100, 150, 150)]
    p.last_rec_names = [None]

    p.accumulate_vote()

    assert list(p.recog_obs) == [None]


# ── pull_latest_recognition() ───────────────────────────────────────────────

def test_pull_latest_recognition_empty_returns_false(perc_no_recognizer):
    """No queue traffic -> nothing to consume."""
    assert perc_no_recognizer.pull_latest_recognition() is False


def test_pull_latest_recognition_keeps_only_newest(perc_no_recognizer):
    """Drain the queue and hold on to the newest bundle. If the worker
    put 3 bundles while the task was busy, we skip the two stale ones.

    Production uses maxsize=1, so this scenario is defensive rather than
    load-bearing — but the ``while True: get_nowait()`` loop in
    pull_latest_recognition() is written to handle it either way, so
    swap in a bigger queue and prove it does."""
    from queue import Queue
    p = perc_no_recognizer
    p._rec_out = Queue(maxsize=10)
    p._rec_out.put_nowait(([_mk_face(0, 0, 10, 10)], ["oldest"]))
    p._rec_out.put_nowait(([_mk_face(0, 0, 20, 20)], ["middle"]))
    p._rec_out.put_nowait(([_mk_face(0, 0, 30, 30)], ["newest"]))

    got = p.pull_latest_recognition()

    assert got is True
    assert p.last_rec_names == ["newest"]


# ── capture_enroll_candidate — sharpness gate ──────────────────────────────

def _mk_face_with_landmarks(x, y, w, h):
    """Face tuple with a plausible 5-point landmark array so the biggest-face
    picker can index it. Contents don't matter — capture_enroll_candidate
    doesn't inspect them, it just carries them into the buffer entry."""
    lm = np.array([[x + w * 0.35, y + h * 0.40],
                   [x + w * 0.65, y + h * 0.40],
                   [x + w * 0.50, y + h * 0.55],
                   [x + w * 0.40, y + h * 0.70],
                   [x + w * 0.60, y + h * 0.70]], dtype=np.float32)
    return (x, y, w, h, lm)


def test_capture_drops_blurry_frame(perc_no_recognizer):
    """A uniform-grey face region has Laplacian variance ~= 0 — well
    below CAPTURE_SHARPNESS_MIN — and must be dropped rather than
    consuming a slot in the fixed-size candidate deque."""
    p = perc_no_recognizer
    p.last_complete_faces = [_mk_face_with_landmarks(10, 10, 60, 60)]
    flat_frame = np.full((200, 200, 3), 128, dtype=np.uint8)

    assert len(p.enroll_candidates) == 0
    p.capture_enroll_candidate(flat_frame)
    assert len(p.enroll_candidates) == 0


def test_capture_appends_sharp_frame(perc_no_recognizer):
    """A high-variance (noise) face region easily clears
    CAPTURE_SHARPNESS_MIN and lands in the buffer with all metadata."""
    p = perc_no_recognizer
    face = _mk_face_with_landmarks(10, 10, 60, 60)
    p.last_complete_faces = [face]
    rng = np.random.default_rng(42)
    sharp_frame = rng.integers(0, 256, size=(200, 200, 3),
                               dtype=np.uint8).astype(np.uint8)

    p.capture_enroll_candidate(sharp_frame)

    assert len(p.enroll_candidates) == 1
    entry = p.enroll_candidates[0]
    assert entry["bbox"] == (10, 10, 60, 60)
    assert entry["sharpness"] > CAPTURE_SHARPNESS_MIN
    assert entry["landmarks"] is face[4]


# ── Constants sanity ────────────────────────────────────────────────────────

def test_recog_constants_relationship():
    """Sanity check that the vote / agree constants make sense together.

    RECOG_VOTES_REQUIRED should be at least RECOG_AGREE_MIN, otherwise
    tally() could hit the vote cap before enough agreement is possible.
    """
    assert RECOG_VOTES_REQUIRED >= RECOG_AGREE_MIN >= 1
