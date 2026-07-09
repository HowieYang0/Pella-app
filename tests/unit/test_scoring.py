"""Tests for the enrollment candidate scoring functions.

Both scoring functions are stateless — they take a candidate dict and
return a numeric score (plus a reason string for the gated one). We
exercise each hard quality gate and confirm ``ok`` candidates get a
reasonable score in [0, 1].

The candidate dict shape is what capture_enroll_candidate produces:
    {
      "frame":     np.ndarray  (BGR),
      "landmarks": np.ndarray  (shape (5, 2) — YuNet's 5 landmark points),
      "bbox":      (x, y, w, h),
      "sharpness": float,
    }
"""

import numpy as np
import pytest

from perception import (
    _enroll_candidate_score,
    _enroll_candidate_relaxed_score,
)
from vision import (
    ENROLL_MAX_YAW, ENROLL_MAX_ROLL, ENROLL_MIN_IOD,
    ENROLL_BRIGHT_LO, ENROLL_BRIGHT_HI,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def make_candidate(sharpness=150.0,
                   iod=120,             # px between eyes
                   yaw_offset=0.0,      # nose offset from eye midpoint / iod
                   roll_offset=0.0,     # eye Y difference / iod
                   frame_mean=128,      # grey value of the whole frame
                   frame_std_target=70, # noise half-range so gray std clears
                                        # ENROLL_BRIGHT_MIN_STD (=25); channel
                                        # averaging into gray costs ~35 % of
                                        # the per-channel std, so per-channel
                                        # noise has to overshoot the target.
                   landmarks=True,
                   bbox=(50, 50, 200, 200)):
    """Build a candidate dict that starts fully passing all gates, then
    let each test dial in the one failure mode it cares about.

    landmarks = [ left_eye, right_eye, nose, mouth_l, mouth_r ]

    Eye row placed at y=100; nose horizontally offset by yaw_offset*iod
    from the eye midpoint; right eye vertically offset by roll_offset*iod
    to simulate roll.
    """
    if landmarks:
        left_x  = 100
        right_x = left_x + iod
        eye_y   = 100
        mid_x   = (left_x + right_x) * 0.5
        nose_x  = mid_x + yaw_offset * iod
        nose_y  = 130
        right_y = eye_y + roll_offset * iod
        lm = np.array([
            [left_x,  eye_y],
            [right_x, right_y],
            [nose_x,  nose_y],
            [90,      160],
            [right_x, 160],
        ], dtype=np.float32)
    else:
        lm = None

    # Build a frame that fills the bbox with a mid-grey plus a bit of
    # texture so the std check doesn't trip. Simple approach: BGR frame
    # with a grid pattern of value near frame_mean.
    bx, by, bw, bh = bbox
    frame = np.full((by + bh + 20, bx + bw + 20, 3),
                    frame_mean, dtype=np.uint8)
    # Sprinkle noise to bring the std above the "flat" gate.
    rng = np.random.default_rng(0)
    noise = rng.integers(-frame_std_target, frame_std_target + 1,
                         size=(bh, bw, 3), dtype=np.int16)
    region = frame[by:by + bh, bx:bx + bw].astype(np.int16) + noise
    frame[by:by + bh, bx:bx + bw] = np.clip(region, 0, 255).astype(np.uint8)

    return {
        "sharpness": sharpness,
        "landmarks": lm,
        "bbox":      bbox,
        "frame":     frame,
    }


# ── _enroll_candidate_score — hard-gate failures ────────────────────────────

def test_score_no_landmarks_fails():
    """Without landmarks we can't compute pose or IOD — reject outright."""
    c = make_candidate(landmarks=False)
    score, reason = _enroll_candidate_score(c)
    assert score == float("-inf")
    assert reason == "no_landmarks"


def test_score_iod_too_small_fails():
    """Face too far / too small — embedding quality collapses."""
    c = make_candidate(iod=int(ENROLL_MIN_IOD) - 5)
    score, reason = _enroll_candidate_score(c)
    assert score == float("-inf")
    assert reason.startswith("iod=")


def test_score_yaw_too_large_fails():
    """Head turned too far sideways — nose way off the eye midpoint."""
    c = make_candidate(yaw_offset=ENROLL_MAX_YAW + 0.1)
    score, reason = _enroll_candidate_score(c)
    assert score == float("-inf")
    assert reason.startswith("yaw=")


def test_score_roll_too_large_fails():
    """Head tilted too much — one eye significantly higher than the other."""
    c = make_candidate(roll_offset=ENROLL_MAX_ROLL + 0.05)
    score, reason = _enroll_candidate_score(c)
    assert score == float("-inf")
    assert reason.startswith("roll=")


def test_score_dark_fails():
    """Under-exposed face region — dark(mean=...) reject."""
    c = make_candidate(frame_mean=int(ENROLL_BRIGHT_LO) - 20,
                       frame_std_target=5)
    score, reason = _enroll_candidate_score(c)
    assert score == float("-inf")
    assert reason.startswith("dark(")


def test_score_overexposed_fails():
    """Blown-out face region — bright(mean=...) reject."""
    c = make_candidate(frame_mean=int(ENROLL_BRIGHT_HI) + 20,
                       frame_std_target=5)
    score, reason = _enroll_candidate_score(c)
    assert score == float("-inf")
    assert reason.startswith("bright(")


def test_score_flat_fails():
    """Very low std in the face region — washed-out / featureless."""
    c = make_candidate(frame_std_target=0)   # uniform grey inside bbox
    score, reason = _enroll_candidate_score(c)
    assert score == float("-inf")
    assert reason.startswith("flat(")


# ── _enroll_candidate_score — happy path ────────────────────────────────────

def test_score_good_candidate_passes():
    """Well-lit, sharp, frontal face — score in [0, 1] with reason 'ok'."""
    c = make_candidate(sharpness=200.0, iod=120,
                       yaw_offset=0.02, roll_offset=0.02,
                       frame_mean=128)
    score, reason = _enroll_candidate_score(c)
    assert reason == "ok"
    assert 0.0 <= score <= 1.0


def test_score_ranks_sharper_higher():
    """Between two otherwise-identical candidates, sharper wins."""
    dull  = make_candidate(sharpness=80.0)
    sharp = make_candidate(sharpness=300.0)
    dull_score,  _ = _enroll_candidate_score(dull)
    sharp_score, _ = _enroll_candidate_score(sharp)
    assert sharp_score > dull_score


def test_score_ranks_frontal_higher_than_off_axis():
    """A frontal face outranks one with visible yaw offset."""
    frontal  = make_candidate(yaw_offset=0.0, roll_offset=0.0)
    off_axis = make_candidate(yaw_offset=ENROLL_MAX_YAW - 0.02,
                              roll_offset=0.0)
    frontal_score,  _ = _enroll_candidate_score(frontal)
    off_axis_score, _ = _enroll_candidate_score(off_axis)
    assert frontal_score > off_axis_score


# ── _enroll_candidate_relaxed_score (fallback path) ────────────────────────

def test_relaxed_score_without_landmarks_uses_half_sharpness():
    """No landmarks -> we can only trust sharpness (halved as a penalty)."""
    c = make_candidate(sharpness=100.0, landmarks=False)
    score = _enroll_candidate_relaxed_score(c)
    assert score == pytest.approx(50.0)


def test_relaxed_score_with_frontal_landmarks_beats_no_landmarks():
    """Frontal face with landmarks scores above the half-sharpness baseline."""
    c_frontal  = make_candidate(sharpness=100.0, yaw_offset=0.0)
    c_no_lm    = make_candidate(sharpness=100.0, landmarks=False)
    assert _enroll_candidate_relaxed_score(c_frontal) > \
           _enroll_candidate_relaxed_score(c_no_lm)


def test_relaxed_score_ranks_frontal_higher_than_off_axis():
    """The relaxed scorer still prefers frontal to off-axis."""
    c_frontal  = make_candidate(sharpness=100.0, yaw_offset=0.0)
    c_off_axis = make_candidate(sharpness=100.0, yaw_offset=0.4)
    assert _enroll_candidate_relaxed_score(c_frontal) > \
           _enroll_candidate_relaxed_score(c_off_axis)


def test_relaxed_score_ranks_sharper_higher():
    """Sharper wins in the fallback path too."""
    dull  = make_candidate(sharpness=80.0)
    sharp = make_candidate(sharpness=300.0)
    assert _enroll_candidate_relaxed_score(sharp) > \
           _enroll_candidate_relaxed_score(dull)
