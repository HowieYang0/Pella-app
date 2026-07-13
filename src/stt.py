#!/usr/bin/env python3
"""Speech-to-text pipeline: Whisper transcription, VAD, and USB mic capture."""

import concurrent.futures
import os
import re
import threading
import time
from collections import deque
from queue import Queue, Empty
from typing import Optional

import numpy as np
import soxr
import wave
import webrtcvad as _webrtcvad

try:
    from scipy.signal import butter, sosfilt
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import noisereduce as _nr
    _HAS_NR = True
except ImportError:
    _nr = None
    _HAS_NR = False

# ── STT / VAD constants ────────────────────────────────────────────────────────
ROBOT_SAMPLE_RATE     = 48000
ASR_CHANNELS          = 2
ASR_SAMPLE_RATE       = 16000
NO_SPEECH_THRESHOLD   = 0.3    # Whisper s no_speech_prob cutoff. Lowered from
                               # 0.5 -> 0.3 to be more forgiving of marginal
                               # speech (quiet voice at distance, brief
                               # utterances inside a noisier 6 s VAD buffer).
                               # Trade-off: slightly more hallucinations on
                               # pure-noise clips — those are now caught by
                               # the name-confirmation flow in
                               # recog_greeting and by the echo filter
                               # below for TTS-leakage cases.
TTS_MUTE_SEC          = 2.0    # mute mic this long after TTS plays. This is
                               # the FALLBACK floor — only used when
                               # rt/audiohub/player/state can t shorten the
                               # mute via the actual stopped event (e.g.
                               # subscription drop, Wi-Fi hiccup). Lowered
                               # from 4 -> 2 because the state-topic-based
                               # shortening makes the typical mute end at
                               # actual_audio_end + REVERB_PAD_SEC, so the
                               # floor only matters in the failure case.
SPEAKER_VOLUME        = 7      # robot speaker volume 0–10

VAD_AGGRESSIVENESS    = 1      # webrtcvad: 0 = permissive, 3 = strict
                               # Lowered from 3 → 1 after new mic bracket
                               # introduced LiDAR-pulse background; at level 3
                               # webrtcvad was rejecting voice frames as
                               # "not speech enough" amid the noise.
VAD_FRAME_MS          = 20     # webrtcvad frame size (10, 20, or 30 ms)
VAD_FRAME_SAMPLES     = int(VAD_FRAME_MS / 1000 * ROBOT_SAMPLE_RATE)  # 960 @ 48 kHz
VAD_SPEECH_FRAMES     = 3      # consecutive speech frames to enter speech mode
VAD_SILENCE_FRAMES    = 50     # consecutive silence frames to end utterance (~1 s)
VAD_PRE_ROLL_FRAMES   = 8      # frames kept before onset to avoid clipping word starts
MIN_SPEECH_SEC        = 0.8    # discard clips shorter than this
MAX_SPEECH_SEC        = 6.0    # hard cap. Reduced from 15 -> 6 for the new mic
                               # bracket: LiDAR noise keeps the VAD in "speech"
                               # mode continuously, so utterances were only
                               # ever flushing at the 15 s cap. 6 s is enough
                               # for "My name is William" + a beat of silence,
                               # and the resulting clip transcribes in Whisper
                               # in ~3-4 s instead of ~10.
MAX_CLIP_AGE_SEC      = 15.0   # skip Whisper for any clip whose speech_start_t
                               # is already this old by the time the serial
                               # asr_executor reaches it. With max_workers=1
                               # and 1-3 s Whisper latency per clip, queues
                               # can grow several seconds during a noisy
                               # listening window — anything stale will be
                               # rejected by the consumer anyway.
NOISE_FLOOR_EMA_ALPHA = 0.005  # slow EMA adaptation (~200 frames ≈ 4 s)
NOISE_FLOOR_INIT      = 1500.0 # starting estimate for robot motor noise floor
NOISE_FLOOR_FACTOR    = 1.4    # RMS must exceed floor × factor to count as speech
VAD_WARMUP_FRAMES     = 500    # ~10 s: fast-adapt EMA but suppress all triggers

# ── USB microphone (Samson Meteor on dock) ────────────────────────────────────
USE_USB_MIC           = True     # True = PyAudio USB mic; False = robot WebRTC mic
USB_MIC_DEVICE_NAME   = "Samson" # substring match against PyAudio device names
USB_MIC_SAMPLE_RATE   = 16000    # capture at ASR rate — no resampling needed
USB_MIC_CHANNELS      = 1
USB_NOISE_FLOOR_INIT   = 300.0   # lower starting estimate — no motors on the dock
USB_NOISE_FLOOR_CEIL   = 600.0   # hard ceiling on the adaptive EMA. Without it the
                                 # floor drifts monotonically up over a long session
                                 # (300 -> 2000+ observed) — leaked speech and
                                 # quiet motor noise contaminate the non-speech
                                 # EMA buckets. Once the floor crosses ~1000 the
                                 # peak/floor SNR collapses and Whisper starts
                                 # rejecting or hallucinating speech that earlier
                                 # in the session would have transcribed cleanly.
USB_NOISE_FLOOR_FACTOR = 1.2     # RMS entry threshold; speech at 3ft peaks well above floor×factor
                                 # Lowered from 1.5 → 1.2 with new mic bracket
                                 # so voice frames clear the entry threshold
                                 # despite a low (327ish) ambient floor.
USB_SILENCE_HOLD_FACTOR = 1.8   # hysteresis: only resets silence counter for strong speech
USB_SPIKE_REJECT       = 8       # skip frames above this × floor (LiDAR impacts, physical contact)
USB_WARMUP_FRAMES      = 100     # ~2 s fast-adapt then normal EMA
USB_NR_PROP_DECREASE   = 0.75    # 0.85 -> 0.75 after real captures showed
                                 # voice harmonics were being over-subtracted
                                 # (filtered output sounded slightly hollow).
                                 # 0.75 preserves more of the voice signal
                                 # while still removing the bulk of the
                                 # stationary fan / LiDAR / hum noise.
USB_VAD_SPEECH_FRAMES  = 2       # consecutive frames needed to enter speech mode (40 ms)
                                 # Lowered from 3 → 2 with new mic bracket so a
                                 # brief vowel onset enters speech mode faster.
USB_VAD_SILENCE_FRAMES = 60      # consecutive frames needed to end utterance (~1.2 s @ 20 ms)
                                 # Reduced from 120 -> 60 with the new mic
                                 # bracket — pauses between words register as
                                 # silence often enough to flush the buffer
                                 # within the enrollment window when there
                                 # genuinely IS a name-shaped utterance.
USB_PRE_ROLL_FRAMES    = 80      # audio kept before onset (~1.6 s) to capture soft sentence starts


# Shared mutable: TTS sets this so the USB mic loop mutes itself during playback.
tts_mute_until = [0.0]

# Shared mutable: TTS sets this to time.monotonic() the moment the audiohub
# player reports is_playing -> false. The USB mic loop reads it once each
# time the mute clears to log the actual gap between Pella s speech ending
# and the mic accepting the first user frame. Used purely for diagnostics —
# tuning REVERB_PAD_SEC + MUTE_BUFFER_SEC needs a real distribution of
# session-observed gaps, not just a stopwatch guess.
tts_last_stopped_at = [0.0]


# ── Diagnostic audio-clip dump ────────────────────────────────────────────────
#
# When enabled (via configure_debug_audio_dir), EVERY clip that VAD flushes
# is saved as <timestamp>_<seq>.wav with a sidecar <timestamp>_<seq>.txt
# listing the most-recent TTS (conversational context), the clip duration,
# the age-when-sent, Whisper s transcript, and any flag ("stale" / "echo").
# Independent of the Q-A state machine — turning it on at startup is enough.
#
# The point is to listen back when Pella doesn t catch an utterance and
# decide whether the audio actually contained recognisable speech or was
# genuinely empty / too quiet / too noisy. Disk usage grows with use; clean
# the directory periodically.
_debug_audio_dir = [None]
_debug_audio_counter = [0]


def configure_debug_audio_dir(path):
    """Enable (or disable) diagnostic clip dumping.

    Pass a directory path to enable; pass None or "" to disable. The
    directory is created if missing. Once enabled, every VAD-flushed clip
    is saved — no further opt-in required.
    """
    if path:
        os.makedirs(path, exist_ok=True)
        _debug_audio_dir[0] = path
        print(f"Debug audio: enabled, saving every clip to {path}",
              flush=True)
    else:
        _debug_audio_dir[0] = None


def _debug_save_wav(samples, sample_rate, category="speech",
                    declicked_float=None, filtered_float=None):
    """Save a WAV under <debug_dir>/<category>/<stamp>_<seq>.wav, plus
    optional companion files for each stage of the pre-processing pipeline:
      * <stamp>_<seq>.wav            — raw (input to declick)
      * <stamp>_<seq>_declicked.wav  — after declick, pre HP + NR
      * <stamp>_<seq>_filtered.wav   — after full pipeline (what Whisper saw)
    Compare any two in Audacity to isolate the effect of one stage.

    `category` routes by Whisper outcome:
      * "speech" — Whisper produced a transcript (real speech or echo)
      * "noise"  — VAD triggered but Whisper rejected as no_speech

    Returns the raw WAV path (or None on failure).
    """
    if _debug_audio_dir[0] is None:
        return None
    import wave
    from datetime import datetime
    subdir = os.path.join(_debug_audio_dir[0], category)
    try:
        os.makedirs(subdir, exist_ok=True)
    except Exception as e:
        print(f"Debug audio: mkdir {subdir} failed: {e}", flush=True)
        return None
    _debug_audio_counter[0] += 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base  = f"{stamp}_{_debug_audio_counter[0]:04d}"
    path  = os.path.join(subdir, base + ".wav")
    try:
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(samples.tobytes())
    except Exception as e:
        print(f"Debug audio: WAV save failed: {e}", flush=True)
        return None

    def _save_companion(suffix, float_arr):
        if float_arr is None:
            return
        comp_path = os.path.join(subdir, base + suffix)
        try:
            int16 = np.clip(float_arr * 32768.0,
                            -32768, 32767).astype(np.int16)
            with wave.open(comp_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(int16.tobytes())
        except Exception as e:
            print(f"Debug audio: companion WAV save failed "
                  f"({suffix}): {e}", flush=True)

    _save_companion("_declicked.wav", declicked_float)
    _save_companion("_filtered.wav",  filtered_float)
    return path


def _debug_save_txt(wav_path, clip_sec, speech_start_t,
                    transcript=None, reason=None):
    """Sidecar metadata next to a dumped WAV.

    Context (the most recent TTS Pella said) is pulled from _recent_tts —
    same source the echo filter uses — so the saved metadata always
    reflects "what conversational state was Pella in" when this clip
    arrived, without needing the caller to thread the context through.
    """
    if not wav_path:
        return
    txt_path = wav_path[:-4] + ".txt"
    now = time.monotonic()
    # Only treat a TTS as "recent context" if it was playing within the
    # last 10 s — otherwise this clip likely has nothing to do with it.
    recent_context = ""
    if _recent_tts["text"] and now - _recent_tts["ends_at"] < 10.0:
        recent_context = _recent_tts["text"]
    try:
        with open(txt_path, "w") as f:
            f.write(f"recent_tts: {recent_context}\n")
            f.write(f"clip_sec: {clip_sec:.2f}\n")
            f.write(f"age_when_sent_sec: "
                    f"{now - speech_start_t:.2f}\n")
            f.write(f"transcript: "
                    f"{transcript if transcript else '<no speech detected>'}\n")
            if reason:
                f.write(f"flagged: {reason}\n")
    except Exception as e:
        print(f"Debug audio: txt save failed: {e}", flush=True)


# ── TTS-echo filter ───────────────────────────────────────────────────────────
#
# Even with the mic-mute mechanism, fragments of Pella s own speech can
# occasionally leak into the capture (long playback tail, network jitter,
# room reverb past the mute deadline). When that happens Whisper transcribes
# something like "Pella" or "What is your name" which then gets routed to
# the task as if the user said it.
#
# tts.py calls note_tts_play(text, duration) when each phrase starts playing.
# After Whisper transcribes, _is_echo() checks whether the transcript looks
# like a fragment of that recent TTS — and if so, the transcript is dropped
# before reaching the queue.
_recent_tts = {
    "text":    "",
    "ends_at": 0.0,
}


def note_tts_play(text, duration_sec):
    """Record what TTS just started playing so the ASR can filter echoes."""
    _recent_tts["text"]    = text or ""
    _recent_tts["ends_at"] = time.monotonic() + max(0.0, duration_sec)


def _is_echo(transcript, speech_start_t):
    """Conservative: only flag as echo when speech started during/just after
    the TTS audio AND every word in the transcript is a word from the TTS.
    Won t filter unrelated speech that incidentally shares a word."""
    if not _recent_tts["text"]:
        return False
    # If the user started speaking after TTS had finished plus a small grace,
    # leakage can t be the source — must be genuine speech.
    if speech_start_t > _recent_tts["ends_at"] + 1.0:
        return False
    tts_lower = _recent_tts["text"].lower()
    test = re.sub(r"[^a-z ]", " ", transcript.lower())
    test = re.sub(r"\s+", " ", test).strip()
    if not test:
        return False
    tts_words  = set(re.findall(r"[a-z]+", tts_lower))
    test_words = test.split()
    if not test_words:
        return False
    return all(w in tts_words for w in test_words)


# ── Whisper model ─────────────────────────────────────────────────────────────

def _load_whisper():
    """Load Whisper: faster-whisper CUDA → faster-whisper CPU int8 → openai-whisper."""
    try:
        from faster_whisper import WhisperModel
        for compute_type in ("float16", "int8_float16"):
            try:
                print(f"Loading Whisper small.en (faster-whisper CUDA {compute_type})...",
                      flush=True)
                model = WhisperModel("small.en", device="cuda", compute_type=compute_type)
                return model, "faster_whisper"
            except Exception as e:
                print(f"  faster-whisper CUDA {compute_type} failed: {e}", flush=True)
        # CUDA unavailable — CPU int8 has better beam search than openai-whisper
        print("Loading Whisper small.en (faster-whisper CPU int8)...", flush=True)
        model = WhisperModel("small.en", device="cpu", compute_type="int8")
        return model, "faster_whisper"
    except ImportError:
        pass

    import whisper
    print("Loading Whisper small.en (openai-whisper PyTorch)...", flush=True)
    model = whisper.load_model("small.en")
    return model, "openai_whisper"


_whisper, _WHISPER_BACKEND = _load_whisper()

# Warm up Whisper now — first CUDA inference triggers kernel compilation (~10–15 s).
try:
    _dummy = np.zeros(16000, dtype=np.float32)
    if _WHISPER_BACKEND == "faster_whisper":
        list(_whisper.transcribe(_dummy, language="en")[0])
    else:
        import torch as _torch
        _whisper.transcribe(_dummy, language="en", fp16=_torch.cuda.is_available())
    del _dummy
except Exception:
    pass
print(f"Whisper ready ({_WHISPER_BACKEND})", flush=True)

# Warm up noisereduce — numba JIT compile on first call takes ~17 s otherwise.
if _HAS_NR:
    try:
        _dummy = np.zeros(16000, dtype=np.float32)
        _nr.reduce_noise(y=_dummy, sr=16000, stationary=True, prop_decrease=0.75)
        del _dummy
    except Exception:
        pass
    print("noisereduce ready (JIT warmed up)", flush=True)


# ── Noise pre-processing: high-pass + saved profile ──────────────────────────
#
# Diagnostic on a real failing capture showed 96 % of background-noise
# energy below 200 Hz with a dominant 56 Hz peak (mains harmonic + fan
# rumble coupling into the mic body on this dock). Two layered filters
# surgically attack that profile:
#
#   1. 8th-order Butterworth high-pass at HIGHPASS_CUTOFF_HZ. Tested
#      against the failing clip, this drops <200 Hz energy from 96 % ->
#      4 % and lifts the speech band (200-3000 Hz) from 4 % -> 92 %,
#      adding ~11 dB SNR. Voice formants live mostly in 500-1000 Hz so
#      they re fully preserved. The high order is necessary because a
#      4th-order at 250 Hz still left 40 % of energy in the noise band —
#      the noise extends past 56 Hz into low-mid frequencies.
#   2. noisereduce with a known-silent profile loaded from
#      data/noise_profile.wav. Removes residual mid-frequency motor
#      harmonics the high-pass misses. With a fixed reference profile,
#      the subtraction stays surgical even when the clip is almost
#      entirely voice — the old per-clip estimation otherwise included
#      voice harmonics and subtracted them from themselves.
HIGHPASS_CUTOFF_HZ   = 250
HIGHPASS_ORDER       = 6     # Lowered from 8 -> 6 after the first round of
                             # real captures sounded over-processed in the
                             # 200-300 Hz range. At 6th order the corner is
                             # almost as sharp (36 dB/oct vs 48 dB/oct) so
                             # noise rejection drops only a little (<200Hz
                             # band: 2 % -> 8 % of total energy) while the
                             # voice tail above the cutoff is attenuated
                             # less steeply — voice sounds more natural.
_HP_SOS = (butter(HIGHPASS_ORDER, HIGHPASS_CUTOFF_HZ, btype="highpass",
                  fs=ASR_SAMPLE_RATE, output="sos")
           if _HAS_SCIPY else None)
if _HP_SOS is not None:
    print(f"High-pass filter ready "
          f"(order {HIGHPASS_ORDER}, {HIGHPASS_CUTOFF_HZ} Hz cutoff)",
          flush=True)


# ── De-clicker ────────────────────────────────────────────────────────────────
#
# Impulsive transients ("explosive clicks") show up in the dock capture from
# USB packet hiccups, LiDAR motor pulses leaking into the mic body, or
# mechanical taps on the surface. They are very brief (~1-2 ms), broadband,
# and produce sample-to-sample jumps far larger than anything in speech.
# Whisper is sensitive to them: a click in the middle of "Joy" easily
# becomes a different acoustic token in the encoder.
#
# Detection is two-stage to avoid clipping voiced consonants:
#   1. Candidate spike: a sample-to-sample |diff| above
#      DECLICK_THRESHOLD_FACTOR * clip_rms (4x by default).
#   2. Narrow-impulse confirmation: within DECLICK_MAX_WIDTH samples
#      after the candidate, there must be an opposite-sign jump of
#      similar magnitude. That s the signature of a true impulse
#      returning to baseline. Speech onsets (/p/, /t/, /k/, fricatives)
#      attack hard then stay elevated for ~10-50 ms as formants
#      resonate — they never produce the opposing-sign rebound and
#      so are left alone.
# Repair: replace ±DECLICK_PAD_SAMPLES around the impulse with a linear
#   interpolation between the boundary samples just outside the patched
#   region.
DECLICK_THRESHOLD_FACTOR = 4.5     # 4.0 -> 4.5: real captures showed
                                   # zero false positives already; bumping
                                   # slightly higher reserves de-click for
                                   # only the most unambiguous impulses.

# Loudness normalization targets applied AFTER high-pass + noisereduce.
# Pick values close to Whisper s training distribution so its VAD filter
# and no_speech_prob calibration stay sane. Real captures landed around
# -41 to -45 dBFS RMS pre-normalize; these targets boost ~16-22 dB.
NORMALIZE_TARGET_RMS = 0.05        # ~-26 dBFS — typical speech level
NORMALIZE_PEAK_CEIL  = 0.70        # ~-3 dBFS — hard ceiling so we never
                                   # clip even when target_rms would
                                   # require a much higher gain.
DECLICK_MAX_WIDTH        = 16     # 1 ms @ 16 kHz: opposite-sign jump
                                  # must arrive within this window for the
                                  # candidate to count as an impulse.
DECLICK_PAD_SAMPLES      = 8      # ±0.5 ms padding around each patched
                                  # impulse; keeps interpolation away from
                                  # the click s ringing tail.


def _declick(audio: np.ndarray) -> np.ndarray:
    """Detect impulsive clicks and replace them with linear interpolation.

    Operates on a float32 array sampled at ASR_SAMPLE_RATE. Returns a
    new array (input unmodified). Safe to call even when no clicks are
    present — the worst case is a few CPU microseconds and no-op output.

    Only patches *narrow* impulses (≤ DECLICK_MAX_WIDTH samples wide,
    where width = distance to the matching opposite-sign jump).
    Wide-but-loud transients like plosives or hand claps are preserved.
    """
    if audio.size < 3:
        return audio
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    if rms <= 0:
        return audio
    threshold = DECLICK_THRESHOLD_FACTOR * rms
    diff = np.diff(audio)
    abs_diff = np.abs(diff)
    candidates = np.flatnonzero(abs_diff > threshold)
    if candidates.size == 0:
        return audio

    n     = audio.size
    pad   = DECLICK_PAD_SAMPLES
    max_w = DECLICK_MAX_WIDTH
    diff_len = diff.size

    # For each candidate, look for an opposite-sign jump of similar
    # magnitude within max_w samples ahead. If found, that confirms a
    # narrow impulse; otherwise leave the candidate alone.
    intervals = []
    last_patched = -1
    for i in candidates:
        if i <= last_patched:
            continue        # already inside a previously-patched window
        sign_i = diff[i]
        # Search window: diff[i+1 .. i+max_w]
        end = min(i + max_w + 1, diff_len)
        if sign_i > 0:
            opp = np.flatnonzero(diff[i + 1:end] < -threshold)
        else:
            opp = np.flatnonzero(diff[i + 1:end] > threshold)
        if opp.size == 0:
            continue        # no rebound — not an impulse (likely speech)
        j = int(i + 1 + opp[0])     # index of the opposing jump in diff
        # Patch span: from spike start to one sample past the rebound,
        # padded on both sides so the interpolation reaches stable audio.
        lo = max(0, int(i) - pad)
        hi = min(n - 1, j + 1 + pad)
        if intervals and lo <= intervals[-1][1] + 1:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], hi))
        else:
            intervals.append((lo, hi))
        last_patched = hi

    if not intervals:
        return audio

    out = audio.copy()
    for lo, hi in intervals:
        if lo <= 0 or hi >= n - 1:
            edge = out[hi + 1] if hi < n - 1 \
                   else (out[lo - 1] if lo > 0 else 0.0)
            out[lo:hi + 1] = edge
        else:
            out[lo:hi + 1] = np.linspace(out[lo - 1], out[hi + 1],
                                         hi - lo + 1, dtype=out.dtype)
    return out

_NOISE_PROFILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "noise_profile.wav",
)
NOISE_PROFILE_CAPTURE_SEC = 3.0
_noise_profile = None     # float32 array @ ASR_SAMPLE_RATE, or None


def _save_noise_profile(samples_int16: np.ndarray) -> None:
    """Persist the captured profile to data/noise_profile.wav so subsequent
    runs skip re-capturing. Delete the file to force a fresh capture.
    """
    try:
        os.makedirs(os.path.dirname(_NOISE_PROFILE_PATH), exist_ok=True)
        with wave.open(_NOISE_PROFILE_PATH, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(ASR_SAMPLE_RATE)
            w.writeframes(samples_int16.astype(np.int16).tobytes())
    except Exception as e:
        print(f"Noise profile: save failed: {e}", flush=True)


def _load_noise_profile() -> None:
    """Load the saved profile into _noise_profile (None on failure)."""
    global _noise_profile
    try:
        with wave.open(_NOISE_PROFILE_PATH, "rb") as w:
            raw = w.readframes(w.getnframes())
        _noise_profile = (np.frombuffer(raw, dtype=np.int16)
                          .astype(np.float32) / 32768.0)
        print(f"Noise profile: loaded "
              f"({len(_noise_profile)/ASR_SAMPLE_RATE:.1f}s) "
              f"from {_NOISE_PROFILE_PATH}", flush=True)
    except Exception:
        _noise_profile = None


def ensure_noise_profile(read_chunk_fn, sample_rate: int) -> None:
    """Load the noise profile from disk, or capture a fresh one.

    Called once from run_usb_mic right after the stream opens, before the
    main capture loop. `read_chunk_fn()` should return a fresh int16
    numpy array of audio samples — we just stream a few seconds of it
    while assuming the user is silent.

    Skips capture if the file already exists (fast restart). Delete
    data/noise_profile.wav to force a re-capture, e.g. after moving to
    a new room.
    """
    global _noise_profile
    if os.path.exists(_NOISE_PROFILE_PATH):
        _load_noise_profile()
        return

    print(f"Noise profile: capturing {NOISE_PROFILE_CAPTURE_SEC:.1f}s of "
          f"ambient (please stay silent)...", flush=True)
    target_samples = int(NOISE_PROFILE_CAPTURE_SEC * sample_rate)
    buf = []
    have = 0
    while have < target_samples:
        chunk = read_chunk_fn()
        if chunk is None or len(chunk) == 0:
            continue
        buf.append(chunk)
        have += len(chunk)
    captured = np.concatenate(buf)[:target_samples].astype(np.int16)
    _noise_profile = captured.astype(np.float32) / 32768.0
    _save_noise_profile(captured)
    rms = float(np.sqrt((captured.astype(np.float32) ** 2).mean()))
    print(f"Noise profile: captured ({NOISE_PROFILE_CAPTURE_SEC:.1f}s, "
          f"rms={int(rms)}) -> {_NOISE_PROFILE_PATH}", flush=True)


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe(samples: np.ndarray,
               src_sr: int = ROBOT_SAMPLE_RATE,
               nr_prop: float = 0.75,
               *,
               out_declicked: Optional[list] = None,
               out_filtered:  Optional[list] = None) -> str:
    """Convert mono int16 audio to text via Whisper.

    Resamples to 16 kHz if needed, runs the three-stage preprocessing
    pipeline (declick -> high-pass -> noisereduce), then runs Whisper.
    Returns empty string when no speech is detected.

    out_declicked / out_filtered: optional lists for A/B diagnostics.
    When provided, intermediate float32 audio is appended to each:
      * out_declicked — after declick only (pre HP + NR)
      * out_filtered  — after the full pipeline (what Whisper saw)
    The caller can persist these as companion WAVs to compare the
    individual stages in Audacity.
    """
    try:
        audio_f = samples.astype(np.float32) / 32768.0
        audio_16k = audio_f if src_sr == ASR_SAMPLE_RATE else \
                    soxr.resample(audio_f, src_sr, ASR_SAMPLE_RATE, quality="HQ")

        # 1. De-click: replace impulsive transients with linear
        #    interpolation. Runs first so the downstream filters see
        #    a clean signal — clicks otherwise smear through the
        #    high-pass / noisereduce stages and end up as residual
        #    ringing that Whisper still hears.
        try:
            audio_16k = _declick(audio_16k)
        except Exception:
            pass
        if out_declicked is not None:
            try:
                out_declicked.append(
                    np.asarray(audio_16k, dtype=np.float32).copy())
            except Exception:
                pass

        # 2. High-pass: kill the 0-150 Hz band where >96 % of the dock s
        #    background noise lives (mains harmonics + fan rumble). Voice
        #    energy starts ~200 Hz, so the speech band is untouched.
        if _HP_SOS is not None:
            try:
                audio_16k = sosfilt(_HP_SOS, audio_16k).astype(np.float32)
            except Exception:
                pass

        # 3. Spectral noise reduction. Use the pre-captured profile if we
        #    have one — surgical subtraction of the actual ambient.
        #    Otherwise fall back to per-clip estimation (noisier, but
        #    still better than nothing).
        if _HAS_NR:
            try:
                kwargs = dict(sr=ASR_SAMPLE_RATE, stationary=True,
                              prop_decrease=nr_prop)
                if _noise_profile is not None:
                    kwargs["y_noise"] = _noise_profile
                audio_16k = _nr.reduce_noise(y=audio_16k, **kwargs)
            except Exception:
                pass

        # 4. Loudness normalization. The high-pass + noisereduce stages
        #    typically leave voice content very quiet (real-world clips
        #    measure RMS around -41 to -45 dBFS, ~20 dB below typical
        #    speech). Whisper s log-mel features are technically
        #    log-scale-invariant, but its built-in VAD and no_speech_prob
        #    behave noticeably better with input at training-corpus levels
        #    (~-23 dBFS RMS). Boost to a target RMS, capped by a safe peak
        #    so we never clip into distortion.
        try:
            peak = float(np.abs(audio_16k).max())
            if peak > 1e-6:
                rms = float(np.sqrt(np.mean(audio_16k ** 2)))
                if rms > 1e-6:
                    gain_rms  = NORMALIZE_TARGET_RMS  / rms
                    gain_peak = NORMALIZE_PEAK_CEIL   / peak
                    gain = min(gain_rms, gain_peak)
                    audio_16k = (audio_16k * gain).astype(np.float32)
        except Exception:
            pass

        # Hand the post-filter audio back to the caller so it can save a
        # "_filtered" companion WAV next to the raw dump. Copy to avoid
        # later in-place mutations by Whisper backends.
        if out_filtered is not None:
            try:
                out_filtered.append(np.asarray(audio_16k, dtype=np.float32).copy())
            except Exception:
                pass

        if _WHISPER_BACKEND == "faster_whisper":
            segments, _ = _whisper.transcribe(
                audio_16k, language="en", beam_size=5,
                vad_filter=True, condition_on_previous_text=False,
                no_speech_threshold=NO_SPEECH_THRESHOLD,
                # Whisper s built-in hallucination guards. Tightened from
                # defaults (2.4 / -1.0) because real captures showed the
                # language model inventing plausible English ("Peace be
                # upon you.", "William, Pella.") for clips with only a
                # fraction of a second of actual voice. Stricter values
                # cause Whisper to reject low-confidence decodes upfront
                # instead of committing to them.
                compression_ratio_threshold=2.0,   # repetitive / templated
                                                   # decodes get rejected
                log_prob_threshold=-0.7,           # low-confidence decodes
                                                   # get rejected
                # Language-model hint that biases Whisper toward the name-
                # introduction patterns we actually listen for, away from
                # frequent-English hallucinations ("It's Alison" / "Hey
                # Mr. Ellison" instead of "My name is Alison"). Common
                # short names anchor the model toward proper-noun
                # interpretation of similar-sounding tokens.
                initial_prompt=("Pella, Bella. My name is Alison. "
                                "My name is William. Call me Sam. "
                                "I am Joy."),
            )
            parts = [seg.text.strip() for seg in segments
                     if seg.no_speech_prob < NO_SPEECH_THRESHOLD]
        else:
            import torch as _torch
            result = _whisper.transcribe(
                audio_16k, language="en",
                fp16=_torch.cuda.is_available(),
                condition_on_previous_text=False,
            )
            parts = [s["text"].strip() for s in result.get("segments", [])
                     if s.get("no_speech_prob", 0) < NO_SPEECH_THRESHOLD]

        return " ".join(parts).strip()

    except Exception as e:
        print(f"ASR error: {e}", flush=True)
        return ""


# ── Full-pipeline warmup ──────────────────────────────────────────────────────
# Initial zero-audio warmup (above) compiles kernels but Whisper's VAD skips
# silent input, so the inference path stays cold. Run the full transcribe()
# twice with a synthetic speech-like signal to prime resample → NR → encoder →
# decoder paths and CPU caches.
try:
    _t      = np.arange(0, 1.5, 1.0 / ASR_SAMPLE_RATE, dtype=np.float32)
    # Three formant-band tones with a 4 Hz amplitude envelope (~speech rhythm).
    _signal = (np.sin(2 * np.pi *  500 * _t) * 0.3 +
               np.sin(2 * np.pi * 1500 * _t) * 0.2 +
               np.sin(2 * np.pi * 2500 * _t) * 0.1)
    _signal *= np.clip(np.sin(2 * np.pi * 4 * _t), 0, None) ** 2
    _warm    = (_signal * 8000).astype(np.int16)
    for _ in range(2):
        transcribe(_warm, src_sr=ASR_SAMPLE_RATE)
    del _t, _signal, _warm
    print("STT pipeline warmed up (full path)", flush=True)
except Exception as e:
    print(f"STT pipeline warmup error: {e}", flush=True)


# ── USB mic capture thread ────────────────────────────────────────────────────

def run_usb_mic(transcript_queue: Queue, stop_event: threading.Event):
    """Capture audio from a USB microphone and transcribe speech via VAD + Whisper.

    Runs in a dedicated thread. Transcription is dispatched to a background
    executor so the capture loop never stalls during Whisper inference.
    """
    try:
        import pyaudio
    except ImportError:
        print("pyaudio not installed — USB mic disabled. Run: pip3 install pyaudio",
              flush=True)
        return

    FRAME_SAMPLES = int(VAD_FRAME_MS / 1000 * USB_MIC_SAMPLE_RATE)  # 320 @ 16 kHz
    pa = pyaudio.PyAudio()

    # Scan and select input device by name.
    device_index = None
    print("USB mic: scanning audio devices...", flush=True)
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  [{i}] {info['name']}  ch={info['maxInputChannels']}"
                  f"  rate={int(info['defaultSampleRate'])}", flush=True)
            if USB_MIC_DEVICE_NAME and USB_MIC_DEVICE_NAME.lower() in info["name"].lower():
                device_index = i

    if device_index is not None:
        name = pa.get_device_info_by_index(device_index)["name"]
        print(f"USB mic: using device [{device_index}] '{name}'", flush=True)
    else:
        print(f"USB mic: '{USB_MIC_DEVICE_NAME}' not found — using system default",
              flush=True)

    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=USB_MIC_CHANNELS,
            rate=USB_MIC_SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=FRAME_SAMPLES * 16,  # large buffer survives Whisper latency
        )
    except Exception as e:
        print(f"USB mic open failed: {e}", flush=True)
        pa.terminate()
        return

    print("USB mic streaming started", flush=True)

    # Capture (or load) the ambient-noise profile while the room is still
    # quiet — before the TTS warmup completes and the user starts
    # interacting. Subsequent transcribe() calls will use this profile
    # for surgical spectral subtraction. Delete data/noise_profile.wav
    # to force a re-capture.
    def _read_chunk():
        raw = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
        return np.frombuffer(raw, dtype=np.int16)
    try:
        ensure_noise_profile(_read_chunk, USB_MIC_SAMPLE_RATE)
    except Exception as e:
        print(f"Noise profile: setup failed: {e}", flush=True)

    # Single-worker executor: Whisper runs off the capture thread, results arrive
    # in order, and the capture loop never blocks waiting for inference.
    asr_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    vad_engine = _webrtcvad.Vad(VAD_AGGRESSIVENESS)
    vad = {
        "buf":           deque(),
        "pre_roll":      deque(maxlen=USB_PRE_ROLL_FRAMES),
        "in_speech":     False,
        "speech_frames": 0,
        "silence_frames": 0,
        "speech_samples": 0,
        "frame_count":   0,
        "max_rms":       0.0,
        "noise_floor":   USB_NOISE_FLOOR_INIT,
        "speech_start_t": 0.0,   # monotonic time the user actually started
                                 # speaking. Stamped on each transcript so
                                 # downstream consumers can decide if the
                                 # speech fell inside a listening window —
                                 # robust to Whisper transcription latency.
    }
    max_samples = int(MAX_SPEECH_SEC * USB_MIC_SAMPLE_RATE)
    min_samples = int(MIN_SPEECH_SEC * USB_MIC_SAMPLE_RATE)

    def _flush():
        samples         = np.array(list(vad["buf"]), dtype=np.int16)
        speech_start_t  = vad["speech_start_t"]
        vad["buf"].clear()
        vad["in_speech"]       = False
        vad["speech_frames"]   = 0
        vad["silence_frames"]  = 0
        vad["speech_samples"]  = 0
        vad["speech_start_t"]  = 0.0
        if len(samples) < min_samples:
            return
        clip_sec = len(samples) / USB_MIC_SAMPLE_RATE
        # speech_end_t = end of audible speech ≈ start + buffered duration.
        # The trailing silence trailer (USB_VAD_SILENCE_FRAMES) is part of
        # `samples` so this is a slight over-estimate (by ~USB_SILENCE_FRAMES
        # * 20ms), but that's fine for the downstream stitch-gap test which
        # uses ~3 s windows.
        speech_end_t = speech_start_t + clip_sec
        print(f"ASR: sending {clip_sec:.1f}s of audio "
              f"(speech started {time.monotonic() - speech_start_t:.1f}s ago)",
              flush=True)

        def _run():
            # Drop stale clips: with max_workers=1 the executor serialises,
            # and on a CPU-only Jetson a 6 s burst of speech can sit ~10 s
            # in the queue before Whisper gets to it. By the time a stale
            # clip would emerge no consumer (enrollment, chat, …) accepts
            # it anyway — skip the transcribe to free the next slot sooner.
            age = time.monotonic() - speech_start_t
            if age > MAX_CLIP_AGE_SEC:
                print(f"ASR: skipping stale {clip_sec:.1f}s clip "
                      f"({age:.1f}s old, > {MAX_CLIP_AGE_SEC:.0f}s cutoff)",
                      flush=True)
                return
            # Capture intermediate stages for A/B review. transcribe()
            # appends float32 arrays; we convert to int16 inside
            # _debug_save_wav. Two stages: post-declick (pre HP+NR) and
            # post-full-pipeline (what Whisper actually saw).
            declicked_holder = []
            filtered_holder  = []
            text = transcribe(samples, src_sr=USB_MIC_SAMPLE_RATE,
                              nr_prop=USB_NR_PROP_DECREASE,
                              out_declicked=declicked_holder,
                              out_filtered=filtered_holder)
            declicked_audio = declicked_holder[0] if declicked_holder else None
            filtered_audio  = filtered_holder[0]  if filtered_holder  else None
            echo = bool(text) and _is_echo(text, speech_start_t)
            if text and not echo:
                print(f"Heard: {text}", flush=True)
                try:
                    transcript_queue.put_nowait(
                        (text, speech_start_t, speech_end_t))
                except Exception:
                    pass
            elif echo:
                print(f"ASR: discarding likely TTS echo of "
                      f"{_recent_tts['text']!r}: {text!r}", flush=True)
            else:
                print("ASR: no speech detected in clip", flush=True)

            # Save EVERY clip — but route by Whisper outcome so the user
            # can scan one subdir for utterances and the other for
            # background-noise samples without one drowning the other.
            #   speech/  — Whisper transcribed text (real speech or echo)
            #   noise/   — VAD triggered but Whisper rejected as silence
            # Each raw <stamp>_<seq>.wav gets two companions:
            #   _declicked.wav — after declick only
            #   _filtered.wav  — after declick + HP + noisereduce
            # so we can A/B compare each preprocessing stage in Audacity.
            category = "speech" if text else "noise"
            wav_path = _debug_save_wav(
                samples, USB_MIC_SAMPLE_RATE, category=category,
                declicked_float=declicked_audio,
                filtered_float=filtered_audio)
            _debug_save_txt(
                wav_path, clip_sec, speech_start_t,
                transcript=text or "",
                reason=("echo" if echo else None),
            )

        asr_executor.submit(_run)

    try:
        while not stop_event.is_set():
            try:
                raw = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
            except Exception as e:
                print(f"USB mic read error: {e}", flush=True)
                break

            if time.monotonic() < tts_mute_until[0]:
                vad["mic_was_muted"] = True
                continue

            if vad.get("mic_was_muted"):
                vad["mic_was_muted"] = False
                stopped = tts_last_stopped_at[0]
                if stopped > 0.0:
                    gap_ms = (time.monotonic() - stopped) * 1000.0
                    print(f"USB mic: opened {gap_ms:.0f} ms after TTS end",
                          flush=True)
                    tts_last_stopped_at[0] = 0.0

            chunk = np.frombuffer(raw, dtype=np.int16)
            rms   = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            vad["frame_count"] += 1
            vad["max_rms"] = max(vad["max_rms"], rms)

            if vad["frame_count"] % 200 == 0:
                print(f"USB mic: frame={vad['frame_count']}"
                      f"  peak={vad['max_rms']:.0f}"
                      f"  floor={vad['noise_floor']:.0f}", flush=True)
                vad["max_rms"] = 0.0

            # Reject mechanical spikes (LiDAR motor, physical impact on dock).
            if rms > vad["noise_floor"] * USB_SPIKE_REJECT:
                continue

            if not vad["in_speech"]:
                alpha = 0.05 if vad["frame_count"] < USB_WARMUP_FRAMES \
                        else NOISE_FLOOR_EMA_ALPHA
                vad["noise_floor"] = min(
                    USB_NOISE_FLOOR_CEIL,
                    alpha * rms + (1.0 - alpha) * vad["noise_floor"],
                )

            if vad["frame_count"] < USB_WARMUP_FRAMES:
                continue  # floor still stabilising — suppress all triggers

            above_floor = rms > vad["noise_floor"] * USB_NOISE_FLOOR_FACTOR
            if above_floor:
                try:
                    is_speech = vad_engine.is_speech(chunk.tobytes(), USB_MIC_SAMPLE_RATE)
                except Exception:
                    is_speech = False
            else:
                is_speech = False

            if is_speech:
                vad["speech_frames"]  += 1
                # Only reset silence timer for strong speech — prevents background noise
                # from keeping VAD alive after the user stops speaking (hysteresis).
                if rms > vad["noise_floor"] * USB_SILENCE_HOLD_FACTOR:
                    vad["silence_frames"] = 0
                if vad["in_speech"]:
                    vad["buf"].extend(chunk.tolist())
                    vad["speech_samples"] += len(chunk)
                    if len(vad["buf"]) >= max_samples:
                        _flush()
                elif vad["speech_frames"] >= USB_VAD_SPEECH_FRAMES:
                    vad["in_speech"]      = True
                    # Stamp as soon as VAD commits to "this is speech" —
                    # the timestamp travels with the transcript so the
                    # consumer can match user-speech-time, not arrival-time.
                    vad["speech_start_t"] = time.monotonic()
                    for pre in vad["pre_roll"]:
                        vad["buf"].extend(pre)
                        vad["speech_samples"] += len(pre)
                    vad["pre_roll"].clear()
                    vad["buf"].extend(chunk.tolist())
                    vad["speech_samples"] += len(chunk)
                else:
                    vad["pre_roll"].append(chunk.tolist())
            else:
                vad["speech_frames"]  = 0
                vad["silence_frames"] += 1
                if vad["in_speech"]:
                    vad["buf"].extend(chunk.tolist())
                    if (vad["silence_frames"] >= USB_VAD_SILENCE_FRAMES
                            or len(vad["buf"]) >= max_samples):
                        _flush()
                else:
                    vad["pre_roll"].append(chunk.tolist())
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        asr_executor.shutdown(wait=False)
        print("USB mic stopped", flush=True)
