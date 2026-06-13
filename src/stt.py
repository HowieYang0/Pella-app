#!/usr/bin/env python3
"""Speech-to-text pipeline: Whisper transcription, VAD, and USB mic capture."""

import concurrent.futures
import threading
import time
from collections import deque
from queue import Queue, Empty

import numpy as np
import soxr
import webrtcvad as _webrtcvad

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
TTS_MUTE_SEC          = 4.0    # mute mic this long after TTS plays
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
USB_NR_PROP_DECREASE   = 0.85    # aggressive NR — LiDAR/fan noise is very stationary
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


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe(samples: np.ndarray,
               src_sr: int = ROBOT_SAMPLE_RATE,
               nr_prop: float = 0.75) -> str:
    """Convert mono int16 audio to text via Whisper.

    Resamples to 16 kHz if needed, applies optional noise reduction,
    then runs Whisper. Returns empty string when no speech is detected.
    """
    try:
        audio_f = samples.astype(np.float32) / 32768.0
        audio_16k = audio_f if src_sr == ASR_SAMPLE_RATE else \
                    soxr.resample(audio_f, src_sr, ASR_SAMPLE_RATE, quality="HQ")

        if _HAS_NR:
            try:
                audio_16k = _nr.reduce_noise(
                    y=audio_16k, sr=ASR_SAMPLE_RATE,
                    stationary=True, prop_decrease=nr_prop,
                )
            except Exception:
                pass

        if _WHISPER_BACKEND == "faster_whisper":
            segments, _ = _whisper.transcribe(
                audio_16k, language="en", beam_size=5,
                vad_filter=True, condition_on_previous_text=False,
                no_speech_threshold=0.5,
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
            parts = [seg.text.strip() for seg in segments if seg.no_speech_prob < 0.5]
        else:
            import torch as _torch
            result = _whisper.transcribe(
                audio_16k, language="en",
                fp16=_torch.cuda.is_available(),
                condition_on_previous_text=False,
            )
            parts = [s["text"].strip() for s in result.get("segments", [])
                     if s.get("no_speech_prob", 0) < 0.5]

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
            text = transcribe(samples, src_sr=USB_MIC_SAMPLE_RATE,
                              nr_prop=USB_NR_PROP_DECREASE)
            if text:
                print(f"Heard: {text}", flush=True)
                try:
                    transcript_queue.put_nowait(
                        (text, speech_start_t, speech_end_t))
                except Exception:
                    pass
            else:
                print("ASR: no speech detected in clip", flush=True)

        asr_executor.submit(_run)

    try:
        while not stop_event.is_set():
            try:
                raw = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
            except Exception as e:
                print(f"USB mic read error: {e}", flush=True)
                break

            if time.monotonic() < tts_mute_until[0]:
                continue

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
