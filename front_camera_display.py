#!/usr/bin/env python3
"""Pella: Go2 front camera display with face recognition and voice interaction."""

import asyncio
import contextlib
import hashlib
import io
import json
import logging
import os
import sys
import threading
import time
from collections import Counter, deque
from queue import Queue, Empty

import av
av.logging.set_level(av.logging.FATAL)
try:
    import ctypes, ctypes.util
    _libavutil = ctypes.CDLL(ctypes.util.find_library("avutil"))
    _libavutil.av_log_set_level(-8)  # AV_LOG_QUIET — silences swscaler spam
except Exception:
    pass

import cv2
import dotenv
import numpy as np
import pygame
import webrtcvad as _webrtcvad
from pydub import AudioSegment

dotenv.load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
os.environ.setdefault("DISPLAY", ":0")
logging.basicConfig(level=logging.FATAL)

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_APP_DIR, "..", "go2_webrtc_connect"))

from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.webrtc_audiohub import WebRTCAudioHub
from go2_webrtc_driver.constants import RTC_TOPIC

import re

import actions
import stt
import vision
from stt import (
    USE_USB_MIC, TTS_MUTE_SEC, SPEAKER_VOLUME, tts_mute_until,
    ROBOT_SAMPLE_RATE, ASR_CHANNELS, ASR_SAMPLE_RATE,
    VAD_AGGRESSIVENESS, VAD_FRAME_MS, VAD_FRAME_SAMPLES,
    VAD_SPEECH_FRAMES, VAD_SILENCE_FRAMES, VAD_PRE_ROLL_FRAMES,
    MIN_SPEECH_SEC, MAX_SPEECH_SEC,
    NOISE_FLOOR_EMA_ALPHA, NOISE_FLOOR_INIT, NOISE_FLOOR_FACTOR, VAD_WARMUP_FRAMES,
)
from vision import (
    FACE_DETECT_EVERY, ZOOM_DURATION, GREET_COOLDOWN,
    MOTION_COOLDOWN, SEEK_TIMEOUT,
    INTRODUCE_COOLDOWN, ENROLL_TIMEOUT, FACE_IDS_DIR, SHARPNESS_THRESHOLD,
    detect_faces, detect_motion, detect_person, sharpness,
    annotate, zoom_crop, is_face_at_edge,
    load_recognizer, recognition_worker,
)

# ── Display constants ─────────────────────────────────────────────────────────
RECONNECT_DELAY        = 5.0
FRAME_MS               = 20
TRANSCRIPT_DISPLAY_SEC = 8.0
TRACKING = "tracking"
ZOOMED   = "zoomed"

# ── Interaction state machine ────────────────────────────────────────────────
# The interaction state coordinates a single recognition "event" — from the
# moment a person is spotted, through pose recovery, recognition, and the final
# greeting / introduction. Previously each branch fired independently per
# frame, which let recognition flicker between hit/miss trigger a greet
# immediately followed by an introduce, and let the greet branch run while the
# robot was still seated. The state machine forces these to happen as one
# coherent sequence.
INTERACTION_IDLE         = "idle"          # listening for motion / face
INTERACTION_SEEKING      = "seeking"       # adjusting pose to find a face
INTERACTION_RECOGNIZING  = "recognizing"   # complete face visible; voting on identity
INTERACTION_GREETING     = "greeting"      # known person — recovery + say + wiggle
INTERACTION_INTRODUCING  = "introducing"   # unknown — ask name; enrollment runs
INTERACTION_COOLDOWN     = "cooldown"      # grace period before re-engaging

# Recognition is debounced across multiple detection cycles so a single
# misrecognition doesn't drive a wrong greeting/introduction. With
# FACE_DETECT_EVERY=5 at ~30 fps, one observation arrives every ~170 ms, so
# 3 observations yields a decision in ~0.5 s.
RECOG_VOTES_REQUIRED    = 3
RECOG_AGREE_MIN         = 2
RECOG_TIMEOUT_SEC       = 6.0
# After a recovery action (look_level / stand_up) is queued, give the body
# this long to stop moving before we start sampling faces for the verdict.
# Frames captured mid-motion are too blurry for enrollment and produce
# spurious "unknown" votes that contaminate the verdict.
# look_level is a quick body-tilt back to level (~1 s covers it). stand_up
# from sit is a full rise — leg unfold + balance — and runs noticeably
# longer, so its stabilization needs to wait out the whole motion before
# any face is sampled.
RECOG_STABILIZE_LOOK_SEC = 0.5
RECOG_STABILIZE_SIT_SEC  = 1.5
# Face position jitters during pose recovery, so individual detections can miss
# even when the person is visible. Allow brief gaps before declaring face lost.
FACE_LOST_TIMEOUT_SEC   = 3.0
GREETING_DURATION_SEC   = 6.0
INTERACTION_COOLDOWN_SEC = 5.0


def _tally_recognition(obs):
    """Return ``(verdict, kind)`` from a sequence of recognition observations.

    Each observation is a name string (known) or None (unknown).
    Returns:
      ``(name, "known")``        — name has >= RECOG_AGREE_MIN votes
      ``(None, "unknown")``      — None has >= RECOG_AGREE_MIN votes
      ``(None, "ambiguous")``    — no majority
    """
    if not obs:
        return None, "ambiguous"
    val, count = Counter(obs).most_common(1)[0]
    if count < RECOG_AGREE_MIN:
        return None, "ambiguous"
    return val, ("known" if val else "unknown")


# ── TTS helpers ───────────────────────────────────────────────────────────────

def _generate_wav(text: str, text_hash: str) -> str:
    """Convert text to a 44100 Hz WAV via gTTS. Returns the file path."""
    from gtts import gTTS
    mp3_path = f"/tmp/pella_{text_hash}.mp3"
    out_path = f"/tmp/pella_{text_hash}.wav"
    gTTS(text=text, lang="en").save(mp3_path)
    sound = AudioSegment.from_mp3(mp3_path).set_frame_rate(44100)
    sound.export(out_path, format="wav")
    os.unlink(mp3_path)
    return out_path


# ── Display helpers ───────────────────────────────────────────────────────────

def _to_surface(bgr: np.ndarray, size: tuple) -> pygame.Surface:
    rgb  = bgr[:, :, ::-1]
    surf = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
    return pygame.transform.scale(surf, size)


def _make_offline_surface(w: int, h: int) -> pygame.Surface:
    surf = pygame.Surface((w, h))
    surf.fill((0, 0, 0))
    font = pygame.font.SysFont("sans", 36)
    text = font.render("Waiting for Pella...", True, (180, 180, 180))
    surf.blit(text, (w // 2 - text.get_width() // 2, h // 2))
    return surf


def _draw_transcript(screen, font, text: str, w: int, h: int):
    """Render transcript text centred at the bottom of the screen."""
    padding = 12
    surf  = font.render(text, True, (255, 255, 0))
    bg_w  = surf.get_width()  + padding * 2
    bg_h  = surf.get_height() + padding * 2
    bg    = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 160))
    x = (w - bg_w) // 2
    y = h - bg_h - 30
    screen.blit(bg, (x, y))
    screen.blit(surf, (x + padding, y + padding))


# ── Name parsing ─────────────────────────────────────────────────────────────

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
    # Keep only the first sentence — drops everything after . ! ?
    t = re.split(r"[.!?]", t, maxsplit=1)[0].strip()
    # If a name-intro phrase is present anywhere, take what follows it
    m = re.search(r"(?:i am|i'm|my name is)\s+(.+)", t, re.IGNORECASE)
    if m:
        t = m.group(1).strip()
    # Stop at first comma, ' and ', or courtesy phrase ('nice/glad/pleased to meet')
    courtesy = (
        r",|\s+and\b|"
        r"\s+(?:nice|glad|pleased|happy)\s+(?:to\s+)?(?:meet|meeting)\b"
    )
    t = re.split(courtesy, t, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    # Alphabetic tokens only
    words = re.findall(r"[A-Za-z]+", t)
    if not words:
        return ""
    # Reject if any token is a common non-name word — almost certainly not a name.
    if any(w.lower() in _NON_NAME_WORDS for w in words):
        return ""
    return " ".join(words)


_SEEK_ACTIONS = ("look_up", "sit_look_up")


def _drain_seek_actions(q):
    """Remove any pending look_up/sit_look_up from the action queue, preserving order
    of the remaining items. Called when recognition succeeds so stale seek actions
    queued by a prior seek-timeout don't run after the face has already been found.
    """
    keep = []
    try:
        while True:
            item = q.get_nowait()
            if item not in _SEEK_ACTIONS:
                keep.append(item)
    except Empty:
        pass
    for item in keep:
        try:
            q.put_nowait(item)
        except Exception:
            pass


def _queue_contains(q, name):
    """Best-effort check whether `name` is still pending in the action queue.
    Racy with the consumer thread, but used only for ordering heuristics where
    a missed/false-positive will self-correct on the next display iteration.
    """
    try:
        return name in list(q.queue)
    except Exception:
        return False


# ── WebRTC thread ─────────────────────────────────────────────────────────────

_ACTION_MAP = {
    "look_up":     actions.look_up,
    "look_level":  actions.look_level,
    "sit_look_up": actions.sit_look_up,
    "stand_up":    actions.stand_up,
    "wiggle":      actions.wiggle,
    "hello":       actions.hello,
    "dance":       actions.dance,
}


def _run_webrtc(robot_ip: str, frame_queue: Queue, say_queue: Queue,
                transcript_queue: Queue, action_queue: Queue,
                stop_event: threading.Event, enable_audio: bool = True):
    """Connect to the robot, stream video, play TTS, and (optionally) do STT.

    enable_audio=False when the USB mic handles STT instead.
    The WebRTC connection remains active for video and TTS regardless.
    """

    async def recv_video(track):
        while not stop_event.is_set():
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                while frame_queue.qsize() > 2:
                    try:
                        frame_queue.get_nowait()
                    except Empty:
                        break
                frame_queue.put(img)
            except Exception as e:
                print(f"recv_video error: {type(e).__name__}: {e}", flush=True)
                break

    async def _speak(text: str, audiohub, cache: dict):
        """Generate WAV on demand and play it via the robot's AudioHub."""
        try:
            text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            file_name = f"pella_{text_hash}"
            entry = cache.setdefault(text_hash, {"path": None, "uuid": None})

            if entry["path"] is None:
                entry["path"] = await asyncio.get_event_loop().run_in_executor(
                    None, _generate_wav, text, text_hash
                )

            if entry["uuid"] is None:
                resp = await audiohub.get_audio_list()
                audio_list = json.loads(
                    (resp.get("data") or {}).get("data", "{}")
                ).get("audio_list", [])
                for item in audio_list:
                    if item.get("CUSTOM_NAME") == file_name:
                        await audiohub.delete_record(item["UNIQUE_ID"])
                        break
                with contextlib.redirect_stdout(io.StringIO()):
                    await audiohub.upload_audio_file(entry["path"])
                resp = await audiohub.get_audio_list()
                audio_list = json.loads(
                    (resp.get("data") or {}).get("data", "{}")
                ).get("audio_list", [])
                for item in audio_list:
                    if item.get("CUSTOM_NAME") == file_name:
                        entry["uuid"] = item["UNIQUE_ID"]
                        break

            if entry["uuid"]:
                mute_t = time.monotonic() + TTS_MUTE_SEC
                vad_state["mute_until"] = mute_t
                tts_mute_until[0]       = mute_t   # also mutes USB mic path
                preview = (text[:60] + "…") if len(text) > 60 else text
                print(f"TTS: playing {repr(preview)} "
                      f"(mute until +{TTS_MUTE_SEC:.1f}s)", flush=True)
                await audiohub.play_by_uuid(entry["uuid"])
                print(f"TTS: done {repr(preview)}", flush=True)
            else:
                print(f"TTS: NO UUID for {repr(text[:60])} — playback skipped",
                      flush=True)
        except Exception as e:
            print(f"TTS error: {e}", flush=True)

    async def recv_audio(frame):
        """WebRTC robot-mic audio → VAD → Whisper (used when enable_audio=True)."""
        if time.monotonic() < vad_state["mute_until"]:
            return

        raw  = np.frombuffer(frame.to_ndarray(), dtype=np.int16)
        mono = raw.reshape(-1, ASR_CHANNELS).mean(axis=1).astype(np.int16)
        rms  = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2)))
        vad_state["frame_count"] += 1
        vad_state["max_rms"] = max(vad_state["max_rms"], rms)

        if vad_state["frame_count"] % 200 == 0:
            print(f"Audio alive: frame={vad_state['frame_count']}"
                  f"  peak={vad_state['max_rms']:.0f}"
                  f"  floor={vad_state['noise_floor']:.0f}", flush=True)
            vad_state["max_rms"] = 0.0

        if not vad_state["in_speech"]:
            alpha = 0.05 if vad_state["frame_count"] < VAD_WARMUP_FRAMES \
                    else NOISE_FLOOR_EMA_ALPHA
            vad_state["noise_floor"] = (
                alpha * rms + (1.0 - alpha) * vad_state["noise_floor"]
            )

        vad_state["accumulator"].extend(mono.tolist())
        while len(vad_state["accumulator"]) >= VAD_FRAME_SAMPLES:
            chunk = np.array(vad_state["accumulator"][:VAD_FRAME_SAMPLES], dtype=np.int16)
            vad_state["accumulator"] = vad_state["accumulator"][VAD_FRAME_SAMPLES:]

            if vad_state["frame_count"] < VAD_WARMUP_FRAMES:
                continue

            chunk_rms   = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            above_floor = chunk_rms > vad_state["noise_floor"] * NOISE_FLOOR_FACTOR
            if above_floor:
                try:
                    is_speech = vad_engine.is_speech(chunk.tobytes(), ROBOT_SAMPLE_RATE)
                except Exception:
                    is_speech = False
            else:
                is_speech = False

            if is_speech:
                vad_state["speech_frames"]  += 1
                vad_state["silence_frames"]  = 0
                if vad_state["in_speech"]:
                    vad_state["buf"].extend(chunk.tolist())
                    vad_state["speech_samples"] += len(chunk)
                    if len(vad_state["buf"]) >= max_samples:
                        await _flush_audio()
                elif vad_state["speech_frames"] >= VAD_SPEECH_FRAMES:
                    vad_state["in_speech"] = True
                    for pre in vad_state["pre_roll"]:
                        vad_state["buf"].extend(pre)
                        vad_state["speech_samples"] += len(pre)
                    vad_state["pre_roll"].clear()
                    vad_state["buf"].extend(chunk.tolist())
                    vad_state["speech_samples"] += len(chunk)
                else:
                    vad_state["pre_roll"].append(chunk.tolist())
            else:
                vad_state["speech_frames"]  = 0
                vad_state["silence_frames"] += 1
                if vad_state["in_speech"]:
                    vad_state["buf"].extend(chunk.tolist())
                    if (vad_state["silence_frames"] >= VAD_SILENCE_FRAMES
                            or len(vad_state["buf"]) >= max_samples):
                        await _flush_audio()
                else:
                    vad_state["pre_roll"].append(chunk.tolist())

    async def _flush_audio():
        samples = np.array(list(vad_state["buf"]), dtype=np.int16)
        vad_state["buf"].clear()
        vad_state["in_speech"]      = False
        vad_state["speech_frames"]  = 0
        vad_state["silence_frames"] = 0
        vad_state["speech_samples"] = 0
        if len(samples) < min_samples:
            return
        print(f"ASR: sending {len(samples) / ROBOT_SAMPLE_RATE:.1f}s of audio", flush=True)
        text = await asyncio.get_event_loop().run_in_executor(
            None, stt.transcribe, samples
        )
        if text:
            print(f"Heard: {text}", flush=True)
            try:
                transcript_queue.put_nowait(text)
            except Exception:
                pass
        else:
            print("ASR: no speech detected in clip", flush=True)

    async def connect_loop():
        while not stop_event.is_set():
            conn        = None
            audiohub    = None
            audio_cache = {}

            nonlocal vad_state, vad_engine
            vad_engine = _webrtcvad.Vad(VAD_AGGRESSIVENESS)
            vad_state  = {
                "buf":           deque(),
                "accumulator":   [],
                "pre_roll":      deque(maxlen=VAD_PRE_ROLL_FRAMES),
                "in_speech":     False,
                "speech_frames": 0, "silence_frames": 0, "speech_samples": 0,
                "frame_count":   0, "max_rms": 0.0,
                "mute_until":    0.0,
                "noise_floor":   NOISE_FLOOR_INIT,
            }

            try:
                print(f"Connecting to {robot_ip}...", flush=True)
                conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=robot_ip)
                await asyncio.wait_for(conn.connect(), timeout=30.0)
                print("Connected. Starting video and mic...", flush=True)

                conn.video.switchVideoChannel(True)
                conn.video.add_track_callback(recv_video)
                if enable_audio:
                    conn.audio.switchAudioChannel(True)
                    conn.audio.add_track_callback(recv_audio)

                # Forward any transcripts from the robot's onboard chat_go service.
                def _on_gpt_feedback(msg):
                    try:
                        inner = msg.get("data") or msg
                        if isinstance(inner, str):
                            inner = json.loads(inner)
                        text = (inner.get("text") or inner.get("asr_result")
                                or inner.get("result") or "")
                    except Exception:
                        text = ""
                    if text:
                        print(f"chat_go ASR: {text}", flush=True)
                        try:
                            transcript_queue.put_nowait(text)
                        except Exception:
                            pass

                conn.datachannel.pub_sub.subscribe(RTC_TOPIC["GPT_FEEDBACK"],
                                                   _on_gpt_feedback)

                try:
                    audiohub = WebRTCAudioHub(conn)
                    print("AudioHub ready", flush=True)
                    await conn.datachannel.pub_sub.publish_request_new(
                        RTC_TOPIC["VUI"],
                        {"api_id": 1003, "parameter": {"volume": SPEAKER_VOLUME}},
                    )
                    print(f"Speaker volume set to {SPEAKER_VOLUME}", flush=True)
                except Exception as e:
                    print(f"AudioHub init failed (no TTS): {e}", flush=True)

                action_task = None
                pending_release = False
                while not stop_event.is_set() and conn.isConnected:
                    if audiohub:
                        try:
                            text = say_queue.get_nowait()
                            asyncio.ensure_future(_speak(text, audiohub, audio_cache))
                        except Empty:
                            pass
                    # Serialize actions: only pop the next one after the previous
                    # coroutine completes — so e.g. wiggle waits for stand_up
                    # instead of running concurrently and being clobbered.
                    if action_task is None or action_task.done():
                        if action_task is not None and action_task.done():
                            action_task = None
                            # An action just finished. If no follow-up is queued,
                            # release sticky Euler/Sit state once so the joystick
                            # works again. If a follow-up IS queued, skip release
                            # so it doesn't clobber the next action's pose commands.
                            pending_release = True
                        try:
                            action_name = action_queue.get_nowait()
                            fn = _ACTION_MAP.get(action_name)
                            if fn:
                                action_task = asyncio.ensure_future(fn(conn.datachannel))
                                pending_release = False
                        except Empty:
                            if pending_release:
                                action_task = asyncio.ensure_future(
                                    actions.release_control(conn.datachannel))
                                pending_release = False
                    await asyncio.sleep(0.5)

                print("Connection lost, reconnecting...", flush=True)
            except BaseException as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                print(f"WebRTC error: {type(e).__name__}: {e}", flush=True)
            finally:
                if conn:
                    try:
                        await conn.disconnect()
                    except Exception:
                        pass
            if not stop_event.is_set():
                await asyncio.sleep(RECONNECT_DELAY)

    # Initialise here so connect_loop's nonlocal can rebind them each reconnect.
    vad_engine = _webrtcvad.Vad(VAD_AGGRESSIVENESS)
    vad_state  = {}
    max_samples = int(MAX_SPEECH_SEC * ROBOT_SAMPLE_RATE)
    min_samples = int(MIN_SPEECH_SEC * ROBOT_SAMPLE_RATE)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(connect_loop())
    finally:
        loop.close()


# ── Main display loop ─────────────────────────────────────────────────────────

def main():
    robot_ip   = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.161"
    recognizer = load_recognizer()

    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption("Pella Camera")
    W, H = screen.get_size()
    transcript_font = pygame.font.SysFont("sans", 40)
    offline_surf    = _make_offline_surface(W, H)

    frame_queue:      Queue = Queue()
    say_queue:        Queue = Queue(maxsize=1)
    transcript_queue: Queue = Queue(maxsize=5)
    action_queue:     Queue = Queue(maxsize=4)
    stop_event = threading.Event()

    webrtc_thread = threading.Thread(
        target=_run_webrtc,
        args=(robot_ip, frame_queue, say_queue, transcript_queue, action_queue, stop_event),
        kwargs={"enable_audio": not USE_USB_MIC},
        daemon=True,
    )
    webrtc_thread.start()

    if USE_USB_MIC:
        threading.Thread(
            target=stt.run_usb_mic,
            args=(transcript_queue, stop_event),
            daemon=True,
        ).start()

    rec_in:   Queue = Queue(maxsize=1)
    rec_out:  Queue = Queue(maxsize=1)
    rec_thread = None
    if recognizer:
        rec_thread = threading.Thread(
            target=recognition_worker,
            args=(recognizer, rec_in, rec_out, stop_event),
            daemon=True,
        )
        rec_thread.start()

    # ── Display state ─────────────────────────────────────────────────────
    state            = TRACKING
    zoom_surf        = None
    zoom_start       = 0.0
    current_surf     = offline_surf
    transcript_text  = ""
    transcript_time  = 0.0

    # ── Per-frame detection state ─────────────────────────────────────────
    last_faces           = []
    # Subset of last_faces whose bbox is not flush against any frame edge.
    # Cropped faces are treated as "still searching" so seek pose keeps
    # adjusting until a fully-framed face appears.
    last_complete_faces  = []
    # Faces and names from the most recent recognition output, bundled
    # together to avoid index drift if last_faces is updated mid-recognition.
    last_rec_faces       = []
    last_rec_names       = []
    detect_counter       = 0
    last_frame_time      = 0.0
    prev_frame           = None

    # ── Physical body pose ────────────────────────────────────────────────
    camera_pose      = "level"   # "level" | "seeking" | "seeking_sit"
    seek_start       = 0.0
    last_motion_time = 0.0

    # ── Interaction state machine ─────────────────────────────────────────
    interaction          = INTERACTION_IDLE
    interaction_entered  = 0.0
    # Recognition vote buffer for the current RECOGNIZING session.
    recog_obs            = deque(maxlen=RECOG_VOTES_REQUIRED)
    # Best (sharpest) complete face captured during the current RECOGNIZING
    # session — used to enroll an unknown person even if the face has left
    # the frame by the moment the verdict fires.
    recog_best_face      = None
    # Hold-off timer: while now < recog_stabilize_until, skip vote/best-face
    # updates so motion blur from the just-queued recovery action doesn't
    # corrupt the verdict. 0 means no stabilization required.
    recog_stabilize_until = 0.0
    last_complete_seen   = 0.0
    # Latest video frame dimensions, kept around so the recognition output
    # (which doesn't carry dimensions) can be edge-filtered.
    frame_w              = 0
    frame_h              = 0
    # Per-name cooldown so we don't re-greet the same person back-to-back.
    last_greeted         = {}
    last_introduced      = 0.0
    enroll_state         = {"active": False, "frame": None, "landmarks": None,
                            "bbox":   None,  "asked_at": 0.0, "best_sharpness": 0.0}

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            now = time.monotonic()

            # Pull the latest transcript (discard all but newest).
            try:
                while True:
                    transcript_text = transcript_queue.get_nowait()
                    transcript_time = now
                    print(f"Display: {transcript_text}", flush=True)
                    if enroll_state["active"]:
                        name_raw = _parse_name(transcript_text)
                        if not name_raw:
                            enroll_state["active"] = False
                            try:
                                say_queue.put_nowait(
                                    "Sorry, I didn't catch your name.")
                            except Exception:
                                pass
                            print(f"Enrollment skipped: '{transcript_text}' "
                                  f"didn't parse as a name", flush=True)
                            continue
                        dir_name     = name_raw.lower().replace(" ", "_")
                        display_name = name_raw.title()
                        save_dir     = os.path.join(FACE_IDS_DIR, dir_name)
                        if recognizer:
                            enrolled = recognizer.enroll_new(
                                dir_name,
                                enroll_state["frame"],
                                enroll_state["landmarks"],
                                enroll_state.get("bbox"),
                                save_dir,
                            )
                        else:
                            os.makedirs(save_dir, exist_ok=True)
                            cv2.imwrite(os.path.join(save_dir, "001.jpg"),
                                        enroll_state["frame"])
                            enrolled = True
                        enroll_state["active"] = False
                        try:
                            if enrolled:
                                say_queue.put_nowait(
                                    f"Nice to meet you, {display_name}!")
                            else:
                                say_queue.put_nowait(
                                    f"You don't look like the {display_name} "
                                    f"I know. Sorry about that.")
                            print(f"{'Enrolled' if enrolled else 'Rejected'}: "
                                  f"{display_name}", flush=True)
                        except Exception as e:
                            print(f"{'Enrolled' if enrolled else 'Rejected'}: "
                                  f"{display_name} (say_queue full? {e})",
                                  flush=True)
            except Empty:
                pass

            # Cancel enrollment if person didn't respond in time.
            if enroll_state["active"] and now - enroll_state["asked_at"] >= ENROLL_TIMEOUT:
                enroll_state["active"] = False
                print("Enrollment timeout: no name received", flush=True)

            # (Seek-timeout escalation and recovery are handled inside the
            # interaction state machine below.)

            # Zoom state: hold the zoomed view for ZOOM_DURATION seconds.
            # The interaction state machine still advances (timers expire,
            # COOLDOWN transitions) — only the display is locked to the zoom.
            if state == ZOOMED:
                if now - zoom_start >= ZOOM_DURATION:
                    state = TRACKING
                else:
                    screen.blit(zoom_surf, (0, 0))
                    if transcript_text and now - transcript_time < TRANSCRIPT_DISPLAY_SEC:
                        _draw_transcript(screen, transcript_font, transcript_text, W, H)
                    pygame.display.flip()
                    pygame.time.wait(FRAME_MS)
                    continue

            # Pull latest recognition result. Each output is a (faces, names)
            # bundle so name lookups are immune to last_faces index drift.
            # In RECOGNIZING state, each new bundle adds one vote for the
            # biggest-face's identity to the debounce buffer.
            rec_got_new = False
            try:
                while True:
                    last_rec_faces, last_rec_names = rec_out.get_nowait()
                    rec_got_new = True
            except Empty:
                pass
            if (rec_got_new and interaction == INTERACTION_RECOGNIZING
                    and last_rec_faces and now >= recog_stabilize_until):
                # Only vote on fully-framed faces — a cropped face can't be
                # reliably recognized and would poison the verdict with a
                # spurious "unknown". Also skip during stabilization so
                # motion-blurred frames don't contribute.
                complete_rec = [f for f in last_rec_faces
                                if not is_face_at_edge(f, frame_w, frame_h)] \
                               if frame_w and frame_h else []
                if complete_rec:
                    biggest = max(complete_rec, key=lambda f: f[2] * f[3])
                    idx = last_rec_faces.index(biggest)
                    rec_name = last_rec_names[idx] if idx < len(last_rec_names) else None
                    recog_obs.append(rec_name)

            # ── Process newest video frame ────────────────────────────────
            img = None
            try:
                img = frame_queue.get_nowait()
                last_frame_time = now
            except Empty:
                if now - last_frame_time > 3.0:
                    current_surf = offline_surf

            if img is not None:
                # Motion + person check is used by the state machine to decide
                # IDLE → SEEKING. We only care about motion when at level pose;
                # during seek/recognize the pose change itself causes motion.
                motion_seen = False
                if (camera_pose == "level"
                        and now - last_motion_time >= MOTION_COOLDOWN
                        and detect_motion(img, prev_frame)
                        and detect_person(img)):
                    motion_seen = True
                    last_motion_time = now
                prev_frame = img

                detect_counter += 1
                if detect_counter >= FACE_DETECT_EVERY:
                    detect_counter = 0
                    last_faces = detect_faces(img)
                    img_h, img_w = img.shape[:2]
                    frame_w, frame_h = img_w, img_h
                    last_complete_faces = [
                        f for f in last_faces
                        if not is_face_at_edge(f, img_w, img_h)
                    ]
                    if last_complete_faces:
                        last_complete_seen = now
                    if recognizer and last_faces:
                        try:
                            rec_in.put_nowait((img.copy(), last_faces))
                        except Exception:
                            pass
                    # While in RECOGNIZING, keep the sharpest complete face seen
                    # so we can enroll an unknown person even if the face has
                    # left the frame by the time the verdict fires. Skip during
                    # stabilization — those frames are motion-blurred.
                    if (interaction == INTERACTION_RECOGNIZING
                            and last_complete_faces
                            and now >= recog_stabilize_until):
                        biggest = max(last_complete_faces, key=lambda f: f[2] * f[3])
                        bx, by, bw, bh = biggest[:4]
                        region = img[by:by + bh, bx:bx + bw]
                        if region.size > 0:
                            s = sharpness(region)
                            if recog_best_face is None or s > recog_best_face["sharpness"]:
                                recog_best_face = {
                                    "img":       img.copy(),
                                    "bbox":      (bx, by, bw, bh),
                                    "landmarks": biggest[4] if len(biggest) > 4 else None,
                                    "sharpness": s,
                                }
                    # During an active enrollment, keep the sharpest fully-framed
                    # face seen so the saved image is the best one in the window.
                    if enroll_state["active"] and last_complete_faces:
                        biggest = max(last_complete_faces, key=lambda f: f[2] * f[3])
                        bx, by, bw, bh = biggest[:4]
                        region = img[by:by + bh, bx:bx + bw]
                        if region.size > 0:
                            s = sharpness(region)
                            if s > enroll_state["best_sharpness"]:
                                enroll_state.update({
                                    "frame":          img.copy(),
                                    "landmarks":      biggest[4] if len(biggest) > 4 else None,
                                    "bbox":           (bx, by, bw, bh),
                                    "best_sharpness": s,
                                })
                                print(f"Enroll: sharper face captured "
                                      f"(sharpness={s:.1f})", flush=True)

                # ── Interaction state machine ─────────────────────────────
                if interaction == INTERACTION_IDLE:
                    if last_complete_faces:
                        # Face already visible at level — skip seeking.
                        # No recovery action queued, so no stabilization needed.
                        interaction = INTERACTION_RECOGNIZING
                        interaction_entered = now
                        recog_obs.clear()
                        recog_best_face = None
                        recog_stabilize_until = 0.0
                        last_complete_seen = now
                        print("Interaction: -> recognizing (face visible at level)",
                              flush=True)
                    elif motion_seen:
                        # Person spotted but no face — start seeking.
                        interaction = INTERACTION_SEEKING
                        interaction_entered = now
                        camera_pose = "seeking"
                        seek_start  = now
                        try:
                            action_queue.put_nowait("look_up")
                        except Exception:
                            pass
                        print("Interaction: -> seeking (motion + person)",
                              flush=True)

                elif interaction == INTERACTION_SEEKING:
                    if last_complete_faces:
                        # Found a face — transition to RECOGNIZING while
                        # STAYING in the current seek pose (sit_look_up or
                        # look_up). The recovery action (stand_up /
                        # look_level) is queued only when we LEAVE
                        # RECOGNIZING, so the body holds steady throughout
                        # vote accumulation and gives a clean capture
                        # window. Stabilization here covers any residual
                        # motion from the seek action still settling
                        # (the sit transition is noticeably longer than
                        # the look_up tilt).
                        if camera_pose == "seeking_sit":
                            stabilize_sec = RECOG_STABILIZE_SIT_SEC
                        elif camera_pose == "seeking":
                            stabilize_sec = RECOG_STABILIZE_LOOK_SEC
                        else:
                            stabilize_sec = 0.0
                        interaction = INTERACTION_RECOGNIZING
                        interaction_entered = now
                        recog_obs.clear()
                        recog_best_face = None
                        recog_stabilize_until = (now + stabilize_sec
                                                 if stabilize_sec > 0 else 0.0)
                        last_complete_seen = now
                        print(f"Interaction: -> recognizing "
                              f"(holding {camera_pose} pose"
                              + (f", stabilizing for {stabilize_sec:.1f}s)"
                                 if stabilize_sec > 0 else ")"),
                              flush=True)
                    elif (camera_pose == "seeking"
                            and now - seek_start >= SEEK_TIMEOUT
                            and not _queue_contains(action_queue, "look_up")):
                        # look_up didn't find a face — escalate to sit.
                        camera_pose = "seeking_sit"
                        seek_start  = now
                        try:
                            action_queue.put_nowait("sit_look_up")
                        except Exception:
                            pass
                        print("Seek timeout: escalating to sit_look_up",
                              flush=True)
                    elif (camera_pose == "seeking_sit"
                            and now - seek_start >= SEEK_TIMEOUT):
                        # Sit also failed — stand back up and cool down.
                        try:
                            action_queue.put_nowait("stand_up")
                        except Exception:
                            pass
                        camera_pose = "level"
                        last_motion_time = now
                        interaction = INTERACTION_COOLDOWN
                        interaction_entered = now
                        print("Seek give-up: -> cooldown", flush=True)

                elif interaction == INTERACTION_RECOGNIZING:
                    decide_now = (len(recog_obs) >= RECOG_VOTES_REQUIRED
                                  or now - interaction_entered >= RECOG_TIMEOUT_SEC)
                    face_lost  = now - last_complete_seen > FACE_LOST_TIMEOUT_SEC

                    # If we're about to leave RECOGNIZING, queue the recovery
                    # action now (after the body has held the seek pose for
                    # the whole capture window). For greetings the wiggle
                    # queued by the known-verdict branch lands AFTER this
                    # recovery, so the consumer order becomes
                    # recovery -> wiggle.
                    if (decide_now or face_lost) and camera_pose != "level":
                        recovery = ("stand_up" if camera_pose == "seeking_sit"
                                    else "look_level")
                        _drain_seek_actions(action_queue)
                        try:
                            action_queue.put_nowait(recovery)
                        except Exception:
                            pass
                        camera_pose = "level"
                        last_motion_time = now
                        print(f"Leaving recognizing: queueing {recovery}",
                              flush=True)

                    if decide_now:
                        # Verdict takes precedence over face_lost: as long as we
                        # accumulated enough votes (or hit timeout) we'll act on
                        # them — the captured recog_best_face lets us enroll an
                        # unknown person even if they've already turned away.
                        verdict, kind = _tally_recognition(recog_obs)
                        if kind == "known":
                            display_name = verdict.replace("_", " ").title()
                            if (now - last_greeted.get(verdict, 0.0)) < GREET_COOLDOWN:
                                interaction = INTERACTION_COOLDOWN
                                interaction_entered = now
                                print(f"Already greeted {display_name} recently "
                                      f"-> cooldown", flush=True)
                            else:
                                try:
                                    say_queue.put_nowait(f"Hi, {display_name}")
                                    action_queue.put_nowait("wiggle")
                                except Exception:
                                    pass
                                last_greeted[verdict] = now
                                last_introduced       = now   # also suppress introduce
                                last_motion_time      = now
                                # Prefer the live frame for the zoom label, but fall
                                # back to the captured snapshot if the person has
                                # already moved out of frame.
                                if last_complete_faces:
                                    biggest = max(last_complete_faces,
                                                  key=lambda f: f[2] * f[3])
                                    crop = zoom_crop(img, biggest)
                                elif recog_best_face is not None:
                                    crop = zoom_crop(recog_best_face["img"],
                                                     recog_best_face["bbox"])
                                else:
                                    crop = None
                                if crop is not None:
                                    zoom_bgr = crop.copy()
                                    cv2.putText(zoom_bgr, verdict,
                                                (20, zoom_bgr.shape[0] - 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                                (0, 255, 0), 2, cv2.LINE_AA)
                                    zoom_surf    = _to_surface(zoom_bgr, (W, H))
                                    state        = ZOOMED
                                    zoom_start   = now
                                    current_surf = zoom_surf
                                interaction = INTERACTION_GREETING
                                interaction_entered = now
                                print(f"Greeting: {display_name}", flush=True)
                        elif kind == "unknown":
                            if enroll_state["active"]:
                                interaction = INTERACTION_COOLDOWN
                                interaction_entered = now
                                print("Unknown but enrollment already active "
                                      "-> cooldown", flush=True)
                            elif now - last_introduced < INTRODUCE_COOLDOWN:
                                interaction = INTERACTION_COOLDOWN
                                interaction_entered = now
                                print(f"Unknown but introduce on cooldown "
                                      f"({INTRODUCE_COOLDOWN - (now - last_introduced):.1f}s "
                                      f"remaining) -> cooldown", flush=True)
                            elif (recog_best_face is None
                                  or recog_best_face["sharpness"] < SHARPNESS_THRESHOLD):
                                interaction = INTERACTION_COOLDOWN
                                interaction_entered = now
                                sharp_str = (f"{recog_best_face['sharpness']:.1f}"
                                             if recog_best_face else "none")
                                print(f"Unknown face not sharp enough "
                                      f"(best sharpness={sharp_str}, "
                                      f"threshold={SHARPNESS_THRESHOLD}) -> cooldown",
                                      flush=True)
                            else:
                                # Enroll from the best face we captured during
                                # recognition — works even if the person already
                                # turned away.
                                try:
                                    say_queue.put_nowait(
                                        "Hello, I am Pella. What is your name?")
                                except Exception as e:
                                    print(f"WARN: say_queue full, "
                                          f"name-ask not queued: {e}",
                                          flush=True)
                                enroll_state.update({
                                    "active":         True,
                                    "frame":          recog_best_face["img"],
                                    "landmarks":      recog_best_face["landmarks"],
                                    "bbox":           recog_best_face["bbox"],
                                    "asked_at":       now,
                                    "best_sharpness": recog_best_face["sharpness"],
                                })
                                last_introduced  = now
                                last_motion_time = now
                                crop = zoom_crop(recog_best_face["img"],
                                                 recog_best_face["bbox"])
                                zoom_surf    = _to_surface(crop, (W, H))
                                state        = ZOOMED
                                zoom_start   = now
                                current_surf = zoom_surf
                                interaction = INTERACTION_INTRODUCING
                                interaction_entered = now
                                print(f"Introducing: asking for name "
                                      f"(sharpness={recog_best_face['sharpness']:.1f})",
                                      flush=True)
                        else:  # ambiguous votes after full window — give up
                            interaction = INTERACTION_COOLDOWN
                            interaction_entered = now
                            print("Recognition ambiguous -> cooldown", flush=True)
                    elif face_lost:
                        # No verdict yet AND face is gone — give up.
                        interaction = INTERACTION_COOLDOWN
                        interaction_entered = now
                        print("Face lost before recognition completed -> cooldown",
                              flush=True)

                elif interaction == INTERACTION_GREETING:
                    if now - interaction_entered >= GREETING_DURATION_SEC:
                        interaction = INTERACTION_COOLDOWN
                        interaction_entered = now
                        print("Greeting complete -> cooldown", flush=True)

                elif interaction == INTERACTION_INTRODUCING:
                    # Driven by the transcript handler / enrollment timeout —
                    # both flip enroll_state["active"] back to False when done.
                    if not enroll_state["active"]:
                        interaction = INTERACTION_COOLDOWN
                        interaction_entered = now
                        print("Introducing complete -> cooldown", flush=True)

                elif interaction == INTERACTION_COOLDOWN:
                    if now - interaction_entered >= INTERACTION_COOLDOWN_SEC:
                        interaction = INTERACTION_IDLE
                        interaction_entered = now
                        print("Cooldown done -> idle", flush=True)

                # ── Live (non-zoom) display ───────────────────────────────
                if state != ZOOMED:
                    annotated    = annotate(img, last_faces) if last_faces else img
                    current_surf = _to_surface(annotated, (W, H))

            screen.blit(current_surf, (0, 0))
            if transcript_text and now - transcript_time < TRANSCRIPT_DISPLAY_SEC:
                _draw_transcript(screen, transcript_font, transcript_text, W, H)
            pygame.display.flip()
            pygame.time.wait(FRAME_MS)

    finally:
        stop_event.set()
        pygame.quit()
        webrtc_thread.join(timeout=5.0)
        if rec_thread:
            rec_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
