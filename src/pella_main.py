#!/usr/bin/env python3
"""Pella's brain: a thin sense-and-respond shell that dispatches to a task manager.

Each iteration of the main loop:

  1. Drain transcript_queue; forward to the task manager (which routes
     to the active task; tasks use transcripts to finish enrollment,
     respond to commands, etc.).
  2. tick() the task manager — the active task does its own sensing
     (frames, motion, face detection, …) and returns:
        * latest_frame + faces for the live annotated display
        * an optional display_request (image + duration) to pin on screen
        * an optional status event for logging / further routing
  3. Render: pinned image while active, otherwise the annotated live frame.

pella_main has zero knowledge of any concrete task type. Task selection,
state machines, recognition policy — all live behind task_manager. To
add a new behavior, register it in task_manager; pella_main does not
change.

The Go2 only accepts a single WebRTC peer connection; pella_main owns
that connection and hands typed handles to each organ:
  * front_camera (eye)   — video track → frame_queue
  * tts          (mouth) — AudioHub
  * actions      (limbs) — data channel
  * stt          (ear)   — USB-mic thread (unrelated to WebRTC)
"""

import asyncio
import logging
import os
import sys
import threading
import time
from queue import Queue, Empty

import av
av.logging.set_level(av.logging.FATAL)
try:
    import ctypes, ctypes.util
    _libavutil = ctypes.CDLL(ctypes.util.find_library("avutil"))
    _libavutil.av_log_set_level(-8)  # AV_LOG_QUIET — silences swscaler spam
except Exception:
    pass

import dotenv
import numpy as np
import pygame

dotenv.load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
os.environ.setdefault("DISPLAY", ":0")
logging.basicConfig(level=logging.FATAL)

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_APP_DIR, "..", "go2_webrtc_connect"))

from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.webrtc_audiohub import WebRTCAudioHub
from go2_webrtc_driver.constants import RTC_TOPIC

import actions
import front_camera
import stt
import task_manager
import tts
from stt import USE_USB_MIC, SPEAKER_VOLUME
import vision
from vision import annotate


# ── Display constants ─────────────────────────────────────────────────────────
RECONNECT_DELAY        = 5.0
FRAME_MS               = 20
TRANSCRIPT_DISPLAY_SEC = 8.0


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
    """Render heard-transcript text centred at the bottom of the screen.

    Drawn in cyan to make it visually distinct from other overlays
    (e.g. the green name labels in zoom views), reading as "this is
    what Pella heard you say".
    """
    padding = 12
    surf  = font.render(text, True, (0, 255, 255))   # cyan = "heard"
    bg_w  = surf.get_width()  + padding * 2
    bg_h  = surf.get_height() + padding * 2
    bg    = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 160))
    x = (w - bg_w) // 2
    y = h - bg_h - 30
    screen.blit(bg, (x, y))
    screen.blit(surf, (x + padding, y + padding))


# ── Robot connection lifecycle ───────────────────────────────────────────────

def _make_gpt_feedback_handler(transcript_queue):
    """Build a callback for the robot's onboard chat_go ASR pub/sub topic."""
    def _on_gpt_feedback(msg):
        import json
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
                # Robot-side ASR has no capture timestamps — stamp start and
                # end with "now" so downstream treats it as just-spoken speech
                # with zero duration (no stitching possible from this path).
                now = time.monotonic()
                transcript_queue.put_nowait((text, now, now))
            except Exception:
                pass
    return _on_gpt_feedback


def _run_robot_loop(robot_ip, frame_queue, say_queue, prep_queue,
                    transcript_queue, action_queue, stop_event,
                    warm_phrases=None):
    """Maintain the WebRTC connection and run all organ consumers on its loop.

    The Go2 only accepts a single WebRTC peer connection, so video (eye),
    audio output (mouth via audiohub), and sport-mode commands (limbs via
    datachannel) all share one connection here. When the connection drops we
    cancel the organ consumer coroutines, reconnect, and relaunch them.
    """

    async def _video_track_callback(track):
        await front_camera.recv_video(track, frame_queue, stop_event)

    async def connect_loop():
        # Process-lifetime TTS cache: text_hash -> {path, uuid, duration, lock}.
        # Lives outside the reconnect loop so cached gTTS .wav paths survive
        # WebRTC drops. On each reconnect we only invalidate the per-session
        # `uuid` field (the new AudioHub doesn't know about the old session's
        # UUIDs); the .wav files on /tmp stay valid, so warm-up after a
        # reconnect re-uploads to the new AudioHub without re-fetching from
        # gTTS — no internet round-trip needed. Critical on flaky WiFi where
        # the LAN drop and a WAN blip often co-occur.
        audio_cache = {}

        while not stop_event.is_set():
            conn = None
            audiohub = None
            consumer_tasks = []
            # Per-connection session: invalidate AudioHub UUIDs from any
            # prior connection so the next warm-up re-uploads to *this*
            # AudioHub. Preserves path/duration/lock.
            tts.invalidate_audiohub_uuids(audio_cache)

            try:
                print(f"Connecting to {robot_ip}...", flush=True)
                conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA,
                                           ip=robot_ip)
                await asyncio.wait_for(conn.connect(), timeout=30.0)
                print("Connected. Starting video and mic...", flush=True)

                # Eye — wire the video track to front_camera.recv_video.
                conn.video.switchVideoChannel(True)
                conn.video.add_track_callback(_video_track_callback)

                # Onboard chat_go ASR feedback (if the robot publishes on it).
                conn.datachannel.pub_sub.subscribe(
                    RTC_TOPIC["GPT_FEEDBACK"],
                    _make_gpt_feedback_handler(transcript_queue),
                )

                # AudioHub playback state — ~4 Hz heartbeat reporting current
                # play state. tts.py uses the playing -> stopped transition
                # to shorten tts_mute_until and release the mic as soon as
                # actual audio finishes, rather than waiting for the time-
                # based fallback. Schema verified via
                # scripts/probe_audiohub_state.py.
                conn.datachannel.pub_sub.subscribe(
                    RTC_TOPIC["AUDIO_HUB_PLAY_STATE"],
                    tts.make_player_state_callback(),
                )

                # Mouth — initialise the AudioHub and set speaker volume.
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
                    audiohub = None

                # Pre-cache phrases the active task expects to say so the
                # first-time gen+upload latency doesn't show up mid-interaction.
                # Runs in the background; doesn't block other consumers.
                if audiohub and warm_phrases:
                    consumer_tasks.append(asyncio.ensure_future(
                        tts.run_warmup(warm_phrases, audiohub, audio_cache)))

                # Launch organ consumers on the same asyncio loop.
                if audiohub:
                    consumer_tasks.append(asyncio.ensure_future(
                        tts.run_say_consumer(say_queue, audiohub,
                                             stop_event, audio_cache)))
                    # Pre-cache requests from tasks (e.g. names parsed at
                    # runtime) — shares the audio_cache with the say
                    # consumer so a later say() hits the warm UUID.
                    consumer_tasks.append(asyncio.ensure_future(
                        tts.run_prep_consumer(prep_queue, audiohub,
                                              stop_event, audio_cache)))
                consumer_tasks.append(asyncio.ensure_future(
                    actions.run_action_consumer(action_queue,
                                                conn.datachannel, stop_event)))

                # Idle until connection drops or shutdown is signalled.
                while not stop_event.is_set() and conn.isConnected:
                    await asyncio.sleep(0.5)

                print("Connection lost, reconnecting...", flush=True)
            except BaseException as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                print(f"WebRTC error: {type(e).__name__}: {e}", flush=True)
            finally:
                for t in consumer_tasks:
                    if not t.done():
                        t.cancel()
                if conn:
                    try:
                        await conn.disconnect()
                    except Exception:
                        pass
            if not stop_event.is_set():
                await asyncio.sleep(RECONNECT_DELAY)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(connect_loop())
    finally:
        loop.close()


# ── Main entry: sense-and-respond shell ──────────────────────────────────────

def main():
    robot_ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.161"

    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption("Pella Camera")
    W, H = screen.get_size()
    transcript_font = pygame.font.SysFont("sans", 40)
    offline_surf    = _make_offline_surface(W, H)

    # Inter-organ queues — owned here, handed to each organ via thread args.
    frame_queue:      Queue = Queue()
    say_queue:        Queue = Queue(maxsize=1)
    # prep_queue is a generic "warm the TTS cache for this phrase NOW" channel.
    # Tasks push to it when they know they'll soon want to say something whose
    # text wasn't predictable at startup (e.g. "Nice to meet you, <NewName>!"
    # the moment a new name is parsed from STT).
    prep_queue:       Queue = Queue(maxsize=4)
    transcript_queue: Queue = Queue(maxsize=5)
    action_queue:     Queue = Queue(maxsize=4)
    stop_event = threading.Event()

    # The task manager owns all task instantiation, selection, and shared
    # perception resources (e.g. the face recognizer). pella_main only ever
    # tick()s it and forwards transcripts. Build it first so we can hand its
    # warm-phrase list to the robot thread on startup.
    tasks = task_manager.TaskManager(
        frame_queue, action_queue, say_queue, prep_queue, stop_event)
    warm_phrases = tasks.get_warm_phrases()

    # Robot thread — eye + mouth + limbs sharing one WebRTC connection.
    robot_thread = threading.Thread(
        target=_run_robot_loop,
        args=(robot_ip, frame_queue, say_queue, prep_queue, transcript_queue,
              action_queue, stop_event, warm_phrases),
        daemon=True,
    )
    robot_thread.start()

    # Ear — USB-mic VAD + Whisper, writes into transcript_queue.
    if USE_USB_MIC:
        # Enable diagnostic clip dumping. recog_greeting calls
        # stt.arm_debug_audio_window() after each enrollment-prompt TTS so
        # only the user-reply listening windows get saved (not every clip).
        stt.configure_debug_audio_dir(vision.DEBUG_AUDIO_DIR)
        threading.Thread(
            target=stt.run_usb_mic,
            args=(transcript_queue, stop_event),
            daemon=True,
        ).start()

    # ── Display state ─────────────────────────────────────────────────────
    # A task may ask us to pin a specific image on screen for N seconds via
    # TickResult.display_request. While now < pinned_until, that image is
    # shown; otherwise the live annotated frame is shown.
    pinned_image     = None
    pinned_until     = 0.0
    current_surf     = offline_surf
    transcript_text  = ""
    transcript_time  = 0.0
    last_frame_time  = 0.0   # for offline-placeholder fallback

    try:
        while True:
            # pygame events.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            now = time.monotonic()

            # Drain transcripts → forward to the active task and cache the
            # latest for the on-screen overlay.
            try:
                while True:
                    text, capture_t, end_t = transcript_queue.get_nowait()
                    transcript_text = text
                    transcript_time = now
                    print(f"Display: {text}", flush=True)
                    tasks.submit_transcript(now, text, capture_t, end_t)
            except Empty:
                pass

            # Tick the task manager — the active task does its own sensing.
            result = tasks.tick(now)

            if result.status_event is not None:
                print(f"Task event: {result.status_event}", flush=True)

            # If the task asked us to pin an image, store it. pella_main has no
            # opinion on what the image is for — just shows it for the duration.
            if result.display_request is not None:
                pinned_image = result.display_request.image
                pinned_until = now + result.display_request.duration_sec

            # Decide what to render: pinned image (while still active) or the
            # latest live annotated frame.
            if pinned_image is not None and now < pinned_until:
                current_surf = _to_surface(pinned_image, (W, H))
            elif result.latest_frame is not None:
                pinned_image = None
                last_frame_time = now
                if result.latest_faces:
                    annotated = annotate(result.latest_frame,
                                         result.latest_faces)
                else:
                    annotated = result.latest_frame
                current_surf = _to_surface(annotated, (W, H))
            elif now - last_frame_time > 3.0:
                pinned_image = None
                current_surf = offline_surf

            screen.blit(current_surf, (0, 0))
            if transcript_text and now - transcript_time < TRANSCRIPT_DISPLAY_SEC:
                _draw_transcript(screen, transcript_font, transcript_text, W, H)
            pygame.display.flip()
            pygame.time.wait(FRAME_MS)

    finally:
        stop_event.set()
        pygame.quit()
        robot_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
