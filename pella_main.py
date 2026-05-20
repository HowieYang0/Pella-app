#!/usr/bin/env python3
"""Pella's brain: orchestrate the eye, mouth, ear, and limbs.

Pella's organs each live in a focused module:

  - front_camera (eye)   — receives WebRTC video frames -> frame_queue
  - tts          (mouth) — plays text from say_queue via Go2 audiohub
  - stt          (ear)   — USB-mic VAD + Whisper -> transcript_queue
  - actions      (limbs) — motor primitives + action_queue executor

This module is the central command center. It owns:
  - the single Go2 WebRTC connection (shared by eye / mouth / limbs)
  - the inter-organ queues (frame / say / transcript / action)
  - the interaction state machine (IDLE -> SEEKING -> RECOGNIZING ->
    GREETING / INTRODUCING -> COOLDOWN)
  - the pygame display surface
  - transcript-driven enrollment handoff

The current state machine corresponds to one task (face interaction).
Future task_manager (see project plan) will sit above pella_main as a
peer-orchestrator that emits primitive names into the same action_queue
and say_queue.
"""

import asyncio
import logging
import os
import re
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
import tts
from stt import USE_USB_MIC, SPEAKER_VOLUME
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
INTERACTION_IDLE         = "idle"
INTERACTION_SEEKING      = "seeking"
INTERACTION_RECOGNIZING  = "recognizing"
INTERACTION_GREETING     = "greeting"
INTERACTION_INTRODUCING  = "introducing"
INTERACTION_COOLDOWN     = "cooldown"

# Recognition is debounced across multiple detection cycles so a single
# misrecognition doesn't drive a wrong greeting/introduction. At ~6 detection
# cycles per second, 3 observations yields a decision in ~0.5 s.
RECOG_VOTES_REQUIRED    = 3
RECOG_AGREE_MIN         = 2
RECOG_TIMEOUT_SEC       = 6.0
# Look_level is a quick body-tilt back to level (~0.5 s covers it). stand_up
# from sit is a full rise — leg unfold + balance — and runs noticeably longer,
# so its stabilization needs to wait out the whole motion before any face is
# sampled. With the seek pose now held throughout RECOGNIZING (recovery is
# queued at RECOGNIZING exit), these durations target the time before we start
# sampling — i.e. how long we let the body settle in the seek pose first.
RECOG_STABILIZE_LOOK_SEC = 0.5
RECOG_STABILIZE_SIT_SEC  = 1.5
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
                transcript_queue.put_nowait(text)
            except Exception:
                pass
    return _on_gpt_feedback


def _run_robot_loop(robot_ip, frame_queue, say_queue, transcript_queue,
                    action_queue, stop_event):
    """Maintain the WebRTC connection and run all organ consumers on its loop.

    The Go2 only accepts a single WebRTC peer connection, so video (eye), audio
    output (mouth via audiohub), and sport-mode commands (limbs via datachannel)
    all share one connection that lives here. When the connection drops we
    cancel the organ consumer coroutines, reconnect, and relaunch them.
    """

    async def _video_track_callback(track):
        await front_camera.recv_video(track, frame_queue, stop_event)

    async def connect_loop():
        while not stop_event.is_set():
            conn = None
            audiohub = None
            audio_cache = {}
            consumer_tasks = []

            try:
                print(f"Connecting to {robot_ip}...", flush=True)
                conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=robot_ip)
                await asyncio.wait_for(conn.connect(), timeout=30.0)
                print("Connected. Starting video and mic...", flush=True)

                # Eye — wire the video track to front_camera.recv_video.
                conn.video.switchVideoChannel(True)
                conn.video.add_track_callback(_video_track_callback)

                # Onboard chat_go ASR feedback (forwarded into transcript_queue
                # if the robot is configured to publish on this topic).
                conn.datachannel.pub_sub.subscribe(
                    RTC_TOPIC["GPT_FEEDBACK"],
                    _make_gpt_feedback_handler(transcript_queue),
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

                # Launch organ consumers on the same asyncio loop.
                if audiohub:
                    consumer_tasks.append(asyncio.ensure_future(
                        tts.run_say_consumer(say_queue, audiohub,
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
                # Stop organ consumers before tearing down the connection.
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


# ── Main entry: display + interaction state machine ──────────────────────────

def main():
    robot_ip   = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.161"
    recognizer = load_recognizer()

    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption("Pella Camera")
    W, H = screen.get_size()
    transcript_font = pygame.font.SysFont("sans", 40)
    offline_surf    = _make_offline_surface(W, H)

    # Inter-organ queues — owned here, handed to each organ via thread args.
    frame_queue:      Queue = Queue()
    say_queue:        Queue = Queue(maxsize=1)
    transcript_queue: Queue = Queue(maxsize=5)
    action_queue:     Queue = Queue(maxsize=4)
    stop_event = threading.Event()

    # Robot thread (eye + mouth + limbs sharing one WebRTC connection).
    robot_thread = threading.Thread(
        target=_run_robot_loop,
        args=(robot_ip, frame_queue, say_queue, transcript_queue,
              action_queue, stop_event),
        daemon=True,
    )
    robot_thread.start()

    # Ear — USB-mic VAD + Whisper, writes into transcript_queue.
    if USE_USB_MIC:
        threading.Thread(
            target=stt.run_usb_mic,
            args=(transcript_queue, stop_event),
            daemon=True,
        ).start()

    # Face-recognition worker — reads (image, faces) pairs, writes (faces, names).
    rec_in:  Queue = Queue(maxsize=1)
    rec_out: Queue = Queue(maxsize=1)
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
    last_complete_faces  = []
    # Bundle from the most recent recognition output — keeps faces+names
    # in sync so name lookups are immune to last_faces index drift.
    last_rec_faces       = []
    last_rec_names       = []
    detect_counter       = 0
    last_frame_time      = 0.0
    prev_frame           = None

    # ── Physical body pose (tracked logically; the actual pose follows the
    # consumed action_queue items, which may lag this state by ~1 s) ──────
    camera_pose      = "level"   # "level" | "seeking" | "seeking_sit"
    seek_start       = 0.0
    last_motion_time = 0.0

    # ── Interaction state machine ─────────────────────────────────────────
    interaction          = INTERACTION_IDLE
    interaction_entered  = 0.0
    recog_obs            = deque(maxlen=RECOG_VOTES_REQUIRED)
    # Best (sharpest) complete face captured during the current RECOGNIZING
    # session — used to enroll an unknown person even if the face has left
    # the frame by the moment the verdict fires.
    recog_best_face      = None
    # Hold-off timer: while now < recog_stabilize_until, skip vote/best-face
    # updates so motion blur from residual seek-pose settling doesn't corrupt
    # the verdict. 0 means no stabilization required.
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
                            "bbox":   None,  "asked_at": 0.0,
                            "best_sharpness": 0.0}

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            now = time.monotonic()

            # ── Transcript handling (drives enrollment finalization) ──────
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

            # Cancel enrollment if the person didn't respond in time.
            if enroll_state["active"] and now - enroll_state["asked_at"] >= ENROLL_TIMEOUT:
                enroll_state["active"] = False
                print("Enrollment timeout: no name received", flush=True)

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

            # ── Pull latest recognition result (faces, names bundle) ──────
            rec_got_new = False
            try:
                while True:
                    last_rec_faces, last_rec_names = rec_out.get_nowait()
                    rec_got_new = True
            except Empty:
                pass
            if (rec_got_new and interaction == INTERACTION_RECOGNIZING
                    and last_rec_faces and now >= recog_stabilize_until):
                # Vote only on fully-framed faces — a cropped face can't be
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
                # Motion + person check is used by the state machine to
                # decide IDLE -> SEEKING. We only care about motion when at
                # level pose; during seek/recognize the pose change itself
                # causes motion.
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

                    # While in RECOGNIZING, track the sharpest complete face
                    # seen so we can enroll an unknown person even if the
                    # face has left the frame by the time the verdict fires.
                    # Skip during stabilization — those frames are blurry.
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
                    # During active enrollment, keep the sharpest fully-framed
                    # face so the saved image is the best one in the window.
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
                        # vote accumulation and gives a clean capture window.
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
                            and not actions.queue_contains(action_queue, "look_up")):
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
                    # action now (after the body has held the seek pose for the
                    # whole capture window). For greetings, the wiggle queued
                    # by the known-verdict branch lands AFTER this recovery, so
                    # the consumer order becomes recovery -> wiggle.
                    if (decide_now or face_lost) and camera_pose != "level":
                        recovery = ("stand_up" if camera_pose == "seeking_sit"
                                    else "look_level")
                        actions.drain_seek_actions(action_queue)
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
                                # Prefer the live frame for the zoom label, but
                                # fall back to the captured snapshot if the
                                # person has already moved out of frame.
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
        robot_thread.join(timeout=5.0)
        if rec_thread:
            rec_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
