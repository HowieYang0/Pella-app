#!/usr/bin/env python3
"""Pella's mouth: text-to-speech playback through the Go2 audio hub.

Texts arrive on `say_queue` (Queue[str]) from pella_main. For each one:
  1. Generate a WAV via gTTS on a worker thread (cached per text hash).
  2. Upload it to the Go2's audio hub (also cached — the robot keeps the
     uploaded file by UUID, so repeat utterances skip the upload).
  3. Play via `play_by_uuid` on the audio hub.
  4. Set `stt.tts_mute_until[0]` so the USB-mic capture path discards
     audio while Pella is speaking, preventing self-hear.

The say_queue consumer runs as a coroutine on the same asyncio loop that
owns the WebRTC connection (started by pella_main when audiohub is ready).
"""

import asyncio
import contextlib
import hashlib
import io
import json
import os
import time
from queue import Empty

from pydub import AudioSegment

from stt import TTS_MUTE_SEC, tts_mute_until


# ── WAV generation ───────────────────────────────────────────────────────────

def _generate_wav(text: str, text_hash: str) -> str:
    """Convert text to a 44.1 kHz WAV via gTTS. Returns the file path.

    Runs on a worker thread (called via run_in_executor) because gTTS makes
    blocking HTTP requests to Google's TTS service.
    """
    from gtts import gTTS
    mp3_path = f"/tmp/pella_{text_hash}.mp3"
    out_path = f"/tmp/pella_{text_hash}.wav"
    gTTS(text=text, lang="en").save(mp3_path)
    sound = AudioSegment.from_mp3(mp3_path).set_frame_rate(44100)
    sound.export(out_path, format="wav")
    os.unlink(mp3_path)
    return out_path


# ── Playback ─────────────────────────────────────────────────────────────────

async def speak(text: str, audiohub, cache: dict):
    """Generate (if needed), upload (if needed), and play `text`.

    `cache` is a dict mapping text_hash -> {"path": str, "uuid": str} so each
    distinct utterance only pays the gTTS + upload cost once per process run.
    Caller owns the dict.

    Sets `tts_mute_until[0]` to silence the USB-mic capture path for
    TTS_MUTE_SEC seconds starting now, so the robot's speaker output doesn't
    get re-transcribed.
    """
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
            tts_mute_until[0] = mute_t   # USB-mic path reads this to self-mute
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


# ── Say queue consumer ───────────────────────────────────────────────────────

async def run_say_consumer(say_queue, audiohub, stop_event, cache=None):
    """Drain say_queue and play each utterance.

    Each pop fires speak() as a background task; multiple consecutive say
    requests can therefore overlap in upload/playback phases but Audio Hub
    serialises actual speaker output. (The say_queue itself is bounded to
    maxsize=1 by pella_main, so producers naturally backpressure.)
    """
    if cache is None:
        cache = {}
    while not stop_event.is_set():
        try:
            text = say_queue.get_nowait()
            asyncio.ensure_future(speak(text, audiohub, cache))
        except Empty:
            pass
        await asyncio.sleep(0.5)
