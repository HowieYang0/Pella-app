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

def _generate_wav(text: str, text_hash: str):
    """Convert text to a 44.1 kHz WAV via gTTS. Returns (path, duration_sec).

    Runs on a worker thread (called via run_in_executor) because gTTS makes
    blocking HTTP requests to Google's TTS service. The duration is needed
    by the caller to size the TTS-mute window so the USB mic doesn't
    capture (and re-transcribe) Pella's own speech.
    """
    from gtts import gTTS
    mp3_path = f"/tmp/pella_{text_hash}.mp3"
    out_path = f"/tmp/pella_{text_hash}.wav"
    gTTS(text=text, lang="en").save(mp3_path)
    sound = AudioSegment.from_mp3(mp3_path).set_frame_rate(44100)
    sound.export(out_path, format="wav")
    os.unlink(mp3_path)
    return out_path, len(sound) / 1000.0


def _wav_duration(path: str) -> float:
    """Read duration in seconds from a WAV file. 0.0 on failure.

    Used when a cache entry already has a path (e.g. recovered from a
    previous run) but no recorded duration yet.
    """
    import wave
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


# How much extra to add on top of the WAV duration when sizing the mute.
# Covers: play_by_uuid request acknowledge (~0.2 s) + speaker buffering
# (~0.3 s) + room reverb tail (~0.5 s). Generous on purpose — mute lifting
# slightly late costs a fraction of a second of listening; lifting too early
# costs a self-hear and a bogus transcript.
MUTE_BUFFER_SEC = 1.0


def _get_or_create_entry(cache: dict, text_hash: str) -> dict:
    """Return the cache entry for `text_hash`, creating it (with a fresh
    asyncio.Lock) if it doesn't exist yet.

    The lock serialises the gen_wav + audiohub-upload phases across
    concurrent callers (speak() vs prepare(), or two prepare()s) racing
    on the same text. Without it, both callers run `gTTS().save(mp3)`
    against the same /tmp path, one os.unlinks while the other is still
    reading, and ffmpeg fails to decode.

    Lock creation has to happen inside a running asyncio loop (Python 3.8+
    semantics), so we only create it on first access from within an
    async function — never at module import time.
    """
    entry = cache.get(text_hash)
    if entry is None:
        entry = {
            "path": None, "uuid": None, "duration": 0.0,
            "lock": asyncio.Lock(),
        }
        cache[text_hash] = entry
    return entry


# ── Playback ─────────────────────────────────────────────────────────────────

async def speak(text: str, audiohub, cache: dict):
    """Generate (if needed), upload (if needed), and play `text`.

    `cache` is a dict mapping text_hash -> {"path": str, "uuid": str} so each
    distinct utterance only pays the gTTS + upload cost once per process run.
    Caller owns the dict.

    Sets `tts_mute_until[0]` to silence the USB-mic capture path for
    TTS_MUTE_SEC seconds starting now, so the robot's speaker output doesn't
    get re-transcribed.

    Each phase is timed and logged with a tag identifying the utterance, so
    we can see exactly which step (gen / upload / list / play) consumes the
    wall time when there's an end-to-end delay.
    """
    try:
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        file_name = f"pella_{text_hash}"
        entry = _get_or_create_entry(cache, text_hash)
        preview = (text[:60] + "…") if len(text) > 60 else text
        tag = f"[{text_hash}]"

        # Serialise the gen_wav + upload phases against concurrent callers
        # for the same text_hash. Without this, prepare() and speak()
        # racing on "Nice to meet you, William!" would both call gen_wav,
        # collide on /tmp/pella_<hash>.mp3 (the first os.unlinks it while
        # the second is still reading), and ffmpeg fails decoding.
        async with entry["lock"]:
            if entry["path"] is None:
                t0 = time.monotonic()
                entry["path"], entry["duration"] = (
                    await asyncio.get_event_loop().run_in_executor(
                        None, _generate_wav, text, text_hash
                    )
                )
                print(f"TTS {tag} gen_wav: {time.monotonic()-t0:.2f}s "
                      f"({entry['duration']:.2f}s audio) "
                      f"{repr(preview)}", flush=True)
            elif not entry.get("duration"):
                # Recovered path with no duration recorded — read from the WAV.
                entry["duration"] = _wav_duration(entry["path"])

            if entry["uuid"] is None:
                t0 = time.monotonic()
                resp = await audiohub.get_audio_list()
                print(f"TTS {tag} get_list_before: {time.monotonic()-t0:.2f}s",
                      flush=True)
                audio_list = json.loads(
                    (resp.get("data") or {}).get("data", "{}")
                ).get("audio_list", [])
                for item in audio_list:
                    if item.get("CUSTOM_NAME") == file_name:
                        t0 = time.monotonic()
                        await audiohub.delete_record(item["UNIQUE_ID"])
                        print(f"TTS {tag} delete_old: "
                              f"{time.monotonic()-t0:.2f}s", flush=True)
                        break
                t0 = time.monotonic()
                with contextlib.redirect_stdout(io.StringIO()):
                    await audiohub.upload_audio_file(entry["path"])
                print(f"TTS {tag} upload: {time.monotonic()-t0:.2f}s "
                      f"({os.path.getsize(entry['path'])} bytes)", flush=True)
                t0 = time.monotonic()
                resp = await audiohub.get_audio_list()
                print(f"TTS {tag} get_list_after: {time.monotonic()-t0:.2f}s",
                      flush=True)
                audio_list = json.loads(
                    (resp.get("data") or {}).get("data", "{}")
                ).get("audio_list", [])
                for item in audio_list:
                    if item.get("CUSTOM_NAME") == file_name:
                        entry["uuid"] = item["UNIQUE_ID"]
                        break

        if entry["uuid"]:
            # Mute the USB mic for the actual audio length + a safety buffer.
            # Fixed TTS_MUTE_SEC was 4 s, which under-covered long utterances
            # like "Hello, I am Pella. What is your name?" (~3.5 s audio +
            # ~0.5 s playback latency), causing the tail to leak into the
            # mic and be re-transcribed as "Pella".
            mute_dur = max(
                entry.get("duration", 0.0) + MUTE_BUFFER_SEC,
                TTS_MUTE_SEC,  # never shorter than the legacy floor
            )
            mute_t = time.monotonic() + mute_dur
            tts_mute_until[0] = mute_t   # USB-mic path reads this to self-mute
            print(f"TTS {tag} playing {repr(preview)} "
                  f"(mute until +{mute_dur:.1f}s)", flush=True)
            t0 = time.monotonic()
            await audiohub.play_by_uuid(entry["uuid"])
            print(f"TTS {tag} play_by_uuid_call: "
                  f"{time.monotonic()-t0:.2f}s (done)", flush=True)
        else:
            print(f"TTS {tag} NO UUID for {repr(preview)} — playback skipped",
                  flush=True)
    except Exception as e:
        print(f"TTS error: {e}", flush=True)


# ── Cache pre-warm ───────────────────────────────────────────────────────────

async def prepare(text: str, audiohub, cache: dict):
    """Generate WAV + upload to audiohub but DO NOT play.

    Pre-warms the cache so a subsequent speak(text, …) hits the cached UUID
    and the play_by_uuid request fires immediately. Used at startup for
    static prompts and proactively when a task knows it will say something
    soon (e.g. as soon as a name is parsed from STT).

    Same per-phase timing prints as speak(), tagged with the text_hash so
    a prep followed by a speak can be matched up in the journal.
    """
    try:
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        file_name = f"pella_{text_hash}"
        entry = _get_or_create_entry(cache, text_hash)
        preview = (text[:50] + "…") if len(text) > 50 else text
        tag = f"[{text_hash} prep]"

        # Serialise against speak() (or another prepare()) racing on the
        # same text_hash — see comment in speak().
        async with entry["lock"]:
            if entry["path"] is None:
                t0 = time.monotonic()
                entry["path"], entry["duration"] = (
                    await asyncio.get_event_loop().run_in_executor(
                        None, _generate_wav, text, text_hash
                    )
                )
                print(f"TTS {tag} gen_wav: {time.monotonic()-t0:.2f}s "
                      f"({entry['duration']:.2f}s audio) "
                      f"{repr(preview)}", flush=True)
            elif not entry.get("duration"):
                entry["duration"] = _wav_duration(entry["path"])

            if entry["uuid"] is None:
                t0 = time.monotonic()
                resp = await audiohub.get_audio_list()
                print(f"TTS {tag} get_list_before: "
                      f"{time.monotonic()-t0:.2f}s", flush=True)
                audio_list = json.loads(
                    (resp.get("data") or {}).get("data", "{}")
                ).get("audio_list", [])
                for item in audio_list:
                    if item.get("CUSTOM_NAME") == file_name:
                        entry["uuid"] = item["UNIQUE_ID"]
                        break
                if entry["uuid"] is None:
                    t0 = time.monotonic()
                    with contextlib.redirect_stdout(io.StringIO()):
                        await audiohub.upload_audio_file(entry["path"])
                    print(f"TTS {tag} upload: {time.monotonic()-t0:.2f}s "
                          f"({os.path.getsize(entry['path'])} bytes)",
                          flush=True)
                    t0 = time.monotonic()
                    resp = await audiohub.get_audio_list()
                    print(f"TTS {tag} get_list_after: "
                          f"{time.monotonic()-t0:.2f}s", flush=True)
                    audio_list = json.loads(
                        (resp.get("data") or {}).get("data", "{}")
                    ).get("audio_list", [])
                    for item in audio_list:
                        if item.get("CUSTOM_NAME") == file_name:
                            entry["uuid"] = item["UNIQUE_ID"]
                            break
        return entry["uuid"] is not None
    except Exception as e:
        print(f"TTS prepare error for {repr(text[:60])}: {e}", flush=True)
        return False


async def run_warmup(phrases, audiohub, cache: dict):
    """Sequentially pre-cache a list of phrases. Logs each one."""
    if not phrases:
        return
    print(f"TTS: warming cache for {len(phrases)} phrase(s)…", flush=True)
    for text in phrases:
        ok = await prepare(text, audiohub, cache)
        preview = (text[:50] + "…") if len(text) > 50 else text
        print(f"TTS: {'cached' if ok else 'FAILED'} {repr(preview)}",
              flush=True)
    print("TTS: warmup done", flush=True)


# ── Queue consumers ──────────────────────────────────────────────────────────

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


async def run_prep_consumer(prep_queue, audiohub, stop_event, cache=None):
    """Drain prep_queue and pre-cache each phrase without playing.

    Lets a task hand off "I'm about to say this — please start uploading
    now" requests that overlap with the rest of its work. By the time the
    task actually puts the same text on say_queue, speak() hits a warm
    cache and play_by_uuid fires immediately.

    Used today for dynamic, name-dependent phrases the startup warmup
    can't anticipate — most importantly "Nice to meet you, <New Name>!"
    queued the instant the new person's name is parsed from STT.
    """
    if cache is None:
        cache = {}
    while not stop_event.is_set():
        try:
            text = prep_queue.get_nowait()
            asyncio.ensure_future(prepare(text, audiohub, cache))
        except Empty:
            pass
        await asyncio.sleep(0.2)


