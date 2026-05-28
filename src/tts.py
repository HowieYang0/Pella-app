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
import base64
import contextlib
import hashlib
import io
import itertools
import json
import os
import time
from queue import Empty

from pydub import AudioSegment

from stt import TTS_MUTE_SEC, tts_mute_until


# ── Fast AudioHub upload ─────────────────────────────────────────────────────
# The vendor `audiohub.upload_audio_file` uses 4 KB base64 chunks with a
# 0.1 s sleep between every chunk; for a 132 KB WAV that's ~44 chunks of
# pure waiting (>4 s) on top of the per-chunk request round-trip. The
# protocol itself is chunk-size-flexible (`current_block_size` is
# per-chunk metadata; webrtc_audiohub.py even defines an unused
# CHUNK_SIZE = 61440 constant). Bigger chunks + shorter sleep should
# cut upload time ~5×; combined with the 22 kHz native sample rate
# (no upsample to 44.1 kHz) the typical novel-phrase upload drops
# from ~10 s to ~1 s.
FAST_UPLOAD_CHUNK_BYTES     = 4096  # vendor-compatible chunk size; bigger
                                    # chunks made the Go2 firmware silently
                                    # stall (see commit history).
FAST_UPLOAD_CHUNK_TIMEOUT   = 8.0   # max wall-clock for a single chunk's
                                    # response. On hang the asyncio.gather
                                    # propagates the TimeoutError so the
                                    # caller falls back to vendor uploader.
AUDIOHUB_UPLOAD_API_ID      = 2001  # AUDIO_API["UPLOAD_AUDIO_FILE"]
AUDIOHUB_REQUEST_TOPIC      = "rt/api/audiohub/request"

# Monotonic id generator for upload chunk requests. The vendor's default
# (timestamp_ms + random[0,1000]) collides with high probability when many
# requests fire in the same millisecond — birthday-paradox in a 1000-element
# space hits ~30% with 28 chunks. itertools.count guarantees uniqueness.
_chunk_request_id_counter = itertools.count(int(time.time() * 1000))


def _next_chunk_request_id() -> int:
    """Monotonic, collision-free id for a single upload chunk's request."""
    return next(_chunk_request_id_counter) % 2147483648


async def _fast_upload_audio_file(audiohub, wav_path: str):
    """Sub-second AudioHub upload by parallel-tracked send.

    Vendor `upload_audio_file` sends each chunk and awaits its ack
    sequentially, plus a 0.1 s sleep between every chunk — for a 132 KB
    WAV that's 44 chunks × ~150 ms = ~6.5 s of artificial pacing.

    A previous attempt at pure fire-and-forget (channel.send back-to-back,
    only await the LAST chunk's ack) shaved that to ~0.5 s but silently
    dropped middle chunks under load — produced "Hi, W" of a "Hi, William"
    upload that "succeeded" because the FINAL chunk's ack still arrived.

    This version pipelines: each chunk's publish_request_new is launched
    as a separate task via asyncio.gather. All chunks' channel.send fire
    back-to-back (no per-chunk RTT wait inserted by us), but every chunk
    is individually awaited so dropped chunks raise a TimeoutError and
    propagate. The asyncio scheduler keeps the datachannel busy while
    the Go2 firmware processes chunks in pipeline.

    Each chunk uses a monotonic, collision-free request id (the vendor's
    timestamp_ms + random[0,1000] generator collides ~30% of the time
    when 28 chunks fire in the same millisecond).
    """
    with open(wav_path, "rb") as f:
        audio_data = f.read()
    file_md5  = hashlib.md5(audio_data).hexdigest()
    b64       = base64.b64encode(audio_data).decode("utf-8")
    chunks    = [b64[i:i + FAST_UPLOAD_CHUNK_BYTES]
                 for i in range(0, len(b64), FAST_UPLOAD_CHUNK_BYTES)]
    file_name = os.path.splitext(os.path.basename(wav_path))[0]
    total_chunks = len(chunks)

    def _param(i, chunk):
        return {
            "file_name":           file_name,
            "file_type":           "wav",
            "file_size":           len(audio_data),
            "current_block_index": i,
            "total_block_number":  total_chunks,
            "block_content":       chunk,
            "current_block_size":  len(chunk),
            "file_md5":            file_md5,
            "create_time":         int(time.time() * 1000),
        }

    pubsub = audiohub.data_channel.pub_sub

    async def _send_chunk(i, chunk):
        return await asyncio.wait_for(
            pubsub.publish_request_new(
                AUDIOHUB_REQUEST_TOPIC,
                {
                    "id":        _next_chunk_request_id(),
                    "api_id":    AUDIOHUB_UPLOAD_API_ID,
                    "parameter": json.dumps(_param(i, chunk),
                                            ensure_ascii=True),
                },
            ),
            timeout=FAST_UPLOAD_CHUNK_TIMEOUT,
        )

    # All chunks concurrent — channel.send fires synchronously for each
    # task before any await, so the wire-time is back-to-back ~milliseconds,
    # then all chunks' acks resolve as the Go2 pipelines them.
    responses = await asyncio.gather(
        *(_send_chunk(i, c) for i, c in enumerate(chunks, 1))
    )
    return responses[-1]


# ── WAV generation ───────────────────────────────────────────────────────────

def _generate_wav(text: str, text_hash: str):
    """Convert text to a WAV via gTTS. Returns (path, duration_sec).

    Runs on a worker thread (called via run_in_executor) because gTTS makes
    blocking HTTP requests to Google's TTS service. The duration is needed
    by the caller to size the TTS-mute window so the USB mic doesn't
    capture (and re-transcribe) Pella's own speech.

    The WAV is exported at gTTS's native sample rate (22.05 kHz mono),
    not upsampled — 22.05 kHz is plenty for speech and produces a WAV
    half the size of a 44.1 kHz version, which roughly halves the
    audiohub upload time over the Go2's WebRTC datachannel. The WAV
    header carries the sample rate so the player picks it up.
    """
    from gtts import gTTS
    mp3_path = f"/tmp/pella_{text_hash}.mp3"
    out_path = f"/tmp/pella_{text_hash}.wav"
    gTTS(text=text, lang="en").save(mp3_path)
    sound = AudioSegment.from_mp3(mp3_path)
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


def invalidate_audiohub_uuids(cache: dict) -> int:
    """Clear the `uuid` field of every cache entry. Returns the number
    invalidated.

    Called by pella_main on each new WebRTC connection: the previous
    AudioHub session's UUIDs are not valid against the new AudioHub, so
    a subsequent prepare/speak must re-upload the .wav. The .wav `path`
    on /tmp, the `duration`, and the per-entry `lock` are preserved —
    so re-upload is cheap and, crucially, requires no gTTS HTTP call
    (no WAN dependency to recover from a reconnect).
    """
    n = 0
    for entry in cache.values():
        if entry.get("uuid") is not None:
            entry["uuid"] = None
            n += 1
    if n:
        print(f"TTS: invalidated {n} cached AudioHub uuid(s) for new session",
              flush=True)
    return n


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
            # Regenerate if the path was never set, OR if it was set in a
            # previous session and the /tmp file has since been cleaned
            # away by tmpwatch/systemd-tmpfiles. Treating a missing path
            # as "must regenerate" keeps the cache honest across long
            # uptimes.
            if entry["path"] is None or not os.path.exists(entry["path"]):
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
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        await _fast_upload_audio_file(audiohub, entry["path"])
                    upload_method = "fast"
                except Exception as fast_err:
                    # Fall back to the vendor's slower-but-known-good
                    # uploader if our larger-chunk path errors out (e.g.
                    # Go2 firmware rejects 16 KB chunks on some version).
                    print(f"TTS {tag} fast_upload failed "
                          f"({type(fast_err).__name__}: {fast_err}); "
                          f"falling back to vendor uploader", flush=True)
                    with contextlib.redirect_stdout(io.StringIO()):
                        await audiohub.upload_audio_file(entry["path"])
                    upload_method = "vendor"
                print(f"TTS {tag} upload[{upload_method}]: "
                      f"{time.monotonic()-t0:.2f}s "
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
            # Regenerate if the path was never set, OR if it was set in a
            # previous session and the /tmp file has since been cleaned
            # away by tmpwatch/systemd-tmpfiles. Treating a missing path
            # as "must regenerate" keeps the cache honest across long
            # uptimes.
            if entry["path"] is None or not os.path.exists(entry["path"]):
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
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            await _fast_upload_audio_file(audiohub,
                                                          entry["path"])
                        upload_method = "fast"
                    except Exception as fast_err:
                        print(f"TTS {tag} fast_upload failed "
                          f"({type(fast_err).__name__}: {fast_err}); "
                              f"falling back to vendor uploader",
                              flush=True)
                        with contextlib.redirect_stdout(io.StringIO()):
                            await audiohub.upload_audio_file(entry["path"])
                        upload_method = "vendor"
                    print(f"TTS {tag} upload[{upload_method}]: "
                          f"{time.monotonic()-t0:.2f}s "
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


