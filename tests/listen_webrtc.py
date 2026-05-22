#!/usr/bin/env python3
"""Receive audio from Go2's microphone via WebRTC and transcribe with DashScope ASR."""

import asyncio
import os
import sys
import tempfile
import wave
from collections import deque
from http import HTTPStatus

import dotenv
import numpy as np
import soxr

dotenv.load_dotenv()

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "go2_webrtc_connect")
)

from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

import dashscope
from dashscope.audio.asr import Recognition

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope.base_http_api_url = os.getenv("DASHSCOPE_BASE_URL")

ASR_MODEL = os.getenv("ALI_ASR_MODEL", "paraformer-realtime-v2")

# Robot audio stream properties (fixed by WebRTC)
ROBOT_SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2  # bytes — 16-bit PCM

# DashScope ASR expects 16 kHz mono
ASR_SAMPLE_RATE = 16000

# Voice Activity Detection tuning
RMS_THRESHOLD = 300       # RMS below this is treated as silence
MIN_SPEECH_SEC = 0.4      # shorter clips are ignored (noise / clicks)
MAX_SPEECH_SEC = 12.0     # hard cut-off per utterance
SILENCE_TAIL_SEC = 0.8    # silence after speech to finalise utterance


def _transcribe_samples(samples_48k: np.ndarray) -> str:
    """Resample to 16 kHz, write WAV, call DashScope ASR, return transcript."""
    # Resample 48 kHz → 16 kHz
    audio_f = samples_48k.astype(np.float32) / 32768.0
    audio_rs = soxr.resample(audio_f, ROBOT_SAMPLE_RATE, ASR_SAMPLE_RATE, quality="HQ")
    samples_16k = np.clip(audio_rs * 32768.0, -32768, 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(ASR_SAMPLE_RATE)
            wf.writeframes(samples_16k.tobytes())

        recognizer = Recognition(
            model=ASR_MODEL,
            callback=None,
            format="wav",
            sample_rate=ASR_SAMPLE_RATE,
            language_hints=["zh", "en"],
        )
        result = recognizer.call(file=wav_path)

        if result.status_code != HTTPStatus.OK:
            print(f"[ASR error] {result.status_code}: {result.message}")
            return ""

        sentences = result.output.get("sentence", [])
        return " ".join(s["text"].strip() for s in sentences if s.get("text")).strip()

    finally:
        os.unlink(wav_path)


class AudioListener:
    def __init__(self, robot_ip: str, on_transcript=None):
        self.robot_ip = robot_ip
        # on_transcript(text) is called for each recognised utterance.
        # Default: print to stdout.
        self.on_transcript = on_transcript or (lambda t: print(f"[Heard] {t}"))

        self._buf: deque = deque()
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0

        self._silence_cutoff = int(SILENCE_TAIL_SEC * ROBOT_SAMPLE_RATE)
        self._max_samples = int(MAX_SPEECH_SEC * ROBOT_SAMPLE_RATE)
        self._min_samples = int(MIN_SPEECH_SEC * ROBOT_SAMPLE_RATE)

    async def _frame_callback(self, frame):
        # frame.to_ndarray() → interleaved int16 stereo [L, R, L, R, …]
        raw = np.frombuffer(frame.to_ndarray(), dtype=np.int16)
        mono = raw.reshape(-1, CHANNELS).mean(axis=1).astype(np.int16)

        rms = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2)))
        is_voice = rms > RMS_THRESHOLD

        if is_voice:
            self._buf.extend(mono.tolist())
            self._speech_samples += len(mono)
            self._silence_samples = 0
            self._in_speech = True

        elif self._in_speech:
            self._buf.extend(mono.tolist())
            self._silence_samples += len(mono)

            end_of_utterance = (
                self._silence_samples >= self._silence_cutoff
                or self._speech_samples >= self._max_samples
            )
            if end_of_utterance:
                await self._flush()

    async def _flush(self):
        samples = np.array(list(self._buf), dtype=np.int16)
        self._reset()

        if len(samples) < self._min_samples:
            return

        text = await asyncio.to_thread(_transcribe_samples, samples)
        if text:
            self.on_transcript(text)

    def _reset(self):
        self._buf.clear()
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0

    async def run(self):
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.robot_ip)
        await conn.connect()
        try:
            conn.audio.switchAudioChannel(True)
            conn.audio.add_track_callback(self._frame_callback)
            print(f"Listening on Go2 mic at {self.robot_ip} — Ctrl+C to stop")
            while True:
                await asyncio.sleep(1)
        finally:
            await conn.disconnect()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 listen_webrtc.py <robot_ip>")
        sys.exit(1)
    listener = AudioListener(sys.argv[1])
    try:
        asyncio.run(listener.run())
    except KeyboardInterrupt:
        print("\nStopped.")
