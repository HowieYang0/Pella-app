import asyncio
from asyncio import Queue
from collections import deque
import difflib
import re
import unicodedata

from google.cloud import texttospeech
import numpy as np
import sounddevice as sd
import base64
import soxr
import os, dotenv, time, sys

from audio.audio_context import get_audio_config

dotenv.load_dotenv()

PUNCT_TO_KEEP = ".,?!"


def speakable_number(num_str: str) -> str:
    s = num_str.strip()
    if s.startswith('-'):
        s = 'minus ' + s[1:]
    return s.replace('.', ' point ')


def rewrite_gps_for_tts(text: str) -> str:
    def repl(m):
        lat = speakable_number(m.group(1))
        lon = speakable_number(m.group(2))
        return f"latitude {lat}, longitude {lon}"
    return re.sub(
        r'\b(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)\b',
        repl,
        text
    )


def sanitize_for_tts(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = rewrite_gps_for_tts(text)
    text = re.sub(r'(?m)^\s*-\s+', '', text)
    text = re.sub(r"\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}|<[^<>]*>", " ", text)
    text = re.sub(r'(?<=\d)\.(?=\d)', '<DECIMAL>', text)
    text = re.sub(r'(?<!\w)-(?=\d)', '<NEG>', text)
    text = re.sub(
        r'\b(?:[A-Za-z]\.){2,}',
        lambda m: m.group(0).replace('.', '<ABBR_DOT>'),
        text
    )
    replacements = {
        "&": " and ",
        " e.g.": " for example",
        " i.e.": " that is",
        " vs.": " versus",
    }
    for k, v in replacements.items():
        text = re.sub(re.escape(k), v, text, flags=re.IGNORECASE)
    text = re.sub(r"\bhttps?://\S+|\bwww\.\S+|\S+@\S+", " ", text)
    allowed_placeholders = {'<', '>', '_'}
    text = "".join(
        ch if (ch.isalnum() or ch.isspace() or ch in PUNCT_TO_KEEP or ch in allowed_placeholders)
        else " "
        for ch in text
    )
    text = re.sub(rf"\s*([{re.escape(PUNCT_TO_KEEP)}])\s*", r"\1 ", text)
    text = re.sub(r'\b\d{6,}\b', ' degrees', text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[.]{2,}", ".", text)
    text = re.sub(r"[!?]{2,}", lambda m: m.group(0)[0], text)
    text = text.replace('<DECIMAL>', '.')
    text = text.replace('<NEG>', '-')
    text = text.replace('<ABBR_DOT>', '.')
    return text


def clean_string(text):
    text_no_newlines = text.replace('\n', '').replace('\r', '')
    return re.sub(r'\b\d{4,}\b', '', text_no_newlines)


def get_sd_output_device_and_rate(device_name: str | None = None, channels: int = 1, dtype="int16"):
    PREFERRED_RATES = [24000, 22050, 16000, 32000, 48000]
    if device_name is None:
        dev = sd.default.device[1]
        if dev is None or dev < 0:
            dev = sd.query_devices(kind="output")["index"]
    else:
        name_lc = device_name.lower()
        dev = None
        for i, d in enumerate(sd.query_devices()):
            if d["max_output_channels"] > 0 and name_lc in d["name"].lower():
                dev = i
                break
        if dev is None:
            raise RuntimeError(f"No output device matching '{device_name}'")
    for sr in PREFERRED_RATES:
        try:
            sd.check_output_settings(device=dev, samplerate=sr, channels=channels, dtype=dtype)
            return dev, sr
        except Exception:
            continue
    raise RuntimeError(f"No supported samplerate found for device={dev}")


if sys.platform == "linux" or sys.platform == "linux2":
    output_channel, SAMPLE_RATE = get_sd_output_device_and_rate('USB')
elif sys.platform == "darwin":
    output_channel, SAMPLE_RATE = get_sd_output_device_and_rate()
else:
    output_channel, SAMPLE_RATE = 1, 48000

# Google Cloud TTS (English)
client = texttospeech.TextToSpeechClient()
voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Neural2-F")
audio_cfg = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    sample_rate_hertz=SAMPLE_RATE,
)


def convert_to_voice(text: str):
    convert_to_voice_ch(text)


def convert_to_voice_en(text: str):
    resp = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=voice,
        audio_config=audio_cfg,
    )
    try:
        samples = np.frombuffer(resp.audio_content, dtype=np.int16)
        sd.play(samples, samplerate=SAMPLE_RATE, blocking=True)
        sd.wait()
    except sd.PortAudioError:
        print("convert_to_voice_en - Error in device setup")
        OUT_ID = sd.default.device[1]
        samples = np.frombuffer(resp.audio_content, dtype=np.int16)
        sd.play(samples, samplerate=SAMPLE_RATE, blocking=False, device=OUT_ID)
        sd.wait()


# Alibaba DashScope TTS (Chinese)
import dashscope
from requests.exceptions import ConnectionError, ConnectTimeout
from dashscope.common.error import AuthenticationError, DashScopeException
import queue
import threading

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope.base_http_api_url = os.getenv("DASHSCOPE_BASE_URL")
MODEL = os.getenv("ALI_TTS_MODEL")
VOICE = "Cherry"


def resample_int16(audio_i16: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    audio_f = audio_i16.astype(np.float32) / 32768.0
    audio_f_rs = soxr.resample(audio_f, src_sr, dst_sr, quality="HQ")
    return np.clip(audio_f_rs * 32768.0, -32768, 32767).astype(np.int16)


def convert_to_voice_ch(text: str):
    if not text or not text.strip():
        return

    audio_config = get_audio_config()
    output_channel, SAMPLE_RATE = audio_config.output_device_index, audio_config.tts_rate

    CHANNELS = 1
    SRC_SR = 24000
    DST_SR = SAMPLE_RATE
    PREBUFFER_SECONDS = 2
    PREBUFFER_SAMPLES = int(DST_SR * PREBUFFER_SECONDS)

    q: "queue.Queue[np.ndarray | object]" = queue.Queue(maxsize=200)
    END = object()

    def producer():
        try:
            response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
                model=MODEL,
                text=text,
                voice=VOICE,
                sample_rate=SRC_SR,
                language_type="Chinese",
                stream=True,
            )
            for chunk in response:
                if chunk.output is None:
                    continue
                audio = chunk.output.audio
                if not audio or audio.get("data") is None:
                    if getattr(chunk.output, "finish_reason", None) == "stop":
                        break
                    continue
                pcm_bytes = base64.b64decode(audio["data"])
                audio_np = np.frombuffer(pcm_bytes, dtype=np.int16)
                audio_rs = resample_int16(audio_np, SRC_SR, DST_SR)
                q.put(audio_rs)
                if getattr(chunk.output, "finish_reason", None) == "stop":
                    break
        except ConnectTimeout as e:
            print(e.message if hasattr(e, "message") else e)
        except AuthenticationError as e:
            print(e.message if hasattr(e, "message") else "Dashscope authentication error")
        except DashScopeException as e:
            print(e.message if hasattr(e, "message") else f"Dashscope exception: {e}")
        except Exception as e:
            print(f"Exception: {type(e).__name__}, belongs to {type(e).__module__}")
        finally:
            q.put(END)

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    buffered = np.empty((0,), dtype=np.int16)
    while buffered.size < PREBUFFER_SAMPLES:
        item = q.get()
        if item is END:
            break
        buffered = np.concatenate([buffered, item])

    if buffered.size == 0:
        return

    remaining = buffered.copy()

    def callback(outdata, frames, time_info, status):
        nonlocal remaining
        if status:
            print("sounddevice status:", status, "block size:", frames)
        while remaining.size < frames:
            item = q.get()
            if item is END:
                out = np.zeros(frames, dtype=np.int16)
                n = min(remaining.size, frames)
                if n > 0:
                    out[:n] = remaining[:n]
                    remaining = remaining[n:]
                outdata[:] = np.repeat(out.reshape(-1, 1), CHANNELS, axis=1)
                raise sd.CallbackStop()
            remaining = np.concatenate([remaining, item])
        out = remaining[:frames]
        remaining = remaining[frames:]
        if CHANNELS == 1:
            outdata[:] = out.reshape(-1, 1)
        else:
            outdata[:] = np.repeat(out.reshape(-1, 1), CHANNELS, axis=1)

    with sd.OutputStream(
        samplerate=DST_SR,
        channels=CHANNELS,
        dtype="int16",
        callback=callback,
        device=audio_config.output_device_index,
        blocksize=0,
    ):
        producer_thread.join()
        while remaining.size > 0 or producer_thread.is_alive():
            time.sleep(0.05)
        time.sleep(0.2)


class EchoGuard:
    def __init__(self, window_sec: float = 8.0, max_chars: int = 200, threshold: float = 0.88):
        self.window_sec = window_sec
        self.max_chars = max_chars
        self.threshold = threshold
        self._history = deque()
        self._lock = asyncio.Lock()

    @staticmethod
    def _normalize(text: str) -> str:
        text = (text or "").strip().lower()
        return " ".join(text.split())

    def _truncate(self, text: str) -> str:
        return text[-self.max_chars:] if len(text) > self.max_chars else text

    async def record_tts(self, text: str):
        self._last_spoken_text = self._truncate(self._normalize(text))
        self._last_spoken_time = time.monotonic()

    async def is_echo(self, stt_text: str) -> bool:
        norm_stt = self._truncate(self._normalize(stt_text))
        if not norm_stt:
            return False
        if not hasattr(self, '_last_spoken_time') or time.monotonic() - self._last_spoken_time > 30:
            return False
        ratio = difflib.SequenceMatcher(None, norm_stt, self._last_spoken_text).ratio()
        if ratio >= self.threshold:
            print(f"[EchoGuard] similarity={ratio:.3f} → echo ignored")
            return True
        print(f"[EchoGuard] similarity={ratio:.3f} → not an echo")
        return False


async def speak_enqueue(out_q: asyncio.Queue[str], echo_guard: EchoGuard, text: str):
    text = (text or "").strip()
    if not text:
        return
    await echo_guard.record_tts(text)
    await out_q.put(text)


async def tts_sender(out_q: Queue[str], echo_guard: EchoGuard, speaking_event: asyncio.Event):
    print("TTS sender started, waiting for text to speak...")
    try:
        while True:
            text = await out_q.get()
            text = (text or "").strip()
            if not text:
                continue
            print("\n\nTTS: ", text)
            speaking_event.set()
            speech = sanitize_for_tts(text)
            await asyncio.to_thread(convert_to_voice, text=speech)
            await asyncio.sleep(3.0)
            speaking_event.clear()
    except asyncio.exceptions.CancelledError:
        print("TTS sender task cancelled, exiting.")
    except Exception as e:
        print("TTS sender task failed, exiting.")
