#!/usr/bin/env python3
"""Speak text through the Go2's built-in speaker via WebRTC AudioHub."""

import asyncio
import hashlib
import json
import os
import sys
import tempfile

import dotenv

dotenv.load_dotenv()

# go2_webrtc_connect lives as a sibling directory
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "go2_webrtc_connect")
)

from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.webrtc_audiohub import WebRTCAudioHub

import re
import unicodedata

_PUNCT_TO_KEEP = ".,?!"


def _speakable_number(s: str) -> str:
    s = s.strip()
    if s.startswith("-"):
        s = "minus " + s[1:]
    return s.replace(".", " point ")


def _rewrite_gps(text: str) -> str:
    def _repl(m):
        return f"latitude {_speakable_number(m.group(1))}, longitude {_speakable_number(m.group(2))}"
    return re.sub(r"\b(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)\b", _repl, text)


def sanitize_for_tts(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _rewrite_gps(text)
    text = re.sub(r"(?m)^\s*-\s+", "", text)
    text = re.sub(r"\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}|<[^<>]*>", " ", text)
    text = re.sub(r"(?<=\d)\.(?=\d)", "<DECIMAL>", text)
    text = re.sub(r"(?<!\w)-(?=\d)", "<NEG>", text)
    text = re.sub(
        r"\b(?:[A-Za-z]\.){2,}",
        lambda m: m.group(0).replace(".", "<ABBR_DOT>"),
        text,
    )
    for k, v in {"&": " and ", " e.g.": " for example", " i.e.": " that is", " vs.": " versus"}.items():
        text = re.sub(re.escape(k), v, text, flags=re.IGNORECASE)
    text = re.sub(r"\bhttps?://\S+|\bwww\.\S+|\S+@\S+", " ", text)
    text = "".join(
        ch if (ch.isalnum() or ch.isspace() or ch in _PUNCT_TO_KEEP or ch in {"<", ">", "_"}) else " "
        for ch in text
    )
    text = re.sub(rf"\s*([{re.escape(_PUNCT_TO_KEEP)}])\s*", r"\1 ", text)
    text = re.sub(r"\b\d{6,}\b", " degrees", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[.]{2,}", ".", text)
    text = re.sub(r"[!?]{2,}", lambda m: m.group(0)[0], text)
    return text.replace("<DECIMAL>", ".").replace("<NEG>", "-").replace("<ABBR_DOT>", ".")


def _tts_to_mp3(text: str, mp3_path: str) -> None:
    from gtts import gTTS
    gTTS(text=text, lang="en").save(mp3_path)


async def speak_webrtc(text: str, robot_ip: str) -> None:
    """Generate TTS and play it on the Go2's speaker via WebRTC.

    Uploads the clip to the robot's AudioHub the first time; subsequent calls
    with the same text reuse the cached clip (identified by an MD5-based name).
    Requires Go2 Pro or Edu (Air does not have a speaker).
    """
    text = sanitize_for_tts(text)
    if not text.strip():
        return

    clip_name = "pella_" + hashlib.md5(text.encode()).hexdigest()[:8]

    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=robot_ip)
    await conn.connect()

    try:
        hub = WebRTCAudioHub(conn)

        audio_list = await _get_audio_list(hub)
        existing = _find_clip(audio_list, clip_name)

        if not existing:
            mp3_path = os.path.join(tempfile.gettempdir(), clip_name + ".mp3")
            try:
                _tts_to_mp3(text, mp3_path)
                await hub.upload_audio_file(mp3_path)
            finally:
                if os.path.exists(mp3_path):
                    os.unlink(mp3_path)

            audio_list = await _get_audio_list(hub)
            existing = _find_clip(audio_list, clip_name)

        if existing:
            await hub.play_by_uuid(existing["UNIQUE_ID"])
        else:
            print(f"say_webrtc: could not find uploaded clip '{clip_name}' on robot")
    finally:
        await conn.disconnect()


async def _get_audio_list(hub: WebRTCAudioHub) -> list:
    response = await hub.get_audio_list()
    if response and isinstance(response, dict):
        data_str = response.get("data", {}).get("data", "{}")
        return json.loads(data_str).get("audio_list", [])
    return []


def _find_clip(audio_list: list, clip_name: str) -> dict | None:
    return next((a for a in audio_list if a["CUSTOM_NAME"] == clip_name), None)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 say_webrtc.py <text> <robot_ip>")
        sys.exit(1)
    asyncio.run(speak_webrtc(sys.argv[1], sys.argv[2]))
