#!/usr/bin/env python3
"""Record 5 seconds from the Go2 microphone and save to /tmp/mic_test.wav."""

import asyncio
import logging
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "go2_webrtc_connect"))

from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

logging.basicConfig(level=logging.FATAL)

SAMPLE_RATE  = 48000
CHANNELS     = 2
RECORD_SECS  = 5
OUT_PATH     = "/tmp/mic_test.wav"

frames_recorded = 0
target_frames   = RECORD_SECS * SAMPLE_RATE
wf              = None


async def recv_audio(frame):
    global frames_recorded
    if frames_recorded >= target_frames:
        return
    data = np.frombuffer(frame.to_ndarray(), dtype=np.int16)
    wf.writeframes(data.tobytes())
    frames_recorded += len(data) // CHANNELS
    elapsed = frames_recorded / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
    print(f"\r  {elapsed:.1f}s / {RECORD_SECS}s   RMS={rms:6.0f}   ", end="", flush=True)


async def main():
    global wf
    robot_ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.161"
    print(f"Connecting to {robot_ip}...")
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=robot_ip)
    await asyncio.wait_for(conn.connect(), timeout=30.0)
    print("Connected. Recording for 5 seconds — make some noise near the robot!\n")

    wf = wave.open(OUT_PATH, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)

    conn.audio.switchAudioChannel(True)
    conn.audio.add_track_callback(recv_audio)

    while frames_recorded < target_frames:
        await asyncio.sleep(0.1)

    wf.close()
    await conn.disconnect()
    print(f"\n\nSaved to {OUT_PATH}")
    print(f"Copy to laptop:  scp unitree@192.168.123.18:{OUT_PATH} ~/mic_test.wav")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        if wf:
            wf.close()
        print("\nStopped.")
