#!/usr/bin/env python3
"""Subscribe to rt/gptflowfeedback and rt/api/assistant_recorder to see what the
robot's built-in voice assistant publishes when you speak to it."""

import asyncio
import json
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "go2_webrtc_connect"))

from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC

logging.basicConfig(level=logging.FATAL)


async def main():
    robot_ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.161"
    print(f"Connecting to {robot_ip}...")
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=robot_ip)
    await asyncio.wait_for(conn.connect(), timeout=30.0)
    print("Connected. Subscribing to voice/AI topics...\n")

    def on_gpt_feedback(msg):
        print(f"[GPT_FEEDBACK] {json.dumps(msg, indent=2)}")

    def on_assistant(msg):
        print(f"[ASSISTANT_RECORDER] {json.dumps(msg, indent=2)}")

    # Subscribe to feedback topics
    conn.datachannel.pub_sub.subscribe(RTC_TOPIC["GPT_FEEDBACK"], on_gpt_feedback)

    # Also try to subscribe to sport mode state to see if voice commands come through there
    def on_service_state(msg):
        print(f"[SERVICE_STATE] {json.dumps(msg, indent=2)}")

    conn.datachannel.pub_sub.subscribe(RTC_TOPIC["SERVICE_STATE"], on_service_state)

    print("Listening. Try speaking a voice command to the robot (e.g. the wake word + 'follow me').")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass

    await conn.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
