#!/usr/bin/env python3
"""Interactive test for actions.py — connects via WebRTC and runs actions on demand.

Usage:
    python3 test_actions.py [robot_ip]

Keys:
    u  — look_up
    l  — look_level
    w  — wiggle
    q  — quit
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "go2_webrtc_connect"))

from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
import actions


async def main():
    robot_ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.161"
    print(f"Connecting to {robot_ip}...")
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=robot_ip)
    await conn.connect()
    print("Connected.")
    print("  u = look_up   l = look_level   w = wiggle   h = hello   d = dance   q = quit")

    while True:
        key = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
        key = key.strip().lower()
        if key == "u":
            await actions.look_up(conn.datachannel)
        elif key == "l":
            await actions.look_level(conn.datachannel)
        elif key == "w":
            await actions.wiggle(conn.datachannel)
        elif key == "h":
            await actions.hello(conn.datachannel)
        elif key == "d":
            await actions.dance(conn.datachannel)
        elif key == "q":
            break
        else:
            print("  u = look_up   l = look_level   w = wiggle   h = hello   d = dance   q = quit")

    await conn.disconnect()
    print("Disconnected.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
