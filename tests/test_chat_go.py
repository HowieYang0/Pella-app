#!/usr/bin/env python3
"""Probe the Go2 for a way to start the chat_go voice-assistant service.

Approaches tried in order:
  1. Subscribe to rt/api/bashrunner/response, then fire bash commands
  2. Switch to "ai" motion mode and watch SERVICE_STATE
  3. Try undocumented service-control topics

Usage:
    python3 test_chat_go.py [robot_ip]
"""

import asyncio
import json
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "go2_webrtc_connect"))

from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, DATA_CHANNEL_TYPE

logging.basicConfig(level=logging.FATAL)

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.161"

# Additional topics not in constants
BASH_RESP   = "rt/api/bashrunner/response"
SVC_CTRL    = "rt/api/service_manager/request"   # hypothetical
ROBOT_STATE = "rt/api/robot_state/request"        # hypothetical


def _show(label, msg):
    print(f"\n[{label}]\n{json.dumps(msg, indent=2)}", flush=True)


async def _bash(conn, cmd, *, timeout=8.0):
    """Send a bash command via fire-and-forget (response comes via subscription)."""
    print(f"\n  $ {cmd}", flush=True)
    payload = {
        "header": {"identity": {"id": 12345678, "api_id": 1001}},
        "parameter": json.dumps({"cmd": cmd}),
    }
    conn.datachannel.pub_sub.publish_without_callback(
        topic=RTC_TOPIC["BASH_REQ"],
        data=payload,
        msg_type=DATA_CHANNEL_TYPE["REQUEST"],
    )
    await asyncio.sleep(timeout)


async def main():
    print(f"Connecting to {ROBOT_IP}...", flush=True)
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP)
    await asyncio.wait_for(conn.connect(), timeout=30.0)
    print("Connected.\n", flush=True)

    # ── Subscribe to everything we care about ────────────────────────────────
    conn.datachannel.pub_sub.subscribe(RTC_TOPIC["SERVICE_STATE"],
                                        lambda m: _show("SERVICE_STATE", m))
    conn.datachannel.pub_sub.subscribe(RTC_TOPIC["GPT_FEEDBACK"],
                                        lambda m: _show("GPT_FEEDBACK", m))
    conn.datachannel.pub_sub.subscribe(BASH_RESP,
                                        lambda m: _show("BASH_RESP", m))

    await asyncio.sleep(1)

    # ── Step 1: bash runner via fire-and-forget ───────────────────────────────
    print("=" * 60, flush=True)
    print("Step 1: bash runner (fire-and-forget, listening for response)", flush=True)
    await _bash(conn, "ps aux | grep -E 'chat_go|pet_go|voice|asr' | grep -v grep")
    await _bash(conn, "systemctl list-units --type=service | grep -iE 'chat|voice|asr|pet'")
    await _bash(conn, "ls /home/unitree/ 2>/dev/null || ls /root/ 2>/dev/null")

    # ── Step 2: try switching to "ai" mode and monitor service state ─────────
    print("=" * 60, flush=True)
    print("Step 2: switching to 'ai' mode, watching SERVICE_STATE for 10s", flush=True)
    try:
        resp = await asyncio.wait_for(
            conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1002, "parameter": {"name": "ai"}},
            ),
            timeout=5.0,
        )
        print(f"  ai-mode response: {json.dumps(resp, indent=2)}", flush=True)
    except Exception as e:
        print(f"  ai-mode switch failed: {e}", flush=True)
    await asyncio.sleep(10)

    # ── Step 3: try hypothetical service-control topics ──────────────────────
    print("=" * 60, flush=True)
    print("Step 3: probing undocumented service-control topics", flush=True)
    for topic, label in [
        (SVC_CTRL,    "service_manager"),
        (ROBOT_STATE, "robot_state"),
    ]:
        for api_id in [1001, 1002, 1003]:
            print(f"  {label} api_id={api_id}...", flush=True)
            try:
                resp = await asyncio.wait_for(
                    conn.datachannel.pub_sub.publish_request_new(
                        topic, {"api_id": api_id}
                    ),
                    timeout=3.0,
                )
                print(f"  RESPONSE: {json.dumps(resp, indent=2)}", flush=True)
            except Exception as e:
                print(f"  no response ({e})", flush=True)

    # ── Step 4: listen for GPT_FEEDBACK ─────────────────────────────────────
    print("=" * 60, flush=True)
    print("Step 4: listening for GPT_FEEDBACK for 60 s — speak to the robot now", flush=True)
    try:
        await asyncio.sleep(60)
    except KeyboardInterrupt:
        pass

    # ── Step 5: switch back to normal mode ───────────────────────────────────
    print("Switching back to normal mode...", flush=True)
    try:
        await asyncio.wait_for(
            conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1002, "parameter": {"name": "normal"}},
            ),
            timeout=5.0,
        )
    except Exception:
        pass

    await conn.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
