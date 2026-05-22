#!/usr/bin/env python3
"""Probe the ASSISTANT_RECORDER and VUI APIs to find any noise-cancelled audio path."""
import asyncio, json, logging, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "go2_webrtc_connect"))
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, DATA_CHANNEL_TYPE
logging.basicConfig(level=logging.FATAL)

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.161"


async def main():
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP)
    await asyncio.wait_for(conn.connect(), timeout=30.0)
    print("Connected.\n")

    # Subscribe to anything that might carry transcripts or audio events
    for topic in [
        RTC_TOPIC["GPT_FEEDBACK"],
        RTC_TOPIC["SERVICE_STATE"],
        "rt/api/assistant_recorder/response",
        "rt/api/vui/response",
        "rt/voiceprint/result",
        "rt/asr/result",
    ]:
        conn.datachannel.pub_sub.subscribe(
            topic, lambda m, t=topic: print(f"[{t}]\n{json.dumps(m, indent=2)}", flush=True)
        )

    await asyncio.sleep(1)

    # Probe ASSISTANT_RECORDER with api_ids 1001-1005
    print("── Probing ASSISTANT_RECORDER ──")
    for api_id in [1001, 1002, 1003, 1004, 1005]:
        print(f"  api_id={api_id}...", flush=True)
        try:
            resp = await asyncio.wait_for(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["ASSISTANT_RECORDER"],
                    {"api_id": api_id, "parameter": json.dumps({})},
                ),
                timeout=3.0,
            )
            print(f"  RESPONSE: {json.dumps(resp, indent=2)}", flush=True)
        except Exception as e:
            print(f"  no response ({e})", flush=True)

    # Probe VUI with api_ids around recording
    print("\n── Probing VUI (higher api_ids) ──")
    for api_id in [1001, 1002, 1003, 1004, 1005, 2001, 3001]:
        print(f"  api_id={api_id}...", flush=True)
        try:
            resp = await asyncio.wait_for(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["VUI"],
                    {"api_id": api_id, "parameter": json.dumps({})},
                ),
                timeout=3.0,
            )
            print(f"  RESPONSE: {json.dumps(resp, indent=2)}", flush=True)
        except Exception as e:
            print(f"  no response ({e})", flush=True)

    print("\nListening for 30s — speak to the robot now.")
    try:
        await asyncio.sleep(30)
    except KeyboardInterrupt:
        pass

    await conn.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
