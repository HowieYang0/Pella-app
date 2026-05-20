#!/usr/bin/env python3
"""Try to enable the VUI (wake-word/voice chain) and watch whether
vui_service and chat_go start as a result."""
import asyncio, json, logging, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "go2_webrtc_connect"))
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC
logging.basicConfig(level=logging.FATAL)

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.161"
VUI = RTC_TOPIC["VUI"]


async def vui(conn, api_id, param=None):
    opts = {"api_id": api_id, "parameter": json.dumps(param or {})}
    try:
        r = await asyncio.wait_for(
            conn.datachannel.pub_sub.publish_request_new(VUI, opts),
            timeout=4.0,
        )
        data = r.get("data", {})
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                pass
        inner = data.get("data", "") if isinstance(data, dict) else data
        code = (data.get("header", {}).get("status", {}).get("code", "?")
                if isinstance(data, dict) else "?")
        return code, inner
    except Exception as e:
        return "timeout", str(e)


async def main():
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP)
    await asyncio.wait_for(conn.connect(), timeout=30.0)
    print("Connected.\n")

    conn.datachannel.pub_sub.subscribe(
        RTC_TOPIC["SERVICE_STATE"],
        lambda m: print(f"\n[SERVICE_STATE CHANGE]", flush=True) or _show_services(m),
    )
    conn.datachannel.pub_sub.subscribe(
        RTC_TOPIC["GPT_FEEDBACK"],
        lambda m: print(f"\n[GPT_FEEDBACK] {json.dumps(m, indent=2)}", flush=True),
    )

    def _show_services(msg):
        raw = msg.get("data", "")
        try:
            lst = json.loads(raw) if isinstance(raw, str) else raw
            for s in lst:
                if s["name"] in ("audio_hub", "vui_service", "chat_go"):
                    print(f"  {s['name']:20s} status={s['status']}", flush=True)
        except Exception:
            pass

    # ── Current state ──────────────────────────────────────────────────────
    print("── VUI current state ──")
    code, data = await vui(conn, 1002)  # get enable
    print(f"  enable query: code={code}  data={data!r}")
    code, data = await vui(conn, 1004)  # get volume
    print(f"  volume query: code={code}  data={data!r}")

    # ── Try all plausible "set enable" api_ids ─────────────────────────────
    print("\n── Trying to enable VUI (set enable=1) ──")
    for api_id in [1001, 1003, 1005, 1006, 2001]:
        print(f"  api_id={api_id} param={{enable:1}} ...", flush=True)
        code, data = await vui(conn, api_id, {"enable": 1})
        print(f"    → code={code}  data={data!r}", flush=True)
        await asyncio.sleep(1)
        # Re-check enable status
        c2, d2 = await vui(conn, 1002)
        print(f"    enable now: {d2!r}", flush=True)
        if d2 and "1" in str(d2):
            print(f"    *** VUI enabled by api_id={api_id}! ***", flush=True)
            break

    # ── Also try setting volume in case that unlocks something ────────────
    print("\n── Setting volume=5 ──")
    code, data = await vui(conn, 1003, {"volume": 5})
    print(f"  api_id=1003 volume=5 → code={code}  data={data!r}")
    await asyncio.sleep(1)
    code, data = await vui(conn, 1004)
    print(f"  volume now: {data!r}")

    print("\nWaiting 15s — watching SERVICE_STATE for audio_hub/vui_service/chat_go changes…")
    await asyncio.sleep(15)

    await conn.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
