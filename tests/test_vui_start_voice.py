#!/usr/bin/env python3
"""Probe wider VUI api_ids while also running DDS ServiceSwitch concurrently."""
import asyncio, json, logging, subprocess, sys, os, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "go2_webrtc_connect"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unitree_sdk2_python"))
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC
logging.basicConfig(level=logging.FATAL)

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.161"
VUI = RTC_TOPIC["VUI"]


async def vreq(conn, api_id, param=None, timeout=3.0):
    try:
        r = await asyncio.wait_for(
            conn.datachannel.pub_sub.publish_request_new(
                VUI, {"api_id": api_id, "parameter": json.dumps(param or {})}
            ),
            timeout=timeout,
        )
        code = r.get("data", {}).get("header", {}).get("status", {}).get("code", "?") \
               if isinstance(r.get("data"), dict) else "?"
        inner = r.get("data", {}).get("data", "") \
                if isinstance(r.get("data"), dict) else r.get("data", "")
        return code, inner
    except Exception as e:
        return "timeout", str(e)[:40]


def _run_dds_switch():
    """Run in a background thread: DDS ServiceSwitch while WebRTC is live."""
    time.sleep(3)          # let WebRTC settle and VUI enable first
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
        ChannelFactoryInitialize(0, "eth0")
        rsc = RobotStateClient()
        rsc.SetTimeout(5.0)
        rsc.Init()
        time.sleep(1)
        for svc in ("audio_hub", "vui_service", "chat_go"):
            code = rsc.ServiceSwitch(svc, True)
            print(f"  DDS ServiceSwitch({svc}) → code={code}", flush=True)
            time.sleep(2)
            _, lst = rsc.ServiceList()
            for s in lst:
                if s.name == svc:
                    print(f"    {s.name} status={s.status}", flush=True)
    except Exception as e:
        print(f"  DDS thread error: {e}", flush=True)


async def main():
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP)
    await asyncio.wait_for(conn.connect(), timeout=30.0)
    print("Connected.\n")

    conn.datachannel.pub_sub.subscribe(
        RTC_TOPIC["SERVICE_STATE"],
        lambda m: _on_svc(m),
    )
    conn.datachannel.pub_sub.subscribe(
        RTC_TOPIC["GPT_FEEDBACK"],
        lambda m: print(f"\n*** GPT_FEEDBACK: {json.dumps(m, indent=2)}", flush=True),
    )

    def _on_svc(msg):
        raw = msg.get("data", "")
        try:
            lst = json.loads(raw) if isinstance(raw, str) else raw
            changed = [s for s in lst if s["name"] in ("audio_hub", "vui_service", "chat_go") and s["status"] == 1]
            if changed:
                print(f"\n*** SERVICE STARTED: {[s['name'] for s in changed]} ***", flush=True)
        except Exception:
            pass

    # Enable VUI first
    code, _ = await vreq(conn, 1001, {"enable": 1})
    print(f"VUI enable=1 → code={code}")

    # Probe undiscovered api_ids (6-20, 100-110, 2001-2005)
    print("\n── Probing more VUI api_ids ──")
    for api_id in list(range(1006, 1016)) + [1020, 1050, 1100, 2001, 2002, 2003, 3001, 3002]:
        code, data = await vreq(conn, api_id, timeout=2.0)
        if code != "timeout":
            print(f"  api_id={api_id:5d}  code={code}  data={data!r}", flush=True)

    # Start DDS thread to fire ServiceSwitch while WebRTC is live
    print("\n── Starting DDS ServiceSwitch in background thread ──")
    t = threading.Thread(target=_run_dds_switch, daemon=True)
    t.start()

    print("\nWaiting 30s — watching for service starts and GPT_FEEDBACK…")
    await asyncio.sleep(30)
    t.join(timeout=5)

    await conn.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
