#!/usr/bin/env python3
"""Probe the Go2's `rt/audiohub/player/state` topic.

We want to replace the guess-the-buffer mute mechanism in tts.py with a
callback-driven one — set tts_mute_until off the actual playback-start
event, and clear it off the playback-end event. To do that we first need
to know what messages the topic publishes and when, since the vendor
SDK doesn't document them.

What this script does:
  1. Connect to the robot.
  2. Subscribe to rt/audiohub/player/state and log every message with
     a monotonic timestamp.
  3. Pick the first UUID from get_audio_list() (any cached TTS clip
     works — we just want a known-good audio to trigger playback).
  4. Call play_by_uuid() and log the call/return wall-clock.
  5. Sit and listen for state events for PROBE_LISTEN_SEC seconds.
  6. Print a chronological summary.

Run from the dock (where the robot is reachable). The output tells us:
  * Does the topic emit a discrete "started" message? Roughly when
    relative to play_by_uuid returning?
  * Does it emit a "completed" / "stopped" message at end of audio?
  * What's the message schema (so we can write a stable callback)?

    python3 scripts/probe_audiohub_state.py 192.168.123.161
"""

import asyncio
import json
import sys
import time

from go2_webrtc_driver.webrtc_audiohub import WebRTCAudioHub
from go2_webrtc_driver.webrtc_driver import (
    Go2WebRTCConnection, WebRTCConnectionMethod,
)


STATE_TOPIC       = "rt/audiohub/player/state"
PROBE_LISTEN_SEC  = 15.0   # how long to listen for state after triggering play
CONNECT_TIMEOUT   = 30.0
SETTLE_SEC        = 1.0    # quick settle after connect before subscribing


def _ts(t0):
    return time.monotonic() - t0


def _safe_dump(obj) -> str:
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return repr(obj)


async def main(robot_ip: str) -> int:
    t0 = time.monotonic()
    events: list = []        # (elapsed_sec, kind, payload)

    def log(kind: str, payload):
        elapsed = _ts(t0)
        events.append((elapsed, kind, payload))
        print(f"[{elapsed:7.3f}s] {kind}: {payload}", flush=True)

    def on_state(message):
        # message shape unknown — keep both raw and any obvious unwrappings.
        elapsed = _ts(t0)
        unwrapped = None
        if isinstance(message, dict):
            inner = message.get("data")
            if isinstance(inner, str):
                try:
                    unwrapped = json.loads(inner)
                except Exception:
                    unwrapped = inner
            elif isinstance(inner, dict):
                # Some envelopes nest a JSON string under data.data.
                deeper = inner.get("data") if isinstance(inner, dict) else None
                if isinstance(deeper, str):
                    try:
                        unwrapped = json.loads(deeper)
                    except Exception:
                        unwrapped = deeper
                else:
                    unwrapped = inner
        payload = {
            "raw":       _safe_dump(message),
            "unwrapped": _safe_dump(unwrapped) if unwrapped is not None else None,
        }
        events.append((elapsed, "STATE", payload))
        print(f"[{elapsed:7.3f}s] STATE raw={payload['raw']}", flush=True)
        if payload["unwrapped"] is not None:
            print(f"[{elapsed:7.3f}s]       unwrapped={payload['unwrapped']}",
                  flush=True)

    log("INFO", f"Connecting to {robot_ip}…")
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=robot_ip)
    try:
        await asyncio.wait_for(conn.connect(), timeout=CONNECT_TIMEOUT)
    except Exception as e:
        log("ERROR", f"connect failed: {e}")
        return 1
    log("INFO", "Connected.")

    conn.datachannel.pub_sub.subscribe(STATE_TOPIC, on_state)
    log("INFO", f"Subscribed to {STATE_TOPIC}.")

    # Small settle so any "idle" state baseline arrives before we trigger play.
    await asyncio.sleep(SETTLE_SEC)

    audiohub = WebRTCAudioHub(conn)
    log("INFO", "Fetching audio list…")
    try:
        resp = await audiohub.get_audio_list()
    except Exception as e:
        log("ERROR", f"get_audio_list failed: {e}")
        return 1

    # Unwrap the standard envelope: resp["data"]["data"] is a JSON string
    # whose payload contains audio_list.
    audio_list = []
    try:
        inner = (resp.get("data") or {}).get("data", "{}")
        audio_list = json.loads(inner).get("audio_list", []) or []
    except Exception as e:
        log("ERROR", f"audio_list parse failed: {e}; resp={_safe_dump(resp)[:300]}")
        return 1
    if not audio_list:
        log("ERROR", "audio_list is empty — upload at least one TTS first "
                    "(e.g. start pella-camera once to warm the cache, then "
                    "rerun this probe).")
        return 1

    first = audio_list[0]
    uuid  = first.get("UNIQUE_ID")
    name  = first.get("CUSTOM_NAME", "<unknown>")
    log("INFO", f"Will play UUID={uuid!r} name={name!r}")

    # Trigger playback and time the call.
    log("INFO", "Calling play_by_uuid…")
    t_call = time.monotonic()
    try:
        await audiohub.play_by_uuid(uuid)
    except Exception as e:
        log("ERROR", f"play_by_uuid failed: {e}")
        return 1
    t_return = time.monotonic()
    log("INFO", f"play_by_uuid returned after {(t_return - t_call)*1000:.0f} ms")

    # Listen for state events for a while after the call. Anything that
    # arrives here is the topic actually reporting playback progress.
    log("INFO", f"Listening for state events for {PROBE_LISTEN_SEC:.1f}s…")
    await asyncio.sleep(PROBE_LISTEN_SEC)

    # Chronological summary so it's easy to copy/paste back.
    print("\n" + "=" * 60)
    print(f"Summary: {len(events)} events captured over "
          f"{_ts(t0):.1f}s.")
    print("=" * 60)
    state_events = [(e, p) for e, k, p in events if k == "STATE"]
    print(f"State messages received: {len(state_events)}")
    for elapsed, payload in state_events:
        line = payload.get("unwrapped") or payload["raw"]
        print(f"  +{elapsed:7.3f}s  {line}")
    if not state_events:
        print("  (none — the topic never published during the listen window)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <robot_ip>", file=sys.stderr)
        print(f"  e.g. {sys.argv[0]} 192.168.123.161", file=sys.stderr)
        sys.exit(2)
    sys.exit(asyncio.run(main(sys.argv[1])))
