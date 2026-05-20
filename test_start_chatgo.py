#!/usr/bin/env python3
"""Start the Go2's chat_go voice service via DDS RobotStateClient.

chat_go uses the robot's hardware noise-cancelled mic array and publishes
ASR results to rt/gptflowfeedback.

Usage:
    python3 test_start_chatgo.py <network_interface> [domain_id]

Example (real robot on eth0):
    python3 test_start_chatgo.py eth0
    python3 test_start_chatgo.py enx000ec6768747
"""

import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unitree_sdk2_python"))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <network_interface> [domain_id]")
        sys.exit(1)

    iface     = sys.argv[1]
    domain_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    print(f"Initializing DDS on interface={iface} domain={domain_id}...")
    ChannelFactoryInitialize(domain_id, iface)

    rsc = RobotStateClient()
    rsc.SetTimeout(5.0)
    rsc.Init()
    time.sleep(1)

    # ── List current services ───────────────────────────────────────────────
    print("\n── Current services ──")
    code, lst = rsc.ServiceList()
    if code != 0:
        print(f"ServiceList failed (code={code}). Check interface name and DDS domain.")
        sys.exit(1)

    chat_go_running = False
    for s in lst:
        marker = " ◀" if s.name == "chat_go" else ""
        print(f"  {s.name:30s}  status={s.status}  protect={s.protect}{marker}")
        if s.name == "chat_go":
            chat_go_running = (s.status == 1)

    # ── Helper ──────────────────────────────────────────────────────────────
    def switch_and_wait(name, enable, wait=3):
        current = {s.name: s.status for s in lst}
        if current.get(name) == (1 if enable else 0):
            print(f"  {name} already {'running' if enable else 'stopped'}.")
            return True
        action = "Starting" if enable else "Stopping"
        print(f"  {action} {name}...")
        c = rsc.ServiceSwitch(name, enable)
        print(f"    ServiceSwitch('{name}', {enable}) → code={c}")
        time.sleep(wait)
        c2, lst2 = rsc.ServiceList()
        if c2 == 0:
            st = {s.name: s.status for s in lst2}.get(name, -1)
            ok = (st == 1) if enable else (st == 0)
            print(f"    {name} status={st}  ({'OK' if ok else 'FAILED'})")
            return ok
        return False

    # ── Start dependency chain ───────────────────────────────────────────────
    if chat_go_running:
        print("\nchat_go is already running.")
    else:
        print()
        # audio_hub must be up before chat_go can open the mic
        switch_and_wait("audio_hub", True, wait=3)
        # vui_service handles wake-word detection that chat_go relies on
        switch_and_wait("vui_service", True, wait=3)
        switch_and_wait("chat_go", True, wait=5)

    # ── Final state ─────────────────────────────────────────────────────────
    print("\n── Final service states ──")
    code, lst = rsc.ServiceList()
    if code == 0:
        for name in ("audio_hub", "vui_service", "chat_go"):
            for s in lst:
                if s.name == name:
                    print(f"  {s.name:30s}  status={s.status}  protect={s.protect}")
        for s in lst:
            if s.name == "chat_go":
                if s.status == 1:
                    print("\nchat_go is running. Speak to the robot —")
                    print("subscribe to rt/gptflowfeedback via WebRTC to receive transcripts.")
                else:
                    print("\nchat_go status is still 0 — service did not start.")


if __name__ == "__main__":
    main()
