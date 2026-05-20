#!/usr/bin/env python3
"""Pella robot action primitives.

Each function is async and takes the WebRTC datachannel as its first argument
so it can be called from within the existing WebRTC event loop in
front_camera_display.py without opening a second connection.
"""

import asyncio

from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

# Go2 pitch convention: negative = nose up (rear lower, camera points up).
# -0.7 rad ≈ 40° — at ~1.5 m distance the camera centres on a face at ~1.6 m height.
LOOK_UP_PITCH = -0.75


async def _sport(datachannel, cmd: str, **params):
    """Send one sport-mode command over the WebRTC data channel."""
    await datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD[cmd], "parameter": params if params else {}},
    )


async def release_control(datachannel):
    """Clear any active pose/velocity target so the joystick controls locomotion again.

    Euler / Sit / Dance commands can leave the robot with a sticky target that
    intercepts joystick input (forward becomes yaw etc.). Sending Move(0,0,0)
    clears it; re-asserting 'normal' motion mode is a safety net.

    Called once by the action consumer when the queue drains, so any sequence
    of pose actions (e.g. look_level → wiggle) cleans up at the end without
    intermediate actions interfering with each other.
    """
    try:
        await datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1002, "parameter": {"name": "normal"}},
        )
    except Exception:
        pass
    await _sport(datachannel, "Move", x=0.0, y=0.0, z=0.0)


async def look_up(datachannel):
    """Tilt body back so the camera points up toward a standing person's face."""
    await _sport(datachannel, "BalanceStand")
    await asyncio.sleep(0.3)
    await _sport(datachannel, "Euler", x=0.0, y=LOOK_UP_PITCH, z=0.0)
    print(f"Action: look_up (pitch={LOOK_UP_PITCH} rad)", flush=True)


async def look_level(datachannel):
    """Return body to level — camera horizontal.

    Cleanup of sticky-Euler state is handled by the action consumer's
    queue-drain release_control, so a wiggle queued immediately after
    is not clobbered, and joystick recovery still happens when nothing
    follows.
    """
    await _sport(datachannel, "BalanceStand")
    await asyncio.sleep(0.3)
    await _sport(datachannel, "Euler", x=0.0, y=0.0, z=0.0)
    print("Action: look_level", flush=True)


async def sit_look_up(datachannel):
    """Sit down — camera points up more steeply than the 0.75 rad Euler limit."""
    await _sport(datachannel, "Sit")
    print("Action: sit_look_up", flush=True)


async def stand_up(datachannel):
    """Rise from a sit and return to level balance.

    Cleanup is handled by the action consumer's queue-drain release_control.
    """
    await _sport(datachannel, "RiseSit")
    await asyncio.sleep(1.5)
    await _sport(datachannel, "BalanceStand")
    print("Action: stand_up", flush=True)


async def wiggle(datachannel, cycles: int = 2, amplitude: float = 0.3):
    """Body yaw oscillation — swings body left/right at ~1 Hz via Euler command."""
    await _sport(datachannel, "BalanceStand")
    await asyncio.sleep(0.3)
    for _ in range(cycles):
        await _sport(datachannel, "Euler", x=0.0, y=0.0, z=amplitude)
        await asyncio.sleep(0.5)
        await _sport(datachannel, "Euler", x=0.0, y=0.0, z=-amplitude)
        await asyncio.sleep(0.5)
    await _sport(datachannel, "Euler", x=0.0, y=0.0, z=0.0)
    print("Action: wiggle", flush=True)


async def hello(datachannel):
    """Hello greeting gesture."""
    await _sport(datachannel, "Hello")
    print("Action: hello", flush=True)


async def dance(datachannel):
    """Dance gesture."""
    await _sport(datachannel, "Dance1")
    print("Action: dance", flush=True)
