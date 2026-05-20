#!/usr/bin/env python3
"""Pella's limbs: motor primitives + the action-queue executor.

The primitives (look_up, sit_look_up, stand_up, wiggle, …) are async
coroutines that publish sport commands over the Go2 WebRTC data channel.
They are the leaves of the motion stack — when a higher-level
task_manager appears (see the project plan), it composes plans and
eventually emits these primitive names into the action queue.

run_action_consumer is the serializing executor that pops names from
the queue, dispatches via _ACTION_MAP, and waits for each coroutine to
finish before starting the next. When the queue drains it queues
release_control once to clear sticky Euler/Sit state so the joystick
controls locomotion again.

This module also exposes two small helpers for callers that need to
peek or filter the queue without consuming it (drain_seek_actions /
queue_contains) — used by pella_main's interaction state machine.
"""

import asyncio
from queue import Empty

from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

# Go2 pitch convention: negative = nose up (rear lower, camera points up).
# -0.7 rad ≈ 40° — at ~1.5 m distance the camera centres on a face at ~1.6 m height.
LOOK_UP_PITCH = -0.75


# ── Sport command helper ─────────────────────────────────────────────────────

async def _sport(datachannel, cmd: str, **params):
    """Send one sport-mode command over the WebRTC data channel."""
    await datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD[cmd], "parameter": params if params else {}},
    )


# ── Motion primitives ────────────────────────────────────────────────────────

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


# ── Action dispatch ──────────────────────────────────────────────────────────

ACTION_MAP = {
    "look_up":     look_up,
    "look_level":  look_level,
    "sit_look_up": sit_look_up,
    "stand_up":    stand_up,
    "wiggle":      wiggle,
    "hello":       hello,
    "dance":       dance,
}

# Seek-pose actions — used by drain_seek_actions to selectively flush stale
# seek requests when recognition succeeds while they're still queued.
SEEK_ACTIONS = ("look_up", "sit_look_up")


# ── Action queue helpers ─────────────────────────────────────────────────────

def drain_seek_actions(q):
    """Remove any pending look_up/sit_look_up from the action queue, preserving
    the order of the remaining items.

    Called by the interaction state machine when recognition succeeds so stale
    seek actions queued by a prior seek-timeout don't run after the face has
    already been found.
    """
    keep = []
    try:
        while True:
            item = q.get_nowait()
            if item not in SEEK_ACTIONS:
                keep.append(item)
    except Empty:
        pass
    for item in keep:
        try:
            q.put_nowait(item)
        except Exception:
            pass


def queue_contains(q, name):
    """Best-effort check whether `name` is still pending in the action queue.

    Racy with the consumer running on another thread, but used only for ordering
    heuristics where a missed/false-positive will self-correct on the next
    display iteration.
    """
    try:
        return name in list(q.queue)
    except Exception:
        return False


# ── Action consumer (serializing executor) ───────────────────────────────────

async def run_action_consumer(action_queue, datachannel, stop_event):
    """Pop action names from `action_queue` and run them one at a time.

    Each name is looked up in ACTION_MAP; the resulting coroutine is awaited
    to completion before the next item is popped, so e.g. wiggle waits for
    stand_up to finish instead of running concurrently and being clobbered.

    When an action completes and the queue is empty, release_control runs once
    to clear any sticky Euler/Sit target — this keeps the joystick responsive
    after pose-changing actions like look_level. A follow-up action arriving
    before the queue drains short-circuits the release (no intermediate cleanup
    between e.g. look_level → wiggle).
    """
    action_task = None
    pending_release = False
    while not stop_event.is_set():
        if action_task is None or action_task.done():
            if action_task is not None and action_task.done():
                action_task = None
                pending_release = True
            try:
                action_name = action_queue.get_nowait()
                fn = ACTION_MAP.get(action_name)
                if fn:
                    action_task = asyncio.ensure_future(fn(datachannel))
                    pending_release = False
            except Empty:
                if pending_release:
                    action_task = asyncio.ensure_future(release_control(datachannel))
                    pending_release = False
        await asyncio.sleep(0.5)
