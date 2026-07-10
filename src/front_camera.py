#!/usr/bin/env python3
"""Pella's eye: receive video frames from the Go2 front camera over WebRTC.

This module owns just the frame intake — pulling decoded BGR frames from
the WebRTC video track and pushing them onto a shared frame_queue for
pella_main to consume. Face detection / recognition / motion sensing
still live in `vision.py` (perception primitives) and are coordinated
by pella_main's interaction state machine.

Keeping the eye narrow at this stage makes it easy to swap in additional
inputs later (e.g. depth/lidar) without entangling them with the
WebRTC-specific decode loop.
"""

import asyncio
from queue import Empty


async def recv_video(track, frame_queue, stop_event):
    """Pull frames from the WebRTC video track and push to frame_queue.

    The queue is bounded to 2 frames — older frames are dropped so the
    consumer (pygame display + face detection) always processes near-real-time
    imagery, never stale buffered frames piling up after a video stall.

    Logs the frame shape on the first frame per (re)connect — the Go2's
    WebRTC video channel resolution is fixed by the vendor firmware and
    not currently configurable from our side, so this line answers "what
    are we actually getting" once per session for face-quality tuning.
    """
    first_frame = True
    while not stop_event.is_set():
        try:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            if first_frame:
                h, w = img.shape[:2]
                print(f"front_camera: video track opened at {w}x{h}",
                      flush=True)
                first_frame = False
            while frame_queue.qsize() > 2:
                try:
                    frame_queue.get_nowait()
                except Empty:
                    break
            frame_queue.put(img)
        except Exception as e:
            print(f"front_camera.recv_video error: "
                  f"{type(e).__name__}: {e}", flush=True)
            break
