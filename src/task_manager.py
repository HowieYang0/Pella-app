#!/usr/bin/env python3
"""Task manager: owns the set of available tasks and dispatches between them.

This is the *only* module that knows about specific task types. pella_main
talks to TaskManager — it has no knowledge of recog_greeting, follow_person,
fetch, or any other concrete task implementation. As new tasks are added,
this module grows (task registry, switching policy, sub-task stack);
pella_main does not.

Today the task set is minimal: a single perpetual recog_greeting task.

Surface to pella_main:
  * TaskManager(frame_queue, action_queue, say_queue, stop_event)
       — wires up shared queues, owns task-shared perception resources
         (e.g. the face recognizer), instantiates the active task(s).
  * tick(now) -> TickResult
       — runs the active task and returns whatever it asks pella_main to
         do (pin a display, emit a status event, render the latest frame).
  * submit_transcript(now, text) -> bool
       — forwards transcripts to the active task; returns True if
         the task consumed it.
"""

import chat
import recog_greeting
from vision import load_recognizer


class TaskManager:
    """Holds the currently-active task and routes pella_main's calls to it.

    Future expansions of this class will manage a task stack with switching
    rules, handle sub-task decomposition, and gate dispatch on robot context.
    None of that changes the surface seen by pella_main.
    """

    def __init__(self, frame_queue, action_queue, say_queue, prep_queue,
                 stop_event):
        # Shared perception resources (used by tasks that need identity).
        # Loaded here so pella_main doesn't need to know any task cares.
        recognizer = load_recognizer()

        # Today: a single perpetual recog_greeting task. Future: a registry
        # or stack of tasks, plus switching policy.
        self._active_task = recog_greeting.RecogGreetingTask(
            frame_queue, action_queue, say_queue, prep_queue,
            recognizer, stop_event,
        )

        # Retain a reference for routing chat replies (when neither the
        # active task nor anything else consumed the transcript).
        self._say_queue = say_queue

    def tick(self, now):
        """Run one iteration of whichever task is currently active."""
        return self._active_task.tick(now)

    def submit_transcript(self, now, text, capture_t) -> bool:
        """Forward a transcript line to the active task, then to chat.

        `capture_t` is the monotonic time the user actually started
        speaking (stamped by stt.py when VAD entered speech mode), not
        the time the transcript arrived from Whisper. Tasks use this
        to match speech against listening windows that can be much
        shorter than the worst-case transcription latency.

        Cascade: the active task gets first dibs (e.g. recog_greeting may
        be in INTRODUCING and waiting for a name). If the task doesn't
        consume the line, chat.respond_to() is tried — when it returns
        a reply string, we push it onto say_queue for TTS playback.
        Returns True if anything handled the transcript.
        """
        if self._active_task.submit_transcript(now, text, capture_t):
            return True
        reply = chat.respond_to(text)
        if reply:
            try:
                self._say_queue.put_nowait(reply)
            except Exception:
                pass
            print(f"Chat: {repr(text)} -> {repr(reply)}", flush=True)
            return True
        return False

    def get_warm_phrases(self) -> list:
        """Collect phrases that pella_main should pre-cache at startup.

        Pulls from both the active task (its known greetings + prompts)
        and the chat module (its static replies). pella_main treats the
        result as opaque strings.
        """
        return self._active_task.get_warm_phrases() + chat.get_warm_phrases()
