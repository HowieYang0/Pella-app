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

import recog_greeting
from vision import load_recognizer


class TaskManager:
    """Holds the currently-active task and routes pella_main's calls to it.

    Future expansions of this class will manage a task stack with switching
    rules, handle sub-task decomposition, and gate dispatch on robot context.
    None of that changes the surface seen by pella_main.
    """

    def __init__(self, frame_queue, action_queue, say_queue, stop_event):
        # Shared perception resources (used by tasks that need identity).
        # Loaded here so pella_main doesn't need to know any task cares.
        recognizer = load_recognizer()

        # Today: a single perpetual recog_greeting task. Future: a registry
        # or stack of tasks, plus switching policy.
        self._active_task = recog_greeting.RecogGreetingTask(
            frame_queue, action_queue, say_queue, recognizer, stop_event,
        )

    def tick(self, now):
        """Run one iteration of whichever task is currently active."""
        return self._active_task.tick(now)

    def submit_transcript(self, now, text) -> bool:
        """Forward a transcript line to the active task."""
        return self._active_task.submit_transcript(now, text)

    def get_warm_phrases(self) -> list:
        """Collect phrases the active task wants pre-cached at startup."""
        return self._active_task.get_warm_phrases()
