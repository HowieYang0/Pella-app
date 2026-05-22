#!/usr/bin/env python3
"""Tiny pattern-matched chat responder.

When task_manager has a transcript that the active task didn't consume,
it asks chat.respond_to() whether any of the known questions match. If
one does, the matching reply gets pushed onto say_queue for TTS playback.

This is a deliberately stupid placeholder — eventually replaced by an
LLM-backed responder that handles open-ended conversation. The point
today is to give Pella a visible voice loop ("ask her a question, she
answers") that exercises the STT -> task_manager -> say_queue -> TTS
pipeline end-to-end.

Adding a new canned Q&A is just a (regex, reply) entry in _RESPONSES;
adding the reply text to WARM_PHRASES makes it play instantly on first
use (otherwise gTTS + upload takes ~15s first time).
"""

import re


# (compiled pattern, reply) in priority order. First match wins.
_RESPONSES = [
    (re.compile(r"what(?:'?s| is) your name", re.IGNORECASE),
     "I am Pella."),
    (re.compile(r"can you hear me", re.IGNORECASE),
     "Yes, I can hear you."),
]


# Static reply texts that pella_main should pre-cache at startup so
# play_by_uuid fires instantly when one of them is the chosen reply.
WARM_PHRASES = [reply for _, reply in _RESPONSES]


def respond_to(text: str):
    """Return Pella's reply string if `text` matches a known question.

    Returns None if no pattern matched — task_manager treats that as
    "transcript not consumed" and ignores it.
    """
    if not text:
        return None
    for pattern, reply in _RESPONSES:
        if pattern.search(text):
            return reply
    return None


def get_warm_phrases():
    """Phrases task_manager should hand to the startup TTS warmup."""
    return list(WARM_PHRASES)
