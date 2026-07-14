"""Post-transcription filters over Whisper output.

Kept in its own file — free of numpy / soxr / webrtcvad / faster-whisper /
torch imports — so the unit tests can exercise this logic without dragging
the whole ML stack in at collection time.

Everything here is pure text-in / bool-out. If you're adding a filter that
needs the audio samples or the Whisper model, it belongs in stt.py, not
here.
"""

# ── Post-transcription hallucination filter ─────────────────────────────────
#
# Whisper is prone to producing certain short English phrases from silent
# or near-silent audio, especially when the initial_prompt biases the
# language model toward a specific completion pattern. Rather than let
# these reach the task as if they were real speech, we recognise the
# specific patterns and reject them post-transcription.
#
# Every entry here is either:
#   * an intro phrase with NO name after it — structurally impossible as a
#     real reply to "What is your name?" (Whisper cut off the reply, or
#     invented the intro from nothing)
#   * a well-known Whisper training-artifact hallucination (its training
#     set included a lot of YouTube captions, so silent clips often decode
#     to "Thanks for watching" / "Please subscribe" / etc.)
#
# Real user speech never lands in this set. Keep the set narrow — only
# add a phrase after observing it actually get hallucinated in production,
# not preemptively, so we don't start eating legitimate short replies.
_HALLUCINATION_ONLY = frozenset({
    # Intro fragments — the name got cut off (or was never there)
    "my name is",
    "i am",
    "i'm",
    "call me",
    "this is",
    "it's",
    "that's",
    # YouTube-caption hallucinations from silent clips
    "thank you",
    "thanks for watching",
    "please subscribe",
})


def _is_hallucination_only(text: str) -> bool:
    """True iff ``text`` is one of the known no-name / no-content phrases
    Whisper hallucinates from silent audio.

    Compares after lowercasing and stripping trailing punctuation only —
    surrounding whitespace and terminal ".!?, " get normalised so that
    "My name is." and "my name is" both match.
    """
    return text.lower().strip(" .,?!") in _HALLUCINATION_ONLY
