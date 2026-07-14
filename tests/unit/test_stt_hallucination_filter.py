"""Tests for the post-transcription hallucination filter in stt_filters.py.

The filter is a pure function over a text string — trivially unit-testable
without touching Whisper, PyAudio, or the CUDA path. Real user speech must
NEVER match the filter (false positive here = a silent lost interaction);
the specific known-hallucination phrases MUST match (false negative here =
"Sorry, I didn't catch your name" loops on silence).

Imports from ``stt_filters`` directly rather than ``stt`` so the whole
soxr / webrtcvad / torch / faster-whisper stack stays out of test
collection — the filter module is deliberately dependency-free.
"""

import pytest

from stt_filters import _is_hallucination_only, _HALLUCINATION_ONLY


# ── Positive matches: MUST be filtered ─────────────────────────────────────

@pytest.mark.parametrize("phrase", [
    "my name is",
    "My name is",
    "MY NAME IS",
    "My name is.",              # trailing period from Whisper
    "  my name is  ",           # surrounding whitespace
    "my name is?",              # trailing question mark
    "i am",
    "I am",
    "I'm",
    "i'm",
    "call me",
    "Call me",
    "this is",
    "it's",
    "that's",
    "thank you",
    "thanks for watching",
    "please subscribe",
])
def test_filters_known_hallucinations(phrase):
    """Every entry in _HALLUCINATION_ONLY must match, along with common
    Whisper punctuation/whitespace variations."""
    assert _is_hallucination_only(phrase), \
        f"Filter should have caught {phrase!r} — it's a known hallucination"


# ── Negative matches: MUST pass through ────────────────────────────────────

@pytest.mark.parametrize("phrase", [
    # Any intro phrase with an actual name attached is real user speech
    "my name is Joy",
    "my name is George Bush",
    "i am Joy",
    "I'm Alexander",
    "call me Sam",
    "this is Priya",
    # Bare names — the "Joy." confirmation-path case
    "Joy",
    "Joy.",
    "William",
    "Alexander",
    "Priya",
    "Xiuying",
    "Kwame",
    # Confirmation replies — never conflate with the intro fragments
    "yes",
    "no",
    "yeah",
    "nope",
    "no, my name is Joy",
    # Empty / non-name utterances the parser handles separately
    "",
    "hello",
    "sorry",
    "what did you say",
])
def test_does_not_filter_real_speech(phrase):
    """Anything that could plausibly be a real user reply must NOT match.
    A false positive here silently drops user interaction."""
    assert not _is_hallucination_only(phrase), \
        f"Filter incorrectly flagged {phrase!r} as a hallucination"


# ── Set invariants — quick sanity ──────────────────────────────────────────

def test_hallucination_set_is_lowercase():
    """The lookup lowercases the input, so every set entry must already be
    lowercased or it can never match."""
    for phrase in _HALLUCINATION_ONLY:
        assert phrase == phrase.lower(), \
            f"Set entry {phrase!r} is not lowercased — will never match"


def test_hallucination_set_has_no_trailing_punctuation():
    """The lookup strips trailing space/punctuation before comparing.
    Entries with trailing punctuation would also never match."""
    for phrase in _HALLUCINATION_ONLY:
        assert phrase == phrase.strip(" .,?!"), \
            f"Set entry {phrase!r} has trailing punctuation — will never match"


def test_hallucination_set_contains_the_observed_case():
    """Regression guard for the exact string that triggered the loop
    (transcript: My name is), verified against the actual field content."""
    assert "my name is" in _HALLUCINATION_ONLY
