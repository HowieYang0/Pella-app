"""Tests for the phrase generators + name canonicalization in greeting.py.

These functions define every English string Pella ever says. They're
trivial one-liners individually, but the tests here lock in:

  - the exact wording used at each moment in the flow (so a stray edit
    doesn't accidentally change what the user hears)
  - the canonicalization contract (dir_name <-> display_name) since
    it's used from multiple call sites and must stay consistent
  - the warm-phrase composition matches what the task queues at
    runtime, so the pre-cache is aligned with the real TTS

If a phrase legitimately needs to change (localisation, tone tweak),
update the assertion here in the same commit as the string.
"""

import pytest

from greeting import (
    # Canonicalization
    canonicalize,
    display_name_from_dir_name,
    # Phrase generators
    intro_prompt,
    name_apology,
    clarity_apology,
    confirmation_prompt,
    greet_phrase,
    welcome_phrase,
    reject_phrase,
    correction_ack,
    intro_warm_phrases,
    correction_warm_phrases,
    static_warm_phrases,
    per_person_warm_phrases,
)


# ── Canonicalization ────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected_dir, expected_display", [
    ("Joy",              "joy",           "Joy"),
    ("joy",              "joy",           "Joy"),
    ("JOY",              "joy",           "Joy"),
    ("George Bush",      "george_bush",   "George Bush"),
    ("george bush",      "george_bush",   "George Bush"),
    ("Mary Jane Watson", "mary_jane_watson", "Mary Jane Watson"),
])
def test_canonicalize_pairs(raw, expected_dir, expected_display):
    """Canonicalize is the single source of truth for the dir_name /
    display_name pair — both derive from the same input."""
    dir_name, display_name = canonicalize(raw)
    assert dir_name == expected_dir
    assert display_name == expected_display


@pytest.mark.parametrize("dir_name, expected", [
    ("joy",              "Joy"),
    ("george_bush",      "George Bush"),
    ("mary_jane_watson", "Mary Jane Watson"),
])
def test_display_name_from_dir_name(dir_name, expected):
    """The inverse: dir_name -> display_name, used when the recognizer
    hands us a stored id and we want to greet the person."""
    assert display_name_from_dir_name(dir_name) == expected


def test_canonicalize_round_trip():
    """canonicalize followed by display_name_from_dir_name preserves the
    display form. Useful when the task saves a name, then later reads
    it back from the recognizer to say 'Hi, X'."""
    for raw in ("Joy", "george bush", "Mary Jane Watson"):
        dir_name, display_name = canonicalize(raw)
        assert display_name_from_dir_name(dir_name) == display_name


# ── Static phrase generators (name-independent) ────────────────────────────

def test_intro_prompt():
    assert intro_prompt() == "Hello, I am Pella. What is your name?"


def test_name_apology():
    assert name_apology() == "Sorry, I didn't catch your name."


def test_clarity_apology():
    assert clarity_apology() == "Sorry, I cannot see you clearly."


# ── Name-parameterised phrase generators ───────────────────────────────────

def test_confirmation_prompt():
    assert confirmation_prompt("Joy") == "Did you say Joy?"


def test_greet_phrase():
    assert greet_phrase("Joy") == "Hi, Joy"


def test_welcome_phrase():
    """Note the exclamation mark — 'Nice to meet you, X!' — has been the
    contract since the intro flow was written. Losing it changes the TTS
    pacing/intonation."""
    assert welcome_phrase("Joy") == "Nice to meet you, Joy!"


def test_reject_phrase():
    """Two-sentence phrase used when the enrolment mechanism rejected the
    captured face (e.g. it looks like a different enrolled person)."""
    assert reject_phrase("Joy") == (
        "You don't look like the Joy I know. Sorry about that."
    )


def test_correction_ack():
    assert correction_ack("Joy") == "Sorry, Joy. Got it."


def test_phrases_handle_multi_word_names():
    """None of the phrase generators do any special-casing on names — they
    hand off to f-string substitution — but a regression test makes sure
    no future change adds e.g. .split() that would break multi-word names."""
    assert confirmation_prompt("George Bush") == "Did you say George Bush?"
    assert greet_phrase("George Bush") == "Hi, George Bush"
    assert welcome_phrase("George Bush") == "Nice to meet you, George Bush!"


# ── Warm-phrase lists ──────────────────────────────────────────────────────

def test_intro_warm_phrases_pair():
    """The intro flow pre-caches the two phrases most likely to fire next:
    the successful enrolment welcome, and the mismatch rejection."""
    phrases = intro_warm_phrases("Joy")
    assert list(phrases) == [
        "Nice to meet you, Joy!",
        "You don't look like the Joy I know. Sorry about that.",
    ]


def test_correction_warm_phrases_pair():
    """After a rename, the next greeting is likely — pre-cache the new
    'Hi, X' plus its 'Nice to meet you, X!' fallback."""
    phrases = correction_warm_phrases("Joy")
    assert list(phrases) == [
        "Hi, Joy",
        "Nice to meet you, Joy!",
    ]


def test_static_warm_phrases_content():
    """pella_main pre-caches these three at startup so first-time
    generation latency doesn't show up mid-interaction."""
    phrases = static_warm_phrases()
    assert phrases == [
        "Hello, I am Pella. What is your name?",
        "Sorry, I didn't catch your name.",
        "Sorry, I cannot see you clearly.",
    ]


def test_per_person_warm_phrases_uses_display_form():
    """Takes a dir_name (from recognizer.known) and produces phrases with
    the display form (so 'george_bush' -> 'Hi, George Bush')."""
    phrases = per_person_warm_phrases("george_bush")
    assert phrases == ["Hi, George Bush", "Nice to meet you, George Bush!"]


def test_per_person_warm_phrases_single_word_name():
    phrases = per_person_warm_phrases("joy")
    assert phrases == ["Hi, Joy", "Nice to meet you, Joy!"]
