"""Tests for the text parsers in greeting.py.

These functions convert raw Whisper transcripts into structured decisions
about the enrollment / confirmation flow. They're pure text logic — no
threads, no queues, no image data — so they're the highest-ROI thing to
have real tests on. If any of these regresses, the effect is a mis-heard
name silently enrolled or a valid one silently rejected.
"""

import pytest

from greeting import (
    parse_name as _parse_name,
    parse_confirmation as _parse_confirmation,
    has_intro_phrase as _has_intro_phrase,
)


# ── _has_intro_phrase ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "My name is Joy",
    "my name is joy",           # case-insensitive
    "My name's Joy",            # apostrophe contraction
    "I'm Joy",
    "I am Joy",
    "Call me Joy",
    "This is Joy",
    "It's Joy",                 # weak proper-noun pointer added later
    "That's Joy",
    "Well, my name is Joy",     # phrase appears mid-sentence
])
def test_has_intro_phrase_present(text):
    """Confidently detects the common name-introduction shapes."""
    assert _has_intro_phrase(text) is True


@pytest.mark.parametrize("text", [
    "Joy",
    "Hello",
    "Yes",
    "Fine, thanks",
    "",
])
def test_has_intro_phrase_absent(text):
    """Bare replies and unrelated phrases return False."""
    assert _has_intro_phrase(text) is False


# ── _parse_name — happy paths ──────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("Joy",                       "Joy"),
    ("My name is Joy",            "Joy"),
    ("My name is Joy.",           "Joy"),
    ("my name is joy",            "joy"),         # preserves user's case
                                                  # (title-casing happens
                                                  # in _commit_enrollment)
    ("I am Joy",                  "Joy"),
    ("I'm Joy",                   "Joy"),
    ("Call me Joy",               "Joy"),
    ("It's Joy",                  "Joy"),
    ("That's Joy",                "Joy"),
    ("This is Joy",               "Joy"),
    ("My name's Joy",             "Joy"),
    ("Name's Joy",                "Joy"),
])
def test_parse_name_common_intros(text, expected):
    """Every widely-used intro phrasing pulls the same name out."""
    assert _parse_name(text) == expected


def test_parse_name_multi_word():
    """Multi-token names survive the title-case pass."""
    assert _parse_name("My name is George Bush") == "George Bush"


def test_parse_name_strips_courtesy_tail():
    """The trailing 'nice to meet you' after the name is not part of it."""
    assert _parse_name("My name is Sam, nice to meet you") == "Sam"


def test_parse_name_handles_ellipsis_and_stitch():
    """VAD-split stitching: 'My name is... my name is Joy' still parses.

    Whisper emits '...' for trailing-off speech and the parser has to
    collapse those ellipses before splitting on sentence punctuation.
    """
    assert _parse_name("My name is... my name is Joy") == "Joy"


# ── _parse_name — rejects (the common misread failure modes) ───────────────

@pytest.mark.parametrize("text", [
    "",                          # empty
    "Hello, sir",                # greeting, no name
    "Yes",                       # yes reply
    "OK",                        # ack
    "Fine, thanks",              # state reply
    "We're here",                # state reply with pronoun
    "It's fine",                 # 'it's' pointer followed by state word
    "It's me",                   # pronoun after 'it's'
    "That's right",              # state word
    "It's cool",                 # state word
    "Hmm",                       # filler
    "Huh",                       # filler
    "He's illisoned",            # 'he' pronoun
    "Hey, Mr. Ellison",          # 'hey' filler + 'mr' honorific
])
def test_parse_name_rejects_non_names(text):
    """Any transcript that's clearly not a name-introduction is empty-parsed."""
    assert _parse_name(text) == ""


def test_parse_name_require_intro_phrase_rejects_bare():
    """Correction mode: 'Joy' alone doesn't count as a rename."""
    assert _parse_name("Joy", require_intro_phrase=True) == ""


def test_parse_name_require_intro_phrase_accepts_intro():
    """Correction mode: 'My name is Joy' DOES count as a rename."""
    assert _parse_name("My name is Joy", require_intro_phrase=True) == "Joy"


# ── _parse_confirmation — yes/no/correction routing ────────────────────────

@pytest.mark.parametrize("text", [
    "Yes", "Yeah", "Yep", "Yup", "Sure", "OK", "Okay",
    "Right", "Correct", "That's right", "That is right",
])
def test_parse_confirmation_yes(text):
    """Every affirmation shape returns ('yes', None)."""
    verdict, new_name = _parse_confirmation(text)
    assert verdict == "yes"
    assert new_name is None


@pytest.mark.parametrize("text", [
    "No", "Nope", "Nah", "Wrong", "Incorrect", "Not quite",
    "That's wrong",
])
def test_parse_confirmation_no_alone(text):
    """A bare 'no' returns ('no', None) — will trigger a re-ask."""
    verdict, new_name = _parse_confirmation(text)
    assert verdict == "no"
    assert new_name is None


@pytest.mark.parametrize("text, expected_name", [
    ("No, Joy",                  "Joy"),
    ("No, my name is Joy",       "Joy"),
    ("No, it's Joy",             "Joy"),
    ("Nope, call me Sam",        "Sam"),
    ("Wrong, it's William",      "William"),
])
def test_parse_confirmation_no_plus_correction(text, expected_name):
    """'No, X' or 'No, my name is X' carries the corrected name in the tuple."""
    verdict, new_name = _parse_confirmation(text)
    assert verdict == "no"
    assert new_name == expected_name


@pytest.mark.parametrize("text, expected_name", [
    ("Joy",                      "Joy"),
    ("It's Joy",                 "Joy"),
    ("My name is Joy",           "Joy"),
    ("Call me Sam",              "Sam"),
])
def test_parse_confirmation_implicit_correction(text, expected_name):
    """Bare name / intro without 'no' during confirmation IS an implicit
    correction — the user is telling us the right name instead of just
    saying yes."""
    verdict, new_name = _parse_confirmation(text)
    assert verdict == "no"
    assert new_name == expected_name


@pytest.mark.parametrize("text", ["", "Hmm", "Huh", "What?"])
def test_parse_confirmation_unclear(text):
    """Fillers and empty replies leave the confirmation state unresolved."""
    verdict, new_name = _parse_confirmation(text)
    assert verdict == "unclear"
    assert new_name is None


def test_parse_confirmation_yes_wins_over_ambient_name():
    """Regression: a plain 'Yes' must not be parsed as a correction with
    an ambient 'yes' captured as the new name."""
    verdict, new_name = _parse_confirmation("Yes")
    assert verdict == "yes"
    assert new_name is None
