#!/usr/bin/env python3
"""Greeting text: parsers, phrase templates, name canonicalization.

Everything in this module is a pure text function — takes strings in,
returns strings (or booleans, or small tuples) out. No queues, no
threads, no image data, no state. That's deliberate: RecogGreetingTask
holds all the orchestration state, calls into this module for "how do I
say / interpret X", and never has an English literal in its body.

Organised into three sections:

1. Parsers — decide what a transcript means:
     * has_intro_phrase(text) -> bool
     * parse_name(text, require_intro_phrase=False) -> str
     * parse_confirmation(text) -> (verdict, optional[new_name])

2. Name canonicalization — the one place that decides how a raw name
   becomes a filesystem-safe key + a human-readable display form:
     * canonicalize(name_raw) -> (dir_name, display_name)

3. Phrase generators — everything Pella says. Grouped so localisation
   (or A/B copy tweaks) touches this file and only this file:
     * intro_prompt(), name_apology(), clarity_apology()
     * confirmation_prompt(display_name)
     * greet_phrase(display_name), welcome_phrase(display_name)
     * reject_phrase(display_name), correction_ack(display_name)
     * intro_warm_phrases(display_name)   — pre-cache before enrol
     * correction_warm_phrases(display_name)
     * static_warm_phrases()              — startup pre-cache
     * per_person_warm_phrases(dir_name)  — one per enrolled person
"""

import re


# ── 1. Parsers ──────────────────────────────────────────────────────────────

# Words that a name-token filter should reject. If parse_name pulls any
# of these out as a "name", the transcript is almost certainly not a
# name-introduction — e.g. "Sorry", "Yes", "Hmm", or a state reply like
# "fine" that would follow "I'm" without any actual name.
_NON_NAME_WORDS = {
    # Pronouns
    "i", "me", "my", "mine", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers",
    "it", "its", "we", "us", "our", "they", "them", "their",
    "this", "that", "these", "those",
    # Prepositions / particles
    "for", "to", "from", "with", "of", "in", "on", "at", "by", "about",
    # Auxiliary verbs
    "is", "am", "are", "was", "were", "be",
    "do", "does", "did", "have", "has", "had",
    # Connectives / articles
    "and", "or", "but", "the", "a", "an", "not",
    # Question words
    "what", "where", "when", "who", "why", "how", "which",
    # Interjections / common responses
    "yes", "no", "ok", "okay", "yeah", "nah", "uh", "um", "oh",
    "hmm", "huh", "ugh", "eh", "mm", "mmm", "hm",
    "hi", "hey", "hello", "bye",
    # Polite / imperative words seen in mis-captures
    "please", "thanks", "thank", "sorry",
    "stop", "hold", "wait", "excuse", "tell", "give", "take",
    # Common state replies after "I'm X" / "It's X" / "That's X" — would
    # otherwise be mis-parsed as a name correction by the intro-phrase
    # branch. Expanded to cover state words that follow the weaker
    # "it's"/"that's" pointers added to INTRO_RE.
    "fine", "good", "great", "well", "alright", "tired", "busy",
    "happy", "sad", "lost", "here", "back", "home", "ready",
    "cool", "nice", "hot", "warm", "cold", "bad", "right", "wrong",
    "true", "false", "late", "early", "easy", "hard", "fun", "old",
    "new", "big", "small", "first", "last",
}


# Recognises the leading phrase in a name introduction. Matches the
# common English shapes:
#   "my name is X" / "my name's X" / "name's X"
#   "I am X" / "I'm X" / "I'm called X"
#   "call me X"
#   "this is X"
#   "it's X" / "it is X" / "that's X" / "that is X"   (weak proper-noun
#     pointers that Whisper often substitutes for "my name is X")
_INTRO_RE = re.compile(
    r"(?:i am|i'm|i'?m\s+called|my\s+name\s+is|my\s+name'?s|name'?s|"
    r"call\s+me|this\s+is|it'?s|it\s+is|that'?s|that\s+is)\s+(.+)",
    re.IGNORECASE,
)

# Yes / no patterns for the "Did you say {name}?" confirmation reply.
# Anchored to start because we want the user's leading word to be the
# affirmation/rejection — "yes, joy" / "no, joy" — not a buried token.
_YES_RE = re.compile(
    r"^(?:yes|yeah|yep|yup|sure|right|correct|that'?s\s+right|"
    r"that\s+is\s+right|ok|okay)\b",
    re.IGNORECASE,
)
_NO_RE = re.compile(
    r"^(?:no|nope|nah|wrong|incorrect|not\s+quite|that'?s\s+wrong)\b(.*)",
    re.IGNORECASE,
)


def has_intro_phrase(text: str) -> bool:
    """True iff the transcript contains an explicit name-intro phrase.

    Used to decide whether a parsed name needs confirmation. A bare-word
    transcript ("Joy" -> "Enjoy" via Whisper) goes through "Did you say X?"
    because it's the hallucination failure mode; "My name is Joy" goes
    straight to enrollment because it carries enough acoustic context for
    Whisper to commit confidently.
    """
    return _INTRO_RE.search(text) is not None


def parse_confirmation(text: str):
    """Parse a yes / no / correction reply to "Did you say {name}?".

    Returns a tuple (verdict, new_name):
      verdict   : "yes" | "no" | "unclear"
      new_name  : a name parsed out of a "no, X" / "it's Y" / "my name
                  is Z" reply, or None.

    "yes" → commit the originally-parsed name.
    "no" with new_name → commit the new name instead.
    "no" without new_name → re-prompt for the name (consumes a retry slot).
    "unclear" → ignore and keep waiting; tick() handles the timeout.
    """
    t = re.sub(r"[.,!?]+", " ", text.strip())
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ("unclear", None)

    # Check "no" first, because "no, joy" should NOT be confused with a
    # bare-name parse of "joy" alone — the leading "no" is the signal.
    no_match = _NO_RE.match(t)
    if no_match:
        rest = no_match.group(1).strip(" ,.;:-")
        if rest:
            new_name = parse_name(rest)
            if new_name:
                return ("no", new_name)
        return ("no", None)

    if _YES_RE.match(t):
        return ("yes", None)

    # No explicit yes/no leading word — treat any successfully-parsed name
    # as an implicit correction ("Joy." / "It's Joy." / "My name is Joy.").
    new_name = parse_name(t)
    if new_name:
        return ("no", new_name)

    return ("unclear", None)


def parse_name(text: str, *, require_intro_phrase: bool = False) -> str:
    """Extract a name from a casual reply, ignoring courtesy phrases.

    Returns an empty string when the input doesn't look like a name — e.g.
    contains common non-name words like 'excuse me' or 'hold it for me'.
    Case is preserved from the input; ``canonicalize()`` is the one place
    that decides the display-name capitalisation.

    When `require_intro_phrase=True`, only matches with an explicit
    "my name is X" / "I am X" / "call me X" / etc. — bare names are
    rejected. Used for *corrections* to a stored name: a casual reply
    like "I'm fine, thanks" right after a greeting shouldn't accidentally
    rename the person.
    """
    t = text.strip()
    # Collapse ellipses ("..." or longer) to a single space before splitting
    # on sentence punctuation. Whisper emits "..." for trailing-off speech
    # ("My name is...") and the bare first-sentence split would otherwise
    # cut the utterance at the first dot — losing the actual name that
    # arrives after a stitch from the next clip.
    t = re.sub(r"\.{2,}", " ", t)
    t = re.split(r"[.!?]", t, maxsplit=1)[0].strip()
    # Loop because a stitched transcript can look like "my name is ... my
    # name is Joy" — peeling off only the first intro phrase would leave
    # "my name is Joy" and trip the non-name-word filter on "my". Strip
    # all intro phrases iteratively until none remain.
    matched_any = False
    while True:
        m = _INTRO_RE.search(t)
        if not m:
            break
        matched_any = True
        t = m.group(1).strip()
    if not matched_any and require_intro_phrase:
        return ""
    courtesy = (
        r",|\s+and\b|"
        r"\s+(?:nice|glad|pleased|happy)\s+(?:to\s+)?(?:meet|meeting)\b"
    )
    t = re.split(courtesy, t, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    words = re.findall(r"[A-Za-z]+", t)
    if not words:
        return ""
    if any(w.lower() in _NON_NAME_WORDS for w in words):
        return ""
    return " ".join(words)


# ── 2. Name canonicalization ────────────────────────────────────────────────

def canonicalize(name_raw: str):
    """Return ``(dir_name, display_name)`` for a raw name string.

    dir_name    : filesystem-safe key used under data/face_ids/ and as
                  the identity's canonical id ("george_bush").
    display_name: human-readable form used in all TTS phrases and
                  on-screen labels ("George Bush").

    Both derive from the same input so the pair stays consistent
    everywhere; the one caller that starts from a ``dir_name`` (e.g.
    an already-enrolled person the recognizer just returned) uses
    ``display_name_from_dir_name()`` below instead.
    """
    dir_name     = name_raw.lower().replace(" ", "_")
    display_name = name_raw.title()
    return dir_name, display_name


def display_name_from_dir_name(dir_name: str) -> str:
    """Convert a stored ``dir_name`` back to its display form.

    Used when Pella is about to greet an already-enrolled person and only
    has the recognizer's dir-name key on hand — e.g. after a successful
    tally verdict, or in ``get_warm_phrases`` iterating over
    ``known_names()``.
    """
    return dir_name.replace("_", " ").title()


# ── 3. Phrase generators ────────────────────────────────────────────────────

def intro_prompt() -> str:
    """Opening question to an unknown face."""
    return "Hello, I am Pella. What is your name?"


def name_apology() -> str:
    """After a listening window closed without a parseable name."""
    return "Sorry, I didn't catch your name."


def clarity_apology() -> str:
    """When the best face captured during RECOGNIZING was too blurry to enrol."""
    return "Sorry, I cannot see you clearly."


def confirmation_prompt(display_name: str) -> str:
    """Ask the user to confirm a bare-name parse before enrolment."""
    return f"Did you say {display_name}?"


def greet_phrase(display_name: str) -> str:
    """Greeting an already-enrolled person."""
    return f"Hi, {display_name}"


def welcome_phrase(display_name: str) -> str:
    """Successful enrolment — the "Nice to meet you" that closes the intro."""
    return f"Nice to meet you, {display_name}!"


def reject_phrase(display_name: str) -> str:
    """Enrolment failed after the user gave a name — e.g. the embedding
    for the name already existed under a different face and the recognizer
    refused the new template."""
    return (f"You don't look like the {display_name} I know. "
            f"Sorry about that.")


def correction_ack(display_name: str) -> str:
    """Acknowledge a rename during the CORRECTION_WINDOW after a greet or
    introduce. Uses "Sorry, X" to mirror the way people apologise for
    getting a name wrong the first time."""
    return f"Sorry, {display_name}. Got it."


def intro_warm_phrases(display_name: str):
    """Pre-cache list for the intro flow.

    Kick this off right after Pella asks for the name so the "Nice to meet
    you" / "You don't look like the X" TTS is generated and uploaded in
    parallel with the user speaking, and playback fires instantly when
    the enrolment result lands.
    """
    return (
        welcome_phrase(display_name),
        reject_phrase(display_name),
    )


def correction_warm_phrases(display_name: str):
    """Pre-cache list after a rename — the next greeting will likely
    use the new name, so warm both greet + welcome forms."""
    return (
        greet_phrase(display_name),
        welcome_phrase(display_name),
    )


def static_warm_phrases():
    """Phrases that Pella says regardless of who's in front of her.

    Used by ``pella_main`` at startup to pre-generate + upload each WAV
    to the audiohub so first-time gen+upload latency doesn't show up
    mid-interaction.
    """
    return [
        intro_prompt(),
        name_apology(),
        clarity_apology(),
    ]


def per_person_warm_phrases(dir_name: str):
    """Per-name warmup pair for a currently-enrolled person.

    "Hi, X" fires often on repeated recognition; "Nice to meet you, X!"
    caches for the rare case where a person was un-enrolled and re-
    enrolled under the same name in the same session.
    """
    display = display_name_from_dir_name(dir_name)
    return [greet_phrase(display), welcome_phrase(display)]
