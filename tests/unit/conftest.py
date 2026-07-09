"""Shared pytest setup for the unit test suite.

Two responsibilities:

1. Put ``src/`` on sys.path so tests can ``import perception`` and
   ``import recog_greeting`` directly, matching how the running app
   imports them (via the ``src/`` working directory).

2. Fake the ``face_recognizer`` module before any test imports it.
   The real face_recognizer pulls in onnxruntime + the ArcFace model
   at import time — heavy runtime deps that live only on the dock.
   The fake gives just enough surface (``next_image_index``) for
   Perception's fallback-save-raw-frames branch to be exercisable.
"""

import pathlib
import sys
import types


# Put src/ on the path — matches how pella_main.py imports its siblings.
_SRC = pathlib.Path(__file__).parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# Stub face_recognizer BEFORE perception tries to import it. The stub only
# has to expose the surface Perception actually calls: next_image_index.
if "face_recognizer" not in sys.modules:
    _fake_fr = types.ModuleType("face_recognizer")

    def _next_image_index(dir_path):
        """Fake: pretend the directory is empty so tests get index 1."""
        return 1

    _fake_fr.next_image_index = _next_image_index
    sys.modules["face_recognizer"] = _fake_fr


# Stub actions BEFORE recog_greeting tries to import it. actions.py pulls
# in the go2_webrtc_driver package which only exists on the dock. The
# tests we care about (parsers, scoring, tally logic) never actually
# invoke actions.* — recog_greeting just needs the name to resolve.
if "actions" not in sys.modules:
    sys.modules["actions"] = types.ModuleType("actions")
