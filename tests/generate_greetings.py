#!/usr/bin/env python3
"""Generate greeting MP3 files for each person in data/face_ids/. Run on laptop."""

import os
from gtts import gTTS

_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_IDS_DIR  = os.path.join(_DIR, "..", "data", "face_ids")
GREETINGS_DIR = os.path.join(_DIR, "..", "data", "greetings")

os.makedirs(GREETINGS_DIR, exist_ok=True)

for name in sorted(os.listdir(FACE_IDS_DIR)):
    if not os.path.isdir(os.path.join(FACE_IDS_DIR, name)):
        continue
    display_name = name.replace("_", " ").title()
    text = f"Hi, {display_name}"
    out_path = os.path.join(GREETINGS_DIR, f"{name}.mp3")
    if os.path.exists(out_path):
        print(f"skip  {out_path}")
        continue
    gTTS(text).save(out_path)
    print(f"saved {out_path}  ('{text}')")
