#!/usr/bin/env python3
"""Make Pella say something."""

import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.vui.vui_client import VuiClient
from audio.audio_output import sanitize_for_tts, convert_to_voice

VOLUME_LEVEL = 8  # 1-10


def speak(text: str, network_interface: str = None):
    if network_interface:
        ChannelFactoryInitialize(0, network_interface)
    else:
        ChannelFactoryInitialize(0)

    vui = VuiClient()
    vui.SetTimeout(3.0)
    vui.Init()
    code = vui.SetVolume(VOLUME_LEVEL)
    if code != 0:
        print(f"Warning: could not set VUI volume (code {code})")

    convert_to_voice(sanitize_for_tts(text))


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello! I am Pella, your robot dog."
    interface = sys.argv[2] if len(sys.argv) > 2 else None
    speak(text, interface)
