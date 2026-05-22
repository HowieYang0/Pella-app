from dataclasses import dataclass
from audio.audio_device_manager import AudioDeviceManager

_audio_config = None


@dataclass
class AudioConfig:
    input_device_index: int | None
    output_device_index: int | None
    input_rate: int
    tts_rate: int
    stt_rate: int = 16000


def build_audio_config() -> AudioConfig:
    adm = AudioDeviceManager()
    adm.refresh_devices()

    mic = adm.set_preferred_input_device()
    spk = adm.set_preferred_output_device()

    if mic is None:
        raise RuntimeError("No input audio device found")
    if spk is None:
        raise RuntimeError("No output audio device found")

    input_rate = adm.pick_best_io_rate(mic)
    output_rate = adm.pick_best_io_rate(spk)
    return AudioConfig(
        input_device_index=mic.index,
        output_device_index=spk.index,
        input_rate=input_rate,
        tts_rate=output_rate,
    )


def get_audio_config():
    global _audio_config
    if _audio_config is None:
        _audio_config = build_audio_config()
    return _audio_config
