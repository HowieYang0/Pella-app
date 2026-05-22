import pyaudio
import platform
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json


@dataclass
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    host_api: int
    is_default_input: bool
    is_default_output: bool
    device_type: str  # 'builtin', 'usb', 'bluetooth', 'hdmi', 'unknown'


class AudioDeviceManager:
    def __init__(self):
        self.pa = None
        self.devices = []
        self._initialize_pyaudio()

    def _initialize_pyaudio(self):
        if self.pa is None:
            self.pa = pyaudio.PyAudio()

    def _detect_device_type(self, device_info: dict, index: int) -> str:
        name = device_info['name'].lower()
        system = platform.system().lower()

        if system == 'darwin':
            if any(x in name for x in ['built-in', 'macbook', 'imac', 'mac mini']):
                return 'builtin'
            elif 'usb' in name:
                return 'usb'
            elif 'bluetooth' in name:
                return 'bluetooth'
            elif 'displayport' in name or 'hdmi' in name:
                return 'hdmi'
        elif system == 'linux':
            if any(x in name for x in ['bcm2835', 'vc4-hdmi', 'alsa', 'pulse']):
                if 'hdmi' in name:
                    return 'hdmi'
                elif 'usb' in name:
                    return 'usb'
                elif 'bluetooth' in name:
                    return 'bluetooth'
                else:
                    return 'builtin'
        elif system == 'windows':
            if any(x in name for x in ['speakers', 'realtek', 'high definition audio']):
                return 'builtin'
            elif 'usb' in name:
                return 'usb'
            elif 'bluetooth' in name:
                return 'bluetooth'
            elif 'hdmi' in name or 'display' in name:
                return 'hdmi'

        if 'usb' in name:
            return 'usb'
        elif 'bluetooth' in name or 'bt' in name:
            return 'bluetooth'
        elif 'hdmi' in name or 'display' in name:
            return 'hdmi'
        else:
            return 'unknown'

    def refresh_devices(self):
        self._initialize_pyaudio()
        self.devices = []

        device_count = self.pa.get_device_count()
        default_input = self.pa.get_default_input_device_info()
        default_output = self.pa.get_default_output_device_info()

        for i in range(device_count):
            try:
                device_info = self.pa.get_device_info_by_index(i)
                if device_info['maxInputChannels'] == 0 and device_info['maxOutputChannels'] == 0:
                    continue
                device = AudioDevice(
                    index=i,
                    name=device_info['name'],
                    max_input_channels=int(device_info['maxInputChannels']),
                    max_output_channels=int(device_info['maxOutputChannels']),
                    default_sample_rate=float(device_info['defaultSampleRate']),
                    host_api=int(device_info['hostApi']),
                    is_default_input=(i == default_input['index']),
                    is_default_output=(i == default_output['index']),
                    device_type=self._detect_device_type(device_info, i)
                )
                self.devices.append(device)
            except Exception as e:
                print(f"Warning: Could not get info for device {i}: {e}")
                continue

        return self.devices

    def get_output_devices(self) -> List[AudioDevice]:
        return [d for d in self.devices if d.max_output_channels > 0]

    def get_input_devices(self) -> List[AudioDevice]:
        return [d for d in self.devices if d.max_input_channels > 0]

    def find_device_by_type(self, device_type: str) -> List[AudioDevice]:
        return [d for d in self.devices if d.device_type == device_type]

    def get_default_output_device(self) -> Optional[AudioDevice]:
        for device in self.devices:
            if device.is_default_output:
                return device
        return None

    def get_default_input_device(self) -> Optional[AudioDevice]:
        for device in self.devices:
            if device.is_default_input:
                return device
        return None

    def set_preferred_output_device(self, device_type: str = None,
                                    device_name: str = None) -> Optional[AudioDevice]:
        output_devices = self.get_output_devices()
        if not output_devices:
            return None
        if device_name:
            for device in output_devices:
                if device_name.lower() in device.name.lower():
                    return device
        type_priority = ['usb', 'bluetooth', 'hdmi', 'builtin', 'unknown']
        if device_type:
            for device in output_devices:
                if device.device_type == device_type:
                    return device
        else:
            for preferred_type in type_priority:
                for device in output_devices:
                    if device.device_type == preferred_type:
                        return device
        default = self.get_default_output_device()
        return default if default else output_devices[0]

    def set_preferred_input_device(self, device_type: str = None,
                                   device_name: str = None) -> Optional[AudioDevice]:
        input_devices = self.get_input_devices()
        if not input_devices:
            return None
        if device_name:
            for device in input_devices:
                if device_name.lower() in device.name.lower():
                    return device
        type_priority = ['usb', 'bluetooth', 'builtin', 'unknown']
        if device_type:
            for device in input_devices:
                if device.device_type == device_type:
                    return device
        else:
            for preferred_type in type_priority:
                for device in input_devices:
                    if device.device_type == preferred_type:
                        return device
        default = self.get_default_input_device()
        return default if default else input_devices[0]

    def pick_best_io_rate(self, device: AudioDevice,
                          preferred_rates=(16000, 48000, 44100)) -> int:
        config = self.get_channel_configuration(device.index)
        supported = config.get("supported_rates", [])
        for r in preferred_rates:
            if r in supported:
                return r
        return int(device.default_sample_rate)

    def monitor_device_changes(self, callback=None, interval=2.0):
        previous_devices = set([d.name for d in self.devices])
        while True:
            time.sleep(interval)
            self.refresh_devices()
            current_devices = set([d.name for d in self.devices])
            added = current_devices - previous_devices
            removed = previous_devices - current_devices
            if added or removed:
                print(f"Audio devices changed!")
                if added:
                    print(f"  Added: {added}")
                if removed:
                    print(f"  Removed: {removed}")
                if callback:
                    callback(added, removed, self.devices)
            previous_devices = current_devices

    def print_device_summary(self):
        print("\n" + "=" * 60)
        print(f"AUDIO DEVICE SUMMARY ({platform.system()} {platform.machine()})")
        print("=" * 60)
        for device in self.devices:
            default_markers = []
            if device.is_default_input:
                default_markers.append("DEFAULT_IN")
            if device.is_default_output:
                default_markers.append("DEFAULT_OUT")
            default_str = " [" + ", ".join(default_markers) + "]" if default_markers else ""
            print(f"\nDevice #{device.index}: {device.name}{default_str}")
            print(f"  Type: {device.device_type.upper()}")
            print(f"  Channels: IN={device.max_input_channels}, OUT={device.max_output_channels}")
            print(f"  Sample Rate: {device.default_sample_rate} Hz")
            print(f"  Host API: {device.host_api}")

    def get_channel_configuration(self, device_index: int = None) -> Dict:
        if device_index is None:
            device = self.get_default_output_device()
            if not device:
                return {}
            device_index = device.index
        try:
            device_info = self.pa.get_device_info_by_index(device_index)
            return {
                'device_index': device_index,
                'device_name': device_info['name'],
                'max_input_channels': int(device_info['maxInputChannels']),
                'max_output_channels': int(device_info['maxOutputChannels']),
                'default_low_input_latency': device_info['defaultLowInputLatency'],
                'default_low_output_latency': device_info['defaultLowOutputLatency'],
                'default_high_input_latency': device_info['defaultHighInputLatency'],
                'default_high_output_latency': device_info['defaultHighOutputLatency'],
                'default_sample_rate': device_info['defaultSampleRate'],
                'supported_rates': self._get_supported_sample_rates(device_index)
            }
        except Exception as e:
            print(f"Error getting channel config: {e}")
            return {}

    def _get_supported_sample_rates(self, device_index: int) -> List[float]:
        test_rates = [8000, 11025, 16000, 22050, 32000, 48000, 88200, 96000, 192000]
        supported = []
        for rate in test_rates:
            try:
                stream = self.pa.open(
                    format=pyaudio.paInt16,
                    channels=2,
                    rate=rate,
                    output=True,
                    input=False,
                    frames_per_buffer=1024,
                    output_device_index=device_index,
                    start=False
                )
                stream.close()
                supported.append(rate)
            except:
                continue
        return supported

    def create_output_stream(self, device_index: int = None, **kwargs):
        if device_index is None:
            device = self.set_preferred_output_device()
            if not device:
                raise RuntimeError("No output devices available")
            device_index = device.index
        defaults = {
            'format': pyaudio.paInt16,
            'channels': 2,
            'rate': 48000,
            'output': True,
            'frames_per_buffer': 1024,
            'output_device_index': device_index
        }
        defaults.update(kwargs)
        device_info = self.pa.get_device_info_by_index(device_index)
        max_channels = int(device_info['maxOutputChannels'])
        if defaults['channels'] > max_channels:
            print(f"Warning: Device only supports {max_channels} channels, "
                  f"adjusting from {defaults['channels']}")
            defaults['channels'] = max_channels
        print(f"Creating output stream on device #{device_index}: "
              f"{device_info['name']} ({defaults['channels']} channels)")
        return self.pa.open(**defaults)

    def pick_best_sample_rate(self, device: AudioDevice, preferred=[16000, 48000]):
        config = self.get_channel_configuration(device.index)
        supported = config.get("supported_rates", [])
        for r in preferred:
            if r in supported:
                return r
        return int(device.default_sample_rate)

    def cleanup(self):
        if self.pa:
            self.pa.terminate()
            self.pa = None
