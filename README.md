# Pella

Pella is an interactive companion built on the Unitree Go2 quadruped. It uses
the robot's front camera and an external USB microphone to detect people,
recognise faces it has seen before, learn the names of new people, and
respond to short spoken interactions — all over the Go2's WebRTC channel.

```
+-------------------+
|     pella_main    |   sense-respond brain; owns the WebRTC connection,
|     (brain)       |   queues, and threads. Has zero knowledge of tasks.
+-------------------+
          |
+-------------------+
|   task_manager    |   single place that knows about concrete tasks;
+-------------------+   today it dispatches to recog_greeting.
          |
+-------------------+    +----------+----------+----------+----------+
|   recog_greeting  |    | front_   |   stt    |   tts    | actions  |
|       (task)      |    | camera   |  (ear)   | (mouth)  | (limbs)  |
+-------------------+    +----------+----------+----------+----------+
```

* **`pella_main`** runs the display loop, owns the WebRTC peer connection,
  and threads transcripts / frames / actions / TTS between the organs.
* **`task_manager`** dispatches per-iteration ticks to whichever task is
  currently active. Adding a new task only touches this module.
* **`recog_greeting`** is the only task today: detect → recognise → greet,
  or detect → introduce (ask name) → enrol.
* **`front_camera`** receives H.264 frames from the Go2.
* **`stt`** captures the USB mic, runs WebRTC VAD + faster-whisper, and
  stamps each transcript with the speech-start time.
* **`tts`** generates speech via gTTS, uploads to the Go2 AudioHub, and
  plays through the robot's speakers.
* **`actions`** drives motor primitives (look_up, sit, wiggle, …).
* **`face_recognizer`** wraps ArcFace + YuNet for enrolment and recognition.

## Repository layout

```
pella_app/
├── README.md
├── requirements.txt
├── pella-camera.service       # systemd unit for headless deployment
├── .env                       # secrets, never committed
├── src/                       # production code
│   ├── pella_main.py
│   ├── task_manager.py
│   ├── recog_greeting.py
│   ├── front_camera.py
│   ├── stt.py
│   ├── tts.py
│   ├── actions.py
│   ├── chat.py
│   ├── vision.py
│   └── face_recognizer.py
├── tests/                     # tests + exploratory / archived scripts
├── data/
│   ├── face_ids/              # one folder per enrolled person (jpg + npy)
│   └── models/                # ML model checkpoints
└── .venv/                     # local virtualenv (gitignored)
```

## Prerequisites

* **Hardware** — Unitree Go2 quadruped on the same network as the host
  machine. The host is whichever computer runs `pella_main.py` (a laptop on
  the dev network, or the robot's onboard Jetson for headless operation).
* **Mic** — a USB microphone attached to the host. Pella uses local STT,
  so the Go2's onboard mic is not used.
* **Speaker** — playback uses the Go2's onboard speaker (via AudioHub),
  not the host.
* **Python 3.8+** with `pip` and `venv`.
* **System packages**:
  ```bash
  sudo apt install \
    portaudio19-dev libsndfile1 ffmpeg \
    libgl1 libglib2.0-0
  ```

## Installation

```bash
git clone <repo-url> pella_app
cd pella_app

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Models

The small YuNet face detector (`face_detection_yunet_2023mar.onnx`) is
already tracked under `data/models/`. The larger ArcFace recognition
checkpoint is *not* in the repo — download it once:

```bash
# w600k_r50.onnx (~174 MB) — ArcFace ResNet-50, used by face_recognizer.py
curl -L -o data/models/w600k_r50.onnx \
  https://github.com/deepinsight/insightface/releases/download/v0.7/w600k_r50.onnx
```

If the model is missing at runtime, `vision.load_recognizer()` logs
`ArcFace model not found — recognition disabled` and Pella falls back to
detection-only mode (no recognition, no enrolment).

## Configuration

Create `.env` in the repo root for any secrets used by the WebRTC SDK:

```bash
# Example .env — adjust to your Go2's network setup
PELLA_GO2_IP=192.168.123.161
```

## Running

### Locally, against a Go2 on the LAN

```bash
source .venv/bin/activate
python src/pella_main.py 192.168.123.161
```

The first argument is the Go2's IP address. The program opens a pygame
window showing the front camera, captures from the USB mic, and speaks via
the Go2's onboard speaker.

### As a systemd service on the robot

The `pella-camera.service` unit ships with the repo. To install on the
robot's Jetson:

```bash
# On the robot:
sudo cp pella-camera.service /etc/systemd/system/pella-camera.service
sudo systemctl daemon-reload
sudo systemctl enable --now pella-camera

# Watch the log:
journalctl -u pella-camera -f
```

The unit assumes the repo is at `/home/pella/Development/pella/pella_app/`
and the virtualenv is `.venv/` inside it. If your paths differ, edit
`WorkingDirectory=` and `ExecStart=` in the service file.

### Restarting after a code change

```bash
sudo systemctl restart pella-camera
```

The systemd journal (`journalctl -u pella-camera`) is the canonical place
to read Pella's behaviour — the program logs every state transition,
TTS playback, ASR result, and recognition event with a timestamp.

## Adding a person without using the live enrolment flow

Drop one or more face crops under `data/face_ids/<name>/`:

```bash
mkdir -p data/face_ids/alice
cp /path/to/face1.jpg data/face_ids/alice/001.jpg
cp /path/to/face2.jpg data/face_ids/alice/002.jpg
```

Restart Pella — at startup `FaceRecognizer._enroll` walks every
sub-folder of `data/face_ids/`, embeds each `.jpg` it doesn't already have
a paired `.npy` for, and caches the embedding next to the source image
(`001.jpg` → `001.npy`).

## Correcting a mis-heard name

Pella has a 10-second correction window after every greeting and every
enrolment. If it addresses you wrong:

> Pella: "Hi, Willie."
> You: "My name is William."
> Pella: "Sorry, William. Got it."

…and `data/face_ids/willie/` is renamed to `data/face_ids/william/`.
If a `william/` folder already exists, the captures are merged with
next-index numbering instead of overwriting.

Only explicit intro phrasings (`my name is X`, `I am X`, `call me X`,
`this is X`) trigger a rename. Bare names and state replies like
"I'm fine" are ignored to prevent accidental renames.

## Development

### Running an ad-hoc test script

```bash
source .venv/bin/activate
python tests/test_actions.py        # smoke-test motor primitives
python tests/test_mic.py            # USB mic RMS / VAD sanity check
```

### Common debugging recipes

```bash
# Tail the journal with TTS / ASR lines only
journalctl -u pella-camera -f | grep -E "TTS|ASR|Heard|enrolled|Task event"

# Watch VAD floor adapt (LiDAR / background noise tuning)
journalctl -u pella-camera -f | grep "USB mic:"

# Confirm a specific TTS clip is being uploaded vs cached
journalctl -u pella-camera -f | grep "TTS \[.* gen_wav\|upload\|playing\]"
```

## License

Internal / not yet licensed.
