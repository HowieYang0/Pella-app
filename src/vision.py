#!/usr/bin/env python3
"""Face detection and recognition pipeline."""

import os
import threading
from queue import Queue, Empty

import cv2
import numpy as np

# ── Detection / display constants ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.65
MIN_FACE_RATIO       = 0.05   # face width must be at least this fraction of frame width
DETECT_W, DETECT_H   = 640, 360
FACE_DETECT_EVERY    = 5      # run detector every N frames
ZOOM_PADDING         = 0.6    # padding around face when zooming in
ZOOM_COOLDOWN        = 10.0   # seconds between zoom events
GREET_COOLDOWN       = 10.0   # seconds between greetings for the same person

MOTION_THRESHOLD     = 25     # per-pixel diff threshold (0–255)
MOTION_MIN_AREA      = 0.02   # fraction of frame pixels that must change
MOTION_COOLDOWN      = 5.0    # seconds before motion can retrigger look_up
SEEK_TIMEOUT         = 4.0    # max seconds to wait for a sharp face before escalating
LOOK_UP_HOLD_TIME    = 1.5    # min seconds to stay in look_up before recovery allowed
SIT_HOLD_TIME        = 2.0    # min seconds to stay in sit_look_up (longer to settle)
RECOVERY_SHARPNESS   = 50.0   # captured face must be at least this sharp to trigger recovery

INTRODUCE_COOLDOWN   = 30.0   # seconds before asking an unknown face again
CORRECTION_WINDOW    = 10.0   # seconds after Pella greets/introduces a person
                              # during which a transcript starting with an
                              # explicit intro phrase ("my name is X") is
                              # treated as a name correction and triggers a
                              # rename on disk + in the recognizer.
ENROLL_LISTEN_WINDOW = 10.0   # seconds after intro during which the user is
                              # expected to start speaking their name. A
                              # transcript whose VAD speech-start timestamp
                              # falls inside this window counts; later
                              # speech does not.
ENROLL_LOOKBACK_SEC  = 3.0    # accept speech started up to this many seconds
                              # *before* intro. Users often anticipate the
                              # question and start answering as the prompt is
                              # still playing — without this allowance,
                              # transcripts with speech_start_t just before
                              # asked_at would be rejected even though they
                              # are the real answer. _parse_name's strict
                              # mode still requires "my name is X" / "I am X"
                              # so random pre-question chatter won't pass.
ENROLL_TIMEOUT       = 30.0   # absolute deadline after which enrollment is
                              # abandoned even if no transcript arrived.
                              # Larger than ENROLL_LISTEN_WINDOW to absorb
                              # the worst-case Whisper transcription latency
                              # for a user who spoke just before the window
                              # closed: ~10s of capture + ~6s of MAX_SPEECH
                              # buffer + ~10s of CPU-Whisper = ~26s, so 30
                              # gives a small safety margin.
SHARPNESS_THRESHOLD  = 80.0   # min Laplacian variance for an enrollment-quality face

# ── Model paths ────────────────────────────────────────────────────────────────
# _DIR is .../pella_app/src/. Model checkpoints + enrollment data live under
# the repo's data/ tree (sibling of src/), so model & face paths resolve
# through ../data/.
_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(_DIR, "..", "data")
MODELS_DIR   = os.path.join(DATA_DIR, "models")
YUNET_PATH   = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
_PROTO       = os.path.join(MODELS_DIR, "deploy.prototxt")
_WEIGHTS     = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
FACE_IDS_DIR = os.path.join(DATA_DIR, "face_ids")

# ── Detector initialisation ───────────────────────────────────────────────────
if os.path.exists(YUNET_PATH) and hasattr(cv2, "FaceDetectorYN"):
    _detector      = cv2.FaceDetectorYN.create(
        YUNET_PATH, "", (DETECT_W, DETECT_H), CONFIDENCE_THRESHOLD, 0.3, 5000
    )
    _detector_type = "yunet"
    print("Face detector: YuNet", flush=True)
elif os.path.exists(_PROTO) and os.path.exists(_WEIGHTS):
    _detector      = cv2.dnn.readNetFromCaffe(_PROTO, _WEIGHTS)
    _detector_type = "dnn"
    print("Face detector: DNN (SSD)", flush=True)
else:
    _detector      = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    _detector_type = "haar"
    print("Face detector: Haar cascade", flush=True)


# ── Motion detection ──────────────────────────────────────────────────────────

def detect_motion(frame: np.ndarray, prev_frame: np.ndarray) -> bool:
    """Return True if significant motion is detected between two consecutive frames."""
    if prev_frame is None or frame.shape != prev_frame.shape:
        return False
    g1 = cv2.GaussianBlur(cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY), (5, 5), 0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, thresh = cv2.threshold(cv2.absdiff(g1, g2), MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    return np.count_nonzero(thresh) / thresh.size > MOTION_MIN_AREA


# ── Image quality ─────────────────────────────────────────────────────────────

def sharpness(bgr: np.ndarray) -> float:
    """Return Laplacian variance — higher means a sharper (less blurry) image."""
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── Person / body detection ───────────────────────────────────────────────────
PERSON_DETECT_W = 400  # resize width for HOG speed

_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

_lowerbody = None
_lowerbody_path = cv2.data.haarcascades + "haarcascade_lowerbody.xml"
if os.path.exists(_lowerbody_path):
    _lowerbody = cv2.CascadeClassifier(_lowerbody_path)
    print("Body detector: HOG full-body + Haar lower-body", flush=True)
else:
    print("Body detector: HOG full-body only", flush=True)


def detect_person(bgr: np.ndarray) -> bool:
    """Return True if a human body (full or lower) is detected in the frame."""
    h, w = bgr.shape[:2]
    scale = PERSON_DETECT_W / w if w > PERSON_DETECT_W else 1.0
    small = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else bgr
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    try:
        rects, _ = _hog.detectMultiScale(
            gray, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        if len(rects) > 0:
            return True
    except Exception:
        pass

    if _lowerbody is not None:
        legs = _lowerbody.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 60)
        )
        if len(legs) > 0:
            return True

    return False


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_faces(bgr: np.ndarray) -> list:
    """Return a list of (x, y, w, h, landmarks_or_None) tuples."""
    h, w = bgr.shape[:2]

    if _detector_type == "yunet":
        small = cv2.resize(bgr, (DETECT_W, DETECT_H))
        _detector.setInputSize((DETECT_W, DETECT_H))
        _, dets = _detector.detect(small)
        if dets is None:
            return []
        sx, sy = w / DETECT_W, h / DETECT_H
        faces = []
        for d in dets:
            fx = max(0, int(d[0] * sx))
            fy = max(0, int(d[1] * sy))
            fw = min(int(d[2] * sx), w - fx)
            fh = min(int(d[3] * sy), h - fy)
            if fw > 0 and fh > 0 and fw >= w * MIN_FACE_RATIO:
                lm = d[4:14].reshape(5, 2).copy()
                lm[:, 0] *= sx
                lm[:, 1] *= sy
                faces.append((fx, fy, fw, fh, lm.astype(np.float32)))
        return faces

    if _detector_type == "dnn":
        blob = cv2.dnn.blobFromImage(
            cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        _detector.setInput(blob)
        detections = _detector.forward()
        faces = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf > CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1 and (x2 - x1) >= w * MIN_FACE_RATIO:
                    faces.append((x1, y1, x2 - x1, y2 - y1, None))
        return faces

    # Haar fallback
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dets = _detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=8, minSize=(60, 60)
    )
    return [(x, y, fw, fh, None) for (x, y, fw, fh) in dets] if len(dets) > 0 else []


# ── Annotation and crop helpers ───────────────────────────────────────────────

def annotate(bgr: np.ndarray, faces: list, names: list = None) -> np.ndarray:
    """Draw bounding boxes and optional name labels onto a copy of the frame."""
    out = bgr.copy()
    for i, face in enumerate(faces):
        x, y, fw, fh = face[:4]
        cv2.rectangle(out, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        label = names[i] if names and i < len(names) and names[i] else None
        if label:
            cv2.putText(out, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return out


EDGE_MARGIN_PX = 5


def is_face_at_edge(face, img_w: int, img_h: int, margin: int = EDGE_MARGIN_PX) -> bool:
    """Return True if the face bbox is flush against any image edge (within `margin` px).

    A face touching the frame border is likely cropped — part of it lies outside
    the camera's field of view, so recognition will be unreliable and the robot
    should keep adjusting its pose until the face is fully framed.
    """
    x, y, fw, fh = face[:4]
    return (x <= margin
            or y <= margin
            or x + fw >= img_w - margin
            or y + fh >= img_h - margin)


def zoom_crop(bgr: np.ndarray, face: tuple) -> np.ndarray:
    """Return a padded crop of the frame centred on the given face."""
    img_h, img_w = bgr.shape[:2]
    x, y, fw, fh = face[:4]
    pad_x = int(fw * ZOOM_PADDING)
    pad_y = int(fh * ZOOM_PADDING)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + fw + pad_x)
    y2 = min(img_h, y + fh + pad_y)
    return bgr[y1:y2, x1:x2]


# ── Recognition ───────────────────────────────────────────────────────────────

def load_recognizer():
    """Load the ArcFace recognizer. Returns None if unavailable."""
    try:
        from face_recognizer import FaceRecognizer
        model_path   = os.path.join(MODELS_DIR, "w600k_r50.onnx")
        face_ids_dir = FACE_IDS_DIR
        if not os.path.exists(model_path):
            print("ArcFace model not found — recognition disabled", flush=True)
            return None
        print("Loading ArcFace recognizer...", flush=True)
        rec = FaceRecognizer(model_path, face_ids_dir, yunet_path=YUNET_PATH)
        print("ArcFace recognizer ready", flush=True)
        return rec
    except ImportError:
        print("onnxruntime not available — recognition disabled", flush=True)
        return None
    except Exception as e:
        print(f"Recognizer load error: {e}", flush=True)
        return None


def recognition_worker(recognizer, rec_in: Queue, rec_out: Queue,
                        stop_event: threading.Event):
    """Thread target: read (image, faces) pairs, write (faces, names) bundles.

    Bundling faces with names keeps the consumer immune to index drift if
    last_faces is updated by a newer detection while recognition is in flight.
    """
    while not stop_event.is_set():
        try:
            img, faces = rec_in.get(timeout=0.1)
            names = []
            for face in faces:
                lm = face[4] if len(face) > 4 else None
                name, _ = recognizer.recognize(img, lm)
                names.append(name)
            rec_out.put((faces, names))
        except Empty:
            continue
