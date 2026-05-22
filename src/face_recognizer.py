#!/usr/bin/env python3
"""ArcFace (ResNet-50) face recognition with YuNet landmark alignment."""

import os
import cv2
import numpy as np
import onnxruntime as ort

# 5-point reference landmarks for 112x112 ArcFace canonical crop
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

_ENROLL_W, _ENROLL_H = 640, 360


def align_face(bgr_frame: np.ndarray, landmarks_5x2: np.ndarray) -> np.ndarray:
    """Affine-warp face region from bgr_frame to a 112x112 canonical crop."""
    src = landmarks_5x2.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, _ARCFACE_DST)
    if M is None:
        return cv2.resize(bgr_frame, (112, 112))
    return cv2.warpAffine(bgr_frame, M, (112, 112))


def _yunet_detector(model_path: str):
    if not os.path.exists(model_path) or not hasattr(cv2, "FaceDetectorYN"):
        return None
    return cv2.FaceDetectorYN.create(model_path, "", (_ENROLL_W, _ENROLL_H), 0.5, 0.3, 5000)


def _detect_face_landmarks(det, bgr: np.ndarray):
    """Return 5x2 landmarks of the largest face, or None."""
    h, w = bgr.shape[:2]
    scale = min(_ENROLL_W / w, _ENROLL_H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(bgr, (nw, nh))
    det.setInputSize((nw, nh))
    _, dets = det.detect(small)
    if dets is None:
        return None
    best = max(dets, key=lambda d: d[2] * d[3])
    inv = 1.0 / scale
    lm = best[4:14].reshape(5, 2) * inv
    return lm.astype(np.float32)


def next_image_index(save_dir: str) -> int:
    """Return the next available 1-based image index for a face_ids/<person>/
    folder.

    The folder convention is zero-padded three-digit `NNN.jpg` files (plus
    matching `NNN.npy` for embeddings when produced by FaceRecognizer).
    Used by both the main enrollment path and the no-recognizer fallback in
    recog_greeting so multiple captures for the same person never collide.
    """
    if not os.path.isdir(save_dir):
        return 1
    max_idx = 0
    for f in os.listdir(save_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in (".jpg", ".jpeg", ".png"):
            try:
                max_idx = max(max_idx, int(stem))
            except ValueError:
                pass
    return max_idx + 1


class FaceRecognizer:
    def __init__(self, model_path: str, face_ids_dir: str,
                 yunet_path: str = "", threshold: float = 0.35,
                 enroll_threshold: float = 0.50):
        self.threshold        = threshold
        self.enroll_threshold = enroll_threshold
        self.face_ids_dir     = face_ids_dir  # needed by rename()
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.known: dict = {}  # name -> list of 512-dim embeddings
        self._enroll(face_ids_dir, yunet_path)

    def _embed(self, aligned_112: np.ndarray) -> np.ndarray:
        """Return L2-normalised 512-dim embedding from a 112x112 BGR image."""
        rgb = aligned_112[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
        inp = rgb.transpose(2, 0, 1)[np.newaxis]  # NCHW
        emb = self.session.run(None, {self.input_name: inp})[0][0]
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def _enroll(self, face_ids_dir: str, yunet_path: str):
        if not os.path.isdir(face_ids_dir):
            print(f"[FaceRecognizer] face_ids dir not found: {face_ids_dir}", flush=True)
            return

        det = _yunet_detector(yunet_path) if yunet_path else None

        for name in sorted(os.listdir(face_ids_dir)):
            person_dir = os.path.join(face_ids_dir, name)
            if not os.path.isdir(person_dir):
                continue
            embeddings = []
            files = sorted(os.listdir(person_dir))
            # Stems that have a precomputed .npy — skip re-embedding those .jpg files.
            npy_stems = {os.path.splitext(f)[0]
                         for f in files if f.lower().endswith(".npy")}

            for fname in files:
                stem, ext = os.path.splitext(fname)
                ext = ext.lower()
                path = os.path.join(person_dir, fname)
                if ext == ".npy":
                    try:
                        embeddings.append(np.load(path))
                    except Exception as e:
                        print(f"[FaceRecognizer] failed to load {path}: {e}",
                              flush=True)
                elif ext in (".jpg", ".jpeg", ".png") and stem not in npy_stems:
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    aligned = None
                    if det is not None:
                        lm = _detect_face_landmarks(det, img)
                        if lm is not None:
                            aligned = align_face(img, lm)
                    if aligned is None:
                        aligned = cv2.resize(img, (112, 112))
                    embeddings.append(self._embed(aligned))

            if embeddings:
                self.known[name] = embeddings
                print(f"[FaceRecognizer] enrolled '{name}' "
                      f"({len(embeddings)} embedding(s))", flush=True)

        print(f"[FaceRecognizer] {len(self.known)} person(s) ready", flush=True)

    def enroll_new(self, name: str, bgr_frame: np.ndarray,
                   landmarks_5x2: np.ndarray = None, face_bbox: tuple = None,
                   save_dir: str = "") -> bool:
        """Add a new person live and persist a face crop + .npy embedding.

        Alignment / embedding always use the full frame + landmarks (most accurate).
        Only a padded face crop is saved to disk; the .npy lets future loads skip
        detection and alignment entirely.

        If a person with this name already exists, the new embedding must be
        similar enough (cosine ≥ self.threshold) to the existing embeddings,
        otherwise the save is rejected and False is returned.
        """
        aligned = (align_face(bgr_frame, landmarks_5x2)
                   if landmarks_5x2 is not None
                   else cv2.resize(bgr_frame, (112, 112)))
        emb = self._embed(aligned)

        if name in self.known:
            best_sim = max(float(np.dot(emb, e)) for e in self.known[name])
            if best_sim < self.enroll_threshold:
                print(f"[FaceRecognizer] rejecting enrollment for '{name}': "
                      f"similarity {best_sim:.2f} < {self.enroll_threshold:.2f} "
                      f"(face doesn't match existing photos)", flush=True)
                return False
            self.known[name].append(emb)
        else:
            self.known[name] = [emb]

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            idx = next_image_index(save_dir)

            if face_bbox is not None:
                x, y, fw, fh = face_bbox[:4]
                ih, iw = bgr_frame.shape[:2]
                pad_x = int(fw * 0.3)
                pad_y = int(fh * 0.3)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(iw, x + fw + pad_x)
                y2 = min(ih, y + fh + pad_y)
                crop = bgr_frame[y1:y2, x1:x2]
            else:
                crop = bgr_frame

            jpg_path = os.path.join(save_dir, f"{idx:03d}.jpg")
            npy_path = os.path.join(save_dir, f"{idx:03d}.npy")
            cv2.imwrite(jpg_path, crop)
            np.save(npy_path, emb)
            print(f"[FaceRecognizer] saved {jpg_path} + .npy", flush=True)

        print(f"[FaceRecognizer] enrolled '{name}' "
              f"({len(self.known[name])} embedding(s))", flush=True)
        return True

    def rename(self, old_name: str, new_name: str) -> bool:
        """Rename a person on disk and in memory.

        If `new_name` already exists, merges `old_name`'s files into it
        with next-index numbering (preserves jpg/npy pairs). Otherwise
        does a plain directory rename. Returns True on success.

        Used by recog_greeting when the user corrects a mis-heard name
        ("My name is William" after Pella said "Hi, Willie").
        """
        if old_name == new_name:
            return False

        old_dir = os.path.join(self.face_ids_dir, old_name)
        new_dir = os.path.join(self.face_ids_dir, new_name)

        if os.path.isdir(old_dir):
            if not os.path.isdir(new_dir):
                os.rename(old_dir, new_dir)
            else:
                # Merge: copy each NNN.jpg + NNN.npy pair into new_dir at
                # next available index. Files are processed stem-by-stem
                # so a .jpg/.npy pair stays together under the same new
                # number.
                next_idx = next_image_index(new_dir)
                files_by_stem: dict = {}
                for fname in os.listdir(old_dir):
                    stem, ext = os.path.splitext(fname)
                    if ext.lower() in (".jpg", ".jpeg", ".png", ".npy"):
                        files_by_stem.setdefault(stem, []).append(fname)
                for stem in sorted(files_by_stem.keys()):
                    for fname in files_by_stem[stem]:
                        _, ext = os.path.splitext(fname)
                        os.rename(
                            os.path.join(old_dir, fname),
                            os.path.join(new_dir, f"{next_idx:03d}{ext.lower()}"),
                        )
                    next_idx += 1
                # Best-effort cleanup of the now-empty old dir.
                try:
                    for fname in os.listdir(old_dir):
                        os.remove(os.path.join(old_dir, fname))
                    os.rmdir(old_dir)
                except OSError as e:
                    print(f"[FaceRecognizer] rename: could not remove "
                          f"{old_dir}: {e}", flush=True)

        # In-memory map.
        if old_name in self.known:
            if new_name in self.known:
                self.known[new_name].extend(self.known.pop(old_name))
            else:
                self.known[new_name] = self.known.pop(old_name)

        print(f"[FaceRecognizer] renamed '{old_name}' -> '{new_name}' "
              f"({len(self.known.get(new_name, []))} embedding(s))",
              flush=True)
        return True

    def recognize(self, bgr_frame: np.ndarray,
                  landmarks_5x2: np.ndarray = None) -> tuple:
        """
        Identify a face in bgr_frame.
        landmarks_5x2: YuNet 5-point landmarks in frame coordinates (preferred).
        Returns (name_or_None, cosine_score).
        """
        if landmarks_5x2 is not None:
            aligned = align_face(bgr_frame, landmarks_5x2)
        else:
            aligned = cv2.resize(bgr_frame, (112, 112))

        emb = self._embed(aligned)
        best_name, best_score = None, -1.0
        for name, embs in self.known.items():
            score = max(float(np.dot(emb, e)) for e in embs)
            if score > best_score:
                best_score, best_name = score, name

        if best_score >= self.threshold:
            return best_name, best_score
        return None, best_score
