"""Optional face identification using the `face_recognition` library.

Only people you enrolled yourself (see enroll_face.py) are recognised, everybody else is
shown as UNKNOWN. Encodings are stored locally in <faces_dir>/encodings.pkl.
Only enrol people who have given their consent.
"""
from __future__ import annotations

import math
import os
import pickle

import cv2
import numpy as np


class FaceIdentifier:
    def __init__(self, faces_dir: str = "faces", interval: float = 1.0, tolerance: float = 0.5):
        import face_recognition  # raises ImportError if not installed

        self.fr = face_recognition
        self.interval = interval
        self.tolerance = tolerance
        self.names: list = []
        self.encodings: list = []
        self._cache: list = []     # [(center, name)]
        self._last = 0.0
        path = os.path.join(faces_dir, "encodings.pkl")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                data = pickle.load(fh)
            self.names, self.encodings = data["names"], data["encodings"]
            print(f"[face_id] loaded {len(self.names)} enrolled face(s)")
        else:
            print("[face_id] no enrolled faces found - everyone will be UNKNOWN")

    @staticmethod
    def _center(bbox):
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def identify(self, rgb, boxes) -> list:
        locations = [(y1, x2, y2, x1) for (x1, y1, x2, y2) in boxes]   # top, right, bottom, left
        names = []
        for enc in self.fr.face_encodings(rgb, locations):
            if not self.encodings:
                names.append("UNKNOWN")
                continue
            dists = self.fr.face_distance(self.encodings, enc)
            i = int(np.argmin(dists))
            names.append(self.names[i].upper() if dists[i] < self.tolerance else "UNKNOWN")
        return names

    def update(self, frame_bgr, faces, now: float) -> None:
        """Assign `face.identity` to every face, re-identifying at most once per interval."""
        if not faces:
            self._cache = []
            return
        if now - self._last >= self.interval:
            self._last = now
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            boxes = [f.bbox for f in faces]
            names = self.identify(rgb, boxes)
            self._cache = [(self._center(b), n) for b, n in zip(boxes, names)]
        for face in faces:
            c = self._center(face.bbox)
            best, best_d = None, 150.0
            for cc, name in self._cache:
                d = math.hypot(c[0] - cc[0], c[1] - cc[1])
                if d < best_d:
                    best, best_d = name, d
            face.identity = best or "SCANNING"
