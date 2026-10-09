"""Perception worker thread: hands, face mesh (+analysis), pose, objects, face identity."""
from __future__ import annotations

import math
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# ----------------------------------------------------------------------------- data
@dataclass
class HandData:
    landmarks: np.ndarray        # (21, 3) normalised x, y, z
    label: str                   # "Left" / "Right" (selfie view)
    score: float = 1.0


@dataclass
class FaceData:
    pts: np.ndarray              # (N, 3) normalised landmarks
    bbox: tuple                  # (x1, y1, x2, y2) in pixels
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    ear: float = 0.3             # eye aspect ratio
    blink: bool = False
    mouth_open: float = 0.0
    smile: float = 0.0
    gaze: tuple = (0.0, 0.0)     # (x, y) offsets, roughly -0.5..0.5
    gaze_label: str = "CENTER"
    expression: str = "NEUTRAL"
    identity: Optional[str] = None


@dataclass
class Detection:
    label: str
    conf: float
    bbox: tuple                  # (x1, y1, x2, y2) pixels


@dataclass
class Perception:
    hands: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    pose: Optional[np.ndarray] = None    # (33, 4) x, y, z, visibility
    objects: list = field(default_factory=list)
    latency_ms: float = 0.0
    frame_id: int = -1
    shape: tuple = (720, 1280)           # (h, w) of the processed frame
    timestamp: float = 0.0


# ----------------------------------------------------------------------------- face analysis
_R_EYE = [33, 160, 158, 133, 153, 144]
_L_EYE = [362, 385, 387, 263, 373, 380]


def _d(a, b) -> float:
    return float(np.linalg.norm(a - b))


def _ear(p: np.ndarray, idx) -> float:
    a, b, c, d, e, f = (p[i] for i in idx)
    return (_d(b, f) + _d(c, e)) / (2.0 * _d(a, d) + 1e-6)


def analyze_face(pts: np.ndarray, shape) -> FaceData:
    """Geometric face analysis from the 468/478 face-mesh landmarks (all estimates)."""
    h, w = shape
    p = pts[:, :2] * np.array([w, h], dtype=np.float32)
    x1, y1 = np.clip(p.min(axis=0), 0, [w - 1, h - 1])
    x2, y2 = np.clip(p.max(axis=0), 0, [w - 1, h - 1])

    face_w = _d(p[234], p[454]) + 1e-6
    mid_x = (p[234][0] + p[454][0]) / 2.0
    yaw = float(np.clip((p[1][0] - mid_x) / face_w * 140.0, -90, 90))

    eye_mid_y = (p[33][1] + p[263][1]) / 2.0
    face_h = p[152][1] - eye_mid_y + 1e-6
    ratio = (p[1][1] - eye_mid_y) / face_h
    pitch = float(np.clip((0.42 - ratio) * 200.0, -60, 60))

    roll = math.degrees(math.atan2(p[263][1] - p[33][1], p[263][0] - p[33][0]))

    ear = (_ear(p, _R_EYE) + _ear(p, _L_EYE)) / 2.0
    mouth_w = _d(p[61], p[291]) + 1e-6
    mouth_open = _d(p[13], p[14]) / mouth_w
    smile = mouth_w / face_w

    gaze, gaze_label = (0.0, 0.0), "CENTER"
    if len(p) > 473:
        def rx(iris, a, b):
            return (p[iris][0] - p[a][0]) / (p[b][0] - p[a][0] + 1e-6)

        def ry(iris, top, bot):
            return (p[iris][1] - p[top][1]) / (p[bot][1] - p[top][1] + 1e-6)

        gx = (rx(468, 33, 133) + rx(473, 362, 263)) / 2.0 - 0.5
        gy = (ry(468, 159, 145) + ry(473, 386, 374)) / 2.0 - 0.5
        gaze = (float(gx), float(gy))
        if gx < -0.07:
            gaze_label = "LEFT"
        elif gx > 0.07:
            gaze_label = "RIGHT"
        elif gy > 0.12:
            gaze_label = "DOWN"
        elif gy < -0.12:
            gaze_label = "UP"

    if mouth_open > 0.45:
        expression = "SURPRISED"
    elif smile > 0.47:
        expression = "HAPPY"
    elif ear < 0.17:
        expression = "EYES CLOSED"
    else:
        expression = "NEUTRAL"

    return FaceData(
        pts=pts, bbox=(int(x1), int(y1), int(x2), int(y2)), yaw=yaw, pitch=pitch, roll=roll,
        ear=float(ear), blink=bool(ear < 0.19), mouth_open=float(mouth_open), smile=float(smile),
        gaze=gaze, gaze_label=gaze_label, expression=expression,
    )


# ----------------------------------------------------------------------------- engine
class PerceptionEngine(threading.Thread):
    def __init__(self, camera, cfg):
        super().__init__(daemon=True, name="perception")
        self.camera = camera
        self.cfg = cfg
        self.error = ""
        self.ready = threading.Event()
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._result = Perception()
        self._objects: list = []
        self.hands = self.face = self.pose = self.yolo = self.face_id = None

    # -- public
    def get(self) -> Perception:
        with self._lock:
            return self._result

    def stop(self) -> None:
        self._stop_evt.set()

    # -- setup
    def _init_models(self) -> None:
        import mediapipe as mp  # imported here so a missing install shows on the HUD

        p = self.cfg.perception
        if p.hands:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=p.max_hands, model_complexity=1,
                min_detection_confidence=p.min_detection_confidence,
                min_tracking_confidence=p.min_tracking_confidence)
        if p.face_mesh:
            self.face = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=p.max_faces, refine_landmarks=True,
                min_detection_confidence=p.min_detection_confidence,
                min_tracking_confidence=p.min_tracking_confidence)
        if p.pose:
            self.pose = mp.solutions.pose.Pose(
                model_complexity=p.pose_complexity,
                min_detection_confidence=p.min_detection_confidence,
                min_tracking_confidence=p.min_tracking_confidence)
        if p.object_detection:
            try:
                from ultralytics import YOLO

                self.yolo = YOLO(p.yolo_model)
            except Exception as exc:  # optional feature
                print(f"[perception] YOLO disabled: {exc}")
        if p.face_recognition:
            try:
                from .face_id import FaceIdentifier

                self.face_id = FaceIdentifier(p.faces_dir)
            except Exception as exc:  # optional feature
                print(f"[perception] face recognition disabled: {exc}")

    # -- loop
    def run(self) -> None:
        try:
            self._init_models()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            return
        self.ready.set()
        last_stamp, cycle = -1, 0
        while not self._stop_evt.is_set():
            frame, stamp = self.camera.read()
            if frame is None or stamp == last_stamp:
                time.sleep(0.002)
                continue
            last_stamp = stamp
            cycle += 1
            try:
                result = self._process(frame, stamp, cycle)
            except Exception:
                traceback.print_exc()
                time.sleep(0.05)
                continue
            with self._lock:
                self._result = result

    def _process(self, frame, stamp: int, cycle: int) -> Perception:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        scale = self.cfg.perception.process_scale
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 0.999 else frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        out = Perception(frame_id=stamp, shape=(h, w), timestamp=time.time())

        if self.hands is not None:
            res = self.hands.process(rgb)
            if res.multi_hand_landmarks:
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    arr = np.array([[q.x, q.y, q.z] for q in lm.landmark], dtype=np.float32)
                    cls = hd.classification[0]
                    out.hands.append(HandData(arr, cls.label, float(cls.score)))

        if self.face is not None:
            res = self.face.process(rgb)
            if res.multi_face_landmarks:
                for lm in res.multi_face_landmarks:
                    arr = np.array([[q.x, q.y, q.z] for q in lm.landmark], dtype=np.float32)
                    out.faces.append(analyze_face(arr, (h, w)))
            if self.face_id is not None and out.faces:
                self.face_id.update(frame, out.faces, time.time())

        if self.pose is not None:
            res = self.pose.process(rgb)
            if res.pose_landmarks:
                out.pose = np.array([[q.x, q.y, q.z, q.visibility] for q in res.pose_landmarks.landmark],
                                    dtype=np.float32)

        if self.yolo is not None:
            if cycle % max(1, self.cfg.perception.yolo_every) == 0:
                self._objects = self._run_yolo(frame)
            out.objects = list(self._objects)

        out.latency_ms = (time.perf_counter() - t0) * 1000.0
        return out

    def _run_yolo(self, frame) -> list:
        res = self.yolo.predict(frame, imgsz=480, conf=self.cfg.perception.yolo_conf, verbose=False)[0]
        names = res.names
        found = []
        for box in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            found.append(Detection(str(names[int(box.cls[0])]).upper(), float(box.conf[0]), (x1, y1, x2, y2)))
        return found
