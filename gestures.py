"""Gesture engine.

* static gestures   - rule based, or a trained classifier (train_gesture.py) when available
* pinch             - with hysteresis + start/release edges
* swipes            - open-palm trajectory analysis
* two-hand pinch    - scale / rotation deltas for manipulating virtual objects
"""
from __future__ import annotations

import json
import math
import os
import pickle
from collections import deque
from dataclasses import dataclass, field

import numpy as np

HAND_TIPS = (4, 8, 12, 16, 20)
FINGER_JOINTS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}  # (pip, tip)


def landmark_features(lm: np.ndarray) -> np.ndarray:
    """Translation / scale invariant feature vector (63 values) for the classifier."""
    pts = lm[:, :3] - lm[0, :3]
    scale = float(np.linalg.norm(pts[9, :2])) + 1e-6
    return (pts / scale).flatten().astype(np.float32)


@dataclass
class GestureEvent:
    name: str
    hand: str = ""
    time: float = 0.0


@dataclass
class HandState:
    key: str
    label: str
    landmarks: np.ndarray
    px: np.ndarray                 # (21, 2) pixel coordinates
    size: float                    # wrist -> middle-MCP distance in pixels
    fingers: dict
    gesture: str = "UNKNOWN"
    confidence: float = 1.0
    pinch_ratio: float = 1.0
    pinching: bool = False
    pinch_started: bool = False
    pinch_released: bool = False

    @property
    def pinch_point(self) -> tuple:
        mid = (self.px[4] + self.px[8]) / 2.0
        return float(mid[0]), float(mid[1])

    @property
    def index_tip(self) -> tuple:
        return float(self.px[8][0]), float(self.px[8][1])

    @property
    def palm_center(self) -> tuple:
        c = self.px[[0, 5, 9, 13, 17]].mean(axis=0)
        return float(c[0]), float(c[1])


@dataclass
class TwoHandState:
    active: bool = False
    center: tuple = (0.0, 0.0)
    dist: float = 0.0
    angle: float = 0.0
    scale_ratio: float = 1.0
    d_angle: float = 0.0


class GestureEngine:
    def __init__(self, cfg):
        self.cfg = cfg.gestures
        self.pinch_threshold = float(self.cfg.pinch_threshold)
        self.model = None
        self.hands: dict = {}
        self.events: list = []
        self.two_hand = TwoHandState()
        self._pinch_prev: dict = {}
        self._traj: dict = {}
        self._last_swipe = 0.0
        self._th_prev = None
        self._load_model()
        self._load_calibration()

    # ------------------------------------------------------------------ setup
    def _load_model(self) -> None:
        path = self.cfg.model_path
        if not (self.cfg.use_model and os.path.exists(path)):
            return
        try:
            with open(path, "rb") as fh:
                self.model = pickle.load(fh)["model"]
            print(f"[gestures] loaded classifier: {list(self.model.classes_)}")
        except Exception as exc:
            print(f"[gestures] could not load classifier: {exc}")

    def _load_calibration(self) -> None:
        path = self.cfg.calibration_file
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self.pinch_threshold = float(json.load(fh)["pinch_threshold"])
                print(f"[gestures] pinch threshold {self.pinch_threshold:.2f} (calibrated)")
            except Exception:
                pass

    def set_pinch_threshold(self, value: float) -> None:
        self.pinch_threshold = float(min(max(value, 0.12), 0.8))
        try:
            with open(self.cfg.calibration_file, "w", encoding="utf-8") as fh:
                json.dump({"pinch_threshold": self.pinch_threshold}, fh)
        except OSError:
            pass

    # ------------------------------------------------------------------ main update
    def update(self, perception, now: float) -> dict:
        self.events = []
        h, w = perception.shape
        states: dict = {}
        for hd in perception.hands:
            key = hd.label
            if key in states:
                key += "2"
            st = self._analyze(key, hd, (w, h))
            prev = self._pinch_prev.get(key, False)
            limit = self.pinch_threshold * (1.25 if prev else 1.0)
            on = st.pinch_ratio < limit
            st.pinching = on
            st.pinch_started = on and not prev
            st.pinch_released = prev and not on
            if on:
                st.gesture = "PINCH"
            self._pinch_prev[key] = on
            self._track_swipe(st, (w, h), now)
            states[key] = st
        for key in list(self._pinch_prev):
            if key not in states:
                del self._pinch_prev[key]
                self._traj.pop(key, None)
        self.hands = states
        self._update_two_hand(states)
        return states

    # ------------------------------------------------------------------ analysis
    def _analyze(self, key, hd, shape) -> HandState:
        w, h = shape
        lm = hd.landmarks
        px = lm[:, :2] * np.array([w, h], dtype=np.float32)
        size = float(np.linalg.norm(px[0] - px[9])) + 1e-6
        fingers = self._fingers(px)
        pinch_ratio = float(np.linalg.norm(px[4] - px[8])) / size
        gesture, conf = self._classify(lm, fingers, px, size)
        return HandState(key=key, label=hd.label, landmarks=lm, px=px, size=size, fingers=fingers,
                         gesture=gesture, confidence=conf, pinch_ratio=pinch_ratio)

    @staticmethod
    def _fingers(px: np.ndarray) -> dict:
        wrist = px[0]
        ext = {}
        for name, (pip, tip) in FINGER_JOINTS.items():
            ext[name] = bool(np.linalg.norm(px[tip] - wrist) > np.linalg.norm(px[pip] - wrist) * 1.08)
        ext["thumb"] = bool(np.linalg.norm(px[4] - px[17]) > np.linalg.norm(px[3] - px[17]) * 1.05)
        return ext

    def _classify(self, lm, fingers, px, size):
        if self.model is not None:
            try:
                proba = self.model.predict_proba(landmark_features(lm)[None, :])[0]
                i = int(np.argmax(proba))
                if proba[i] >= self.cfg.model_min_conf:
                    return str(self.model.classes_[i]).upper(), float(proba[i])
            except Exception:
                pass
        return self._rules(fingers, px, size), 1.0

    @staticmethod
    def _rules(f: dict, px: np.ndarray, size: float) -> str:
        n = sum(f[k] for k in ("index", "middle", "ring", "pinky"))
        if n == 4:
            return "OPEN_PALM"
        if n == 0:
            if f["thumb"] and px[4][1] < px[2][1] and px[4][1] < px[0][1] - 0.7 * size:
                return "THUMBS_UP"
            return "FIST"
        if n == 1 and f["index"]:
            return "POINT"
        if n == 2 and f["index"] and f["middle"]:
            return "PEACE"
        if n == 2 and f["index"] and f["pinky"]:
            return "ROCK"
        if n == 3:
            return "THREE"
        return "UNKNOWN"

    # ------------------------------------------------------------------ swipes
    def _track_swipe(self, st: HandState, shape, now: float) -> None:
        w, h = shape
        dq = self._traj.setdefault(st.key, deque())
        cx, cy = st.palm_center
        dq.append((now, cx / w, cy / h))
        while dq and now - dq[0][0] > self.cfg.swipe_time:
            dq.popleft()
        if (st.pinching or st.gesture != "OPEN_PALM" or len(dq) < 4
                or now - self._last_swipe < self.cfg.swipe_cooldown):
            return
        dx, dy = dq[-1][1] - dq[0][1], dq[-1][2] - dq[0][2]
        dist = self.cfg.swipe_distance
        name = None
        if abs(dx) >= dist and abs(dy) < 0.6 * abs(dx):
            name = "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
        elif abs(dy) >= dist * 1.1 and abs(dx) < 0.6 * abs(dy):
            name = "SWIPE_DOWN" if dy > 0 else "SWIPE_UP"
        if name:
            self.events.append(GestureEvent(name, st.label, now))
            self._last_swipe = now
            dq.clear()

    # ------------------------------------------------------------------ two-hand
    def _update_two_hand(self, states: dict) -> None:
        pinching = [s for _, s in sorted(states.items()) if s.pinching]
        if len(pinching) < 2:
            self.two_hand = TwoHandState()
            self._th_prev = None
            return
        a, b = pinching[0].pinch_point, pinching[1].pinch_point
        dist = math.hypot(b[0] - a[0], b[1] - a[1]) + 1e-6
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        ratio, d_angle = 1.0, 0.0
        if self._th_prev is not None:
            ratio = dist / self._th_prev[0]
            d_angle = (angle - self._th_prev[1] + math.pi) % (2 * math.pi) - math.pi
        self._th_prev = (dist, angle)
        self.two_hand = TwoHandState(True, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), dist, angle, ratio, d_angle)
