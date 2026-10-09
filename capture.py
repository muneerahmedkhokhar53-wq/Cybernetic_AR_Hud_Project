"""Threaded camera capture - always exposes the newest frame, never blocks the UI."""
from __future__ import annotations

import os
import threading
import time

import cv2


class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, mirror: bool = True):
        self.index = index
        self.width = width
        self.height = height
        self.mirror = mirror
        self.cap = None
        self.error = ""
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0
        self._running = False
        self._thread = None

    def start(self) -> "Camera":
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.index, backend)
        if not self.cap.isOpened():
            self.error = f"Cannot open camera {self.index}"
            return self
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera")
        self._thread.start()
        return self

    def _loop(self) -> None:
        failures = 0
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                failures += 1
                if failures > 100:
                    self.error = "Camera stream lost"
                time.sleep(0.01)
                continue
            failures = 0
            self.error = ""
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame
                self._stamp += 1

    def read(self):
        """Return (frame, stamp). Frames are never modified in place by consumers."""
        with self._lock:
            return self._frame, self._stamp

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
