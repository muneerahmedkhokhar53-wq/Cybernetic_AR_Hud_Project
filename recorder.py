"""Screenshot + video recording of the composed HUD output."""
from __future__ import annotations

import os
import time

import cv2


class Recorder:
    def __init__(self, directory: str = "captures", fps: int = 30):
        self.dir = directory
        self.fps = fps
        self.writer = None
        self._t0 = 0.0

    @property
    def recording(self) -> bool:
        return self.writer is not None

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0 if self.recording else 0.0

    def _path(self, prefix: str, ext: str) -> str:
        os.makedirs(self.dir, exist_ok=True)
        return os.path.join(self.dir, f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.{ext}")

    def toggle(self, size) -> str:
        """Start/stop recording. `size` is (width, height). Returns a status message."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            return "RECORDING SAVED"
        path = self._path("hud", "mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, tuple(size))
        if not writer.isOpened():
            return "RECORDER ERROR"
        self.writer = writer
        self._t0 = time.time()
        return "RECORDING STARTED"

    def write(self, frame) -> None:
        if self.writer is not None:
            self.writer.write(frame)

    def screenshot(self, frame) -> str:
        path = self._path("shot", "png")
        cv2.imwrite(path, frame)
        return path

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
