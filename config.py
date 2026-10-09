"""Configuration: built-in defaults, optional YAML overrides, attribute access."""
from __future__ import annotations

import copy
import os

try:
    import yaml
except ImportError:  # PyYAML is optional; defaults are used without it
    yaml = None

DEFAULTS: dict = {
    "camera": {"index": 0, "width": 1280, "height": 720, "mirror": True},
    "perception": {
        "hands": True,
        "max_hands": 2,
        "face_mesh": True,
        "max_faces": 2,
        "pose": True,
        "pose_complexity": 0,
        "object_detection": False,   # needs: pip install ultralytics
        "yolo_model": "yolov8n.pt",
        "yolo_every": 3,             # run YOLO every N perception cycles
        "yolo_conf": 0.45,
        "face_recognition": False,   # needs: pip install face_recognition (+ enrolled faces)
        "faces_dir": "faces",
        "process_scale": 0.6,        # downscale before inference (speed)
        "min_detection_confidence": 0.6,
        "min_tracking_confidence": 0.5,
    },
    "gestures": {
        "model_path": "models/gesture_model.pkl",
        "use_model": True,
        "model_min_conf": 0.8,
        "swipe_distance": 0.28,      # fraction of frame width/height
        "swipe_time": 0.45,          # seconds
        "swipe_cooldown": 0.9,
        "pinch_threshold": 0.35,     # thumb-index distance / hand size
        "calibration_file": "calibration.json",
    },
    "ui": {
        "window_name": "CYBERNETIC AR HUD",
        "theme": "cyan",
        "fullscreen": False,
        "scanlines": True,
        "glow": True,
        "vignette": True,
        "glitch": True,
        "boot_sequence": True,
        "dim": 0.72,
        "show_mesh": True,
    },
    "voice": {"enabled": False, "language": "en-US"},
    "recording": {"dir": "captures", "fps": 30},
    "plugins": {"dir": "plugins"},
}


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


class Cfg:
    """Thin wrapper so nested dicts can be read/written as cfg.section.key."""

    __slots__ = ("_d",)

    def __init__(self, data: dict):
        object.__setattr__(self, "_d", data)

    def __getattr__(self, name: str):
        data = object.__getattribute__(self, "_d")
        try:
            value = data[name]
        except KeyError:
            raise AttributeError(name) from None
        return Cfg(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value) -> None:
        self._d[name] = value

    def to_dict(self) -> dict:
        return copy.deepcopy(self._d)


def load_config(path: str | None = "config.yaml") -> Cfg:
    data = copy.deepcopy(DEFAULTS)
    if path and os.path.exists(path):
        if yaml is None:
            print("[config] PyYAML not installed - using defaults")
        else:
            with open(path, "r", encoding="utf-8") as fh:
                _merge(data, yaml.safe_load(fh) or {})
    return Cfg(data)
