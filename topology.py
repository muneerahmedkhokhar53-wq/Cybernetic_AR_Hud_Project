"""Landmark connection topology (lazy so the HUD can start without MediaPipe)."""
from __future__ import annotations

_CACHE = None

_NAMES = {
    "face_oval": "FACEMESH_FACE_OVAL",
    "lips": "FACEMESH_LIPS",
    "left_eye": "FACEMESH_LEFT_EYE",
    "right_eye": "FACEMESH_RIGHT_EYE",
    "left_brow": "FACEMESH_LEFT_EYEBROW",
    "right_brow": "FACEMESH_RIGHT_EYEBROW",
}


def get_topology() -> dict:
    global _CACHE
    if _CACHE is None:
        topo = {k: [] for k in _NAMES}
        try:
            import mediapipe as mp

            face_mesh = mp.solutions.face_mesh
            for key, attr in _NAMES.items():
                topo[key] = sorted(getattr(face_mesh, attr, []))
        except Exception:
            pass
        _CACHE = topo
    return _CACHE
