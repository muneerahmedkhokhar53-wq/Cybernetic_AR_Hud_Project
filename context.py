"""Per-frame context object handed to panels and modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Context:
    app: Any
    cfg: Any
    theme: Any
    t: float = 0.0              # seconds since start
    dt: float = 0.033           # seconds since previous frame
    fps: float = 0.0
    size: tuple = (1280, 720)   # (width, height)
    perception: Any = None
    hands: dict = field(default_factory=dict)   # key -> HandState
    two_hand: Any = None
    events: list = field(default_factory=list)
    base: Any = None            # graded camera frame (panel backgrounds are darkened here)
    overlay: Any = None         # additive HUD layer (everything bright is drawn here)

    @property
    def panels(self):
        return self.app.panels

    def toast(self, text: str, color=None, duration: float = 1.8) -> None:
        self.app.add_toast(text, color, duration)
