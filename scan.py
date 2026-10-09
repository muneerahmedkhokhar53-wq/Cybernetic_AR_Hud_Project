"""SCAN mode - full perception HUD (face analysis, hands, pose, objects)."""
from __future__ import annotations

from . import HUDModule
from ..draw import put_text, shade


class ScanModule(HUDModule):
    name = "SCAN"

    def draw(self, ctx) -> None:
        th, ov = ctx.theme, ctx.overlay
        w, h = ctx.size
        p = ctx.perception
        if ctx.hands:
            gest = "  ".join(s.gesture.replace("_", " ") for s in ctx.hands.values())
        else:
            gest = "NO HAND"
        bio = "FACE DETECTED" if p.faces else "NO FACE"
        put_text(ov, f"GESTURE: {gest}", (30, h - 62), th.secondary, 0.62, 2)
        put_text(ov, f"BIOMETRIC: {bio}", (30, h - 38), th.primary, 0.62, 2)
        hints = ("SHOW YOUR HAND GESTURES:", "> OPEN PALM  : OPEN OVERLAYS", "> PINCH + DRAG: MOVE PANELS",
                 "> SWIPE L/R  : CHANGE MODE", "> PRESS C    : CALIBRATE PINCH")
        for i, s in enumerate(hints):
            put_text(ov, s, (w // 2 - 130, h - 120 + i * 17), shade(th.secondary, 0.75 if i else 1.0), 0.42)
