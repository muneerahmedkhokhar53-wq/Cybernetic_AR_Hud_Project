"""DRAW mode - draw in the air with your index finger. PEACE sign clears the canvas."""
from __future__ import annotations

import cv2
import numpy as np

from . import HUDModule
from ..draw import AA, P, put_text


class DrawModule(HUDModule):
    name = "DRAW"
    show_face = False
    show_pose = False
    show_objects = False

    def __init__(self):
        self.canvas = None
        self.pos: dict = {}
        self.color_idx = 0
        self._last_clear = -10.0

    def _palette(self, theme) -> list:
        return [theme.secondary, theme.primary, theme.alert, (255, 255, 255)]

    def on_enter(self, ctx) -> None:
        ctx.toast("AIR DRAW: POINT TO DRAW - PEACE CLEARS - B CHANGES COLOUR")

    def update(self, ctx) -> None:
        w, h = ctx.size
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), np.uint8)
        color = self._palette(ctx.theme)[self.color_idx % 4]
        for key in list(self.pos):
            if key not in ctx.hands:
                del self.pos[key]
        for key, st in ctx.hands.items():
            if st.gesture == "POINT" and not ctx.panels.busy(key):
                pt = np.array(st.index_tip)
                prev = self.pos.get(key)
                if prev is not None:
                    pt = prev * 0.6 + pt * 0.4
                    cv2.line(self.canvas, P(prev), P(pt), color, 5, AA)
                self.pos[key] = pt
            else:
                self.pos.pop(key, None)
            if st.gesture == "PEACE" and ctx.t - self._last_clear > 1.0:
                self.canvas[:] = 0
                self._last_clear = ctx.t
                ctx.toast("CANVAS CLEARED")

    def draw(self, ctx) -> None:
        if self.canvas is not None:
            cv2.add(ctx.overlay, self.canvas, dst=ctx.overlay)
        color = self._palette(ctx.theme)[self.color_idx % 4]
        for st in ctx.hands.values():
            if st.gesture == "POINT":
                c = P(st.index_tip)
                cv2.circle(ctx.overlay, c, 12, color, 2, AA)
                cv2.circle(ctx.overlay, c, 3, (255, 255, 255), -1, AA)
        put_text(ctx.overlay, "AIR DRAW  //  POINT = DRAW   PEACE = CLEAR   B = COLOUR   X = CLEAR",
                 (ctx.size[0] // 2, ctx.size[1] - 40), ctx.theme.secondary, 0.5, 1, "center")

    def on_key(self, ctx, key: int) -> bool:
        if key == ord("b"):
            self.color_idx += 1
            return True
        if key == ord("x"):
            if self.canvas is not None:
                self.canvas[:] = 0
            return True
        return False
