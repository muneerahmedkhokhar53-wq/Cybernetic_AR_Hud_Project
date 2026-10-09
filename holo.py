"""HOLO mode - manipulate a 3D wireframe hologram with one or two hands."""
from __future__ import annotations

from . import HUDModule
from ..draw import put_text
from ..widgets import Hologram


class HoloModule(HUDModule):
    name = "HOLO"
    show_face = False
    show_pose = False
    show_objects = False

    def __init__(self):
        self.holo = Hologram()

    def on_enter(self, ctx) -> None:
        ctx.toast("HOLO: PINCH+DRAG ROTATE - TWO-HAND PINCH SCALE/ROLL - PEACE SHAPE")

    def update(self, ctx) -> None:
        self.holo.update(ctx)

    def draw(self, ctx) -> None:
        self.holo.draw(ctx)
        put_text(ctx.overlay, "PINCH+DRAG: ROTATE   TWO-HAND PINCH: SCALE + ROLL   PEACE: CHANGE SHAPE",
                 (ctx.size[0] // 2, ctx.size[1] - 40), ctx.theme.secondary, 0.5, 1, "center")
