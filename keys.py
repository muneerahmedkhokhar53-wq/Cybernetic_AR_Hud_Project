"""KEYS mode - air keyboard. Pinch a key to type."""
from __future__ import annotations

from . import HUDModule
from ..widgets import VirtualKeyboard


class KeysModule(HUDModule):
    name = "KEYS"
    show_face = False
    show_pose = False
    show_objects = False
    show_panels = False

    def __init__(self):
        self.kb = VirtualKeyboard()

    def on_enter(self, ctx) -> None:
        ctx.toast("AIR KEYBOARD: PINCH A KEY TO TYPE")

    def update(self, ctx) -> None:
        self.kb.update(ctx)

    def draw(self, ctx) -> None:
        self.kb.draw(ctx)
