"""Example plugin: adds a CLOCK mode and a draggable uptime panel.

Drop any *.py file with a `register(registry)` function into plugins/ and it is loaded at
start-up. `registry.app` gives access to the running app (panels, theme, perception, ...).
"""
import time

from hud.draw import put_text, shade
from hud.modules import HUDModule
from hud.widgets import Panel


class ClockModule(HUDModule):
    name = "CLOCK"
    show_face = False
    show_pose = False
    show_objects = False

    def draw(self, ctx):
        th, ov = ctx.theme, ctx.overlay
        w, h = ctx.size
        put_text(ov, time.strftime("%H:%M:%S"), (w // 2, h // 2), th.secondary, 3.0, 5, "center")
        put_text(ov, time.strftime("%A  %d %B %Y").upper(), (w // 2, h // 2 + 50), th.primary, 0.9, 2, "center")


def register(registry):
    registry.register(ClockModule())
    app = registry.app
    started = time.time()

    def uptime(ctx, rect):
        x, y, w, h = rect
        put_text(ctx.overlay, "UPTIME", (x + 10, y + 22), shade(ctx.theme.primary, 0.85), 0.45)
        put_text(ctx.overlay, time.strftime("%H:%M:%S", time.gmtime(time.time() - started)),
                 (x + w - 10, y + 22), ctx.theme.secondary, 0.55, 1, "right")

    app.panels.add(Panel("UPTIME", 350, 64, 210, 62, uptime))
