"""Interactive HUD widgets driven by hand gestures."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .draw import AA, P, brackets, darken, notched_rect, put_text, shade

HEADER_H = 26


# ----------------------------------------------------------------------------- panels
class Panel:
    """A translucent HUD window. Pinch its header and drag to move it."""

    def __init__(self, title: str, x: float, y: float, w: int, h: int, draw_fn, visible: bool = True):
        self.title = title
        self.x, self.y = float(x), float(y)
        self.w, self.h = int(w), int(h)
        self.draw_fn = draw_fn          # draw_fn(ctx, (x, y, w, h)) - content rect below the header
        self.visible = visible

    def header_contains(self, pt) -> bool:
        return self.x <= pt[0] <= self.x + self.w and self.y <= pt[1] <= self.y + HEADER_H

    def draw(self, ctx, grabbed: bool) -> None:
        th, ov = ctx.theme, ctx.overlay
        x, y, w, h = int(self.x), int(self.y), self.w, self.h
        darken(ctx.base, x, y, x + w, y + h, 0.30)
        col = th.secondary if grabbed else th.primary
        notched_rect(ov, x, y, x + w, y + h, col, cut=14, thick=1)
        cv2.rectangle(ov, (x + 1, y + 1), (x + w - 1, y + HEADER_H), shade(col, 0.30), -1)
        put_text(ov, self.title, (x + 10, y + 18), col, 0.5)
        for i in range(3):
            cv2.circle(ov, (x + w - 14 - i * 9, y + 13), 2, col, -1, AA)
        self.draw_fn(ctx, (x, y + HEADER_H, w, h - HEADER_H))


class PanelManager:
    """Owns panels and arbitrates which hand is busy with what (`claims`)."""

    def __init__(self):
        self.panels: list = []
        self.claims: dict = {}          # hand key -> owner (Panel, Slider, ...)
        self._offsets: dict = {}

    def add(self, panel: Panel) -> Panel:
        self.panels.append(panel)
        return panel

    def busy(self, key: str) -> bool:
        return key in self.claims

    def update(self, ctx) -> None:
        w, h = ctx.size
        for key in list(self.claims):
            st = ctx.hands.get(key)
            if st is None or not st.pinching:
                self.claims.pop(key, None)
                self._offsets.pop(key, None)
        for key, st in ctx.hands.items():
            pt = st.pinch_point
            owner = self.claims.get(key)
            if owner is None and st.pinch_started:
                for panel in reversed(self.panels):
                    if panel.visible and panel.header_contains(pt):
                        self.claims[key] = panel
                        self._offsets[key] = (pt[0] - panel.x, pt[1] - panel.y)
                        self.panels.remove(panel)       # bring to front
                        self.panels.append(panel)
                        break
            elif isinstance(owner, Panel):
                ox, oy = self._offsets[key]
                owner.x = max(0.0, min(pt[0] - ox, w - owner.w))
                owner.y = max(0.0, min(pt[1] - oy, h - owner.h))

    def draw(self, ctx) -> None:
        for panel in self.panels:
            if panel.visible:
                grabbed = any(o is panel for o in self.claims.values())
                panel.draw(ctx, grabbed)


# ----------------------------------------------------------------------------- slider
class Slider:
    def __init__(self, label: str, value: float = 0.5):
        self.label = label
        self.value = float(value)

    def run(self, ctx, x: int, y: int, w: int) -> None:
        th, ov = ctx.theme, ctx.overlay
        ty = y + 22
        for key, st in ctx.hands.items():
            px, py = st.pinch_point
            inside = x - 14 <= px <= x + w + 14 and ty - 18 <= py <= ty + 18
            if st.pinch_started and inside and key not in ctx.panels.claims:
                ctx.panels.claims[key] = self
            if ctx.panels.claims.get(key) is self and st.pinching:
                self.value = min(max((px - x) / max(w, 1), 0.0), 1.0)
        put_text(ov, self.label, (x, y + 12), shade(th.primary, 0.85), 0.42)
        put_text(ov, f"{int(self.value * 100):3d}%", (x + w, y + 12), th.secondary, 0.42, 1, "right")
        cv2.line(ov, (x, ty), (x + w, ty), shade(th.primary, 0.45), 2, AA)
        kx = int(x + self.value * w)
        cv2.line(ov, (x, ty), (kx, ty), th.secondary, 3, AA)
        cv2.circle(ov, (kx, ty), 8, th.secondary, -1, AA)
        cv2.circle(ov, (kx, ty), 12, shade(th.secondary, 0.5), 1, AA)


# ----------------------------------------------------------------------------- hologram
def _cube():
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    e = [(i, j) for i in range(8) for j in range(i + 1, 8) if int(np.sum(v[i] != v[j])) == 1]
    return v, e


def _octa():
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    e = [(i, j) for i in range(6) for j in range(i + 1, 6) if not np.allclose(v[i], -v[j])]
    return v, e


def _pyramid():
    v = np.array([[-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1], [0, -1.3, 0]], float)
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]
    return v, e


SHAPES = {"CUBE": _cube(), "OCTAHEDRON": _octa(), "PYRAMID": _pyramid()}


def _rot(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Hologram:
    """Wireframe object: pinch-drag to rotate, two-hand pinch to scale/roll, PEACE to change shape."""

    def __init__(self):
        self.names = list(SHAPES)
        self.idx = 0
        self.center = None
        self.scale = 110.0
        self.rot = np.array([0.5, 0.7, 0.0])
        self.spin = np.array([0.25, 0.4, 0.0])
        self._drag: dict = {}
        self._last_cycle = -10.0

    def update(self, ctx) -> None:
        w, h = ctx.size
        if self.center is None:
            self.center = np.array([w * 0.5, h * 0.5])
        th = ctx.two_hand
        busy = False
        if th is not None and th.active:
            self.scale = float(np.clip(self.scale * th.scale_ratio, 40, 420))
            self.rot[2] += th.d_angle
            self.center += (np.array(th.center) - self.center) * 0.35
            self._drag.clear()
            busy = True
        else:
            for key in list(self._drag):
                if key not in ctx.hands or not ctx.hands[key].pinching:
                    self._drag.pop(key, None)
            for key, st in ctx.hands.items():
                if not st.pinching:
                    continue
                p = np.array(st.pinch_point)
                if st.pinch_started and not ctx.panels.busy(key) \
                        and np.linalg.norm(p - self.center) < self.scale * 1.8:
                    self._drag[key] = p
                elif key in self._drag:
                    d = p - self._drag[key]
                    self.rot[1] += d[0] * 0.012
                    self.rot[0] += d[1] * 0.012
                    self._drag[key] = p
                    busy = True
        if not busy:
            self.rot += self.spin * ctx.dt
        for st in ctx.hands.values():
            if st.gesture == "PEACE" and ctx.t - self._last_cycle > 1.2:
                self.idx = (self.idx + 1) % len(self.names)
                self._last_cycle = ctx.t
                ctx.toast(f"SHAPE: {self.names[self.idx]}")

    def draw(self, ctx) -> None:
        th, ov = ctx.theme, ctx.overlay
        verts, edges = SHAPES[self.names[self.idx]]
        pts = verts @ _rot(*self.rot).T
        zn = pts[:, 2]
        k = 1.0 / (1.0 - 0.15 * zn)
        proj = pts[:, :2] * self.scale * k[:, None] + self.center
        for a, b in edges:
            depth = (zn[a] + zn[b]) / 2.0
            col = shade(th.primary, 0.65 + 0.3 * math.tanh(depth))
            cv2.line(ov, P(proj[a]), P(proj[b]), col, 2, AA)
        for p in proj:
            cv2.circle(ov, P(p), 4, th.secondary, -1, AA)
        cx, cy = self.center
        base_y = cy + self.scale * 1.65
        cv2.ellipse(ov, (int(cx), int(base_y)), (int(self.scale * 1.3), int(self.scale * 0.34)), 0, 0, 360,
                    shade(th.primary, 0.6), 1, AA)
        cv2.ellipse(ov, (int(cx), int(base_y)), (int(self.scale * 1.3), int(self.scale * 0.34)), 0,
                    ctx.t * 90, ctx.t * 90 + 70, th.secondary, 2, AA)
        put_text(ov, f"HOLO-OBJECT // {self.names[self.idx]}  SCALE x{self.scale / 110.0:.2f}",
                 (cx, base_y + self.scale * 0.55), th.secondary, 0.5, 1, "center")


# ----------------------------------------------------------------------------- virtual keyboard
class VirtualKeyboard:
    ROWS = ("QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM")

    def __init__(self):
        self.text = ""
        self.keys: list = []
        self.hover: set = set()
        self.flash: dict = {}
        self._last = -10.0

    def layout(self, w: int, h: int) -> list:
        kw = int(min(64, w / 15))
        gap = 8
        y0 = int(h - 4 * (kw + gap) - 70)
        keys = []
        for r, row in enumerate(self.ROWS):
            total = len(row) * (kw + gap) - gap
            x0 = (w - total) / 2
            for i, ch in enumerate(row):
                keys.append((ch, (int(x0 + i * (kw + gap)), y0 + r * (kw + gap), kw, kw)))
        specials = [("SPACE", 5 * kw + 4 * gap), ("BACK", 2 * kw + gap), ("CLEAR", 2 * kw + gap)]
        total = sum(sw for _, sw in specials) + gap * (len(specials) - 1)
        x = (w - total) / 2
        yb = y0 + 3 * (kw + gap)
        for name, sw in specials:
            keys.append((name, (int(x), yb, int(sw), kw)))
            x += sw + gap
        return keys

    def _hit(self, pt):
        for name, (x, y, kw, kh) in self.keys:
            if x <= pt[0] <= x + kw and y <= pt[1] <= y + kh:
                return name
        return None

    def update(self, ctx) -> None:
        w, h = ctx.size
        self.keys = self.layout(w, h)
        self.hover = set()
        for key, st in ctx.hands.items():
            if ctx.panels.busy(key):
                continue
            hit = self._hit(st.pinch_point)
            if hit:
                self.hover.add(hit)
                if st.pinch_started and ctx.t - self._last > 0.18:
                    self._press(hit, ctx.t)

    def _press(self, name: str, t: float) -> None:
        if name == "SPACE":
            self.text += " "
        elif name == "BACK":
            self.text = self.text[:-1]
        elif name == "CLEAR":
            self.text = ""
        else:
            self.text += name
        self.text = self.text[-40:]
        self.flash[name] = t
        self._last = t

    def draw(self, ctx) -> None:
        th, ov = ctx.theme, ctx.overlay
        w, h = ctx.size
        for name, (x, y, kw, kh) in self.keys:
            darken(ctx.base, x, y, x + kw, y + kh, 0.30)
            flashing = ctx.t - self.flash.get(name, -10.0) < 0.25
            hot = name in self.hover
            col = th.alert if flashing else (th.secondary if hot else th.primary)
            notched_rect(ov, x, y, x + kw, y + kh, col, cut=8, thick=2 if hot else 1)
            if flashing:
                cv2.rectangle(ov, (x + 3, y + 3), (x + kw - 3, y + kh - 3), shade(col, 0.45), -1)
            put_text(ov, name, (x + kw // 2, y + kh // 2 + 6), col, 0.5 if len(name) == 1 else 0.45, 1, "center")
        x0, y0 = self.keys[0][1][0], self.keys[0][1][1]
        cursor = "_" if int(ctx.t * 2) % 2 == 0 else " "
        put_text(ov, "> " + self.text + cursor, (x0, y0 - 24), th.secondary, 0.8, 2)
        put_text(ov, "PINCH A KEY TO TYPE", (w // 2, y0 - 56), shade(th.primary, 0.7), 0.45, 1, "center")
