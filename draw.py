"""Drawing primitives, HUD elements and the final compositor (grade / glow / glitch).

Rendering model
  base     - the camera frame, colour graded (tint, vignette, scanlines). Panel backgrounds
             are darkened directly on it.
  overlay  - black image where every bright HUD element is drawn. It is added on top of the
             base and blurred at low resolution to create the neon glow.
All text is ASCII only (OpenCV's Hershey fonts do not support Unicode).
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np

from .topology import get_topology

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
HAND_TIPS = (4, 8, 12, 16, 20)
POSE_LINKS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
              (23, 25), (25, 27), (24, 26), (26, 28)]
POSE_JOINTS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


# ----------------------------------------------------------------------------- helpers
def P(p) -> tuple:
    return int(p[0]), int(p[1])


def shade(color, k: float) -> tuple:
    return tuple(int(max(0, min(255, c * k))) for c in color)


def text_width(s: str, scale: float = 0.5, thick: int = 1) -> int:
    return cv2.getTextSize(s, FONT, scale, thick)[0][0]


def put_text(img, s: str, org, color, scale: float = 0.5, thick: int = 1, align: str = "left") -> int:
    tw = text_width(s, scale, thick)
    x, y = org
    if align == "center":
        x -= tw // 2
    elif align == "right":
        x -= tw
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, color, thick, AA)
    return tw


def brackets(img, x1, y1, x2, y2, color, length: int = 16, thick: int = 2) -> None:
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    for x, y, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (x, y), (x + dx * length, y), color, thick, AA)
        cv2.line(img, (x, y), (x, y + dy * length), color, thick, AA)


def notched_rect(img, x1, y1, x2, y2, color, cut: int = 12, thick: int = 1) -> None:
    pts = np.array([(x1 + cut, y1), (x2, y1), (x2, y2 - cut), (x2 - cut, y2), (x1, y2), (x1, y1 + cut)], np.int32)
    cv2.polylines(img, [pts], True, color, thick, AA)


def darken(img, x1, y1, x2, y2, k: float = 0.35) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 > x1 and y2 > y1:
        img[y1:y2, x1:x2] = cv2.convertScaleAbs(img[y1:y2, x1:x2], alpha=k)


def bar(img, x, y, w, h, frac: float, color) -> None:
    x, y, w, h = int(x), int(y), int(w), int(h)
    cv2.rectangle(img, (x, y), (x + w, y + h), shade(color, 0.6), 1)
    fw = int((w - 2) * max(0.0, min(1.0, frac)))
    if fw > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + 1 + fw, y + h - 1), color, -1)


def sparkline(img, rect, values, color, lo: float = 0.0, hi: float = 1.0) -> None:
    x, y, w, h = (int(v) for v in rect)
    cv2.rectangle(img, (x, y), (x + w, y + h), shade(color, 0.35), 1)
    if len(values) < 2:
        return
    vals = list(values)[-w:]
    span = max(hi - lo, 1e-6)
    pts = []
    for i, v in enumerate(vals):
        px = x + int(i * (w - 1) / max(1, len(vals) - 1))
        py = y + h - 2 - int((min(max(v, lo), hi) - lo) / span * (h - 4))
        pts.append((px, py))
    cv2.polylines(img, [np.array(pts, np.int32)], False, color, 1, AA)


# ----------------------------------------------------------------------------- perception overlays
def draw_hand(ov, hs, theme, t: float) -> None:
    px = hs.px
    for a, b in HAND_LINKS:
        cv2.line(ov, P(px[a]), P(px[b]), theme.primary, 2, AA)
    for i in range(21):
        tip = i in HAND_TIPS
        cv2.circle(ov, P(px[i]), 6 if tip else 3, theme.secondary if tip else theme.primary, -1, AA)
    pulse = int(11 + 3 * math.sin(t * 6))
    for i in HAND_TIPS:
        cv2.circle(ov, P(px[i]), pulse, shade(theme.primary, 0.55), 1, AA)
    x1, y1 = px.min(axis=0) - 22
    x2, y2 = px.max(axis=0) + 22
    brackets(ov, x1, y1, x2, y2, shade(theme.secondary, 0.8), 14, 1)
    put_text(ov, f"{hs.label.upper()} HAND // {hs.gesture.replace('_', ' ')}", (x1, y1 - 8), theme.secondary, 0.45)
    col = theme.alert if hs.pinching else shade(theme.primary, 0.5)
    cv2.line(ov, P(px[4]), P(px[8]), col, 2 if hs.pinching else 1, AA)
    if hs.pinching:
        pp = hs.pinch_point
        cv2.circle(ov, P(pp), 14, theme.alert, 2, AA)
        cv2.circle(ov, P(pp), 4, (255, 255, 255), -1, AA)


def draw_face(ov, face, idx: int, theme, t: float, show_mesh: bool = True) -> None:
    h, w = ov.shape[:2]
    p = face.pts[:, :2] * np.array([w, h], dtype=np.float32)
    x1, y1, x2, y2 = face.bbox
    pad = 14
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(w - 1, x2 + pad), min(h - 1, y2 + pad)
    box_col = theme.alert

    if show_mesh:
        topo = get_topology()
        for key, k in (("face_oval", 0.7), ("lips", 0.9), ("left_eye", 0.9), ("right_eye", 0.9),
                       ("left_brow", 0.6), ("right_brow", 0.6)):
            col = shade(theme.primary if key == "face_oval" else theme.secondary, k)
            for a, b in topo[key]:
                cv2.line(ov, P(p[a]), P(p[b]), col, 1, AA)
    if len(p) > 473:
        for i in (468, 473):
            cv2.circle(ov, P(p[i]), 4, theme.secondary, 1, AA)

    brackets(ov, x1, y1, x2, y2, box_col, 22, 2)
    title = "FACIAL RECOGNITION ACTIVE" if face.identity is not None else "FACE TRACKING ACTIVE"
    put_text(ov, title, (x1, y1 - 8), box_col, 0.5)
    if face.identity is not None:
        put_text(ov, f"ID: {face.identity}", (x1, y2 + 20), theme.secondary, 0.5)

    scan_y = int(y1 + ((t * 0.7 + idx * 0.3) % 1.0) * (y2 - y1))
    cv2.line(ov, (x1, scan_y), (x2, scan_y), shade(box_col, 0.8), 1, AA)

    nx, ny = P(p[1])
    cv2.line(ov, (nx - 10, ny), (nx + 10, ny), theme.secondary, 1, AA)
    cv2.line(ov, (nx, ny - 10), (nx, ny + 10), theme.secondary, 1, AA)

    rows = [
        (f"YAW   {face.yaw:+06.1f}", theme.primary),
        (f"PITCH {face.pitch:+06.1f}", theme.primary),
        (f"ROLL  {face.roll:+06.1f}", theme.primary),
        (f"GAZE  {face.gaze_label}", theme.secondary),
        (f"EAR   {face.ear:.2f}{'  BLINK' if face.blink else ''}", theme.alert if face.blink else theme.secondary),
        (f"EXPR  {face.expression}", theme.secondary),
    ]
    right = x2 + 16 + 150 <= w
    tx = x2 + 16 if right else x1 - 16
    for i, (s, col) in enumerate(rows):
        put_text(ov, s, (tx, y1 + 16 + i * 18), col, 0.42, 1, "left" if right else "right")


def draw_pose(ov, pose, theme) -> None:
    h, w = ov.shape[:2]
    for a, b in POSE_LINKS:
        if pose[a, 3] < 0.5 or pose[b, 3] < 0.5:
            continue
        cv2.line(ov, P((pose[a, 0] * w, pose[a, 1] * h)), P((pose[b, 0] * w, pose[b, 1] * h)),
                 shade(theme.primary, 0.7), 2, AA)
    for i in POSE_JOINTS:
        if pose[i, 3] >= 0.5:
            c = P((pose[i, 0] * w, pose[i, 1] * h))
            cv2.circle(ov, c, 5, theme.secondary, -1, AA)
            cv2.circle(ov, c, 9, shade(theme.secondary, 0.5), 1, AA)


def draw_objects(ov, objects, theme) -> None:
    for o in objects:
        x1, y1, x2, y2 = o.bbox
        brackets(ov, x1, y1, x2, y2, theme.primary, 14, 2)
        put_text(ov, f"{o.label} {int(o.conf * 100)}%", (x1, y1 - 6), theme.primary, 0.45)


# ----------------------------------------------------------------------------- decorative elements
def draw_neural_net(ov, x, y, w, h, t: float, theme, layers=(4, 6, 6, 3)) -> None:
    cols = len(layers)
    xs = [x + i * w / (cols - 1) for i in range(cols)]
    nodes = [[(xs[i], y + (j + 0.5) * h / n) for j in range(n)] for i, n in enumerate(layers)]
    for i in range(cols - 1):
        for ai, a in enumerate(nodes[i]):
            for bi, b in enumerate(nodes[i + 1]):
                cv2.line(ov, P(a), P(b), shade(theme.primary, 0.28), 1, AA)
                if (ai + bi + int(t)) % 3 == 0:
                    ph = (t * 0.7 + i * 0.31 + ai * 0.13 + bi * 0.07) % 1.0
                    pt = (a[0] + (b[0] - a[0]) * ph, a[1] + (b[1] - a[1]) * ph)
                    cv2.circle(ov, P(pt), 2, theme.secondary, -1, AA)
    for i, col in enumerate(nodes):
        for j, n in enumerate(col):
            act = 0.45 + 0.55 * abs(math.sin(t * 1.7 + i * 1.1 + j * 0.7))
            cv2.circle(ov, P(n), 7, shade(theme.primary, act), -1, AA)
            cv2.circle(ov, P(n), 9, theme.primary, 1, AA)


def draw_radar(ov, cx: int, cy: int, r: int, t: float, theme, blips) -> None:
    cv2.circle(ov, (cx, cy), r, shade(theme.primary, 0.7), 1, AA)
    for k in (0.66, 0.33):
        cv2.circle(ov, (cx, cy), int(r * k), shade(theme.primary, 0.3), 1, AA)
    cv2.line(ov, (cx - r, cy), (cx + r, cy), shade(theme.primary, 0.3), 1, AA)
    cv2.line(ov, (cx, cy - r), (cx, cy + r), shade(theme.primary, 0.3), 1, AA)
    ang = (t * 1.6) % (2 * math.pi)
    for i in range(14):
        a = ang - i * 0.06
        k = 1.0 - i / 14.0
        cv2.line(ov, (cx, cy), (int(cx + r * math.cos(a)), int(cy + r * math.sin(a))),
                 shade(theme.secondary, 0.9 * k), 1, AA)
    for nx, ny, col in blips:
        dx, dy = (nx - 0.5) * 2 * r * 0.92, (ny - 0.5) * 2 * r * 0.92
        dist = math.hypot(dx, dy)
        if dist > r * 0.95:
            dx, dy = dx * r * 0.95 / dist, dy * r * 0.95 / dist
        c = (int(cx + dx), int(cy + dy))
        cv2.circle(ov, c, 4, col, -1, AA)
        cv2.circle(ov, c, 7, shade(col, 0.5), 1, AA)
    put_text(ov, "RADAR", (cx, cy - r - 8), theme.primary, 0.42, 1, "center")


# ----------------------------------------------------------------------------- compositor
class Compositor:
    """Colour grade, neon glow, scanlines, vignette and occasional glitch."""

    def __init__(self, cfg):
        self.ui = cfg.ui
        self._grade = None
        self._key = None
        self._glitch_frames = 0

    def _build_grade(self, w: int, h: int, theme) -> None:
        tint = np.array([0.55 + 0.45 * c / 255.0 for c in theme.primary], np.float32)
        grade = np.ones((h, w, 3), np.float32) * tint
        if self.ui.vignette:
            yy, xx = np.ogrid[:h, :w]
            d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
            grade *= np.clip(1.15 - 0.55 * d ** 2, 0.35, 1.0)[:, :, None]
        if self.ui.scanlines:
            grade[::3] *= 0.82
        self._grade = np.clip(grade * 255, 0, 255).astype(np.uint8)

    def grade(self, frame, theme, dim: float):
        h, w = frame.shape[:2]
        key = (w, h, theme.name, bool(self.ui.vignette), bool(self.ui.scanlines))
        if key != self._key:
            self._build_grade(w, h, theme)
            self._key = key
        base = cv2.convertScaleAbs(frame, alpha=dim, beta=-6)
        return cv2.multiply(base, self._grade, scale=1 / 255.0)

    def finish(self, base, overlay, glow_gain: float):
        out = cv2.add(base, overlay)
        if self.ui.glow and glow_gain > 0.01:
            h, w = overlay.shape[:2]
            small = cv2.resize(overlay, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), 2.5)
            glow = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            out = cv2.add(out, cv2.convertScaleAbs(glow, alpha=glow_gain))
        if self.ui.glitch:
            self._apply_glitch(out)
        return out

    def _apply_glitch(self, out) -> None:
        h, w = out.shape[:2]
        if self._glitch_frames > 0:
            self._glitch_frames -= 1
            for _ in range(3):
                y = random.randint(0, max(0, h - 30))
                hh = random.randint(6, 28)
                out[y:y + hh] = np.roll(out[y:y + hh], random.randint(-25, 25), axis=1)
        elif random.random() < 0.004:
            self._glitch_frames = random.randint(2, 5)
