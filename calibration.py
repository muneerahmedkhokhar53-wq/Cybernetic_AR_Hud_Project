"""Pinch calibration wizard (press C): measures your open-hand and pinch distances."""
from __future__ import annotations

import numpy as np

from .draw import bar, put_text, shade

STEPS = (
    ("OPEN YOUR HAND WIDE AND HOLD STILL", "open"),
    ("PINCH THUMB AND INDEX TOGETHER AND HOLD", "pinch"),
)
WARMUP, DURATION = 0.8, 2.4


class Calibrator:
    def __init__(self):
        self.active = False
        self.step = 0
        self.values: dict = {}
        self._samples: list = []
        self._t0 = None

    def start(self) -> None:
        self.active = True
        self.step = 0
        self.values = {}
        self._samples = []
        self._t0 = None

    def cancel(self) -> None:
        self.active = False

    def update(self, ctx, engine) -> None:
        if not self.active:
            return
        hand = next(iter(ctx.hands.values()), None)
        if hand is None:
            self._t0, self._samples = None, []
            return
        if self._t0 is None:
            self._t0, self._samples = ctx.t, []
        elapsed = ctx.t - self._t0
        if elapsed > WARMUP:
            self._samples.append(hand.pinch_ratio)
        if elapsed >= DURATION:
            if self._samples:
                self.values[STEPS[self.step][1]] = float(np.median(self._samples))
            self.step += 1
            self._t0, self._samples = None, []
            if self.step >= len(STEPS):
                self._finish(ctx, engine)

    def _finish(self, ctx, engine) -> None:
        self.active = False
        opened, pinched = self.values.get("open"), self.values.get("pinch")
        if opened and pinched and opened > pinched * 1.5:
            threshold = pinched + 0.4 * (opened - pinched)
            engine.set_pinch_threshold(threshold)
            ctx.toast(f"CALIBRATED - PINCH THRESHOLD {engine.pinch_threshold:.2f}")
        else:
            ctx.toast("CALIBRATION FAILED - TRY AGAIN (C)", ctx.theme.alert, 2.5)

    def draw(self, ctx) -> None:
        if not self.active:
            return
        th, ov = ctx.theme, ctx.overlay
        w, h = ctx.size
        msg = STEPS[min(self.step, len(STEPS) - 1)][0]
        put_text(ov, f"CALIBRATION {self.step + 1}/{len(STEPS)}", (w // 2, h // 2 - 40), th.secondary, 0.7, 2, "center")
        put_text(ov, msg, (w // 2, h // 2), th.primary, 0.65, 1, "center")
        if self._t0 is None:
            put_text(ov, "SHOW YOUR HAND TO THE CAMERA", (w // 2, h // 2 + 34), th.alert, 0.5, 1, "center")
        else:
            bar(ov, w // 2 - 160, h // 2 + 20, 320, 12, min(1.0, (ctx.t - self._t0) / DURATION), th.secondary)
