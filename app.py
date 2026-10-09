"""HUDApp - ties camera, perception, gestures, modules and rendering together."""
from __future__ import annotations

import queue
import time
from collections import deque

import cv2
import numpy as np

from .calibration import Calibrator
from .capture import Camera
from .context import Context
from .draw import (Compositor, bar, brackets, draw_face, draw_hand, draw_neural_net, draw_objects,
                   draw_pose, draw_radar, put_text, shade, sparkline, text_width)
from .gestures import GestureEngine
from .modules import ModuleRegistry, load_plugins
from .modules.draw_mode import DrawModule
from .modules.holo import HoloModule
from .modules.keys import KeysModule
from .modules.scan import ScanModule
from .perception import PerceptionEngine
from .recorder import Recorder
from .theme import get_theme, next_theme
from .voice import VoiceControl
from .widgets import Panel, PanelManager, Slider

BOOT_STEPS = (("OPTICAL CORE", 0.3), ("GESTURE ENGINE", 0.9), ("FACIAL MESH", 1.5),
              ("NEURAL LINK", 2.1), ("HOLO RENDERER", 2.7))
BOOT_MIN = 3.4


class HUDApp:
    def __init__(self, cfg):
        self.cfg = cfg
        ui = cfg.ui
        self.camera = Camera(cfg.camera.index, cfg.camera.width, cfg.camera.height, cfg.camera.mirror)
        self.perception = PerceptionEngine(self.camera, cfg)
        self.gestures = GestureEngine(cfg)
        self.compositor = Compositor(cfg)
        self.recorder = Recorder(cfg.recording.dir, cfg.recording.fps)
        self.calibrator = Calibrator()
        self.panels = PanelManager()
        self.theme = get_theme(ui.theme)
        self.dim = float(ui.dim)
        self.glow_gain = 1.6
        self.fullscreen = bool(ui.fullscreen)
        self.toasts: list = []
        self.fps_hist = deque(maxlen=120)
        self.lat_hist = deque(maxlen=120)
        self.voice_queue: queue.Queue = queue.Queue()
        self.voice = None

        self.registry = ModuleRegistry(self)
        for module in (ScanModule(), DrawModule(), HoloModule(), KeysModule()):
            self.registry.register(module)
        self.loaded_plugins = load_plugins(self.registry, cfg.plugins.dir)
        self.active = 0

        self._sliders = {"glow": Slider("GLOW", (self.glow_gain - 0.4) / 2.0),
                         "dim": Slider("BRIGHTNESS", (self.dim - 0.35) / 0.65)}
        self._panels_ready = False
        self._entered = False
        self._fps = 0.0
        self._t0 = self._last_t = time.time()
        self._boot_done = None
        self._last_out = None

    # ------------------------------------------------------------------ properties / helpers
    @property
    def module(self):
        return self.registry.modules[self.active]

    def add_toast(self, text: str, color=None, duration: float = 1.8) -> None:
        now = time.time()
        self.toasts.append({"text": text, "color": color, "t0": now, "t1": now + duration})
        del self.toasts[:-4]

    def set_module(self, index: int, ctx) -> None:
        index %= len(self.registry.modules)
        if index == self.active:
            return
        self.module.on_exit(ctx)
        self.active = index
        self.module.on_enter(ctx)
        self.add_toast(f"MODE: {self.module.name}")

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        self.camera.start()
        self.perception.start()
        if self.cfg.voice.enabled:
            self.voice = VoiceControl(self.cfg.voice.language, self.voice_queue.put)
            self.voice.start()
        name = self.cfg.ui.window_name
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        ctx = None
        try:
            while True:
                frame, _ = self.camera.read()
                if frame is None:
                    self._show_waiting(name)
                    if cv2.waitKey(30) & 0xFF in (27, ord("q")):
                        break
                    continue
                h, w = frame.shape[:2]
                now = time.time()
                dt = max(1e-3, now - self._last_t)
                self._last_t = now
                self._fps = self._fps * 0.9 + (1.0 / dt) * 0.1 if self._fps else 1.0 / dt

                perception = self.perception.get()
                hands = self.gestures.update(perception, now)
                ctx = Context(app=self, cfg=self.cfg, theme=self.theme, t=now - self._t0, dt=dt, fps=self._fps,
                              size=(w, h), perception=perception, hands=hands,
                              two_hand=self.gestures.two_hand, events=list(self.gestures.events))

                self._ensure_panels(ctx)
                if not self._entered:
                    self.module.on_enter(ctx)
                    self._entered = True
                self._drain_voice(ctx)
                for event in ctx.events:
                    self._handle_event(event, ctx)
                self.panels.update(ctx)
                self.calibrator.update(ctx, self.gestures)
                self.module.update(ctx)
                self.fps_hist.append(self._fps)
                self.lat_hist.append(perception.latency_ms)

                base = self.compositor.grade(frame, self.theme, self.dim)
                ctx.base, ctx.overlay = base, np.zeros_like(base)
                if self._booting(ctx):
                    self._draw_boot(ctx)
                else:
                    self._draw_scene(ctx)
                    fade = min(1.0, (ctx.t - self._boot_done) / 0.8)
                    if fade < 1.0:
                        ctx.overlay = cv2.convertScaleAbs(ctx.overlay, alpha=fade)
                out = self.compositor.finish(base, ctx.overlay, self.glow_gain)
                self._last_out = out
                self.recorder.write(out)
                cv2.imshow(name, out)

                key = cv2.waitKey(1) & 0xFF
                if key != 255 and not self._handle_key(key, ctx):
                    break
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        self.recorder.close()
        self.perception.stop()
        if self.voice is not None:
            self.voice.stop()
        self.camera.stop()
        cv2.destroyAllWindows()

    def _show_waiting(self, name: str) -> None:
        img = np.zeros((720, 1280, 3), np.uint8)
        msg = self.camera.error or "WAITING FOR CAMERA..."
        put_text(img, msg.upper(), (640, 340), self.theme.primary, 0.9, 2, "center")
        put_text(img, "CHECK camera.index IN config.yaml - PRESS Q TO QUIT", (640, 380),
                 shade(self.theme.primary, 0.6), 0.55, 1, "center")
        cv2.imshow(name, img)

    # ------------------------------------------------------------------ events / input
    def _handle_event(self, event, ctx) -> None:
        if event.name == "SWIPE_RIGHT":
            self.set_module(self.active + 1, ctx)
        elif event.name == "SWIPE_LEFT":
            self.set_module(self.active - 1, ctx)
        else:
            self.module.on_event(ctx, event)

    def _handle_key(self, key: int, ctx) -> bool:
        if key in (27, ord("q")):
            return False
        if self.module.on_key(ctx, key):
            return True
        if key == ord("t"):
            self.theme = next_theme(self.theme.name)
            self.add_toast(f"THEME: {self.theme.name.upper()}")
        elif key == ord("r"):
            h, w = ctx.base.shape[:2]
            self.add_toast(self.recorder.toggle((w, h)))
        elif key == ord("s") and self._last_out is not None:
            self.add_toast(f"SAVED {self.recorder.screenshot(self._last_out)}")
        elif key == ord("c"):
            self.calibrator.start()
        elif key == ord("m"):
            self.cfg.ui.show_mesh = not self.cfg.ui.show_mesh
        elif key == ord("f"):
            self.fullscreen = not self.fullscreen
            prop = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(self.cfg.ui.window_name, cv2.WND_PROP_FULLSCREEN, prop)
        elif ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(self.registry.modules):
                self.set_module(idx, ctx)
        return True

    def _drain_voice(self, ctx) -> None:
        while True:
            try:
                text = self.voice_queue.get_nowait()
            except queue.Empty:
                return
            self._handle_voice(text, ctx)

    def _handle_voice(self, text: str, ctx) -> None:
        ctx.toast(f'VOICE: "{text.upper()}"')
        aliases = {"hologram": "holo", "keyboard": "keys", "drawing": "draw", "scanner": "scan"}
        for word, target in aliases.items():
            text = text.replace(word, target)
        idx = self.registry.index_of(text)
        if idx is not None:
            self.set_module(idx, ctx)
        if "screenshot" in text and self._last_out is not None:
            self.add_toast(f"SAVED {self.recorder.screenshot(self._last_out)}")
        elif "stop recording" in text and self.recorder.recording:
            self.add_toast(self.recorder.toggle((ctx.size[0], ctx.size[1])))
        elif "record" in text and not self.recorder.recording:
            self.add_toast(self.recorder.toggle((ctx.size[0], ctx.size[1])))
        elif "next mode" in text:
            self.set_module(self.active + 1, ctx)
        elif "calibrate" in text:
            self.calibrator.start()
        for theme_name in ("cyan", "red", "green", "amber"):
            if "theme" in text and theme_name in text:
                self.theme = get_theme(theme_name)

    # ------------------------------------------------------------------ boot sequence
    def _booting(self, ctx) -> bool:
        if not self.cfg.ui.boot_sequence:
            if self._boot_done is None:
                self._boot_done = ctx.t - 1.0
            return False
        if self._boot_done is not None:
            return False
        models_done = self.perception.ready.is_set() or bool(self.perception.error)
        if ctx.t >= BOOT_MIN and models_done:
            self._boot_done = ctx.t
            return False
        return True

    def _draw_boot(self, ctx) -> None:
        th, ov = ctx.theme, ctx.overlay
        w, h = ctx.size
        cx, cy = w // 2, h // 2
        brackets(ov, 14, 14, w - 14, h - 14, th.primary, 60, 2)
        put_text(ov, "CYBERNETIC AR HUD", (cx, cy - 150), th.secondary, 1.1, 2, "center")
        put_text(ov, "INITIALIZING SYSTEMS", (cx, cy - 118), shade(th.primary, 0.8), 0.55, 1, "center")
        for i, (label, t_start) in enumerate(BOOT_STEPS):
            if ctx.t < t_start:
                break
            done = ctx.t > t_start + 0.4
            yy = cy - 70 + i * 28
            put_text(ov, label, (cx - 190, yy), th.primary, 0.6)
            put_text(ov, "OK" if done else "...", (cx + 190, yy), th.secondary if done else th.alert, 0.6, 1, "right")
        loading = not (self.perception.ready.is_set() or self.perception.error)
        frac = min(1.0, ctx.t / BOOT_MIN)
        if loading:
            frac = min(frac, 0.95)
        bar(ov, cx - 190, cy + 100, 380, 14, frac, th.secondary)
        status = "LOADING NEURAL MODELS" + "." * (int(ctx.t * 3) % 4) if loading else "SYSTEMS ONLINE"
        put_text(ov, status, (cx, cy + 140), th.secondary, 0.55, 1, "center")

    # ------------------------------------------------------------------ scene
    def _draw_scene(self, ctx) -> None:
        th, ov, m = ctx.theme, ctx.overlay, self.module
        p = ctx.perception
        if m.show_pose and p.pose is not None:
            draw_pose(ov, p.pose, th)
        if m.show_objects:
            draw_objects(ov, p.objects, th)
        if m.show_face:
            for i, face in enumerate(p.faces):
                draw_face(ov, face, i, th, ctx.t, bool(self.cfg.ui.show_mesh))
        for st in ctx.hands.values():
            draw_hand(ov, st, th, ctx.t)
        self._draw_chrome(ctx)
        if m.show_panels:
            self.panels.draw(ctx)
        m.draw(ctx)
        if self.perception.error:
            w, h = ctx.size
            put_text(ov, "PERCEPTION OFFLINE", (w // 2, h // 2 - 20), th.alert, 0.9, 2, "center")
            put_text(ov, self.perception.error[:90], (w // 2, h // 2 + 14), th.alert, 0.5, 1, "center")
            put_text(ov, "pip install mediapipe==0.10.14", (w // 2, h // 2 + 40), th.primary, 0.5, 1, "center")
        self._draw_toasts(ctx)
        self.calibrator.draw(ctx)

    def _draw_chrome(self, ctx) -> None:
        th, ov, m = ctx.theme, ctx.overlay, self.module
        w, h = ctx.size
        p = ctx.perception
        brackets(ov, 14, 14, w - 14, h - 14, th.primary, 60, 2)

        # top bar
        cv2.line(ov, (24, 46), (w - 24, 46), shade(th.primary, 0.5), 1)
        put_text(ov, "CYBERNETIC AR HUD", (30, 34), th.secondary, 0.65, 2)
        labels = [f"[{i + 1}] {mod.name}" for i, mod in enumerate(self.registry.modules)]
        widths = [text_width(s, 0.5) for s in labels]
        x = (w - (sum(widths) + 26 * (len(labels) - 1))) // 2
        for i, (s, tw) in enumerate(zip(labels, widths)):
            active = i == self.active
            put_text(ov, s, (x, 34), th.secondary if active else shade(th.primary, 0.55), 0.5, 2 if active else 1)
            if active:
                cv2.line(ov, (x, 40), (x + tw, 40), th.secondary, 2)
            x += tw + 26
        right = f"{time.strftime('%H:%M:%S')}   FPS {ctx.fps:4.1f}"
        if self.recorder.recording:
            right = f"[REC {int(self.recorder.elapsed):03d}s]  " + right
        put_text(ov, right, (w - 30, 34), th.alert if self.recorder.recording else th.primary, 0.5, 1, "right")

        # bottom hint bar
        put_text(ov, "Q QUIT | 1-9 MODE | T THEME | R REC | S SHOT | C CALIBRATE | M MESH | F FULLSCREEN | SWIPE L/R MODE",
                 (w // 2, h - 18), shade(th.primary, 0.6), 0.4, 1, "center")

        if m.show_panels:
            draw_neural_net(ov, 36, int(h * 0.40), 150, 190, ctx.t, th)
            blips = [(((f.bbox[0] + f.bbox[2]) / 2) / w, ((f.bbox[1] + f.bbox[3]) / 2) / h, th.alert) for f in p.faces]
            blips += [(s.palm_center[0] / w, s.palm_center[1] / h, th.secondary) for s in ctx.hands.values()]
            blips += [(((o.bbox[0] + o.bbox[2]) / 2) / w, ((o.bbox[1] + o.bbox[3]) / 2) / h, th.primary)
                      for o in p.objects]
            draw_radar(ov, w - 130, h - 150, 90, ctx.t, th, blips)

    def _draw_toasts(self, ctx) -> None:
        now = time.time()
        self.toasts = [t for t in self.toasts if t["t1"] > now]
        w, _ = ctx.size
        for i, toast in enumerate(self.toasts):
            k = min(1.0, (toast["t1"] - now) / 0.4)
            color = toast["color"] or ctx.theme.secondary
            put_text(ctx.overlay, toast["text"], (w // 2, 74 + i * 26), shade(color, k), 0.6, 1, "center")

    # ------------------------------------------------------------------ panels
    def _ensure_panels(self, ctx) -> None:
        if self._panels_ready:
            return
        w, _ = ctx.size
        self.panels.add(Panel("TELEMETRY", 30, 64, 300, 150, self._telemetry_panel))
        self.panels.add(Panel("SYSTEM STATUS", w - 340, 64, 310, 190, self._status_panel))
        self.panels.add(Panel("SYSTEM CONTROL", w - 340, 270, 310, 130, self._control_panel))
        self._panels_ready = True

    def _status_panel(self, ctx, rect) -> None:
        x, y, w, h = rect
        th, ov, p = ctx.theme, ctx.overlay, ctx.perception
        gest = ", ".join(f"{s.label[0]}:{s.gesture.replace('_', ' ')}" for s in ctx.hands.values()) or "NONE"
        rows = (("MODE", self.module.name), ("GESTURE", gest),
                ("HANDS / FACES", f"{len(p.hands)} / {len(p.faces)}"), ("OBJECTS", str(len(p.objects))),
                ("LATENCY", f"{p.latency_ms:.1f} MS"), ("FPS", f"{ctx.fps:.1f}"),
                ("THEME", th.name.upper()), ("RECORDING", "ON" if self.recorder.recording else "OFF"))
        for i, (k, v) in enumerate(rows):
            yy = y + 18 + i * 18
            put_text(ov, k, (x + 10, yy), shade(th.primary, 0.85), 0.4)
            put_text(ov, v, (x + 130, yy), th.secondary, 0.4)

    def _telemetry_panel(self, ctx, rect) -> None:
        x, y, w, h = rect
        th, ov = ctx.theme, ctx.overlay
        rows = (("FPS", self.fps_hist, 0, 60, f"{ctx.fps:.1f}"),
                ("LATENCY MS", self.lat_hist, 0, 120, f"{ctx.perception.latency_ms:.1f}"))
        for i, (label, hist, lo, hi, val) in enumerate(rows):
            yy = y + 8 + i * 58
            put_text(ov, label, (x + 10, yy + 10), shade(th.primary, 0.85), 0.4)
            put_text(ov, val, (x + w - 10, yy + 10), th.secondary, 0.45, 1, "right")
            sparkline(ov, (x + 10, yy + 18, w - 20, 30), hist, th.secondary, lo, hi)

    def _control_panel(self, ctx, rect) -> None:
        x, y, w, h = rect
        self._sliders["glow"].run(ctx, x + 14, y + 6, w - 28)
        self._sliders["dim"].run(ctx, x + 14, y + 56, w - 28)
        self.glow_gain = 0.4 + 2.0 * self._sliders["glow"].value
        self.dim = 0.35 + 0.65 * self._sliders["dim"].value
