# Cybernetic AR HUD

A real-time augmented-reality heads-up display built with Python, OpenCV and MediaPipe.
Gesture recognition, face tracking and analysis, body pose and optional object detection feed a cyberpunk HUD with
neon glow, draggable panels, a hologram, an air keyboard, voice control and a plugin system.

## Features

| Area | What you get |
|------|--------------|
| Vision | Two-hand tracking, 478-point face mesh with head pose, gaze, blink and expression estimates, upper-body pose, optional YOLO objects, optional face identification (enrolled faces only) |
| Gestures | Open palm, fist, point, peace, thumbs-up, rock, pinch; swipe left/right/up/down; two-hand pinch (scale + roll); optional custom classifier you train yourself |
| Interaction | Pinch-drag panels, pinch sliders, **DRAW** (air drawing), **HOLO** (3D wireframe), **KEYS** (air keyboard), voice commands |
| Visuals | Neon glow, scanlines, vignette, glitch, boot sequence, radar, neural-net animation, telemetry sparklines, 4 themes |
| Engineering | Threaded camera and perception (UI never blocks on inference), modular package, YAML config, plugin loader, recorder, calibration wizard |

## Install

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Optional extras: `pip install ultralytics` (objects), `face_recognition` (face ID), `SpeechRecognition PyAudio` (voice).
MediaPipe 0.10.14 supports Python 3.8-3.12.

## Run

```bash
python main.py                       # uses config.yaml
python main.py --theme red --objects # red theme + YOLO object detection
python main.py --no-pose --no-face   # fastest: hands only
python main.py --voice               # voice commands
```

## Controls

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `1`-`9` | switch mode | `T` | cycle theme |
| `R` | start/stop recording | `S` | screenshot |
| `C` | calibrate pinch | `M` | toggle face mesh |
| `F` | fullscreen | `Q` / `Esc` | quit |
| `B` / `X` | (DRAW) brush colour / clear | | |

**Gestures:** swipe an open palm left/right to change mode - pinch a panel header and drag to move it - pinch inside a slider
to set it - in DRAW point to draw and show peace to clear - in HOLO pinch near the object and drag to rotate, pinch with both
hands to scale and roll, show peace to change shape - in KEYS pinch a key to type.

**Voice** (English keywords): "scan", "draw", "hologram", "keyboard", "clock", "next mode", "screenshot", "record",
"stop recording", "calibrate", "theme red / cyan / green / amber".

## Train your own gestures

```bash
python train_gesture.py collect --labels open_palm fist point peace thumbs_up rock
python train_gesture.py train      # writes models/gesture_model.pkl, loaded automatically
```

Record both hands, different angles and distances (200+ samples per gesture). Keep the names `open_palm`, `point`, `peace`
so swipes, DRAW and HOLO keep working.

## Face identification (optional, consent only)

```bash
pip install face_recognition
python enroll_face.py --name Ali      # only enroll people who agreed
python main.py --face-id
```

Only face encodings are stored (`faces/encodings.pkl`), locally. Delete the file to forget everyone. Anyone not enrolled
shows as UNKNOWN.

## Architecture

```
main.py                 entry point + CLI flags
config.yaml             all tunables
hud/
  app.py                main loop, chrome, panels, boot, events, key + voice handling
  capture.py            threaded camera (newest frame only)
  perception.py         worker thread: hands, face mesh + analysis, pose, YOLO, face ID
  gestures.py           static gestures, ML classifier, pinch, swipes, two-hand
  draw.py               primitives, HUD elements, compositor (grade, glow, glitch)
  widgets.py            Panel, PanelManager, Slider, Hologram, VirtualKeyboard
  calibration.py        pinch calibration wizard
  recorder.py voice.py  video/screenshot, voice thread
  modules/              SCAN, DRAW, HOLO, KEYS + HUDModule base class + plugin loader
plugins/clock_plugin.py example plugin (new mode + draggable panel)
```

Data flow per frame: camera thread -> perception thread (latest result) -> `GestureEngine.update` -> `Context` ->
panels/modules update -> draw into `overlay` -> compositor adds glow -> window/recorder.

### Write a plugin

```python
# plugins/my_mode.py
from hud.modules import HUDModule
from hud.draw import put_text

class MyMode(HUDModule):
    name = "MINE"
    show_face = False           # which shared overlays stay visible
    def update(self, ctx): ...  # ctx.hands, ctx.perception, ctx.two_hand, ctx.t, ctx.dt
    def draw(self, ctx):
        put_text(ctx.overlay, "HELLO", (100, 100), ctx.theme.secondary, 1.0, 2)

def register(registry):
    registry.register(MyMode())
```

## Tuning

- Low FPS: `--no-pose`, lower `process_scale` (0.4-0.5), `pose_complexity: 0`, `max_faces: 1`, disable `glow`.
- Pinch triggers too easily or not at all: press `C` to calibrate.
- Wrong camera: set `camera.index` (try 1 or 2) and a lower `width/height` if the stream is slow.
- Face angles, gaze and expression are geometric estimates, good for a HUD effect, not for measurement.

## Privacy and responsible use

Camera frames and face-related data can be sensitive. Face identification is optional and should only be enabled with informed consent. The README describes local storage of enrolled encodings; inspect your configuration and recording settings before use, avoid committing captured media or encodings, and delete local data when it is no longer needed. Geometric gaze and expression outputs are approximate visual effects, not medical or psychological measurements.

## Notes

Face identification is optional, local and opt-in. Please only enroll people who consent.
