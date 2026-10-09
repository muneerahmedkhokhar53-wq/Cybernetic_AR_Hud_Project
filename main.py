"""Cybernetic AR HUD - entry point.

    python main.py                      # default config.yaml
    python main.py --theme red --objects --voice
"""
import argparse

from hud.app import HUDApp
from hud.config import load_config


def parse_args():
    ap = argparse.ArgumentParser(description="Cybernetic AR HUD")
    ap.add_argument("--config", default="config.yaml", help="path to YAML config")
    ap.add_argument("--camera", type=int, help="camera index")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--theme", choices=["cyan", "red", "green", "amber"])
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--no-pose", action="store_true", help="disable body pose (faster)")
    ap.add_argument("--no-face", action="store_true", help="disable face mesh")
    ap.add_argument("--objects", action="store_true", help="enable YOLO object detection")
    ap.add_argument("--face-id", action="store_true", help="enable face identification (enrolled faces only)")
    ap.add_argument("--voice", action="store_true", help="enable voice commands")
    ap.add_argument("--no-boot", action="store_true", help="skip the boot animation")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.camera is not None:
        cfg.camera.index = args.camera
    if args.width:
        cfg.camera.width = args.width
    if args.height:
        cfg.camera.height = args.height
    if args.theme:
        cfg.ui.theme = args.theme
    if args.fullscreen:
        cfg.ui.fullscreen = True
    if args.no_pose:
        cfg.perception.pose = False
    if args.no_face:
        cfg.perception.face_mesh = False
    if args.objects:
        cfg.perception.object_detection = True
    if args.face_id:
        cfg.perception.face_recognition = True
    if args.voice:
        cfg.voice.enabled = True
    if args.no_boot:
        cfg.ui.boot_sequence = False
    HUDApp(cfg).run()


if __name__ == "__main__":
    main()
