"""Colour themes (BGR tuples, as OpenCV expects)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    name: str
    primary: tuple
    secondary: tuple
    alert: tuple


THEMES = {
    "cyan": Theme("cyan", (255, 225, 40), (140, 255, 110), (70, 70, 255)),
    "red": Theme("red", (70, 70, 255), (60, 170, 255), (255, 230, 40)),
    "green": Theme("green", (90, 255, 90), (200, 255, 150), (70, 70, 255)),
    "amber": Theme("amber", (0, 185, 255), (90, 235, 255), (255, 120, 60)),
}
ORDER = list(THEMES)


def get_theme(name: str) -> Theme:
    return THEMES.get(name, THEMES["cyan"])


def next_theme(name: str) -> Theme:
    idx = ORDER.index(name) if name in ORDER else -1
    return THEMES[ORDER[(idx + 1) % len(ORDER)]]
