"""Render synthetic road scenes used by the tests and for quick demos.

The road is painted in a metric top-down canvas and then projected into a
camera view with the same trapezoid the detector uses by default, which gives
deterministic ground truth for the number of lanes.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Tuple

import cv2
import numpy as np

from road_detect.config import Config
from road_detect import perspective

WHITE = (245, 245, 245)
YELLOW = (60, 210, 245)


def _x(cfg: Config, x_m: float) -> int:
    return int(round(x_m / cfg.m_per_px_x()))


def _y(cfg: Config, y_m: float) -> int:
    return int(round(cfg.bev_height - y_m / cfg.m_per_px_y()))


def paint_road(
    cfg: Config,
    solid: Iterable[Tuple[float, Tuple[int, int, int]]],
    dashed: Iterable[Tuple[float, Tuple[int, int, int]]],
    chevron: Tuple[float, float] | None = None,
) -> np.ndarray:
    bev = np.full((cfg.bev_height, cfg.bev_width, 3), 70, np.uint8)
    bev += np.random.randint(-6, 6, bev.shape, dtype=np.int16).astype(np.uint8)
    thickness = max(3, _x(cfg, 0.15))

    for x_m, color in solid:
        cv2.line(bev, (_x(cfg, x_m), 0), (_x(cfg, x_m), cfg.bev_height), color, thickness)
    for x_m, color in dashed:
        step_m, mark_m = 9.0, 4.0
        y = 0.0
        while y < cfg.bev_length_m:
            cv2.line(
                bev,
                (_x(cfg, x_m), _y(cfg, y)),
                (_x(cfg, x_m), _y(cfg, min(y + mark_m, cfg.bev_length_m))),
                color, thickness,
            )
            y += step_m
    if chevron is not None:
        x0, x1 = chevron
        cv2.line(bev, (_x(cfg, x0), 0), (_x(cfg, x0), cfg.bev_height), WHITE, thickness)
        cv2.line(bev, (_x(cfg, x1), 0), (_x(cfg, x1), cfg.bev_height), WHITE, thickness)
        y = 1.0
        while y < cfg.bev_length_m - 2:
            cv2.line(
                bev,
                (_x(cfg, x0), _y(cfg, y)),
                (_x(cfg, x1), _y(cfg, y + 2.5)),
                WHITE, thickness,
            )
            y += 3.0
    return bev


def to_camera_view(bev: np.ndarray, cfg: Config, size=(1280, 720)) -> np.ndarray:
    w, h = size
    _, m_inv = perspective.transforms((h, w), cfg)
    sky = np.full((h, w, 3), 120, np.uint8)
    sky[: int(h * 0.55)] = (200, 180, 150)
    road = cv2.warpPerspective(bev, m_inv, (w, h), borderValue=(70, 70, 70))
    mask = cv2.warpPerspective(
        np.full(bev.shape[:2], 255, np.uint8), m_inv, (w, h)
    )
    out = np.where(mask[..., None] > 0, road, sky)
    return cv2.GaussianBlur(out, (3, 3), 0)


SCENES = {
    "three_lanes": dict(
        solid=[(0.75, YELLOW), (11.25, WHITE)],
        dashed=[(4.25, WHITE), (7.75, WHITE)],
        chevron=None,
        lanes=3,
    ),
    "two_lanes_chevron": dict(
        solid=[(0.75, YELLOW), (4.25, WHITE)],
        dashed=[],
        chevron=(7.75, 11.25),
        lanes=2,
    ),
    "single_lane": dict(
        solid=[(3.5, WHITE), (7.0, WHITE)], dashed=[], chevron=None, lanes=1
    ),
}


def build(name: str, cfg: Config | None = None) -> Tuple[np.ndarray, int]:
    cfg = cfg or Config()
    scene = SCENES[name]
    bev = paint_road(cfg, scene["solid"], scene["dashed"], scene["chevron"])
    return to_camera_view(bev, cfg), scene["lanes"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="samples")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for name in SCENES:
        img, lanes = build(name)
        path = os.path.join(args.outdir, f"{name}.png")
        cv2.imwrite(path, img)
        print(f"{path}  (ground truth: {lanes} lanes)")


if __name__ == "__main__":
    main()
