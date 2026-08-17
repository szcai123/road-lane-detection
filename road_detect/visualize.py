"""Drawing helpers (bird's eye debug view and overlay on the original frame)."""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .detector import DetectionResult
from . import perspective

_COLOR = {"white": (255, 255, 255), "yellow": (0, 215, 255)}


def draw_bev(result: DetectionResult, cfg: Config) -> np.ndarray:
    canvas = result.bev_bgr.copy()
    overlay = canvas.copy()
    ys = np.arange(cfg.bev_height)

    for lane in result.lanes:
        x0 = int(lane.left_x_m / result.m_per_px_x)
        x1 = int(lane.right_x_m / result.m_per_px_x)
        cv2.rectangle(overlay, (x0, 0), (x1, cfg.bev_height), (0, 160, 0), -1)
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    for zone in result.hatch_zones:
        cv2.polylines(canvas, [zone.polygon], True, (255, 0, 255), 2)
        x, y = zone.polygon[:, 0, 0].min(), zone.polygon[:, 0, 1].min()
        cv2.putText(
            canvas, "hatch", (int(x), max(15, int(y) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA,
        )

    for line in result.lines:
        xs = line.x_at(ys)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        dash = 1 if line.style == "solid" else 0
        color = _COLOR[line.color]
        if dash:
            cv2.polylines(canvas, [pts], False, color, 3)
        else:
            for k in range(0, len(pts) - 20, 40):
                cv2.polylines(canvas, [pts[k : k + 20]], False, color, 3)
        cv2.putText(
            canvas, f"{line.style[0]}", (int(line.x_px) - 5, cfg.bev_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )

    for lane in result.lanes:
        cx = int((lane.left_x_m + lane.right_x_m) / 2 / result.m_per_px_x)
        cv2.putText(
            canvas, str(lane.index), (cx - 8, cfg.bev_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA,
        )
    return canvas


def draw_overlay(image_bgr: np.ndarray, result: DetectionResult, cfg: Config) -> np.ndarray:
    layer = np.zeros_like(result.bev_bgr)
    ys = np.arange(cfg.bev_height)

    for lane in result.lanes:
        x0 = int(lane.left_x_m / result.m_per_px_x)
        x1 = int(lane.right_x_m / result.m_per_px_x)
        cv2.rectangle(layer, (x0, 0), (x1, cfg.bev_height), (0, 150, 0), -1)
    for zone in result.hatch_zones:
        cv2.fillPoly(layer, [zone.polygon], (180, 0, 180))
    for line in result.lines:
        pts = np.stack([line.x_at(ys), ys], axis=1).astype(np.int32)
        cv2.polylines(layer, [pts], False, _COLOR[line.color], 6)

    unwarped = perspective.unwarp(layer, result.m_inv, image_bgr.shape)
    out = cv2.addWeighted(image_bgr, 1.0, unwarped, 0.45, 0)

    cv2.rectangle(out, (10, 10), (330, 100), (0, 0, 0), -1)
    cv2.putText(out, f"Lanes: {result.lane_count}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        out,
        f"lines: {len(result.lines)}  hatch: {len(result.hatch_zones)}",
        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


def draw_lines(result: DetectionResult, cfg: Config) -> np.ndarray:
    """Extracted markings only: the binary paint mask plus the fitted lines."""
    canvas = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
    canvas = (canvas * 0.45).astype(np.uint8)
    ys = np.arange(cfg.bev_height)

    for zone in result.hatch_zones:
        cv2.fillPoly(canvas, [zone.polygon], (120, 0, 120))
    for i, line in enumerate(result.lines, start=1):
        pts = np.stack([line.x_at(ys), ys], axis=1).astype(np.int32)
        cv2.polylines(canvas, [pts], False, _COLOR[line.color], 2)
        cv2.putText(
            canvas, f"{i}:{line.color[0]}{line.style[0]}",
            (max(2, int(line.x_px) - 20), 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
        )
    cv2.putText(
        canvas, f"lines: {len(result.lines)}  lanes: {result.lane_count}",
        (10, cfg.bev_height - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
    )
    return canvas


def draw_roi(image_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    out = image_bgr.copy()
    quad = perspective.roi_quad(image_bgr.shape, cfg).astype(np.int32)
    cv2.polylines(out, [quad], True, (0, 0, 255), 2)
    return out
