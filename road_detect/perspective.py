"""Bird's eye (inverse perspective) mapping."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import Config


def roi_quad(shape: Tuple[int, int], cfg: Config) -> np.ndarray:
    """Return the source trapezoid in pixel coordinates (bl, tl, tr, br)."""
    h, w = shape[:2]
    return np.float32(
        [
            [cfg.roi_bottom_left_x * w, cfg.roi_bottom_y * h],
            [cfg.roi_top_left_x * w, cfg.roi_top_y * h],
            [cfg.roi_top_right_x * w, cfg.roi_top_y * h],
            [cfg.roi_bottom_right_x * w, cfg.roi_bottom_y * h],
        ]
    )


def transforms(shape: Tuple[int, int], cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Return (M, M_inv) mapping image <-> bird's eye view."""
    src = roi_quad(shape, cfg)
    dst = np.float32(
        [
            [0, cfg.bev_height],
            [0, 0],
            [cfg.bev_width, 0],
            [cfg.bev_width, cfg.bev_height],
        ]
    )
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    return m, m_inv


def warp(image: np.ndarray, m: np.ndarray, cfg: Config) -> np.ndarray:
    return cv2.warpPerspective(
        image, m, (cfg.bev_width, cfg.bev_height), flags=cv2.INTER_LINEAR
    )


def unwarp(image: np.ndarray, m_inv: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape[:2]
    return cv2.warpPerspective(image, m_inv, (w, h), flags=cv2.INTER_LINEAR)
