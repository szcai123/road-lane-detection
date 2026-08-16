"""Extraction of painted road markings from a bird's eye view image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Config


def paint_mask(bev_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """Binary mask of white / yellow paint in a bird's eye view image.

    Combines a colour threshold (robust for saturated yellow and bright white)
    with a horizontal top-hat, which isolates thin bright structures on a darker
    background and therefore survives shadows and worn out paint.
    """
    blur = cv2.GaussianBlur(bev_bgr, (5, 5), 0)
    hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)

    white = ((l >= cfg.white_l_min) & (s <= cfg.white_s_max)).astype(np.uint8)
    yellow = (
        (h >= cfg.yellow_h[0])
        & (h <= cfg.yellow_h[1])
        & (s >= cfg.yellow_s_min)
        & (l >= cfg.yellow_l_min)
    ).astype(np.uint8)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.tophat_width, 1))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    thin = (tophat >= cfg.tophat_thresh).astype(np.uint8)

    mask = ((white | yellow | thin) * 255).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    return mask


@dataclass
class HatchZone:
    """A chevron / herringbone ("鱼骨线", 导流线) painted area."""

    stripes: int
    angle_deg: float
    x_range_m: Tuple[float, float]
    y_range_m: Tuple[float, float]
    polygon: np.ndarray  # convex hull in bird's eye pixels

    def as_dict(self) -> dict:
        return {
            "stripes": self.stripes,
            "angle_deg": round(self.angle_deg, 1),
            "x_range_m": [round(v, 2) for v in self.x_range_m],
            "y_range_m": [round(v, 2) for v in self.y_range_m],
        }


def _stripe_angle(rect) -> Tuple[float, float]:
    """Return (deviation from the vertical axis in degrees, elongation)."""
    (_, _), (w, h), angle = rect
    if w == 0 or h == 0:
        return 0.0, 0.0
    if w >= h:  # OpenCV's angle refers to the `w` edge
        long_side, short_side, long_angle = w, h, angle
    else:
        long_side, short_side, long_angle = h, w, angle + 90.0
    # long_angle is measured from the +x axis; lane lines run along +y (90 deg)
    deviation = abs(((long_angle - 90.0) + 90.0) % 180.0 - 90.0)
    return deviation, long_side / max(short_side, 1e-6)


def _is_brighter_than_road(
    gray: np.ndarray, pts: np.ndarray, road_level: float, cfg: Config
) -> bool:
    brightness = float(np.mean(gray[pts[:, 1], pts[:, 0]]))
    return brightness - road_level >= cfg.hatch_min_contrast


def _spatial_groups(candidates: List[tuple], cfg: Config) -> List[List[tuple]]:
    """Split angle-compatible stripes into spatially connected groups."""
    max_gap_px = cfg.hatch_max_stripe_spacing_m / cfg.m_per_px_y()
    ordered = sorted(candidates, key=lambda c: c[2][1])  # by centre y
    groups: List[List[tuple]] = []
    for cand in ordered:
        if groups and abs(cand[2][1] - groups[-1][-1][2][1]) <= max_gap_px:
            groups[-1].append(cand)
        else:
            groups.append([cand])
    return groups


def _is_regular(group: List[tuple], cfg: Config) -> bool:
    """Chevron stripes repeat: similar length, roughly constant spacing."""
    lengths = np.array([c[4] for c in group], dtype=np.float64)
    if lengths.max() / max(lengths.min(), 1e-6) > cfg.hatch_max_length_ratio:
        return False
    centres = np.sort(np.array([c[2][1] for c in group], dtype=np.float64))
    spacing = np.diff(centres)
    if len(spacing) < 2:
        return True
    mean = spacing.mean()
    return mean > 0 and spacing.std() / mean <= cfg.hatch_max_spacing_cv


def find_hatch_zones(
    mask: np.ndarray, cfg: Config, bev_bgr: Optional[np.ndarray] = None
) -> Tuple[List[HatchZone], np.ndarray]:
    """Detect chevron/fishbone areas and return them plus a cleaned mask.

    Chevron markings are groups of parallel stripes that are strongly tilted
    with respect to the driving direction.  In the bird's eye view lane lines
    are close to vertical, so the tilt is a reliable discriminator.  Detected
    stripes are removed from the returned mask so that they cannot be mistaken
    for lane boundaries.
    """
    # Chevron stripes usually touch the solid lines that delimit the area, which
    # would merge everything into one blob.  Remove the near-vertical structures
    # first so that every stripe becomes its own connected component.
    vertical = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
    )
    vertical = cv2.dilate(
        vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    )
    residual = cv2.bitwise_and(mask, cv2.bitwise_not(vertical))
    residual = cv2.morphologyEx(
        residual, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )

    gray = None
    road_level = 0.0
    if bev_bgr is not None:
        gray = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2GRAY)
        road_level = float(np.median(gray[mask == 0])) if (mask == 0).any() else 0.0

    min_len_px = cfg.hatch_min_stripe_len_m / cfg.m_per_px_y()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(residual, 8)
    candidates = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 80:
            continue
        pts = cv2.findNonZero((labels == i).astype(np.uint8)).reshape(-1, 2)
        rect = cv2.minAreaRect(pts)
        deviation, elongation = _stripe_angle(rect)
        if elongation < cfg.hatch_min_elongation or max(rect[1]) < min_len_px:
            continue
        if deviation < cfg.hatch_min_angle_deg:
            continue
        if gray is not None and not _is_brighter_than_road(gray, pts, road_level, cfg):
            continue
        candidates.append((i, deviation, rect[0], pts, max(rect[1])))

    # group stripes by orientation first, then by spatial proximity
    by_angle: List[List[tuple]] = []
    for cand in sorted(candidates, key=lambda c: c[1]):
        for group in by_angle:
            if abs(group[0][1] - cand[1]) <= cfg.hatch_angle_tolerance_deg:
                group.append(cand)
                break
        else:
            by_angle.append([cand])
    groups = [g for angle_group in by_angle for g in _spatial_groups(angle_group, cfg)]

    zones: List[HatchZone] = []
    cleaned = mask.copy()
    mx, my = cfg.m_per_px_x(), cfg.m_per_px_y()
    for group in groups:
        if len(group) < cfg.hatch_min_stripes or not _is_regular(group, cfg):
            continue
        all_pts = np.vstack([c[3] for c in group])
        if np.ptp(all_pts[:, 0]) > cfg.hatch_max_width_ratio * mask.shape[1]:
            continue  # a "stripe pattern" across the whole road is road texture
        hull = cv2.convexHull(all_pts)
        xs = all_pts[:, 0]
        ys = all_pts[:, 1]
        zones.append(
            HatchZone(
                stripes=len(group),
                angle_deg=float(np.mean([c[1] for c in group])),
                x_range_m=(float(xs.min() * mx), float(xs.max() * mx)),
                y_range_m=(float(ys.min() * my), float(ys.max() * my)),
                polygon=hull,
            )
        )
        stripe_mask = np.zeros_like(mask)
        for *_, pts, _len in group:
            stripe_mask[pts[:, 1], pts[:, 0]] = 255
        stripe_mask = cv2.dilate(
            stripe_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        )
        cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(stripe_mask))
    return zones, cleaned
