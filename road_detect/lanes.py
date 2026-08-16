"""Lane line tracing and lane counting in the bird's eye view."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .markings import HatchZone


@dataclass
class LaneLine:
    """A single longitudinal road marking (车道分割线)."""

    x_px: float                 # x position at the bottom of the bird's eye view
    x_m: float                  # same, in metres from the left edge of the BEV
    fit: np.ndarray             # 2nd order polynomial x = f(y) in BEV pixels
    fill_ratio: float
    style: str                  # "solid" | "dashed"
    color: str                  # "white" | "yellow"
    points: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 2)))

    def as_dict(self) -> dict:
        return {
            "x_m": round(self.x_m, 2),
            "style": self.style,
            "color": self.color,
            "fill_ratio": round(self.fill_ratio, 2),
        }

    def x_at(self, y: np.ndarray) -> np.ndarray:
        return np.polyval(self.fit, y)


@dataclass
class Lane:
    """A drivable lane, i.e. the space between two consecutive lines."""

    index: int
    left_x_m: float
    right_x_m: float
    inferred: bool  # True when no painted line separated it (wide gap split)

    @property
    def width_m(self) -> float:
        return self.right_x_m - self.left_x_m

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "left_x_m": round(self.left_x_m, 2),
            "right_x_m": round(self.right_x_m, 2),
            "width_m": round(self.width_m, 2),
            "inferred": self.inferred,
        }


def _find_peaks(hist: np.ndarray, min_value: float, min_distance: int) -> List[int]:
    """Greedy 1-D peak picking: strongest first, suppressing close neighbours."""
    peaks: List[int] = []
    working = hist.copy()
    while True:
        idx = int(np.argmax(working))
        if working[idx] < min_value:
            break
        peaks.append(idx)
        lo = max(0, idx - min_distance)
        hi = min(len(working), idx + min_distance + 1)
        working[lo:hi] = 0
    return sorted(peaks)


def _trace(mask: np.ndarray, x_start: int, cfg: Config) -> Optional[np.ndarray]:
    """Sliding window search upwards from ``x_start``; returns Nx2 points."""
    h, w = mask.shape
    win_h = h // cfg.n_windows
    margin = max(8, int(cfg.window_margin_m / cfg.m_per_px_x()))
    nz_y, nz_x = mask.nonzero()
    x_current = x_start
    collected: List[np.ndarray] = []
    for i in range(cfg.n_windows):
        y_hi = h - i * win_h
        y_lo = y_hi - win_h
        sel = (
            (nz_y >= y_lo)
            & (nz_y < y_hi)
            & (nz_x >= x_current - margin)
            & (nz_x < x_current + margin)
        )
        idx = sel.nonzero()[0]
        if len(idx) > 25:
            collected.append(idx)
            x_current = int(np.mean(nz_x[idx]))
    if not collected:
        return None
    idx = np.concatenate(collected)
    return np.stack([nz_x[idx], nz_y[idx]], axis=1)


def _classify_style(points: np.ndarray, cfg: Config) -> Tuple[float, str]:
    rows = np.unique(points[:, 1])
    span = points[:, 1].max() - points[:, 1].min() + 1
    fill = len(rows) / max(span, 1)
    return fill, "solid" if fill >= cfg.solid_fill_ratio else "dashed"


def _classify_color(bev_bgr: np.ndarray, points: np.ndarray, cfg: Config) -> str:
    hls = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HLS)
    sample = hls[points[:, 1], points[:, 0]]
    h, _, s = sample[:, 0], sample[:, 1], sample[:, 2]
    yellow = (h >= cfg.yellow_h[0]) & (h <= cfg.yellow_h[1]) & (s >= cfg.yellow_s_min)
    return "yellow" if yellow.mean() > 0.4 else "white"


def find_lane_lines(mask: np.ndarray, bev_bgr: np.ndarray, cfg: Config) -> List[LaneLine]:
    h, w = mask.shape
    hist = (mask[h // 2 :, :] > 0).sum(axis=0).astype(np.float32)
    hist = cv2.GaussianBlur(hist.reshape(1, -1), (0, 0), 3).ravel()
    min_value = cfg.peak_min_ratio * (h / 2)
    min_distance = max(4, int(cfg.peak_min_distance_m / cfg.m_per_px_x()))

    lines: List[LaneLine] = []
    for peak in _find_peaks(hist, min_value, min_distance):
        pts = _trace(mask, peak, cfg)
        if pts is None or len(pts) < 60:
            continue
        if pts[:, 1].max() - pts[:, 1].min() < h * 0.25:
            continue
        rows = len(np.unique(pts[:, 1]))
        if len(pts) / max(rows, 1) > cfg.max_line_width_m / cfg.m_per_px_x():
            continue  # too wide to be paint (kerb, barrier, glare)
        fit = np.polyfit(pts[:, 1], pts[:, 0], 2)
        fill, style = _classify_style(pts, cfg)
        color = _classify_color(bev_bgr, pts, cfg)
        x_px = float(np.polyval(fit, h - 1))
        lines.append(
            LaneLine(
                x_px=x_px,
                x_m=x_px * cfg.m_per_px_x(),
                fit=fit,
                fill_ratio=fill,
                style=style,
                color=color,
                points=pts,
            )
        )

    # deduplicate lines that converged onto the same marking
    lines.sort(key=lambda ln: ln.x_px)
    deduped: List[LaneLine] = []
    for line in lines:
        if deduped and abs(line.x_px - deduped[-1].x_px) < min_distance:
            if line.fill_ratio > deduped[-1].fill_ratio:
                deduped[-1] = line
            continue
        deduped.append(line)
    return deduped


def estimate_scale(lines: List[LaneLine], cfg: Config) -> float:
    """Metres per pixel derived from the detected markings.

    The ROI trapezoid is rarely calibrated, so absolute distances from the
    config are unreliable.  Assuming the *median* spacing between neighbouring
    markings is a nominal lane makes the lane count robust against a sloppy ROI.
    Falls back to the static config value when there is nothing to measure.
    """
    if not cfg.auto_scale:
        return cfg.m_per_px_x()
    xs = sorted(line.x_px for line in lines)
    min_distance = max(4.0, cfg.peak_min_distance_m / cfg.m_per_px_x())
    gaps = [b - a for a, b in zip(xs, xs[1:]) if b - a >= min_distance]
    if not gaps:
        return cfg.m_per_px_x()
    return cfg.nominal_lane_width_m / float(np.median(gaps))


def _gap_is_hatched(left_m: float, right_m: float, zones: List[HatchZone]) -> bool:
    """True when a chevron area covers most of the space between two lines."""
    width = right_m - left_m
    if width <= 0:
        return False
    for zone in zones:
        overlap = min(right_m, zone.x_range_m[1]) - max(left_m, zone.x_range_m[0])
        if overlap / width > 0.6:
            return True
    return False


def count_lanes(lines: List[LaneLine], zones: List[HatchZone], cfg: Config) -> List[Lane]:
    """Turn detected lines into drivable lanes.

    * gaps narrower than ``min_lane_width_m`` are ignored (double lines, kerb);
    * gaps covered by a chevron area are not drivable lanes;
    * gaps much wider than a lane are split into several lanes, which recovers
      lanes whose separator is worn out or occluded by traffic.
    """
    lanes: List[Lane] = []
    ordered = sorted(lines, key=lambda ln: ln.x_m)
    for left, right in zip(ordered, ordered[1:]):
        width = right.x_m - left.x_m
        if width < cfg.min_lane_width_m:
            continue
        if _gap_is_hatched(left.x_m, right.x_m, zones):
            continue
        if width <= cfg.max_lane_width_m:
            lanes.append(Lane(len(lanes) + 1, left.x_m, right.x_m, inferred=False))
            continue
        n = max(1, int(round(width / cfg.nominal_lane_width_m)))
        step = width / n
        for k in range(n):
            lanes.append(
                Lane(
                    len(lanes) + 1,
                    left.x_m + k * step,
                    left.x_m + (k + 1) * step,
                    inferred=True,
                )
            )
    for i, lane in enumerate(lanes, start=1):
        lane.index = i
    return lanes
