"""High level API: image / video -> lane count + marking description."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import Config
from .lanes import Lane, LaneLine, count_lanes, estimate_scale, find_lane_lines
from .markings import HatchZone, find_hatch_zones, paint_mask
from . import perspective


@dataclass
class DetectionResult:
    lane_count: int
    lanes: List[Lane]
    lines: List[LaneLine]
    hatch_zones: List[HatchZone]
    #: metres per bird's eye pixel actually used (see :func:`estimate_scale`)
    m_per_px_x: float = 0.0
    bev_bgr: np.ndarray = field(repr=False, default=None)
    mask: np.ndarray = field(repr=False, default=None)
    m_inv: np.ndarray = field(repr=False, default=None)
    image_shape: tuple = ()

    def as_dict(self) -> dict:
        return {
            "lane_count": self.lane_count,
            "lanes": [lane.as_dict() for lane in self.lanes],
            "lines": [line.as_dict() for line in self.lines],
            "hatch_zones": [zone.as_dict() for zone in self.hatch_zones],
            "m_per_px_x": round(self.m_per_px_x, 4),
        }


class RoadMarkingDetector:
    """Detects longitudinal lane markings and chevron areas, and counts lanes."""

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        cfg = self.cfg
        m, m_inv = perspective.transforms(image_bgr.shape, cfg)
        bev = perspective.warp(image_bgr, m, cfg)

        mask = paint_mask(bev, cfg)
        zones, cleaned = find_hatch_zones(mask, cfg, bev)
        lines = find_lane_lines(cleaned, bev, cfg)

        scale = estimate_scale(lines, cfg)
        for line in lines:
            line.x_m = line.x_px * scale
        for zone in zones:
            xs = zone.polygon[:, 0, 0]
            zone.x_range_m = (float(xs.min() * scale), float(xs.max() * scale))

        lanes = count_lanes(lines, zones, cfg)

        return DetectionResult(
            lane_count=len(lanes),
            lanes=lanes,
            lines=lines,
            hatch_zones=zones,
            m_per_px_x=scale,
            bev_bgr=bev,
            mask=cleaned,
            m_inv=m_inv,
            image_shape=image_bgr.shape,
        )
