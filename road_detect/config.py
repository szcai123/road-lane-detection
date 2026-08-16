"""Configuration for the road marking detector.

All geometry is expressed in fractions of the input image so that the same
config works for different resolutions.  The only value that really needs to be
calibrated per camera is :attr:`Config.bev_width_m` together with the ROI
trapezoid: they define the pixel -> metre mapping used for lane counting.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple
import json


@dataclass
class Config:
    # ------------------------------------------------------------------ ROI
    # Trapezoid (in image fractions) that is warped to a bird's eye view.
    # Order: bottom-left, top-left, top-right, bottom-right.
    roi_bottom_y: float = 0.95
    roi_top_y: float = 0.62
    roi_bottom_left_x: float = 0.12
    roi_bottom_right_x: float = 0.88
    roi_top_left_x: float = 0.44
    roi_top_right_x: float = 0.56

    # -------------------------------------------------------------- bird eye
    bev_width: int = 600
    bev_height: int = 800
    #: real world width covered by the warped image, in metres
    bev_width_m: float = 12.0
    #: real world length covered by the warped image, in metres
    bev_length_m: float = 30.0

    # ------------------------------------------------------------ thresholds
    white_l_min: int = 175       # HLS lightness for white paint
    white_s_max: int = 90        # HLS saturation for white paint
    yellow_h: Tuple[int, int] = (15, 38)
    yellow_s_min: int = 70
    yellow_l_min: int = 70
    #: width (px, bird's eye) of the top-hat kernel used to isolate thin paint
    tophat_width: int = 17
    tophat_thresh: int = 25

    # ----------------------------------------------------------- lane lines
    #: minimum column-histogram score (fraction of rows) for a line candidate
    peak_min_ratio: float = 0.12
    #: minimum distance between two detected lines, in metres
    peak_min_distance_m: float = 1.2
    #: a traced line covering more than this fraction of rows is "solid"
    solid_fill_ratio: float = 0.72
    #: painted lines are thin; wider structures (kerbs, barriers, wet asphalt
    #: reflections) are rejected.  Uses the static config scale, not auto_scale.
    max_line_width_m: float = 0.7
    n_windows: int = 12
    window_margin_m: float = 0.5

    # ---------------------------------------------------------- hatch / 鱼骨线
    #: minimum length/width ratio for a stripe of a chevron (fishbone) area
    hatch_min_elongation: float = 2.2
    #: a stripe must deviate from the lane direction by at least this angle
    hatch_min_angle_deg: float = 22.0
    #: number of parallel stripes required to call it a hatched area
    hatch_min_stripes: int = 3
    #: minimum length of one stripe, in metres
    hatch_min_stripe_len_m: float = 1.0
    #: neighbouring stripes of one area must be closer than this, in metres
    hatch_max_stripe_spacing_m: float = 4.0
    #: a stripe must be this much brighter than the surrounding road surface
    hatch_min_contrast: int = 25
    #: stripes of one group must not differ by more than this angle
    hatch_angle_tolerance_deg: float = 18.0
    #: chevron stripes repeat regularly: length ratio and spacing variation caps
    hatch_max_length_ratio: float = 3.0
    hatch_max_spacing_cv: float = 0.6
    #: a stripe pattern spanning most of the road is asphalt texture, not paint
    hatch_max_width_ratio: float = 0.6

    # ---------------------------------------------------------- lane counting
    min_lane_width_m: float = 2.2
    max_lane_width_m: float = 4.6
    #: gaps wider than max_lane_width_m are split into N implicit lanes
    nominal_lane_width_m: float = 3.5
    #: derive the pixel->metre scale from the detected markings by assuming the
    #: median line spacing equals ``nominal_lane_width_m``.  Makes the lane count
    #: independent of the ROI calibration; disable if the camera is calibrated.
    auto_scale: bool = True

    def m_per_px_x(self) -> float:
        return self.bev_width_m / self.bev_width

    def m_per_px_y(self) -> float:
        return self.bev_length_m / self.bev_height

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json_file(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        if "yellow_h" in data:
            data["yellow_h"] = tuple(data["yellow_h"])
        return cls(**data)
