import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from road_detect import Config, RoadMarkingDetector  # noqa: E402
from road_detect.lanes import LaneLine, count_lanes, estimate_scale  # noqa: E402
from tools.make_synthetic import SCENES, build  # noqa: E402


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_lane_count_matches_ground_truth(scene):
    image, expected = build(scene)
    result = RoadMarkingDetector().detect(image)
    assert result.lane_count == expected


def test_chevron_area_is_detected_and_not_counted_as_lane():
    image, _ = build("two_lanes_chevron")
    result = RoadMarkingDetector().detect(image)
    assert len(result.hatch_zones) == 1
    assert result.hatch_zones[0].stripes >= 3


def test_line_style_and_color():
    image, _ = build("three_lanes")
    result = RoadMarkingDetector().detect(image)
    styles = [line.style for line in result.lines]
    assert styles == ["solid", "dashed", "dashed", "solid"]
    assert result.lines[0].color == "yellow"


def test_plain_road_has_no_lanes():
    image = np.full((720, 1280, 3), 70, np.uint8)
    result = RoadMarkingDetector().detect(image)
    assert result.lane_count == 0
    assert result.lines == []


def _line(x_px: float, cfg: Config) -> LaneLine:
    return LaneLine(
        x_px=x_px,
        x_m=x_px * cfg.m_per_px_x(),
        fit=np.array([0.0, 0.0, x_px]),
        fill_ratio=1.0,
        style="solid",
        color="white",
    )


def test_wide_gap_is_split_into_several_lanes():
    cfg = Config(auto_scale=False)
    lines = [_line(0, cfg), _line(7.0 / cfg.m_per_px_x(), cfg)]
    lanes = count_lanes(lines, [], cfg)
    assert len(lanes) == 2
    assert all(lane.inferred for lane in lanes)


def test_double_line_is_not_a_lane():
    cfg = Config(auto_scale=False)
    lines = [_line(0, cfg), _line(0.3 / cfg.m_per_px_x(), cfg)]
    assert count_lanes(lines, [], cfg) == []


def test_auto_scale_uses_median_spacing():
    cfg = Config()
    lines = [_line(x, cfg) for x in (100, 200, 300)]
    assert estimate_scale(lines, cfg) == pytest.approx(cfg.nominal_lane_width_m / 100)
