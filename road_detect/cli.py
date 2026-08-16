"""Command line interface.

Examples
--------
    python -m road_detect.cli image road.jpg --out out/road.png --json out/road.json
    python -m road_detect.cli video drive.mp4 --out out/drive.mp4
    python -m road_detect.cli roi road.jpg --out out/roi.png   # tune the trapezoid
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import cv2

from .config import Config
from .detector import RoadMarkingDetector
from . import visualize


def _load_cfg(path: Optional[str]) -> Config:
    return Config.from_json_file(path) if path else Config()


def _ensure_parent(path: Optional[str]) -> None:
    if path:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)


def run_image(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    image = cv2.imread(args.input)
    if image is None:
        raise SystemExit(f"cannot read image: {args.input}")

    result = RoadMarkingDetector(cfg).detect(image)
    payload = result.as_dict()
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.json:
        _ensure_parent(args.json)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
    if args.out:
        _ensure_parent(args.out)
        cv2.imwrite(args.out, visualize.draw_overlay(image, result, cfg))
    if args.bev:
        _ensure_parent(args.bev)
        cv2.imwrite(args.bev, visualize.draw_bev(result, cfg))
    return 0


def run_video(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.input}")

    detector = RoadMarkingDetector(cfg)
    writer = None
    counts = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = detector.detect(frame)
        counts.append(result.lane_count)
        if args.out:
            if writer is None:
                _ensure_parent(args.out)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            writer.write(visualize.draw_overlay(frame, result, cfg))
    cap.release()
    if writer is not None:
        writer.release()

    if counts:
        # the mode is much more stable than any single frame
        stable = max(set(counts), key=counts.count)
        print(json.dumps({"frames": len(counts), "lane_count": stable}, indent=2))
    return 0


def run_roi(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    image = cv2.imread(args.input)
    if image is None:
        raise SystemExit(f"cannot read image: {args.input}")
    out = args.out or "roi.png"
    _ensure_parent(out)
    cv2.imwrite(out, visualize.draw_roi(image, cfg))
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="road_detect", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, handler in (("image", run_image), ("video", run_video), ("roi", run_roi)):
        p = sub.add_parser(name)
        p.add_argument("input")
        p.add_argument("--out", help="annotated output file")
        p.add_argument("--config", help="JSON config file")
        p.set_defaults(func=handler)
        if name == "image":
            p.add_argument("--json", help="write the result as JSON")
            p.add_argument("--bev", help="write the bird's eye debug view")

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
