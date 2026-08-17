"""Command line interface.

Examples
--------
    python -m road_detect.cli image road.jpg --out out/road.png --json out/road.json
    python -m road_detect.cli video drive.mp4 --out out/drive.mp4
    python -m road_detect.cli roi road.jpg --out out/roi.png   # tune the trapezoid
    python -m road_detect.cli camera 0 --record out/cam.mp4      # live webcam
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, deque
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


def _open_camera(source: str, width: int, height: int) -> cv2.VideoCapture:
    """Open a webcam index or a stream URL, trying the OS specific backends."""
    if source.isdigit():
        index = int(source)
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_ANY):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                break
            cap.release()
        else:
            raise SystemExit(
                f"cannot open camera {index}: try another index (0/1/2), close other "
                "apps using the camera, or check the Windows camera privacy setting"
            )
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f"cannot open stream: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def run_camera(args: argparse.Namespace) -> int:
    """Live detection from a webcam or a phone/IP camera stream.

    Keys: ``q``/ESC quit, ``s`` snapshot, ``b`` bird's eye view, ``r`` ROI.
    """
    cfg = _load_cfg(args.config)
    cap = _open_camera(args.input, args.width, args.height)
    detector = RoadMarkingDetector(cfg)

    writer = None
    history: deque = deque(maxlen=max(1, args.smooth))
    show_bev = False
    show_roi = False
    frames = 0
    started = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera returned no frame, stopping")
                break
            frames += 1
            result = detector.detect(frame)
            history.append(result.lane_count)
            # the mode over the last frames is far steadier than a single frame
            stable = Counter(history).most_common(1)[0][0]

            view = visualize.draw_overlay(frame, result, cfg)
            if show_roi:
                view = visualize.draw_roi(view, cfg)
            fps = frames / max(time.time() - started, 1e-6)
            cv2.putText(
                view, f"stable lanes: {stable}   {fps:.1f} fps", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
            )

            if args.record:
                if writer is None:
                    _ensure_parent(args.record)
                    h, w = view.shape[:2]
                    writer = cv2.VideoWriter(
                        args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h)
                    )
                writer.write(view)

            if args.headless:
                print(
                    json.dumps({"frame": frames, "lane_count": result.lane_count,
                                "stable": stable}),
                    flush=True,
                )
                if args.max_frames and frames >= args.max_frames:
                    break
                continue

            cv2.imshow("road_detect  (q quit, s save, b bev, r roi)", view)
            if show_bev:
                cv2.imshow("bird eye", visualize.draw_bev(result, cfg))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("b"):
                show_bev = not show_bev
                if not show_bev:
                    cv2.destroyWindow("bird eye")
            if key == ord("r"):
                show_roi = not show_roi
            if key == ord("s"):
                path = os.path.join(args.snapshot_dir, f"cam_{int(time.time())}.png")
                _ensure_parent(path)
                cv2.imwrite(path, view)
                print(f"saved {path}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
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

    for name, handler in (
        ("image", run_image),
        ("video", run_video),
        ("roi", run_roi),
        ("camera", run_camera),
    ):
        p = sub.add_parser(name)
        p.set_defaults(func=handler)
        p.add_argument("--config", help="JSON config file")
        if name == "camera":
            p.add_argument(
                "input", nargs="?", default="0",
                help="camera index (0, 1, ...) or a stream URL",
            )
            p.add_argument("--width", type=int, default=1280)
            p.add_argument("--height", type=int, default=720)
            p.add_argument("--smooth", type=int, default=15,
                           help="frames used for the stable lane count")
            p.add_argument("--record", help="write the annotated live view to a video")
            p.add_argument("--snapshot-dir", default="out")
            p.add_argument("--headless", action="store_true",
                           help="no preview window (server / WSL without GUI)")
            p.add_argument("--max-frames", type=int, default=0,
                           help="stop after N frames in headless mode")
            continue
        p.add_argument("input")
        p.add_argument("--out", help="annotated output file")
        if name == "image":
            p.add_argument("--json", help="write the result as JSON")
            p.add_argument("--bev", help="write the bird's eye debug view")

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
