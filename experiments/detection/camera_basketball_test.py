#!/usr/bin/env python3
"""Camera-only basketball detection test for AACVision.

This program does NOT import or control GPIO, motors, encoders, LiDAR, or SLAM.
It uses only the Intel RealSense color/depth streams and an Ultralytics YOLO
model. COCO's ``sports ball`` detection is checked for orange/brown color and
roughly round proportions so a normal basketball can be identified.

Examples:
    python aacvision_camera_basketball_test.py --duration 60
    python aacvision_camera_basketball_test.py --duration 60 --display
    python aacvision_camera_basketball_test.py --duration 60 --loose
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pyrealsense2 as rs

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
DEFAULT_MODEL = "yolo26n.pt"
DEFAULT_OUTPUT_DIR = Path("/home/pi/aacvision_basketball_test")

# OpenCV HSV hue range is 0..179. These limits cover standard orange/brown
# basketballs under a fairly broad range of indoor lighting.
ORANGE_LOWER = np.array([2, 50, 30], dtype=np.uint8)
ORANGE_UPPER = np.array([36, 255, 255], dtype=np.uint8)

RUNNING = True


def stop_handler(_signum: int, _frame: Any) -> None:
    global RUNNING
    RUNNING = False


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def class_name_for(result: Any, class_id: int) -> str:
    names = getattr(result, "names", {})
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def appearance_evidence(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    x1 = clamp(x1, 0, width - 1)
    x2 = clamp(x2, x1 + 1, width)
    y1 = clamp(y1, 0, height - 1)
    y2 = clamp(y2, y1 + 1, height)

    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    roundness = min(box_width, box_height) / max(box_width, box_height)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0, roundness

    # Ignore ten percent of the box border, where background pixels are common.
    border_y = max(1, int(crop.shape[0] * 0.10))
    border_x = max(1, int(crop.shape[1] * 0.10))
    if crop.shape[0] > 2 * border_y and crop.shape[1] > 2 * border_x:
        crop = crop[border_y:-border_y, border_x:-border_x]

    hsv = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    orange_fraction = float(np.count_nonzero(mask)) / max(1, mask.size)
    return orange_fraction, roundness


def depth_from_bbox(
    depth_image: np.ndarray,
    bbox: tuple[int, int, int, int],
    depth_scale: float,
) -> Optional[float]:
    x1, y1, x2, y2 = bbox
    height, width = depth_image.shape[:2]
    inset_x = int((x2 - x1) * 0.25)
    inset_y = int((y2 - y1) * 0.25)
    ix1 = clamp(x1 + inset_x, 0, width - 1)
    ix2 = clamp(x2 - inset_x, ix1 + 1, width)
    iy1 = clamp(y1 + inset_y, 0, height - 1)
    iy2 = clamp(y2 - inset_y, iy1 + 1, height)

    crop_m = depth_image[iy1:iy2, ix1:ix2].astype(np.float32) * depth_scale
    valid = crop_m[(crop_m > 0.15) & (crop_m < 8.0) & np.isfinite(crop_m)]
    if valid.size < 10:
        return None
    return float(np.percentile(valid, 35.0))


def select_basketball(
    result: Any,
    frame: np.ndarray,
    depth_image: np.ndarray,
    depth_scale: float,
    confidence: float,
    orange_min: float,
    roundness_min: float,
    loose: bool,
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    boxes = getattr(result, "boxes", None)
    candidates: list[dict[str, Any]] = []
    if boxes is None or len(boxes) == 0:
        return None, candidates

    frame_area = float(frame.shape[0] * frame.shape[1])
    for box in boxes:
        try:
            class_id = int(box.cls[0].item())
            ai_confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
        except Exception:
            continue

        class_name = class_name_for(result, class_id).strip().lower()
        if class_name not in ("sports ball", "basketball"):
            continue
        if ai_confidence < confidence:
            continue

        x1 = clamp(x1, 0, frame.shape[1] - 1)
        x2 = clamp(x2, x1 + 1, frame.shape[1])
        y1 = clamp(y1, 0, frame.shape[0] - 1)
        y2 = clamp(y2, y1 + 1, frame.shape[0])
        bbox = (x1, y1, x2, y2)
        area_fraction = ((x2 - x1) * (y2 - y1)) / frame_area
        if area_fraction < 0.0005:
            continue

        orange_fraction, roundness = appearance_evidence(frame, bbox)
        accepted = loose or (
            orange_fraction >= orange_min and roundness >= roundness_min
        )
        distance_m = depth_from_bbox(depth_image, bbox, depth_scale)
        score = ai_confidence + 0.30 * orange_fraction + 0.10 * roundness
        candidate = {
            "bbox": bbox,
            "class_name": class_name,
            "confidence": ai_confidence,
            "orange": orange_fraction,
            "roundness": roundness,
            "distance_m": distance_m,
            "accepted": accepted,
            "score": score,
        }
        candidates.append(candidate)

    accepted_candidates = [item for item in candidates if item["accepted"]]
    best = max(accepted_candidates, key=lambda item: item["score"], default=None)
    return best, candidates


def annotate(
    frame: np.ndarray,
    best: Optional[dict[str, Any]],
    candidates: list[dict[str, Any]],
    fps: float,
    loose: bool,
) -> np.ndarray:
    output = frame.copy()

    # Draw rejected sports-ball candidates in amber so color-filter failures are visible.
    for item in candidates:
        x1, y1, x2, y2 = item["bbox"]
        accepted = bool(item["accepted"])
        color = (0, 255, 0) if accepted else (0, 180, 255)
        thickness = 3 if accepted else 2
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        label = (
            f"sports ball {item['confidence']:.2f} "
            f"orange {item['orange']:.2f} round {item['roundness']:.2f}"
        )
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    status = "BASKETBALL FOUND" if best is not None else "SEARCHING"
    status_color = (0, 255, 0) if best is not None else (0, 180, 255)
    cv2.putText(
        output,
        status,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        status_color,
        2,
        cv2.LINE_AA,
    )
    mode = "LOOSE: any YOLO sports ball" if loose else "NORMAL: sports ball + orange + round"
    cv2.putText(
        output,
        f"{mode} | loop {fps:.1f} FPS",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if best is not None:
        distance = "unknown" if best["distance_m"] is None else f"{best['distance_m']:.2f} m"
        cv2.putText(
            output,
            f"confidence {best['confidence']:.2f} | depth {distance}",
            (16, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test only the RealSense basketball detector; motors and LiDAR are never used."
    )
    parser.add_argument("--duration", type=float, default=60.0, help="test duration in seconds; 0 runs until Ctrl+C")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model name or path")
    parser.add_argument("--confidence", type=float, default=0.20, help="minimum YOLO confidence")
    parser.add_argument("--orange-min", type=float, default=0.025, help="minimum orange fraction in ball box")
    parser.add_argument("--roundness-min", type=float, default=0.35, help="minimum width/height roundness score")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO inference image size")
    parser.add_argument("--display", action="store_true", help="show a live OpenCV window; requires a graphical display")
    parser.add_argument("--loose", action="store_true", help="accept any YOLO sports-ball result; useful for diagnosing the color filter")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for latest and detection images")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    if args.display and not os.environ.get("DISPLAY"):
        print("WARNING: --display was requested but DISPLAY is not set. Continuing headless.")
        args.display = False

    try:
        from ultralytics import YOLO
    except ImportError as error:
        print("ERROR: ultralytics is not installed in this Python environment.")
        print("Activate it first: source /home/pi/aacvision-env/bin/activate")
        raise SystemExit(2) from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.output_dir / "latest_camera_test.jpg"

    print("=" * 72)
    print("AACVISION CAMERA-ONLY BASKETBALL TEST")
    print("No GPIO, motors, encoders, LiDAR, or SLAM will be initialized.")
    print(f"Model: {args.model}")
    print(f"Mode: {'LOOSE sports-ball diagnostic' if args.loose else 'normal orange basketball'}")
    print(f"Output: {args.output_dir}")
    print("Place the basketball 0.5-4 m in front of the RealSense camera.")
    print("Press Ctrl+C to stop.")
    print("=" * 72)

    print("Loading YOLO model...")
    model = YOLO(args.model)
    print("Starting RealSense...")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)
    config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)

    try:
        profile = pipeline.start(config)
    except Exception as error:
        print(f"ERROR: RealSense could not start: {error}")
        return 3

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)

    try:
        for _ in range(8):
            frames = pipeline.wait_for_frames(timeout_ms=3000)
            align.process(frames)
    except Exception as error:
        pipeline.stop()
        print(f"ERROR: RealSense warm-up failed: {error}")
        return 4

    print("Camera ready. Looking for a basketball...")
    started = time.monotonic()
    last_status = 0.0
    last_save = 0.0
    last_detection_save = 0.0
    last_loop = time.monotonic()
    loop_fps = 0.0
    found_frames = 0
    total_inferences = 0

    try:
        while RUNNING:
            now = time.monotonic()
            if args.duration > 0 and now - started >= args.duration:
                break

            frames = pipeline.wait_for_frames(timeout_ms=3000)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.confidence,
                verbose=False,
                device="cpu",
            )
            total_inferences += 1
            result = results[0] if results else None
            if result is None:
                best, candidates = None, []
            else:
                best, candidates = select_basketball(
                    result,
                    frame,
                    depth_image,
                    depth_scale,
                    args.confidence,
                    args.orange_min,
                    args.roundness_min,
                    args.loose,
                )

            current = time.monotonic()
            elapsed_loop = max(1e-6, current - last_loop)
            instant_fps = 1.0 / elapsed_loop
            loop_fps = instant_fps if loop_fps == 0.0 else 0.15 * instant_fps + 0.85 * loop_fps
            last_loop = current

            annotated = annotate(frame, best, candidates, loop_fps, args.loose)

            # Always refresh a single latest image for headless SSH testing.
            if current - last_save >= 1.0:
                cv2.imwrite(str(latest_path), annotated)
                last_save = current

            if best is not None:
                found_frames += 1
                distance = "unknown" if best["distance_m"] is None else f"{best['distance_m']:.2f} m"
                print(
                    "FOUND basketball | "
                    f"YOLO={best['confidence']:.2f} orange={best['orange']:.2f} "
                    f"round={best['roundness']:.2f} depth={distance}"
                )
                if current - last_detection_save >= 2.0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    detection_path = args.output_dir / f"basketball_detected_{timestamp}.jpg"
                    cv2.imwrite(str(detection_path), annotated)
                    print(f"Saved detection image: {detection_path}")
                    last_detection_save = current
            elif current - last_status >= 2.0:
                if candidates:
                    strongest = max(candidates, key=lambda item: item["confidence"])
                    print(
                        "YOLO sees a sports ball, but basketball checks rejected it | "
                        f"YOLO={strongest['confidence']:.2f} "
                        f"orange={strongest['orange']:.2f}/{args.orange_min:.2f} "
                        f"round={strongest['roundness']:.2f}/{args.roundness_min:.2f}"
                    )
                else:
                    print("Searching... no YOLO sports-ball detection")
                last_status = current

            if args.display:
                cv2.imshow("AACVision Basketball Camera Test", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    except Exception as error:
        print(f"ERROR during camera test: {error}")
        return 5
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        if args.display:
            cv2.destroyAllWindows()

    print("=" * 72)
    print("CAMERA TEST COMPLETE")
    print(f"Inferences: {total_inferences}")
    print(f"Frames with accepted basketball: {found_frames}")
    print(f"Latest annotated image: {latest_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
