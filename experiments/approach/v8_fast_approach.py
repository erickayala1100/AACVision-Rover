#!/usr/bin/env python3
"""
AACVision Stepwise Basketball Search and Room Mapping
----------------------------------------------
Different strategy from continuous reactive obstacle avoidance:
instead of reacting to sensor readings every loop tick while driving
(which is fragile -- readings differ wildly depending on exactly
where the rover is mid-maneuver, especially in doorways), this uses a
discrete STOP -> SCAN -> DECIDE -> MOVE cycle:

  1. Stop completely.
  2. Wait for one fresh, full 360-degree LiDAR scan.
  3. Divide that scan into sectors around the rover and find the
     sector with the most open space (biased toward continuing
     roughly forward rather than doubling back).
  4. Rotate in place to face that direction, using SLAM's
     corrected heading for accuracy.
  5. Drive forward a bounded distance toward it (a simple, narrow
     "is something suddenly right in front of me" check is the only
     thing monitored during the drive itself -- the direction
     decision itself is never re-litigated mid-step).
  6. Repeat.

This finds doorways/openings naturally, since an opening just shows
up as "the most open sector" in a full scan -- no special-casing
needed, and no risk of dithering between two nearly-equal readings
mid-turn, since the decision is made once per stop, not every tick.

SAFETY
-------
Test with wheels lifted first, then in a clear room. Ctrl+C stops
immediately at any time.

Encoder pins (BCM):
    Front-left:  A=GPIO4,  B=GPIO17
    Front-right: A=GPIO27, B=GPIO22
    Rear-left:   A=GPIO10, B=GPIO9
    Rear-right:  A=GPIO11, B=GPIO8

Motor pins (BCM):
    front_left:  enable=12, in1=5,  in2=6
    front_right: enable=13, in1=20, in2=21
    rear_left:   enable=18, in1=23, in2=24
    rear_right:  enable=19, in1=15, in2=26
"""

import math
import os
import re
import signal
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import time
from datetime import datetime
from pathlib import Path

import RPi.GPIO as GPIO
import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image

from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser

# ============================================================
# CONFIRMED CALIBRATION
# ============================================================

WHEEL_DIAMETER_METERS = 0.080
ENCODER_COUNTS_PER_WHEEL_REV = 1232.0
TRACK_WIDTH_METERS = 0.296

ENCODER_POLARITY = {
    "front_left": 1,
    "front_right": -1,
    "rear_left": 1,
    "rear_right": -1,
}

MOTOR_POLARITY = {
    "front_left": 1,
    "front_right": 1,
    "rear_left": 1,
    "rear_right": 1,
}

# ============================================================
# GPIO PIN MAP (BCM numbering)
# ============================================================

ENCODERS = {
    "front_left": {"a": 4, "b": 17},
    "front_right": {"a": 27, "b": 22},
    "rear_left": {"a": 10, "b": 9},
    "rear_right": {"a": 11, "b": 8},
}

MOTORS = {
    "front_left": {"enable": 12, "in1": 5, "in2": 6},
    "front_right": {"enable": 13, "in1": 20, "in2": 21},
    "rear_left": {"enable": 18, "in1": 23, "in2": 24},
    "rear_right": {"enable": 19, "in1": 15, "in2": 26},
}

PWM_FREQUENCY = 1000

# ============================================================
# STEPWISE EXPLORATION SETTINGS
# ============================================================

EXPLORE_SPEED = 22
TURN_SPEED = 20
REVERSE_SPEED = 18

SECTOR_COUNT = 32                  # 11.25 degrees per sector (compromise)
MIN_STEP_CLEARANCE_M = 0.35        # ignore sectors tighter than this
MAX_USEFUL_RANGE_M = 4.0           # cap "how open" for scoring purposes
MAX_STEP_METERS = 0.8              # never drive more than this in one step
STEP_DISTANCE_FRACTION = 0.6       # drive this fraction of measured clearance

# Only turn halfway toward the chosen direction rather than pointing
# straight at it -- keeps the rover more forward-biased instead of
# fully committing to sharp turns every cycle.
TURN_ANGLE_SCALE = 0.5

FRONT_EMERGENCY_DISTANCE = 0.25    # last-resort stop during a drive step
NARROW_FRONT_START = 345
NARROW_FRONT_END = 15

TURN_TOLERANCE_DEG = 8.0
MAX_TURN_SECONDS = 5.0
MAX_STEP_SECONDS = 6.0

# No exploration time limit: run until the basketball is reached or Ctrl+C.
MAX_CONSECUTIVE_STUCK = 3
STUCK_REVERSE_SECONDS = 0.8

# ============================================================
# LIDAR SETTINGS
# ============================================================

LIDAR_PROGRAM = "/home/pi/rplidar_sdk/output/Linux/Release/ultra_simple"
LIDAR_PORT = "/dev/ttyUSB0"
LIDAR_BAUD = "460800"

LIDAR_PATTERN = re.compile(
    r"theta:\s*([0-9.]+)\s+"
    r"Dist:\s*([0-9.]+)\s+"
    r"Q:\s*([0-9]+)"
)

LIDAR_STARTUP_GRACE = 10.0
LIDAR_TIMEOUT = 3.0

# ============================================================
# SLAM / LASER SETTINGS
# ============================================================

SCAN_SIZE = 360
SCAN_RATE_HZ = 10.0
DETECTION_ANGLE_DEGREES = 360
DISTANCE_NO_DETECTION_MM = 12000
DETECTION_MARGIN = 0
LIDAR_OFFSET_MM = 0

MAP_SIZE_PIXELS = 800
MAP_SIZE_METERS = 8.0
MAP_SAVE_INTERVAL = 3.0

PI_HOME = Path("/home/pi")
MAP_DIRECTORY = PI_HOME / "aacvision_maps"

# ============================================================
# INTEL REALSENSE D435 / BASKETBALL MISSION SETTINGS
# ============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
CAMERA_INFERENCE_SIZE = 320
CAMERA_DIRECTORY = PI_HOME / "aacvision_basketball_images"
CAMERA_LATEST_PATH = CAMERA_DIRECTORY / "latest_basketball_view.jpg"
DEFAULT_BASKETBALL_MODEL = "yolo26n.pt"

# These are the same thresholds that worked in the camera-only test.
BASKETBALL_CLASS_NAMES = ("sports ball", "basketball")
BASKETBALL_AI_MIN_CONFIDENCE = 0.20
BASKETBALL_MIN_BOX_AREA_FRACTION = 0.0005
ORANGE_HSV_LOWER = np.array([2, 50, 30], dtype=np.uint8)
ORANGE_HSV_UPPER = np.array([36, 255, 255], dtype=np.uint8)
ORANGE_MIN_FRACTION = 0.025
BASKETBALL_MIN_ROUNDNESS = 0.35
BASKETBALL_CONFIRM_SCORE = 0.70
BASKETBALL_MAX_DETECTION_AGE_SECONDS = 3.5
BASKETBALL_REACQUIRE_SECONDS = 8.0

# Stop and approach settings.
BASKETBALL_STOP_DISTANCE_M = 0.80
BASKETBALL_CENTER_TOLERANCE_PIXELS = 75
BASKETBALL_CAPTURE_CENTER_TOLERANCE_PIXELS = 75

# Faster approach profile. The Pi only completes roughly one YOLO inference
# every 1.5-2 seconds, so tiny 0.1-0.2 second movements waste most of the
# mission waiting while stationary. V8 uses distance-scaled movement pulses.
BASKETBALL_ALIGNMENT_TURN_SPEED = 18
BASKETBALL_APPROACH_FAST_SPEED = 22
BASKETBALL_APPROACH_MEDIUM_SPEED = 18
BASKETBALL_APPROACH_NEAR_SPEED = 13
BASKETBALL_FAR_DISTANCE_M = 2.00
BASKETBALL_NEAR_DISTANCE_M = 1.15
BASKETBALL_FAR_FORWARD_PULSE_SECONDS = 0.75
BASKETBALL_MEDIUM_FORWARD_PULSE_SECONDS = 0.45
BASKETBALL_NEAR_FORWARD_PULSE_SECONDS = 0.20
BASKETBALL_MIN_TURN_PULSE_SECONDS = 0.08
BASKETBALL_MAX_TURN_PULSE_SECONDS = 0.30
BASKETBALL_POST_MOTION_SETTLE_SECONDS = 0.03
BASKETBALL_CLOSE_CONFIRM_FRAMES = 1
BASKETBALL_DEPTH_MIN_VALID_PIXELS = 10
BASKETBALL_DEPTH_PERCENTILE = 35.0
CAMERA_STARTUP_TIMEOUT_SECONDS = 120.0


# ============================================================
# HELPERS
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def angle_in_sector(angle, start, end):
    angle %= 360.0
    start %= 360.0
    end %= 360.0

    if start <= end:
        return start <= angle <= end
    return angle >= start or angle <= end


def robust_distance(values):
    if not values:
        return float("inf")

    values = sorted(values)
    count = min(8, len(values))
    return sum(values[:count]) / count


def median_distance(values, min_points=3):
    """
    Used for exploration DECISION-making, not safety-critical stops.
    Flat surfaces hit at an angle (not straight-on) reflect LiDAR
    beams poorly -- a few grazing-angle readings can come back
    falsely short, which the closest-N-average in robust_distance
    treats as gospel. Median is far less swayed by a handful of noisy
    outliers. If too few points landed in a sector to trust a median,
    treat it as open/unknown rather than assuming it's blocked --
    sparse sectors are often a grazing edge, not a solid wall dead
    ahead.
    """
    if len(values) < min_points:
        return float("inf")

    values = sorted(values)
    mid = len(values) // 2

    if len(values) % 2 == 0:
        return (values[mid - 1] + values[mid]) / 2.0
    return values[mid]


def bin_scan_to_mm(scan):
    bins = [0] * SCAN_SIZE

    for angle_deg, distance_m, quality in scan:
        if quality <= 0 or distance_m <= 0:
            continue

        index = int(angle_deg) % SCAN_SIZE
        bins[index] = int(distance_m * 1000.0)

    return bins


def clean_map_image(image):
    threshold = 210

    lookup_table = []
    for value in range(256):
        if value >= threshold:
            lookup_table.append(255)
        else:
            ratio = value / threshold
            lookup_table.append(int((ratio ** 2) * threshold))

    return image.point(lookup_table)


def get_sector_clearances(scan, sector_count=SECTOR_COUNT):
    """
    Divide a full scan into equal angular sectors and return a
    decision-oriented clearance distance in each one (median-based,
    not closest-N-average -- see median_distance), capped at
    MAX_USEFUL_RANGE_M for scoring purposes.
    """
    sector_width = 360.0 / sector_count
    buckets = [[] for _ in range(sector_count)]

    for angle_deg, distance_m, quality in scan:
        if quality <= 0 or distance_m <= 0:
            continue

        index = int(angle_deg // sector_width) % sector_count
        buckets[index].append(distance_m)

    clearances = []
    for values in buckets:
        distance = median_distance(values)
        if not math.isfinite(distance):
            distance = MAX_USEFUL_RANGE_M
        clearances.append(min(distance, MAX_USEFUL_RANGE_M))

    return clearances, sector_width


def choose_target_sector(clearances, sector_width):
    """
    Pick the most promising direction: prefer sectors with more open
    space, mildly biased toward directions closer to "straight ahead"
    (0 degrees relative) so the rover doesn't spin randomly when
    several directions are similarly open. Returns
    (relative_angle_degrees, clearance_m) or None if nothing is open
    enough to move toward.
    """
    best_index = None
    best_score = -1.0

    for index, clearance in enumerate(clearances):
        if clearance < MIN_STEP_CLEARANCE_M:
            continue

        sector_center_deg = index * sector_width + sector_width / 2.0
        relative_angle = sector_center_deg
        if relative_angle > 180.0:
            relative_angle -= 360.0

        turn_penalty = abs(relative_angle) / 180.0  # 0 ahead .. 1 behind
        score = clearance * (1.0 - 0.4 * turn_penalty)

        if score > best_score:
            best_score = score
            best_index = index

    if best_index is None:
        return None

    sector_center_deg = best_index * sector_width + sector_width / 2.0
    relative_angle = sector_center_deg
    if relative_angle > 180.0:
        relative_angle -= 360.0

    return relative_angle, clearances[best_index]


def get_instant_center_clearance(lidar_state):
    """Quick narrow-cone check used only as a safety net while driving a step."""
    scan = lidar_state.snapshot()["scan"]
    values = [
        distance_m
        for angle, distance_m, quality in scan
        if quality > 0
        and distance_m > 0
        and angle_in_sector(angle, NARROW_FRONT_START, NARROW_FRONT_END)
    ]
    return robust_distance(values)


# ============================================================
# INTEL REALSENSE D435 BASKETBALL DETECTOR
# ============================================================

@dataclass
class BasketballObservation:
    visible: bool
    confirmed: bool
    confidence: float
    center_x: Optional[float]
    center_y: Optional[float]
    distance_m: Optional[float]
    orange_fraction: float
    roundness: float
    bbox: Optional[Tuple[int, int, int, int]]
    last_seen_time: float
    frame_time: float


class BasketballDetector(threading.Thread):
    """Runs the proven YOLO + orange + roundness RealSense detector."""

    def __init__(self, target_found_event, model_spec=DEFAULT_BASKETBALL_MODEL):
        super().__init__(daemon=True)
        self.target_found_event = target_found_event
        self.model_spec = model_spec
        self.running = True
        self.pipeline = None
        self.model = None
        self.error_message = None
        self.startup_stage = "created"
        self.depth_scale = 0.001
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_observation = self._empty_observation(0.0)
        self.confirm_score = 0.0
        self.smoothed_center_x = None
        self.smoothed_center_y = None
        self.smoothed_distance = None
        self.last_seen_time = float("-inf")
        self.last_latest_save = 0.0

    @staticmethod
    def _empty_observation(frame_time):
        return BasketballObservation(
            visible=False,
            confirmed=False,
            confidence=0.0,
            center_x=None,
            center_y=None,
            distance_m=None,
            orange_fraction=0.0,
            roundness=0.0,
            bbox=None,
            last_seen_time=float("-inf"),
            frame_time=frame_time,
        )

    @staticmethod
    def _class_name(result, class_id):
        names = getattr(result, "names", {})
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    @staticmethod
    def _appearance_evidence(frame, bbox):
        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]
        x1 = int(clamp(x1, 0, width - 1))
        x2 = int(clamp(x2, x1 + 1, width))
        y1 = int(clamp(y1, 0, height - 1))
        y2 = int(clamp(y2, y1 + 1, height))

        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        roundness = min(box_width, box_height) / max(box_width, box_height)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0, roundness

        border_y = max(1, int(crop.shape[0] * 0.10))
        border_x = max(1, int(crop.shape[1] * 0.10))
        if crop.shape[0] > 2 * border_y and crop.shape[1] > 2 * border_x:
            crop = crop[border_y:-border_y, border_x:-border_x]

        hsv = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ORANGE_HSV_LOWER, ORANGE_HSV_UPPER)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        orange_fraction = float(np.count_nonzero(mask)) / max(1, mask.size)
        return orange_fraction, roundness

    def _depth_from_bbox(self, depth_image, bbox):
        x1, y1, x2, y2 = bbox
        height, width = depth_image.shape[:2]
        inset_x = int((x2 - x1) * 0.25)
        inset_y = int((y2 - y1) * 0.25)
        ix1 = int(clamp(x1 + inset_x, 0, width - 1))
        ix2 = int(clamp(x2 - inset_x, ix1 + 1, width))
        iy1 = int(clamp(y1 + inset_y, 0, height - 1))
        iy2 = int(clamp(y2 - inset_y, iy1 + 1, height))

        crop_m = depth_image[iy1:iy2, ix1:ix2].astype(np.float32) * self.depth_scale
        valid = crop_m[(crop_m > 0.15) & (crop_m < 8.0) & np.isfinite(crop_m)]
        if valid.size < BASKETBALL_DEPTH_MIN_VALID_PIXELS:
            return None
        return float(np.percentile(valid, BASKETBALL_DEPTH_PERCENTILE))

    def _select_basketball(self, result, frame, depth_image):
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        frame_area = float(frame.shape[0] * frame.shape[1])
        best = None
        for box in boxes:
            try:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            except Exception:
                continue

            class_name = self._class_name(result, class_id).strip().lower()
            if class_name not in BASKETBALL_CLASS_NAMES:
                continue
            if confidence < BASKETBALL_AI_MIN_CONFIDENCE:
                continue

            x1 = int(clamp(x1, 0, frame.shape[1] - 1))
            x2 = int(clamp(x2, x1 + 1, frame.shape[1]))
            y1 = int(clamp(y1, 0, frame.shape[0] - 1))
            y2 = int(clamp(y2, y1 + 1, frame.shape[0]))
            bbox = (x1, y1, x2, y2)
            area_fraction = ((x2 - x1) * (y2 - y1)) / frame_area
            if area_fraction < BASKETBALL_MIN_BOX_AREA_FRACTION:
                continue

            orange_fraction, roundness = self._appearance_evidence(frame, bbox)
            if orange_fraction < ORANGE_MIN_FRACTION:
                continue
            if roundness < BASKETBALL_MIN_ROUNDNESS:
                continue

            distance_m = self._depth_from_bbox(depth_image, bbox)
            score = confidence + 0.30 * orange_fraction + 0.10 * roundness
            candidate = {
                "score": score,
                "confidence": confidence,
                "center_x": (x1 + x2) / 2.0,
                "center_y": (y1 + y2) / 2.0,
                "distance_m": distance_m,
                "orange_fraction": orange_fraction,
                "roundness": roundness,
                "bbox": bbox,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        return best

    def _annotate(self, frame, observation):
        output = frame.copy()
        if observation.bbox is not None:
            x1, y1, x2, y2 = observation.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
        status = "BASKETBALL FOUND" if observation.visible else "SEARCHING FOR BASKETBALL"
        color = (0, 255, 0) if observation.visible else (0, 180, 255)
        cv2.putText(output, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        if observation.visible:
            distance = "unknown" if observation.distance_m is None else f"{observation.distance_m:.2f}m"
            cv2.putText(
                output,
                f"YOLO {observation.confidence:.2f} orange {observation.orange_fraction:.2f} round {observation.roundness:.2f} depth {distance}",
                (16, 61),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                1,
                cv2.LINE_AA,
            )
        return output

    def _publish(self, detection, frame, now):
        if detection is not None:
            increment = 0.60 if detection["confidence"] >= 0.40 else 0.40
            self.confirm_score = min(1.0, self.confirm_score + increment)
            alpha = 0.70
            if self.smoothed_center_x is None:
                self.smoothed_center_x = detection["center_x"]
                self.smoothed_center_y = detection["center_y"]
            else:
                self.smoothed_center_x = alpha * detection["center_x"] + (1.0 - alpha) * self.smoothed_center_x
                self.smoothed_center_y = alpha * detection["center_y"] + (1.0 - alpha) * self.smoothed_center_y
            if detection["distance_m"] is not None:
                if self.smoothed_distance is None:
                    self.smoothed_distance = detection["distance_m"]
                else:
                    self.smoothed_distance = alpha * detection["distance_m"] + (1.0 - alpha) * self.smoothed_distance
            self.last_seen_time = now
            confirmed = self.confirm_score >= BASKETBALL_CONFIRM_SCORE
            observation = BasketballObservation(
                visible=True,
                confirmed=confirmed,
                confidence=detection["confidence"],
                center_x=self.smoothed_center_x,
                center_y=self.smoothed_center_y,
                distance_m=self.smoothed_distance,
                orange_fraction=detection["orange_fraction"],
                roundness=detection["roundness"],
                bbox=detection["bbox"],
                last_seen_time=self.last_seen_time,
                frame_time=now,
            )
            if confirmed:
                self.target_found_event.set()
        else:
            self.confirm_score = max(0.0, self.confirm_score - 0.20)
            previous = self.latest_observation
            observation = BasketballObservation(
                visible=False,
                confirmed=False,
                confidence=0.0,
                center_x=self.smoothed_center_x,
                center_y=self.smoothed_center_y,
                distance_m=self.smoothed_distance,
                orange_fraction=previous.orange_fraction,
                roundness=previous.roundness,
                bbox=previous.bbox,
                last_seen_time=self.last_seen_time,
                frame_time=now,
            )

        annotated = self._annotate(frame, observation)
        with self.lock:
            self.latest_frame = frame.copy()
            self.latest_observation = observation

        if now - self.last_latest_save >= 2.0:
            CAMERA_DIRECTORY.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(CAMERA_LATEST_PATH), annotated)
            self.last_latest_save = now

    def snapshot(self):
        with self.lock:
            observation = self.latest_observation
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        return observation, frame

    def reset_target(self):
        with self.lock:
            self.confirm_score = 0.0
            self.smoothed_center_x = None
            self.smoothed_center_y = None
            self.smoothed_distance = None
            self.last_seen_time = float("-inf")
            self.latest_observation = self._empty_observation(time.monotonic())
        self.target_found_event.clear()

    def save_target_photo(self):
        observation, frame = self.snapshot()
        if frame is None:
            raise RuntimeError("No RealSense frame is available to save.")
        CAMERA_DIRECTORY.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = CAMERA_DIRECTORY / f"basketball_reached_{timestamp}.jpg"
        annotated = self._annotate(frame, observation)
        cv2.putText(
            annotated,
            "MISSION COMPLETE - BASKETBALL REACHED",
            (16, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"Could not save {output_path}")
        try:
            os.chown(output_path, 1000, 1000)
        except PermissionError:
            pass
        return output_path

    def run(self):
        try:
            self.startup_stage = "importing Ultralytics"
            from ultralytics import YOLO

            self.startup_stage = "loading YOLO model"
            self.model = YOLO(self.model_spec)

            self.startup_stage = "starting RealSense"
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)
            config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
            profile = self.pipeline.start(config)
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            align = rs.align(rs.stream.color)

            self.startup_stage = "warming up RealSense"
            for _ in range(5):
                if not self.running:
                    return
                aligned = align.process(self.pipeline.wait_for_frames(timeout_ms=3000))
                color = aligned.get_color_frame()
                if color:
                    with self.lock:
                        self.latest_frame = np.asanyarray(color.get_data()).copy()

            self.startup_stage = "ready"
            while self.running:
                aligned = align.process(self.pipeline.wait_for_frames(timeout_ms=3000))
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                now = time.monotonic()
                with self.lock:
                    self.latest_frame = frame.copy()

                results = self.model.predict(
                    source=frame,
                    imgsz=CAMERA_INFERENCE_SIZE,
                    conf=BASKETBALL_AI_MIN_CONFIDENCE,
                    verbose=False,
                    device="cpu",
                )
                result = results[0] if results else None
                detection = None if result is None else self._select_basketball(result, frame, depth_image)
                self._publish(detection, frame, now)

        except Exception as error:
            self.startup_stage = "failed"
            self.error_message = str(error)
        finally:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass

    def stop_detector(self):
        self.running = False


# ============================================================
# BASKETBALL APPROACH
# ============================================================

def approach_basketball(motors, lidar_state, detector):
    """Center quickly, then use distance-scaled forward pulses to reach the ball."""
    motors.stop()
    last_frame_time = float("-inf")
    last_horizontal_error = 0.0

    print("\nBasketball confirmed. Beginning fast center-and-approach mode.")

    def motion_pulse(action, seconds):
        """Move for one bounded pulse, with live LiDAR emergency checking."""
        action()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            front_clearance = get_instant_center_clearance(lidar_state)
            if front_clearance <= FRONT_EMERGENCY_DISTANCE:
                motors.stop()
                print(f"\nEmergency stop during approach: obstacle at {front_clearance:.2f}m.")
                return False
            time.sleep(0.02)
        motors.stop()
        time.sleep(BASKETBALL_POST_MOTION_SETTLE_SECONDS)
        return True

    def turn_pulse_for_error(horizontal_error):
        # Large image errors receive a larger turn; small errors receive a short trim.
        normalized = min(1.0, abs(horizontal_error) / (CAMERA_WIDTH / 2.0))
        return (
            BASKETBALL_MIN_TURN_PULSE_SECONDS
            + normalized
            * (BASKETBALL_MAX_TURN_PULSE_SECONDS - BASKETBALL_MIN_TURN_PULSE_SECONDS)
        )

    while True:
        if detector.error_message:
            motors.stop()
            return "camera_error"

        observation, _frame = detector.snapshot()
        now = time.monotonic()

        # Wait only for a genuinely new YOLO result. The detector thread continues
        # collecting RealSense frames while the rover performs each motion pulse.
        if observation.frame_time <= last_frame_time + 1e-6:
            time.sleep(0.02)
            continue
        last_frame_time = observation.frame_time

        visible = (
            observation.visible
            and now - observation.frame_time <= BASKETBALL_MAX_DETECTION_AGE_SECONDS
            and observation.center_x is not None
        )

        if not visible:
            motors.stop()
            if now - observation.last_seen_time > BASKETBALL_REACQUIRE_SECONDS:
                print("\nBasketball lost. Resuming stepwise exploration.")
                return "lost"

            # Search in the direction where the ball was last seen, using a useful
            # turn rather than many tiny cautious pulses.
            turn_seconds = 0.18
            if last_horizontal_error < 0:
                ok = motion_pulse(
                    lambda: motors.rotate_left(BASKETBALL_ALIGNMENT_TURN_SPEED),
                    turn_seconds,
                )
            else:
                ok = motion_pulse(
                    lambda: motors.rotate_right(BASKETBALL_ALIGNMENT_TURN_SPEED),
                    turn_seconds,
                )
            if not ok:
                return "blocked"
            continue

        horizontal_error = observation.center_x - CAMERA_WIDTH / 2.0
        last_horizontal_error = horizontal_error
        distance_m = observation.distance_m
        distance_text = "unknown" if distance_m is None else f"{distance_m:.2f}m"

        print(
            "\r"
            f"FAST APPROACH YOLO:{observation.confidence:.2f} "
            f"orange:{observation.orange_fraction:.2f} "
            f"error:{horizontal_error:+5.0f}px depth:{distance_text:>7s}",
            end="",
            flush=True,
        )

        # At the target distance, one centered observation is enough. Requiring
        # multiple slow YOLO frames added several unnecessary seconds.
        if distance_m is not None and distance_m <= BASKETBALL_STOP_DISTANCE_M:
            motors.stop()
            if abs(horizontal_error) <= BASKETBALL_CAPTURE_CENTER_TOLERANCE_PIXELS:
                photo_path = detector.save_target_photo()
                print(f"\nBasketball reached. Photo saved to: {photo_path}")
                return "captured"

            turn_seconds = turn_pulse_for_error(horizontal_error)
            if horizontal_error < 0:
                ok = motion_pulse(
                    lambda: motors.rotate_left(BASKETBALL_ALIGNMENT_TURN_SPEED),
                    turn_seconds,
                )
            else:
                ok = motion_pulse(
                    lambda: motors.rotate_right(BASKETBALL_ALIGNMENT_TURN_SPEED),
                    turn_seconds,
                )
            if not ok:
                return "blocked"
            continue

        # Correct only meaningful misalignment. A wider center corridor lets the
        # rover keep making forward progress instead of endlessly trimming heading.
        if abs(horizontal_error) > BASKETBALL_CENTER_TOLERANCE_PIXELS:
            turn_seconds = turn_pulse_for_error(horizontal_error)
            if horizontal_error < 0:
                ok = motion_pulse(
                    lambda: motors.rotate_left(BASKETBALL_ALIGNMENT_TURN_SPEED),
                    turn_seconds,
                )
            else:
                ok = motion_pulse(
                    lambda: motors.rotate_right(BASKETBALL_ALIGNMENT_TURN_SPEED),
                    turn_seconds,
                )
            if not ok:
                return "blocked"
            continue

        if distance_m is None:
            motors.stop()
            continue

        front_clearance = get_instant_center_clearance(lidar_state)
        if front_clearance <= FRONT_EMERGENCY_DISTANCE + 0.08:
            motors.stop()
            print(f"\nSafety stop: LiDAR obstacle at {front_clearance:.2f}m.")
            return "blocked"

        # Long/faster pulses while far away, tapering only near the stop distance.
        if distance_m > BASKETBALL_FAR_DISTANCE_M:
            speed = BASKETBALL_APPROACH_FAST_SPEED
            pulse_seconds = BASKETBALL_FAR_FORWARD_PULSE_SECONDS
        elif distance_m > BASKETBALL_NEAR_DISTANCE_M:
            speed = BASKETBALL_APPROACH_MEDIUM_SPEED
            pulse_seconds = BASKETBALL_MEDIUM_FORWARD_PULSE_SECONDS
        else:
            speed = BASKETBALL_APPROACH_NEAR_SPEED
            pulse_seconds = BASKETBALL_NEAR_FORWARD_PULSE_SECONDS

        if not motion_pulse(lambda: motors.forward(speed), pulse_seconds):
            return "blocked"


# ============================================================
# MOTORS
# ============================================================

class MotorController:
    def __init__(self):
        self.pwm = {}

        for name, pins in MOTORS.items():
            GPIO.setup(pins["enable"], GPIO.OUT)
            GPIO.setup(pins["in1"], GPIO.OUT)
            GPIO.setup(pins["in2"], GPIO.OUT)

            GPIO.output(pins["in1"], GPIO.LOW)
            GPIO.output(pins["in2"], GPIO.LOW)

            pwm = GPIO.PWM(pins["enable"], PWM_FREQUENCY)
            pwm.start(0)
            self.pwm[name] = pwm

        self.stop()

    def set_motor(self, name, direction, speed):
        pins = MOTORS[name]
        direction *= MOTOR_POLARITY[name]
        speed = clamp(float(speed), 0.0, 100.0)

        if direction > 0:
            GPIO.output(pins["in1"], GPIO.HIGH)
            GPIO.output(pins["in2"], GPIO.LOW)
            self.pwm[name].ChangeDutyCycle(speed)
        elif direction < 0:
            GPIO.output(pins["in1"], GPIO.LOW)
            GPIO.output(pins["in2"], GPIO.HIGH)
            self.pwm[name].ChangeDutyCycle(speed)
        else:
            self.pwm[name].ChangeDutyCycle(0)
            GPIO.output(pins["in1"], GPIO.LOW)
            GPIO.output(pins["in2"], GPIO.LOW)

    def stop(self):
        for name in MOTORS:
            self.set_motor(name, 0, 0)

    def forward(self, speed):
        for name in MOTORS:
            self.set_motor(name, 1, speed)

    def reverse(self, speed):
        for name in MOTORS:
            self.set_motor(name, -1, speed)

    def rotate_left(self, speed):
        # Verified convention: rotate_left increases heading (radians)
        # in this rover's odometry math.
        self.set_motor("front_left", -1, speed)
        self.set_motor("rear_left", -1, speed)
        self.set_motor("front_right", 1, speed)
        self.set_motor("rear_right", 1, speed)

    def rotate_right(self, speed):
        self.set_motor("front_left", 1, speed)
        self.set_motor("rear_left", 1, speed)
        self.set_motor("front_right", -1, speed)
        self.set_motor("rear_right", -1, speed)

    def cleanup(self):
        self.stop()

        for pwm in list(self.pwm.values()):
            try:
                pwm.stop()
            except Exception:
                pass

        self.pwm.clear()


# ============================================================
# ENCODERS
# ============================================================

class EncoderReader:
    def __init__(self):
        self.lock = threading.Lock()
        self.counts = {name: 0 for name in ENCODERS}

        for name, pins in ENCODERS.items():
            GPIO.setup(pins["a"], GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(pins["b"], GPIO.IN, pull_up_down=GPIO.PUD_UP)

            GPIO.add_event_detect(
                pins["a"],
                GPIO.BOTH,
                callback=self._make_callback(name),
            )

    def _make_callback(self, name):
        def callback(_channel):
            pins = ENCODERS[name]
            a_state = GPIO.input(pins["a"])
            b_state = GPIO.input(pins["b"])

            direction = 1 if a_state == b_state else -1
            direction *= ENCODER_POLARITY[name]

            with self.lock:
                self.counts[name] += direction

        return callback

    def snapshot(self):
        with self.lock:
            return self.counts.copy()

    def cleanup(self):
        for pins in ENCODERS.values():
            try:
                GPIO.remove_event_detect(pins["a"])
            except Exception:
                pass


class OdometryDeltaTracker:
    def __init__(self, encoder_reader):
        self.encoder_reader = encoder_reader
        self.previous_counts = encoder_reader.snapshot()

        self.meters_per_count = (
            math.pi * WHEEL_DIAMETER_METERS
            / ENCODER_COUNTS_PER_WHEEL_REV
        )

        self.lock = threading.Lock()
        self.accum_distance_mm = 0.0
        self.accum_heading_deg = 0.0

    def update(self):
        counts = self.encoder_reader.snapshot()

        deltas = {
            name: counts[name] - self.previous_counts[name]
            for name in counts
        }
        self.previous_counts = counts

        left_counts = (
            deltas["front_left"] + deltas["rear_left"]
        ) / 2.0
        right_counts = (
            deltas["front_right"] + deltas["rear_right"]
        ) / 2.0

        left_distance = left_counts * self.meters_per_count
        right_distance = right_counts * self.meters_per_count

        center_distance = (left_distance + right_distance) / 2.0
        heading_change_rad = (
            right_distance - left_distance
        ) / TRACK_WIDTH_METERS

        with self.lock:
            self.accum_distance_mm += center_distance * 1000.0
            self.accum_heading_deg += math.degrees(heading_change_rad)

    def pop_deltas(self):
        with self.lock:
            distance_mm = self.accum_distance_mm
            heading_deg = self.accum_heading_deg
            self.accum_distance_mm = 0.0
            self.accum_heading_deg = 0.0
        return distance_mm, heading_deg


# ============================================================
# LIDAR
# ============================================================

class LidarState:
    def __init__(self):
        self.lock = threading.Lock()
        self.scan = []
        self.last_update = 0.0

    def update(self, scan):
        with self.lock:
            self.scan = list(scan)
            self.last_update = time.monotonic()

    def snapshot(self):
        with self.lock:
            return {
                "scan": list(self.scan),
                "last_update": self.last_update,
            }


class LidarReader(threading.Thread):
    def __init__(self, state):
        super().__init__(daemon=True)
        self.state = state
        self.process = None
        self.running = True
        self.error_message = None

    def run(self):
        command = [
            "stdbuf",
            "-oL",
            LIDAR_PROGRAM,
            "--channel",
            "--serial",
            LIDAR_PORT,
            LIDAR_BAUD,
        ]

        current_scan = []
        previous_angle = None
        last_publish = time.monotonic()

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid,
            )

            if self.process.stdout is None:
                self.error_message = "LiDAR process produced no output stream."
                return

            for raw_line in self.process.stdout:
                if not self.running:
                    break

                match = LIDAR_PATTERN.search(raw_line)

                if match is None:
                    text = raw_line.strip()
                    if text and (
                        "error" in text.lower()
                        or "fail" in text.lower()
                        or "cannot" in text.lower()
                    ):
                        self.error_message = text
                    continue

                angle = float(match.group(1))
                distance_m = float(match.group(2)) / 1000.0
                quality = int(match.group(3))

                angle_wrapped = (
                    previous_angle is not None
                    and angle + 30.0 < previous_angle
                )

                now = time.monotonic()
                timed_publish = (
                    current_scan
                    and now - last_publish >= 0.30
                )

                if (angle_wrapped or timed_publish) and current_scan:
                    self.state.update(current_scan)
                    current_scan = []
                    last_publish = now

                if quality > 0 and 0.05 <= distance_m <= 12.0:
                    current_scan.append((angle, distance_m, quality))

                previous_angle = angle

            if (
                self.running
                and self.process is not None
                and self.process.poll() is not None
            ):
                self.error_message = (
                    f"LiDAR SDK exited with code "
                    f"{self.process.returncode}."
                )

        except Exception as error:
            self.error_message = str(error)

    def stop_reader(self):
        self.running = False

        if self.process is not None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass


# ============================================================
# SHARED STATE + SLAM WORKER
# ============================================================

class PoseHolder:
    def __init__(self):
        self.lock = threading.Lock()
        self.pose_m = (0.0, 0.0, 0.0)

    def set(self, x_m, y_m, heading_rad):
        with self.lock:
            self.pose_m = (x_m, y_m, heading_rad)

    def get(self):
        with self.lock:
            return self.pose_m


class SlamWorker(threading.Thread):
    def __init__(self, lidar_state, odometry_tracker, pose_holder):
        super().__init__(daemon=True)
        self.lidar_state = lidar_state
        self.odometry_tracker = odometry_tracker
        self.pose_holder = pose_holder
        self.running = True

        laser = Laser(
            SCAN_SIZE,
            SCAN_RATE_HZ,
            DETECTION_ANGLE_DEGREES,
            DISTANCE_NO_DETECTION_MM,
            DETECTION_MARGIN,
            LIDAR_OFFSET_MM,
        )

        self.slam = RMHC_SLAM(
            laser,
            MAP_SIZE_PIXELS,
            MAP_SIZE_METERS,
            map_quality=80,      # default is 50; higher = more confident
                                 # per-scan updates, converges to solid
                                 # black/white faster instead of lingering
                                 # in the mid-gray streak zone
            hole_width_mm=400,  # default is 600; thinner value keeps
                                 # walls tighter/crisper instead of
                                 # widening them into a fuzzy band
        )

        self.last_processed_scan_time = 0.0
        self.last_map_save = time.monotonic()
        self.last_slam_update_time = time.monotonic()
        self.scans_processed = 0

    def save_map(self, filename):
        MAP_DIRECTORY.mkdir(parents=True, exist_ok=True)
        output_path = MAP_DIRECTORY / filename

        map_bytes = bytearray(MAP_SIZE_PIXELS * MAP_SIZE_PIXELS)
        self.slam.getmap(map_bytes)

        image = Image.frombuffer(
            "L",
            (MAP_SIZE_PIXELS, MAP_SIZE_PIXELS),
            bytes(map_bytes),
            "raw",
            "L",
            0,
            1,
        )
        image = clean_map_image(image)
        image.save(output_path)

        try:
            os.chown(output_path, 1000, 1000)
        except PermissionError:
            pass

        return output_path

    def run(self):
        while self.running:
            self.odometry_tracker.update()
            lidar = self.lidar_state.snapshot()

            if lidar["last_update"] > self.last_processed_scan_time:
                self.last_processed_scan_time = lidar["last_update"]

                scan_mm = bin_scan_to_mm(lidar["scan"])
                distance_mm, heading_deg = self.odometry_tracker.pop_deltas()

                now = time.monotonic()
                dt_seconds = now - self.last_slam_update_time
                self.last_slam_update_time = now

                self.slam.update(
                    scan_mm,
                    pose_change=(distance_mm, heading_deg, dt_seconds),
                )

                x_mm, y_mm, theta_deg = self.slam.getpos()

                self.pose_holder.set(
                    x_mm / 1000.0,
                    y_mm / 1000.0,
                    math.radians(theta_deg),
                )
                self.scans_processed += 1

            now = time.monotonic()
            if now - self.last_map_save >= MAP_SAVE_INTERVAL:
                self.save_map("basketball_stepwise_live_map.png")
                self.last_map_save = now

            time.sleep(0.02)

    def stop_worker(self):
        self.running = False


# ============================================================
# STEP EXECUTION (rotate-to-heading, drive-a-step)
# ============================================================

def rotate_to_heading(motors, pose_holder, target_heading_rad, mission_complete_event):
    _, _, current_heading = pose_holder.get()
    error = normalize_angle(target_heading_rad - current_heading)

    if abs(error) <= math.radians(TURN_TOLERANCE_DEG):
        return

    # Direction decided ONCE, before the loop starts -- this is what
    # avoids the old dithering problem. We don't re-decide mid-turn.
    turning_left = error > 0

    if turning_left:
        motors.rotate_left(TURN_SPEED)
    else:
        motors.rotate_right(TURN_SPEED)

    start_time = time.monotonic()

    while True:
        if mission_complete_event.is_set():
            break

        if time.monotonic() - start_time > MAX_TURN_SECONDS:
            break

        _, _, current_heading = pose_holder.get()
        error = normalize_angle(target_heading_rad - current_heading)

        if abs(error) <= math.radians(TURN_TOLERANCE_DEG):
            break

        time.sleep(0.05)

    motors.stop()


def drive_step(
    motors, pose_holder, lidar_state, target_distance_m, mission_complete_event
):
    start_x, start_y, _ = pose_holder.get()
    start_time = time.monotonic()

    motors.forward(EXPLORE_SPEED)

    while True:
        if mission_complete_event.is_set():
            break

        now = time.monotonic()

        if now - start_time > MAX_STEP_SECONDS:
            break

        x, y, _ = pose_holder.get()
        traveled = math.hypot(x - start_x, y - start_y)

        if traveled >= target_distance_m:
            break

        # Safety net only -- not the primary decision logic. If
        # something unexpected is suddenly right in front, stop.
        if get_instant_center_clearance(lidar_state) <= FRONT_EMERGENCY_DISTANCE:
            break

        time.sleep(0.05)

    motors.stop()


# ============================================================
# STEPWISE EXPLORATION LOOP
# ============================================================

def wait_for_fresh_scan(lidar_state, after_time):
    wait_start = time.monotonic()

    while True:
        snapshot = lidar_state.snapshot()

        if snapshot["last_update"] > after_time + 0.05:
            return snapshot["scan"]

        if time.monotonic() - wait_start > LIDAR_TIMEOUT:
            return snapshot["scan"]

        time.sleep(0.05)


def stepwise_explore(
    motors, lidar_state, pose_holder, slam_worker, target_found_event, detector
):
    """Preserve the uploaded STOP -> SCAN -> DECIDE -> TURN -> DRIVE behavior."""
    consecutive_stuck = 0

    while True:
        if detector.error_message:
            print(f"\nBasketball camera failed: {detector.error_message}")
            motors.stop()
            return "camera_error"

        if target_found_event.is_set():
            print("\nBasketball detected. Stopping exploration for approach.")
            motors.stop()
            return "target"

        motors.stop()
        time.sleep(0.15)

        before_stop_time = lidar_state.snapshot()["last_update"]
        scan = wait_for_fresh_scan(lidar_state, before_stop_time)

        if not scan:
            print("\nNo LiDAR data available. Stopping for safety.")
            motors.stop()
            return "lidar_error"

        clearances, sector_width = get_sector_clearances(scan)
        choice = choose_target_sector(clearances, sector_width)

        if choice is None:
            consecutive_stuck += 1
            print(
                f"\nNo clear direction found (attempt {consecutive_stuck}/"
                f"{MAX_CONSECUTIVE_STUCK}). Backing up a little."
            )
            motors.reverse(REVERSE_SPEED)
            time.sleep(STUCK_REVERSE_SECONDS)
            motors.stop()

            if consecutive_stuck >= MAX_CONSECUTIVE_STUCK:
                print("Still boxed in. Performing one short recovery turn, then continuing.")
                motors.rotate_left(TURN_SPEED)
                time.sleep(0.65)
                motors.stop()
                consecutive_stuck = 0
            continue

        consecutive_stuck = 0
        relative_angle_deg, clearance = choice
        scaled_turn_deg = relative_angle_deg * TURN_ANGLE_SCALE

        _, _, current_heading = pose_holder.get()
        target_heading = normalize_angle(current_heading + math.radians(scaled_turn_deg))

        x_m, y_m, _ = pose_holder.get()
        print(
            f"\nAt x:{x_m:5.2f}m y:{y_m:5.2f}m -- "
            f"turning {scaled_turn_deg:+.0f} deg (of {relative_angle_deg:+.0f} deg found) toward "
            f"{clearance:.2f}m of clearance "
            f"(scans processed: {slam_worker.scans_processed})"
        )

        rotate_to_heading(motors, pose_holder, target_heading, target_found_event)

        if target_found_event.is_set():
            motors.stop()
            return "target"

        # Re-check the fresh forward view after the turn before committing to a drive.
        after_turn_time = lidar_state.snapshot()["last_update"]
        fresh_scan = wait_for_fresh_scan(lidar_state, after_turn_time)
        if fresh_scan:
            center_clearance = get_instant_center_clearance(lidar_state)
            clearance = min(clearance, center_clearance if math.isfinite(center_clearance) else clearance)

        step_distance = min(clearance * STEP_DISTANCE_FRACTION, MAX_STEP_METERS)
        if step_distance < 0.12:
            print("Forward path became too short after the turn; rescanning.")
            continue

        print(f"Driving forward {step_distance:.2f}m...")
        drive_step(motors, pose_holder, lidar_state, step_distance, target_found_event)

        if target_found_event.is_set():
            motors.stop()
            return "target"

    motors.stop()


# ============================================================
# MAIN
# ============================================================

def main():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    motors = None
    encoders = None
    lidar_reader = None
    slam_worker = None
    basketball_detector = None
    target_found_event = threading.Event()
    mission_result = "stopped"

    try:
        print("=" * 72)
        print("AACVISION STEPWISE BASKETBALL SEARCH + LIDAR MAPPING V7")
        print("=" * 72)
        print("Your original travel logic: STOP -> SCAN -> DECIDE -> TURN -> DRIVE.")
        print("The proven RealSense/YOLO detector searches continuously.")
        print("No run timer: Ctrl+C stops it, or it ends after reaching the ball.")
        print("Test with wheels lifted first and keep immediate access to motor power.")
        print("=" * 72)
        input("Press Enter when ready to start...")

        motors = MotorController()
        encoders = EncoderReader()
        odometry_tracker = OdometryDeltaTracker(encoders)
        pose_holder = PoseHolder()

        lidar_state = LidarState()
        lidar_reader = LidarReader(lidar_state)
        lidar_reader.start()

        print("Waiting for LiDAR to start...")
        wait_start = time.monotonic()
        while lidar_state.snapshot()["last_update"] <= 0.0:
            if time.monotonic() - wait_start > LIDAR_STARTUP_GRACE:
                print("No LiDAR data received. Check the LiDAR connection.")
                if lidar_reader.error_message:
                    print(f"LiDAR SDK message: {lidar_reader.error_message}")
                return
            time.sleep(0.1)

        print(f"LiDAR is running. Starting RealSense and YOLO model: {DEFAULT_BASKETBALL_MODEL}")
        basketball_detector = BasketballDetector(target_found_event)
        basketball_detector.start()

        camera_wait_start = time.monotonic()
        last_stage = None
        while basketball_detector.latest_frame is None and basketball_detector.error_message is None:
            stage = basketball_detector.startup_stage
            if stage != last_stage:
                print(f"  Camera startup: {stage}...")
                last_stage = stage
            if time.monotonic() - camera_wait_start > CAMERA_STARTUP_TIMEOUT_SECONDS:
                print("Basketball camera startup timed out.")
                return
            time.sleep(0.1)

        if basketball_detector.error_message is not None:
            print(f"RealSense/YOLO could not start: {basketball_detector.error_message}")
            return

        print("RealSense and YOLO are ready. Starting SLAM and autonomous search.")
        slam_worker = SlamWorker(lidar_state, odometry_tracker, pose_holder)
        slam_worker.start()
        time.sleep(0.8)

        while True:
            exploration_result = stepwise_explore(
                motors,
                lidar_state,
                pose_holder,
                slam_worker,
                target_found_event,
                basketball_detector,
            )

            if exploration_result != "target":
                mission_result = exploration_result
                break

            approach_result = approach_basketball(motors, lidar_state, basketball_detector)
            if approach_result == "captured":
                mission_result = "basketball_reached"
                print("\nMISSION COMPLETE: basketball reached; rover stopped.")
                break

            if approach_result == "lost":
                basketball_detector.reset_target()
                print("Returning to stop-scan-decide mapping search.")
                time.sleep(0.5)
                continue

            mission_result = approach_result
            print(f"\nApproach ended for safety: {approach_result}")
            break

    except KeyboardInterrupt:
        mission_result = "manual_stop"
        print("\nEmergency stop requested with Ctrl+C.")

    finally:
        if motors is not None:
            motors.stop()

        if slam_worker is not None:
            slam_worker.stop_worker()
            slam_worker.join(timeout=2.0)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                map_path = slam_worker.save_map(f"basketball_stepwise_final_{timestamp}.png")
                print(f"\nFinal SLAM map saved to: {map_path}")
            except Exception as save_error:
                print(f"\nCould not save final map: {save_error}")

        if basketball_detector is not None:
            basketball_detector.stop_detector()
            basketball_detector.join(timeout=3.0)
            if basketball_detector.error_message:
                print(f"RealSense/YOLO detector message: {basketball_detector.error_message}")

        if lidar_reader is not None:
            lidar_reader.stop_reader()

        if encoders is not None:
            encoders.cleanup()

        if motors is not None:
            motors.cleanup()

        GPIO.cleanup()
        print(f"Mission result: {mission_result}")
        print("Motors stopped and hardware released.")


if __name__ == "__main__":
    main()
