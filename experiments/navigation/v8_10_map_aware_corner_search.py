#!/usr/bin/env python3
"""
AACVision Basketball Search + Mapping V8.10 Map-Aware Corner Search
-----------------------------------------------------------------
Uses the simple local state-machine navigation style from the user's
``yellow_target_navigation.py`` reference:

* scan/search with short timed turns,
* move forward in short patrol pulses when the front is clear,
* reverse briefly and turn toward the clearer side when blocked,
* after a confirmed basketball detection, center it with short turns,
* approach using aligned RealSense depth and LiDAR emergency safety,
* stop about 0.80 m from the basketball and save a photograph.

BreezySLAM mapping runs in the background and is consulted only when a
corner offers both a left and right escape. The rover chooses the side with
more reachable unknown map area, while retaining simple local obstacle travel.
This is not a full frontier planner or a complete-room coverage guarantee.

Test with wheels raised first. Ctrl+C stops immediately.
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
# YELLOW-STYLE LOCAL NAVIGATION SETTINGS
# ============================================================

# These values follow the simple short-pulse state machine from the uploaded
# yellow-target navigation code. The added forward patrol pulse is necessary
# so the rover can search beyond the camera view at its starting position.
SEARCH_TURN_SPEED = 24
SEARCH_FORWARD_SPEED = 22
SEARCH_REVERSE_SPEED = 18
TARGET_ALIGN_SPEED = 20
TARGET_APPROACH_SPEED = 23
TARGET_SLOW_APPROACH_SPEED = 16

SEARCH_TURN_PULSE_SECONDS = 0.16
SEARCH_FORWARD_PULSE_SECONDS = 0.47
SEARCH_REVERSE_PULSE_SECONDS = 0.34
SEARCH_ESCAPE_TURN_SECONDS = 0.18
SEARCH_SETTLE_SECONDS = 0.07
SEARCH_CAMERA_SWEEP_EVERY_FORWARD_PULSES = 6

# A stopped camera sweep looks left and right, then approximately returns to
# the original heading. The dwell is long enough for 3-frame confirmation.
CAMERA_SWEEP_TURN_SPEED = 18
CAMERA_SWEEP_PULSE_SECONDS = 0.10
CAMERA_SWEEP_DWELL_SECONDS = 0.20
CAMERA_SWEEP_STEPS_EACH_SIDE = 4

# Once a wall escape direction is selected, keep it for several turn pulses.
# This prevents left/right oscillation caused by small LiDAR fluctuations.
WALL_ESCAPE_TURN_PULSES = 5
WALL_ESCAPE_RETRY_TURN_PULSES = 4
WALL_ESCAPE_MAX_RETRIES = 2
WALL_ESCAPE_FRONT_RELEASE_M = 0.92
WALL_ESCAPE_DIRECTION_MARGIN_M = 0.12
WALL_ESCAPE_REVERSE_SECONDS = 0.42
WALL_ESCAPE_PROGRESS_FORWARD_PULSES = 2

# Map-aware corner choice. BreezySLAM normally stores unknown cells near the
# middle-gray range, free cells near white, and obstacles near black. The
# search samples reachable rays to each side and favors the side containing
# more unknown cells before the first mapped obstacle.
MAP_UNKNOWN_LOW = 90
MAP_UNKNOWN_HIGH = 210
MAP_FREE_THRESHOLD = 220
MAP_OCCUPIED_THRESHOLD = 80
MAP_SIDE_MIN_RADIUS_M = 0.55
MAP_SIDE_MAX_RADIUS_M = 3.20
MAP_SIDE_RADIUS_STEP_M = 0.14
MAP_SIDE_HALF_ANGLE_DEG = 48.0
MAP_SIDE_ANGLE_SAMPLES = 13
MAP_SIDE_SCORE_DECISION_MARGIN = 0.035
MAP_SIDE_MIN_VALID_SAMPLES = 35

TARGET_ALIGN_PULSE_SECONDS = 0.075
TARGET_FORWARD_PULSE_FAR_SECONDS = 0.16
TARGET_FORWARD_PULSE_NEAR_SECONDS = 0.09
TARGET_SETTLE_SECONDS = 0.045
TARGET_CENTER_TOLERANCE_PIXELS = 38
TARGET_LOST_FRAME_LIMIT = 8
TARGET_CLOSE_CONFIRM_FRAMES = 2

LOCAL_FRONT_EMERGENCY_DISTANCE_M = 0.38
LOCAL_FRONT_CLEAR_DISTANCE_M = 0.78
LOCAL_FRONT_GOOD_DISTANCE_M = 1.05
LOCAL_SIDE_CLEAR_DISTANCE_M = 0.62
LOCAL_REAR_CLEAR_DISTANCE_M = 0.65
LOCAL_LIDAR_TIMEOUT_SECONDS = 2.0

LOCAL_FRONT_START = 340
LOCAL_FRONT_END = 20
LOCAL_LEFT_START = 30
LOCAL_LEFT_END = 110
LOCAL_RIGHT_START = 250
LOCAL_RIGHT_END = 330
LOCAL_REAR_START = 155
LOCAL_REAR_END = 205

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

MAP_SIZE_PIXELS = 1000
MAP_SIZE_METERS = 16.0
MAP_SAVE_INTERVAL = 3.0

PI_HOME = Path("/home/pi")
MAP_DIRECTORY = PI_HOME / "aacvision_maps"

# ============================================================
# INTEL REALSENSE D435 / BASKETBALL MISSION SETTINGS
# ============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_INFERENCE_SIZE = 384
CAMERA_DIRECTORY = PI_HOME / "aacvision_basketball_images"
CAMERA_LATEST_PATH = CAMERA_DIRECTORY / "latest_basketball_view.jpg"
DEFAULT_BASKETBALL_MODEL = "yolo26n.pt"

# V8.7 deliberately returns to the simple V8 stepwise explorer. Only the
# basketball detector and the final ball-approach controller are strengthened.
BASKETBALL_CLASS_NAMES = ("sports ball", "basketball")
BASKETBALL_AI_MIN_CONFIDENCE = 0.15
BASKETBALL_MIN_BOX_AREA_FRACTION = 0.0010
BASKETBALL_MAX_BOX_AREA_FRACTION = 0.35
ORANGE_HSV_LOWER = np.array([2, 65, 40], dtype=np.uint8)
ORANGE_HSV_UPPER = np.array([34, 255, 255], dtype=np.uint8)
ORANGE_MIN_FRACTION = 0.035
BASKETBALL_MIN_ROUNDNESS = 0.48
BASKETBALL_CONFIRM_SCORE = 0.74
BASKETBALL_CONFIRM_CONSECUTIVE_FRAMES = 3
BASKETBALL_COLOR_CONFIRM_CONSECUTIVE_FRAMES = 3
BASKETBALL_MIN_CENTER_Y_FRACTION = 0.18
BASKETBALL_MAX_CENTER_JUMP_PIXELS = 105.0
BASKETBALL_MAX_DEPTH_RATIO_CHANGE = 0.55
BASKETBALL_MAX_AREA_RATIO_CHANGE = 2.8
BASKETBALL_EDGE_MARGIN_PIXELS = 10
BASKETBALL_MIN_VALID_DISTANCE_M = 0.25
BASKETBALL_MAX_VALID_DISTANCE_M = 5.0
BASKETBALL_MAX_DETECTION_AGE_SECONDS = 3.5
BASKETBALL_REACQUIRE_SECONDS = 6.0

# Stop and approach settings. V8.7 uses small, verified corrections rather
# than long turns or continuous motion on an old camera frame.
BASKETBALL_STOP_DISTANCE_M = 0.80
BASKETBALL_STOP_DISTANCE_TOLERANCE_M = 0.08
BASKETBALL_FAR_CENTER_TOLERANCE_PIXELS = 50
BASKETBALL_NEAR_CENTER_TOLERANCE_PIXELS = 32
BASKETBALL_CAPTURE_CENTER_TOLERANCE_PIXELS = 45
BASKETBALL_APPROACH_TRACK_FRAMES = 2
BASKETBALL_CLOSE_CONFIRM_FRAMES = 2

BASKETBALL_ALIGNMENT_TURN_SPEED = 18
BASKETBALL_APPROACH_FAST_SPEED = 22
BASKETBALL_APPROACH_MEDIUM_SPEED = 18
BASKETBALL_APPROACH_NEAR_SPEED = 13
BASKETBALL_FAR_DISTANCE_M = 2.00
BASKETBALL_NEAR_DISTANCE_M = 1.20
BASKETBALL_FAR_FORWARD_PULSE_SECONDS = 0.36
BASKETBALL_MEDIUM_FORWARD_PULSE_SECONDS = 0.22
BASKETBALL_NEAR_FORWARD_PULSE_SECONDS = 0.10
BASKETBALL_MIN_TURN_PULSE_SECONDS = 0.045
BASKETBALL_MAX_TURN_PULSE_SECONDS = 0.16
BASKETBALL_REACQUIRE_TURN_PULSE_SECONDS = 0.07
BASKETBALL_REACQUIRE_ALTERNATE_EVERY = 3
BASKETBALL_REACQUIRE_MAX_STEPS = 12
BASKETBALL_POST_MOTION_SETTLE_SECONDS = 0.06
BASKETBALL_STEERING_GAIN = 0.055
BASKETBALL_MAX_STEERING_DELTA = 3.5
BASKETBALL_TRACK_MAX_CENTER_JUMP_PIXELS = 110.0
BASKETBALL_TRACK_MAX_DEPTH_CHANGE_M = 0.75
BASKETBALL_DEPTH_MIN_VALID_PIXELS = 10
BASKETBALL_DEPTH_PERCENTILE = 35.0

# Fast orange-ball fallback. This runs before YOLO and lets the rover react to
# a close, clearly orange circular basketball at camera frame rate. YOLO remains
# the long-range/general detector. The shape checks reject most orange walls,
# cabinets, and floor patches.
COLOR_FALLBACK_MIN_AREA_PIXELS = 650
COLOR_FALLBACK_MAX_AREA_FRACTION = 0.30
COLOR_FALLBACK_MIN_CIRCULARITY = 0.58
COLOR_FALLBACK_MIN_ASPECT_RATIO = 0.72
COLOR_FALLBACK_MIN_FILL_RATIO = 0.45
COLOR_FALLBACK_MIN_SOLIDITY = 0.82
COLOR_FALLBACK_MIN_ORANGE_FRACTION = 0.24
COLOR_FALLBACK_MAX_DISTANCE_M = 4.5
LATEST_VIEW_SAVE_INTERVAL_SECONDS = 0.5

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
    """Runs a dual basketball detector: fast orange-circle tracking plus YOLO."""

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
        self.candidate_streak = 0
        self.last_candidate_center = None
        self.last_candidate_distance = None
        self.last_candidate_area = None
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
            if (
                area_fraction < BASKETBALL_MIN_BOX_AREA_FRACTION
                or area_fraction > BASKETBALL_MAX_BOX_AREA_FRACTION
            ):
                continue

            # Ignore tiny edge slivers, which frequently produce sports-ball
            # false positives on furniture, lamps, and partially visible objects.
            if (
                x1 <= BASKETBALL_EDGE_MARGIN_PIXELS
                or y1 <= BASKETBALL_EDGE_MARGIN_PIXELS
                or x2 >= frame.shape[1] - BASKETBALL_EDGE_MARGIN_PIXELS
                or y2 >= frame.shape[0] - BASKETBALL_EDGE_MARGIN_PIXELS
            ):
                continue

            orange_fraction, roundness = self._appearance_evidence(frame, bbox)

            # Every YOLO candidate must be reasonably round. Moderate-confidence
            # detections must also look orange; only a very strong model result
            # may tolerate reduced orange caused by shadows or black seams.
            if roundness < BASKETBALL_MIN_ROUNDNESS:
                continue
            strong_yolo = confidence >= 0.55
            appearance_ok = orange_fraction >= ORANGE_MIN_FRACTION
            if not strong_yolo and not appearance_ok:
                continue

            center_y = (y1 + y2) / 2.0
            if center_y < frame.shape[0] * BASKETBALL_MIN_CENTER_Y_FRACTION:
                continue

            distance_m = self._depth_from_bbox(depth_image, bbox)
            if (
                distance_m is None
                or distance_m < BASKETBALL_MIN_VALID_DISTANCE_M
                or distance_m > BASKETBALL_MAX_VALID_DISTANCE_M
            ):
                continue

            score = confidence + 0.40 * orange_fraction + 0.15 * roundness
            candidate = {
                "score": score,
                "confidence": confidence,
                "center_x": (x1 + x2) / 2.0,
                "center_y": (y1 + y2) / 2.0,
                "distance_m": distance_m,
                "orange_fraction": orange_fraction,
                "roundness": roundness,
                "bbox": bbox,
                "area_fraction": area_fraction,
                "source": "yolo",
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        return best

    def _select_color_basketball(self, frame, depth_image):
        """Find a close orange circular ball without waiting for YOLO.

        This is intentionally conservative: it requires a compact, roughly
        circular orange region with a plausible RealSense depth. It makes
        close-range acquisition and approach much more responsive while YOLO
        remains available when the color fallback has no strong candidate.
        """
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ORANGE_HSV_LOWER, ORANGE_HSV_UPPER)

        # Close black basketball seams and small lighting gaps, then remove noise.
        close_kernel = np.ones((11, 11), dtype=np.uint8)
        open_kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(frame.shape[0] * frame.shape[1])
        best = None

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < COLOR_FALLBACK_MIN_AREA_PIXELS:
                continue
            area_fraction = area / frame_area
            if area_fraction > COLOR_FALLBACK_MAX_AREA_FRACTION:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < COLOR_FALLBACK_MIN_CIRCULARITY:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            aspect_ratio = min(w, h) / max(w, h)
            if aspect_ratio < COLOR_FALLBACK_MIN_ASPECT_RATIO:
                continue

            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / max(hull_area, 1.0)
            if solidity < COLOR_FALLBACK_MIN_SOLIDITY:
                continue

            (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
            if radius < 8.0:
                continue
            circle_area = math.pi * radius * radius
            fill_ratio = area / max(circle_area, 1.0)
            if fill_ratio < COLOR_FALLBACK_MIN_FILL_RATIO:
                continue

            pad = int(max(4.0, radius * 0.10))
            x1 = int(clamp(x - pad, 0, frame.shape[1] - 1))
            y1 = int(clamp(y - pad, 0, frame.shape[0] - 1))
            x2 = int(clamp(x + w + pad, x1 + 1, frame.shape[1]))
            y2 = int(clamp(y + h + pad, y1 + 1, frame.shape[0]))
            bbox = (x1, y1, x2, y2)

            if circle_y < frame.shape[0] * BASKETBALL_MIN_CENTER_Y_FRACTION:
                continue

            if (
                x1 <= BASKETBALL_EDGE_MARGIN_PIXELS
                or y1 <= BASKETBALL_EDGE_MARGIN_PIXELS
                or x2 >= frame.shape[1] - BASKETBALL_EDGE_MARGIN_PIXELS
                or y2 >= frame.shape[0] - BASKETBALL_EDGE_MARGIN_PIXELS
            ):
                continue

            orange_fraction, box_roundness = self._appearance_evidence(frame, bbox)
            if orange_fraction < COLOR_FALLBACK_MIN_ORANGE_FRACTION:
                continue

            distance_m = self._depth_from_bbox(depth_image, bbox)
            if distance_m is None or distance_m > COLOR_FALLBACK_MAX_DISTANCE_M:
                continue

            # Color-only candidates use stricter geometry and require four
            # consistent frames before they may interrupt exploration.
            confidence = min(0.34, 0.18 + 0.10 * circularity + 0.08 * fill_ratio)
            score = (
                0.35 * circularity
                + 0.20 * aspect_ratio
                + 0.20 * fill_ratio
                + 0.15 * solidity
                + 0.10 * min(1.0, orange_fraction)
            )
            candidate = {
                "score": score,
                "confidence": confidence,
                "center_x": float(circle_x),
                "center_y": float(circle_y),
                "distance_m": distance_m,
                "orange_fraction": orange_fraction,
                "roundness": max(box_roundness, circularity),
                "bbox": bbox,
                "area_fraction": area_fraction,
                "source": "color",
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

        return best

    def _annotate(self, frame, observation):
        output = frame.copy()
        if observation.bbox is not None and observation.visible:
            x1, y1, x2, y2 = observation.bbox
            box_color = (0, 255, 0) if observation.confirmed else (0, 215, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 3)
        if observation.confirmed:
            status = "BASKETBALL CONFIRMED"
            color = (0, 255, 0)
        elif observation.visible:
            status = (
                f"BALL CANDIDATE {self.candidate_streak}/"
                f"{BASKETBALL_CONFIRM_CONSECUTIVE_FRAMES}"
            )
            color = (0, 215, 255)
        else:
            status = "SEARCHING FOR BASKETBALL"
            color = (0, 180, 255)
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
            center = (float(detection["center_x"]), float(detection["center_y"]))
            distance_m = detection["distance_m"]
            area_fraction = float(detection.get("area_fraction", 0.0))

            consistent = False
            if self.last_candidate_center is not None:
                center_jump = math.hypot(
                    center[0] - self.last_candidate_center[0],
                    center[1] - self.last_candidate_center[1],
                )
                depth_ok = True
                if self.last_candidate_distance is not None and distance_m is not None:
                    denominator = max(0.20, min(self.last_candidate_distance, distance_m))
                    depth_ok = (
                        abs(distance_m - self.last_candidate_distance) / denominator
                        <= BASKETBALL_MAX_DEPTH_RATIO_CHANGE
                    )
                area_ok = True
                if self.last_candidate_area is not None and area_fraction > 0.0:
                    ratio = max(
                        area_fraction / max(self.last_candidate_area, 1e-6),
                        self.last_candidate_area / max(area_fraction, 1e-6),
                    )
                    area_ok = ratio <= BASKETBALL_MAX_AREA_RATIO_CHANGE
                consistent = (
                    center_jump <= BASKETBALL_MAX_CENTER_JUMP_PIXELS
                    and depth_ok
                    and area_ok
                )

            if consistent:
                self.candidate_streak += 1
            else:
                self.candidate_streak = 1
                self.smoothed_center_x = None
                self.smoothed_center_y = None
                self.smoothed_distance = None

            self.last_candidate_center = center
            self.last_candidate_distance = distance_m
            self.last_candidate_area = area_fraction

            evidence_score = (
                0.45 * min(1.0, detection["confidence"] / 0.55)
                + 0.30 * min(1.0, detection["orange_fraction"] / 0.24)
                + 0.25 * min(1.0, detection["roundness"] / 0.75)
            )
            self.confirm_score = min(1.0, evidence_score)

            alpha = 0.70
            if self.smoothed_center_x is None:
                self.smoothed_center_x = detection["center_x"]
                self.smoothed_center_y = detection["center_y"]
            else:
                self.smoothed_center_x = alpha * detection["center_x"] + (1.0 - alpha) * self.smoothed_center_x
                self.smoothed_center_y = alpha * detection["center_y"] + (1.0 - alpha) * self.smoothed_center_y
            if distance_m is not None:
                if self.smoothed_distance is None:
                    self.smoothed_distance = distance_m
                else:
                    self.smoothed_distance = alpha * distance_m + (1.0 - alpha) * self.smoothed_distance

            self.last_seen_time = now
            required_frames = (
                BASKETBALL_COLOR_CONFIRM_CONSECUTIVE_FRAMES
                if detection.get("source") == "color"
                else BASKETBALL_CONFIRM_CONSECUTIVE_FRAMES
            )
            confirmed = (
                self.candidate_streak >= required_frames
                and self.confirm_score >= BASKETBALL_CONFIRM_SCORE
            )
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
            self.confirm_score = max(0.0, self.confirm_score - 0.25)
            self.candidate_streak = max(0, self.candidate_streak - 1)
            if self.candidate_streak == 0:
                self.last_candidate_center = None
                self.last_candidate_distance = None
                self.last_candidate_area = None
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

        if now - self.last_latest_save >= LATEST_VIEW_SAVE_INTERVAL_SECONDS:
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
            self.candidate_streak = 0
            self.last_candidate_center = None
            self.last_candidate_distance = None
            self.last_candidate_area = None
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

                # Fast path: a close orange circular basketball is tracked with
                # OpenCV at camera speed. When that has no reliable candidate,
                # fall back to the slower but more general YOLO sports-ball model.
                detection = self._select_color_basketball(frame, depth_image)
                if detection is None:
                    results = self.model.predict(
                        source=frame,
                        imgsz=CAMERA_INFERENCE_SIZE,
                        conf=BASKETBALL_AI_MIN_CONFIDENCE,
                        verbose=False,
                        device="cpu",
                    )
                    result = results[0] if results else None
                    detection = (
                        None
                        if result is None
                        else self._select_basketball(result, frame, depth_image)
                    )
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
    """Use the uploaded yellow-target center-then-approach state machine.

    Every motion is a short timed pulse. The rover always stops after a pulse,
    waits for a fresh camera frame, and then makes the next decision. This keeps
    the behavior simple and prevents large accumulated corrections.
    """
    motors.stop()
    last_frame_time = float("-inf")
    lost_frames = 0
    close_frames = 0
    last_error = 0.0

    print("\nBasketball confirmed. Starting center-and-approach navigation.")

    def pulse(action, duration, check_front=False):
        action()
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if check_front:
                front = get_local_clearances(lidar_state)[0]
                if front <= LOCAL_FRONT_EMERGENCY_DISTANCE_M:
                    motors.stop()
                    print(f"\nEmergency stop: LiDAR front clearance {front:.2f}m.")
                    return False
            time.sleep(0.012)
        motors.stop()
        time.sleep(TARGET_SETTLE_SECONDS)
        return True

    while True:
        if detector.error_message:
            motors.stop()
            return "camera_error"

        observation, _frame = detector.snapshot()
        now = time.monotonic()

        # Wait for a genuinely new RealSense frame before issuing another move.
        if observation.frame_time <= last_frame_time + 1e-6:
            motors.stop()
            time.sleep(0.012)
            continue
        last_frame_time = observation.frame_time

        visible = (
            observation.visible
            and now - observation.frame_time <= BASKETBALL_MAX_DETECTION_AGE_SECONDS
            and observation.center_x is not None
            and observation.distance_m is not None
        )

        if not visible:
            motors.stop()
            lost_frames += 1
            close_frames = 0

            if lost_frames >= TARGET_LOST_FRAME_LIMIT:
                print("\nBasketball lost for several frames. Returning to local search patrol.")
                return "lost"

            # Look briefly toward the last known side. No 180-degree sweep.
            if last_error < 0:
                pulse(lambda: motors.rotate_left(TARGET_ALIGN_SPEED), 0.055)
                direction = "left"
            else:
                pulse(lambda: motors.rotate_right(TARGET_ALIGN_SPEED), 0.055)
                direction = "right"
            print(
                f"\rBALL LOST - checking {direction:>5s} "
                f"{lost_frames}/{TARGET_LOST_FRAME_LIMIT}",
                end="",
                flush=True,
            )
            continue

        lost_frames = 0
        horizontal_error = observation.center_x - CAMERA_WIDTH / 2.0
        last_error = horizontal_error
        distance_m = observation.distance_m
        front, _left, _right, _rear, lidar_age = get_local_clearances(lidar_state)

        print(
            "\r"
            f"BALL error:{horizontal_error:+5.0f}px "
            f"depth:{distance_m:4.2f}m "
            f"front:{front:4.2f}m "
            f"conf:{observation.confidence:.2f}",
            end="",
            flush=True,
        )

        if lidar_age > LOCAL_LIDAR_TIMEOUT_SECONDS:
            motors.stop()
            print("\nLiDAR data became stale during basketball approach.")
            return "lidar_error"

        centered = abs(horizontal_error) <= TARGET_CENTER_TOLERANCE_PIXELS
        within_stop = distance_m <= (
            BASKETBALL_STOP_DISTANCE_M + BASKETBALL_STOP_DISTANCE_TOLERANCE_M
        )

        if within_stop and centered:
            motors.stop()
            close_frames += 1
            if close_frames >= TARGET_CLOSE_CONFIRM_FRAMES:
                photo_path = detector.save_target_photo()
                print(f"\nBasketball reached. Photo saved to: {photo_path}")
                return "captured"
            time.sleep(TARGET_SETTLE_SECONDS)
            continue
        close_frames = 0

        # Center first, then advance using fixed small corrections.
        if horizontal_error < -TARGET_CENTER_TOLERANCE_PIXELS:
            pulse(lambda: motors.rotate_left(TARGET_ALIGN_SPEED), TARGET_ALIGN_PULSE_SECONDS)
            continue
        if horizontal_error > TARGET_CENTER_TOLERANCE_PIXELS:
            pulse(lambda: motors.rotate_right(TARGET_ALIGN_SPEED), TARGET_ALIGN_PULSE_SECONDS)
            continue

        if not math.isfinite(distance_m):
            motors.stop()
            time.sleep(0.05)
            continue

        if front <= LOCAL_FRONT_EMERGENCY_DISTANCE_M:
            motors.stop()
            print(f"\nApproach blocked by an obstacle at {front:.2f}m.")
            return "blocked"

        if distance_m > BASKETBALL_NEAR_DISTANCE_M:
            speed = TARGET_APPROACH_SPEED
            duration = TARGET_FORWARD_PULSE_FAR_SECONDS
        else:
            speed = TARGET_SLOW_APPROACH_SPEED
            duration = TARGET_FORWARD_PULSE_NEAR_SECONDS

        if not pulse(lambda: motors.forward(speed), duration, check_front=True):
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

    def forward_steer(self, base_speed, steering_delta):
        """Drive forward continuously with differential steering.

        Positive steering_delta curves right; negative curves left.
        """
        left_speed = clamp(base_speed + steering_delta, 0.0, 100.0)
        right_speed = clamp(base_speed - steering_delta, 0.0, 100.0)
        self.set_motor("front_left", 1, left_speed)
        self.set_motor("rear_left", 1, left_speed)
        self.set_motor("front_right", 1, right_speed)
        self.set_motor("rear_right", 1, right_speed)

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
        self.slam_lock = threading.Lock()

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
        with self.slam_lock:
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

    def get_raw_map_array(self):
        """Return a thread-safe copy of the current raw SLAM map."""
        map_bytes = bytearray(MAP_SIZE_PIXELS * MAP_SIZE_PIXELS)
        try:
            with self.slam_lock:
                self.slam.getmap(map_bytes)
        except Exception:
            return None
        return np.frombuffer(bytes(map_bytes), dtype=np.uint8).reshape(
            (MAP_SIZE_PIXELS, MAP_SIZE_PIXELS)
        )

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

                with self.slam_lock:
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
                self.save_map("basketball_map_aware_live_map.png")
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
# YELLOW-STYLE LOCAL SEARCH / PATROL
# ============================================================

def get_local_clearances(lidar_state):
    snapshot = lidar_state.snapshot()
    scan = snapshot["scan"]

    def sector(start, end):
        values = [
            distance_m
            for angle, distance_m, quality in scan
            if quality > 0
            and distance_m > 0
            and angle_in_sector(angle, start, end)
        ]
        return robust_distance(values)

    age = time.monotonic() - snapshot["last_update"]
    return (
        sector(LOCAL_FRONT_START, LOCAL_FRONT_END),
        sector(LOCAL_LEFT_START, LOCAL_LEFT_END),
        sector(LOCAL_RIGHT_START, LOCAL_RIGHT_END),
        sector(LOCAL_REAR_START, LOCAL_REAR_END),
        age,
    )


def side_unmapped_score(slam_worker, pose_holder, direction):
    """Estimate reachable unknown map area to the rover's left or right.

    Rays start outside the rover footprint and stop at the first mapped
    obstacle. Unknown cells are weighted slightly more at longer range so a
    genuinely unexplored corridor beats a side that is already well mapped.
    Returns None until enough map samples are available.
    """
    raw_map = slam_worker.get_raw_map_array()
    if raw_map is None or slam_worker.scans_processed < 4:
        return None

    x_m, y_m, heading = pose_holder.get()
    pixels_per_meter = MAP_SIZE_PIXELS / MAP_SIZE_METERS
    relative_center = math.pi / 2.0 if direction == "LEFT" else -math.pi / 2.0
    half_angle = math.radians(MAP_SIDE_HALF_ANGLE_DEG)
    angle_offsets = np.linspace(
        -half_angle,
        half_angle,
        MAP_SIDE_ANGLE_SAMPLES,
    )
    radii = np.arange(
        MAP_SIDE_MIN_RADIUS_M,
        MAP_SIDE_MAX_RADIUS_M + 0.5 * MAP_SIDE_RADIUS_STEP_M,
        MAP_SIDE_RADIUS_STEP_M,
    )

    unknown_weight = 0.0
    valid_weight = 0.0
    frontier_bonus = 0.0
    valid_samples = 0

    for offset in angle_offsets:
        ray_angle = heading + relative_center + float(offset)
        saw_free = False
        saw_unknown_after_free = False

        for radius in radii:
            world_x = x_m + math.cos(ray_angle) * float(radius)
            world_y = y_m + math.sin(ray_angle) * float(radius)
            px = int(world_x * pixels_per_meter)
            py = int(world_y * pixels_per_meter)

            if not (0 <= px < MAP_SIZE_PIXELS and 0 <= py < MAP_SIZE_PIXELS):
                break

            value = int(raw_map[py, px])
            weight = 1.0 + 0.20 * float(radius)
            valid_weight += weight
            valid_samples += 1

            if value <= MAP_OCCUPIED_THRESHOLD:
                break

            if value >= MAP_FREE_THRESHOLD:
                saw_free = True
                continue

            if MAP_UNKNOWN_LOW <= value <= MAP_UNKNOWN_HIGH:
                unknown_weight += weight
                if saw_free and not saw_unknown_after_free:
                    frontier_bonus += 1.0
                    saw_unknown_after_free = True

    if valid_samples < MAP_SIDE_MIN_VALID_SAMPLES or valid_weight <= 0.0:
        return None

    unknown_ratio = unknown_weight / valid_weight
    frontier_ratio = frontier_bonus / max(1.0, float(MAP_SIDE_ANGLE_SAMPLES))
    return unknown_ratio + 0.12 * frontier_ratio


def yellow_style_search(
    motors,
    lidar_state,
    target_found_event,
    detector,
    pose_holder,
    slam_worker,
):
    """Reactive patrol with deliberate camera sweeps and locked wall escape.

    The rover still uses the simple local navigation style, but it does not
    choose left/right again after every scan. When a wall blocks the front, it
    commits to one escape direction until the front corridor is genuinely open.
    """
    forward_pulses = 0
    initial_sweep_done = False
    escape_direction = None
    escape_turns_remaining = 0
    escape_retries = 0
    post_escape_forward_remaining = 0

    def pulse(action, duration, settle=SEARCH_SETTLE_SECONDS):
        action()
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if target_found_event.is_set():
                motors.stop()
                return "target"
            time.sleep(0.015)
        motors.stop()
        if settle > 0:
            wait_until = time.monotonic() + settle
            while time.monotonic() < wait_until:
                if target_found_event.is_set():
                    return "target"
                time.sleep(0.015)
        return None

    def choose_escape_direction(left, right, previous=None):
        # Keep a committed escape unless the opposite side is physically much
        # clearer. This prevents left/right oscillation during the turn.
        if previous == "LEFT" and left + WALL_ESCAPE_DIRECTION_MARGIN_M >= right:
            return "LEFT", None, None, "locked"
        if previous == "RIGHT" and right + WALL_ESCAPE_DIRECTION_MARGIN_M >= left:
            return "RIGHT", None, None, "locked"

        # Never choose a clearly cramped side merely because the map behind a
        # wall is unknown. Physical LiDAR clearance remains the first filter.
        if left < LOCAL_SIDE_CLEAR_DISTANCE_M <= right:
            return "RIGHT", None, None, "clearance"
        if right < LOCAL_SIDE_CLEAR_DISTANCE_M <= left:
            return "LEFT", None, None, "clearance"

        left_map = side_unmapped_score(slam_worker, pose_holder, "LEFT")
        right_map = side_unmapped_score(slam_worker, pose_holder, "RIGHT")

        if left_map is not None and right_map is not None:
            difference = left_map - right_map
            if abs(difference) >= MAP_SIDE_SCORE_DECISION_MARGIN:
                choice = "LEFT" if difference > 0.0 else "RIGHT"
                return choice, left_map, right_map, "unmapped"

        # Early in a run the map may not yet contain enough evidence. Fall back
        # to the side with greater measured clearance.
        choice = "RIGHT" if right > left else "LEFT"
        return choice, left_map, right_map, "clearance"

    def turn_once(direction, duration=SEARCH_ESCAPE_TURN_SECONDS):
        if direction == "LEFT":
            return pulse(lambda: motors.rotate_left(SEARCH_TURN_SPEED), duration)
        return pulse(lambda: motors.rotate_right(SEARCH_TURN_SPEED), duration)

    def camera_sweep():
        """Scan both sides while pausing long enough to confirm a basketball."""
        print("\nCamera sweep: checking left, right, then returning to travel heading.")
        sequence = (
            [("LEFT", CAMERA_SWEEP_STEPS_EACH_SIDE)]
            + [("RIGHT", CAMERA_SWEEP_STEPS_EACH_SIDE * 2)]
            + [("LEFT", CAMERA_SWEEP_STEPS_EACH_SIDE)]
        )
        for direction, count in sequence:
            for _ in range(count):
                if direction == "LEFT":
                    result = pulse(
                        lambda: motors.rotate_left(CAMERA_SWEEP_TURN_SPEED),
                        CAMERA_SWEEP_PULSE_SECONDS,
                        settle=CAMERA_SWEEP_DWELL_SECONDS,
                    )
                else:
                    result = pulse(
                        lambda: motors.rotate_right(CAMERA_SWEEP_TURN_SPEED),
                        CAMERA_SWEEP_PULSE_SECONDS,
                        settle=CAMERA_SWEEP_DWELL_SECONDS,
                    )
                if result:
                    return result
        return None

    while True:
        if detector.error_message:
            motors.stop()
            print(f"\nBasketball camera failed: {detector.error_message}")
            return "camera_error"

        if target_found_event.is_set():
            motors.stop()
            print("\nBasketball confirmed. Leaving search patrol for target approach.")
            return "target"

        front, left, right, rear, lidar_age = get_local_clearances(lidar_state)
        x_m, y_m, _ = pose_holder.get()
        mode = "ESCAPE" if escape_direction is not None else "PATROL"

        print(
            "\r"
            f"{mode:6s} x:{x_m:5.2f} y:{y_m:5.2f} "
            f"F:{front:4.2f} L:{left:4.2f} R:{right:4.2f} B:{rear:4.2f}",
            end="",
            flush=True,
        )

        if lidar_age > LOCAL_LIDAR_TIMEOUT_SECONDS:
            motors.stop()
            print("\nLiDAR data is stale. Stopping search for safety.")
            return "lidar_error"

        # Search the full camera field before the first patrol movement.
        if not initial_sweep_done:
            result = camera_sweep()
            initial_sweep_done = True
            if result:
                return result
            continue

        # ----------------------------------------------------
        # LOCKED WALL-ESCAPE MODE
        # ----------------------------------------------------
        if escape_direction is not None:
            # Release only when the front is comfortably open. Do not switch
            # direction merely because the left/right readings trade places.
            if front >= WALL_ESCAPE_FRONT_RELEASE_M:
                print(f"\nWall escape complete toward {escape_direction.lower()}.")
                escape_direction = None
                escape_turns_remaining = 0
                escape_retries = 0
                post_escape_forward_remaining = WALL_ESCAPE_PROGRESS_FORWARD_PULSES
                forward_pulses = 0
                continue

            if escape_turns_remaining > 0:
                result = turn_once(escape_direction)
                if result:
                    return result
                escape_turns_remaining -= 1
                continue

            # The committed turn was not enough. Back away, then continue in
            # the SAME direction. Only switch once after repeated failed tries.
            if rear > LOCAL_REAR_CLEAR_DISTANCE_M:
                result = pulse(
                    lambda: motors.reverse(SEARCH_REVERSE_SPEED),
                    WALL_ESCAPE_REVERSE_SECONDS,
                )
                if result:
                    return result

            escape_retries += 1
            if escape_retries > WALL_ESCAPE_MAX_RETRIES:
                old_direction = escape_direction
                escape_direction = "RIGHT" if escape_direction == "LEFT" else "LEFT"
                escape_retries = 0
                print(
                    f"\nEscape toward {old_direction.lower()} remained blocked; "
                    f"making one committed switch to {escape_direction.lower()}."
                )
            escape_turns_remaining = WALL_ESCAPE_RETRY_TURN_PULSES
            continue

        # After escaping a wall, drive forward briefly before allowing another
        # wall decision. This prevents turning back toward the same wall.
        if post_escape_forward_remaining > 0 and front > LOCAL_FRONT_CLEAR_DISTANCE_M:
            result = pulse(
                lambda: motors.forward(SEARCH_FORWARD_SPEED),
                SEARCH_FORWARD_PULSE_SECONDS,
            )
            if result:
                return result
            post_escape_forward_remaining -= 1
            forward_pulses += 1
            continue
        post_escape_forward_remaining = 0

        # A blocked or merely cramped front starts one committed wall escape.
        if front < LOCAL_FRONT_CLEAR_DISTANCE_M:
            motors.stop()
            if front <= LOCAL_FRONT_EMERGENCY_DISTANCE_M and rear > LOCAL_REAR_CLEAR_DISTANCE_M:
                result = pulse(
                    lambda: motors.reverse(SEARCH_REVERSE_SPEED),
                    SEARCH_REVERSE_PULSE_SECONDS,
                )
                if result:
                    return result
                front, left, right, rear, _ = get_local_clearances(lidar_state)

            (
                escape_direction,
                left_unmapped,
                right_unmapped,
                choice_reason,
            ) = choose_escape_direction(left, right, escape_direction)
            escape_turns_remaining = WALL_ESCAPE_TURN_PULSES
            escape_retries = 0
            forward_pulses = 0
            map_text = "map warming up"
            if left_unmapped is not None and right_unmapped is not None:
                map_text = (
                    f"unmapped L:{left_unmapped:.2f} R:{right_unmapped:.2f}"
                )
            print(
                f"\nFront wall/corner detected. Choosing {escape_direction.lower()} "
                f"by {choice_reason} ({map_text}); committing for "
                f"{escape_turns_remaining} scan-turns."
            )
            continue

        # ----------------------------------------------------
        # NORMAL FORWARD PATROL
        # ----------------------------------------------------
        result = pulse(
            lambda: motors.forward(SEARCH_FORWARD_SPEED),
            SEARCH_FORWARD_PULSE_SECONDS,
        )
        if result:
            return result
        forward_pulses += 1

        # Periodically stop and scan both sides. Long dwells let the detector
        # collect enough consistent frames instead of sweeping past the ball.
        if forward_pulses >= SEARCH_CAMERA_SWEEP_EVERY_FORWARD_PULSES:
            result = camera_sweep()
            forward_pulses = 0
            if result:
                return result


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
        print("AACVISION BASKETBALL SEARCH + MAPPING V8.10 MAP-AWARE CORNER SEARCH")
        print("=" * 72)
        print("Navigation style: faster forward patrol, camera sweeps, and map-aware locked corner escape.")
        print("At corners, the rover favors the safer side with more reachable unmapped SLAM area.")
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

        print("RealSense and YOLO are ready. Starting SLAM and map-aware corner search.")
        slam_worker = SlamWorker(lidar_state, odometry_tracker, pose_holder)
        slam_worker.start()
        time.sleep(0.8)

        while True:
            exploration_result = yellow_style_search(
                motors,
                lidar_state,
                target_found_event,
                basketball_detector,
                pose_holder,
                slam_worker,
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
                print("Returning to map-aware local search patrol.")
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
                map_path = slam_worker.save_map(f"basketball_map_aware_final_{timestamp}.png")
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
