#!/usr/bin/env python3
"""
AACVision Frontier-Based Basketball Search and Room Mapping V10
----------------------------------------------------------------
This version keeps the working RealSense/YOLO basketball detector,
small stop-and-recheck basketball approach, motor calibration, encoder
odometry, LiDAR reader, and BreezySLAM map generation from V8.3.

Room exploration is changed to frontier-based navigation:

  1. Stop and wait for a fresh 360-degree LiDAR scan.
  2. Read the current BreezySLAM occupancy map.
  3. Find frontiers: safe known-free cells that touch unknown space.
  4. Cluster and rank the frontiers.
  5. Plan a collision-aware A* path through known free space.
  6. Turn toward a short look-ahead waypoint and drive one bounded step.
  7. Stop, update the map, replan, and repeat.

If no frontier remains, the rover performs a stepped 360-degree scan and
then patrols a low-visit safe part of the known map so the camera can keep
searching. The mission has no timer. It ends when the rover reaches the
basketball, a safety-critical fault occurs, or Ctrl+C is pressed.

SAFETY
------
Test with wheels lifted first, then in a clear room. Keep immediate access
to motor power. Do not operate near stairs, pets, people, or fragile items.
"""

import heapq
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

# Frontier navigation uses a coarser planning grid derived from the SLAM map.
# This keeps A* fast while the saved SLAM image remains high resolution.
FRONTIER_GRID_RESOLUTION_M = 0.08
FRONTIER_GRID_CELLS = 200
FRONTIER_FREE_PIXEL_THRESHOLD = 200
FRONTIER_OCCUPIED_PIXEL_THRESHOLD = 80
FRONTIER_FREE_BLOCK_FRACTION = 0.35
FRONTIER_OCCUPIED_BLOCK_FRACTION = 0.02
FRONTIER_ROBOT_INFLATION_RADIUS_M = 0.28
FRONTIER_MIN_CLUSTER_CELLS = 5
FRONTIER_MIN_GOAL_DISTANCE_M = 0.45
FRONTIER_MAX_CANDIDATES_TO_PLAN = 10
FRONTIER_CLUSTER_SIZE_CREDIT_M = 0.012
FRONTIER_LOOKAHEAD_DISTANCE_M = 0.55
FRONTIER_MAX_DRIVE_STEP_M = 0.55
FRONTIER_MIN_DRIVE_STEP_M = 0.12
FRONTIER_SWEEP_INCREMENT_DEG = 45.0
FRONTIER_SWEEP_SETTLE_SECONDS = 0.25
FRONTIER_BLACKLIST_SECONDS = 18.0
FRONTIER_PATROL_MIN_DISTANCE_M = 0.90
FRONTIER_VISIT_RADIUS_CELLS = 2

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
# FRONTIER MAP CLASSIFICATION + A* PLANNING
# ============================================================

def _dilate_boolean_mask(mask, radius_cells):
    """Circular binary dilation implemented with NumPy only."""
    if radius_cells <= 0:
        return mask.copy()

    height, width = mask.shape
    result = mask.copy()

    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_cells * radius_cells:
                continue

            src_y0 = max(0, -dy)
            src_y1 = min(height, height - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(width, width - dx)
            dst_y0 = src_y0 + dy
            dst_y1 = src_y1 + dy
            dst_x0 = src_x0 + dx
            dst_x1 = src_x1 + dx

            result[dst_y0:dst_y1, dst_x0:dst_x1] |= mask[
                src_y0:src_y1,
                src_x0:src_x1,
            ]

    return result


def _path_length(points):
    total = 0.0
    for first, second in zip(points, points[1:]):
        total += math.hypot(second[0] - first[0], second[1] - first[1])
    return total


class FrontierPlanner:
    """Derives frontiers and A* paths from the current BreezySLAM map."""

    _NEIGHBORS = (
        (-1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, math.sqrt(2.0)),
        (1, 0, 1.0),
        (1, 1, math.sqrt(2.0)),
    )

    def __init__(self, slam_worker):
        self.slam_worker = slam_worker
        self.visit_counts = np.zeros(
            (FRONTIER_GRID_CELLS, FRONTIER_GRID_CELLS),
            dtype=np.uint16,
        )
        self.blacklisted_goals = []

    def world_to_grid(self, x_m, y_m):
        gx = int(x_m / FRONTIER_GRID_RESOLUTION_M)
        gy = int(y_m / FRONTIER_GRID_RESOLUTION_M)
        return gx, gy

    def grid_to_world(self, gx, gy):
        return (
            (gx + 0.5) * FRONTIER_GRID_RESOLUTION_M,
            (gy + 0.5) * FRONTIER_GRID_RESOLUTION_M,
        )

    def valid(self, gx, gy):
        return (
            0 <= gx < FRONTIER_GRID_CELLS
            and 0 <= gy < FRONTIER_GRID_CELLS
        )

    def mark_visited(self, pose):
        gx, gy = self.world_to_grid(pose[0], pose[1])
        radius = FRONTIER_VISIT_RADIUS_CELLS
        y0 = max(0, gy - radius)
        y1 = min(FRONTIER_GRID_CELLS, gy + radius + 1)
        x0 = max(0, gx - radius)
        x1 = min(FRONTIER_GRID_CELLS, gx + radius + 1)
        if x0 < x1 and y0 < y1:
            region = self.visit_counts[y0:y1, x0:x1].astype(np.uint32) + 1
            self.visit_counts[y0:y1, x0:x1] = np.minimum(region, 65535).astype(np.uint16)

    def blacklist(self, goal_world):
        gx, gy = self.world_to_grid(*goal_world)
        self.blacklisted_goals.append(
            (gx, gy, time.monotonic() + FRONTIER_BLACKLIST_SECONDS)
        )

    def _is_blacklisted(self, gx, gy):
        now = time.monotonic()
        self.blacklisted_goals = [item for item in self.blacklisted_goals if item[2] > now]
        for bx, by, _expiry in self.blacklisted_goals:
            if (gx - bx) ** 2 + (gy - by) ** 2 <= 4 ** 2:
                return True
        return False

    def _planning_layers(self):
        raw = self.slam_worker.get_raw_map_array()
        if raw is None:
            return None

        factor = MAP_SIZE_PIXELS // FRONTIER_GRID_CELLS
        usable = factor * FRONTIER_GRID_CELLS
        raw = raw[:usable, :usable]
        blocks = raw.reshape(
            FRONTIER_GRID_CELLS,
            factor,
            FRONTIER_GRID_CELLS,
            factor,
        )

        occupied_fraction = np.mean(
            blocks <= FRONTIER_OCCUPIED_PIXEL_THRESHOLD,
            axis=(1, 3),
        )
        free_fraction = np.mean(
            blocks >= FRONTIER_FREE_PIXEL_THRESHOLD,
            axis=(1, 3),
        )

        occupied = occupied_fraction >= FRONTIER_OCCUPIED_BLOCK_FRACTION
        known_free = (
            free_fraction >= FRONTIER_FREE_BLOCK_FRACTION
        ) & ~occupied
        unknown = ~(known_free | occupied)

        inflation_cells = max(
            1,
            int(math.ceil(
                FRONTIER_ROBOT_INFLATION_RADIUS_M
                / FRONTIER_GRID_RESOLUTION_M
            )),
        )
        inflated = _dilate_boolean_mask(occupied, inflation_cells)
        safe_free = known_free & ~inflated

        return safe_free, unknown, occupied, inflated

    def _frontier_mask(self, safe_free, unknown):
        neighbor_unknown = np.zeros_like(unknown)
        neighbor_unknown[1:, :] |= unknown[:-1, :]
        neighbor_unknown[:-1, :] |= unknown[1:, :]
        neighbor_unknown[:, 1:] |= unknown[:, :-1]
        neighbor_unknown[:, :-1] |= unknown[:, 1:]

        frontier = safe_free & neighbor_unknown
        frontier[0, :] = False
        frontier[-1, :] = False
        frontier[:, 0] = False
        frontier[:, -1] = False
        return frontier

    def _clusters(self, mask):
        ys, xs = np.nonzero(mask)
        remaining = set(zip(ys.tolist(), xs.tolist()))
        clusters = []

        while remaining:
            seed = remaining.pop()
            stack = [seed]
            members = [seed]

            while stack:
                cy, cx = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        neighbor = (cy + dy, cx + dx)
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            stack.append(neighbor)
                            members.append(neighbor)

            if len(members) >= FRONTIER_MIN_CLUSTER_CELLS:
                clusters.append(members)

        return clusters

    def _nearest_safe_cell(self, safe_free, gx, gy, max_radius=14):
        if self.valid(gx, gy) and safe_free[gy, gx]:
            return gx, gy

        for radius in range(1, max_radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = gx + dx, gy + dy
                    if self.valid(nx, ny) and safe_free[ny, nx]:
                        return nx, ny
        return None

    def _astar(self, safe_free, start_world, goal_cell):
        sx, sy = self.world_to_grid(*start_world)
        start = self._nearest_safe_cell(safe_free, sx, sy)
        goal = self._nearest_safe_cell(safe_free, goal_cell[0], goal_cell[1])
        if start is None or goal is None:
            return None

        open_heap = [(0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        closed = set()
        expansions = 0
        max_expansions = FRONTIER_GRID_CELLS * FRONTIER_GRID_CELLS

        while open_heap:
            _priority, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            expansions += 1
            if expansions > max_expansions:
                return None

            if current == goal:
                cells = [current]
                while current in came_from:
                    current = came_from[current]
                    cells.append(current)
                cells.reverse()
                return [self.grid_to_world(gx, gy) for gx, gy in cells]

            cx, cy = current
            for dx, dy, step_cost in self._NEIGHBORS:
                nx, ny = cx + dx, cy + dy
                if not self.valid(nx, ny) or not safe_free[ny, nx]:
                    continue

                if dx != 0 and dy != 0:
                    if not safe_free[cy, nx] or not safe_free[ny, cx]:
                        continue

                neighbor = (nx, ny)
                tentative = g_score[current] + step_cost
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative
                heuristic = math.hypot(nx - goal[0], ny - goal[1])
                heapq.heappush(open_heap, (tentative + heuristic, neighbor))

        return None

    def _representative_cell(self, members, robot_cell):
        minimum_cells = FRONTIER_MIN_GOAL_DISTANCE_M / FRONTIER_GRID_RESOLUTION_M
        ranked = sorted(
            members,
            key=lambda cell: (
                (cell[1] - robot_cell[0]) ** 2
                + (cell[0] - robot_cell[1]) ** 2
            ),
        )
        for gy, gx in ranked:
            distance_cells = math.hypot(gx - robot_cell[0], gy - robot_cell[1])
            if distance_cells >= minimum_cells and not self._is_blacklisted(gx, gy):
                return gx, gy
        return None

    def plan_best_frontier(self, pose):
        layers = self._planning_layers()
        if layers is None:
            return None
        safe_free, unknown, _occupied, _inflated = layers
        frontier_mask = self._frontier_mask(safe_free, unknown)
        clusters = self._clusters(frontier_mask)
        robot_cell = self.world_to_grid(pose[0], pose[1])

        ranked = []
        for members in clusters:
            representative = self._representative_cell(members, robot_cell)
            if representative is None:
                continue
            gx, gy = representative
            distance = math.hypot(
                gx - robot_cell[0],
                gy - robot_cell[1],
            ) * FRONTIER_GRID_RESOLUTION_M
            pre_score = distance - FRONTIER_CLUSTER_SIZE_CREDIT_M * len(members)
            ranked.append((pre_score, representative, len(members)))

        ranked.sort(key=lambda item: item[0])
        best = None

        for _pre_score, goal_cell, cluster_size in ranked[:FRONTIER_MAX_CANDIDATES_TO_PLAN]:
            path = self._astar(safe_free, (pose[0], pose[1]), goal_cell)
            if not path or len(path) < 2:
                continue

            length = _path_length(path)
            gx, gy = goal_cell
            visit_penalty = float(self.visit_counts[gy, gx]) * 0.08
            goal_world = self.grid_to_world(gx, gy)
            goal_heading = math.atan2(goal_world[1] - pose[1], goal_world[0] - pose[0])
            heading_penalty = abs(normalize_angle(goal_heading - pose[2])) * 0.08
            score = (
                length
                - FRONTIER_CLUSTER_SIZE_CREDIT_M * cluster_size
                + visit_penalty
                + heading_penalty
            )

            candidate = {
                "mode": "frontier",
                "goal": goal_world,
                "goal_cell": goal_cell,
                "cluster_size": cluster_size,
                "path": path,
                "path_length": length,
                "score": score,
                "frontier_clusters": len(clusters),
            }
            if best is None or candidate["score"] < best["score"]:
                best = candidate

        return best

    def plan_low_visit_patrol(self, pose):
        layers = self._planning_layers()
        if layers is None:
            return None
        safe_free, _unknown, _occupied, _inflated = layers
        robot_gx, robot_gy = self.world_to_grid(pose[0], pose[1])

        choices = []
        stride = 4
        for gy in range(0, FRONTIER_GRID_CELLS, stride):
            for gx in range(0, FRONTIER_GRID_CELLS, stride):
                if not safe_free[gy, gx] or self._is_blacklisted(gx, gy):
                    continue
                distance = math.hypot(gx - robot_gx, gy - robot_gy) * FRONTIER_GRID_RESOLUTION_M
                if distance < FRONTIER_PATROL_MIN_DISTANCE_M:
                    continue
                visits = float(self.visit_counts[gy, gx])
                score = distance - visits * 0.65
                choices.append((score, gx, gy, visits))

        choices.sort(reverse=True)
        for _score, gx, gy, visits in choices[:20]:
            path = self._astar(safe_free, (pose[0], pose[1]), (gx, gy))
            if path and len(path) >= 2:
                return {
                    "mode": "coverage-patrol",
                    "goal": self.grid_to_world(gx, gy),
                    "goal_cell": (gx, gy),
                    "cluster_size": 0,
                    "path": path,
                    "path_length": _path_length(path),
                    "score": _score,
                    "frontier_clusters": 0,
                    "visits": visits,
                }
        return None

    def lookahead_waypoint(self, pose, path):
        chosen = path[1] if len(path) > 1 else path[0]
        for point in path[1:]:
            distance = math.hypot(point[0] - pose[0], point[1] - pose[1])
            if distance <= FRONTIER_LOOKAHEAD_DISTANCE_M:
                chosen = point
            else:
                break
        return chosen


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

# These are the same thresholds that worked in the camera-only test.
BASKETBALL_CLASS_NAMES = ("sports ball", "basketball")
BASKETBALL_AI_MIN_CONFIDENCE = 0.08
BASKETBALL_MIN_BOX_AREA_FRACTION = 0.0005
ORANGE_HSV_LOWER = np.array([2, 50, 30], dtype=np.uint8)
ORANGE_HSV_UPPER = np.array([36, 255, 255], dtype=np.uint8)
ORANGE_MIN_FRACTION = 0.008
BASKETBALL_MIN_ROUNDNESS = 0.25
BASKETBALL_CONFIRM_SCORE = 0.55
BASKETBALL_MAX_DETECTION_AGE_SECONDS = 3.5
BASKETBALL_REACQUIRE_SECONDS = 8.0

# Stop and approach settings.
BASKETBALL_STOP_DISTANCE_M = 0.80
BASKETBALL_CENTER_TOLERANCE_PIXELS = 60
BASKETBALL_CAPTURE_CENTER_TOLERANCE_PIXELS = 70

# V8.3 uses small stop-and-recheck steps. It is quicker than the original
# cautious mode, but it never keeps driving on an old camera observation.
BASKETBALL_ALIGNMENT_TURN_SPEED = 19
BASKETBALL_APPROACH_FAST_SPEED = 22
BASKETBALL_APPROACH_MEDIUM_SPEED = 19
BASKETBALL_APPROACH_NEAR_SPEED = 14
BASKETBALL_FAR_DISTANCE_M = 2.00
BASKETBALL_NEAR_DISTANCE_M = 1.15
BASKETBALL_FAR_FORWARD_PULSE_SECONDS = 0.45
BASKETBALL_MEDIUM_FORWARD_PULSE_SECONDS = 0.30
BASKETBALL_NEAR_FORWARD_PULSE_SECONDS = 0.14
BASKETBALL_MIN_TURN_PULSE_SECONDS = 0.07
BASKETBALL_MAX_TURN_PULSE_SECONDS = 0.22
BASKETBALL_REACQUIRE_TURN_PULSE_SECONDS = 0.11
BASKETBALL_POST_MOTION_SETTLE_SECONDS = 0.05
BASKETBALL_CLOSE_CONFIRM_FRAMES = 1
BASKETBALL_DEPTH_MIN_VALID_PIXELS = 10
BASKETBALL_DEPTH_PERCENTILE = 35.0

# Fast orange-ball fallback. This runs before YOLO and lets the rover react to
# a close, clearly orange circular basketball at camera frame rate. YOLO remains
# the long-range/general detector. The shape checks reject most orange walls,
# cabinets, and floor patches.
COLOR_FALLBACK_MIN_AREA_PIXELS = 450
COLOR_FALLBACK_MAX_AREA_FRACTION = 0.55
COLOR_FALLBACK_MIN_CIRCULARITY = 0.42
COLOR_FALLBACK_MIN_ASPECT_RATIO = 0.62
COLOR_FALLBACK_MIN_FILL_RATIO = 0.32
COLOR_FALLBACK_MIN_ORANGE_FRACTION = 0.12
COLOR_FALLBACK_MAX_DISTANCE_M = 5.5
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

            # A strong YOLO sports-ball result is accepted even when motion blur,
            # black seams, shadows, or partial cropping reduce the orange score.
            # Lower-confidence YOLO results still need orange and shape evidence.
            strong_yolo = confidence >= 0.35
            appearance_ok = (
                orange_fraction >= ORANGE_MIN_FRACTION
                and roundness >= BASKETBALL_MIN_ROUNDNESS
            )
            if not strong_yolo and not appearance_ok:
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

            orange_fraction, box_roundness = self._appearance_evidence(frame, bbox)
            if orange_fraction < COLOR_FALLBACK_MIN_ORANGE_FRACTION:
                continue

            distance_m = self._depth_from_bbox(depth_image, bbox)
            if distance_m is None or distance_m > COLOR_FALLBACK_MAX_DISTANCE_M:
                continue

            # Pseudo-confidence remains below a strong YOLO confidence so two
            # consecutive color frames are normally required for confirmation.
            confidence = min(0.34, 0.18 + 0.10 * circularity + 0.08 * fill_ratio)
            score = (
                0.45 * circularity
                + 0.25 * aspect_ratio
                + 0.20 * fill_ratio
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
    """Align and approach with brisk, bounded stop-and-recheck steps.

    Every turn or forward step is followed by a fresh camera observation. This
    prevents the rover from continuing on a stale target position while still
    moving noticeably faster than the original cautious approach.
    """
    motors.stop()
    last_frame_time = float("-inf")
    last_horizontal_error = 0.0

    print("\nBasketball confirmed. Beginning smooth step-and-recheck approach mode.")

    def motion_pulse(action, seconds):
        action()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            front_clearance = get_instant_center_clearance(lidar_state)
            if front_clearance <= FRONT_EMERGENCY_DISTANCE:
                motors.stop()
                print(
                    f"\nEmergency stop during approach: obstacle at "
                    f"{front_clearance:.2f}m."
                )
                return False
            time.sleep(0.015)

        motors.stop()
        time.sleep(BASKETBALL_POST_MOTION_SETTLE_SECONDS)
        return True

    def turn_pulse_for_error(horizontal_error):
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

        # Do not issue another movement command until the detector has produced
        # a new observation after the previous movement.
        if observation.frame_time <= last_frame_time + 1e-6:
            motors.stop()
            time.sleep(0.015)
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

            # Search in small steps toward the most recent target direction.
            if last_horizontal_error < 0:
                ok = motion_pulse(
                    lambda: motors.rotate_left(BASKETBALL_ALIGNMENT_TURN_SPEED),
                    BASKETBALL_REACQUIRE_TURN_PULSE_SECONDS,
                )
                search_text = "left"
            else:
                ok = motion_pulse(
                    lambda: motors.rotate_right(BASKETBALL_ALIGNMENT_TURN_SPEED),
                    BASKETBALL_REACQUIRE_TURN_PULSE_SECONDS,
                )
                search_text = "right"

            print(
                f"\rBALL TEMPORARILY LOST - short search step {search_text:>5s}",
                end="",
                flush=True,
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
            f"STEP APPROACH conf:{observation.confidence:.2f} "
            f"orange:{observation.orange_fraction:.2f} "
            f"error:{horizontal_error:+5.0f}px depth:{distance_text:>7s}",
            end="",
            flush=True,
        )

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

        # Align first. Forward motion is allowed only when the ball is inside
        # the center corridor in the newest observation.
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
        """Thread-safe copy of the raw BreezySLAM grayscale map."""
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
                self.save_map("basketball_frontier_live_map.png")
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


def perform_frontier_scan_sweep(
    motors,
    lidar_state,
    pose_holder,
    target_found_event,
    detector,
):
    """Stepped 360-degree sweep to expose additional map frontiers."""
    print("\nNo reachable frontier yet. Performing a stepped 360-degree scan sweep...")
    _x, _y, starting_heading = pose_holder.get()
    steps = int(round(360.0 / FRONTIER_SWEEP_INCREMENT_DEG))

    for index in range(1, steps + 1):
        if target_found_event.is_set() or detector.error_message:
            motors.stop()
            return

        target_heading = normalize_angle(
            starting_heading
            + math.radians(index * FRONTIER_SWEEP_INCREMENT_DEG)
        )
        rotate_to_heading(
            motors,
            pose_holder,
            target_heading,
            target_found_event,
        )
        motors.stop()
        time.sleep(FRONTIER_SWEEP_SETTLE_SECONDS)
        previous_scan_time = lidar_state.snapshot()["last_update"]
        wait_for_fresh_scan(lidar_state, previous_scan_time)


def stepwise_explore(
    motors,
    lidar_state,
    pose_holder,
    slam_worker,
    target_found_event,
    detector,
    frontier_planner,
):
    """Frontier selection + A* path planning with bounded stop-and-go motion."""
    consecutive_stuck = 0
    no_plan_cycles = 0

    while True:
        if detector.error_message:
            print(f"\nBasketball camera failed: {detector.error_message}")
            motors.stop()
            return "camera_error"

        if target_found_event.is_set():
            print("\nBasketball detected. Stopping frontier exploration for approach.")
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

        pose = pose_holder.get()
        frontier_planner.mark_visited(pose)
        plan = frontier_planner.plan_best_frontier(pose)

        if plan is None:
            no_plan_cycles += 1
            if no_plan_cycles == 1 or no_plan_cycles % 3 == 0:
                perform_frontier_scan_sweep(
                    motors,
                    lidar_state,
                    pose_holder,
                    target_found_event,
                    detector,
                )
                if target_found_event.is_set():
                    return "target"
                pose = pose_holder.get()
                frontier_planner.mark_visited(pose)
                plan = frontier_planner.plan_best_frontier(pose)

            if plan is None:
                plan = frontier_planner.plan_low_visit_patrol(pose)

            if plan is None:
                print("No safe frontier or patrol path available. Waiting for a new map update.")
                time.sleep(0.8)
                continue
        else:
            no_plan_cycles = 0

        waypoint = frontier_planner.lookahead_waypoint(pose, plan["path"])
        waypoint_distance = math.hypot(
            waypoint[0] - pose[0],
            waypoint[1] - pose[1],
        )
        if waypoint_distance < FRONTIER_MIN_DRIVE_STEP_M:
            frontier_planner.blacklist(plan["goal"])
            continue

        target_heading = math.atan2(
            waypoint[1] - pose[1],
            waypoint[0] - pose[0],
        )

        if plan["mode"] == "frontier":
            plan_label = (
                f"frontier size={plan['cluster_size']} "
                f"clusters={plan['frontier_clusters']}"
            )
        else:
            plan_label = f"coverage patrol visits={plan.get('visits', 0):.0f}"

        print(
            f"\nAt x:{pose[0]:5.2f}m y:{pose[1]:5.2f}m -- "
            f"{plan_label}; A* path={plan['path_length']:.2f}m; "
            f"waypoint=({waypoint[0]:.2f}, {waypoint[1]:.2f})"
        )

        rotate_to_heading(
            motors,
            pose_holder,
            target_heading,
            target_found_event,
        )
        if target_found_event.is_set():
            motors.stop()
            return "target"

        after_turn_time = lidar_state.snapshot()["last_update"]
        fresh_scan = wait_for_fresh_scan(lidar_state, after_turn_time)
        if not fresh_scan:
            motors.stop()
            return "lidar_error"

        center_clearance = get_instant_center_clearance(lidar_state)
        if not math.isfinite(center_clearance):
            center_clearance = FRONTIER_MAX_DRIVE_STEP_M

        safe_distance = max(
            0.0,
            center_clearance - FRONT_EMERGENCY_DISTANCE - 0.08,
        )
        step_distance = min(
            waypoint_distance,
            FRONTIER_MAX_DRIVE_STEP_M,
            safe_distance,
        )

        if step_distance < FRONTIER_MIN_DRIVE_STEP_M:
            consecutive_stuck += 1
            frontier_planner.blacklist(plan["goal"])
            print(
                f"Planned path is blocked after turning "
                f"(attempt {consecutive_stuck}/{MAX_CONSECUTIVE_STUCK})."
            )

            if consecutive_stuck >= MAX_CONSECUTIVE_STUCK:
                print("Backing up and rotating briefly before replanning.")
                motors.reverse(REVERSE_SPEED)
                time.sleep(STUCK_REVERSE_SECONDS)
                motors.stop()
                motors.rotate_left(TURN_SPEED)
                time.sleep(0.45)
                motors.stop()
                consecutive_stuck = 0
            continue

        consecutive_stuck = 0
        print(f"Driving {step_distance:.2f}m along the planned path...")
        drive_step(
            motors,
            pose_holder,
            lidar_state,
            step_distance,
            target_found_event,
        )
        motors.stop()

        frontier_planner.mark_visited(pose_holder.get())
        if target_found_event.is_set():
            return "target"


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
        print("AACVISION FRONTIER BASKETBALL SEARCH + LIDAR MAPPING V10")
        print("=" * 72)
        print("Frontier exploration: SLAM map -> frontier clusters -> A* path -> bounded step.")
        print("Robust orange/YOLO detection with the V8.3 stop-and-recheck ball approach.")
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
        time.sleep(1.2)
        frontier_planner = FrontierPlanner(slam_worker)

        while True:
            exploration_result = stepwise_explore(
                motors,
                lidar_state,
                pose_holder,
                slam_worker,
                target_found_event,
                basketball_detector,
                frontier_planner,
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
                print("Returning to frontier mapping search.")
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
                map_path = slam_worker.save_map(f"basketball_frontier_final_{timestamp}.png")
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
