#!/usr/bin/env python3
"""
C1 LiDAR + Intel RealSense D435 yellow-target explorer V2.

Improvements over the first explorer:
- Uses a wall-escape state machine.
- Locks one turn direction during obstacle avoidance.
- Does not switch left/right every sensor update.
- Requires forward progress after a turn before choosing another turn.
- Reverses briefly when boxed in.
- Searches for a yellow square while driving around.
- Centers on the target, approaches it, and stops in front of it.

This is reactive exploration, not full SLAM.
"""

import math
import os
import random
import re
import signal
import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs
import RPi.GPIO as GPIO


# ============================================================
# MOTOR GPIO CONFIGURATION
# ============================================================

MOTORS = {
    "front_left": {"enable": 12, "in1": 5, "in2": 6},
    "front_right": {"enable": 13, "in1": 20, "in2": 21},
    "rear_left": {"enable": 18, "in1": 23, "in2": 24},
    "rear_right": {"enable": 19, "in1": 15, "in2": 26},
}

MOTOR_POLARITY = {
    "front_left": 1,
    "front_right": 1,
    "rear_left": 1,
    "rear_right": 1,
}


# ============================================================
# SPEEDS
# ============================================================

PWM_FREQUENCY = 1000

EXPLORE_SPEED = 32
CAUTIOUS_SPEED = 25
TURN_SPEED = 34
ALIGN_TURN_SPEED = 27
APPROACH_SPEED = 29
SLOW_APPROACH_SPEED = 22
REVERSE_SPEED = 25


# ============================================================
# MOTION TIMING
# ============================================================

# Avoidance turn is committed for a fixed short period.
AVOID_TURN_TIME = 0.48

# After turning, drive forward before reconsidering direction.
ESCAPE_FORWARD_TIME = 0.70

# Emergency recovery.
REVERSE_TIME = 0.42
RECOVERY_TURN_TIME = 0.58

# Periodic camera sweep during normal exploration.
SCAN_INTERVAL_SECONDS = 8.0
SCAN_TURN_TIME = 0.24

# Do not choose the opposite avoidance direction immediately.
TURN_DIRECTION_HOLD_TIME = 2.0


# ============================================================
# TARGET SETTINGS
# ============================================================

TARGET_STOP_DISTANCE = 0.60
TARGET_SLOW_DISTANCE = 1.20

CENTER_TOLERANCE_PIXELS = 35
MIN_TARGET_AREA = 900

TARGET_CONFIRM_FRAMES = 4
TARGET_LOST_FRAMES = 7

YELLOW_LOWER = np.array([18, 110, 100], dtype=np.uint8)
YELLOW_UPPER = np.array([38, 255, 255], dtype=np.uint8)


# ============================================================
# OBSTACLE SETTINGS
# ============================================================

FRONT_EMERGENCY_DISTANCE = 0.38
FRONT_TURN_DISTANCE = 0.72
FRONT_CLEAR_DISTANCE = 0.95

SIDE_EMERGENCY_DISTANCE = 0.38
SIDE_COMFORT_DISTANCE = 0.62

REAR_CLEAR_DISTANCE = 0.65
LIDAR_TIMEOUT = 2.0

# Opposite side must be substantially more open before switching.
TURN_SWITCH_MARGIN = 0.45


# ============================================================
# C1 LIDAR SETTINGS
# ============================================================

LIDAR_PROGRAM = "/home/pi/rplidar_sdk/output/Linux/Release/ultra_simple"
LIDAR_PORT = "/dev/ttyUSB0"
LIDAR_BAUD = "460800"

FRONT_START = 340
FRONT_END = 20

LEFT_START = 30
LEFT_END = 110

RIGHT_START = 250
RIGHT_END = 330

REAR_START = 155
REAR_END = 205

LIDAR_PATTERN = re.compile(
    r"theta:\s*([0-9.]+)\s+"
    r"Dist:\s*([0-9.]+)\s+"
    r"Q:\s*([0-9]+)"
)


# ============================================================
# MOTOR CONTROLLER
# ============================================================

class MotorController:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

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
        speed = max(0, min(100, speed))

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

    def forward(self, speed=EXPLORE_SPEED):
        for name in MOTORS:
            self.set_motor(name, 1, speed)

    def reverse(self):
        for name in MOTORS:
            self.set_motor(name, -1, REVERSE_SPEED)

    def rotate_left(self, speed=TURN_SPEED):
        self.set_motor("front_left", -1, speed)
        self.set_motor("rear_left", -1, speed)
        self.set_motor("front_right", 1, speed)
        self.set_motor("rear_right", 1, speed)

    def rotate_right(self, speed=TURN_SPEED):
        self.set_motor("front_left", 1, speed)
        self.set_motor("rear_left", 1, speed)
        self.set_motor("front_right", -1, speed)
        self.set_motor("rear_right", -1, speed)

    def curve_left(self, speed=EXPLORE_SPEED):
        inner = max(0, speed - 8)

        self.set_motor("front_left", 1, inner)
        self.set_motor("rear_left", 1, inner)
        self.set_motor("front_right", 1, speed)
        self.set_motor("rear_right", 1, speed)

    def curve_right(self, speed=EXPLORE_SPEED):
        inner = max(0, speed - 8)

        self.set_motor("front_left", 1, speed)
        self.set_motor("rear_left", 1, speed)
        self.set_motor("front_right", 1, inner)
        self.set_motor("rear_right", 1, inner)

    def cleanup(self):
        self.stop()

        for pwm in self.pwm.values():
            pwm.stop()

        GPIO.cleanup()


# ============================================================
# LIDAR PROCESSING
# ============================================================

def angle_in_sector(angle, start, end):
    angle %= 360.0
    start %= 360.0
    end %= 360.0

    if start <= end:
        return start <= angle <= end

    return angle >= start or angle <= end


def get_sector(points, start, end):
    values = []

    for angle, distance_mm, quality in points:
        if quality <= 0 or distance_mm <= 0:
            continue

        if angle_in_sector(angle, start, end):
            values.append(distance_mm / 1000.0)

    return values


def robust_distance(values):
    if not values:
        return float("inf")

    values = sorted(values)
    count = min(8, len(values))

    return sum(values[:count]) / count


class LidarState:
    def __init__(self):
        self.lock = threading.Lock()

        self.front = float("inf")
        self.left = float("inf")
        self.right = float("inf")
        self.rear = float("inf")
        self.last_update = 0.0

    def update(self, front, left, right, rear):
        with self.lock:
            self.front = front
            self.left = left
            self.right = right
            self.rear = rear
            self.last_update = time.monotonic()

    def read(self):
        with self.lock:
            return (
                self.front,
                self.left,
                self.right,
                self.rear,
                self.last_update,
            )


class LidarReader(threading.Thread):
    def __init__(self, state):
        super().__init__(daemon=True)

        self.state = state
        self.process = None
        self.running = True

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

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )

        current_scan = []

        try:
            if self.process.stdout is None:
                return

            for raw_line in self.process.stdout:
                if not self.running:
                    break

                match = LIDAR_PATTERN.search(raw_line)

                if match is None:
                    continue

                angle = float(match.group(1))
                distance = float(match.group(2))
                quality = int(match.group(3))

                new_scan = raw_line.lstrip().startswith("S")

                if new_scan and current_scan:
                    front = robust_distance(
                        get_sector(current_scan, FRONT_START, FRONT_END)
                    )
                    left = robust_distance(
                        get_sector(current_scan, LEFT_START, LEFT_END)
                    )
                    right = robust_distance(
                        get_sector(current_scan, RIGHT_START, RIGHT_END)
                    )
                    rear = robust_distance(
                        get_sector(current_scan, REAR_START, REAR_END)
                    )

                    self.state.update(front, left, right, rear)
                    current_scan = []

                current_scan.append((angle, distance, quality))

        except Exception as error:
            print(f"\nLIDAR error: {error}")

    def stop_reader(self):
        self.running = False

        if self.process is not None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass


# ============================================================
# TARGET DETECTION
# ============================================================

def detect_yellow_square(color_image):
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best = None
    best_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_TARGET_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        x, y, width, height = cv2.boundingRect(contour)

        if height <= 0:
            continue

        aspect_ratio = width / height
        rectangularity = area / max(1.0, width * height)

        square_like = 0.65 <= aspect_ratio <= 1.35
        filled = rectangularity >= 0.55
        simple_shape = 4 <= len(polygon) <= 8

        if square_like and filled and simple_shape and area > best_area:
            best = {
                "center_x": x + width // 2,
                "center_y": y + height // 2,
                "area": area,
            }
            best_area = area

    return best


def target_depth(depth_frame, target):
    width = depth_frame.get_width()
    height = depth_frame.get_height()

    center_x = int(np.clip(target["center_x"], 0, width - 1))
    center_y = int(np.clip(target["center_y"], 0, height - 1))

    samples = []

    for offset_y in range(-5, 6, 2):
        for offset_x in range(-5, 6, 2):
            x = int(np.clip(center_x + offset_x, 0, width - 1))
            y = int(np.clip(center_y + offset_y, 0, height - 1))

            distance = depth_frame.get_distance(x, y)

            if 0.10 < distance < 10.0:
                samples.append(distance)

    if not samples:
        return float("inf")

    return float(np.median(samples))


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    motors = MotorController()

    lidar_state = LidarState()
    lidar_reader = LidarReader(lidar_state)

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.depth,
        424,
        240,
        rs.format.z16,
        15,
    )

    config.enable_stream(
        rs.stream.color,
        424,
        240,
        rs.format.bgr8,
        15,
    )

    align = rs.align(rs.stream.color)

    detection_history = deque(maxlen=TARGET_CONFIRM_FRAMES)

    target_acquired = False
    lost_count = 0

    state = "STARTING"

    motion_phase = None
    motion_until = 0.0

    active_turn = None
    last_turn_direction = random.choice(["LEFT", "RIGHT"])
    last_turn_time = 0.0

    last_scan_time = time.monotonic()

    def set_state(new_state):
        nonlocal state

        if new_state != state:
            state = new_state
            print(f"\nSTATE: {state}")

    def select_escape_direction(left, right, now):
        nonlocal last_turn_direction

        difference = left - right

        # Hold the previous direction briefly unless the other
        # side is clearly more open.
        if now - last_turn_time < TURN_DIRECTION_HOLD_TIME:
            if last_turn_direction == "LEFT":
                if difference < -TURN_SWITCH_MARGIN:
                    return "RIGHT"
                return "LEFT"

            if difference > TURN_SWITCH_MARGIN:
                return "LEFT"
            return "RIGHT"

        if difference > TURN_SWITCH_MARGIN:
            return "LEFT"

        if difference < -TURN_SWITCH_MARGIN:
            return "RIGHT"

        return last_turn_direction

    def begin_avoidance(direction, now):
        nonlocal motion_phase
        nonlocal motion_until
        nonlocal active_turn
        nonlocal last_turn_direction
        nonlocal last_turn_time

        active_turn = direction
        last_turn_direction = direction
        last_turn_time = now

        if direction == "LEFT":
            motors.rotate_left()
        else:
            motors.rotate_right()

        motion_phase = "AVOID_TURN"
        motion_until = now + AVOID_TURN_TIME

        set_state(f"LOCKED AVOID {direction}")

    try:
        motors.stop()
        lidar_reader.start()

        print("=" * 80)
        print("YELLOW TARGET EXPLORER V2")
        print("=" * 80)
        print("Wall-escape direction locking enabled")
        print(f"Target stop distance: {TARGET_STOP_DISTANCE:.2f} m")
        print("Press Ctrl+C to stop.")
        print("=" * 80)

        pipeline.start(config)
        time.sleep(2.0)

        set_state("EXPLORING")

        while True:
            now = time.monotonic()

            frames = pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                motors.stop()
                set_state("STOPPED - CAMERA FRAME LOST")
                continue

            color_image = np.asanyarray(color_frame.get_data())
            target = detect_yellow_square(color_image)

            (
                lidar_front,
                lidar_left,
                lidar_right,
                lidar_rear,
                lidar_update,
            ) = lidar_state.read()

            if now - lidar_update > LIDAR_TIMEOUT:
                motors.stop()
                set_state("STOPPED - LIDAR TIMEOUT")
                continue

            detected = target is not None
            detection_history.append(detected)

            if detected:
                lost_count = 0
            else:
                lost_count += 1

            if (
                not target_acquired
                and len(detection_history) == TARGET_CONFIRM_FRAMES
                and all(detection_history)
            ):
                target_acquired = True
                motion_phase = None
                motors.stop()
                set_state("TARGET ACQUIRED")

            if target_acquired and lost_count >= TARGET_LOST_FRAMES:
                target_acquired = False
                detection_history.clear()
                motors.stop()
                set_state("TARGET LOST - EXPLORING")

            # ------------------------------------------------
            # COMPLETE LOCKED MOTION PHASES
            # ------------------------------------------------

            if motion_phase == "AVOID_TURN":
                if now < motion_until:
                    time.sleep(0.03)
                    continue

                motors.forward(CAUTIOUS_SPEED)
                motion_phase = "ESCAPE_FORWARD"
                motion_until = now + ESCAPE_FORWARD_TIME

                set_state("ESCAPE FORWARD")
                continue

            if motion_phase == "ESCAPE_FORWARD":
                # Emergency interruption is still allowed.
                if lidar_front <= FRONT_EMERGENCY_DISTANCE:
                    motors.stop()
                    motion_phase = None
                elif now < motion_until:
                    time.sleep(0.03)
                    continue
                else:
                    motors.stop()
                    motion_phase = None
                    active_turn = None
                    set_state("ESCAPE COMPLETE")
                    time.sleep(0.08)

            if motion_phase == "REVERSE":
                if now < motion_until:
                    time.sleep(0.03)
                    continue

                direction = active_turn or last_turn_direction

                if direction == "LEFT":
                    motors.rotate_left()
                else:
                    motors.rotate_right()

                motion_phase = "RECOVERY_TURN"
                motion_until = now + RECOVERY_TURN_TIME

                set_state(f"RECOVERY TURN {direction}")
                continue

            if motion_phase == "RECOVERY_TURN":
                if now < motion_until:
                    time.sleep(0.03)
                    continue

                motors.forward(CAUTIOUS_SPEED)
                motion_phase = "ESCAPE_FORWARD"
                motion_until = now + ESCAPE_FORWARD_TIME

                set_state("RECOVERY FORWARD")
                continue

            if motion_phase == "SCAN":
                if now < motion_until:
                    time.sleep(0.03)
                    continue

                motors.stop()
                motion_phase = None
                last_scan_time = now
                time.sleep(0.08)

            if motion_phase == "ALIGN":
                if now < motion_until:
                    time.sleep(0.03)
                    continue

                motors.stop()
                motion_phase = None
                time.sleep(0.05)

            # ------------------------------------------------
            # TARGET TRACKING
            # ------------------------------------------------

            if target_acquired and target is not None:
                image_center_x = color_image.shape[1] // 2
                horizontal_error = target["center_x"] - image_center_x
                distance = target_depth(depth_frame, target)

                print(
                    "\r"
                    f"TARGET err:{horizontal_error:4d}px "
                    f"depth:{distance:4.2f}m "
                    f"LF:{lidar_front:4.2f} "
                    f"{state:30}",
                    end="",
                    flush=True,
                )

                if lidar_front <= FRONT_EMERGENCY_DISTANCE:
                    motors.stop()
                    set_state("STOPPED - OBSTACLE BEFORE TARGET")
                    time.sleep(0.10)
                    continue

                if math.isfinite(distance) and distance <= TARGET_STOP_DISTANCE:
                    motors.stop()
                    set_state("ARRIVED - STOPPED AT TARGET")
                    print("\nYellow target reached.")
                    break

                if horizontal_error < -CENTER_TOLERANCE_PIXELS:
                    motors.rotate_left(ALIGN_TURN_SPEED)
                    motion_phase = "ALIGN"
                    motion_until = now + 0.10
                    set_state("ALIGNING LEFT")
                    continue

                if horizontal_error > CENTER_TOLERANCE_PIXELS:
                    motors.rotate_right(ALIGN_TURN_SPEED)
                    motion_phase = "ALIGN"
                    motion_until = now + 0.10
                    set_state("ALIGNING RIGHT")
                    continue

                if not math.isfinite(distance):
                    motors.stop()
                    set_state("TARGET DEPTH UNKNOWN")
                    time.sleep(0.10)
                    continue

                if distance <= TARGET_SLOW_DISTANCE:
                    motors.forward(SLOW_APPROACH_SPEED)
                    set_state("SLOW APPROACH")
                else:
                    motors.forward(APPROACH_SPEED)
                    set_state("APPROACHING TARGET")

                time.sleep(0.06)
                continue

            # ------------------------------------------------
            # EXPLORATION
            # ------------------------------------------------

            print(
                "\r"
                f"EXPLORE F:{lidar_front:4.2f} "
                f"L:{lidar_left:4.2f} "
                f"R:{lidar_right:4.2f} "
                f"B:{lidar_rear:4.2f} "
                f"{state:30}",
                end="",
                flush=True,
            )

            left_too_close = lidar_left <= SIDE_EMERGENCY_DISTANCE
            right_too_close = lidar_right <= SIDE_EMERGENCY_DISTANCE

            # Boxed in or at a wall.
            if (
                lidar_front <= FRONT_EMERGENCY_DISTANCE
                or (left_too_close and right_too_close)
            ):
                motors.stop()

                direction = select_escape_direction(
                    lidar_left,
                    lidar_right,
                    now,
                )

                active_turn = direction
                last_turn_direction = direction
                last_turn_time = now

                if lidar_rear > REAR_CLEAR_DISTANCE:
                    motors.reverse()
                    motion_phase = "REVERSE"
                    motion_until = now + REVERSE_TIME
                    set_state("WALL RECOVERY - REVERSE")
                else:
                    begin_avoidance(direction, now)

                continue

            # Front obstacle: choose one direction and commit.
            if lidar_front < FRONT_TURN_DISTANCE:
                direction = select_escape_direction(
                    lidar_left,
                    lidar_right,
                    now,
                )

                begin_avoidance(direction, now)
                continue

            # Very close side wall: curve away without switching sides.
            if left_too_close:
                motors.curve_right(CAUTIOUS_SPEED)
                set_state("WALL FOLLOW - AWAY RIGHT")
                time.sleep(0.08)
                continue

            if right_too_close:
                motors.curve_left(CAUTIOUS_SPEED)
                set_state("WALL FOLLOW - AWAY LEFT")
                time.sleep(0.08)
                continue

            # Periodic small camera scan.
            if now - last_scan_time >= SCAN_INTERVAL_SECONDS:
                scan_direction = last_turn_direction

                if scan_direction == "LEFT":
                    motors.rotate_left(ALIGN_TURN_SPEED)
                else:
                    motors.rotate_right(ALIGN_TURN_SPEED)

                motion_phase = "SCAN"
                motion_until = now + SCAN_TURN_TIME

                set_state(f"CAMERA SCAN {scan_direction}")
                continue

            # Normal forward exploration with mild wall centering.
            side_difference = lidar_left - lidar_right

            if side_difference > SIDE_COMFORT_DISTANCE:
                motors.curve_left(EXPLORE_SPEED)
                set_state("EXPLORING - CURVE LEFT")

            elif side_difference < -SIDE_COMFORT_DISTANCE:
                motors.curve_right(EXPLORE_SPEED)
                set_state("EXPLORING - CURVE RIGHT")

            else:
                motors.forward(EXPLORE_SPEED)
                set_state("EXPLORING FORWARD")

            time.sleep(0.07)

    except KeyboardInterrupt:
        print("\nStopping rover...")

    except RuntimeError as error:
        print(f"\nRealSense error: {error}")

    except Exception as error:
        print(f"\nUnexpected error: {error}")

    finally:
        motors.stop()
        lidar_reader.stop_reader()

        try:
            pipeline.stop()
        except Exception:
            pass

        motors.cleanup()
        print("Motors stopped and sensors closed.")


if __name__ == "__main__":
    main()
