#!/usr/bin/env bash
set -u

PYTHON_BIN="${AACVISION_PYTHON:-/home/pi/aacvision-env/bin/python}"
LIDAR_BIN="${AACVISION_LIDAR_BIN:-/home/pi/rplidar_sdk/output/Linux/Release/ultra_simple}"
FAIL=0

check_file() {
  local label="$1"
  local path="$2"
  if [[ -e "$path" ]]; then
    echo "[OK]   $label: $path"
  else
    echo "[MISS] $label: $path"
    FAIL=1
  fi
}

check_file "Python environment" "$PYTHON_BIN"
check_file "RPLIDAR ultra_simple" "$LIDAR_BIN"
check_file "RPLIDAR serial device" "/dev/ttyUSB0"

if [[ -x "$PYTHON_BIN" ]]; then
  "$PYTHON_BIN" - <<'PY' || FAIL=1
import importlib
modules = [
    "cv2",
    "numpy",
    "PIL",
    "pyrealsense2",
    "RPi.GPIO",
    "ultralytics",
    "breezyslam.algorithms",
]
for name in modules:
    try:
        importlib.import_module(name)
        print(f"[OK]   Python import: {name}")
    except Exception as exc:
        print(f"[FAIL] Python import: {name}: {exc}")
        raise SystemExit(1)
PY
fi

if command -v vcgencmd >/dev/null 2>&1; then
  echo "[INFO] Power status: $(vcgencmd get_throttled)"
else
  echo "[INFO] vcgencmd is unavailable on this system"
fi

exit "$FAIL"
