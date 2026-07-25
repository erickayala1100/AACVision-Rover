#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${AACVISION_PYTHON:-/home/pi/aacvision-env/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN="$REPO_ROOT/src/aacvision_stepwise_basketball_rover_v8_3_smooth_step_approach.py"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$MAIN" ]]; then
  echo "Main V8.3 program not found: $MAIN" >&2
  exit 1
fi

exec "$PYTHON_BIN" -u "$MAIN"
