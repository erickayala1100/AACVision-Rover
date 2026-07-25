#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
find "$REPO_ROOT/src" "$REPO_ROOT/experiments" -name '*.py' -print0 \
  | xargs -0 -n1 python3 -m py_compile
echo "All Python files passed syntax checking."
