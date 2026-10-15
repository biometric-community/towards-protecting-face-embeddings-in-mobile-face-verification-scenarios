#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PYTHON="${PYTHON:-/home/public/tbiom/.venv/bin/python}"
CFG="${1:-configs/default.yaml}"
IMG="${2:?image path required}"
"$PYTHON" -m polyprotect.predict --config "$CFG" --image "$IMG" --protect
