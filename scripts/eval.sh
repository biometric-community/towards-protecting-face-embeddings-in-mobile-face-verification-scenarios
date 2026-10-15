#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PYTHON="${PYTHON:-/home/public/tbiom/.venv/bin/python}"
CFG="${1:-configs/default.yaml}"
"$PYTHON" -m polyprotect.eval --config "$CFG"
