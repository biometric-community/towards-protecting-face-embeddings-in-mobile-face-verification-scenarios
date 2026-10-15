#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
bash scripts/setup_env.sh
bash scripts/train.sh configs/smoke.yaml
bash scripts/eval.sh configs/smoke.yaml
bash scripts/report.sh configs/smoke.yaml
echo "Smoke done"
