#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 scripts/benchmark_suite.py --device cpu "$@"
python3 scripts/benchmark_suite.py --device gpu "$@"
