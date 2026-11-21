#!/bin/sh
set -e
if ! python -c "from src.registry import ModelRegistry; ModelRegistry.load()" >/dev/null 2>&1; then
  echo "[entrypoint] no loadable models (missing, or saved by a different scikit-learn) -> generating data and training"
  python scripts/train.py
  python scripts/benchmark.py > /dev/null
fi
exec "$@"
