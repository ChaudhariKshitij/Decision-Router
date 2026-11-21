"""Prepare a REAL dataset for the pipeline.

    python scripts/prepare_real_data.py --dataset creditcard --csv /path/to/creditcard.csv [--split temporal|stratified]

Writes data/processed/{train,val,test}.csv + dataset.json (all under DR_DATA_DIR if set). Afterwards run
train.py -> benchmark.py -> stress_test.py -> drift_test.py -> make_report.py with DR_CONFIG pointing at the dataset's config
(e.g. configs/creditcard.yaml) and, to keep your synthetic results intact, DR_ARTIFACTS_DIR / DR_DATA_DIR set to new folders.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from src.config import load_config
from src.datasets import load_spec, prepare_real


def main(dataset: str, csv: str, split: str | None = None) -> dict:
    cfg = load_config()
    info = prepare_real(load_spec(dataset), csv, cfg["seed"], split)
    print(json.dumps(info, indent=2))
    for w in info["warnings"]:
        print("WARNING:", w)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="creditcard")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", choices=["temporal", "stratified"])
    a = ap.parse_args()
    main(a.dataset, a.csv, a.split)
