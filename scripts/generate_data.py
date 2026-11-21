"""Generate the synthetic dataset and stratified splits (deterministic given configs/routing.yaml)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.config import data_dir, load_config
from src.datasets import marker_path
from src.data import file_fingerprint, generate_transactions, split_dataset


def main(n: int | None = None, quiet: bool = False) -> dict:
    cfg = load_config()
    d = cfg["data"]
    df = generate_transactions(n or d["n_samples"], cfg["seed"], d["target_fraud_rate"], d["label_noise"])
    raw, proc = data_dir() / "raw", data_dir() / "processed"
    raw.mkdir(parents=True, exist_ok=True); proc.mkdir(parents=True, exist_ok=True)
    marker_path().unlink(missing_ok=True)      # switching back to synthetic data
    df.to_csv(raw / "synthetic_fraud.csv", index=False)
    splits = split_dataset(df, cfg["seed"], d["splits"])
    for k, v in splits.items():
        v.to_csv(proc / f"{k}.csv", index=False)
    info = {"rows": len(df), "fraud_rate": float(df.fraud.mean()), "sha256_12": file_fingerprint(raw / "synthetic_fraud.csv"),
            **{f"{k}_rows": len(v) for k, v in splits.items()}, **{f"{k}_fraud_rate": float(v.fraud.mean()) for k, v in splits.items()}}
    if not quiet:
        print(info)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int); main(ap.parse_args().n)
