"""Dataset specs, feature builders and real-data preparation.

Two feature builders share one interface (``transform``, ``heuristic_risk``, ``risk_flags``, ``feature_names``,
``required_columns``): ``SyntheticFeatures`` (the original generator's fields) and ``TabularFeatures`` (any numeric/
categorical CSV described by a YAML spec in configs/datasets/). The builder is saved inside the model registry so
serving always applies exactly the preprocessing that was fitted on the TRAIN split.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import ROOT, data_dir
from .data import split_dataset
from .features import FEATURE_COLUMNS, RAW_INPUTS, build_features, heuristic_risk_score, risk_flags

log = logging.getLogger(__name__)
MIN_FRAUDS_PER_SPLIT = 30


class SyntheticFeatures:
    """Original synthetic-fraud features (see src/features.py). Stateless."""
    kind, name = "synthetic", "synthetic"
    feature_names = FEATURE_COLUMNS
    required_columns = RAW_INPUTS

    def fit(self, df: pd.DataFrame) -> "SyntheticFeatures":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return build_features(df)[FEATURE_COLUMNS]

    def heuristic_risk(self, df: pd.DataFrame) -> np.ndarray:
        return heuristic_risk_score(df)

    def risk_flags(self, row: dict) -> list[str]:
        return risk_flags(row)


class TabularFeatures:
    """Generic builder for a real tabular dataset. Everything data-dependent is fitted on TRAIN only.

    numeric      -> passed through (missing -> train median)
    log1p        -> extra log1p(max(x, 0)) column for skewed amounts
    hour_from_seconds -> sin/cos of hour-of-day from a seconds-since-start column (the raw column itself is NOT used as a
                    feature: a monotone clock extrapolates badly under a temporal split)
    categorical  -> frequency encoding learned on train (unseen category -> 0)
    """
    kind = "tabular"

    def __init__(self, spec: dict) -> None:
        self.spec, self.name = spec, spec["name"]
        self.numeric = list(spec.get("numeric", []))
        self.categorical = list(spec.get("categorical", []))
        self.log1p = list(spec.get("log1p", []))
        self.hour_col = spec.get("hour_from_seconds")
        self.medians: dict[str, float] = {}
        self.freq: dict[str, dict[str, float]] = {}
        self.feature_names = (self.numeric + [f"log1p_{c}" for c in self.log1p] + (["hour_sin", "hour_cos"] if self.hour_col else [])
                              + [f"freq_{c}" for c in self.categorical])

    @property
    def required_columns(self) -> list[str]:
        cols = self.numeric + self.categorical + ([self.hour_col] if self.hour_col else [])
        return list(dict.fromkeys(cols))

    def fit(self, df: pd.DataFrame) -> "TabularFeatures":
        self._check(df)
        self.medians = {c: float(pd.to_numeric(df[c], errors="coerce").median()) for c in self.numeric}
        self.freq = {c: df[c].astype(str).value_counts(normalize=True).to_dict() for c in self.categorical}
        return self

    def _check(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check(df)
        out = pd.DataFrame(index=df.index)
        for c in self.numeric:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.medians.get(c, 0.0))
        for c in self.log1p:
            out[f"log1p_{c}"] = np.log1p(out[c].clip(lower=0))
        if self.hour_col:
            ang = 2 * np.pi * ((pd.to_numeric(df[self.hour_col], errors="coerce").fillna(0) % 86400) / 3600.0) / 24.0
            out["hour_sin"], out["hour_cos"] = np.sin(ang), np.cos(ang)
        for c in self.categorical:
            out[f"freq_{c}"] = df[c].astype(str).map(self.freq.get(c, {})).fillna(0.0)
        return out[self.feature_names].astype(float)

    def heuristic_risk(self, df: pd.DataFrame) -> np.ndarray:
        """No domain rules exist for an arbitrary dataset, so the rule-based stage-1 gate is neutral (0)."""
        return np.zeros(len(df))

    def risk_flags(self, row: dict) -> list[str]:
        return []


def load_spec(name: str) -> dict:
    p = ROOT / "configs" / "datasets" / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no dataset spec {p}")
    spec = yaml.safe_load(p.read_text())
    spec.setdefault("name", name)
    return spec


def marker_path() -> Path:
    return data_dir() / "processed" / "dataset.json"


def active_dataset() -> dict | None:
    """The prepared real dataset (written by scripts/prepare_real_data.py), or None when using synthetic data."""
    p = marker_path()
    return json.loads(p.read_text()) if p.exists() else None


def prepare_real(spec: dict, csv_path: str | Path, seed: int, split: str | None = None) -> dict:
    """Validate a real CSV against ``spec`` and write data/processed/{train,val,test}.csv in the project's schema
    (``transaction_id``, raw feature columns, ``fraud``)."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    target = spec["target"]
    need = [c for c in TabularFeatures(spec).required_columns if c not in ("transaction_id",)] + [target]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns required by spec '{spec['name']}': {missing}. Found: {list(df.columns)[:12]}...")
    df = df.dropna(subset=[target]).copy()
    labels = pd.to_numeric(df[target], errors="coerce")
    if labels.isna().any() or not set(labels.unique()) <= {0, 1}:
        raise ValueError(f"target column '{target}' must be binary 0/1; saw {sorted(map(str, labels.unique()))[:6]}")
    mode = split or spec.get("split", "stratified")
    time_col = spec.get("time_column")
    if mode == "temporal":
        if time_col not in df.columns:
            raise ValueError("temporal split requires spec.time_column present in the CSV")
        df = df.sort_values(time_col, kind="stable")
    df["fraud"] = labels.loc[df.index].astype(int)
    if target != "fraud":
        df = df.drop(columns=[target])
    df.insert(0, "transaction_id", [f"R{i:07d}" for i in range(len(df))])
    fr = spec.get("split_fracs", {"train": 0.6, "val": 0.2, "test": 0.2})
    if mode == "temporal":
        n = len(df); a, b = int(n * fr["train"]), int(n * (fr["train"] + fr["val"]))
        splits = {"train": df.iloc[:a], "val": df.iloc[a:b], "test": df.iloc[b:]}
        splits = {k: v.reset_index(drop=True) for k, v in splits.items()}
    else:
        splits = split_dataset(df, seed, fr)
    warnings = [f"split '{k}' has only {int(v.fraud.sum())} frauds: PR-AUC/recall will be very noisy" for k, v in splits.items() if v.fraud.sum() < MIN_FRAUDS_PER_SPLIT]
    proc = data_dir() / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    for k, v in splits.items():
        v.to_csv(proc / f"{k}.csv", index=False)
    info = {"name": spec["name"], "kind": "real", "source_file": csv_path.name, "split_mode": mode, "rows": len(df),
            "fraud_rate": float(df.fraud.mean()), "sha256_12": hashlib.sha256(csv_path.read_bytes()).hexdigest()[:12],
            **{f"{k}_rows": len(v) for k, v in splits.items()}, **{f"{k}_frauds": int(v.fraud.sum()) for k, v in splits.items()},
            **{f"{k}_fraud_rate": float(v.fraud.mean()) for k, v in splits.items()}, "warnings": warnings}
    marker_path().write_text(json.dumps(info, indent=2))
    return info
