"""Model registry: model objects + cost/latency proxies + versions + metadata, persisted with joblib."""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, precision_recall_curve

from .calibration import ProbabilityCalibrator, choose_calibrator
from .config import TIERS, models_dir
from .datasets import SyntheticFeatures
from .models import raw_fraud_proba
from .uncertainty import Evidence


@dataclass
class TierModel:
    """One routable model tier. ``predict_proba`` returns the usual (n, 2) calibrated array."""
    name: str
    estimator: Any
    calibrator: ProbabilityCalibrator
    cost: float
    latency_ms: float
    version: str
    metadata: dict = field(default_factory=dict)

    def raw_proba(self, X) -> np.ndarray:
        return raw_fraud_proba(self.estimator, X)

    def proba(self, X) -> np.ndarray:
        """Calibrated P(fraud), 1-D."""
        return self.calibrator.transform(self.raw_proba(X))

    def predict_proba(self, X) -> np.ndarray:
        p = self.proba(X)
        return np.column_stack([1 - p, p])


class ModelRegistry:
    def __init__(self, models: dict[str, TierModel], fraud_threshold: float, meta: dict | None = None, feature_builder=None) -> None:
        missing = [t for t in TIERS if t not in models]
        if missing:
            raise ValueError(f"registry missing tiers: {missing}")
        self.models, self.fraud_threshold, self.meta = models, float(fraud_threshold), meta or {}
        self.feature_builder = feature_builder if feature_builder is not None else SyntheticFeatures()

    def compute_evidence(self, df: pd.DataFrame) -> Evidence:
        """Run all three tiers on raw transactions. (Cost accounting for this is handled by the policies.)"""
        X = self.feature_builder.transform(df)
        return Evidence(self.models["fast"].proba(X), self.models["accurate"].proba(X),
                        self.models["deep"].proba(X), self.feature_builder.heuristic_risk(df))

    def info(self) -> list[dict]:
        return [{"name": m.name, "version": m.version, "cost_units": m.cost, "latency_ms_proxy": m.latency_ms,
                 "estimator": type(m.estimator).__name__ if not hasattr(m.estimator, "steps") else
                 "+".join(type(s).__name__ for _, s in m.estimator.steps),
                 "calibration": m.calibrator.method, "metadata": m.metadata} for m in self.models.values()]

    def save(self, directory: Path | None = None) -> None:
        d = Path(directory or models_dir())
        d.mkdir(parents=True, exist_ok=True)
        for name, m in self.models.items():
            joblib.dump(m, d / f"{name}.joblib")
        joblib.dump(self.feature_builder, d / "features.joblib")
        meta = {**self.meta, "fraud_threshold": self.fraud_threshold, "model_versions": {n: m.version for n, m in self.models.items()}, "feature_columns": list(self.feature_builder.feature_names), "dataset": self.feature_builder.name,
                "sklearn": sklearn.__version__, "python": platform.python_version()}
        (d / "registry.json").write_text(json.dumps(meta, indent=2, default=str))

    @classmethod
    def load(cls, directory: Path | None = None) -> "ModelRegistry":
        d = Path(directory or models_dir())
        meta_path = d / "registry.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"no trained models in {d}; run `python scripts/train.py` first")
        meta = json.loads(meta_path.read_text())
        trained_with = meta.get("sklearn")
        try:
            models = {n: joblib.load(d / f"{n}.joblib") for n in TIERS}
        except Exception as exc:  # noqa: BLE001 - pickles are tied to the scikit-learn version that wrote them
            raise RuntimeError(
                f"could not load saved models ({type(exc).__name__}: {exc}). They were trained with scikit-learn {trained_with}; "
                f"this environment has {sklearn.__version__}. Retrain locally: `python scripts/train.py && python scripts/benchmark.py`") from exc
        fb = d / "features.joblib"
        builder = joblib.load(fb) if fb.exists() else None      # older registries are synthetic-only
        return cls(models, meta["fraud_threshold"], meta, builder)


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising F1 on the given (validation) data."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def build_registry(estimators: dict, X_val, y_val, cfg: dict, data_version: str, extra_meta: dict | None = None) -> tuple[ModelRegistry, dict]:
    """Calibrate on VALIDATION, pick the fraud threshold on VALIDATION, and wrap everything in a registry."""
    seed = cfg["seed"]
    y_val = np.asarray(y_val)
    models, cal_reports = {}, {}
    for name in TIERS:
        raw = raw_fraud_proba(estimators[name], X_val)
        cal, rep = choose_calibrator(raw, y_val, seed)
        cal_reports[name] = rep
        models[name] = TierModel(name, estimators[name], cal, float(cfg["cost"][name]), float(cfg["latency_ms"][name]),
                                 f"{name}-{data_version[:8]}-s{seed}", {"calibration": rep})
    thr = cfg["decision"]["fraud_threshold"]
    if thr is None:
        thr = best_f1_threshold(y_val, models["accurate"].proba(X_val))
    meta = {"data_version": data_version, "seed": seed, **(extra_meta or {})}
    return ModelRegistry(models, thr, meta), cal_reports
