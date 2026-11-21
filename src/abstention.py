"""Selective prediction: FRAUD / LEGIT / REVIEW decisions and coverage-vs-error curves.

With ~6% prevalence ``max(p, 1-p)`` is high for almost everyone, so "confidence" for abstention is the
decision MARGIN |p - fraud_threshold| (distance from the decision boundary), combined with ensemble disagreement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def decide(p_final: np.ndarray, disagreement: np.ndarray, fraud_threshold: float, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (decision, review_reason). decision in {'FRAUD','LEGIT','REVIEW'}."""
    a = cfg["abstention"]
    p, d = np.asarray(p_final, float), np.asarray(disagreement, float)
    low_margin = np.abs(p - fraud_threshold) < a["min_margin"]
    high_dis = d >= a["review_disagreement"]
    review = low_margin | high_dis
    decision = np.where(review, "REVIEW", np.where(p >= fraud_threshold, "FRAUD", "LEGIT")).astype(object)
    reason = np.full(len(p), None, dtype=object)
    reason[low_margin] = "low_margin"
    reason[high_dis] = "high_disagreement"
    return decision, np.where(review, reason, None)


def coverage_error_curve(y_true: np.ndarray, p_final: np.ndarray, fraud_threshold: float, uncertainty: np.ndarray,
                         coverages: np.ndarray | None = None, seed: int = 0) -> pd.DataFrame:
    """Selective-risk curve: keep the ``coverage`` fraction with LOWEST uncertainty, report error among kept.

    Also reports a random-abstention baseline for the same coverages (expected error = overall error).
    """
    y, p, u = np.asarray(y_true), np.asarray(p_final), np.asarray(uncertainty)
    pred = (p >= fraud_threshold).astype(int)
    err = (pred != y).astype(float)
    order = np.argsort(u, kind="stable")
    cov = coverages if coverages is not None else np.round(np.arange(0.5, 1.0001, 0.05), 2)
    rows = []
    for c in cov:
        k = max(int(round(c * len(y))), 1)
        keep = order[:k]
        kept_y = y[keep]
        rows.append({"coverage": float(c), "n_kept": k, "error_rate": float(err[keep].mean()),
                     "random_baseline_error": float(err.mean()),
                     "fraud_recall_kept": float(((pred[keep] == 1) & (kept_y == 1)).sum() / max((kept_y == 1).sum(), 1))})
    return pd.DataFrame(rows)
