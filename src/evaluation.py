"""Metrics, Pareto utilities, latency benchmarking and plotting helpers."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)

from .config import TIERS


def classification_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """Threshold-free (ROC-AUC, PR-AUC, Brier, log-loss) and thresholded metrics."""
    y, p = np.asarray(y).astype(int), np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "roc_auc": float(roc_auc_score(y, p)) if both else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if both else np.nan,
        "accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn), "threshold": float(thr),
    }


def policy_row(name: str, y: np.ndarray, frame: pd.DataFrame, thr: float, deep_cost: float | None = None) -> dict:
    """One benchmark row for a routed frame (see DecisionRouter.route)."""
    m = classification_metrics(y, frame["p_final"].to_numpy(), thr)
    fm = frame["final_model"]
    row = {"policy": name, **{k: m[k] for k in ("pr_auc", "roc_auc", "accuracy", "recall", "precision", "f1", "brier", "log_loss")},
           "avg_cost_incurred": float(frame["incurred_cost"].mean()), "avg_cost_selected_tier": float(frame["selected_cost"].mean()),
           "avg_latency_ms_proxy": float(frame["incurred_latency_ms"].mean()),
           "p95_latency_ms_proxy": float(np.percentile(frame["incurred_latency_ms"], 95)),
           **{f"{t}_pct": float((fm == t).mean() * 100) for t in TIERS},
           "escalation_rate_pct": float((fm != "fast").mean() * 100),
           "safety_override_pct": float(frame["safety_override"].mean() * 100),
           "review_rate_pct": float((frame["decision"] == "REVIEW").mean() * 100),
           "missed_fraud": m["fn"], "fraud_caught": m["tp"]}
    return row


def add_relative_columns(df: pd.DataFrame, base: str = "always_deep") -> pd.DataFrame:
    """Savings / latency reduction / quality retention relative to the Always-Deep baseline."""
    b = df.loc[df["policy"] == base].iloc[0]
    out = df.copy()
    out["cost_savings_vs_deep_pct"] = (1 - out["avg_cost_incurred"] / b["avg_cost_incurred"]) * 100
    out["latency_reduction_vs_deep_pct"] = (1 - out["avg_latency_ms_proxy"] / b["avg_latency_ms_proxy"]) * 100
    out["pr_auc_retention_vs_deep"] = out["pr_auc"] / b["pr_auc"]
    out["recall_retention_vs_deep"] = out["recall"] / b["recall"]
    return out


def pareto_mask(cost: np.ndarray, quality: np.ndarray) -> np.ndarray:
    """True for points not dominated (lower-or-equal cost AND higher-or-equal quality, one strictly better)."""
    c, q = np.asarray(cost, float), np.asarray(quality, float)
    keep = np.ones(len(c), bool)
    for i in range(len(c)):
        dom = (c <= c[i]) & (q >= q[i]) & ((c < c[i]) | (q > q[i]))
        keep[i] = not dom.any()
    return keep


def measure_latency(predict_fn, X_row, iterations: int = 300, warmup: int = 20) -> dict:
    """Wall-clock single-row latency in ms (p50/p95/p99, mean). Host-dependent: report as measured, not as a guarantee."""
    for _ in range(warmup):
        predict_fn(X_row)
    t = np.empty(iterations)
    for i in range(iterations):
        s = time.perf_counter()
        predict_fn(X_row)
        t[i] = (time.perf_counter() - s) * 1000
    return {"mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "p99_ms": float(np.percentile(t, 99))}
