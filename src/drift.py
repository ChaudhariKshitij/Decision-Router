"""Synthetic distribution-shift experiments: covariate shift and mechanism (concept) shift."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import expected_calibration_error
from .config import TIERS
from .data import generate_transactions
from .evaluation import classification_metrics

SCENARIOS = {
    "baseline_fresh_sample": {},
    "amount_scale_x1.3": {"amount_scale": 1.3},
    "intl_volume_x3": {"intl_multiplier": 3.0},
    "intl_fraud_mechanism_x2": {"intl_fraud_weight": 2.0},
    "combined": {"amount_scale": 1.3, "intl_multiplier": 3.0, "intl_fraud_weight": 2.0},
}


def run_drift(router, cfg: dict, n: int = 20000) -> pd.DataFrame:
    """Evaluate frozen models/calibrators/thresholds on shifted fresh samples (seed differs from train/val/test)."""
    thr, rows = router.registry.fraud_threshold, []
    for i, (name, shift) in enumerate(SCENARIOS.items()):
        df = generate_transactions(n, cfg["seed"] + 1000 + i, cfg["data"]["target_fraud_rate"], cfg["data"]["label_noise"], shift)
        ev = router.evidence(df)
        y = df["fraud"].to_numpy()
        row = {"scenario": name, "shift": str(shift), "n": n, "fraud_rate": float(y.mean()),
               "intl_fraud_rate": float(y[df.international == 1].mean()), "intl_share": float(df.international.mean()),
               "mean_disagreement": float(ev.disagreement.mean()), "p95_disagreement": float(np.percentile(ev.disagreement, 95)),
               "mean_entropy_fast": float(ev.entropy_fast.mean()), "mean_fast_confidence": float(ev.fast_confidence.mean())}
        for t, p in zip(TIERS, (ev.p_fast, ev.p_accurate, ev.p_deep)):
            m = classification_metrics(y, p, thr)
            row.update({f"{t}_pr_auc": m["pr_auc"], f"{t}_recall": m["recall"], f"{t}_brier": m["brier"], f"{t}_ece": expected_calibration_error(y, p)})
        for pol in ("risk_aware", "confidence"):
            fr = router.route(df, pol, ev=ev).frame
            m = classification_metrics(y, fr["p_final"].to_numpy(), thr)
            row.update({f"{pol}_pr_auc": m["pr_auc"], f"{pol}_recall": m["recall"], f"{pol}_avg_cost": float(fr["incurred_cost"].mean()),
                        f"{pol}_deep_pct": float((fr["final_model"] == "deep").mean() * 100),
                        f"{pol}_safety_override_pct": float(fr["safety_override"].mean() * 100)})
        rows.append(row)
    return pd.DataFrame(rows)


def run_temporal_windows(router, cfg: dict, test_df: pd.DataFrame, windows: int = 4, time_col: str | None = None) -> pd.DataFrame:
    """Real-data drift check: evaluate the frozen models on consecutive time windows of the (time-ordered) test split.

    Real data has no generating process to perturb, so we look at how disagreement, escalation and quality move over time.
    """
    df = test_df.sort_values(time_col, kind="stable").reset_index(drop=True) if time_col and time_col in test_df else test_df.reset_index(drop=True)
    thr, rows = router.registry.fraud_threshold, []
    for i, idx in enumerate(np.array_split(np.arange(len(df)), windows)):
        part = df.iloc[idx].reset_index(drop=True)
        ev = router.evidence(part)
        y = part["fraud"].to_numpy()
        row = {"scenario": f"test_window_{i + 1}_of_{windows}", "n": len(part), "frauds": int(y.sum()), "fraud_rate": float(y.mean()),
               "mean_disagreement": float(ev.disagreement.mean()), "p95_disagreement": float(np.percentile(ev.disagreement, 95)),
               "mean_entropy_fast": float(ev.entropy_fast.mean()), "mean_fast_confidence": float(ev.fast_confidence.mean())}
        for t, p in zip(TIERS, (ev.p_fast, ev.p_accurate, ev.p_deep)):
            m = classification_metrics(y, p, thr)
            row.update({f"{t}_pr_auc": m["pr_auc"], f"{t}_recall": m["recall"], f"{t}_brier": m["brier"], f"{t}_ece": expected_calibration_error(y, p)})
        for pol in ("risk_aware", "confidence"):
            fr = router.route(part, pol, ev=ev).frame
            m = classification_metrics(y, fr["p_final"].to_numpy(), thr)
            row.update({f"{pol}_pr_auc": m["pr_auc"], f"{pol}_recall": m["recall"], f"{pol}_avg_cost": float(fr["incurred_cost"].mean()),
                        f"{pol}_deep_pct": float((fr["final_model"] == "deep").mean() * 100),
                        f"{pol}_safety_override_pct": float(fr["safety_override"].mean() * 100)})
        rows.append(row)
    return pd.DataFrame(rows)
