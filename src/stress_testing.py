"""Slice analysis and hand-built adversarial scenarios."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TIERS
from .evaluation import classification_metrics

SLICES = {
    "all": lambda d: np.ones(len(d), bool),
    "high_amount (>1000)": lambda d: d["amount"] > 1000,
    "new_device": lambda d: d["device_seen_before"] == 0,
    "new_merchant": lambda d: d["merchant_seen_before"] == 0,
    "international": lambda d: d["international"] == 1,
    "high_velocity (>=6 tx/24h)": lambda d: d["transactions_24h"] >= 6,
    "unusual_hour (<5)": lambda d: d["hour"] < 5,
    "low_account_age (<60d)": lambda d: d["account_age_days"] < 60,
}

SCENARIOS = {
    "high_amount_new_device": dict(amount=2500, hour=14, account_age_days=400, device_seen_before=0, merchant_seen_before=1,
                                   transactions_24h=2, avg_amount_30d=110, distance_from_home_km=20, international=0),
    "high_amount_new_account": dict(amount=1800, hour=15, account_age_days=12, device_seen_before=1, merchant_seen_before=1,
                                    transactions_24h=1, avg_amount_30d=90, distance_from_home_km=10, international=0),
    "international_unusual_hour": dict(amount=420, hour=3, account_age_days=700, device_seen_before=0, merchant_seen_before=0,
                                       transactions_24h=2, avg_amount_30d=120, distance_from_home_km=4200, international=1),
    "high_velocity_new_merchant": dict(amount=160, hour=19, account_age_days=300, device_seen_before=1, merchant_seen_before=0,
                                       transactions_24h=12, avg_amount_30d=100, distance_from_home_km=15, international=0),
    "benign_control": dict(amount=65, hour=13, account_age_days=900, device_seen_before=1, merchant_seen_before=1,
                           transactions_24h=1, avg_amount_30d=70, distance_from_home_km=8, international=0),
}


def slices_from_queries(queries: dict[str, str]) -> dict:
    """Turn {name: pandas-query-expression} (from a dataset spec) into slice callables, plus an 'all' slice."""
    return {"all": SLICES["all"], **{name: (lambda d, q=q: d.eval(q)) for name, q in queries.items()}}


def slice_report(policy: str, df: pd.DataFrame, frame: pd.DataFrame, thr: float, slices: dict | None = None) -> pd.DataFrame:
    rows = []
    y = df["fraud"].to_numpy()
    for name, fn in (slices or SLICES).items():
        m = np.asarray(fn(df), bool)
        n = int(m.sum())
        if n == 0:
            continue
        fr, yy = frame[m], y[m]
        pred = (fr["p_final"].to_numpy() >= thr).astype(int)
        pos = int(yy.sum())
        met = classification_metrics(yy, fr["p_final"].to_numpy(), thr) if pos >= 3 and pos < n else {}
        rows.append({"policy": policy, "slice": name, "n": n, "fraud_n": pos, "fraud_rate": float(yy.mean()),
                     **{f"{t}_pct": float((fr["final_model"] == t).mean() * 100) for t in TIERS},
                     "deep_escalation_pct": float((fr["final_model"] == "deep").mean() * 100),
                     "false_negative_rate": float(((pred == 0) & (yy == 1)).sum() / pos) if pos else np.nan,
                     "recall": met.get("recall", np.nan), "pr_auc": met.get("pr_auc", np.nan),
                     "avg_cost": float(fr["incurred_cost"].mean()), "safety_override_pct": float(fr["safety_override"].mean() * 100),
                     "review_pct": float((fr["decision"] == "REVIEW").mean() * 100)})
    return pd.DataFrame(rows)
