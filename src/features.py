"""Feature engineering. Deterministic, stateless, shared by training and serving."""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_INPUTS = [
    "amount", "hour", "account_age_days", "device_seen_before", "merchant_seen_before",
    "transactions_24h", "avg_amount_30d", "distance_from_home_km", "international",
]
RAW_FEATURES = RAW_INPUTS + ["merchant_risk", "velocity_score", "amount_deviation"]
ENGINEERED = [
    "log_amount", "log_avg_amount", "amount_ratio", "hour_sin", "hour_cos", "is_night",
    "log_distance", "log_tx24h", "log_age", "new_account",
]
FEATURE_COLUMNS = RAW_FEATURES + ENGINEERED
DEFAULT_MERCHANT_RISK = 0.20  # population mean of the generator's Beta(2, 8)
NEW_ACCOUNT_DAYS = 60


def derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add merchant_risk / velocity_score / amount_deviation when absent (API requests omit them)."""
    out = df.copy()
    if "merchant_risk" not in out:
        out["merchant_risk"] = DEFAULT_MERCHANT_RISK
    new_acct = (out["account_age_days"] < NEW_ACCOUNT_DAYS).astype(float)
    out["velocity_score"] = np.log1p(out["transactions_24h"]) / np.log1p(20.0) * (1.0 + 0.5 * new_acct)
    out["amount_deviation"] = (out["amount"] - out["avg_amount_30d"]) / out["avg_amount_30d"].clip(lower=1e-6)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model input matrix (FEATURE_COLUMNS). Raises ValueError on missing inputs."""
    missing = [c for c in RAW_INPUTS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    out = derive_columns(df)
    out["log_amount"] = np.log1p(out["amount"])
    out["log_avg_amount"] = np.log1p(out["avg_amount_30d"])
    out["amount_ratio"] = out["amount"] / (out["avg_amount_30d"] + 1.0)
    ang = 2 * np.pi * out["hour"] / 24.0
    out["hour_sin"], out["hour_cos"] = np.sin(ang), np.cos(ang)
    out["is_night"] = ((out["hour"] < 5) | (out["hour"] >= 23)).astype(float)
    out["log_distance"] = np.log1p(out["distance_from_home_km"])
    out["log_tx24h"] = np.log1p(out["transactions_24h"])
    out["log_age"] = np.log1p(out["account_age_days"])
    out["new_account"] = (out["account_age_days"] < NEW_ACCOUNT_DAYS).astype(float)
    return out[FEATURE_COLUMNS].astype(float)


def heuristic_risk_score(df: pd.DataFrame) -> np.ndarray:
    """Cheap rule-based risk in [0, 1] (no model call). Weighted sum of well-known red flags."""
    d = derive_columns(df)
    night = ((d["hour"] < 5) | (d["hour"] >= 23)).astype(float)
    score = (
        0.25 * (1 - d["device_seen_before"]) + 0.20 * d["international"] + 0.15 * night
        + 0.15 * (d["transactions_24h"] >= 6) + 0.15 * (d["amount_deviation"] > 3.0)
        + 0.10 * (1 - d["merchant_seen_before"])
    )
    return score.to_numpy(dtype=float)


RISK_FLAG_RULES = [
    ("unseen device", lambda r: r["device_seen_before"] == 0),
    ("unseen merchant", lambda r: r["merchant_seen_before"] == 0),
    ("international transaction", lambda r: r["international"] == 1),
    ("unusual hour (23:00-05:00)", lambda r: r["hour"] < 5 or r["hour"] >= 23),
    ("high transaction velocity (>=6 in 24h)", lambda r: r["transactions_24h"] >= 6),
    ("amount far above 30-day average (>3x)", lambda r: r["amount_deviation"] > 3.0),
    ("new account (<60 days)", lambda r: r["account_age_days"] < NEW_ACCOUNT_DAYS),
    ("far from home (>200 km)", lambda r: r["distance_from_home_km"] > 200),
]


def risk_flags(row: dict) -> list[str]:
    """Human-readable red flags for one transaction (deterministic)."""
    d = derive_columns(pd.DataFrame([row])).iloc[0].to_dict()
    return [name for name, rule in RISK_FLAG_RULES if rule(d)]
