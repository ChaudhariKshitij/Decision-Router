"""Reproducible synthetic fraud generator and stratified splits.

The label is Bernoulli(sigmoid(z + b)) where z contains additive effects, nonlinear terms,
interactions and an UNOBSERVED latent noise term, so even a perfect model cannot reach AUC 1.0.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .features import NEW_ACCOUNT_DAYS, derive_columns

SIGNAL_SCALE = 1.7   # steepens the logit so the task is learnable but still imperfect
DEFAULT_SHIFT = {"amount_scale": 1.0, "intl_multiplier": 1.0, "intl_fraud_weight": 1.0}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _simulate(n: int, seed: int, shift: dict | None) -> tuple[pd.DataFrame, np.ndarray]:
    """Draw covariates and the latent fraud logit z (without intercept)."""
    s = {**DEFAULT_SHIFT, **(shift or {})}
    rng = np.random.default_rng(seed)
    age = np.clip(np.round(rng.gamma(1.4, 420.0, n)), 1, 5000)
    avg_amt = rng.lognormal(4.6, 0.75, n) * s["amount_scale"]
    ratio = rng.lognormal(0.0, 0.45, n)
    burst = rng.random(n) < 0.04
    ratio[burst] *= rng.lognormal(1.3, 0.6, burst.sum())
    amount = np.round(avg_amt * ratio, 2)
    hour = (rng.normal(14, 4.5, n) % 24)
    odd = rng.random(n) < 0.12
    hour[odd] = rng.uniform(0, 24, odd.sum())
    hour = np.floor(hour).astype(int) % 24
    established = age >= NEW_ACCOUNT_DAYS
    device = (rng.random(n) < np.where(established, 0.94, 0.60)).astype(int)
    merchant = (rng.random(n) < 0.80).astype(int)
    tx24 = rng.poisson(1.5, n) + np.where(rng.random(n) < 0.03, rng.poisson(8, n), 0)
    intl = (rng.random(n) < 0.06 * s["intl_multiplier"]).astype(int)
    dist = rng.exponential(25.0, n)
    far = rng.random(n) < 0.08
    dist[far] = rng.lognormal(5.0, 0.9, far.sum())
    dist = np.where(intl == 1, dist + rng.lognormal(5.5, 0.8, n), dist)
    merchant_risk = rng.beta(2, 8, n)
    df = pd.DataFrame({
        "amount": amount, "hour": hour, "account_age_days": age.astype(int),
        "device_seen_before": device, "merchant_seen_before": merchant,
        "transactions_24h": tx24.astype(int), "avg_amount_30d": np.round(avg_amt, 2),
        "distance_from_home_km": np.round(dist, 1), "international": intl,
        "merchant_risk": np.round(merchant_risk, 4),
    })
    df = derive_columns(df)
    d_night = np.abs(hour - 3.0)
    d_night = np.minimum(d_night, 24 - d_night)
    night = np.exp(-((d_night / 3.0) ** 2))            # smooth "unusual hour" bump
    dev_dev = df["amount_deviation"].to_numpy()
    z = (
        1.3 * night
        + 1.4 * (1 - device)
        + 0.9 * (1 - merchant)
        + 1.3 * s["intl_fraud_weight"] * intl
        + 0.55 * np.maximum(tx24 - 3, 0)                # velocity hinge
        + 1.3 * np.log1p(np.maximum(dev_dev, 0.0))      # amount far above normal
        + 2.0 * (~established) * (amount > 300)         # new account x large amount
        + 1.8 * (1 - device) * night                    # new device at night
        + 1.6 * intl * (dist > 200)                     # international x far
        + 1.5 * (1 - device) * (1 - merchant) * (tx24 >= 4)
        + 2.2 * merchant_risk
        + 1.2 * np.minimum(dev_dev, 6.0) * night        # big deviation at night
        + rng.normal(0.0, 0.45, n)                      # unobserved latent factor => ambiguity
    )
    z = SIGNAL_SCALE * z
    return df, z


@lru_cache(maxsize=8)
def reference_intercept(target_rate: float) -> float:
    """Solve the intercept on a fixed reference sample so the BASELINE fraud rate hits target_rate."""
    _, z = _simulate(60000, 987654, None)
    lo, hi = -15.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate_transactions(n: int, seed: int, target_rate: float = 0.06, label_noise: float = 0.003,
                          shift: dict | None = None) -> pd.DataFrame:
    """Generate ``n`` transactions. ``shift`` perturbs covariates/mechanism (drift experiments) while the
    intercept stays at its baseline value, so shifts change the fraud rate naturally."""
    df, z = _simulate(n, seed, shift)
    rng = np.random.default_rng(seed + 1)
    p = _sigmoid(z + reference_intercept(target_rate))
    fraud = (rng.random(n) < p).astype(int)
    flip = rng.random(n) < label_noise
    df["fraud"] = np.where(flip, 1 - fraud, fraud)
    df.insert(0, "transaction_id", [f"T{seed:04d}{i:07d}" for i in range(n)])
    return df


def split_dataset(df: pd.DataFrame, seed: int, fracs: dict) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test split. Test rows are never used for fitting, calibration or tuning."""
    train, rest = train_test_split(df, train_size=fracs["train"], stratify=df["fraud"], random_state=seed)
    val_share = fracs["val"] / (fracs["val"] + fracs["test"])
    val, test = train_test_split(rest, train_size=val_share, stratify=rest["fraud"], random_state=seed)
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True), "test": test.reset_index(drop=True)}


def file_fingerprint(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
