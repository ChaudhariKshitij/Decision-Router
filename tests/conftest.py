import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit import AuditLogger
from src.config import load_config
from src.data import generate_transactions, split_dataset
from src.features import build_features
from src.jev_adapter import JevAdapter
from src.models import fit_estimators
from src.registry import build_registry
from src.router import DecisionRouter

TXN = dict(amount=450, hour=2, account_age_days=30, device_seen_before=0, merchant_seen_before=1, transactions_24h=8,
           avg_amount_30d=120, distance_from_home_km=350, international=1)
BENIGN = dict(amount=60, hour=13, account_age_days=900, device_seen_before=1, merchant_seen_before=1, transactions_24h=1,
              avg_amount_30d=70, distance_from_home_km=5, international=0)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def splits(cfg):
    df = generate_transactions(6000, cfg["seed"], cfg["data"]["target_fraud_rate"], cfg["data"]["label_noise"])
    return split_dataset(df, cfg["seed"], cfg["data"]["splits"])


@pytest.fixture(scope="session")
def registry(cfg, splits):
    tr, va = splits["train"], splits["val"]
    est = fit_estimators(build_features(tr), tr["fraud"], cfg["seed"], quick=True)
    reg, _ = build_registry(est, build_features(va), va["fraud"], cfg, "testdata")
    return reg


@pytest.fixture()
def router(cfg, registry, tmp_path):
    return DecisionRouter(registry, cfg, JevAdapter(cfg, enabled=False), AuditLogger(tmp_path / "audit.jsonl"))
