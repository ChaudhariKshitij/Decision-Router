"""Real-data path, exercised on a SMALL STAND-IN CSV that has the ULB credit-card layout (Time, V1-V28, Amount, Class).

The stand-in is random test fixture data used only to prove that loading, splitting, training, benchmarking, serving and
reporting work for a non-synthetic schema. It says nothing about real fraud; real results need the real download.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def make_fixture_csv(path: Path, n: int = 12000, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, n)).round(0))
    df["Amount"] = rng.lognormal(3.5, 1.2, n).round(2)
    z = -4.5 - 1.1 * df.V14 - 0.9 * df.V3 + 0.7 * df.V10 * (df.V12 < 0) + rng.normal(0, 0.8, n)
    df["Class"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def real_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("real")
    csv = make_fixture_csv(base / "creditcard.csv")
    saved = {k: os.environ.get(k) for k in ("DR_CONFIG", "DR_ARTIFACTS_DIR", "DR_DATA_DIR")}
    os.environ.update(DR_CONFIG=str(ROOT / "configs" / "creditcard.yaml"), DR_ARTIFACTS_DIR=str(base / "art"), DR_DATA_DIR=str(base / "data"))
    try:
        yield csv, base
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_prepare_rejects_bad_files(real_env, tmp_path):
    from src.datasets import load_spec, prepare_real
    csv, _ = real_env
    df = pd.read_csv(csv)
    bad = tmp_path / "missing.csv"; df.drop(columns=["V5"]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="V5"):
        prepare_real(load_spec("creditcard"), bad, 42)
    nb = tmp_path / "nonbinary.csv"; df.assign(Class=df.Class * 2).to_csv(nb, index=False)
    with pytest.raises(ValueError, match="binary"):
        prepare_real(load_spec("creditcard"), nb, 42)


def test_full_pipeline_on_real_schema(real_env):
    from scripts import benchmark, drift_test, make_report, prepare_real_data, stress_test, train
    from src.config import artifacts_dir, data_dir

    csv, _ = real_env
    readme_before = (ROOT / "README.md").read_text()
    info = prepare_real_data.main("creditcard", str(csv))
    tr, te = pd.read_csv(data_dir() / "processed" / "train.csv"), pd.read_csv(data_dir() / "processed" / "test.csv")
    assert info["split_mode"] == "temporal" and tr.Time.max() <= te.Time.min()      # no look-ahead
    assert {"transaction_id", "fraud"} <= set(tr.columns) and "Class" not in tr.columns
    train.main(quick=True)
    benchmark.main(); stress_test.main(); drift_test.main(); make_report.main()
    a = artifacts_dir()
    for f in ("model_metrics.json", "routing_benchmark.csv", "stress_test_results.csv", "drift_results.csv", "reports/final_report.md"):
        assert (a / f).exists(), f
    rep = (a / "reports" / "final_report.md").read_text()
    assert "Real dataset `creditcard`" in rep and "Time windows of the test split" in rep
    bench = pd.read_csv(a / "routing_benchmark.csv").set_index("policy")
    assert bench.loc["always_fast", "avg_cost_incurred"] == 1 and bench.loc["always_deep", "avg_cost_incurred"] == 8
    assert (ROOT / "README.md").read_text() == readme_before        # a real-data/scratch run must never rewrite the README


def test_registry_and_api_serve_real_features(real_env):
    from app.api import create_app
    from src.config import data_dir
    from src.registry import ModelRegistry
    from src.router import DecisionRouter
    from src.config import load_config

    reg = ModelRegistry.load()
    assert reg.feature_builder.kind == "tabular" and "log1p_Amount" in reg.feature_builder.feature_names
    row = pd.read_csv(data_dir() / "processed" / "test.csv").iloc[0][reg.feature_builder.required_columns].astype(float).to_dict()
    rec = DecisionRouter(reg, load_config()).route_one(row, "risk_aware", log=False)
    assert rec["decision"] in {"FRAUD", "LEGIT", "REVIEW"}
    with TestClient(create_app(DecisionRouter(reg, load_config()))) as c:
        ok = c.post("/route/raw", json={"features": row})
        assert ok.status_code == 200 and ok.json()["selected_model"] in {"fast", "accurate", "deep"}
        assert c.get("/model-info").json()["dataset"] == "creditcard"
        missing = {k: v for k, v in row.items() if k != "V1"}
        assert c.post("/route/raw", json={"features": missing}).status_code == 422
        b = c.post("/route/raw/batch", json={"rows": [row, row]})
        assert b.status_code == 200 and b.json()["total_transactions"] == 2
