"""Distribution-shift experiment (frozen models, shifted fresh samples)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from src.config import artifacts_dir, load_config
import pandas as pd

from src.config import data_dir
from src.datasets import active_dataset, load_spec
from src.drift import run_drift, run_temporal_windows
from src.registry import ModelRegistry
from src.router import DecisionRouter


def main(n: int = 20000) -> None:
    cfg = load_config()
    router = DecisionRouter(ModelRegistry.load(), cfg)
    real = active_dataset()
    if real:      # no generator to perturb: look at consecutive time windows of the test split
        te = pd.read_csv(data_dir() / "processed" / "test.csv")
        df = run_temporal_windows(router, cfg, te, 4, load_spec(real["name"]).get("time_column"))
    else:
        df = run_drift(router, cfg, n)
    df.to_csv(artifacts_dir() / "drift_results.csv", index=False)
    base = df.iloc[0]
    rel = {r.scenario: {"d_mean_disagreement": float(r.mean_disagreement - base.mean_disagreement),
                        "d_fast_pr_auc": float(r.fast_pr_auc - base.fast_pr_auc), "d_accurate_pr_auc": float(r.accurate_pr_auc - base.accurate_pr_auc),
                        "d_deep_pr_auc": float(r.deep_pr_auc - base.deep_pr_auc),
                        "d_risk_aware_deep_pct": float(r.risk_aware_deep_pct - base.risk_aware_deep_pct),
                        "d_risk_aware_cost": float(r.risk_aware_avg_cost - base.risk_aware_avg_cost)} for r in df.itertuples()}
    (artifacts_dir() / "drift_results.json").write_text(json.dumps(rel, indent=2))
    cols = ["scenario", "fraud_rate"] + (["intl_fraud_rate"] if "intl_fraud_rate" in df else []) + ["fast_pr_auc", "accurate_pr_auc", "deep_pr_auc", "mean_disagreement", "risk_aware_pr_auc",
            "risk_aware_avg_cost", "risk_aware_deep_pct", "risk_aware_safety_override_pct"]
    print(df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
