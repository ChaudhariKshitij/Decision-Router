import numpy as np
import pandas as pd

from scripts.benchmark import run_policies
from src.evaluation import add_relative_columns, classification_metrics, pareto_mask


def test_pareto_mask():
    cost = np.array([1, 3, 8, 2.0]); q = np.array([0.5, 0.6, 0.55, 0.52])
    assert list(pareto_mask(cost, q)) == [True, True, False, True]


def test_classification_metrics_perfect_and_bad():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert m["roc_auc"] == 1.0 and m["recall"] == 1.0 and m["precision"] == 1.0
    assert classification_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]), 0.5)["recall"] == 0.0


def test_benchmark_output(router, splits, registry):
    te = splits["test"]
    ev = router.evidence(te)
    table, frames = run_policies(router, te, ev, registry.fraud_threshold)
    table = add_relative_columns(table)
    assert {"always_fast", "always_accurate", "always_deep", "confidence_router", "risk_aware_router"} <= set(table.policy)
    row = table.set_index("policy")
    assert row.loc["always_deep", "cost_savings_vs_deep_pct"] == 0
    assert row.loc["always_fast", "avg_cost_incurred"] < row.loc["always_deep", "avg_cost_incurred"]
    assert row.loc["risk_aware_router", "avg_cost_incurred"] < row.loc["always_deep", "avg_cost_incurred"]
    for c in ("pr_auc", "roc_auc", "recall", "precision", "f1", "brier", "fast_pct", "accurate_pct", "deep_pct", "safety_override_pct"):
        assert c in table
    assert np.allclose(table[["fast_pct", "accurate_pct", "deep_pct"]].sum(axis=1), 100)
