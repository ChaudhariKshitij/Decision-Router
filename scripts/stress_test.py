"""Slice analysis (several policies) + hand-built adversarial scenarios + data-driven conflicting-evidence cases."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd

from src.config import artifacts_dir, data_dir, load_config
from src.datasets import active_dataset, load_spec
from src.registry import ModelRegistry
from src.router import DecisionRouter
from src.stress_testing import SCENARIOS, slice_report, slices_from_queries


def main() -> None:
    cfg = load_config()
    reg = ModelRegistry.load()
    router = DecisionRouter(reg, cfg)
    te = pd.read_csv(data_dir() / "processed" / "test.csv")
    ev = router.evidence(te)
    thr = reg.fraud_threshold
    real = active_dataset()
    slices = slices_from_queries(load_spec(real["name"]).get("slices", {})) if real else None
    req = list(reg.feature_builder.required_columns)
    parts = []
    for pol in ("risk_aware", "confidence", "fast", "deep"):
        parts.append(slice_report(pol, te, router.route(te, pol, ev=ev).frame, thr, slices))
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(artifacts_dir() / "stress_test_results.csv", index=False)

    pop = {"mean_disagreement": float(ev.disagreement.mean()), "mean_entropy_fast": float(ev.entropy_fast.mean()),
           "mean_risk": float(ev.risk_score.mean())}
    scen = []
    # data-driven "conflicting model evidence": largest FAST-vs-DEEP gaps on the test set
    gap = np.abs(ev.p_fast - ev.p_deep)
    conflict = {f"conflict_case_{i + 1}": te.iloc[j][req].to_dict() for i, j in enumerate(np.argsort(-gap)[:3])}
    for name, txn in {**({} if real else SCENARIOS), **conflict}.items():
        txn = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in txn.items()}
        r = router.route_one(txn, "risk_aware", log=False)
        scen.append({"scenario": name, "transaction": txn, "probabilities": {t: r[f"{t}_probability"] for t in ("fast", "accurate", "deep")},
                     "disagreement": r["model_disagreement"], "entropy": r["entropy"], "risk_score": r["risk_score"],
                     "initial_model": r["initial_model"], "selected_model": r["selected_model"], "safety_override": r["safety_override"],
                     "decision": r["decision"], "estimated_cost": r["estimated_cost_units"], "models_run": r["models_run"]})
    adv = [s for s in scen if s["scenario"] != "benign_control"]
    ctrl = next((s for s in scen if s["scenario"] == "benign_control"), None)
    summary = {
        "population_means_test": pop, "scenarios": scen,
        "checks": {
            "mean_deep_selected_adversarial": float(np.mean([s["selected_model"] == "deep" for s in adv])),
            "benign_control_selected": ctrl["selected_model"] if ctrl else None,
            "adversarial_mean_disagreement_vs_population": float(np.mean([s["disagreement"] for s in adv if s["disagreement"] is not None])) - pop["mean_disagreement"],
            "safety_overrides_triggered": int(sum(s["safety_override"] is not None for s in adv)),
        },
        "note": "Disagreement/entropy are over the tiers that actually ran; FAST-only decisions have 0 observed disagreement by construction.",
    }
    (artifacts_dir() / "stress_test_results.json").write_text(json.dumps(summary, indent=2, default=float))
    print(res[res.policy == "risk_aware"].round(3).drop(columns=["policy"]).to_string(index=False))
    for s in scen:
        print(f"{s['scenario']:30s} P={s['probabilities']} dis={s['disagreement']} -> {s['initial_model']}->{s['selected_model']} override={s['safety_override']} {s['decision']}")
    print(summary["checks"])


if __name__ == "__main__":
    main()
