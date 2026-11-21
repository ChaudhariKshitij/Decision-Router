"""Tune the FAST-stop threshold on the VALIDATION split and write a tuned config.

    DR_CONFIG=configs/creditcard.yaml python scripts/tune_thresholds.py --out configs/creditcard.tuned.yaml

Why: confidence = max(p, 1-p) depends on class prevalence, so thresholds that suit one dataset can send everything to DEEP on
another. Candidates are quantiles of FAST confidence on validation (so they adapt to the score scale), expressed as the share of
traffic that stops at FAST. For each routing policy we pick the CHEAPEST candidate whose validation PR-AUC is within 1% of the best
candidate, then set confidence.fast_threshold, confidence.accurate_threshold (= same value) and risk_aware.fast_confidence /
entropy_threshold (= entropy at that cut). Other thresholds (disagreement, risk, escalate_entropy, gate, abstention) are left as
configured: review them. Test data is never touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import copy
import os

import numpy as np
import pandas as pd
import yaml

from src.config import ROOT, artifacts_dir, data_dir, load_config
from src.evaluation import classification_metrics
from src.registry import ModelRegistry
from src.router import DecisionRouter
from src.uncertainty import binary_entropy

STOP_SHARES = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98]


def main(out: str | None = None) -> dict:
    cfg = load_config()
    reg = ModelRegistry.load()
    va = pd.read_csv(data_dir() / "processed" / "val.csv")
    router = DecisionRouter(reg, cfg)
    ev = router.evidence(va)
    y, thr = va["fraud"].to_numpy(), reg.fraud_threshold
    rows = []
    for share in STOP_SHARES:
        t = float(np.quantile(ev.fast_confidence, 1 - share))          # `share` of rows have confidence >= t
        for pol in ("confidence", "risk_aware"):
            c = copy.deepcopy(cfg)
            c["routing"]["confidence"].update(fast_threshold=t, accurate_threshold=t)
            c["routing"]["risk_aware"].update(fast_confidence=t, entropy_threshold=float(binary_entropy(np.array([1 - t]))[0]))
            fr = DecisionRouter(reg, c).route(va, pol, ev=ev).frame
            m = classification_metrics(y, fr["p_final"].to_numpy(), thr)
            rows.append({"policy": pol, "fast_stop_share": share, "fast_confidence_threshold": t, "val_pr_auc": m["pr_auc"], "val_recall": m["recall"],
                         "avg_cost": float(fr["incurred_cost"].mean()), "fast_pct": float((fr.final_model == "fast").mean() * 100),
                         "deep_pct": float((fr.final_model == "deep").mean() * 100)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(artifacts_dir() / "threshold_tuning.csv", index=False)
    tuned = copy.deepcopy(cfg)
    picks = {}
    for pol, g in tbl.groupby("policy"):
        best = g[g.val_pr_auc >= g.val_pr_auc.max() * 0.99].sort_values("avg_cost").iloc[0]
        picks[pol] = best.to_dict()
    t_conf, t_risk = picks["confidence"]["fast_confidence_threshold"], picks["risk_aware"]["fast_confidence_threshold"]
    tuned["routing"]["confidence"].update(fast_threshold=round(t_conf, 6), accurate_threshold=round(t_conf, 6))
    tuned["routing"]["risk_aware"].update(fast_confidence=round(t_risk, 6), entropy_threshold=round(float(binary_entropy(np.array([1 - t_risk]))[0]), 5))
    src = Path(os.getenv("DR_CONFIG", ROOT / "configs" / "routing.yaml"))
    dest = Path(out) if out else src.with_name(src.stem + ".tuned.yaml")
    dest.write_text("# Tuned on the validation split by scripts/tune_thresholds.py (test data untouched). Review the other thresholds.\n" + yaml.safe_dump(tuned, sort_keys=False))
    print(tbl.round(4).to_string(index=False))
    print({k: {kk: round(v, 4) for kk, v in p.items() if kk in ("fast_confidence_threshold", "val_pr_auc", "avg_cost", "deep_pct")} for k, p in picks.items()})
    print("wrote", dest, "-> use it with DR_CONFIG=" + str(dest))
    return {"picks": picks, "config": str(dest)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out")
    main(ap.parse_args().out)
