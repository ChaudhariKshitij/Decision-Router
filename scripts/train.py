"""Train the three tiers on TRAIN, calibrate + choose threshold on VALIDATION, evaluate once on TEST."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from scripts.generate_data import main as generate
from src import plots
from src.calibration import calibration_table, expected_calibration_error
from src.config import TIERS, artifacts_dir, data_dir, load_config, models_dir, reports_dir
from src.evaluation import classification_metrics, measure_latency
from src.experiments import record_experiment
from src.datasets import SyntheticFeatures, TabularFeatures, active_dataset, load_spec
from src.models import fit_estimators
from src.registry import build_registry


def main(quick: bool = False, n: int | None = None) -> None:
    cfg = load_config()
    real = active_dataset()          # set by scripts/prepare_real_data.py; None => synthetic generator
    info = real if real else generate(n, quiet=True)
    proc = data_dir() / "processed"
    tr, va, te = (pd.read_csv(proc / f"{k}.csv") for k in ("train", "val", "test"))
    builder = TabularFeatures(load_spec(real["name"])).fit(tr) if real else SyntheticFeatures()   # fitted on TRAIN only
    FEATURE_COLUMNS = list(builder.feature_names)
    Xtr, Xva, Xte = (builder.transform(d) for d in (tr, va, te))
    t0 = time.time()
    est = fit_estimators(Xtr, tr["fraud"], cfg["seed"], quick)
    train_s = time.time() - t0
    reg, cal_reports = build_registry(est, Xva, va["fraud"], cfg, info["sha256_12"], {"train_seconds": train_s, "rows": info})
    reg.feature_builder = builder
    reg.save()
    thr = reg.fraud_threshold
    art, rep = artifacts_dir(), reports_dir()

    metrics, rows, cal_rows, preds, cms, ctabs = {}, [], [], {"transaction_id": te["transaction_id"], "fraud": te["fraud"]}, {}, {}
    probs = {}
    for name in TIERS:
        m = reg.models[name]
        raw_p, p = m.raw_proba(Xte), m.proba(Xte)
        probs[name], preds[f"p_{name}"], preds[f"p_{name}_raw"] = p, p, raw_p
        res = classification_metrics(te["fraud"], p, thr)
        res["ece_calibrated"] = expected_calibration_error(te["fraud"], p)
        res["ece_raw"] = expected_calibration_error(te["fraud"], raw_p)
        res["brier_raw"] = classification_metrics(te["fraud"], raw_p, thr)["brier"]
        res["calibration_chosen"] = m.calibrator.method
        res["calibration_cv_brier_on_val"] = cal_reports[name]["cv_brier"]
        res["model_size_kb"] = round(Path(models_dir() / f"{name}.joblib").stat().st_size / 1024, 1)
        res["latency_measured"] = measure_latency(lambda X, m=m: m.proba(X), Xte.iloc[[0]], cfg["experiments"]["latency_iterations"])
        res["cost_proxy"], res["latency_proxy_ms"], res["version"] = m.cost, m.latency_ms, m.version
        metrics[name] = res
        cms[name] = (res["tn"], res["fp"], res["fn"], res["tp"])
        for kind, pp in (("calibrated", p), ("raw", raw_p)):
            t = calibration_table(te["fraud"], pp); t.insert(0, "model", name); t.insert(1, "probabilities", kind)
            cal_rows.append(t)
        ctabs[name] = calibration_table(te["fraud"], p)
        rows.append({"model": name, **{k: res[k] for k in ("pr_auc", "roc_auc", "recall", "precision", "f1", "accuracy", "brier", "log_loss", "ece_calibrated", "ece_raw")},
                     "latency_p50_ms": res["latency_measured"]["p50_ms"], "latency_p95_ms": res["latency_measured"]["p95_ms"],
                     "latency_p99_ms": res["latency_measured"]["p99_ms"], "size_kb": res["model_size_kb"],
                     "cost_proxy": m.cost, "calibration": m.calibrator.method})
    metrics["_meta"] = {"fraud_threshold": thr, "threshold_source": "F1-optimal on VALIDATION for ACCURATE (calibrated)",
                        "test_rows": len(te), "test_fraud_rate": float(te.fraud.mean()), "dataset": info, "seed": cfg["seed"],
                        "train_seconds": train_s, "quick": quick,
                        "notes": "Latency is measured single-row wall-clock on this host; cost/latency proxies are config values."}
    (art / "model_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    pd.DataFrame(rows).to_csv(art / "model_comparison.csv", index=False)
    pd.concat(cal_rows).to_csv(art / "calibration.csv", index=False)
    pd.DataFrame(preds).to_csv(art / "test_predictions.csv", index=False)

    # permutation importance (ACCURATE, validation, PR-AUC) -> explainability artifact
    sub = Xva.sample(min(3000, len(Xva)), random_state=cfg["seed"])
    pi = permutation_importance(est["accurate"], sub, va.loc[sub.index, "fraud"], scoring="average_precision",
                                n_repeats=3, random_state=cfg["seed"], n_jobs=1)
    imp = sorted(({"feature": f, "importance": float(m), "std": float(s)} for f, m, s in zip(FEATURE_COLUMNS, pi.importances_mean, pi.importances_std)),
                 key=lambda r: -r["importance"])
    lr = est["fast"]
    coefs = dict(zip(FEATURE_COLUMNS, map(float, lr.named_steps["clf"].coef_[0])))
    (art / "feature_importance.json").write_text(json.dumps({"permutation_importance_accurate_val_pr_auc": imp, "fast_logistic_coefficients_standardised": coefs}, indent=2))

    fig = rep / "figures"
    plots.curves(te["fraud"], probs, fig / "models")
    plots.calibration_plot(ctabs, fig / "calibration.png")
    plots.confusion_plot(cms, fig / "confusion_matrices.png")
    eid = record_experiment("train", cfg, reg.meta | {"model_versions": {n: reg.models[n].version for n in TIERS}}, None,
                            {n: {k: metrics[n][k] for k in ("pr_auc", "roc_auc", "recall", "precision", "brier")} for n in TIERS})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"fraud_threshold={thr:.4f}  train_seconds={train_s:.1f}  experiment={eid}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--n", type=int)
    a = ap.parse_args(); main(a.quick, a.n)
