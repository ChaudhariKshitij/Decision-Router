"""Routing benchmark on the held-out TEST split + threshold sweep + abstention + disagreement analyses."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import abstention, plots
from src.config import TIERS, artifacts_dir, data_dir, load_config, reports_dir
from src.evaluation import add_relative_columns, classification_metrics, pareto_mask, policy_row
from src.experiments import record_experiment
from src.jev_adapter import JevAdapter
from src.registry import ModelRegistry
from src.router import DecisionRouter
from src.uncertainty import binary_entropy

BASELINES = [("always_fast", "fast", True), ("always_accurate", "accurate", True), ("always_deep", "deep", True),
             ("confidence_router", "confidence", True), ("risk_aware_router", "risk_aware", True),
             ("confidence_router_no_safety", "confidence", False), ("risk_aware_router_no_safety", "risk_aware", False)]


def run_policies(router: DecisionRouter, df: pd.DataFrame, ev, thr: float) -> tuple[pd.DataFrame, dict]:
    y = df["fraud"].to_numpy()
    rows, frames = [], {}
    for name, pol, safety in BASELINES:
        fr = router.route(df, pol, ev=ev, use_safety=safety).frame
        frames[name] = fr
        rows.append(policy_row(name, y, fr, thr))
    return pd.DataFrame(rows), frames


def sweep(router: DecisionRouter, df: pd.DataFrame, ev, thr: float, split: str) -> pd.DataFrame:
    """Sweep the FAST confidence threshold for both cascade routers (other thresholds fixed from config)."""
    base = router.cfg
    out = []
    y = df["fraud"].to_numpy()
    for t in base["experiments"]["threshold_sweep"]:
        for pol, key in (("confidence", ("routing", "confidence", "fast_threshold")), ("risk_aware", ("routing", "risk_aware", "fast_confidence"))):
            cfg = json.loads(json.dumps(base))
            cfg[key[0]][key[1]][key[2]] = t
            r = DecisionRouter(router.registry, cfg)
            fr = r.route(df, pol, ev=ev).frame
            row = policy_row(pol, y, fr, thr)
            out.append({"split": split, "policy": pol, "fast_confidence_threshold": t, "pr_auc": row["pr_auc"], "recall": row["recall"],
                        "avg_cost_incurred": row["avg_cost_incurred"], "avg_latency_ms_proxy": row["avg_latency_ms_proxy"],
                        "fast_pct": row["fast_pct"], "accurate_pct": row["accurate_pct"], "deep_pct": row["deep_pct"],
                        "safety_override_pct": row["safety_override_pct"]})
    return pd.DataFrame(out)


def main() -> None:
    cfg = load_config()
    reg = ModelRegistry.load()
    thr = reg.fraud_threshold
    jev = JevAdapter(cfg)
    router = DecisionRouter(reg, cfg, jev)
    proc = data_dir() / "processed"
    te, va = pd.read_csv(proc / "test.csv"), pd.read_csv(proc / "val.csv")
    ev_te, ev_va = router.evidence(te), router.evidence(va)
    y = te["fraud"].to_numpy()
    art, fig = artifacts_dir(), reports_dir() / "figures"

    table, frames = run_policies(router, te, ev_te, thr)
    # Jev router: only if enabled with a key; evaluated on a fixed random sample (API calls are slow/paid).
    jev_note = None
    if jev.available:
        n = min(cfg["experiments"]["jev_benchmark_sample"], len(te))
        idx = np.random.default_rng(cfg["seed"]).choice(len(te), n, replace=False)
        sub, evs = te.iloc[idx].reset_index(drop=True), ev_te.subset(idx)
        fr = router.route(sub, "jev", ev=evs).frame
        row = policy_row("jev_router", sub["fraud"].to_numpy(), fr, thr)
        row["n_rows"] = n
        row["jev_ok_pct"] = float((fr["policy_source"] == "jev").mean() * 100)
        loc = router.route(sub, "risk_aware", ev=evs).frame
        row["agreement_with_local_pct"] = float((fr["initial_model"] == loc["initial_model"]).mean() * 100)
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
        frames["jev_router"] = fr
    else:
        jev_note = f"jev_router NOT evaluated: JEV_ENABLED={jev.enabled}, api_key_configured={bool(jev._api_key)}"
        print(jev_note)

    table = add_relative_columns(table)
    table["n_rows"] = table["n_rows"].fillna(len(te)).astype(int) if "n_rows" in table else len(te)
    table["on_pareto"] = False
    main_mask = table["n_rows"] == len(te)
    table.loc[main_mask, "on_pareto"] = pareto_mask(table.loc[main_mask, "avg_cost_incurred"], table.loc[main_mask, "pr_auc"])
    table.to_csv(art / "routing_benchmark.csv", index=False)

    sw_te, sw_va = sweep(router, te, ev_te, thr, "test"), sweep(router, va, ev_va, thr, "val")
    pd.concat([sw_va, sw_te]).to_csv(art / "threshold_sweep.csv", index=False)
    plots.pareto_plot(table[main_mask], "avg_cost_incurred", "Average incurred inference cost (proxy units)", fig / "pareto_cost.png", sw_te)
    plots.pareto_plot(table[main_mask], "avg_latency_ms_proxy", "Average latency (proxy ms)", fig / "pareto_latency.png", sw_te.rename(columns={"avg_latency_ms_proxy": "avg_latency_ms_proxy"}))

    # ---- RQ2: does disagreement correlate with errors?  (full 3-model disagreement, ACCURATE tier errors)
    full_dis = ev_te.disagreement
    err_acc = ((ev_te.p_accurate >= thr).astype(int) != y).astype(int)
    err_fast = ((ev_te.p_fast >= thr).astype(int) != y).astype(int)
    q = pd.qcut(pd.Series(full_dis), 5, duplicates="drop")
    by_q = pd.DataFrame({"q": q.astype(str), "err_fast": err_fast, "err_accurate": err_acc, "fraud": y}).groupby("q", observed=True).mean().reset_index()
    dis_fa = np.abs(ev_te.p_fast - ev_te.p_accurate)
    dis_report = {
        "auc_full_disagreement_predicting_fast_error": float(roc_auc_score(err_fast, full_dis)),
        "auc_full_disagreement_predicting_accurate_error": float(roc_auc_score(err_acc, full_dis)),
        "auc_fast_accurate_gap_predicting_fast_error": float(roc_auc_score(err_fast, dis_fa)),
        "auc_fast_entropy_predicting_fast_error": float(roc_auc_score(err_fast, ev_te.entropy_fast)),
        "auc_fast_1_minus_conf_predicting_fast_error": float(roc_auc_score(err_fast, 1 - ev_te.fast_confidence)),
        "spearman_disagreement_vs_accurate_error": float(pd.Series(full_dis).corr(pd.Series(err_acc), method="spearman")),
        "error_rate_by_disagreement_quintile": by_q.to_dict(orient="records"),
        "note": "AUC 0.5 = no signal. Errors are w.r.t. the fixed F1-optimal threshold; fraud is rare so errors concentrate in positives.",
    }
    (art / "disagreement_analysis.json").write_text(json.dumps(dis_report, indent=2, default=float))

    # ---- RQ5: abstention / selective risk (uncertainty = |p - thr| margin for ACCURATE and combined with disagreement)
    ra = frames["risk_aware_router"]
    curve_margin = abstention.coverage_error_curve(y, ra["p_final"].to_numpy(), thr, -np.abs(ra["p_final"].to_numpy() - thr))
    curve_margin["uncertainty"] = "margin_to_threshold"
    curve_dis = abstention.coverage_error_curve(y, ra["p_final"].to_numpy(), thr, ra["model_disagreement"].to_numpy())
    curve_dis["uncertainty"] = "model_disagreement"
    curve_ent = abstention.coverage_error_curve(y, ra["p_final"].to_numpy(), thr, binary_entropy(ra["p_final"].to_numpy()))
    curve_ent["uncertainty"] = "entropy_of_final_p"
    pd.concat([curve_margin, curve_dis, curve_ent]).to_csv(art / "selective_risk.csv", index=False)
    kept = ra["decision"] != "REVIEW"
    pred = (ra["p_final"] >= thr).astype(int)
    hc_err = ((pred != y) & (ra["p_final"].between(0, 1)) & (np.abs(ra["p_final"] - thr) >= cfg["abstention"]["min_margin"]))
    abst = {"review_rate_pct": float((~kept).mean() * 100), "coverage_pct": float(kept.mean() * 100),
            "error_rate_all": float((pred != y).mean()), "error_rate_on_non_review": float((pred[kept] != y[kept]).mean()),
            "error_rate_on_review_rows": float((pred[~kept] != y[~kept]).mean()) if (~kept).any() else None,
            "fraud_rate_in_review": float(y[~kept].mean()) if (~kept).any() else None,
            "fraud_rate_in_non_review": float(y[kept].mean()),
            "fraud_missed_all": int(((pred == 0) & (y == 1)).sum()), "fraud_missed_non_review": int(((pred[kept] == 0) & (y[kept] == 1)).sum()),
            "fraud_total": int(y.sum()), "fraud_sent_to_review": int((y[~kept] == 1).sum())}
    (art / "abstention_summary.json").write_text(json.dumps(abst, indent=2))

    eid = record_experiment("benchmark", cfg, reg.meta, "all", {"table": table[["policy", "pr_auc", "avg_cost_incurred", "recall"]].to_dict(orient="records")},
                            {"jev_note": jev_note})
    show = ["policy", "pr_auc", "roc_auc", "recall", "precision", "brier", "avg_cost_incurred", "avg_latency_ms_proxy", "fast_pct", "accurate_pct", "deep_pct",
            "safety_override_pct", "review_rate_pct", "cost_savings_vs_deep_pct", "pr_auc_retention_vs_deep", "on_pareto"]
    print(table[show].round(3).to_string(index=False))
    print("\nsweep (val):\n", sw_va.round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in dis_report.items() if k.startswith(("auc", "spearman"))}, indent=1))
    print(json.dumps(abst, indent=1)); print("experiment", eid)


if __name__ == "__main__":
    main()
