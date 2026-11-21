"""Assemble artifacts/reports/final_report.md (+ README results block) strictly from generated artifacts."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import ROOT, artifacts_dir, data_dir, load_config, reports_dir
from src.datasets import active_dataset
from src.registry import ModelRegistry
from src.router import DecisionRouter

B = 400


def md(df: pd.DataFrame, floats: int = 3) -> str:
    d = df.copy()
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "|".join("---" for _ in d.columns) + "|\n"
    rows = ["| " + " | ".join(f"{v:.{floats}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return head + "\n".join(rows) + "\n"


def paired_bootstrap(y, pa, pb, thr, rng) -> dict:
    """95% CI of PR-AUC(a)-PR-AUC(b) and recall(a)-recall(b), resampling test rows jointly."""
    n, d_pr, d_rc = len(y), [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        yy = y[i]
        if yy.sum() == 0:
            continue
        d_pr.append(average_precision_score(yy, pa[i]) - average_precision_score(yy, pb[i]))
        d_rc.append((((pa[i] >= thr) & (yy == 1)).sum() - ((pb[i] >= thr) & (yy == 1)).sum()) / yy.sum())
    f = lambda v: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {"d_pr_auc": f(d_pr), "d_recall": f(d_rc)}


def verdict(ci) -> str:
    return "difference not distinguishable from zero (95% CI spans 0)" if ci[1] <= 0 <= ci[2] else ("significantly higher" if ci[1] > 0 else "significantly lower")


def main() -> None:
    cfg = load_config(); art = artifacts_dir()
    reg = ModelRegistry.load(); router = DecisionRouter(reg, cfg); thr = reg.fraud_threshold
    te = pd.read_csv(data_dir() / "processed" / "test.csv"); y = te.fraud.to_numpy()
    ev = router.evidence(te)
    bench = pd.read_csv(art / "routing_benchmark.csv")
    main_b = bench[bench.n_rows == bench.n_rows.max()].set_index("policy")
    comp = pd.read_csv(art / "model_comparison.csv")
    mm = json.loads((art / "model_metrics.json").read_text())
    sweep = pd.read_csv(art / "threshold_sweep.csv")
    dis = json.loads((art / "disagreement_analysis.json").read_text())
    abst = json.loads((art / "abstention_summary.json").read_text())
    stress = pd.read_csv(art / "stress_test_results.csv")
    drift = pd.read_csv(art / "drift_results.csv")
    sj = json.loads((art / "stress_test_results.json").read_text())

    # paired bootstrap CIs on the test split
    rng = np.random.default_rng(cfg["seed"])
    P = {n: router.route(te, pol, ev=ev).frame["p_final"].to_numpy() for n, pol in
         (("always_fast", "fast"), ("always_accurate", "accurate"), ("always_deep", "deep"), ("confidence_router", "confidence"), ("risk_aware_router", "risk_aware"))}
    pairs = [("risk_aware_router", "always_deep"), ("risk_aware_router", "always_accurate"), ("risk_aware_router", "always_fast"),
             ("confidence_router", "always_fast"), ("confidence_router", "always_deep"), ("always_accurate", "always_deep")]
    boot = {f"{a} - {b}": paired_bootstrap(y, P[a], P[b], thr, rng) for a, b in pairs}
    (art / "bootstrap_ci.json").write_text(json.dumps({"B": B, "pairs": boot}, indent=2))
    boot_tbl = pd.DataFrame([{"comparison": k, "ΔPR-AUC": v["d_pr_auc"][0], "95% CI": f"[{v['d_pr_auc'][1]:+.3f}, {v['d_pr_auc'][2]:+.3f}]",
                              "Δrecall": v["d_recall"][0], "recall 95% CI": f"[{v['d_recall'][1]:+.3f}, {v['d_recall'][2]:+.3f}]"} for k, v in boot.items()])

    # threshold selection rule (validation only)
    sv = sweep[sweep.split == "val"]
    picks = {}
    for pol, g in sv.groupby("policy"):
        ok = g[g.pr_auc >= g.pr_auc.max() * 0.99].sort_values("avg_cost_incurred").iloc[0]
        picks[pol] = (float(ok.fast_confidence_threshold), float(ok.avg_cost_incurred), float(ok.pr_auc))

    r = main_b.loc["risk_aware_router"]; c = main_b.loc["confidence_router"]; d = main_b.loc["always_deep"]; a = main_b.loc["always_accurate"]; f = main_b.loc["always_fast"]
    rns = main_b.loc["risk_aware_router_no_safety"]; cns = main_b.loc["confidence_router_no_safety"]
    pareto = ", ".join(main_b.index[main_b.on_pareto.astype(bool)])
    base, comb = drift.iloc[0], drift.iloc[-1]
    ci_ra_deep, ci_ra_acc, ci_c_fast = boot["risk_aware_router - always_deep"]["d_pr_auc"], boot["risk_aware_router - always_accurate"]["d_pr_auc"], boot["confidence_router - always_fast"]["d_pr_auc"]
    sl = stress[stress.policy == "risk_aware"].set_index("slice")
    real = active_dataset()
    ranked = sl.drop(index="all", errors="ignore").sort_values("deep_escalation_pct", ascending=False)
    s1n, s2n = ranked.index[0], ranked.index[min(1, len(ranked) - 1)]
    hv, intl = ranked.loc[s1n], ranked.loc[s2n]       # the two slices that escalate most
    chk = sj["checks"]

    tier_deltas = [float(drift[f"{t}_pr_auc"].iloc[1:].min() - base[f"{t}_pr_auc"]) for t in ("fast", "accurate", "deep")]
    qual_sentence = ("Absolute PR-AUC did not drop below the fresh baseline in any scenario (worst tier delta " + f"{min(tier_deltas):+.3f}" + "; higher fraud prevalence raises PR-AUC), so there was no quality degradation for routing to compensate for; the shifts mainly change cost."
                     if min(tier_deltas) >= -0.01 else
                     "Some tiers lost PR-AUC under shift (worst tier delta " + f"{min(tier_deltas):+.3f}" + "); see the drift table.")
    deep_best = d.pr_auc >= max(a.pr_auc, f.pr_auc)
    rq1_lead = ("Against Always-Deep: yes." if (r.cost_savings_vs_deep_pct > 0 and ci_ra_deep[2] >= 0 and ci_ra_deep[1] > -0.05) else
                "Against Always-Deep: not on this data, the router loses PR-AUC.") + f" ({'synthetic task' if not real else 'real dataset ' + real['name']})"
    rq1_caveat = (f"**Caveat:** Always-Accurate ({a.avg_cost_incurred:.0f} units) has PR-AUC {a.pr_auc:.3f} vs {d.pr_auc:.3f} for Always-Deep, i.e. the most expensive tier is *not* the best here, so part of the saving comes from simply not using DEEP."
                  if a.pr_auc > d.pr_auc else
                  f"Always-Deep has the highest or equal PR-AUC among the fixed tiers ({d.pr_auc:.3f}); the router's value is the cost reduction relative to it.")
    auc_f, auc_e = dis['auc_fast_accurate_gap_predicting_fast_error'], dis['auc_fast_entropy_predicting_fast_error']
    rq2_lead = "It carries signal" if dis['auc_full_disagreement_predicting_fast_error'] > 0.6 else "It carries little signal"
    rq2_cmp = ("about as well" if abs(dis['auc_full_disagreement_predicting_fast_error'] - auc_e) < 0.02 else
               "better" if dis['auc_full_disagreement_predicting_fast_error'] > auc_e else "worse")
    if real:
        rq7 = (f"This is a time-window check, not a controlled shift: over {len(drift)} consecutive test windows, fraud rate ranged {drift.fraud_rate.min():.4f} to {drift.fraud_rate.max():.4f}, "
               f"mean disagreement {drift.mean_disagreement.min():.4f} to {drift.mean_disagreement.max():.4f}, router cost {drift.risk_aware_avg_cost.min():.2f} to {drift.risk_aware_avg_cost.max():.2f} units "
               f"and DEEP share {drift.risk_aware_deep_pct.min():.1f}% to {drift.risk_aware_deep_pct.max():.1f}%. See the window table; windows with few frauds have unreliable PR-AUC.")
    else:
        rq7 = (f"Mixed. Mean disagreement rises from {base.mean_disagreement:.3f} (fresh baseline) to {comb.mean_disagreement:.3f} under the combined shift, and the router's cost rises {base.risk_aware_avg_cost:.2f} → {comb.risk_aware_avg_cost:.2f} "
               f"(DEEP share {base.risk_aware_deep_pct:.1f}% → {comb.risk_aware_deep_pct:.1f}%), so disagreement does respond to shift. {qual_sentence} See the drift table.")
    findings = f"""
**RQ1 – Can adaptive routing cut cost while keeping quality?** {rq1_lead} The risk-aware router costs {r.avg_cost_incurred:.2f} proxy units per transaction vs {d.avg_cost_incurred:.2f} for Always-Deep ({r.cost_savings_vs_deep_pct:.1f}% less, cascade cost counted in full) and keeps {r.pr_auc_retention_vs_deep:.3f}× of Always-Deep PR-AUC. Paired bootstrap ΔPR-AUC vs Always-Deep = {ci_ra_deep[0]:+.3f} [{ci_ra_deep[1]:+.3f}, {ci_ra_deep[2]:+.3f}] → {verdict(ci_ra_deep)}. {rq1_caveat} Versus Always-Accurate the router saves {(1 - r.avg_cost_incurred / a.avg_cost_incurred) * 100:.1f}% cost but ΔPR-AUC = {ci_ra_acc[0]:+.3f} [{ci_ra_acc[1]:+.3f}, {ci_ra_acc[2]:+.3f}] ({verdict(ci_ra_acc)}); recall {r.recall:.3f} vs {a.recall:.3f}.

**RQ2 – Is disagreement a useful escalation signal?** {rq2_lead}: AUC for predicting FAST-tier errors = {dis['auc_full_disagreement_predicting_fast_error']:.3f} (3-tier spread) and {dis['auc_fast_accurate_gap_predicting_fast_error']:.3f} (FAST–ACCURATE gap). FAST's own entropy/confidence predicts FAST errors with AUC {auc_e:.3f}; three-tier disagreement does {rq2_cmp if rq2_cmp == 'about as well' else rq2_cmp + ' than that'}{'.' if rq2_cmp == 'about as well' else ' (point estimates, no confidence interval computed).'} {'It is therefore not shown to be superior to a cheap single-model uncertainty.' if rq2_cmp != 'better' else 'It may be the stronger error signal here, but it needs every tier to run.'} Note the full 3-tier spread requires running every tier, so it is only free in offline analysis.

**RQ3 – Does confidence routing beat a fixed model?** Confidence router vs Always-Fast: ΔPR-AUC {ci_c_fast[0]:+.3f} [{ci_c_fast[1]:+.3f}, {ci_c_fast[2]:+.3f}] ({verdict(ci_c_fast)}), at {c.avg_cost_incurred:.2f} vs {f.avg_cost_incurred:.2f} cost units. Recall {c.recall:.3f} vs {f.recall:.3f}.

**RQ4 – Does risk-aware routing improve safety?** Risk-aware vs confidence router: fraud caught {int(r.fraud_caught)} vs {int(c.fraud_caught)} of {abst['fraud_total']}, precision {r.precision:.3f} vs {c.precision:.3f}, cost {r.avg_cost_incurred:.2f} vs {c.avg_cost_incurred:.2f}. Safety-gate ablation: risk-aware with gate caught {int(r.fraud_caught)} vs {int(rns.fraud_caught)} without it (cost {r.avg_cost_incurred:.2f} vs {rns.avg_cost_incurred:.2f}; confidence router {int(c.fraud_caught)} vs {int(cns.fraud_caught)}). The gate fired on {r.safety_override_pct:.2f}% of transactions; its measured effect on test PR-AUC is {r.pr_auc - rns.pr_auc:+.4f}, i.e. small. Its value here is a hard guarantee (high-risk/high-disagreement always reach DEEP) rather than a large metric gain. On the {s1n} slice the gate fires on {hv.safety_override_pct:.1f}% of transactions.

**RQ5 – Does abstention reduce high-confidence errors?** REVIEW covers {abst['review_rate_pct']:.2f}% of transactions and {abst['fraud_sent_to_review']} of {abst['fraud_total']} frauds. Error rate on non-REVIEW rows {abst['error_rate_on_non_review']:.4f} vs {abst['error_rate_all']:.4f} overall; error on REVIEW rows {abst['error_rate_on_review_rows']:.3f}. Missed frauds fall from {abst['fraud_missed_all']} to {abst['fraud_missed_non_review']} among auto-decided rows (the rest are routed to humans).

**RQ6 – Does Jev add measurable value?** **Not evaluated.** No `TYPESAFE_API_KEY` was available when this report was generated, so the Jev router and shadow comparison were not run and no Jev numbers are reported. The adapter, fallback and shadow logic are covered by tests with a fake client only. Run `scripts/benchmark.py` with `JEV_ENABLED=true` and a key to populate the `jev_router` row.

**RQ7 – Is routing more valuable under shift?** {rq7}
"""
    when = f"""
**When does routing help?** When most traffic is easy and one cheap model is confident: {r.fast_pct:.1f}% of test traffic is resolved by FAST.
**When does it hurt?** Cascades pay for every tier they run: a transaction escalated all the way costs 1+3+8 = 12 units, more than Always-Deep (8). Average cost stays low only because escalations are rare.
**When should the system pay for DEEP?** The slices that escalate most are {s2n} ({intl.deep_escalation_pct:.1f}% to DEEP, {intl.fraud_rate * 100:.1f}% fraud rate) and {s1n} ({hv.deep_escalation_pct:.1f}% to DEEP, {hv.fraud_rate * 100:.1f}% fraud rate); overall {r.deep_pct:.2f}% of traffic reaches DEEP. Whether DEEP then beats ACCURATE is a separate question: here DEEP {'does' if d.pr_auc > a.pr_auc else 'does not'} have higher PR-AUC than ACCURATE.
**Does disagreement correlate with errors?** See RQ2: AUC {dis['auc_full_disagreement_predicting_accurate_error']:.3f} against ACCURATE-tier errors, Spearman {dis['spearman_disagreement_vs_accurate_error']:.3f}.
"""
    ds = mm["_meta"]["dataset"]
    lat = pd.DataFrame([{"model": n, "measured p50 ms": mm[n]["latency_measured"]["p50_ms"], "p95 ms": mm[n]["latency_measured"]["p95_ms"], "p99 ms": mm[n]["latency_measured"]["p99_ms"],
                         "proxy ms": mm[n]["latency_proxy_ms"], "cost proxy": mm[n]["cost_proxy"], "size KB": mm[n]["model_size_kb"]} for n in ("fast", "accurate", "deep")])
    cols = ["policy", "pr_auc", "roc_auc", "recall", "precision", "f1", "brier", "avg_cost_incurred", "avg_latency_ms_proxy", "p95_latency_ms_proxy", "fast_pct", "accurate_pct", "deep_pct",
            "safety_override_pct", "review_rate_pct", "cost_savings_vs_deep_pct", "latency_reduction_vs_deep_pct", "pr_auc_retention_vs_deep", "on_pareto"]
    bt = main_b.reset_index()[cols]
    short = bt[["policy", "pr_auc", "recall", "avg_cost_incurred", "avg_latency_ms_proxy", "deep_pct", "cost_savings_vs_deep_pct", "pr_auc_retention_vs_deep"]]
    sl_tbl = stress[stress.policy == "risk_aware"][["slice", "n", "fraud_rate", "fast_pct", "accurate_pct", "deep_pct", "false_negative_rate", "avg_cost", "safety_override_pct", "review_pct"]]
    sc_tbl = pd.DataFrame([{"scenario": s["scenario"], "p_fast": s["probabilities"]["fast"], "p_acc": s["probabilities"]["accurate"], "p_deep": s["probabilities"]["deep"], "disagreement": s["disagreement"],
                            "initial": s["initial_model"], "final": s["selected_model"], "override": s["safety_override"], "decision": s["decision"]} for s in sj["scenarios"]]).fillna("–")
    dcols = ["scenario", "fraud_rate"] + (["intl_fraud_rate"] if "intl_fraud_rate" in drift else []) + ["fast_pr_auc", "accurate_pr_auc", "deep_pr_auc", "mean_disagreement", "risk_aware_pr_auc", "risk_aware_avg_cost", "risk_aware_deep_pct"]
    jev_line = "Jev router: **not evaluated** (no API key at report time)." if "jev_router" not in set(bench.policy) else "Jev router row present in routing_benchmark.csv."

    results = f"""### Setup
{('Real dataset `' + real['name'] + '` (' + real['split_mode'] + ' split, file ' + real['source_file'] + ')') if real else 'Synthetic data'}, seed {cfg['seed']}: {ds['rows']:,} rows, fraud rate {ds['fraud_rate']:.3%}; test split {mm['_meta']['test_rows']:,} rows / {int(round(mm['_meta']['test_fraud_rate'] * mm['_meta']['test_rows']))} frauds. Fraud threshold {thr:.3f} (F1-optimal on validation for the ACCURATE tier). Cost/latency are proxies from `configs/routing.yaml`. {jev_line}

### Model quality (test)
{md(comp[['model', 'pr_auc', 'roc_auc', 'recall', 'precision', 'f1', 'accuracy', 'brier', 'log_loss', 'ece_raw', 'ece_calibrated', 'calibration']], 4)}
### Latency & cost per model
{md(lat, 3)}
Measured latency is single-row wall-clock `predict_proba` on the generating host (including Python overhead); it is host-dependent and does not preserve the 1:3:8 proxy ratios.

### Routing benchmark (test; cost = cascade cost incl. every tier run)
{md(bt, 3)}
Pareto-optimal (cost vs PR-AUC) among these rows: **{pareto}**.

### Paired bootstrap (B={B}, test rows resampled jointly)
{md(boot_tbl, 3)}
### Threshold choice (validation sweep, cheapest threshold within 1% of best validation PR-AUC)
{', '.join(f"{k}: fast-confidence {v[0]} (cost {v[1]:.2f}, val PR-AUC {v[2]:.3f})" for k, v in picks.items())}. Configured values: confidence {cfg['routing']['confidence']['fast_threshold']}, risk-aware {cfg['routing']['risk_aware']['fast_confidence']}.

### Slice analysis (risk-aware router, test)
{md(sl_tbl, 3)}
### Adversarial scenarios (risk-aware router)
{md(sc_tbl, 3)}
Checks: {json.dumps(chk)}

### {"Time windows of the test split (frozen models)" if real else "Distribution shift (frozen models, fresh shifted samples)"}
{md(drift[dcols], 3)}
### Key findings
{findings}
{when}"""
    limits = f"""
### Limitations
{"- Single real dataset, one split and one seed: results may not transfer to other portfolios, periods or fraud patterns. Anonymised features limit interpretation of rule-based flags and slices." if real else "- Synthetic data generated by a known process: absolute metrics say nothing about real fraud; the *relative* comparison of policies is the contribution."}
- {abst['fraud_total']} test frauds: PR-AUC differences smaller than the bootstrap intervals are within noise. Thresholds were tuned on validation, evaluated once on test.
- Cost and latency are proxies; measured latencies are single-row sklearn calls on one host. No batching, GPU, network or queueing effects.
- Disagreement is a heuristic, not calibrated epistemic uncertainty; the 3-tier spread needs all tiers to run.
- DEEP is a small sklearn MLP (no dropout), not a large model{'; it does not outperform ACCURATE on this data' if a.pr_auc >= d.pr_auc else ''}.
- Jev was not evaluated (no credentials in the build environment).
"""
    report = f"# DecisionRouter Pro – Final Benchmark Report\n\n_Generated by scripts/make_report.py from files in artifacts/. No numbers are hand-typed._\n\n{results}\n{limits}"
    (reports_dir() / "final_report.md").write_text(report)
    (reports_dir() / "results_block.md").write_text(results + limits)
    readme = ROOT / "README.md"
    # Only the default synthetic run may rewrite the README: real-data or scratch runs (DR_ARTIFACTS_DIR set) write their own
    # results_block.md and must never replace the README's results.
    if readme.exists() and not real and "DR_ARTIFACTS_DIR" not in os.environ:
        t = readme.read_text()
        new = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->", lambda m: "<!-- RESULTS:START -->\n" + results + "\n<!-- RESULTS:END -->", t, flags=re.S)
        readme.write_text(new)
    print("wrote", reports_dir() / "final_report.md")
    print(findings)


if __name__ == "__main__":
    main()