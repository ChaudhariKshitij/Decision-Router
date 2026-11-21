"""DecisionRouter Pro dashboard.  Run: streamlit run app/streamlit_app.py

Reads artifacts produced by scripts/*.py. Routing here is LOCAL ONLY: no Jev calls and no API keys exist in this process.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.audit import AuditLogger
from src.config import artifacts_dir, load_config
from src.explainability import feature_contributions
from src.config import data_dir
from src.jev_adapter import JevAdapter, build_state
from src.registry import ModelRegistry
from src.router import DecisionRouter

st.set_page_config(page_title="DecisionRouter Pro", page_icon="🧭", layout="wide")
PALETTE = {"fast": "#2a9d8f", "accurate": "#e9c46a", "deep": "#e76f51", "review": "#6d597a"}
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.kpi {border:1px solid rgba(128,128,128,.25); border-radius:10px; padding:12px 16px; background:rgba(128,128,128,.06);}
.kpi .l {font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; opacity:.65;}
.kpi .v {font-size:1.55rem; font-weight:650; line-height:1.25;}
.kpi .s {font-size:.75rem; opacity:.6;}
.pill {display:inline-block; padding:2px 12px; border-radius:99px; color:white; font-weight:600;}
</style>""", unsafe_allow_html=True)

ART = artifacts_dir()


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


@st.cache_resource
def get_router() -> tuple[DecisionRouter | None, str | None]:
    """Cached local router (Jev off). The API key is never passed into any cached function."""
    try:
        cfg = load_config()
        return DecisionRouter(ModelRegistry.load(), cfg), None
    except (FileNotFoundError, RuntimeError) as exc:
        return None, str(exc)


@st.cache_data
def read_csv(name: str) -> pd.DataFrame | None:
    p = ART / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def read_csv_data(name: str) -> pd.DataFrame:
    return pd.read_csv(data_dir() / "processed" / name)


@st.cache_data
def read_json(name: str):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


router, load_error = get_router()
bench, comp = read_csv("routing_benchmark.csv"), read_csv("model_comparison.csv")
if router is None or bench is None:
    st.error(load_error or "Artifacts missing. Run `python scripts/train.py` and `python scripts/benchmark.py` first.")
    st.stop()


def _clear_key() -> None:
    st.session_state["jev_key_input"] = ""
    st.session_state["jev_on"] = False
    for k in ("jev_last", "jev_test_result", "jev_bench"):
        st.session_state.pop(k, None)


def jev_adapter() -> JevAdapter | None:
    """Adapter built per run from this browser session's key (or TYPESAFE_API_KEY in the server env). Never cached, never saved."""
    key = (st.session_state.get("jev_key_input") or "").strip() or os.getenv("TYPESAFE_API_KEY", "").strip()
    if not (st.session_state.get("jev_on") and key):
        return None
    return JevAdapter(router.cfg, enabled=True, api_key=key, model=(st.session_state.get("jev_model") or "jev-latest").strip())

st.sidebar.title("🧭 DecisionRouter Pro")
st.sidebar.caption("Cost-aware model routing · uncertainty · deterministic safety gate · optional Jev layer")
page = st.sidebar.radio("View", ["Overview", "Single transaction", "Model comparison", "Routing benchmark", "Routing distribution",
                                 "Stress testing", "Distribution shift", "Jev"])
st.sidebar.info("Cost and latency are **proxy units** from `configs/routing.yaml`, not cloud prices. Data is synthetic.")
with st.sidebar.expander("🔑 Jev (optional)", expanded=False):
    st.caption("Paste your TypeSafe API key to let Jev choose the model tier. Jev only picks a tier; the deterministic safety gate still runs after it.")
    env_key = bool(os.getenv("TYPESAFE_API_KEY", "").strip())
    st.text_input("TypeSafe API key", type="password", key="jev_key_input", placeholder="set in server env" if env_key else "paste key", autocomplete="off")
    st.text_input("Jev model", value=os.getenv("JEV_MODEL", "jev-latest"), key="jev_model")
    st.toggle("Enable Jev", key="jev_on")
    b1, b2 = st.columns(2)
    test_clicked = b1.button("Test", key="jev_test_btn", help="Makes one Jev API call")
    b2.button("Clear key", key="jev_clear_btn", on_click=_clear_key)
    _ad = jev_adapter()
    if test_clicked:
        if _ad is None:
            st.session_state["jev_test_result"] = ("warn", "Add a key and switch Enable Jev on first.")
        else:
            _probe = build_state({"fast_probability": 0.30, "accurate_probability": 0.55, "deep_probability": None, "fast_confidence": 0.70,
                                  "model_disagreement": 0.25, "entropy": 0.9, "risk_score": 0.55, "transaction_context": {"international": 1}}, router.cfg)
            _res = _ad.recommend(_probe)
            st.session_state["jev_test_result"] = (("ok", f"Jev answered: **{_res.choice}** ({_res.latency_ms:.0f} ms)") if _res.choice
                                                   else ("err", f"{_res.status}: {_res.error or ''}"))
    _tr = st.session_state.get("jev_test_result")
    if _tr:
        {"ok": st.success, "warn": st.warning, "err": st.error}[_tr[0]](_tr[1])
    st.caption("The key lives only in this browser session on the Streamlit server process. It is never written to disk, logs or the audit file. "
               "Only use this over HTTPS or on your own machine.")
main_rows = bench[bench["n_rows"] == bench["n_rows"].max()]
ra = main_rows.set_index("policy").loc["risk_aware_router"]

if page == "Overview":
    st.title("Overview")
    st.caption("Held-out test split, risk-aware router with the deterministic safety gate.")
    c = st.columns(6)
    kpi(c[0], "Transactions routed", f"{int(ra['n_rows']):,}", "test split")
    kpi(c[1], "Avg cost", f"{ra['avg_cost_incurred']:.2f}", f"vs 8.00 always-deep ({ra['cost_savings_vs_deep_pct']:.0f}% less)")
    kpi(c[2], "Avg latency (proxy)", f"{ra['avg_latency_ms_proxy']:.1f} ms", "vs 20 ms always-deep")
    kpi(c[3], "Deep escalation", f"{ra['deep_pct']:.1f}%", "")
    kpi(c[4], "REVIEW rate", f"{ra['review_rate_pct']:.1f}%", "abstention")
    kpi(c[5], "Safety overrides", f"{ra['safety_override_pct']:.2f}%", "forced to DEEP")
    st.subheader("Architecture")
    st.code("Transaction → features → FAST ─(confident & low rule-risk)→ decision\n"
            "                          └→ ACCURATE ─(risk / disagreement / entropy high)→ DEEP\n"
            "router choice → [optional Jev recommendation] → deterministic SAFETY GATE (final authority) → FRAUD / LEGIT / REVIEW + audit record",
            language="text")
    st.markdown("**Models are evidence generators, the router is a bounded decision layer, the safety gate is deterministic and cannot be overridden by Jev.**")
    st.dataframe(main_rows[["policy", "pr_auc", "recall", "avg_cost_incurred", "cost_savings_vs_deep_pct", "pr_auc_retention_vs_deep"]].round(3), hide_index=True, width="stretch")

elif page == "Single transaction":
    st.title("Single transaction")
    left, right = st.columns([1, 1.5])
    with left:
        pol = st.selectbox("Routing policy", ["risk_aware", "confidence", "fast", "accurate", "deep", "jev"], key="single_policy")
        real_ds = router.registry.feature_builder.kind != "synthetic"
        if real_ds:
            te_real = read_csv_data("test.csv")
            which = st.radio("Test-set rows", ["all", "fraud only", "legit only"], horizontal=True, key="real_filter")
            pool = te_real if which == "all" else te_real[te_real.fraud == (1 if which == "fraud only" else 0)]
            pos = st.number_input(f"Row (0 to {len(pool) - 1})", 0, max(len(pool) - 1, 0), 0, key="real_row")
            row = pool.iloc[int(pos)]
            st.caption(f"Dataset `{router.registry.feature_builder.name}` · true label: **{'FRAUD' if row['fraud'] == 1 else 'legit'}** · features are anonymised, so rows are chosen rather than typed.")
        else:
            amount = st.number_input("Amount", 0.0, 100000.0, 450.0, 10.0)
            avg = st.number_input("30-day average amount", 1.0, 100000.0, 120.0, 10.0)
            hour = st.slider("Hour of day", 0, 23, 2)
            age = st.number_input("Account age (days)", 0, 20000, 30)
            tx = st.slider("Transactions in last 24h", 0, 30, 8)
            dist = st.number_input("Distance from home (km)", 0.0, 20000.0, 350.0, 10.0)
            dev = st.checkbox("Device seen before", False)
            mer = st.checkbox("Merchant seen before", True)
            intl = st.checkbox("International", True)
    if real_ds:
        txn = {k: (float(row[k]) if pd.api.types.is_number(row[k]) else row[k]) for k in router.registry.feature_builder.required_columns}
    else:
        txn = dict(amount=amount, hour=hour, account_age_days=int(age), device_seen_before=int(dev), merchant_seen_before=int(mer),
               transactions_24h=tx, avg_amount_30d=avg, distance_from_home_km=dist, international=int(intl))
    _ad = jev_adapter()
    _jev_router = DecisionRouter(router.registry, router.cfg, _ad) if _ad else None
    _sig = json.dumps(txn, sort_keys=True)
    note = None
    if pol == "jev" and _jev_router is not None:
        with left:
            if st.button("Route with Jev (1 API call)", key="jev_route_btn"):
                st.session_state["jev_last"] = (_sig, _jev_router.route_one(txn, "jev", log=False))
        last = st.session_state.get("jev_last")
        if last and last[0] == _sig:
            r = last[1]
        else:
            r = router.route_one(txn, "risk_aware", log=False)
            note = "Showing the local risk-aware result. Press **Route with Jev** to ask Jev (one API call per press; sliders do not call Jev)."
    else:
        r = router.route_one(txn, pol, log=False)
        if pol == "jev":
            note = "Jev is not enabled in this session (add a key and switch it on in the sidebar), so the local router was used."
    with right:
        if note:
            st.info(note)
        if r["policy"] == "jev":
            st.markdown(f"**Decision source:** `{r['policy_source']}` · Jev status `{r.get('jev_status')}`" + (f" · {r.get('jev_error')}" if r.get("jev_error") else ""))
        col = {"FRAUD": "#c1121f", "LEGIT": "#2a9d8f", "REVIEW": "#6d597a"}[r["decision"]]
        st.markdown(f'### Decision <span class="pill" style="background:{col}">{r["decision"]}</span> &nbsp; routed to '
                    f'<span class="pill" style="background:{PALETTE[r["selected_model"]]}">{r["selected_model"].upper()}</span>', unsafe_allow_html=True)
        k = st.columns(4)
        kpi(k[0], "P(fraud)", f"{r['fraud_probability']:.3f}", f"threshold {router.registry.fraud_threshold:.3f}")
        kpi(k[1], "Risk score", f"{r['risk_score']:.3f}", "max over tiers run")
        kpi(k[2], "FAST confidence", f"{r['fast_confidence']:.3f}", "max(p, 1-p)")
        kpi(k[3], "Entropy", f"{r['entropy']:.3f} bits", "")
        k = st.columns(4)
        kpi(k[0], "Disagreement", f"{r['model_disagreement']:.3f}", "max-min over tiers run")
        kpi(k[1], "Cost (cascade)", f"{r['estimated_cost_units']:.0f}", "units, all tiers run")
        kpi(k[2], "Latency proxy", f"{r['estimated_latency_ms_proxy']:.0f} ms", "")
        kpi(k[3], "Safety override", str(r["safety_override"] or "none"), "")
        probs = pd.DataFrame({"tier": ["fast", "accurate", "deep"], "P(fraud)": [r["fast_probability"], r["accurate_probability"], r["deep_probability"]]})
        probs["status"] = probs["P(fraud)"].map(lambda v: "not run" if pd.isna(v) else "ran")
        st.plotly_chart(px.bar(probs.fillna(0), x="tier", y="P(fraud)", color="tier", color_discrete_map=PALETTE, text=probs["P(fraud)"].map(lambda v: "not run" if pd.isna(v) else f"{v:.3f}"),
                               range_y=[0, 1], height=260), width="stretch")
        st.markdown("**Why was this model selected?** *(deterministic, generated from the audit trace)*")
        for line in r["explanation"]:
            st.markdown(f"- {line}")
        fc = feature_contributions(router.registry, txn)
        st.markdown("**Risk factors:** " + (", ".join(fc["risk_flags"]) or "none"))
        st.dataframe(pd.DataFrame(fc["contributions"]).round(3), hide_index=True, width="stretch")
        st.caption(fc["note"])
        if _jev_router is not None:
            with st.expander("Shadow comparison: local vs Jev (does not change the decision)"):
                if st.button("Ask Jev in shadow mode (1 API call)", key="jev_shadow_btn"):
                    out = _jev_router.shadow_route(txn)
                    log = st.session_state.setdefault("shadow_log", [])
                    log.append({k: out.get(k) for k in ("production_selection", "shadow_selection", "shadow_raw_selection", "agreement", "jev_status", "jev_error")})
                    del log[:-200]
                    st.session_state["shadow_last"] = log[-1]
                last_s = st.session_state.get("shadow_last")
                if last_s:
                    st.json(last_s)
        with st.expander("Audit record"):
            st.json(r)

elif page == "Model comparison":
    st.title("Model comparison")
    st.caption("Test split; probabilities calibrated on validation only when it measurably helped (see calibration column).")
    st.dataframe(comp.round(4), hide_index=True, width="stretch")
    c = st.columns(3)
    for col, m in zip(c, ["pr_auc", "recall", "brier"]):
        col.plotly_chart(px.bar(comp, x="model", y=m, color="model", color_discrete_map=PALETTE, title=m, height=280), width="stretch")
    c = st.columns(2)
    c[0].plotly_chart(px.bar(comp, x="model", y=["latency_p50_ms", "latency_p95_ms", "latency_p99_ms"], barmode="group", title="Measured single-row latency (ms, this host)", height=300), width="stretch")
    c[1].plotly_chart(px.bar(comp, x="model", y="cost_proxy", color="model", color_discrete_map=PALETTE, title="Cost proxy (config)", height=300), width="stretch")
    cal = read_csv("calibration.csv")
    if cal is not None:
        fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="gray"), name="perfect"))
        for (m, k), g in cal.groupby(["model", "probabilities"]):
            fig.add_trace(go.Scatter(x=g.mean_predicted, y=g.observed_rate, mode="lines+markers", name=f"{m} ({k})", line=dict(dash="solid" if k == "calibrated" else "dot"), marker_color=PALETTE[m]))
        fig.update_layout(title="Reliability diagram (test)", xaxis_title="mean predicted", yaxis_title="observed fraud rate", height=420)
        st.plotly_chart(fig, width="stretch")

elif page == "Routing benchmark":
    st.title("Routing benchmark")
    cols = ["policy", "pr_auc", "roc_auc", "recall", "precision", "f1", "brier", "avg_cost_incurred", "avg_latency_ms_proxy", "p95_latency_ms_proxy",
            "fast_pct", "accurate_pct", "deep_pct", "escalation_rate_pct", "safety_override_pct", "review_rate_pct", "cost_savings_vs_deep_pct",
            "latency_reduction_vs_deep_pct", "pr_auc_retention_vs_deep", "on_pareto"]
    st.dataframe(main_rows[cols].round(3), hide_index=True, width="stretch")
    sweep = read_csv("threshold_sweep.csv")
    for xcol, title in (("avg_cost_incurred", "Average incurred cost"), ("avg_latency_ms_proxy", "Average latency (proxy ms)")):
        fig = px.scatter(main_rows, x=xcol, y="pr_auc", color="on_pareto", text="policy", title=f"{title} vs PR-AUC (test)", height=430)
        fig.update_traces(textposition="top center", marker_size=12)
        if sweep is not None:
            for pol, g in sweep[sweep.split == "test"].groupby("policy"):
                fig.add_trace(go.Scatter(x=g[xcol if xcol in g else "avg_cost_incurred"], y=g.pr_auc, mode="lines", name=f"{pol} threshold sweep", opacity=.4))
        st.plotly_chart(fig, width="stretch")
    st.caption("PR-AUC on ~300 test frauds is noisy (see bootstrap CIs in the final report): small differences are not evidence of superiority.")

elif page == "Routing distribution":
    st.title("Routing distribution")
    d = main_rows.melt(id_vars="policy", value_vars=["fast_pct", "accurate_pct", "deep_pct"], var_name="tier", value_name="pct")
    d["tier"] = d["tier"].str.replace("_pct", "")
    st.plotly_chart(px.bar(d, x="policy", y="pct", color="tier", color_discrete_map=PALETTE, title="Final model used (%)", height=420), width="stretch")
    st.plotly_chart(px.bar(main_rows, x="policy", y="review_rate_pct", title="REVIEW rate (%)", height=300), width="stretch")
    sw = read_csv("selective_risk.csv")
    if sw is not None:
        st.plotly_chart(px.line(sw, x="coverage", y="error_rate", color="uncertainty", markers=True, title="Selective risk: error rate vs coverage (risk-aware router, test)", height=360), width="stretch")
    dis = read_json("disagreement_analysis.json")
    if dis:
        st.markdown("**Does disagreement predict errors?** (AUC 0.5 = no signal)")
        st.json({k: round(v, 3) for k, v in dis.items() if k.startswith(("auc", "spearman"))})

elif page == "Stress testing":
    st.title("Stress testing")
    s = read_csv("stress_test_results.csv")
    if s is None:
        st.warning("Run scripts/stress_test.py")
    else:
        pol = st.selectbox("Policy", sorted(s.policy.unique()), index=sorted(s.policy.unique()).index("risk_aware"))
        sel = st.multiselect("Slices", list(s.slice.unique()), default=list(s.slice.unique()))
        g = s[(s.policy == pol) & (s.slice.isin(sel))]
        st.dataframe(g.drop(columns=["policy"]).round(3), hide_index=True, width="stretch")
        st.plotly_chart(px.bar(g, x="slice", y=["fast_pct", "accurate_pct", "deep_pct"], title="Routing distribution by slice (%)", height=380,
                               color_discrete_sequence=[PALETTE["fast"], PALETTE["accurate"], PALETTE["deep"]]), width="stretch")
        st.plotly_chart(px.bar(g, x="slice", y=["false_negative_rate"], title="False-negative rate by slice", height=300), width="stretch")
    sj = read_json("stress_test_results.json")
    if sj:
        st.subheader("Adversarial scenarios")
        st.dataframe(pd.DataFrame([{"scenario": x["scenario"], **{f"p_{k}": v for k, v in x["probabilities"].items()}, "disagreement": x["disagreement"],
                                    "initial": x["initial_model"], "selected": x["selected_model"], "override": x["safety_override"], "decision": x["decision"]} for x in sj["scenarios"]]),
                     hide_index=True, width="stretch")

elif page == "Distribution shift":
    st.title("Distribution shift")
    d = read_csv("drift_results.csv")
    if d is None:
        st.warning("Run scripts/drift_test.py")
    else:
        st.dataframe(d.round(3), hide_index=True, width="stretch")
        st.plotly_chart(px.bar(d, x="scenario", y="mean_disagreement", title="Mean model disagreement under shift", height=340), width="stretch")
        m = d.melt(id_vars="scenario", value_vars=["fast_pr_auc", "accurate_pr_auc", "deep_pr_auc", "risk_aware_pr_auc"], var_name="model", value_name="pr_auc")
        st.plotly_chart(px.bar(m, x="scenario", y="pr_auc", color="model", barmode="group", title="PR-AUC under shift", height=380), width="stretch")

else:
    st.title("Jev (optional decision layer)")
    st.markdown("Jev is **optional**. It chooses a *model tier* from structured evidence; it never predicts fraud and the deterministic safety gate always runs after it. "
                "Enter your key in the sidebar (🔑 Jev) to use it here. Nothing on this page is simulated.")
    ad = jev_adapter()
    jr = DecisionRouter(router.registry, router.cfg, ad) if ad else None
    c = st.columns(4)
    kpi(c[0], "Jev in this session", "ON" if ad else "OFF", "key set + toggle on" if ad else "add key, enable in sidebar")
    shadow_log = st.session_state.get("shadow_log", [])
    comparable = [x for x in shadow_log if x.get("agreement") is not None]
    kpi(c[1], "Shadow agreement", f"{sum(bool(x['agreement']) for x in comparable) / len(comparable):.0%}" if comparable else "n/a", f"{len(comparable)} comparable calls this session")
    jb = bench[bench.policy == "jev_router"]
    kpi(c[2], "Offline benchmark row", "present" if len(jb) else "not run", "scripts/benchmark.py with JEV_ENABLED=true")

    st.subheader("Mini benchmark: Jev vs the local router on test rows")
    st.caption("Calls Jev once per row. Small samples describe routing behaviour (agreement, tier mix, cost); they are NOT a reliable quality estimate.")
    n_rows = st.slider("Rows to send to Jev", 5, 100, 30, 5, key="jev_bench_n")
    if st.button(f"Run Jev on {n_rows} test rows", key="jev_bench_btn", disabled=jr is None):
        te_path = data_dir() / "processed" / "test.csv"
        sample = pd.read_csv(te_path).sample(n_rows, random_state=router.cfg["seed"]).reset_index(drop=True)
        with st.spinner(f"Calling Jev for {n_rows} rows..."):
            ev = router.evidence(sample)
            fj = jr.route(sample, "jev", ev=ev).frame
            fl = router.route(sample, "risk_aware", ev=ev).frame
        ok_mask = fj["policy_source"] == "jev"
        res = {"rows": n_rows, "jev_answered": int(ok_mask.sum()), "fallbacks": int((~ok_mask).sum()),
               "agreement_pct": float((fj.loc[ok_mask, "initial_model"] == fl.loc[ok_mask, "initial_model"]).mean() * 100) if ok_mask.any() else None,
               "errors": sorted({str(e) for e in fj.loc[~ok_mask, "jev_error"].dropna()})[:3],
               "rows_table": pd.DataFrame({
                   "policy": ["local risk_aware", "jev"],
                   "fast_%": [(fl.final_model == "fast").mean() * 100, (fj.final_model == "fast").mean() * 100],
                   "accurate_%": [(fl.final_model == "accurate").mean() * 100, (fj.final_model == "accurate").mean() * 100],
                   "deep_%": [(fl.final_model == "deep").mean() * 100, (fj.final_model == "deep").mean() * 100],
                   "avg_cost": [fl.incurred_cost.mean(), fj.incurred_cost.mean()],
                   "safety_override_%": [fl.safety_override.mean() * 100, fj.safety_override.mean() * 100],
                   "frauds_caught": [int(((fl.p_final >= router.registry.fraud_threshold) & (sample.fraud == 1)).sum()),
                                     int(((fj.p_final >= router.registry.fraud_threshold) & (sample.fraud == 1)).sum())]}).round(2)}
        st.session_state["jev_bench"] = res
    if jr is None:
        st.info("Jev is off in this session. Add a key and switch **Enable Jev** on in the sidebar to run this.")
    jbx = st.session_state.get("jev_bench")
    if jbx:
        m = st.columns(3)
        kpi(m[0], "Jev answered", f"{jbx['jev_answered']} / {jbx['rows']}", f"{jbx['fallbacks']} fell back to local")
        kpi(m[1], "Agreement with local", f"{jbx['agreement_pct']:.0f}%" if jbx["agreement_pct"] is not None else "n/a", "initial tier, before safety gate")
        kpi(m[2], "Safety gate", "always on", "applied after Jev")
        st.dataframe(jbx["rows_table"], hide_index=True, width="stretch")
        if jbx["errors"]:
            st.warning("Jev errors seen: " + " | ".join(jbx["errors"]))

    if len(jb):
        st.subheader("Offline benchmark row (full test split or fixed sample)")
        st.dataframe(jb.round(3), hide_index=True, width="stretch")
    if comparable:
        st.subheader("Shadow comparisons this session")
        st.dataframe(pd.DataFrame(comparable).tail(50), hide_index=True, width="stretch")
