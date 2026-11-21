import copy

import numpy as np
import pandas as pd
import pytest

from src.jev_adapter import JevAdapter
from src.policies import confidence_router, risk_aware_router
from src.router import DecisionRouter
from src.uncertainty import Evidence
from tests.conftest import BENIGN, TXN


def ev(pf, pa, pd_, heur=0.0):
    f = lambda x: np.array([x], float)
    return Evidence(f(pf), f(pa), f(pd_), f(heur))


def test_fixed_policies(router, splits):
    df = splits["test"].head(100)
    for pol, tier in (("fast", "fast"), ("accurate", "accurate"), ("deep", "deep")):
        fr = router.route(df, pol).frame
        assert (fr.final_model == tier).all() and not fr.safety_override.any()
        assert (fr.incurred_cost == router.costs[tier]).all()


def test_confidence_router_cascade_and_cost(cfg):
    c = cfg["routing"]["confidence"]
    out = confidence_router(ev(0.01, 0.5, 0.5), cfg)
    assert out.selection[0] == "fast" and out.run_accurate[0] == False  # noqa: E712
    out = confidence_router(ev(0.5, 0.01, 0.5), cfg)
    assert out.selection[0] == "accurate" and out.run_accurate[0] and not out.run_deep[0]
    out = confidence_router(ev(0.5, 0.5, 0.5), cfg)
    assert out.selection[0] == "deep" and out.run_deep[0]


def test_risk_aware_escalation_paths(cfg):
    assert risk_aware_router(ev(0.01, 0.0, 0.0, 0.0), cfg).selection[0] == "fast"
    assert risk_aware_router(ev(0.01, 0.0, 0.0, 0.9), cfg).selection[0] != "fast"      # rule-based risk blocks FAST path
    assert risk_aware_router(ev(0.15, 0.12, 0.12), cfg).selection[0] == "accurate"      # moderate uncertainty
    assert risk_aware_router(ev(0.3, 0.35, 0.3), cfg).selection[0] == "deep"             # high entropy of the mean
    assert risk_aware_router(ev(0.1, 0.5, 0.5), cfg).selection[0] == "deep"             # FAST/ACC disagreement
    assert risk_aware_router(ev(0.9, 0.9, 0.9), cfg).selection[0] == "deep"             # high risk


def test_router_high_risk_escalation_via_safety(router):
    rec = router.route_one(TXN, "confidence", log=False)
    assert rec["selected_model"] == "deep"


def test_benign_is_cheap(router):
    rec = router.route_one(BENIGN, "risk_aware", log=False)
    assert rec["selected_model"] == "fast" and rec["estimated_cost_units"] == router.costs["fast"] and rec["safety_override"] is None


def test_safety_disagreement_override_in_router(cfg, registry, splits, monkeypatch):
    r = DecisionRouter(registry, cfg)
    df = splits["test"].head(3)
    # FAST unsure (0.5) and ACCURATE confident-legit (0.05) => ACCURATE selected, but observed disagreement 0.45 >= 0.38
    e = Evidence(np.array([0.5] * 3), np.array([0.05] * 3), np.array([0.05] * 3))
    fr = r.route(df, "confidence", ev=e).frame
    assert (fr.initial_model == "accurate").all()
    assert (fr.final_model == "deep").all() and (fr.override_reason == "high_disagreement").all()
    # a FAST-only decision has no observed disagreement, so the gate cannot fire on it (cost-honest design)
    e2 = Evidence(np.array([0.05] * 3), np.array([0.9] * 3), np.array([0.9] * 3))
    assert (r.route(df, "confidence", ev=e2).frame.final_model == "fast").all()


def test_audit_trace_fields_and_lookup(router):
    rec = router.route_one(TXN, "risk_aware")
    for k in ("request_id", "timestamp", "policy", "policy_source", "fast_probability", "risk_score", "model_disagreement", "initial_model",
              "selected_model", "safety_override", "decision", "estimated_cost_units", "latency_ms", "explanation"):
        assert k in rec
    assert router.audit.get(rec["request_id"])["selected_model"] == rec["selected_model"]
    ids = {router.route_one(TXN)["request_id"] for _ in range(5)}
    assert len(ids) == 5


# ----------------------------------------------------------------------- Jev
class FakeResp:
    def __init__(self, choice):
        self.choices = {"tier": type("C", (), {"choice": choice})()}


def fake_factory(choice=None, exc=None):
    class Client:
        def __init__(self, **kw):
            pass
        def system_one(self, state, questions):
            assert set(questions) == {"tier"} and "fast_probability" in state
            if exc:
                raise exc
            return FakeResp(choice)
    return Client


def jev_router(cfg, registry, **kw):
    return DecisionRouter(registry, cfg, JevAdapter(cfg, **kw))


def test_jev_disabled_falls_back(cfg, registry):
    rec = jev_router(cfg, registry, enabled=False).route_one(TXN, "jev", log=False)
    assert rec["policy_source"] == "local_fallback" and rec["fallback_used"] and rec["jev_status"] == "disabled"


def test_jev_missing_api_key(cfg, registry):
    rec = jev_router(cfg, registry, enabled=True, api_key="").route_one(BENIGN, "jev", log=False)
    assert rec["jev_status"] == "missing_api_key" and rec["fallback_used"]


@pytest.mark.parametrize("exc,status", [(RuntimeError("boom"), "error"), (TimeoutError("slow"), "timeout")])
def test_jev_exceptions_never_crash(cfg, registry, exc, status):
    r = jev_router(cfg, registry, enabled=True, api_key="x", client_factory=fake_factory(exc=exc))
    rec = r.route_one(BENIGN, "jev", log=False)
    assert rec["jev_status"] == status and rec["policy_source"] == "local_fallback"


def test_jev_invalid_response_falls_back(cfg, registry):
    r = jev_router(cfg, registry, enabled=True, api_key="x", client_factory=fake_factory("gigantic"))
    assert r.route_one(BENIGN, "jev", log=False)["jev_status"] == "invalid_response"


def test_jev_choice_used_but_safety_still_wins(cfg, registry):
    r = jev_router(cfg, registry, enabled=True, api_key="x", client_factory=fake_factory("fast"))
    rec = r.route_one(TXN, "jev", log=False)          # Jev says fast; high-risk transaction => gate escalates
    assert rec["policy_source"] == "jev" and rec["initial_model"] == "fast"
    assert rec["selected_model"] == "deep" and rec["safety_override"] == "high_risk"
    ok = r.route_one(BENIGN, "jev", log=False)
    assert ok["selected_model"] == "fast" and ok["safety_override"] is None


def test_shadow_mode_never_changes_production(cfg, registry):
    base = DecisionRouter(registry, cfg).route_one(BENIGN, "risk_aware", log=False)
    r = jev_router(cfg, registry, enabled=True, api_key="x", client_factory=fake_factory("deep"))
    out = r.shadow_route(BENIGN)
    assert out["production_selection"] == base["selected_model"] == "fast"
    assert out["shadow_selection"] == "deep" and out["agreement"] is False


def test_shadow_when_jev_unavailable(cfg, registry):
    out = jev_router(cfg, registry, enabled=False).shadow_route(BENIGN)
    assert out["agreement"] is None and out["jev_status"] == "disabled" and out["production_selection"]


def test_unknown_policy(router):
    with pytest.raises(ValueError):
        router.route_one(TXN, "nope")
