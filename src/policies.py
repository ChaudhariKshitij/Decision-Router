"""Routing policies. Each policy maps model evidence to a tier plus the set of tiers it had to RUN.

Cost honesty: cascade policies pay for every tier they execute (FAST, then ACCURATE, then DEEP), not just the
tier whose output is finally used. ``run_*`` flags record that, and the benchmark reports both incurred cost
and selected-tier-only cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .uncertainty import Evidence, binary_entropy


@dataclass
class PolicyOutcome:
    selection: np.ndarray
    run_fast: np.ndarray
    run_accurate: np.ndarray
    run_deep: np.ndarray
    extra: dict = field(default_factory=dict)


def _obj(a) -> np.ndarray:
    return np.asarray(a).astype(object)


def _fixed(tier: str) -> Callable:
    def policy(ev: Evidence, cfg: dict, **_) -> PolicyOutcome:
        n = len(ev)
        t, f = np.ones(n, bool), np.zeros(n, bool)
        return PolicyOutcome(_obj(np.full(n, tier)), t if tier == "fast" else f, t if tier == "accurate" else f,
                             t if tier == "deep" else f)
    policy.__name__ = f"always_{tier}"
    return policy


always_fast, always_accurate, always_deep = _fixed("fast"), _fixed("accurate"), _fixed("deep")


def confidence_router(ev: Evidence, cfg: dict, **_) -> PolicyOutcome:
    """Policy D: FAST if conf(p_fast) >= t1; else ACCURATE if conf(p_acc) >= t2; else DEEP (cascade)."""
    c = cfg["routing"]["confidence"]
    go_fast = ev.fast_confidence >= c["fast_threshold"]
    go_acc = ~go_fast & (ev.accurate_confidence >= c["accurate_threshold"])
    sel = np.where(go_fast, "fast", np.where(go_acc, "accurate", "deep"))
    return PolicyOutcome(_obj(sel), np.ones(len(ev), bool), ~go_fast, sel == "deep")


def risk_aware_router(ev: Evidence, cfg: dict, **_) -> PolicyOutcome:
    """Policy E (cascade). Stage 1 uses FAST + a cheap rule-based risk score; stage 2 adds ACCURATE and
    escalates to DEEP on high risk, FAST/ACCURATE disagreement, or high entropy of their mean."""
    r = cfg["routing"]["risk_aware"]
    stage1 = ((ev.fast_confidence >= r["fast_confidence"]) & (ev.entropy_fast <= r["entropy_threshold"])
              & (ev.heuristic_risk < r["heuristic_risk_max"]))
    d_fa = np.abs(ev.p_fast - ev.p_accurate)
    risk2 = np.maximum(ev.p_fast, ev.p_accurate)
    ent2 = binary_entropy((ev.p_fast + ev.p_accurate) / 2.0)
    esc = (risk2 >= r["escalate_risk"]) | (d_fa >= r["escalate_disagreement"]) | (ent2 >= r["escalate_entropy"])
    sel = np.where(stage1, "fast", np.where(esc, "deep", "accurate"))
    return PolicyOutcome(_obj(sel), np.ones(len(ev), bool), ~stage1, sel == "deep")


LOCAL_POLICIES: dict[str, Callable] = {
    "fast": always_fast, "accurate": always_accurate, "deep": always_deep,
    "confidence": confidence_router, "risk_aware": risk_aware_router,
}
ROUTED_POLICIES = {"confidence", "risk_aware", "jev"}   # safety gate applies to these (not to fixed baselines)
POLICY_ALIASES = {"local": "risk_aware", "always_fast": "fast", "always_accurate": "accurate", "always_deep": "deep"}


def observed_stats(ev: Evidence, rf: np.ndarray, ra: np.ndarray, rd: np.ndarray) -> dict[str, np.ndarray]:
    """Risk / disagreement / entropy using ONLY the tiers that actually ran for each row."""
    P = np.vstack([np.where(rf, ev.p_fast, np.nan), np.where(ra, ev.p_accurate, np.nan), np.where(rd, ev.p_deep, np.nan)])
    mx, mn, mean = np.nanmax(P, axis=0), np.nanmin(P, axis=0), np.nanmean(P, axis=0)
    return {"risk": mx, "disagreement": mx - mn, "entropy": binary_entropy(mean)}
