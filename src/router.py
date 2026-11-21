"""DecisionRouter: evidence -> policy -> (optional Jev) -> deterministic safety gate -> decision + audit."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import abstention
from .audit import AuditLogger, new_request_id, utc_now
from .config import TIERS
from .explainability import explain_routing
from .jev_adapter import JevAdapter, states_from_frame
from .policies import LOCAL_POLICIES, POLICY_ALIASES, ROUTED_POLICIES, PolicyOutcome, observed_stats, risk_aware_router
from .registry import ModelRegistry
from .safety import apply_safety
from .uncertainty import Evidence

VALID_POLICIES = sorted(set(LOCAL_POLICIES) | {"jev", "local"})


@dataclass
class RouteBatch:
    frame: pd.DataFrame
    evidence: Evidence
    policy: str


class DecisionRouter:
    def __init__(self, registry: ModelRegistry, cfg: dict, jev: JevAdapter | None = None, audit: AuditLogger | None = None) -> None:
        self.registry, self.cfg = registry, cfg
        self.jev = jev or JevAdapter(cfg, enabled=False)
        self.audit = audit
        self.costs = {t: float(cfg["cost"][t]) for t in TIERS}
        self.lat = {t: float(cfg["latency_ms"][t]) for t in TIERS}

    # ------------------------------------------------------------------ batch
    def evidence(self, df: pd.DataFrame) -> Evidence:
        return self.registry.compute_evidence(df)

    def _jev_outcome(self, df: pd.DataFrame, ev: Evidence) -> tuple[PolicyOutcome, dict]:
        local = risk_aware_router(ev, self.cfg)
        fr = df[["international", "device_seen_before", "merchant_seen_before", "hour", "transactions_24h"]].copy()
        fr["p_fast"], fr["p_accurate"] = ev.p_fast, ev.p_accurate
        results = self.jev.recommend_batch(states_from_frame(fr, self.cfg))
        choice = np.array([r.choice for r in results], dtype=object)
        ok = np.array([r.choice is not None for r in results], dtype=bool)
        sel = np.where(ok, choice, local.selection).astype(object)
        n = len(ev)
        rf = np.ones(n, bool)
        ra = np.where(ok, True, local.run_accurate)
        rd = np.where(ok, sel == "deep", local.run_deep)
        extra = {"policy_source": np.where(ok, "jev", "local_fallback").astype(object),
                 "jev_status": np.array([r.status for r in results], dtype=object),
                 "jev_error": np.array([r.error for r in results], dtype=object),
                 "jev_choice": choice, "local_selection": local.selection}
        return PolicyOutcome(sel, rf, ra, rd, extra), extra

    def route(self, df: pd.DataFrame, policy: str, ev: Evidence | None = None, use_safety: bool = True) -> RouteBatch:
        """Route every row. ``use_safety=False`` is only for ablations; fixed baselines never use the gate."""
        policy = POLICY_ALIASES.get(policy, policy)
        if policy not in LOCAL_POLICIES and policy != "jev":
            raise ValueError(f"unknown policy {policy!r}; valid: {VALID_POLICIES}")
        ev = ev if ev is not None else self.evidence(df)
        n = len(ev)
        if policy == "jev":
            out, extra = self._jev_outcome(df, ev)
        else:
            out, extra = LOCAL_POLICIES[policy](ev, self.cfg), {}
        initial = out.selection.astype(object)
        rf, ra, rd = out.run_fast.copy(), out.run_accurate.copy(), out.run_deep.copy()
        obs = observed_stats(ev, rf, ra, rd)
        if use_safety and policy in ROUTED_POLICIES:
            sr = apply_safety(initial, obs["risk"], obs["disagreement"], self.cfg, obs["entropy"])
            final, override, reason = sr.final, sr.override, sr.reason
        else:
            final, override, reason = initial.copy(), np.zeros(n, bool), np.full(n, None, dtype=object)
        rd = rd | (final == "deep")
        ra = ra | (final == "accurate")
        obs = observed_stats(ev, rf, ra, rd)
        p_final = ev.select_prob(final)
        decision, rreason = abstention.decide(p_final, obs["disagreement"], self.registry.fraud_threshold, self.cfg)
        c, l = self.costs, self.lat
        frame = pd.DataFrame({
            "initial_model": initial, "final_model": final, "safety_override": override, "override_reason": reason,
            "p_fast": ev.p_fast, "p_accurate": ev.p_accurate, "p_deep": ev.p_deep, "p_final": p_final,
            "run_fast": rf, "run_accurate": ra, "run_deep": rd,
            "risk_score": obs["risk"], "model_disagreement": obs["disagreement"], "entropy": obs["entropy"],
            "decision": decision, "review_reason": rreason,
            "incurred_cost": rf * c["fast"] + ra * c["accurate"] + rd * c["deep"],
            "selected_cost": pd.Series(final).map(c).to_numpy(float),
            "incurred_latency_ms": rf * l["fast"] + ra * l["accurate"] + rd * l["deep"],
            "policy_source": extra.get("policy_source", np.full(n, "local", dtype=object)),
        })
        for k in ("jev_status", "jev_error", "jev_choice", "local_selection"):
            if k in extra:
                frame[k] = extra[k]
        return RouteBatch(frame, ev, policy)

    # ----------------------------------------------------------------- single
    def route_one(self, txn: dict, policy: str = "risk_aware", log: bool = True) -> dict:
        """Route one transaction and return the full response + audit record."""
        t0 = time.perf_counter()
        policy = POLICY_ALIASES.get(policy, policy)
        df = pd.DataFrame([txn])
        batch = self.route(df, policy)
        r = batch.frame.iloc[0]
        rec = self._audit_record(txn, policy, r)
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        rec["explanation"] = explain_routing(rec, self.cfg, self.registry.fraud_threshold)
        if log and self.audit is not None:
            self.audit.log(rec)
        return rec

    def _audit_record(self, txn: dict, policy: str, r: pd.Series) -> dict:
        def prob(tier: str):
            return round(float(r[f"p_{tier}"]), 4) if bool(r[f"run_{tier}"]) else None
        fast_p = float(r["p_fast"])
        override = bool(r["safety_override"])
        source = str(r["policy_source"])
        rec = {
            "request_id": new_request_id(), "timestamp": utc_now(), "policy": policy, "policy_source": source,
            "fast_probability": prob("fast"), "accurate_probability": prob("accurate"), "deep_probability": prob("deep"),
            "fast_confidence": round(max(fast_p, 1 - fast_p), 4),
            "fraud_probability": round(float(r["p_final"]), 4), "risk_score": round(float(r["risk_score"]), 4),
            "model_disagreement": round(float(r["model_disagreement"]), 4), "entropy": round(float(r["entropy"]), 4),
            "initial_model": r["initial_model"], "selected_model": r["final_model"],
            "safety_override": r["override_reason"] if override else None,
            "decision": r["decision"], "review_reason": r["review_reason"],
            "estimated_cost_units": float(r["incurred_cost"]), "estimated_latency_ms_proxy": float(r["incurred_latency_ms"]),
            "models_run": [t for t in TIERS if bool(r[f"run_{t}"])],
            "risk_flags": self.registry.feature_builder.risk_flags(txn),
        }
        if policy == "jev":
            rec.update({"jev_status": r.get("jev_status"), "jev_error": r.get("jev_error"),
                        "fallback_used": source == "local_fallback"})
        return rec

    def shadow_route(self, txn: dict) -> dict:
        """Production = local risk-aware router (authoritative). Jev only recommends; it never changes the outcome."""
        df = pd.DataFrame([txn])
        ev = self.evidence(df)
        prod = self.route(df, "risk_aware", ev=ev)
        rec = self._audit_record(txn, "risk_aware", prod.frame.iloc[0])
        rec["explanation"] = explain_routing(rec, self.cfg, self.registry.fraud_threshold)
        fr = df[["international", "device_seen_before", "merchant_seen_before", "hour", "transactions_24h"]].copy()
        fr["p_fast"], fr["p_accurate"] = ev.p_fast, ev.p_accurate
        res = self.jev.recommend(states_from_frame(fr, self.cfg)[0])
        shadow_raw = res.choice
        shadow_final = None
        if shadow_raw is not None:   # what the gate would have done with Jev's pick (for fair comparison)
            obs = observed_stats(ev, np.ones(1, bool), np.ones(1, bool), np.array([shadow_raw == "deep"]))
            sr = apply_safety(np.array([shadow_raw], dtype=object), obs["risk"], obs["disagreement"], self.cfg, obs["entropy"])
            shadow_final = str(sr.final[0])
        out = {
            "request_id": rec["request_id"], "timestamp": rec["timestamp"],
            "production_selection": rec["selected_model"], "shadow_selection": shadow_final, "shadow_raw_selection": shadow_raw,
            "agreement": (shadow_final == rec["selected_model"]) if shadow_final else None,
            "jev_status": res.status, "jev_error": res.error, "jev_latency_ms": res.latency_ms,
            "production_decision": rec["decision"], "production_audit": rec,
            "note": "Shadow mode: Jev never influences the production selection.",
        }
        if self.audit is not None:
            self.audit.log({**{k: v for k, v in out.items() if k != "production_audit"}, "kind": "shadow",
                            "policy": "shadow"})
            self.audit.log(rec)
        return out
