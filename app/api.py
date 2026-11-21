"""FastAPI service. Run: uvicorn app.api:app --reload   (docs at /docs)."""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.audit import AuditLogger
from src.config import TIERS, artifacts_dir, load_config
from src.explainability import feature_contributions
from src.jev_adapter import JevAdapter
from src.registry import ModelRegistry
from src.router import VALID_POLICIES, DecisionRouter

log = logging.getLogger("decision_router.api")
Flag = Literal[0, 1]


class Transaction(BaseModel):
    """Raw transaction features. merchant_risk is optional (population mean is used when absent)."""
    amount: float = Field(..., ge=0, le=1_000_000, examples=[450])
    hour: int = Field(..., ge=0, le=23, examples=[2])
    account_age_days: int = Field(..., ge=0, le=36500, examples=[30])
    device_seen_before: Flag = Field(..., examples=[0])
    merchant_seen_before: Flag = Field(..., examples=[1])
    transactions_24h: int = Field(..., ge=0, le=1000, examples=[8])
    avg_amount_30d: float = Field(..., gt=0, le=1_000_000, examples=[120])
    distance_from_home_km: float = Field(..., ge=0, le=25000, examples=[350])
    international: Flag = Field(..., examples=[1])
    merchant_risk: Optional[float] = Field(None, ge=0, le=1)

    def to_row(self) -> dict:
        return self.model_dump(exclude_none=True)


class RouteResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    request_id: str
    selected_model: str
    decision: Literal["FRAUD", "LEGIT", "REVIEW"]
    fraud_probability: float
    risk_score: float
    model_disagreement: float
    policy: str
    policy_source: str
    safety_override: Optional[str] = None
    estimated_cost_units: float
    latency_ms: float
    explanation: list[str]


class RawTransaction(BaseModel):
    features: dict[str, Any]


class RawBatch(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1, max_length=5000)


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=5000)


class Stats:
    """Process-local runtime counters for GET /metrics."""

    def __init__(self) -> None:
        self.lock, self.started = threading.Lock(), time.time()
        self.n = 0; self.cost = 0.0; self.lat = 0.0; self.overrides = 0; self.fallbacks = 0
        self.by_model = {t: 0 for t in TIERS}; self.by_decision = {"FRAUD": 0, "LEGIT": 0, "REVIEW": 0}
        self.shadow_n = 0; self.shadow_agree = 0; self.errors = 0

    def add_frame(self, fr: pd.DataFrame) -> None:
        with self.lock:
            self.n += len(fr); self.cost += float(fr["incurred_cost"].sum()); self.lat += float(fr["incurred_latency_ms"].sum())
            self.overrides += int(fr["safety_override"].sum()); self.fallbacks += int((fr["policy_source"] == "local_fallback").sum())
            for t in TIERS:
                self.by_model[t] += int((fr["final_model"] == t).sum())
            for d in self.by_decision:
                self.by_decision[d] += int((fr["decision"] == d).sum())

    def snapshot(self, deep_cost: float) -> dict:
        with self.lock:
            n = max(self.n, 1)
            return {"uptime_s": round(time.time() - self.started, 1), "transactions_routed": self.n, "avg_cost_units": self.cost / n if self.n else None,
                    "estimated_cost_savings_vs_always_deep_pct": (1 - self.cost / (n * deep_cost)) * 100 if self.n else None,
                    "avg_latency_ms_proxy": self.lat / n if self.n else None, "by_model": dict(self.by_model),
                    "by_decision": dict(self.by_decision), "safety_overrides": self.overrides, "jev_fallbacks": self.fallbacks,
                    "shadow": {"n": self.shadow_n, "agreement_rate": (self.shadow_agree / self.shadow_n) if self.shadow_n else None},
                    "errors": self.errors}


def create_app(router: DecisionRouter | None = None) -> FastAPI:
    holder: dict[str, Any] = {"router": router}
    stats = Stats()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if holder["router"] is None:
            cfg = load_config()
            try:
                holder["router"] = DecisionRouter(ModelRegistry.load(), cfg, JevAdapter(cfg), AuditLogger())
                log.info("models loaded")
            except (FileNotFoundError, RuntimeError) as exc:
                holder["load_error"] = str(exc)
                log.warning("%s", exc)
        yield

    app = FastAPI(title="DecisionRouter Pro", version="1.0.0", lifespan=lifespan,
                  description="Cost-aware ML model routing with uncertainty, a deterministic safety gate, abstention, audit trace and an optional Jev decision layer.")

    def need_router() -> DecisionRouter:
        if holder["router"] is None:
            raise HTTPException(503, holder.get("load_error") or "models not trained: run `python scripts/train.py`")
        return holder["router"]

    def check_policy(policy: str) -> str:
        if policy not in VALID_POLICIES:
            raise HTTPException(422, f"unknown policy {policy!r}; valid: {VALID_POLICIES}")
        return policy

    @app.get("/health")
    def health() -> dict:
        r = holder["router"]
        return {"status": "ok", "models_loaded": r is not None, "load_error": holder.get("load_error"), "jev": r.jev.status() if r else JevAdapter(load_config()).status(),
                "policies": VALID_POLICIES}

    @app.post("/route", response_model=RouteResponse)
    def route(txn: Transaction, policy: str = Query("risk_aware", description=f"one of {VALID_POLICIES}")) -> dict:
        r = need_router()
        try:
            rec = r.route_one(txn.to_row(), check_policy(policy))
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            stats.errors += 1
            log.exception("routing failed")
            raise HTTPException(500, f"routing failed: {type(exc).__name__}") from exc
        with stats.lock:
            stats.n += 1; stats.cost += rec["estimated_cost_units"]; stats.lat += rec["estimated_latency_ms_proxy"]
            stats.overrides += int(rec["safety_override"] is not None); stats.fallbacks += int(rec["policy_source"] == "local_fallback")
            stats.by_model[rec["selected_model"]] += 1; stats.by_decision[rec["decision"]] += 1
        return rec

    def run_batch(r: DecisionRouter, df: pd.DataFrame, policy: str, include_rows: bool) -> dict:
        t0 = time.perf_counter()
        fr = r.route(df, policy).frame
        stats.add_frame(fr)
        n, deep = len(fr), float(r.costs["deep"])
        out = {"policy": policy, "total_transactions": n,
               **{f"{t}_pct": float((fr["final_model"] == t).mean() * 100) for t in TIERS},
               "review_pct": float((fr["decision"] == "REVIEW").mean() * 100), "fraud_flagged_pct": float((fr["decision"] == "FRAUD").mean() * 100),
               "safety_override_pct": float(fr["safety_override"].mean() * 100), "average_cost_units": float(fr["incurred_cost"].mean()),
               "estimated_cost_savings_vs_always_deep_pct": float((1 - fr["incurred_cost"].mean() / deep) * 100),
               "average_latency_ms_proxy": float(fr["incurred_latency_ms"].mean()), "wall_clock_ms": round((time.perf_counter() - t0) * 1000, 2)}
        if include_rows:
            out["rows"] = fr[["initial_model", "final_model", "safety_override", "override_reason", "p_final", "risk_score", "model_disagreement",
                              "decision", "incurred_cost"]].to_dict(orient="records")
        return out

    @app.post("/route/batch")
    def route_batch(req: BatchRequest, policy: str = "risk_aware", include_rows: bool = False) -> dict:
        r = need_router()
        check_policy(policy)
        if policy == "jev" and len(req.transactions) > 200:
            raise HTTPException(422, "policy=jev limited to 200 transactions per batch (one API call each)")
        return run_batch(r, pd.DataFrame([t.to_row() for t in req.transactions]), policy, include_rows)

    def check_raw(r: DecisionRouter, rows: list[dict]) -> None:
        need = list(r.registry.feature_builder.required_columns)
        for i, row in enumerate(rows):
            missing = [c for c in need if c not in row]
            if missing:
                raise HTTPException(422, f"row {i}: missing features {missing}")

    @app.post("/route/raw")
    def route_raw(body: RawTransaction, policy: str = Query("risk_aware")) -> dict:
        """Dataset-agnostic routing: {"features": {...}} with the columns listed in /model-info (works for real datasets)."""
        r = need_router()
        check_policy(policy)
        check_raw(r, [body.features])
        rec = r.route_one(body.features, policy)
        with stats.lock:
            stats.n += 1; stats.cost += rec["estimated_cost_units"]; stats.lat += rec["estimated_latency_ms_proxy"]
            stats.overrides += int(rec["safety_override"] is not None); stats.fallbacks += int(rec["policy_source"] == "local_fallback")
            stats.by_model[rec["selected_model"]] += 1; stats.by_decision[rec["decision"]] += 1
        return rec

    @app.post("/route/raw/batch")
    def route_raw_batch(body: RawBatch, policy: str = "risk_aware", include_rows: bool = False) -> dict:
        r = need_router()
        check_policy(policy)
        if policy == "jev" and len(body.rows) > 200:
            raise HTTPException(422, "policy=jev limited to 200 rows per batch (one API call each)")
        check_raw(r, body.rows)
        return run_batch(r, pd.DataFrame(body.rows), policy, include_rows)

    @app.post("/shadow-route")
    def shadow(txn: Transaction) -> dict:
        r = need_router()
        out = r.shadow_route(txn.to_row())
        with stats.lock:
            if out["agreement"] is not None:
                stats.shadow_n += 1; stats.shadow_agree += int(out["agreement"])
        return out

    @app.post("/explain")
    def explain(txn: Transaction) -> dict:
        return feature_contributions(need_router().registry, txn.to_row())

    @app.get("/benchmark")
    def benchmark() -> dict:
        p = artifacts_dir() / "routing_benchmark.csv"
        if not p.exists():
            raise HTTPException(404, "no benchmark yet: run `python scripts/benchmark.py`")
        return {"results": json.loads(pd.read_csv(p).to_json(orient="records"))}

    @app.get("/metrics")
    def metrics() -> dict:
        r = holder["router"]
        return stats.snapshot(float(r.costs["deep"]) if r else 8.0)

    @app.get("/model-info")
    @app.get("/models")
    def model_info() -> dict:
        r = need_router()
        mm = artifacts_dir() / "model_metrics.json"
        return {"dataset": r.registry.feature_builder.name, "required_columns": list(r.registry.feature_builder.required_columns),
                "models": r.registry.info(), "fraud_threshold": r.registry.fraud_threshold, "cost_units": r.costs, "latency_ms_proxy": r.lat,
                "note": "cost/latency are configurable proxies, not cloud prices or production latencies",
                "routing": r.cfg["routing"], "safety": r.cfg["safety"], "abstention": r.cfg["abstention"],
                "test_metrics": json.loads(mm.read_text()) if mm.exists() else None}

    @app.get("/audit/{request_id}")
    def audit(request_id: str) -> dict:
        r = need_router()
        rec = r.audit.get(request_id) if r.audit else None
        if rec is None:
            raise HTTPException(404, "request_id not found")
        return rec

    return app


app = create_app()
