"""Lightweight experiment records (JSON files). Enough for reproducibility without an MLOps stack."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from .config import artifacts_dir


def record_experiment(kind: str, cfg: dict, registry_meta: dict, policy: str | None, metrics: dict, extra: dict | None = None) -> str:
    exp_id = f"{kind}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    rec = {"experiment_id": exp_id, "timestamp": datetime.now(timezone.utc).isoformat(), "kind": kind,
           "random_seed": cfg["seed"], "dataset_version": registry_meta.get("data_version"),
           "model_versions": registry_meta.get("model_versions"), "routing_policy": policy,
           "thresholds": {"routing": cfg["routing"], "safety": cfg["safety"], "abstention": cfg["abstention"]},
           "cost_proxy": cfg["cost"], "latency_proxy_ms": cfg["latency_ms"], "metrics": metrics, **(extra or {})}
    d = artifacts_dir() / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{exp_id}.json").write_text(json.dumps(rec, indent=2, default=float))
    return exp_id
