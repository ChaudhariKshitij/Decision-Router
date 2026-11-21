"""Optional Jev (TypeSafe AI) decision layer.

Jev answers ONE bounded question -- which model tier should process this transaction -- as a typed ``Choice``
(fast / accurate / deep). It never predicts fraud and never executes anything. Every failure mode degrades to
a ``JevResult`` with ``choice=None`` so callers fall back to the local deterministic router. The deterministic
safety gate is applied AFTER Jev and cannot be overridden by it.

SDK usage follows https://docs.typesafe.ai/sdk/python : ``TypeSafeClient(api_key=, model=, timeout=)`` and
``client.system_one(state=..., questions={name: Choice(instructions=..., criteria={...})})`` ->
``response.choices[name].choice``.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .config import TIERS

log = logging.getLogger(__name__)
QUESTION = "tier"
INSTRUCTIONS = (
    "Which ML model tier should process this card transaction? Minimize inference cost while preserving reliability. "
    "Escalate uncertain transactions (low FAST confidence, high entropy, FAST/ACCURATE disagreement) to a stronger tier. "
    "Prefer deep for high-risk or high-disagreement cases. Choose fast only when evidence is confident and consistent. "
    "You are NOT deciding whether the transaction is fraud."
)
CRITERIA = {
    "fast": "Logistic regression: cost 1, ~2 ms. Use when evidence is confident and models agree.",
    "accurate": "Gradient boosting: cost 3, ~8 ms. Use for moderate uncertainty.",
    "deep": "Neural network: cost 8, ~20 ms. Use for high risk, high disagreement or high uncertainty.",
}


@dataclass
class JevResult:
    choice: str | None
    status: str          # ok | disabled | missing_api_key | sdk_missing | timeout | error | invalid_response
    error: str | None = None
    latency_ms: float | None = None
    model: str | None = None

    @property
    def fallback_used(self) -> bool:
        return self.choice is None


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def build_state(row: dict, cfg: dict) -> dict:
    """Structured, non-identifying evidence handed to Jev. ``None`` = that tier was not run."""
    return {
        "fast_probability": row.get("fast_probability"), "accurate_probability": row.get("accurate_probability"),
        "deep_probability": row.get("deep_probability"), "fast_confidence": row.get("fast_confidence"),
        "model_disagreement": row.get("model_disagreement"), "entropy": row.get("entropy"),
        "risk_score": row.get("risk_score"), "transaction_context": row.get("transaction_context", {}),
        "estimated_cost": cfg["cost"], "estimated_latency_ms": cfg["latency_ms"],
    }


class JevAdapter:
    """Thin, failure-proof wrapper around the TypeSafe SDK.

    ``client_factory`` lets tests inject a fake client (no network, no credentials).
    """

    def __init__(self, cfg: dict, enabled: bool | None = None, api_key: str | None = None, model: str | None = None,
                 client_factory: Callable[..., Any] | None = None) -> None:
        self.cfg = cfg
        self.enabled = _env_bool("JEV_ENABLED") if enabled is None else enabled
        self._api_key = api_key if api_key is not None else os.getenv("TYPESAFE_API_KEY", "")
        self.model = model or os.getenv("JEV_MODEL", "jev-latest")
        self.timeout = float(cfg.get("jev", {}).get("timeout_s", 5.0))
        self._factory, self._client = client_factory, None

    @property
    def available(self) -> bool:
        return bool(self.enabled and self._api_key)

    def status(self) -> dict:
        return {"enabled": self.enabled, "api_key_configured": bool(self._api_key), "model": self.model}

    def _get_client(self):
        if self._client is None:
            if self._factory is not None:
                self._client = self._factory(api_key=self._api_key, model=self.model, timeout=self.timeout)
            else:
                from typesafe_sdk import TypeSafeClient
                self._client = TypeSafeClient(api_key=self._api_key, model=self.model, timeout=self.timeout)
        return self._client

    def recommend(self, state: dict) -> JevResult:
        """Ask Jev for a tier. Never raises."""
        if not self.enabled:
            return JevResult(None, "disabled")
        if not self._api_key:
            return JevResult(None, "missing_api_key", "TYPESAFE_API_KEY is not set")
        t0 = time.perf_counter()
        try:
            client = self._get_client()
            if self._factory is not None:
                question = {"instructions": INSTRUCTIONS, "criteria": CRITERIA}
            else:
                from typesafe_sdk import Choice
                question = Choice(instructions=INSTRUCTIONS, criteria=dict(CRITERIA))
            resp = client.system_one(state=state, questions={QUESTION: question})
            choice = resp.choices[QUESTION].choice
        except ImportError as exc:
            return JevResult(None, "sdk_missing", str(exc))
        except Exception as exc:  # noqa: BLE001 - any SDK/network error must degrade gracefully
            name = type(exc).__name__
            status = "timeout" if "Timeout" in name or isinstance(exc, TimeoutError) else "error"
            log.warning("Jev call failed (%s): %s", name, exc)
            msg = f"{name}: {exc}"
            if self._api_key:
                msg = msg.replace(self._api_key, "***")
            return JevResult(None, status, msg, (time.perf_counter() - t0) * 1000, self.model)
        lat = (time.perf_counter() - t0) * 1000
        if choice not in TIERS:
            return JevResult(None, "invalid_response", f"unexpected choice {choice!r}", lat, self.model)
        return JevResult(str(choice), "ok", None, lat, self.model)

    def recommend_batch(self, states: list[dict]) -> list[JevResult]:
        return [self.recommend(s) for s in states]


def states_from_frame(frame, cfg: dict) -> list[dict]:
    """Build Jev states for batch routing from the stage-2 observed evidence (FAST + ACCURATE only)."""
    out = []
    for r in frame.itertuples(index=False):
        pf, pa = float(r.p_fast), float(r.p_accurate)
        from .uncertainty import binary_entropy
        out.append(build_state({
            "fast_probability": round(pf, 4), "accurate_probability": round(pa, 4), "deep_probability": None,
            "fast_confidence": round(max(pf, 1 - pf), 4), "model_disagreement": round(abs(pf - pa), 4),
            "entropy": round(float(binary_entropy((pf + pa) / 2)), 4), "risk_score": round(max(pf, pa), 4),
            "transaction_context": {"international": int(r.international), "device_seen_before": int(r.device_seen_before),
                                    "merchant_seen_before": int(r.merchant_seen_before), "hour": int(r.hour),
                                    "transactions_24h": int(r.transactions_24h)}}, cfg))
    return out
