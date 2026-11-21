"""Deterministic safety gate. Pure function of (selection, risk, disagreement, entropy) and config.

It can only ESCALATE to the deep tier. It never consults a model or an LLM, and it is applied after
any router (local or Jev), so no routing component can bypass it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyResult:
    final: np.ndarray
    override: np.ndarray       # bool
    reason: np.ndarray        # object: None or 'high_risk' | 'high_disagreement' | 'high_uncertainty'


def apply_safety(selection: np.ndarray, risk: np.ndarray, disagreement: np.ndarray, cfg: dict,
                 entropy: np.ndarray | None = None) -> SafetyResult:
    s = cfg["safety"]
    sel = np.asarray(selection, dtype=object).copy()
    trig_risk = np.asarray(risk) >= s["high_risk"]
    trig_dis = np.asarray(disagreement) >= s["high_disagreement"]
    trig_unc = np.zeros(len(sel), dtype=bool)
    if s.get("high_entropy") is not None and entropy is not None:
        trig_unc = np.asarray(entropy) >= s["high_entropy"]
    reason = np.full(len(sel), None, dtype=object)
    reason[trig_unc] = "high_uncertainty"
    reason[trig_dis] = "high_disagreement"
    reason[trig_risk] = "high_risk"          # highest priority
    override = (trig_risk | trig_dis | trig_unc) & (sel != "deep")
    final = np.where(trig_risk | trig_dis | trig_unc, "deep", sel).astype(object)
    reason = np.where(override, reason, None)
    return SafetyResult(final=final, override=override, reason=reason)
