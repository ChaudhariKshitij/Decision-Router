"""Deterministic explanations (no LLM): routing rationale from the audit record, feature contributions from the FAST model."""
from __future__ import annotations

import numpy as np
import pandas as pd



def explain_routing(rec: dict, cfg: dict, fraud_threshold: float) -> list[str]:
    """Why was this tier selected? Built only from fields in the audit record and the config thresholds."""
    ra, conf, s = cfg["routing"]["risk_aware"], cfg["routing"]["confidence"], cfg["safety"]
    out: list[str] = []
    pol, init, final = rec["policy"], rec["initial_model"], rec["selected_model"]
    fc = rec["fast_confidence"]
    if pol in {"fast", "accurate", "deep"}:
        out.append(f"Fixed baseline policy: every transaction uses {pol.upper()} (no routing, no safety gate).")
    elif pol == "confidence":
        if init == "fast":
            out.append(f"FAST confidence {fc:.3f} >= {conf['fast_threshold']}: FAST is sufficient.")
        else:
            out.append(f"FAST confidence {fc:.3f} < {conf['fast_threshold']}: escalated to ACCURATE.")
            if init == "deep":
                out.append(f"ACCURATE confidence was also below {conf['accurate_threshold']}: escalated to DEEP.")
    else:
        if init == "fast":
            out.append(f"FAST confidence {fc:.3f} >= {ra['fast_confidence']}, low entropy and low heuristic risk: FAST is sufficient.")
        else:
            if fc < ra["fast_confidence"]:
                out.append(f"FAST confidence {fc:.3f} was below {ra['fast_confidence']}.")
            else:
                out.append("FAST was confident but entropy or rule-based risk flags required a second opinion.")
            pa, pf = rec.get("accurate_probability"), rec.get("fast_probability")
            if init == "deep" and pa is not None and pf is not None:
                if max(pa, pf) >= ra["escalate_risk"]:
                    out.append(f"Risk {max(pa, pf):.2f} >= {ra['escalate_risk']}: escalated to DEEP.")
                if abs(pa - pf) >= ra["escalate_disagreement"]:
                    out.append(f"FAST/ACCURATE disagreement {abs(pa - pf):.2f} >= {ra['escalate_disagreement']}: escalated to DEEP.")
                if len(out) == 1:
                    out.append("Entropy of the FAST/ACCURATE mean exceeded the escalation threshold: escalated to DEEP.")
            elif init == "accurate":
                out.append("Moderate uncertainty: ACCURATE resolved it without needing DEEP.")
    if rec.get("safety_override"):
        reason = rec["safety_override"]
        detail = {"high_risk": f"risk {rec['risk_score']:.2f} >= {s['high_risk']}",
                  "high_disagreement": f"disagreement {rec['model_disagreement']:.2f} >= {s['high_disagreement']}",
                  "high_uncertainty": f"entropy {rec['entropy']:.2f} >= {s['high_entropy']}"}[reason]
        out.append(f"SAFETY GATE ({reason}): {detail}. Router chose {init.upper()}; deterministic policy escalated to DEEP.")
    if rec.get("policy_source") == "local_fallback":
        out.append(f"Jev unavailable ({rec.get('jev_status')}): local deterministic router used.")
    elif rec.get("policy_source") == "jev":
        out.append("Jev recommended this tier from structured evidence; the safety gate was applied afterwards.")
    d = rec["decision"]
    if d == "REVIEW":
        why = {"low_margin": f"fraud probability is within the abstention margin of the decision threshold ({fraud_threshold:.3f})",
               "high_disagreement": f"model disagreement >= {cfg['abstention']['review_disagreement']} even after routing"}
        out.append(f"Decision REVIEW: {why.get(rec.get('review_reason'), 'uncertain')}.")
    else:
        out.append(f"Decision {d}: P(fraud)={rec['fraud_probability']:.3f} vs threshold {fraud_threshold:.3f}.")
    return out


def feature_contributions(registry, txn: dict, top: int = 6) -> dict:
    """Standardised-logit contributions from the FAST (logistic regression) model + rule-based red flags."""
    fb = registry.feature_builder
    names = list(fb.feature_names)
    X = fb.transform(pd.DataFrame([txn]))
    pipe = registry.models["fast"].estimator
    z = pipe.named_steps["scaler"].transform(X)[0] * pipe.named_steps["clf"].coef_[0]
    order = np.argsort(-np.abs(z))[:top]
    return {"contributions": [{"feature": names[i], "value": float(X.iloc[0, i]), "logit_contribution": float(z[i])} for i in order],
            "risk_flags": fb.risk_flags(txn),
            "note": "Contributions explain the FAST logistic model only (coef x standardised value)."}
