# Interview story — DecisionRouter Pro
(Numbers from `artifacts/`; synthetic data, proxy costs.)

## 30 seconds
I built a router that decides, per transaction, whether a cheap, medium or expensive fraud model should score it, using confidence, entropy and model disagreement, with a deterministic safety gate and a REVIEW outcome. I didn't assume it helps: I benchmarked it against fixed-model baselines. The risk-aware router cost 1.93 proxy units vs 8 for always-deep with statistically indistinguishable PR-AUC — and the benchmark also showed the middle model beats the deep one, which is an honest finding about where the money should go.

## 2 minutes
Why routing: most traffic is easy, so a 1-unit model that is confident is enough. Why three models: logistic regression (cheap, interpretable), gradient boosting (nonlinear interactions), MLP (most compute). Why not always deep: it cost 8 units and, here, did not beat the 3-unit model. Evidence for escalation: FAST confidence `max(p,1-p)`, binary entropy, and disagreement between tiers; plus cheap rule-based risk flags. Cost-honesty: cascades pay for every tier they run, so a full escalation (12 units) costs more than deep alone; savings exist only because escalations are rare (7.1% to deep). Safety: a pure deterministic function can only escalate to deep on high risk/disagreement; Jev, if enabled, is only allowed to pick the tier and sits before the gate. Evaluation: validation for calibration/thresholds, one-shot test, paired bootstrap CIs, slices, drift.

## 5-minute deep dive
1. **Data**: seeded generator with nonlinear terms, interactions, unobserved latent noise and label flips so there's real irreducible error (oracle ROC-AUC ≈ 0.96).
2. **Models & calibration**: 3 tiers; calibration (none/Platt/isotonic) chosen per model by validation CV Brier and only kept if it helps; ECE reported before/after.
3. **Uncertainty**: entropy peaks at 0.5 (concavity, derivative zero); disagreement is a heuristic, not epistemic uncertainty. Measured: disagreement predicts FAST errors with AUC 0.88, but FAST's own entropy does about as well — so I don't claim disagreement is uniquely valuable.
4. **Policies**: fixed baselines, confidence cascade, risk-aware cascade, Jev. Thresholds in YAML, picked via a validation sweep (cheapest within 1% of best PR-AUC).
5. **Safety gate**: overrides recorded with reasons; ablation shows its metric effect is small (0.65% of transactions) — its value is a guarantee, not a score.
6. **Abstention**: REVIEW on 2.2% of traffic, selective-risk curve.
7. **Stress & drift**: adversarial profiles route to deep (86%), benign goes to fast; disagreement rises under shift (0.036 → 0.055).
8. **Limitations**: synthetic data, ~300 test frauds (noisy PR-AUC), proxy costs, small MLP, Jev unmeasured.
9. **Production changes**: see `PRODUCTION_ROADMAP.md`.

## Q&A
**Why Jev?** It returns typed, bounded decisions (a `Choice` among fast/accurate/deep), which fits a routing question and keeps free-form text out of control flow. It is optional, shadow-evaluated, and gated.
**Why not just use an LLM?** Latency/cost per call, nondeterminism, and no guarantees; the deterministic gate and local router mean correctness never depends on it. Whether a decision model adds value is an empirical question I leave to shadow-mode data.
**How do you know routing is better?** Fixed baselines on the same held-out data, cascade cost fully counted, paired bootstrap CIs, ablations, slices and shift. Result: much cheaper at statistically indistinguishable PR-AUC vs deep; *not* shown better than always-accurate on quality (ΔPR-AUC CI spans 0, recall lower).
**How would you prevent unsafe routing?** Deterministic gate after every router, escalate-only, configuration under change control, fixed-model fallback, alert on override/fallback rates.
**How would you monitor it in production?** Per-tier traffic mix, cost/latency percentiles, override/fallback/REVIEW rates, score and disagreement drift, calibration on delayed labels.
**What if Jev is unavailable?** `JevResult(choice=None)` → local risk-aware router; the audit shows `fallback_used`, `jev_status`, `jev_error`; tested for disabled, missing key, timeout, error, invalid answer.
**Detecting drift?** PSI/KS on features and scores, rising disagreement, calibration slope/ECE on matured labels; my synthetic experiment shows disagreement and DEEP share rising under shift.
**Calibrating probabilities?** Platt or isotonic on a held-out split, verified by CV Brier/ECE, re-fit after drift; here Platt/isotonic were chosen where they helped (FAST raw ECE was far worse).
**Optimising thresholds?** Sweep on validation, minimise expected cost subject to a quality constraint (done by a 1% rule here); in production use real cost of misses vs review vs compute and re-tune per segment.
