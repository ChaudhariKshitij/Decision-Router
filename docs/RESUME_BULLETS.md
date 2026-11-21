# Resume bullets
All figures come from `artifacts/` (synthetic data, proxy costs, held-out test split of 4,800 transactions / 297 frauds). No production traffic, real fraud detection or real monetary savings are claimed.

## ML Engineer version
- Built a cost-aware inference router over three model tiers (logistic regression, gradient boosting, MLP); a cascade with entropy/disagreement-based escalation cut average proxy inference cost from 8 to 1.93 units (76% lower, cascade cost fully counted) while PR-AUC stayed at 0.617 vs 0.617 for always-deep (paired-bootstrap ΔPR-AUC 95% CI [-0.009, +0.013]).
- Evaluated routing against fixed-model baselines with calibration (Platt/isotonic chosen by validation CV), bootstrap CIs, threshold sweeps / Pareto frontier, slice and distribution-shift tests; found always-accurate (0.633 PR-AUC at 3 units) dominates always-deep, i.e. reported where the expensive tier does *not* pay off.
- Shipped FastAPI (`/route`, `/route/batch`, `/shadow-route`, audit lookup), Streamlit dashboard, Docker/CI, and a 52-test pytest suite; JSONL audit trace per decision.

## AI Engineer version
- Designed a bounded AI decision layer: TypeSafe Jev (typed `Choice` question via the official Python SDK) may only select a model tier from structured evidence; a deterministic safety gate has final authority and escalates high-risk / high-disagreement cases to the deep model regardless of the router.
- Implemented failure-proof fallback (disabled, missing key, timeout, SDK error, invalid answer → local router) and a shadow mode that records Jev-vs-production agreement without influencing outcomes; verified with fake-client tests. Jev itself has not been benchmarked yet (no credentials), and the report says so.
- Built per-request auditability: request id, per-tier probabilities, uncertainty signals, override reason, cost, and a deterministic natural-language explanation generated from the trace (no LLM in the explanation path).

## Quant / Finance version
- Framed model selection as a cost–risk trade-off: risk-aware cascade resolves 87% of transactions with the cheapest model and escalates 7.1% to the deepest, using confidence, binary entropy and ensemble disagreement as uncertainty signals (disagreement predicts FAST-tier errors with AUC 0.88; FAST entropy alone is comparable).
- Added selective prediction (REVIEW outcome on 2.2% of transactions) and measured the coverage–error trade-off; stress-tested adversarial profiles (86% routed to the deep tier, 4 safety-gate overrides) and covariate/mechanism shift (mean model disagreement 0.036 → 0.055).
- Validated with calibration error, Brier/log-loss and PR-AUC/recall emphasis for class imbalance (~6% fraud), with thresholds tuned on validation and evaluated once on test.
