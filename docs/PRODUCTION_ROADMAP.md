# Production roadmap
**CURRENT IMPLEMENTATION** (this repo) vs **PRODUCTION FUTURE STATE** (not built).

| Area | Current implementation | Production future state |
|---|---|---|
| Data | Synthetic generator, seeded CSVs | Real labelled transactions with delayed/biased labels; label-latency handling; privacy review |
| Features | Pure functions in `src/features.py` shared by train/serve | Feature store with point-in-time correctness and online/offline parity checks |
| Model registry | Local joblib + `registry.json` + version strings | Versioned registry (e.g. MLflow-style) with lineage, approvals, rollback |
| Monitoring | In-process counters (`/metrics`), JSONL audit | Prometheus/OpenTelemetry metrics, per-tier and per-segment dashboards |
| Drift | Offline synthetic shift experiment | PSI/KS on inputs and scores, calibration drift, disagreement-rate alarms, automated retrain triggers |
| Latency | Proxy values + single-row wall clock | p50/p95/p99 under load, batching, autoscaling, distributed inference, tail-latency budgets |
| Deployment | Docker, local uvicorn | Canary + shadow deployments of new routers/models, staged rollout, kill switch to fixed-model fallback |
| Security | None (local) | AuthN/Z, rate limiting, secret manager for `TYPESAFE_API_KEY`, input size limits, audit-log integrity |
| Safety | Deterministic gate + REVIEW | Policy owned by risk/compliance, change-controlled thresholds, human-review queue with SLAs, feedback labels |
| Feedback loop | None | Reviewer outcomes + chargebacks feed retraining and threshold re-optimisation; counterfactual evaluation for routed-away tiers |
| Jev | Optional adapter, shadow mode, fake-client tests, **no measured results** | Run shadow for weeks, compare cost/quality/agreement with CIs, set latency/cost budgets for the extra call, circuit breaker |
| Cost model | Static proxy units | Measured $/1k predictions and latency per tier on target hardware |
| Alerting | None | Alerts on fallback rate, override rate, REVIEW rate, drift, error budget burn |
