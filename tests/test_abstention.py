import numpy as np

from src.abstention import coverage_error_curve, decide


def test_review_on_low_margin_or_high_disagreement(cfg):
    thr = 0.3
    d, why = decide(np.array([0.31, 0.9, 0.05, 0.9]), np.array([0.0, 0.0, 0.0, 0.9]), thr, cfg)
    assert list(d) == ["REVIEW", "FRAUD", "LEGIT", "REVIEW"]
    assert why[0] == "low_margin" and why[3] == "high_disagreement"


def test_selective_risk_curve_is_monotone_for_informative_uncertainty():
    rng = np.random.default_rng(0)
    n = 4000
    u = rng.random(n)
    y = (rng.random(n) < 0.3).astype(int)
    flip = rng.random(n) < u * 0.6            # errors concentrate where uncertainty is high
    p = np.where(flip, 1 - y, y).astype(float)
    c = coverage_error_curve(y, p, 0.5, u)
    assert c.error_rate.iloc[0] < c.error_rate.iloc[-1]
    assert c.coverage.iloc[-1] == 1.0
