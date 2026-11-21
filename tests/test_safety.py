import numpy as np

from src.safety import apply_safety


def run(sel, risk, dis, cfg, ent=None):
    return apply_safety(np.array(sel, dtype=object), np.array(risk), np.array(dis), cfg, None if ent is None else np.array(ent))


def test_high_risk_forces_deep(cfg):
    r = run(["fast", "accurate"], [0.95, 0.95], [0.0, 0.0], cfg)
    assert list(r.final) == ["deep", "deep"] and r.override.all() and set(r.reason) == {"high_risk"}


def test_high_disagreement_forces_deep(cfg):
    r = run(["fast"], [0.1], [0.5], cfg)
    assert r.final[0] == "deep" and r.reason[0] == "high_disagreement"


def test_no_override_when_safe_and_deep_not_counted(cfg):
    r = run(["fast", "deep"], [0.1, 0.99], [0.0, 0.0], cfg)
    assert r.final[0] == "fast" and not r.override[0]
    assert r.final[1] == "deep" and not r.override[1]      # already deep => not an override


def test_never_downgrades_and_is_deterministic(cfg):
    rng = np.random.default_rng(1)
    sel = rng.choice(["fast", "accurate", "deep"], 500).astype(object)
    risk, dis = rng.random(500), rng.random(500)
    a, b = apply_safety(sel, risk, dis, cfg), apply_safety(sel, risk, dis, cfg)
    assert (a.final == b.final).all()
    rank = {"fast": 0, "accurate": 1, "deep": 2}
    assert all(rank[f] >= rank[s] for f, s in zip(a.final, sel))


def test_optional_entropy_rule(cfg):
    import copy
    c = copy.deepcopy(cfg); c["safety"]["high_entropy"] = 0.9
    r = run(["fast"], [0.5], [0.0], c, ent=[0.99])
    assert r.final[0] == "deep" and r.reason[0] == "high_uncertainty"
