import numpy as np

from src.uncertainty import Evidence, binary_entropy, confidence, disagreement


def test_entropy_max_at_half_and_zero_at_edges():
    p = np.linspace(0.001, 0.999, 999)
    h = binary_entropy(p)
    assert abs(p[np.argmax(h)] - 0.5) < 0.002 and abs(h.max() - 1.0) < 1e-3
    assert binary_entropy(np.array([0.0, 1.0])).max() < 1e-9


def test_confidence_and_disagreement():
    np.testing.assert_allclose(confidence(np.array([0.2, 0.9, 0.5])), [0.8, 0.9, 0.5])
    d = disagreement(np.array([0.51]), np.array([0.81]), np.array([0.92]))
    assert abs(d[0] - 0.41) < 1e-9


def test_evidence_properties():
    ev = Evidence(np.array([0.1]), np.array([0.5]), np.array([0.9]))
    assert ev.disagreement[0] == 0.8 and ev.risk_score[0] == 0.9
    assert ev.select_prob(np.array(["deep"], dtype=object))[0] == 0.9
