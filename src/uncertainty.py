"""Uncertainty signals computed from per-model fraud probabilities.

* confidence      = max(p, 1-p)                       (0.5 = coin flip, 1.0 = certain)
* binary entropy  H(p) = -p log2 p - (1-p) log2(1-p)  (bits; 0 at p in {0,1}, 1 at p = 0.5)
  H is maximal at p = 0.5 because that is the point where both outcomes are equally likely; the
  derivative H'(p) = log2((1-p)/p) vanishes only there and H is concave.
* disagreement    = max(p_i) - min(p_i) across tiers. A practical ensemble-disagreement HEURISTIC,
  not a calibrated epistemic-uncertainty estimator.
* risk_score      = max(p_i). Deliberately conservative: any tier raising an alarm raises the risk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-12


def confidence(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.maximum(p, 1.0 - p)


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Binary entropy in bits."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def disagreement(*probs: np.ndarray) -> np.ndarray:
    stack = np.vstack([np.asarray(p, dtype=float) for p in probs])
    return stack.max(axis=0) - stack.min(axis=0)


@dataclass
class Evidence:
    """Per-transaction model evidence (arrays of equal length)."""
    p_fast: np.ndarray
    p_accurate: np.ndarray
    p_deep: np.ndarray
    heuristic_risk: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        self.p_fast, self.p_accurate, self.p_deep = (np.asarray(a, dtype=float) for a in (self.p_fast, self.p_accurate, self.p_deep))
        if self.heuristic_risk is None or len(self.heuristic_risk) == 0:
            self.heuristic_risk = np.zeros(len(self.p_fast))

    def __len__(self) -> int:
        return len(self.p_fast)

    @property
    def fast_confidence(self) -> np.ndarray:
        return confidence(self.p_fast)

    @property
    def accurate_confidence(self) -> np.ndarray:
        return confidence(self.p_accurate)

    @property
    def entropy_fast(self) -> np.ndarray:
        return binary_entropy(self.p_fast)

    @property
    def mean_prob(self) -> np.ndarray:
        return (self.p_fast + self.p_accurate + self.p_deep) / 3.0

    @property
    def entropy_mean(self) -> np.ndarray:
        return binary_entropy(self.mean_prob)

    @property
    def disagreement(self) -> np.ndarray:
        return disagreement(self.p_fast, self.p_accurate, self.p_deep)

    @property
    def risk_score(self) -> np.ndarray:
        return np.maximum.reduce([self.p_fast, self.p_accurate, self.p_deep])

    def select_prob(self, selection: np.ndarray) -> np.ndarray:
        """Probability of the tier chosen for each row."""
        sel = np.asarray(selection)
        return np.where(sel == "fast", self.p_fast, np.where(sel == "accurate", self.p_accurate, self.p_deep))

    def subset(self, idx) -> "Evidence":
        return Evidence(self.p_fast[idx], self.p_accurate[idx], self.p_deep[idx], self.heuristic_risk[idx])
