"""Probability calibration with measured benefit (no blind calibration)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

_CLIP = 1e-6


class ProbabilityCalibrator:
    """Post-hoc calibrator: 'none', 'sigmoid' (Platt on logit) or 'isotonic'."""

    def __init__(self, method: str = "none") -> None:
        if method not in {"none", "sigmoid", "isotonic"}:
            raise ValueError(method)
        self.method = method
        self._model = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, _CLIP, 1 - _CLIP)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        p, y = np.asarray(p, float), np.asarray(y, int)
        if self.method == "sigmoid":
            self._model = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p), y)
        elif self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        if self.method == "sigmoid":
            return self._model.predict_proba(self._logit(p))[:, 1]
        if self.method == "isotonic":
            return np.clip(self._model.predict(p), 0.0, 1.0)
        return p


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Equal-width ECE: weighted mean |observed rate - mean predicted| over probability bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability-diagram data (equal-width bins)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "bin_lo": b / bins, "bin_hi": (b + 1) / bins, "count": int(m.sum()),
                         "mean_predicted": float(p[m].mean()), "observed_rate": float(y[m].mean())})
    return pd.DataFrame(rows)


def choose_calibrator(p_val: np.ndarray, y_val: np.ndarray, seed: int, min_gain: float = 1e-5) -> tuple[ProbabilityCalibrator, dict]:
    """Pick none/sigmoid/isotonic by 2-fold cross-validated Brier on the VALIDATION split, then refit on all of it.

    A calibrator is kept only if it improves held-out Brier by more than ``min_gain``.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(p_val))
    folds = [perm[::2], perm[1::2]]
    scores: dict[str, float] = {}
    for method in ("none", "sigmoid", "isotonic"):
        briers = []
        for i in range(2):
            tr, te = folds[i], folds[1 - i]
            cal = ProbabilityCalibrator(method).fit(p_val[tr], y_val[tr])
            briers.append(brier_score_loss(y_val[te], cal.transform(p_val[te])))
        scores[method] = float(np.mean(briers))
    best = min(("sigmoid", "isotonic"), key=lambda m: scores[m])
    chosen = best if scores["none"] - scores[best] > min_gain else "none"
    cal = ProbabilityCalibrator(chosen).fit(p_val, y_val)
    return cal, {"cv_brier": scores, "chosen": chosen}
