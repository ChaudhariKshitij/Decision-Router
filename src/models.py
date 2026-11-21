"""The three model tiers. All are sklearn-compatible and expose predict_proba."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_fast(seed: int) -> Pipeline:
    """FAST: scaler + class-weighted logistic regression (interpretable, linear)."""
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=seed))])


def build_accurate(seed: int, quick: bool = False) -> HistGradientBoostingClassifier:
    """ACCURATE: histogram gradient boosting (captures nonlinearities and interactions)."""
    return HistGradientBoostingClassifier(
        max_iter=60 if quick else 400, learning_rate=0.05, max_leaf_nodes=24, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=seed)


def build_deep(seed: int, quick: bool = False) -> Pipeline:
    """DEEP: scaled MLP (64-32 ReLU), L2 regularisation + early stopping (sklearn has no dropout)."""
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
                                           early_stopping=True, validation_fraction=0.15, n_iter_no_change=12,
                                           max_iter=60 if quick else 300, random_state=seed))])


def fit_estimators(X_train, y_train, seed: int, quick: bool = False) -> dict:
    """Fit all three tiers on the TRAIN split only."""
    est = {"fast": build_fast(seed), "accurate": build_accurate(seed, quick), "deep": build_deep(seed, quick)}
    for m in est.values():
        m.fit(X_train, y_train)
    return est


def raw_fraud_proba(estimator, X) -> np.ndarray:
    """P(fraud) as a 1-D array."""
    return estimator.predict_proba(X)[:, 1]
