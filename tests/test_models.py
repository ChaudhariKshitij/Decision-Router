import numpy as np
import pandas as pd
import pytest

from src.calibration import ProbabilityCalibrator, expected_calibration_error
from src.config import TIERS
from src.data import generate_transactions
from src.features import FEATURE_COLUMNS, build_features
from src.registry import ModelRegistry


def test_data_is_deterministic_and_imbalanced(cfg):
    a = generate_transactions(2000, 7)
    b = generate_transactions(2000, 7)
    pd.testing.assert_frame_equal(a, b)
    assert 0.03 < a.fraud.mean() < 0.12


def test_splits_stratified_and_disjoint(splits):
    ids = [set(s.transaction_id) for s in splits.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    rates = [s.fraud.mean() for s in splits.values()]
    assert max(rates) - min(rates) < 0.01


def test_feature_generation(splits):
    X = build_features(splits["test"])
    assert list(X.columns) == FEATURE_COLUMNS
    assert not X.isna().any().any() and np.isfinite(X.to_numpy()).all()


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        build_features(pd.DataFrame({"amount": [1.0]}))


@pytest.mark.parametrize("tier", TIERS)
def test_predict_proba_shape_and_range(registry, splits, tier):
    X = build_features(splits["test"].head(200))
    p = registry.models[tier].predict_proba(X)
    assert p.shape == (200, 2)
    assert (p >= 0).all() and (p <= 1).all() and np.allclose(p.sum(1), 1)


def test_models_beat_chance(registry, splits):
    from sklearn.metrics import roc_auc_score
    te = splits["test"]
    X = build_features(te)
    for t in TIERS:
        assert roc_auc_score(te.fraud, registry.models[t].proba(X)) > 0.75


def test_registry_costs_and_roundtrip(registry, tmp_path, cfg):
    assert [registry.models[t].cost for t in TIERS] == [cfg["cost"][t] for t in TIERS]
    registry.save(tmp_path)
    loaded = ModelRegistry.load(tmp_path)
    X = build_features(generate_transactions(50, 3))
    np.testing.assert_allclose(loaded.models["fast"].proba(X), registry.models["fast"].proba(X))
    assert loaded.fraud_threshold == registry.fraud_threshold
    assert all(m["version"] for m in loaded.info())


def test_registry_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelRegistry.load(tmp_path / "nope")


def test_calibrators_reduce_miscalibration():
    rng = np.random.default_rng(0)
    true_p = rng.beta(1, 12, 6000)
    y = (rng.random(6000) < true_p).astype(int)
    overconfident = np.clip(true_p * 3, 0, 1)
    cal = ProbabilityCalibrator("isotonic").fit(overconfident[:3000], y[:3000])
    assert expected_calibration_error(y[3000:], cal.transform(overconfident[3000:])) < expected_calibration_error(y[3000:], overconfident[3000:])


def test_corrupt_model_file_gives_actionable_error(registry, tmp_path):
    registry.save(tmp_path)
    (tmp_path / "accurate.joblib").write_bytes(b"not a pickle")
    with pytest.raises(RuntimeError, match="scripts/train.py"):
        ModelRegistry.load(tmp_path)
