"""Tests for the XGBoost-assisted non-mirrored search helpers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import xgboost as xgb
from mpmath import mp

from experiments import non_mirrored_surrogate_search as surrogate
from solver.two_sided_shooting import TwoSidedSearchPoint


def test_training_seed_generation_has_exact_quotas() -> None:
    """The surrogate training design should be reproducible and exactly sized."""
    with mp.workdps(50):
        left = surrogate._training_seeds(1729)
        right = surrogate._training_seeds(1729)
        counts = Counter(seed.region for seed in left)
        assert left == right
        assert len(left) == 5000
        assert counts == {"near": 700, "medium": 1100, "far": 1200, "asymmetric": 1400, "wide": 600}
        assert all(seed.point.s > surrogate.S_MIN for seed in left)


def test_feature_rows_are_finite_and_named() -> None:
    """Scaled points should become finite feature arrays with one row per point."""
    points = [
        TwoSidedSearchPoint(mp.zero, mp.zero, mp.zero, mp.zero, mp.zero, mp.zero, mp.zero),
        TwoSidedSearchPoint(mp.mpf("0.2"), mp.mpf("-0.1"), mp.mpf("0.3"), mp.mpf("-0.2"), mp.mpf("0.1"), mp.mpf("-0.3"), mp.mpf("0.4")),
    ]
    features = surrogate._features_for_points(points)
    assert features.shape == (2, len(surrogate.FEATURE_NAMES))
    assert np.isfinite(features).all()


def test_xgboost_classifier_and_regressor_smoke(tmp_path: Path) -> None:
    """XGBoost models should train, save, reload, and predict on tiny data."""
    x = np.asarray([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5]], dtype=np.float64)
    y_class = np.asarray([0, 0, 1, 1], dtype=np.int32)
    y_reg = np.asarray([2.0, 1.0, -1.0, -2.0], dtype=np.float64)
    classifier = xgb.train({"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 1}, xgb.DMatrix(x, label=y_class), num_boost_round=3)
    regressor = xgb.train({"objective": "reg:squarederror", "max_depth": 1}, xgb.DMatrix(x, label=y_reg), num_boost_round=3)
    classifier_path = tmp_path / "classifier.json"
    regressor_path = tmp_path / "regressor.json"
    classifier.save_model(classifier_path)
    regressor.save_model(regressor_path)
    loaded_classifier = xgb.Booster()
    loaded_regressor = xgb.Booster()
    loaded_classifier.load_model(classifier_path)
    loaded_regressor.load_model(regressor_path)
    assert loaded_classifier.predict(xgb.DMatrix(x)).shape == (4,)
    assert loaded_regressor.predict(xgb.DMatrix(x)).shape == (4,)


def test_json_events_round_trip(tmp_path: Path) -> None:
    """Surrogate JSONL events should be parseable one event per line."""
    path = tmp_path / "surrogate.jsonl"
    event = surrogate._event("model_metrics", surrogate._model_metrics_payload(surrogate.SurrogateModels(surrogate.ConstantBranchModel(1.0), (surrogate.ConstantResidualModel(-1.0),), {"score": 1.25}), "test"))
    surrogate._write_jsonl_event(path, event)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["event"] == "model_metrics"
    assert loaded["phase"] == "test"
    assert loaded["metrics"]["score"] == 1.25
