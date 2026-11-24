import numpy as np
import pandas as pd
import pytest

from fleet_fault_triage.model import configurations, estimator, metrics, threshold


def test_threshold_keeps_ties_together_and_matches_exhaustive_cost():
    labels = np.array([0, 1, 1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.8, 0.3, 0.2, 0.1])
    selected = threshold(labels, scores)
    costs = []
    for candidate in np.unique(scores):
        report = metrics(labels, scores, candidate, 1.0)
        if report["recall"] >= 0.85:
            costs.append(report["cost"])
    assert metrics(labels, scores, selected, 1.0)["cost"] == min(costs)


def test_missing_class_is_not_silently_assigned_a_quality_score():
    with pytest.raises(ValueError, match="two classes"):
        threshold(np.zeros(5), np.linspace(0, 1, 5))


def test_coverage_can_block_a_good_classifier():
    report = metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5, 0.8)
    assert report["cost"] == 0
    assert report["development_gate_passed"] is False


def test_imputation_fits_training_only_and_preserves_empty_feature():
    model = estimator({"kind": "logistic", "C": 1, "class_weight": None})
    training = pd.DataFrame({"a": [1.0, 3.0, 5.0, 7.0], "empty": [np.nan] * 4})
    model.fit(training, [0, 0, 1, 1])
    before = model[0].statistics_.copy()
    model.predict_proba(pd.DataFrame({"a": [10000.0, np.nan], "empty": [0.0, np.nan]}))
    np.testing.assert_array_equal(model[0].statistics_, before)
    assert before[0] == 4.0


def test_candidate_grid_is_bounded():
    assert len(configurations()) == 14
    assert len(configurations()) * 5 + 4 <= 100


def test_undefined_baseline_precision_is_explicit():
    report = metrics([0, 1], [0.0, 0.0], 0.5, 1.0)
    assert report["precision"] is None
    assert report["cost"] == 500


def test_calibrated_tree_saved_as_skops_loads_and_accepts_nullable_floats(tmp_path):
    import mlflow
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    from fleet_fault_triage.model import log_calibrated_model

    random = np.random.default_rng(42)
    features = pd.DataFrame(random.normal(size=(120, 3)), columns=["a", "b", "c"])
    labels = (features.a + features.b > 0).astype(int)
    tree = estimator(
        {
            "kind": "histogram_boosting",
            "learning_rate": 0.1,
            "max_leaf_nodes": 7,
            "class_weight": None,
        }
    )
    tree.set_params(max_iter=3, min_samples_leaf=2)
    tree.fit(features.iloc[:80], labels.iloc[:80])
    model = CalibratedClassifierCV(FrozenEstimator(tree), method="sigmoid").fit(
        features.iloc[80:], labels.iloc[80:]
    )
    previous_tracking = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.sqlite'}")
        mlflow.set_experiment("synthetic-serialization-test")
        with mlflow.start_run():
            logged = log_calibrated_model(model, features.head(3))
            loaded = mlflow.pyfunc.load_model(logged.model_uri)
            probe = features.head(3).copy()
            probe.loc[0, "a"] = np.nan
            np.testing.assert_allclose(loaded.predict(probe), model.predict_proba(probe), atol=1e-6)
    finally:
        mlflow.set_tracking_uri(previous_tracking)
