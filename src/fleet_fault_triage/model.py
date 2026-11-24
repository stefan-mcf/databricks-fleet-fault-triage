"""Local development protocol. This module never opens the external test file."""

import hashlib
import json
import subprocess
import time
from itertools import product
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fleet_fault_triage.budget import FitBudget
from fleet_fault_triage.data import digest, save_json

# Only used for objects just fitted here, never for accepting external model files.
TRUSTED_MODEL_TYPES = [
    "numpy.dtype",
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def log_calibrated_model(model, example):
    example = example.astype("float64")
    return mlflow.sklearn.log_model(
        model,
        name="classifier",
        input_example=example,
        signature=infer_signature(example, model.predict_proba(example)),
        pyfunc_predict_fn="predict_proba",
        serialization_format="skops",
        skops_trusted_types=TRUSTED_MODEL_TYPES,
    )


def configurations() -> list[dict]:
    linear = [
        {"kind": "logistic", "C": c, "class_weight": weight}
        for c, weight in product([0.1, 1.0, 10.0], [None, "balanced"])
    ]
    trees = [
        {
            "kind": "histogram_boosting",
            "learning_rate": rate,
            "max_leaf_nodes": leaves,
            "class_weight": weight,
        }
        for rate, leaves, weight in product([0.05, 0.1], [15, 31], [None, "balanced"])
    ]
    return linear + trees


def estimator(config: dict):
    settings = {key: value for key, value in config.items() if key != "kind"}
    if config["kind"] == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42, **settings),
        )
    return HistGradientBoostingClassifier(
        max_iter=200, early_stopping=False, random_state=42, **settings
    )


def decision_curve(y, scores) -> list[dict]:
    """Every distinct decision and the no-flags endpoint, without splitting ties."""
    y, scores = np.asarray(y), np.asarray(scores)
    if not set(np.unique(y)) == {0, 1} or not np.isfinite(scores).all():
        raise ValueError("threshold selection requires two classes and finite scores")
    order = np.argsort(-scores, kind="stable")
    ordered_y, ordered_scores = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(y) - 1]
    tp = np.cumsum(ordered_y)[ends]
    fp = ends + 1 - tp
    fn = y.sum() - tp
    result = [
        {
            "threshold": float(s),
            "tp": int(t),
            "fp": int(f),
            "fn": int(n),
            "tn": int(len(y) - y.sum() - f),
            "cost": int(10 * f + 500 * n),
            "recall": float(t / y.sum()),
            "precision": float(t / (i + 1)),
            "flagged_fraction": float((i + 1) / len(y)),
        }
        for i, t, f, n, s in zip(ends, tp, fp, fn, ordered_scores[ends], strict=True)
    ]
    return [
        {
            "threshold": float(np.nextafter(ordered_scores[0], np.inf)),
            "tp": 0,
            "fp": 0,
            "fn": int(y.sum()),
            "tn": int(len(y) - y.sum()),
            "cost": int(500 * y.sum()),
            "recall": 0.0,
            "precision": None,
            "flagged_fraction": 0.0,
        },
        *result,
    ]


def threshold(y, scores) -> float:
    """Minimum cost with recall >=.85; tied scores always share a decision."""
    return min(
        (point for point in decision_curve(y, scores) if point["recall"] >= 0.85),
        key=lambda p: (p["cost"], -p["recall"], p["flagged_fraction"], -p["threshold"]),
    )["threshold"]


def metrics(y, scores, cutoff: float, coverage: float) -> dict:
    y, scores = np.asarray(y), np.asarray(scores)
    prediction = scores >= cutoff
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    cost = int(10 * fp + 500 * fn)
    constant_cost = int(min(500 * y.sum(), 10 * (len(y) - y.sum())))
    result = {
        "rows": len(y),
        "positives": int(y.sum()),
        "coverage": coverage,
        "threshold": cutoff,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "recall": float(tp / (tp + fn)),
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "flagged_fraction": float(prediction.mean()),
        "cost": cost,
        "cost_per_1000": float(cost * 1000 / len(y)),
        "best_constant_cost": constant_cost,
        "average_precision": float(average_precision_score(y, scores)),
        "brier": float(brier_score_loss(y, scores)),
        "log_loss": float(log_loss(y, scores)),
    }
    result["development_gate_passed"] = (
        result["recall"] >= 0.85 and coverage >= 0.95 and cost <= constant_cost * 0.8
    )
    return result


def paired_cost_interval(y, first_scores, first_threshold, second_scores, second_threshold, groups):
    def costs(scores, cutoff):
        return np.where(
            (scores >= cutoff) & (y == 0), 10, np.where((scores < cutoff) & (y == 1), 500, 0)
        )

    difference = costs(first_scores, first_threshold) - costs(second_scores, second_threshold)
    group_costs = pd.Series(difference).groupby(np.asarray(groups)).sum().to_numpy()
    random = np.random.default_rng(42)
    bootstrap = [
        random.choice(group_costs, size=len(group_costs), replace=True).sum() for _ in range(2000)
    ]
    return [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])]


def source_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "src").rglob("*.py")) + [root / "pyproject.toml", root / "uv.lock"]
    snapshot = hashlib.sha256()
    for path in files:
        snapshot.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        "source_tree_sha256": snapshot.hexdigest(),
        "lock_sha256": digest(root / "uv.lock"),
    }


def train(data_dir: Path, output_dir: Path, resume_cv: Path | None = None) -> dict:
    manifest = json.loads((data_dir / "prepared/manifest.json").read_text())
    source_path = data_dir / "prepared/training.parquet"
    if digest(source_path) != manifest["prepared_sha256"]:
        raise ValueError("prepared data hash does not match frozen manifest")
    frame = pd.read_parquet(source_path)
    features = manifest["features"]
    cohorts = {
        name: frame.loc[frame.partition.eq(name)]
        for name in ("development", "calibration", "decision")
    }
    fitted = {name: group.loc[group.eligible] for name, group in cohorts.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(output_dir / 'mlflow.sqlite').resolve()}")
    mlflow.set_experiment("fleet-fault-triage-local")
    identity = source_identity()
    results = []
    recovered = None
    if resume_cv:
        recovered = json.loads(resume_cv.read_text())
        if recovered["data_manifest_sha256"] != digest(data_dir / "prepared/manifest.json"):
            raise ValueError("recovered comparison uses different data/splits")
        results = recovered["configurations"]
        expected = {json.dumps(c, sort_keys=True) for c in configurations()}
        actual = {json.dumps(r["config"], sort_keys=True) for r in results}
        if actual != expected or len(results) != len(expected):
            raise ValueError("recovery requires every frozen configuration exactly once")
        if not recovered.get("complete") or not all(np.isfinite(r["mean_ap"]) for r in results):
            raise ValueError("comparison recovery is incomplete")
        if (
            not recovered.get("source_run_id")
            or not isinstance(recovered.get("source_tree_sha256"), str)
            or len(recovered["source_tree_sha256"]) != 64
        ):
            raise ValueError("comparison recovery requires original run and source identities")
    expected_fits = 4 if resume_cv else len(configurations()) * 5 + 4
    with (
        FitBudget(data_dir, expected_fits, "train") as budget,
        threadpool_limits(limits=2),
        mlflow.start_run(run_name="training-only-development") as run,
    ):
        mlflow.set_tags({"origin": "local_benchmark_development", **identity})
        mlflow.log_dict(manifest, "data-manifest.json")
        mlflow.log_input(
            mlflow.data.from_pandas(fitted["development"][features]), context="training"
        )
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        dev = cohorts["development"]
        for config in [] if resume_cv else configurations():
            start = time.monotonic()
            scores = []
            with mlflow.start_run(nested=True, run_name=config["kind"]):
                mlflow.log_params(config)
                for train_ids, validation_ids in cv.split(dev, dev.label, dev.group_id):
                    fold_train, fold_validation = dev.iloc[train_ids], dev.iloc[validation_ids]
                    # The fold validation data cannot influence its quality threshold or transforms.
                    limit = fold_train[features].isna().mean(axis=1).quantile(0.99)
                    fold_train = fold_train.loc[fold_train[features].isna().mean(axis=1).le(limit)]
                    fold_validation = fold_validation.loc[
                        fold_validation[features].isna().mean(axis=1).le(limit)
                    ]
                    candidate = budget.fit(
                        estimator(config), fold_train[features], fold_train.label
                    )
                    probabilities = candidate.predict_proba(fold_validation[features])[:, 1]
                    scores.append(
                        float(average_precision_score(fold_validation.label, probabilities))
                    )
                result = {
                    "config": config,
                    "fold_ap": scores,
                    "mean_ap": float(np.mean(scores)),
                    "seconds": time.monotonic() - start,
                }
                mlflow.log_metrics({"mean_ap": result["mean_ap"], "seconds": result["seconds"]})
                mlflow.log_metrics({f"fold_{i}_ap": score for i, score in enumerate(scores)})
                results.append(result)
                save_json(
                    output_dir / "cv-results.json",
                    {
                        "data_manifest_sha256": digest(data_dir / "prepared/manifest.json"),
                        "source_run_id": run.info.run_id,
                        "source_tree_sha256": identity["source_tree_sha256"],
                        "configurations": results,
                        "complete": len(results) == len(configurations()),
                    },
                )
                print(
                    json.dumps({"configuration": config, "mean_ap": result["mean_ap"]}), flush=True
                )
        candidates = {}
        decision = fitted["decision"]
        for kind in ("logistic", "histogram_boosting"):
            selected = max(
                (r for r in results if r["config"]["kind"] == kind), key=lambda r: r["mean_ap"]
            )
            base = budget.fit(
                estimator(selected["config"]),
                fitted["development"][features],
                fitted["development"].label,
            )
            calibrated = budget.fit(
                CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid"),
                fitted["calibration"][features],
                fitted["calibration"].label,
            )
            probabilities = calibrated.predict_proba(decision[features])[:, 1]
            cutoff = threshold(decision.label, probabilities)
            candidates[kind] = {
                "model": calibrated,
                "scores": probabilities,
                "config": selected["config"],
                "uncalibrated": metrics(
                    decision.label,
                    base.predict_proba(decision[features])[:, 1],
                    cutoff,
                    len(decision) / len(cohorts["decision"]),
                ),
                "metrics": metrics(
                    decision.label, probabilities, cutoff, len(decision) / len(cohorts["decision"])
                ),
            }
        linear, tree = candidates["logistic"], candidates["histogram_boosting"]
        interval = paired_cost_interval(
            decision.label.to_numpy(),
            tree["scores"],
            tree["metrics"]["threshold"],
            linear["scores"],
            linear["metrics"]["threshold"],
            decision.group_id,
        )
        chosen = "logistic"
        if tree["metrics"]["development_gate_passed"] and (
            interval[1] < 0 or not linear["metrics"]["development_gate_passed"]
        ):
            chosen = "histogram_boosting"
        winner = candidates[chosen]
        bins = []
        for positions in np.array_split(np.argsort(winner["scores"]), 10):
            bins.append(
                {
                    "rows": len(positions),
                    "mean_score": float(winner["scores"][positions].mean()),
                    "positives": int(decision.label.iloc[positions].sum()),
                }
            )
        # All predictors accept missing values: signatures must not infer non-nullable integers.
        example = fitted["development"][features].head(3).astype("float64")
        logged = log_calibrated_model(winner["model"], example)
        tradeoffs = {
            kind: decision_curve(decision.label, candidate["scores"])
            for kind, candidate in candidates.items()
        }
        save_json(output_dir / "threshold-tradeoffs.json", tradeoffs)
        mlflow.log_dict(tradeoffs, "threshold-tradeoffs.json")
        coverage = len(decision) / len(cohorts["decision"])
        report = {
            "schema_version": 1,
            "origin": "local training-only development; not external evaluation",
            **identity,
            "data_manifest_sha256": digest(data_dir / "prepared/manifest.json"),
            "configurations": results,
            "fit_operations": 4 if resume_cv else len(results) * 5 + 4,
            "cumulative_fit_operations": budget.total,
            "fit_ledger_sha256": budget.ledger_sha256,
            "reused_cv_fits": len(results) * 5 if resume_cv else 0,
            "cv_recovery": recovered.get("recovery_note") if recovered else None,
            "reused_cv_source_run_id": recovered["source_run_id"] if recovered else None,
            "reused_cv_source_tree_sha256": recovered["source_tree_sha256"] if recovered else None,
            "selected_kind": chosen,
            "model_uri": logged.model_uri,
            "mlflow_run_id": run.info.run_id,
            "candidates": {
                k: {
                    "config": c["config"],
                    "metrics": c["metrics"],
                    "uncalibrated_metrics": c["uncalibrated"],
                }
                for k, c in candidates.items()
            },
            "tree_minus_logistic_cost_95_interval": interval,
            "selected_metrics": winner["metrics"],
            "baselines": {
                "always_other": metrics(decision.label, np.zeros(len(decision)), 0.5, coverage),
                "always_APS": metrics(decision.label, np.ones(len(decision)), 0.5, coverage),
                "constant_training_prior": metrics(
                    decision.label,
                    np.full(len(decision), float(fitted["development"].label.mean())),
                    0.5,
                    coverage,
                ),
            },
            "reliability_bins": bins,
            "calibration_support": {
                "rows": len(fitted["calibration"]),
                "positives": int(fitted["calibration"].label.sum()),
            },
            "decision_quality_rejects": {
                "rows": int((~cohorts["decision"].eligible).sum()),
                "positives": int(
                    cohorts["decision"].loc[~cohorts["decision"].eligible, "label"].sum()
                ),
            },
            "threshold_tradeoffs_sha256": digest(output_dir / "threshold-tradeoffs.json"),
            "development_gate_passed": winner["metrics"]["development_gate_passed"],
            "external_test_opened_for_analysis": False,
            "cloud_training_performed": False,
            "release_approved": False,
        }
        mlflow.log_metrics(winner["metrics"])
        # Load persisted model and compare, rather than trusting the in-memory estimator.
        reloaded = mlflow.pyfunc.load_model(logged.model_uri)
        expected = winner["model"].predict_proba(example)
        if not np.allclose(reloaded.predict(example), expected, rtol=0, atol=1e-6):
            raise ValueError("persisted model readback differs")
        report["persisted_model_readback_passed"] = True
        mlflow.log_dict(report, "development-report.json")
        save_json(output_dir / "development-report.json", report)
    return report


def reproduce(data_dir: Path, previous_dir: Path, output_dir: Path) -> dict:
    """Refit only the selected recipe in a fresh process; no repeated search or test access."""
    previous = json.loads((previous_dir / "development-report.json").read_text())
    manifest_path = data_dir / "prepared/manifest.json"
    if digest(manifest_path) != previous["data_manifest_sha256"]:
        raise ValueError("reproduction requires the same data/split manifest")
    manifest = json.loads(manifest_path.read_text())
    if digest(data_dir / "prepared/training.parquet") != manifest["prepared_sha256"]:
        raise ValueError("reproduction data changed")
    frame = pd.read_parquet(data_dir / "prepared/training.parquet")
    features = manifest["features"]
    cohorts = {
        name: frame.loc[frame.partition.eq(name) & frame.eligible]
        for name in ("development", "calibration", "decision")
    }
    selected = previous["candidates"][previous["selected_kind"]]
    with FitBudget(data_dir, 2, "reproduce") as budget, threadpool_limits(limits=2):
        base = budget.fit(
            estimator(selected["config"]),
            cohorts["development"][features],
            cohorts["development"].label,
        )
        model = budget.fit(
            CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid"),
            cohorts["calibration"][features],
            cohorts["calibration"].label,
        )
    # The original MLflow run uses a local SQLite/artifact store, never a provider service.
    mlflow.set_tracking_uri(f"sqlite:///{(previous_dir / 'mlflow.sqlite').resolve()}")
    stored = mlflow.pyfunc.load_model(previous["model_uri"])
    decision = cohorts["decision"][features].astype("float64")
    expected = stored.predict(decision)
    actual = model.predict_proba(decision)
    difference = float(np.max(np.abs(expected - actual)))
    if difference > 1e-6:
        raise ValueError(f"reproduction changed predictions: maximum difference {difference}")
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment("fleet-fault-triage-local")
    with mlflow.start_run(run_name="selected-recipe-reproduction") as run:
        example = cohorts["development"][features].head(3).astype("float64")
        logged = log_calibrated_model(model, example)
        restored = mlflow.pyfunc.load_model(logged.model_uri)
        if not np.allclose(restored.predict(decision.astype(float)), actual, atol=1e-6, rtol=0):
            raise ValueError("reproduced model failed persisted readback")
        report = {
            "schema_version": 1,
            "origin": "local selected-recipe reproduction",
            **source_identity(),
            "original_selection_source_tree_sha256": previous["source_tree_sha256"],
            "data_manifest_sha256": digest(manifest_path),
            "original_selection_report_sha256": digest(previous_dir / "development-report.json"),
            "model_uri": logged.model_uri,
            "mlflow_run_id": run.info.run_id,
            "fit_operations": 2,
            "cumulative_fit_operations": budget.total,
            "fit_ledger_sha256": budget.ledger_sha256,
            "max_prediction_difference": difference,
            "decision_rows": len(decision),
            "persisted_model_readback_passed": True,
            "selected_metrics": metrics(
                cohorts["decision"].label,
                actual[:, 1],
                selected["metrics"]["threshold"],
                len(decision) / int(frame.partition.eq("decision").sum()),
            ),
            "external_test_opened_for_analysis": False,
            "cloud_training_performed": False,
        }
        mlflow.log_dict(report, "reproduction-report.json")
        save_json(output_dir / "reproduction-report.json", report)
    return report
