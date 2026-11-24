"""Explicit one-shot external evaluation and identity-bound assistant outcome checks."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fleet_fault_triage.contracts import identity
from fleet_fault_triage.data import TEST, clean, digest
from fleet_fault_triage.model import metrics


def wilson(successes, total):
    if not total:
        return None
    p, z = successes / total, 1.959963984540054
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(centre - half), float(centre + half)]


def external_reference(release, freeze, training_path):
    if training_path is None or digest(training_path) != freeze.get("training_sha256"):
        raise ValueError("exact frozen training reference required before test consumption")
    if digest(training_path.parent / "manifest.json") != release.data_sha256:
        raise ValueError("training reference manifest differs from frozen release")
    return set(pd.read_parquet(training_path, columns=["group_id"]).group_id)


def evaluate_external(data_dir: Path, release, model, store, freeze, guard, *, training_path=None):
    if release.origin != "azure":
        raise ValueError("external evaluation requires an Azure release")
    if freeze.get("release_sha256") != release.sha256 or freeze.get("parity_passed") is not True:
        raise ValueError("exact frozen model and verified Azure parity are required")
    if freeze.get("external_evaluation_authorized") is not True:
        raise ValueError("external evaluation is not approved")
    training_groups = external_reference(release, freeze, training_path)
    guard("external_evaluation")
    path = data_dir / "raw" / TEST
    # Hashing bytes is permitted before consuming the test; parsing starts only after reservation.
    test_hash = digest(path)
    if test_hash != freeze.get("external_test_sha256"):
        raise ValueError("sealed test bytes changed")
    store.reserve_evaluation(release, test_hash)
    with path.open() as stream:
        for skip, line in enumerate(stream):
            if line.startswith("class,"):
                header = next(csv.reader([line]))
                break
        else:
            raise ValueError("external header missing; evaluation remains consumed")
    if header != ["class", *release.features]:
        raise ValueError("external schema differs from frozen features")
    raw = pd.read_csv(path, skiprows=skip, dtype=str, keep_default_na=False)
    if len(raw) != 16_000:
        raise ValueError("unexpected external population; evaluation remains consumed")
    frame, quarantined = clean(raw, test_hash)
    eligible = frame[release.features].isna().mean(axis=1).le(release.missing_fraction_limit)
    selected = frame.loc[eligible]
    if set(selected.label.unique()) != {0, 1}:
        raise ValueError("external eligible population lacks both classes")
    probabilities = np.asarray(model.predict(selected[release.features].astype("float64")))
    if (
        probabilities.shape != (len(selected), 2)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError("external model probability output invalid")
    result = metrics(
        selected.label.to_numpy(), probabilities[:, 1], release.threshold, len(selected) / len(raw)
    )
    result["external_gate_passed"] = result.pop("development_gate_passed")
    result["roc_auc"] = float(roc_auc_score(selected.label, probabilities[:, 1]))
    result["recall_interval"] = wilson(result["tp"], result["tp"] + result["fn"])
    result["precision_interval"] = wilson(result["tp"], result["tp"] + result["fp"])
    overlap = selected.group_id.isin(training_groups)
    result["cross_file_overlap_rows"] = int(overlap.sum())
    nonoverlap = selected.loc[~overlap]
    result["nonoverlap_sensitivity"] = (
        metrics(
            nonoverlap.label.to_numpy(),
            probabilities[~overlap.to_numpy(), 1],
            release.threshold,
            len(nonoverlap) / len(raw),
        )
        if set(nonoverlap.label.unique()) == {0, 1}
        else None
    )
    result["calibration_bins"] = []
    for indices in np.array_split(np.argsort(probabilities[:, 1]), min(10, len(selected))):
        result["calibration_bins"].append(
            {
                "rows": len(indices),
                "mean_probability": float(probabilities[indices, 1].mean()),
                "positive_fraction": float(selected.label.to_numpy()[indices].mean()),
            }
        )
    group_cost = (
        pd.Series(
            np.where(
                (probabilities[:, 1] >= release.threshold) & selected.label.eq(0).to_numpy(),
                10,
                np.where(
                    (probabilities[:, 1] < release.threshold) & selected.label.eq(1).to_numpy(),
                    500,
                    0,
                ),
            )
        )
        .groupby(selected.group_id.to_numpy())
        .sum()
        .to_numpy()
    )
    random = np.random.default_rng(42)
    result["group_bootstrap_cost_interval"] = [
        float(x)
        for x in np.quantile(
            [random.choice(group_cost, len(group_cost), replace=True).sum() for _ in range(2000)],
            [0.025, 0.975],
        )
    ]
    result.update(
        origin="azure_benchmark_evaluation",
        release_sha256=release.sha256,
        external_test_sha256=test_hash,
        input_rows=len(raw),
        quarantined_rows=len(quarantined),
        scope="No vehicle/time/site generalisation claim",
    )
    store.finish_evaluation(release, result)
    return result


def evaluate_briefs(
    cases,
    outcomes,
    human_labels,
    *,
    frozen_suite_hash,
    observed_suite_hash,
    frozen_release_sha256,
    execution,
):
    if frozen_suite_hash != observed_suite_hash or identity(cases) != observed_suite_hash:
        raise ValueError("evaluation suite changed after freeze")
    if len({x["id"] for x in cases}) != len(cases) or set(outcomes) != {x["id"] for x in cases}:
        raise ValueError("evaluation outcomes must match the unique frozen suite")
    if set(human_labels) != set(outcomes):
        return {"status": "human_review_pending", "release_allowed": False}
    if (
        execution.get("origin") != "foundry"
        or execution.get("heldout") is not True
        or execution.get("operation_status") != "terminal"
        or execution.get("suite_sha256") != frozen_suite_hash
        or execution.get("release_sha256") != frozen_release_sha256
        or execution.get("outcomes_sha256") != identity(outcomes)
    ):
        return {"status": "heldout_execution_unverified", "release_allowed": False}
    critical, correct, supported, claims, answerable, completed, numeric_failures = (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    for case in cases:
        outcome, label = outcomes[case["id"]], human_labels[case["id"]]
        if label.get("reviewer_kind") != "human" or not label.get("reviewer"):
            raise ValueError("AI/synthetic rubric is not human acceptance")
        if (
            label.get("outcome_sha256") != identity(outcome)
            or label.get("suite_sha256") != frozen_suite_hash
        ):
            raise ValueError("human label is not bound to this exact suite/output")
        counts = [label.get("supported_factual_claims"), label.get("total_factual_claims")]
        if any(type(n) is not int or n < 0 for n in counts) or counts[0] > counts[1]:
            raise ValueError("invalid human claim counts")
        if (
            outcome.get("case_id") != case["case_id"]
            or outcome.get("release_sha256") != frozen_release_sha256
        ):
            critical += 1
        numeric_failures += int(label.get("numeric_facts_exact") is not True)
        if case["critical"] and label.get("passed") is not True:
            critical += 1
        correct += int(label.get("passed") is True)
        supported += label["supported_factual_claims"]
        claims += label["total_factual_claims"]
        answerable += int(case["answerable"])
        completed += int(
            case["answerable"]
            and label.get("task_completed") is True
            and outcome.get("status") == "draft"
        )
    precision = supported / claims if claims else None
    completion = completed / answerable if answerable else None
    return {
        "status": "human_labelled",
        "cases": len(cases),
        "correct": correct,
        "critical_failures": critical,
        "numeric_fidelity_failures": numeric_failures,
        "cited_factual_precision": precision,
        "answerable_completion": completion,
        "completion_interval": wilson(completed, answerable),
        "interval_assumption": "Descriptive independent-Bernoulli interval; repeated authored cases are not independent field evidence",
        "release_allowed": critical == 0
        and numeric_failures == 0
        and precision is not None
        and precision >= 0.95
        and completion is not None
        and completion >= 0.85,
    }


def retrieval_quality(rankings, human_relevance):
    if not rankings or set(rankings) != set(human_relevance):
        return {"status": "human_relevance_pending", "release_allowed": False}
    rows = []
    for query_id, ids in rankings.items():
        label = human_relevance[query_id]
        if label.get("reviewer_kind") != "human" or not label.get("reviewer"):
            raise ValueError("human retrieval relevance labels required")
        grades = label["grades"]
        if any(type(n) is not int or not 0 <= n <= 3 for n in grades.values()):
            raise ValueError("invalid graded relevance")
        relevant = {key for key, value in grades.items() if value > 0}
        if not relevant or len(ids) != len(set(ids)):
            raise ValueError("answerable unique retrieval ranking required")
        top = ids[:5]
        gain = sum((2 ** grades.get(key, 0) - 1) / np.log2(i + 2) for i, key in enumerate(top))
        ideal = sum(
            (2**grade - 1) / np.log2(i + 2)
            for i, grade in enumerate(sorted(grades.values(), reverse=True)[:5])
        )
        rows.append(
            {
                "query_id": query_id,
                "recall_at_5": len(set(top) & relevant) / len(relevant),
                "ndcg_at_5": float(gain / ideal),
            }
        )
    recall = float(np.mean([row["recall_at_5"] for row in rows]))
    return {
        "status": "human_labelled",
        "rows": rows,
        "mean_recall_at_5": recall,
        "release_allowed": recall >= 0.85,
        "scope": "small reviewed query suite only",
    }


def foundry_judge(query, response, context, model_config, credential, guard):
    """Official hosted evaluators supplement, and never impersonate, human labels."""
    from azure.ai.evaluation import GroundednessEvaluator, RelevanceEvaluator

    if sum(len(x.encode()) for x in (query, response, context)) > 8000:
        raise ValueError("judge input exceeds the frozen request envelope")
    # This pinned SDK owns up to ten retries. Reserve all eleven attempts before
    # invoking it; these are conservative request reservations, not measured calls.
    for _ in range(11):
        guard("judge")
    groundedness = GroundednessEvaluator(model_config=model_config, credential=credential)
    relevance = RelevanceEvaluator(model_config=model_config, credential=credential)
    grounded = groundedness(query=query, response=response, context=context)
    for _ in range(11):
        guard("judge")
    relevant = relevance(query=query, response=response)
    return {
        "groundedness": grounded,
        "relevance": relevant,
        "human_review": False,
        "maximum_attempts_reserved": 22,
        "evaluator_config_sha256": __import__("hashlib")
        .sha256(json.dumps(model_config, sort_keys=True).encode())
        .hexdigest(),
    }
