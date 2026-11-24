"""Bounded input screens and label-gated quality; no automatic model promotion."""

import numpy as np
from scipy.stats import ks_2samp


def monitor_inputs(reference, current, features, reference_version, *, origin):
    if not features or len(features) > 10 or len(set(features)) != len(features):
        raise ValueError("one to ten predeclared features are required")
    if origin not in {"synthetic_rehearsal", "benchmark_replay"} or not reference_version:
        raise ValueError("monitor origin/reference required")
    if list(reference.columns) != list(current.columns) or not set(features) <= set(reference):
        raise ValueError("schema mismatch")
    rows, tests = [], []
    for feature in features:
        left, right = reference[feature], current[feature]
        if not np.isfinite(left.dropna()).all() or not np.isfinite(right.dropna()).all():
            raise ValueError("nonfinite monitor input")
        increase = (
            float(right.isna().mean() - left.isna().mean()) if len(left) and len(right) else None
        )
        supported = len(left.dropna()) >= 1000 and len(right.dropna()) >= 1000
        statistic, pvalue = ks_2samp(left.dropna(), right.dropna()) if supported else (None, None)
        row = {
            "feature": feature,
            "missingness_increase": increase,
            "missingness_alert": increase is not None and increase >= 0.10,
            "ks_effect": float(statistic) if supported else None,
            "p_value": float(pvalue) if supported else None,
            "distribution_status": "tested" if supported else "insufficient_support",
            "distribution_alert": False,
        }
        rows.append(row)
        if supported:
            tests.append((float(pvalue), row))
    ranked = sorted(tests, key=lambda pair: pair[0])
    accepted = [i for i, (p, _) in enumerate(ranked, 1) if p <= 0.05 * i / len(ranked)]
    cutoff = max(accepted, default=0)
    for rank, (_, row) in enumerate(ranked, 1):
        row["distribution_alert"] = rank <= cutoff and row["ks_effect"] >= 0.10
    return {
        "kind": "input_monitor",
        "origin": origin,
        "reference_version": reference_version,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "features": rows,
        "model_quality": "unknown_without_labels",
        "action": "human_investigation_only",
    }


def delayed_quality(scores, labels, release_sha256, *, origin="synthetic_rehearsal"):
    if labels is None:
        return {"origin": origin, "model_quality": "unknown_without_labels", "rows": len(scores)}
    if set(labels) != {x["record_id"] for x in scores}:
        raise ValueError("labels must join the exact unique scoring population")
    if len({x["record_id"] for x in scores}) != len(scores):
        raise ValueError("duplicate score IDs")
    if any(x["release_sha256"] != release_sha256 for x in scores):
        raise ValueError("delayed-label release mismatch")
    if any(value not in (0, 1) or isinstance(value, bool) for value in labels.values()):
        raise ValueError("invalid revealed labels")
    eligible = [x for x in scores if x["probability"] is not None]
    positives = sum(labels[x["record_id"]] for x in eligible)
    if len(eligible) < 3000 or positives < 30:
        return {
            "origin": origin,
            "model_quality": "insufficient_support",
            "rows": len(eligible),
            "positives": positives,
        }
    tp = fp = fn = 0
    for row in eligible:
        prediction, label = row["probability"] >= row["threshold"], labels[row["record_id"]]
        tp += int(prediction and label == 1)
        fp += int(prediction and label == 0)
        fn += int(not prediction and label == 1)
    return {
        "origin": origin,
        "model_quality": "label_backed_benchmark_only",
        "rows": len(eligible),
        "coverage": len(eligible) / len(scores),
        "recall": tp / positives,
        "recall_interval": __import__("fleet_fault_triage.evaluation", fromlist=["wilson"]).wilson(
            tp, positives
        ),
        "benchmark_cost_per_1000": (10 * fp + 500 * fn) * 1000 / len(eligible),
        "cost_uncertainty": "Paired group bootstrap required before asserting degradation or improvement",
        "action": "human_investigation_only",
    }


def monitor_runtime(events, reference_version):
    if not events or not reference_version:
        raise ValueError("runtime events/reference required")
    allowed = {
        "success",
        "schema_rejected",
        "scoring_outage",
        "retrieval_unavailable",
        "invalid_model_identity",
        "late_import",
        "duplicate",
        "reordered",
        "budget_exhausted",
    }
    counts = {name: 0 for name in sorted(allowed)}
    for event in events:
        if event.get("kind") not in allowed or event.get("origin") not in {
            "synthetic_rehearsal",
            "azure",
        }:
            raise ValueError("unknown/unattributed runtime event")
        counts[event["kind"]] += 1
    return {
        "reference_version": reference_version,
        "attempts": len(events),
        "counts": counts,
        "failures": len(events) - counts["success"],
        "action": "human_investigation_only",
    }
