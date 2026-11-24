"""No-network integration rehearsal. Every case, prediction and cost is synthetic."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from fleet_fault_triage.contracts import Release, identity
from fleet_fault_triage.data import digest, save_json
from fleet_fault_triage.evidence import case_report, export_manifest
from fleet_fault_triage.finops import CostLedger
from fleet_fault_triage.investigation import Investigator
from fleet_fault_triage.lifecycle import CaseStore
from fleet_fault_triage.monitoring import delayed_quality, monitor_inputs, monitor_runtime
from fleet_fault_triage.retrieval import Corpus
from fleet_fault_triage.scoring import score_batch


class SyntheticModel:
    def predict(self, frame):
        values = np.clip(frame.iloc[:, 0].to_numpy() / 10, 0, 1)
        return np.column_stack([1 - values, values])


def synthetic_release(root, corpus, **changes):
    content = {
        "origin": "synthetic_rehearsal",
        "code_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "lock_sha256": digest(root / "uv.lock"),
        "data_sha256": identity("SYNTHETIC_DATA"),
        "split_sha256": identity("SYNTHETIC_SPLIT"),
        "features": ["aa", "bb"],
        "model_uri": "models:/synthetic.development.aps/1",
        "model_run_id": "SYNTHETIC_NOT_A_PROVIDER_RUN",
        "threshold": 0.5,
        "missing_fraction_limit": 0.5,
        "prompt_sha256": digest(root / "resources/prompts/investigator.md"),
        "tool_policy_sha256": digest(root / "src/fleet_fault_triage/investigation.py"),
        "corpus_sha256": corpus.version,
        "index_version": "SYNTHETIC_LOCAL_LEXICAL",
        "embedding_version": "NO_EMBEDDING_USED",
        "generation_version": "NO_LLM_USED",
        **changes,
    }
    return Release.model_validate(content)


def rehearse(root: Path, output: Path):
    if output.exists() and any(output.iterdir()):
        raise ValueError("rehearsal destination must be new/empty; preserve previous evidence")
    output.mkdir(parents=True, exist_ok=True)
    corpus = Corpus.load(root / "resources/runbooks")
    release = synthetic_release(root, corpus)
    store = CaseStore(output / "cases.sqlite")
    frame = pd.DataFrame(
        {
            "record_id": ["synthetic-high", "synthetic-low", "synthetic-missing"],
            "aa": [8.0, 2.0, None],
            "bb": [1.0, 2.0, None],
        }
    )
    scores = score_batch(frame, release, SyntheticModel(), "synthetic-batch", store)
    assert score_batch(frame, release, SyntheticModel(), "synthetic-batch", store) == scores
    assert {x["route"] for x in scores} == {
        "aps_review",
        "other_fault_review",
        "insufficient_evidence",
    }
    denied = []
    try:
        score_batch(frame.assign(label=0), release, SyntheticModel(), "bad", store)
    except ValueError:
        denied.append("label_in_scoring_input")
    reviewer = "SYNTHETIC_REVIEWER_NOT_HUMAN_ACCEPTANCE"
    first = store.review(
        scores[0]["case_id"],
        reviewer,
        "aps_review",
        "Synthetic initial disposition",
        None,
        synthetic=True,
    )
    correction = store.review(
        scores[0]["case_id"],
        reviewer,
        "insufficient_evidence",
        "Synthetic append-only correction",
        first["event_hash"],
        synthetic=True,
    )
    assert CaseStore(output / "cases.sqlite").reviews(scores[0]["case_id"])[-1] == correction
    try:
        store.review(scores[0]["case_id"], reviewer, "aps_review", "stale", None, synthetic=True)
    except ValueError:
        denied.append("stale_review")
    gates = {
        name: True
        for name in ("model_quality", "assistant_quality", "permissions", "reproducibility")
    }

    def approval(item):
        return {
            "release_sha256": item.sha256,
            "reviewer": reviewer,
            "source": "synthetic fixture",
            "approved_utc": "2026-10-25T00:00:00+00:00",
            "synthetic": True,
        }

    store.promote(release, None, approval(release), gates)
    second = synthetic_release(root, corpus, model_uri="models:/synthetic.development.aps/2")
    try:
        store.promote(second, release.sha256, approval(second), {**gates, "model_quality": False})
    except ValueError:
        denied.append("rejected_promotion")
    assert store.alias() == release.sha256
    store.promote(second, release.sha256, approval(second), gates)
    store.promote(release, second.sha256, approval(release), gates)
    assert store.alias() == release.sha256
    investigator = Investigator(
        store, release, corpus, corpus.keyword, [x["case_id"] for x in scores]
    )
    briefs = {}
    for score in scores:
        documents = corpus.keyword("quality missingness APS threshold evidence", corpus.version, 3)
        brief = investigator.assemble(
            {
                "case_id": score["case_id"],
                "status": "draft",
                "citations": [x["id"] for x in documents],
                "missing_evidence": ["timing", "field"],
            },
            score["case_id"],
            origin="synthetic_rehearsal",
        )
        briefs[score["case_id"]] = brief.model_dump()
    try:
        investigator.get_case_evidence("unauthorised")
    except PermissionError:
        denied.append("out_of_scope_case")
    ledger = CostLedger(output / "synthetic-cost.sqlite")
    for i, component in enumerate(("training", "scoring", "brief")):
        ledger.record_usage(
            {
                "record_id": f"synthetic-{i}",
                "component": component,
                "run_id": "SYNTHETIC_NOT_PROVIDER",
                "currency": "AUD",
                "cost_aud": ".5",
                "record_type": "ORIGINAL",
                "origin": "synthetic_rehearsal",
            }
        )
    costs = {
        **ledger.report({"training": 1, "scoring": 3, "brief": 0}, ".3"),
        "origin": "synthetic_rehearsal",
        "actual_cloud_cost": None,
    }
    reference = pd.DataFrame({"aa": [1.0] * 1000, "bb": [2.0] * 1000})
    shifted = reference.assign(aa=[None] * 200 + [5.0] * 800)
    monitoring = {
        "input": monitor_inputs(
            reference, shifted, ["aa", "bb"], release.sha256, origin="synthetic_rehearsal"
        ),
        "labels_hidden": delayed_quality(scores, None, release.sha256),
        "labels_revealed": delayed_quality(
            scores, dict(zip(frame.record_id, [1, 0, 1], strict=True)), release.sha256
        ),
        "runtime": monitor_runtime(
            [
                {"kind": kind, "origin": "synthetic_rehearsal"}
                for kind in (
                    "late_import",
                    "duplicate",
                    "reordered",
                    "schema_rejected",
                    "scoring_outage",
                    "retrieval_unavailable",
                    "invalid_model_identity",
                    "success",
                )
            ],
            release.sha256,
        ),
    }
    summary = {
        "schema_version": 1,
        "origin": "synthetic_rehearsal",
        "cloud_execution": False,
        "real_data_fits_added": 0,
        "external_test_opened": False,
        "hosted_quality_claimed": False,
        "routes": sorted({x["route"] for x in scores}),
        "denied_scenarios": denied,
        "rollback_readback_passed": store.alias() == release.sha256,
        "review_restart_readback_passed": True,
        "batch_replay_readback_passed": True,
    }
    for name, value in (
        ("summary.json", summary),
        ("scores.json", scores),
        ("briefs.json", briefs),
        ("monitoring.json", monitoring),
        ("finops.json", costs),
        ("release.json", release.model_dump()),
    ):
        save_json(output / name, value)
    reviews = {x["case_id"]: store.reviews(x["case_id"]) for x in scores}
    save_json(output / "reviews.json", reviews)
    case_report(output, scores, briefs, reviews, "synthetic_rehearsal; no cloud/LLM inference")
    names = [x.name for x in output.iterdir() if x.suffix in {".json", ".html"}]
    export_manifest(output, names, release.code_sha, "synthetic_rehearsal")
    if json.loads((output / "summary.json").read_text()) != summary:
        raise ValueError("rehearsal summary readback failed")
    return summary
