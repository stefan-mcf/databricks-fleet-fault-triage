"""Synthetic lifecycle/security regressions; never touch the real benchmark or Azure."""

import asyncio
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fleet_fault_triage.contracts import Release
from fleet_fault_triage.evidence import export_manifest, verify_manifest
from fleet_fault_triage.investigation import Investigator
from fleet_fault_triage.lifecycle import CaseStore
from fleet_fault_triage.monitoring import delayed_quality, monitor_inputs
from fleet_fault_triage.rehearsal import SyntheticModel, rehearse, synthetic_release
from fleet_fault_triage.retrieval import Corpus
from fleet_fault_triage.scoring import score_batch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def application(tmp_path):
    corpus = Corpus.load(ROOT / "resources/runbooks")
    release = synthetic_release(ROOT, corpus)
    store = CaseStore(tmp_path / "cases.sqlite")
    frame = pd.DataFrame(
        {
            "record_id": ["fixture-high", "fixture-low", "fixture-reject"],
            "aa": [9.0, 1.0, None],
            "bb": [1.0, 2.0, None],
        }
    )
    return corpus, release, store, frame


def test_scoring_is_atomic_replayable_and_label_free(application):
    _, release, store, frame = application
    scores = score_batch(frame, release, SyntheticModel(), "batch", store)
    assert [x["route"] for x in scores] == [
        "aps_review",
        "other_fault_review",
        "insufficient_evidence",
    ]
    assert scores[-1]["probability"] is None
    assert score_batch(frame, release, SyntheticModel(), "batch", store) == scores
    changed = frame.copy()
    changed.loc[0, "aa"] = 8
    with pytest.raises(ValueError, match="replay"):
        score_batch(changed, release, SyntheticModel(), "batch", store)
    assert store.batch("batch") == scores
    assert all("label" not in x for x in scores)


@pytest.mark.parametrize("mutation", ["label", "wrong_order", "infinity", "duplicate_id"])
def test_invalid_input_never_publishes_cases(application, mutation):
    _, release, store, frame = application
    if mutation == "label":
        frame["label"] = 0
    elif mutation == "wrong_order":
        frame = frame[["record_id", "bb", "aa"]]
    elif mutation == "infinity":
        frame.loc[0, "aa"] = np.inf
    else:
        frame.loc[1, "record_id"] = frame.loc[0, "record_id"]
    with pytest.raises(ValueError):
        score_batch(frame, release, SyntheticModel(), "invalid", store)
    assert store.batch("invalid") == []


def test_provider_failure_does_not_leave_half_batch(application):
    _, release, store, frame = application

    class BrokenModel:
        def predict(self, _):
            raise TimeoutError("controlled outage")

    with pytest.raises(TimeoutError):
        score_batch(frame, release, BrokenModel(), "outage", store)
    assert store.batch("outage") == []
    assert len(score_batch(frame, release, SyntheticModel(), "outage", store)) == 3


def test_review_correction_survives_restart_and_rejects_stale_write(application):
    _, release, store, frame = application
    case = score_batch(frame, release, SyntheticModel(), "review", store)[0]
    first = store.review(
        case["case_id"], "synthetic-reviewer", "aps_review", "fixture reason", None, synthetic=True
    )
    with pytest.raises(ValueError, match="stale"):
        store.review(
            case["case_id"],
            "synthetic-reviewer",
            "other_fault_review",
            "stale",
            None,
            synthetic=True,
        )
    second = store.review(
        case["case_id"],
        "synthetic-reviewer",
        "insufficient_evidence",
        "correction",
        first["event_hash"],
        synthetic=True,
    )
    restarted = CaseStore(store.path)
    assert restarted.reviews(case["case_id"])[-1] == second
    with sqlite3.connect(store.path) as db:
        value = json.loads(
            db.execute(
                "SELECT payload FROM reviews WHERE event_hash=?", (first["event_hash"],)
            ).fetchone()[0]
        )
        value["reason"] = "tampered"
        db.execute(
            "UPDATE reviews SET payload=? WHERE event_hash=?",
            (json.dumps(value), first["event_hash"]),
        )
    with pytest.raises(ValueError, match="corrupt"):
        restarted.reviews(case["case_id"])


def test_runtime_tools_cannot_cross_case_model_or_corpus_scope(application):
    corpus, release, store, frame = application
    scores = score_batch(frame, release, SyntheticModel(), "scope", store)
    agent = Investigator(store, release, corpus, corpus.keyword, [scores[0]["case_id"]])
    with pytest.raises(PermissionError):
        agent.get_case_evidence(scores[1]["case_id"])
    with pytest.raises(PermissionError):
        agent.get_model_card("models:/another.scope.model/1")
    with pytest.raises(ValueError):
        asyncio.run(agent.search_runbooks("quality", "stale-corpus", 1))
    with pytest.raises(ValueError):
        asyncio.run(agent.search_runbooks("quality", corpus.version, 6))
    assert not any(hasattr(agent, x) for x in ("shell", "sql", "http", "promote", "review"))


def test_untrusted_model_output_cannot_invent_numeric_facts_or_approve(application):
    corpus, release, store, frame = application
    case = score_batch(frame, release, SyntheticModel(), "brief", store)[0]
    agent = Investigator(store, release, corpus, corpus.keyword, [case["case_id"]])
    selection = {
        "case_id": case["case_id"],
        "status": "draft",
        "citations": ["quality"],
        "missing_evidence": [],
    }
    with pytest.raises(ValueError):
        agent.assemble({**selection, "status": "approved"}, case["case_id"])
    with pytest.raises(ValueError):
        agent.assemble({**selection, "probability": 0.99}, case["case_id"])
    with pytest.raises(ValueError):
        agent.assemble({**selection, "citations": ["deleted"]}, case["case_id"])
    brief = agent.assemble(selection, case["case_id"], origin="synthetic_rehearsal")
    assert "0.9; threshold 0.5" in brief.estimates[0]
    assert len(brief.missing_evidence) >= 3
    assert store.reviews(case["case_id"]) == []


def test_external_eval_is_not_available_to_synthetic_release(application):
    _, release, store, _ = application
    with pytest.raises(ValueError, match="Azure"):
        store.reserve_evaluation(release)


def test_complete_rehearsal_is_explicit_and_hash_verified(tmp_path):
    summary = rehearse(ROOT, tmp_path / "rehearsal")
    assert summary["cloud_execution"] is False
    assert summary["real_data_fits_added"] == 0
    assert len(summary["routes"]) == 3
    root = tmp_path / "rehearsal"
    assert verify_manifest(root)["evidence_class"] == "synthetic_rehearsal"
    (root / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        verify_manifest(root)


def test_evidence_export_rejects_path_escape_and_credentials(tmp_path):
    external = tmp_path / "external.json"
    external.write_text("{}")
    root = tmp_path / "package"
    root.mkdir()
    with pytest.raises(ValueError, match="outside"):
        export_manifest(root, ["../external.json"], "0" * 40, "local")
    (root / "secret.json").write_text('{"url":"https://example.test/?sig=SECRET"}')
    with pytest.raises(ValueError, match="secret"):
        export_manifest(root, ["secret.json"], "0" * 40, "local")


def test_missing_labels_and_small_support_do_not_claim_degradation(application):
    _, release, store, frame = application
    scores = score_batch(frame, release, SyntheticModel(), "monitor", store)
    assert (
        delayed_quality(scores, None, release.sha256)["model_quality"] == "unknown_without_labels"
    )
    labels = {x["record_id"]: 1 for x in scores}
    assert (
        delayed_quality(scores, labels, release.sha256)["model_quality"] == "insufficient_support"
    )
    with pytest.raises(ValueError, match="release"):
        delayed_quality(scores, labels, "changed")
    with pytest.raises(ValueError, match="join"):
        delayed_quality(scores, {}, release.sha256)


def test_drift_is_an_investigation_trigger_and_reports_small_samples():
    reference = pd.DataFrame({"aa": [1.0] * 1200})
    current = pd.DataFrame({"aa": [5.0] * 1200})
    report = monitor_inputs(
        reference, current, ["aa"], "frozen-reference", origin="synthetic_rehearsal"
    )
    assert report["features"][0]["distribution_alert"] is True
    assert report["model_quality"] == "unknown_without_labels"
    small = monitor_inputs(
        reference.head(2), current.head(2), ["aa"], "ref", origin="synthetic_rehearsal"
    )
    assert small["features"][0]["p_value"] is None


def test_release_rejects_aliases_and_label_features(application):
    _, release, _, _ = application
    with pytest.raises(ValueError):
        Release.model_validate({**release.model_dump(), "model_uri": "models:/a.b.c@champion"})
    with pytest.raises(ValueError):
        Release.model_validate({**release.model_dump(), "features": ["aa", "label"]})
