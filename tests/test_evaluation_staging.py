"""Synthetic opaque-byte fixtures; no benchmark data or cloud calls."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from fleet_fault_triage.contracts import identity
from fleet_fault_triage.data import TEST, digest, save_json
from fleet_fault_triage.operations import evaluation_files, launch_job
from fleet_fault_triage.rehearsal import synthetic_release
from fleet_fault_triage.retrieval import Corpus

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evaluation_input(tmp_path):
    config = json.loads((ROOT / "resources/cloud.example.json").read_text())
    release = synthetic_release(ROOT, Corpus.load(ROOT / "resources/runbooks"), origin="azure")
    prepared = tmp_path / "data/prepared"
    prepared.mkdir(parents=True)
    (prepared / "training.parquet").write_bytes(b"SYNTHETIC_TRAINING_BYTES")
    save_json(
        prepared / "manifest.json",
        {
            "features": release.features,
            "split_sha256": release.split_sha256,
            "prepared_sha256": digest(prepared / "training.parquet"),
        },
    )
    release = release.model_copy(update={"data_sha256": digest(prepared / "manifest.json")})
    raw = tmp_path / "data/raw" / TEST
    raw.parent.mkdir()
    raw.write_bytes(b"SYNTHETIC_SEALED_BYTES_NOT_VALID_CSV")
    private = tmp_path / "outputs/cloud-private/SYNTHETIC"
    save_json(private / "release.json", release.model_dump())
    save_json(
        private / "freeze.json",
        {
            "release_sha256": release.sha256,
            "parity_passed": True,
            "external_evaluation_authorized": True,
            "external_test_sha256": digest(raw),
            "training_sha256": digest(prepared / "training.parquet"),
        },
    )
    control = "/Volumes/fleet_fault_triage/evidence/control/evaluation"
    config.update(
        release_path=control + "/release.json",
        freeze_path=control + "/freeze.json",
        source_code_sha=release.code_sha,
        dependency_lock_sha256=release.lock_sha256,
    )
    paths = {
        "release": (private / "release.json", config["release_path"]),
        "freeze": (private / "freeze.json", config["freeze_path"]),
        "training_manifest": (
            prepared / "manifest.json",
            config["training_data_dir"] + "/prepared/manifest.json",
        ),
        "external_test": (raw, config["evaluation_data_dir"] + "/raw/" + TEST),
    }
    manifest = {
        name: {
            "local": str(local.relative_to(tmp_path)),
            "destination": target,
            "sha256": digest(local),
        }
        for name, (local, target) in paths.items()
    }
    file = private / "manifest.json"
    save_json(file, manifest)
    config["evaluation_manifest_sha256"] = digest(file)
    return tmp_path, config, file


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "changed_test",
        "destination",
        "missing_file",
        "unapproved",
        "training_hash",
        "release_hash",
        "another_volume",
        "source",
        "manifest_hash",
        "stale_training_manifest",
    ],
)
def test_evaluation_inputs_are_exact_and_test_bytes_stay_opaque(
    evaluation_input, monkeypatch, fault
):
    root, config, file = evaluation_input
    manifest = json.loads(file.read_text())
    if fault == "changed_test":
        (root / manifest["external_test"]["local"]).write_bytes(b"CHANGED")
    elif fault == "destination":
        manifest["external_test"]["destination"] = "/Volumes/OTHER/sealed/test.csv"
    elif fault == "missing_file":
        del manifest["training_manifest"]
    elif fault in {"unapproved", "training_hash", "release_hash", "stale_training_manifest"}:
        freeze_path = root / manifest["freeze"]["local"]
        freeze = json.loads(freeze_path.read_text())
        if fault == "stale_training_manifest":
            training_path = root / "data/prepared/training.parquet"
            training_path.write_bytes(b"REPLACED_SYNTHETIC_TRAINING")
            freeze["training_sha256"] = digest(training_path)
        else:
            freeze[
                {
                    "unapproved": "external_evaluation_authorized",
                    "training_hash": "training_sha256",
                    "release_hash": "release_sha256",
                }[fault]
            ] = False if fault == "unapproved" else "0" * 64
        save_json(freeze_path, freeze)
        manifest["freeze"]["sha256"] = digest(freeze_path)
    elif fault == "another_volume":
        config["evaluation_data_dir"] = "/Volumes/OTHER/sealed/data"
    elif fault == "source":
        config["source_code_sha"] = "0" * 40
    save_json(file, manifest)
    config["evaluation_manifest_sha256"] = "0" * 64 if fault == "manifest_hash" else digest(file)

    def forbidden(*args, **kwargs):
        raise AssertionError("sealed test was parsed during preparation")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    if fault:
        with pytest.raises(ValueError):
            evaluation_files(config, root, file)
    else:
        files = evaluation_files(config, root, file)
        assert set(files) == {"release", "freeze", "training_manifest", "external_test"}
        assert files["external_test"][0] == root / "data/raw" / TEST


@pytest.mark.parametrize("fault", [None, "readback", "changed_after_preflight"])
def test_evaluation_upload_is_reserved_and_verified_before_permit(
    evaluation_input, monkeypatch, fault
):
    import databricks.sdk
    from databricks.sdk.service._internal import Wait
    from databricks.sdk.service.jobs import JobSettings, Run, RunNowResponse

    root, config, manifest = evaluation_input
    principal = "SYNTHETIC_EVALUATOR"
    settings = JobSettings.from_dict(
        {
            "max_concurrent_runs": 1,
            "timeout_seconds": 1800,
            "run_as": {"service_principal_name": principal},
            "tasks": [
                {
                    "timeout_seconds": 1800,
                    "max_retries": 0,
                    "python_wheel_task": {"entry_point": "fft"},
                    "new_cluster": {
                        "num_workers": 0,
                        "node_type_id": config["node_type"],
                        "spark_version": config["runtime"],
                    },
                }
            ],
        }
    )
    config["principals"]["evaluator"] = principal
    config["job_ids"]["evaluate"] = 1
    config["job_settings_sha256"]["evaluate"] = identity(settings.as_dict())
    events, saved = [], {}

    def begin(*args):
        events.append("reserved")
        if fault == "changed_after_preflight":
            (root / "data/raw" / TEST).write_bytes(b"CHANGED")

    def upload(path, stream, *, overwrite):
        assert events[0] == "reserved"
        assert overwrite is (path == config["runtime_config_path"])
        assert path not in saved
        saved[path] = stream.read()
        events.append("permit" if "/permits/" in path else "uploaded")

    def download(path):
        events.append("readback")
        body = saved[path]
        if fault == "readback" and path.endswith(TEST):
            body += b"CHANGED"
        return SimpleNamespace(contents=io.BytesIO(body))

    def run_now(**kwargs):
        events.append("submitted")
        assert len(saved) == 6 and events[-2] == "permit"
        run = Run.from_dict(
            {
                "run_id": 1,
                "job_id": 1,
                "job_parameters": [
                    {"name": k, "value": v} for k, v in kwargs["job_parameters"].items()
                ],
                "state": {"life_cycle_state": "TERMINATED", "result_state": "FAILED"},
            }
        )
        return Wait(lambda **kw: run, response=RunNowResponse(run_id=1), run_id=1)

    client = SimpleNamespace(
        jobs=SimpleNamespace(get=lambda job: SimpleNamespace(settings=settings), run_now=run_now),
        files=SimpleNamespace(upload=upload, download=download, create_directory=lambda path: None),
    )
    monkeypatch.setattr(databricks.sdk, "WorkspaceClient", lambda **kwargs: client)
    gate = SimpleNamespace(
        config=config,
        begin=begin,
        approval={"expires_utc": "2099-01-01T00:00:00+00:00", "source": "SYNTHETIC_TEST"},
        ledger=SimpleNamespace(finish_execution=lambda *a, **kw: events.append("terminal")),
    )
    output = root / "outputs/cloud-private/run"
    message = {
        None: "provider job failed",
        "readback": "saved evaluation input differs",
        "changed_after_preflight": "bytes changed after preflight",
    }[fault]
    with pytest.raises(ValueError, match=message):
        launch_job(gate, "evaluate", "10", output, root=root, evaluation_manifest=manifest)
    if fault:
        assert "permit" not in events and "submitted" not in events and "terminal" not in events
    else:
        assert events[-2:] == ["submitted", "terminal"]
        readback = json.loads((output / "evaluation-inputs-readback.json").read_text())
        assert len(readback) == 4
