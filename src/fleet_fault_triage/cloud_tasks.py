"""Databricks wheel entry points, invoked only by exact approved job permits."""

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fleet_fault_triage.cloud import identifier, score_delta, train_registered
from fleet_fault_triage.contracts import Release, identity
from fleet_fault_triage.data import save_json
from fleet_fault_triage.evaluation import evaluate_external, external_reference
from fleet_fault_triage.lifecycle import CaseStore
from fleet_fault_triage.monitoring import monitor_inputs


def validate_permit(config, permit, task):
    if config.get("cloud_spend_authorized") is not True or permit.get("approved") is not True:
        raise ValueError("paid task remains blocked")
    if permit.get("synthetic") is not False or permit.get("task") != task:
        raise ValueError("invalid task/synthetic permit")
    if permit.get("config_sha256") != identity(config) or not permit.get("reservation_id"):
        raise ValueError("task permit lacks exact configuration/cost reservation")
    expiry = datetime.fromisoformat(permit["expires_utc"])
    if expiry.tzinfo is None or datetime.now(timezone.utc) >= expiry:
        raise ValueError("task permit expired")
    if config.get("prices_verified") is not True or config.get("entitlements_verified") is not True:
        raise ValueError("live resource preflight incomplete")
    if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        raise ValueError("wheel task must execute inside the approved Databricks runtime")


def verify_workspace_identity(workspace, config):
    expected = config.get("databricks_workspace_id")
    if not isinstance(expected, str) or not re.fullmatch(r"[1-9][0-9]*", expected):
        raise ValueError("approved provider workspace ID is required")
    # Runtime authentication can supply another URL for the same workspace.
    # get_workspace_id() can return a configured/cached ID without a provider call.
    header = "X-Databricks-Org-Id"
    observed = workspace.api_client.do(
        "GET", "/api/2.0/preview/scim/v2/Me", response_headers=[header]
    )
    if observed.get(header) != expected:
        raise ValueError("task is executing in the wrong workspace")
    return identity(expected)


def run_task(config_path: Path, permit_path: Path, task):
    config, permit = json.loads(config_path.read_text()), json.loads(permit_path.read_text())
    validate_permit(config, permit, task)
    from fleet_fault_triage.runtime import verify_runtime

    verify_runtime(config)
    from databricks.sdk import WorkspaceClient
    from pyspark.sql import SparkSession

    workspace = WorkspaceClient()
    workspace_identity = verify_workspace_identity(workspace, config)
    principal = workspace.current_user.me()
    role = {"train": "trainer", "evaluate": "evaluator", "score": "scorer", "monitor": "scorer"}[
        task
    ]
    if principal.user_name != config["principals"][role]:
        raise ValueError("task is executing as the wrong principal")
    spark = SparkSession.builder.getOrCreate()
    catalog = identifier(config["catalog"])
    schema = identifier(
        config["evaluation_schema"] if task == "evaluate" else config["evidence_schema"]
    )
    output = Path(config["output_dir"]) / permit["reservation_id"]
    if task == "train":
        result = train_registered(
            config, Path(config["recipe_dir"]), Path(config["training_data_dir"]), spark
        )
    else:
        import mlflow
        import pandas as pd

        release = Release.model_validate_json(Path(config["release_path"]).read_text())
        if release.origin != "azure":
            raise ValueError("synthetic release cannot become a hosted result")
        if (
            release.code_sha != config["source_code_sha"]
            or release.lock_sha256 != config["dependency_lock_sha256"]
        ):
            raise ValueError("release differs from approved source/dependency lock")
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        model = mlflow.pyfunc.load_model(release.model_uri)
        if task == "evaluate":
            freeze = json.loads(Path(config["freeze_path"]).read_text())
            training_path = Path(config["training_data_dir"]) / "prepared/training.parquet"
            external_reference(release, freeze, training_path)
            marker = f"{catalog}.{schema}.external_evaluation_reservations"
            spark.sql(
                f"CREATE TABLE IF NOT EXISTS {marker} (test_hash STRING, release_hash STRING) USING DELTA"
            )
            # Single evaluator job, max_concurrent_runs=1 and evaluator-only writes serialize this gate.
            test_hash = freeze["external_test_sha256"]
            if not __import__("re").fullmatch(r"[0-9a-f]{64}", test_hash):
                raise ValueError("invalid frozen test hash")
            if spark.sql(
                f"SELECT count(*) AS n FROM {marker} WHERE test_hash='{test_hash}'"
            ).first()["n"]:
                raise ValueError(
                    "external test already consumed; disclosed correction requires owner review"
                )
            spark.createDataFrame(
                [(test_hash, release.sha256)], "test_hash string, release_hash string"
            ).write.mode("append").saveAsTable(marker)
            with tempfile.TemporaryDirectory() as temporary:
                result = evaluate_external(
                    Path(config["evaluation_data_dir"]),
                    release,
                    model,
                    CaseStore(Path(temporary) / "evaluation.sqlite"),
                    freeze,
                    lambda action: (
                        None
                        if action == "external_evaluation"
                        else (_ for _ in ()).throw(ValueError("unexpected operation"))
                    ),
                    training_path=training_path,
                )
        elif task == "score":
            frame = pd.read_parquet(config["scoring_input_path"])
            result = score_delta(
                frame, release, model, config["batch_id"], spark, f"{catalog}.{schema}.scoring"
            )
            result["origin"] = "azure"
        else:
            result = monitor_inputs(
                pd.read_parquet(config["monitor_reference_path"]),
                pd.read_parquet(config["monitor_current_path"]),
                config["monitor_features"],
                release.sha256,
                origin="benchmark_replay",
            )
    result.update(
        task=task,
        provider_execution="azure",
        config_sha256=identity(config),
        reservation_id=permit["reservation_id"],
        source_code_sha=config["source_code_sha"],
        workspace_identity_sha256=workspace_identity,
    )
    save_json(output / "result.json", result)
    if json.loads((output / "result.json").read_text()) != result:
        raise ValueError("saved task evidence readback failed")
    return {"task": task, "result_sha256": identity(result), "provider_execution": True}
