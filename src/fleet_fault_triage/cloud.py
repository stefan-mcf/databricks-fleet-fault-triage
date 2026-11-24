"""Guarded project workflows using official clients; importing never contacts Azure."""

import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

from fleet_fault_triage.contracts import Release, identity
from fleet_fault_triage.data import digest, save_json
from fleet_fault_triage.finops import CostLedger, money


def identifier(value):
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", value):
        raise ValueError("invalid project SQL identifier")
    return value


def operation_bound(config, action):
    quote = config.get("quotes", {}).get(action)
    if not quote or quote.get("currency") != "AUD":
        raise ValueError("a priced operation envelope in AUD is required")
    observed = datetime.fromisoformat(quote["observed_utc"])
    if observed.tzinfo is None:
        raise ValueError("operation quote lacks timezone")
    age = datetime.now(timezone.utc) - observed
    if observed.tzinfo is None or not timedelta(0) <= age <= timedelta(hours=24):
        raise ValueError("operation quote is stale or lacks timezone")
    rows = quote.get("components")
    if not rows or any(not x.get("meter") for x in rows):
        raise ValueError("component price coverage is missing")
    total = Decimal(0)
    for row in rows:
        rate, quantity = money(row.get("rate_aud")), money(row.get("maximum_quantity"))
        if rate < 0 or quantity <= 0 or (rate == 0 and row.get("verified_free") is not True):
            raise ValueError("invalid/unverified free meter or quantity")
        total += rate * quantity
    multiplier = money(quote.get("tax_and_uncertainty_multiplier"))
    if multiplier < Decimal("1.10"):
        raise ValueError("tax/uncertainty margin required")
    return total * multiplier


class CloudGate:
    """An approval-bound local launcher; it cannot create an Azure hard billing cap."""

    def __init__(self, config, approval, ledger: CostLedger, reconciliation):
        self.config, self.approval, self.ledger, self.reconciliation = (
            config,
            approval,
            ledger,
            reconciliation,
        )
        self.started = False
        self.calls = {"embedding": 0, "inference": 0, "judge": 0, "retrieval": 0, "index_write": 0}

    def begin(self, action, batch_id, maximum_aud, *, resume_existing=False):
        if self.approval.get("synthetic") is not False:
            raise ValueError("synthetic/missing approval cannot authorise cloud work")
        if self.approval.get("config_sha256") != identity(self.config):
            raise ValueError("approved cloud configuration changed")
        if action not in self.approval.get("actions", []):
            raise ValueError("action outside the exact approval")
        if self.config.get("prices_verified") is not True:
            raise ValueError("exact SKU/model prices remain unverified")
        if self.config.get("entitlements_verified") is not True:
            raise ValueError("quota/runtime/model entitlements remain unverified")
        if self.config.get("cloud_spend_authorized") is not True:
            raise ValueError("configuration retains the no-spend gate")
        if money(maximum_aud) < operation_bound(self.config, action):
            raise ValueError("requested reservation is below the priced worst-case envelope")
        context = json.loads(
            subprocess.check_output(
                [
                    "az",
                    "account",
                    "show",
                    "--output",
                    "json",
                    "--only-show-errors",
                ],
                text=True,
            )
        )
        role = {
            "assistant": "investigator",
            "judge": "investigator",
            "index_publish": "index_publisher",
            "promote": "model_owner",
        }.get(action)
        expected_identity = (
            self.config.get("principals", {}).get(role)
            if role
            else self.config.get("owner_identity")
        )
        if not expected_identity:
            raise ValueError("explicit operation identity is required")
        if (context.get("id"), context.get("tenantId"), context.get("user", {}).get("name")) != (
            self.config.get("subscription_id"),
            self.config.get("tenant_id"),
            expected_identity,
        ):
            raise ValueError("unexpected Azure account; no cloud calls were made")
        proof = self.reconciliation.get("initial_unused_subscription")
        if (
            self.reconciliation.get("complete_through_observed_time") is not True
            and proof is not None
            and (
                not isinstance(proof, dict)
                or (
                    proof.get("subscription_id"),
                    proof.get("tenant_id"),
                    proof.get("owner_identity"),
                )
                != (context["id"], context["tenantId"], context["user"]["name"])
            )
        ):
            raise ValueError("unused-subscription evidence belongs to another Azure identity")
        audit = self.reconciliation.get("held_envelope_audit")
        if audit is not None and (
            not isinstance(audit, dict)
            or any(
                audit.get(k) != self.config.get(k)
                for k in ("subscription_id", "tenant_id", "owner_identity")
            )
        ):
            raise ValueError("held-envelope audit belongs to another Azure configuration")
        self.ledger.reserve(
            batch_id,
            maximum_aud,
            self.approval,
            self.reconciliation,
            execution_kind=action,
            resume_existing=resume_existing,
        )
        self.started = True
        self.action = action
        self.batch_id = batch_id

    def __call__(self, action):
        if not self.started:
            raise ValueError("paid action requires an approved, reserved operation")
        if action in self.calls:
            permitted = {
                "assistant": {"inference", "embedding", "retrieval", "judge"},
                "judge": {"judge"},
                "index_publish": {"embedding", "index_write"},
            }
            if action not in permitted.get(self.action, set()):
                raise ValueError("service operation outside this reserved scope")
            if self.calls[action] >= self.config["request_limits"][action]:
                raise ValueError("operation request allowance exhausted")
            self.ledger.consume_request(
                self.batch_id, action, self.config["request_limits"][action]
            )
            self.calls[action] += 1
        elif action != self.action:
            raise ValueError("action outside this reserved operation")

    def embedding_inputs(self, texts):
        # For this bounded text corpus, UTF-8 bytes conservatively bound tokens.
        self.ledger.reserve_embedding_tokens(self.batch_id, sum(len(x.encode()) for x in texts))


def validate_endpoints(config):
    allowed = {
        "databricks_host": ".azuredatabricks.net",
        "foundry_endpoint": ".services.ai.azure.com",
        "search_endpoint": ".search.windows.net",
        "openai_endpoint": ".openai.azure.com",
    }
    for name, suffix in allowed.items():
        value = urlsplit(config[name])
        if value.scheme != "https" or not value.hostname or not value.hostname.endswith(suffix):
            raise ValueError(f"invalid approved {name}")
        if value.username or value.password or value.query or value.fragment:
            raise ValueError("credentials/query fragments cannot be embedded in endpoints")
    for name in ("catalog", "development_schema", "evaluation_schema", "evidence_schema"):
        identifier(config[name])


def clients(config, gate):
    if not gate.started or gate.action not in {"assistant", "index_publish"}:
        raise ValueError("model/Search clients require an approved reserved operation")
    validate_endpoints(config)
    from agent_framework.foundry import FoundryChatClient
    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.aio import AIProjectClient as AsyncAIProjectClient
    from azure.identity import AzureCliCredential
    from azure.identity.aio import AzureCliCredential as AsyncAzureCliCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient

    credential = AzureCliCredential(tenant_id=config["tenant_id"])
    project = AIProjectClient(endpoint=config["foundry_endpoint"], credential=credential)
    async_credential = AsyncAzureCliCredential(tenant_id=config["tenant_id"])
    async_project = AsyncAIProjectClient(
        endpoint=config["foundry_endpoint"],
        credential=async_credential,
    )
    chat = FoundryChatClient(
        project_client=async_project,
        model=config["generation_deployment"],
        function_invocation_configuration={
            "max_iterations": 3,
            "max_function_calls": 6,
            "max_duration_seconds": 90,
            "allow_concurrent_invocation": False,
            "terminate_on_unknown_calls": True,
            "include_detailed_errors": False,
        },
    )
    chat.client = chat.client.with_options(max_retries=0, timeout=90)
    return {
        "credential": credential,
        "async_credential": async_credential,
        "project": project,
        "chat": chat,
        "openai": project.get_openai_client(max_retries=0, timeout=90),
        "search": SearchClient(
            config["search_endpoint"], config["search_index"], credential, retry_total=0
        ),
        "indexes": SearchIndexClient(config["search_endpoint"], credential, retry_total=0),
    }


def freeze_inputs(data_dir, previous_dir, output_dir, *, code_sha, lock_sha256):
    """Export the selected recipe and parity inputs without a new fit or test read."""
    import mlflow
    import pandas as pd

    report_path = previous_dir / "development-report.json"
    report = json.loads(report_path.read_text())
    manifest_path = data_dir / "prepared/manifest.json"
    if digest(manifest_path) != report["data_manifest_sha256"]:
        raise ValueError("frozen data/selection identity changed")
    manifest = json.loads(manifest_path.read_text())
    training = data_dir / "prepared/training.parquet"
    if digest(training) != manifest["prepared_sha256"]:
        raise ValueError("training artifact changed")
    frame = pd.read_parquet(training)
    decision = frame.loc[
        frame.partition.eq("decision") & frame.eligible, manifest["features"]
    ].astype("float64")
    mlflow.set_tracking_uri(f"sqlite:///{(previous_dir / 'mlflow.sqlite').resolve()}")
    model = mlflow.pyfunc.load_model(report["model_uri"])
    predicted = model.predict(decision)
    output_dir.mkdir(parents=True, exist_ok=True)
    decision.to_parquet(output_dir / "parity-input.parquet", index=False)
    pd.DataFrame(predicted, columns=["other", "aps"]).to_parquet(
        output_dir / "parity-expected.parquet", index=False
    )
    selected = report["candidates"][report["selected_kind"]]
    recipe = {
        "schema_version": 1,
        "code_sha": code_sha,
        "lock_sha256": lock_sha256,
        "selection_report_sha256": digest(report_path),
        "data_manifest_sha256": digest(manifest_path),
        "training_sha256": digest(training),
        "config": selected["config"],
        "threshold": selected["metrics"]["threshold"],
        "features": manifest["features"],
        "missing_fraction_limit": manifest["missing_fraction_limit"],
        "split_sha256": manifest["split_sha256"],
        "parity_input_sha256": digest(output_dir / "parity-input.parquet"),
        "parity_expected_sha256": digest(output_dir / "parity-expected.parquet"),
        "external_test_opened": False,
        "fits_per_cloud_training_attempt": 2,
        "existing_local_fit_operations": sum(
            x["count"]
            for x in json.loads((data_dir / "prepared/fit-ledger.json").read_text())["attempts"]
        ),
    }
    save_json(output_dir / "recipe.json", recipe)
    return recipe


def train_registered(config, recipe_dir: Path, data_dir: Path, spark, *, tracking_client=None):
    """Databricks job body: two selected fits, MLflow/Delta input lineage and numeric UC version."""
    import mlflow
    import numpy as np
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from threadpoolctl import threadpool_limits

    from fleet_fault_triage.model import estimator, log_calibrated_model

    catalog, schema = identifier(config["catalog"]), identifier(config["development_schema"])
    model_name = f"{catalog}.{schema}.aps_classifier"
    recipe = json.loads((recipe_dir / "recipe.json").read_text())
    if digest(data_dir / "prepared/training.parquet") != recipe["training_sha256"]:
        raise ValueError("approved training bytes changed")
    for name, field in (
        ("parity-input.parquet", "parity_input_sha256"),
        ("parity-expected.parquet", "parity_expected_sha256"),
    ):
        if digest(recipe_dir / name) != recipe[field]:
            raise ValueError("parity artifact changed")
    frame = pd.read_parquet(data_dir / "prepared/training.parquet")
    table = f"{catalog}.{schema}.training"
    if not spark.catalog.tableExists(table):
        spark.createDataFrame(frame).write.format("delta").mode("errorifexists").saveAsTable(table)
    # An interrupted attempt can reuse only the identical immutable training snapshot.
    pd.testing.assert_frame_equal(
        spark.table(table).toPandas().sort_values("record_id").reset_index(drop=True),
        frame.sort_values("record_id").reset_index(drop=True),
        check_dtype=False,
    )
    # Serial job ownership plus Delta-persisted reservations preserve failed/interrupted fits.
    fit_table = f"{catalog}.{schema}.fit_attempts"
    spark.sql(f"CREATE TABLE IF NOT EXISTS {fit_table} (run_id STRING, fits INT) USING DELTA")
    used = spark.sql(f"SELECT coalesce(sum(fits),0) AS used FROM {fit_table}").first()["used"]
    if used + recipe["existing_local_fit_operations"] + 2 > 100:
        raise ValueError("cumulative local/cloud fit allowance exceeded")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(config["experiment_path"])
    with mlflow.start_run(run_name="frozen-recipe-cloud-proof") as run, threadpool_limits(limits=2):
        spark.createDataFrame([(run.info.run_id, 2)], "run_id string, fits int").write.mode(
            "append"
        ).saveAsTable(fit_table)
        mlflow.log_dict(recipe, "frozen-recipe.json")
        mlflow.log_params(
            {
                "source_code_sha": config["source_code_sha"],
                "lock_sha256": config["dependency_lock_sha256"],
                "data_manifest_sha256": recipe["data_manifest_sha256"],
                "split_sha256": recipe["split_sha256"],
                "threshold": recipe["threshold"],
                "fits_reserved": 2,
            }
        )
        mlflow.log_input(
            mlflow.data.from_spark(spark.table(table), table_name=table), context="training"
        )
        development = frame.loc[frame.partition.eq("development") & frame.eligible]
        calibration = frame.loc[frame.partition.eq("calibration") & frame.eligible]
        base = estimator(recipe["config"]).fit(development[recipe["features"]], development.label)
        model = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(
            calibration[recipe["features"]], calibration.label
        )
        parity = pd.read_parquet(recipe_dir / "parity-input.parquet").astype("float64")
        expected = pd.read_parquet(recipe_dir / "parity-expected.parquet").to_numpy()
        actual = model.predict_proba(parity)
        difference = float(np.max(np.abs(expected - actual)))
        mlflow.log_metric("parity_max_absolute_difference", difference)
        if not np.allclose(expected, actual, atol=1e-6, rtol=0):
            raise ValueError("Azure/local parity failed; external test stays sealed")
        logged = log_calibrated_model(model, development[recipe["features"]].head(3))
        restored = mlflow.pyfunc.load_model(logged.model_uri)
        if not np.allclose(restored.predict(parity), actual, atol=1e-6, rtol=0):
            raise ValueError("stored model prediction readback failed")
        version = mlflow.register_model(logged.model_uri, model_name)
        registry = tracking_client or mlflow.MlflowClient()
        saved = registry.get_model_version(model_name, version.version)
        if saved.run_id != run.info.run_id:
            raise ValueError("registered run identity differs")
        result = {
            "origin": "azure",
            "model_uri": f"models:/{model_name}/{saved.version}",
            "model_run_id": run.info.run_id,
            "parity_max_difference": difference,
            "persisted_model_readback_passed": True,
            "recipe_sha256": digest(recipe_dir / "recipe.json"),
        }
        mlflow.log_dict(result, "azure-training-readback.json")
    return result


def score_delta(frame, release: Release, model, batch_id, spark, table):
    """Serial scorer principal; Delta commit is atomic and immutable batch replays must agree."""
    import tempfile

    from fleet_fault_triage.lifecycle import CaseStore
    from fleet_fault_triage.scoring import input_identity, score_batch

    for part in table.split("."):
        identifier(part)
    if len(table.split(".")) != 3:
        raise ValueError("fully qualified project table required")
    with tempfile.TemporaryDirectory() as temporary:
        rows = score_batch(
            frame, release, model, batch_id, CaseStore(Path(temporary) / "batch.sqlite")
        )
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {table} (batch_id STRING, payload STRING, output_hash STRING) USING DELTA"
    )
    existing = (
        spark.table(table)
        .where(__import__("pyspark.sql.functions", fromlist=["col"]).col("batch_id") == batch_id)
        .collect()
    )
    payload = {
        "input_sha256": input_identity(frame, release.features),
        "release_sha256": release.sha256,
        "scores": rows,
    }
    output_hash = identity(payload)
    if existing:
        if len(existing) != 1 or existing[0]["output_hash"] != output_hash:
            raise ValueError("cloud batch replay changed identity/output")
    else:
        spark.createDataFrame(
            [(batch_id, json.dumps(payload, allow_nan=False), output_hash)],
            "batch_id string, payload string, output_hash string",
        ).write.mode("append").saveAsTable(table)
    saved = (
        spark.table(table)
        .where(__import__("pyspark.sql.functions", fromlist=["col"]).col("batch_id") == batch_id)
        .collect()
    )
    if len(saved) != 1 or identity(json.loads(saved[0]["payload"])) != output_hash:
        raise ValueError("cloud scoring readback failed")
    return payload
