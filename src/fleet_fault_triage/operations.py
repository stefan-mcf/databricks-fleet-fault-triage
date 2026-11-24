"""Manual project operations: explicit scope, official clients and persisted readback."""

import hashlib
import io
import json
import re
import subprocess
import time
import uuid
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

from fleet_fault_triage.cloud import (
    CloudGate,
    clients,
    identifier,
    operation_bound,
    validate_endpoints,
)
from fleet_fault_triage.contracts import Release, canonical, identity
from fleet_fault_triage.data import TEST, digest, save_json
from fleet_fault_triage.finops import CostLedger
from fleet_fault_triage.investigation import Investigator, prompt_text
from fleet_fault_triage.lifecycle import CaseStore
from fleet_fault_triage.retrieval import Corpus, SearchRunbooks, index_definition


def gate_from_files(config_path, approval_path, reconciliation_path, ledger_path):
    config = json.loads(config_path.read_text())
    approval = json.loads(approval_path.read_text())
    reconciled = json.loads(reconciliation_path.read_text())
    return CloudGate(config, approval, CostLedger(ledger_path), reconciled)


def verify_job_policy(settings, config, task):
    """Read back both the approved whole definition and its fixed cost boundaries."""
    maximum = 1800 if task in {"train", "evaluate"} else 900
    if identity(settings) != config.get("job_settings_sha256", {}).get(task):
        raise ValueError("deployed job definition changed from approved readback")
    if (
        settings.get("max_concurrent_runs") != 1
        or any(settings.get(k) for k in ("schedule", "continuous", "trigger"))
        or not 0 < settings.get("timeout_seconds", 0) <= maximum
    ):
        raise ValueError("unapproved deployed job scheduling/timeout policy")
    tasks = settings.get("tasks", [])
    if len(tasks) != 1:
        raise ValueError("exactly one bounded wheel task required")
    definition = tasks[0]
    cluster = definition.get("new_cluster", {})
    if (
        definition.get("max_retries", 0) != 0
        or not 0 < definition.get("timeout_seconds", 0) <= maximum
        or cluster.get("num_workers") != 0
        or cluster.get("spark_version") != config.get("runtime")
        or cluster.get("node_type_id") != config.get("node_type")
        or cluster.get("driver_node_type_id", cluster.get("node_type_id"))
        != config.get("node_type")
        or any(
            cluster.get(k)
            for k in (
                "autoscale",
                "instance_pool_id",
                "driver_instance_pool_id",
                "init_scripts",
                "docker_image",
            )
        )
        or not definition.get("python_wheel_task")
        or definition.get("existing_cluster_id")
    ):
        raise ValueError("unapproved deployed compute/task/retry policy")


def evaluation_files(config, root, manifest_path):
    """Bind four evaluation inputs without parsing the sealed benchmark file."""
    if digest(manifest_path) != config.get("evaluation_manifest_sha256"):
        raise ValueError("evaluation manifest differs from approved configuration")
    base = "/Volumes/fleet_fault_triage"
    if (
        config["catalog"] != "fleet_fault_triage"
        or config["training_data_dir"] != base + "/development/inputs/data"
        or config["evaluation_data_dir"] != base + "/evaluation/sealed/data"
        or config["release_path"] != base + "/evidence/control/evaluation/release.json"
        or config["freeze_path"] != base + "/evidence/control/evaluation/freeze.json"
    ):
        raise ValueError("evaluation inputs must remain in the approved project volumes")
    paths = {
        "release": config["release_path"],
        "freeze": config["freeze_path"],
        "training_manifest": config["training_data_dir"] + "/prepared/manifest.json",
        "external_test": config["evaluation_data_dir"] + "/raw/" + TEST,
    }
    manifest = json.loads(manifest_path.read_text())
    if set(manifest) != set(paths):
        raise ValueError("exact four evaluation inputs required")
    root = root.resolve()
    files, size = {}, 0
    for name, destination in paths.items():
        entry = manifest[name]
        local = (root / entry["local"]).resolve()
        expected_local = {
            "training_manifest": root / "data/prepared/manifest.json",
            "external_test": root / "data/raw" / TEST,
        }.get(name)
        if (
            not local.is_relative_to(root)
            or not local.is_file()
            or (expected_local is not None and local != expected_local)
            or (expected_local is None and not local.is_relative_to(root / "outputs/cloud-private"))
            or entry["destination"] != destination
            or digest(local) != entry["sha256"]
        ):
            raise ValueError("evaluation input path or bytes changed")
        size += local.stat().st_size
        files[name] = (local, destination, entry["sha256"])
    if size > 256 * 1024 * 1024:
        raise ValueError("evaluation upload exceeds the256MiB bound")
    release = Release.model_validate_json(files["release"][0].read_text())
    freeze = json.loads(files["freeze"][0].read_text())
    training = json.loads(files["training_manifest"][0].read_text())
    if (
        release.origin != "azure"
        or release.code_sha != config["source_code_sha"]
        or release.lock_sha256 != config["dependency_lock_sha256"]
        or release.data_sha256 != files["training_manifest"][2]
        or release.features != training["features"]
        or release.split_sha256 != training["split_sha256"]
        or freeze.get("release_sha256") != release.sha256
        or freeze.get("parity_passed") is not True
        or freeze.get("external_evaluation_authorized") is not True
        or freeze.get("external_test_sha256") != files["external_test"][2]
        or freeze.get("training_sha256") != training.get("prepared_sha256")
        or freeze.get("training_sha256") != digest(root / "data/prepared/training.parquet")
    ):
        raise ValueError("evaluation inputs differ from the frozen model/data identities")
    return files


def launch_job(gate, task, maximum_aud, private_output, *, root=None, evaluation_manifest=None):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import OperationFailed

    config = gate.config
    validate_endpoints(config)
    if task not in {"train", "evaluate", "score", "monitor"}:
        raise ValueError("unknown project job")
    if (task == "evaluate" and (evaluation_manifest is None or root is None)) or (
        task != "evaluate" and evaluation_manifest is not None
    ):
        raise ValueError(
            "evaluation requires its exact input manifest; other jobs cannot upload it"
        )
    evaluation_inputs = (
        evaluation_files(config, root, evaluation_manifest) if task == "evaluate" else None
    )
    batch_id = uuid.uuid4().hex
    gate.begin(task, batch_id, maximum_aud)
    workspace = WorkspaceClient(host=config["databricks_host"], auth_type="azure-cli")
    job_id = config["job_ids"][task]
    job = workspace.jobs.get(job_id)
    verify_job_policy(job.settings.as_dict(), config, task)
    role = {"train": "trainer", "evaluate": "evaluator", "score": "scorer", "monitor": "scorer"}[
        task
    ]
    if job.settings.run_as.service_principal_name != config["principals"][role]:
        raise ValueError(
            "deployed job principal differs from approved scope; reservation remains pending"
        )
    if job.settings.max_concurrent_runs != 1 or job.settings.schedule or job.settings.continuous:
        raise ValueError("deployed job execution policy differs from approved scope")
    permit = {
        "approved": True,
        "synthetic": False,
        "task": task,
        "reservation_id": batch_id,
        "config_sha256": identity(config),
        "expires_utc": gate.approval["expires_utc"],
        "source": gate.approval["source"],
    }
    permit_dir = config["permit_dir"].rstrip("/")
    control = (
        f"/Volumes/{identifier(config['catalog'])}/{identifier(config['evidence_schema'])}/control"
    )
    if (
        permit_dir != control + "/permits"
        or config["runtime_config_path"] != control + "/config.json"
    ):
        raise ValueError("configuration/permit storage outside the owner-written control volume")
    config_bytes = canonical(config).encode()
    workspace.files.upload(config["runtime_config_path"], io.BytesIO(config_bytes), overwrite=True)
    with workspace.files.download(config["runtime_config_path"]).contents as stream:
        if stream.read(len(config_bytes) + 1) != config_bytes:
            raise ValueError("saved runtime configuration differs from approved bytes")
    if evaluation_inputs is not None:
        uploaded = {}
        for name, (local, destination, expected) in evaluation_inputs.items():
            with local.open("rb") as stream:
                content = stream.read(256 * 1024 * 1024 + 1)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("evaluation bytes changed after preflight; lease retained")
            workspace.files.create_directory(destination.rsplit("/", 1)[0])
            workspace.files.upload(destination, io.BytesIO(content), overwrite=False)
            with workspace.files.download(destination).contents as stream:
                if stream.read(len(content) + 1) != content:
                    raise ValueError("saved evaluation input differs; lease retained")
            uploaded[name] = expected
        save_json(private_output / "evaluation-inputs-readback.json", uploaded)
    permit_path = permit_dir + "/" + batch_id + ".json"
    workspace.files.upload(permit_path, io.BytesIO(canonical(permit).encode()), overwrite=False)
    private_output.mkdir(parents=True, exist_ok=True)
    save_json(private_output / "permit.json", permit)
    pending = workspace.jobs.run_now(
        job_id=job_id,
        idempotency_token=batch_id,
        job_parameters={
            "permit": permit_path,
            "config": config["runtime_config_path"],
        },
    )
    submitted_run_id = pending.response.run_id
    save_json(
        private_output / "submission.json",
        {
            "provider_run_id": submitted_run_id,
            "config_sha256": identity(config),
            "reservation_id": batch_id,
        },
    )
    try:
        run = pending.result(timeout=timedelta(minutes=40))
    except OperationFailed:
        # The SDK raises on INTERNAL_ERROR instead of returning its terminal run.
        run = workspace.jobs.get_run(run_id=submitted_run_id)
    if run.run_id != submitted_run_id or run.job_id != job_id:
        raise ValueError("provider run differs from submitted job; lease retained")
    parameters = {p.name: p.value if p.value is not None else p.default for p in run.job_parameters}
    if (
        parameters.get("permit") != permit_path
        or parameters.get("config") != config["runtime_config_path"]
    ):
        raise ValueError("provider run parameters differ from approved scope; lease retained")
    state = run.state.as_dict()
    terminal = state.get("life_cycle_state") in {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}
    if not terminal:
        raise ValueError("provider terminal state unverified; lease and cost reservation retained")
    gate.ledger.finish_execution(batch_id, provider_terminal_verified=True)
    receipt = {
        "task": task,
        "provider_run_key": identity(str(run.run_id)),
        "state": state,
        "config_sha256": identity(config),
        "reservation_id": batch_id,
        "cost_reconciled": False,
    }
    save_json(private_output / "job-receipt.json", receipt)
    if state.get("result_state") != "SUCCESS":
        raise ValueError("provider job failed; failed attempt and cost reservation retained")
    result_path = config["output_dir"].rstrip("/") + "/" + batch_id + "/result.json"
    downloaded = workspace.files.download(result_path)
    result = json.loads(downloaded.contents.read())
    if result.get("config_sha256") != identity(config) or result.get("reservation_id") != batch_id:
        raise ValueError("job output differs from the exact reserved configuration")
    save_json(private_output / "result.json", result)
    if json.loads((private_output / "result.json").read_text()) != result:
        raise ValueError("saved result changed")
    return receipt


async def run_assistant(
    gate,
    root,
    state_path,
    release_path,
    case_ids,
    question,
    output,
    *,
    suite=None,
    mode="hybrid",
    heldout=False,
):
    corpus = Corpus.load(root / "resources/runbooks")
    release = Release.model_validate_json(release_path.read_text())
    if release.origin != "azure" or (
        release.prompt_sha256 != digest(root / "resources/prompts/investigator.md")
        or release.tool_policy_sha256 != digest(root / "src/fleet_fault_triage/investigation.py")
        or release.corpus_sha256 != corpus.version
        or release.index_version != gate.config["search_index"]
        or release.embedding_version != gate.config["embedding_version"]
        or release.generation_version != gate.config["generation_version"]
    ):
        raise ValueError("assistant release/prompt/corpus/deployment identity changed")
    if not case_ids or len(case_ids) > 120 or len(set(case_ids)) != len(case_ids):
        raise ValueError("bounded unique approved case list required")
    if mode not in {"keyword", "hybrid"}:
        raise ValueError("select keyword or hybrid grounded generation")
    queries = suite or [{"id": case, "case_id": case, "question": question} for case in case_ids]
    if not 1 <= len(queries) <= 80 or len({q["id"] for q in queries}) != len(queries):
        raise ValueError("bounded unique question suite required")
    if any(q["case_id"] not in case_ids for q in queries):
        raise ValueError("suite case outside the scoped evidence list")
    store = CaseStore(state_path)
    suite_hash = identity(queries)
    classification = gate.config.get("assistant_suites", {}).get(suite_hash)
    if classification not in {"development", "heldout"}:
        raise ValueError("assistant request is outside the approved frozen suites")
    heldout = classification == "heldout" or suite_hash == gate.config.get(
        "assistant_frozen_suite_sha256"
    )
    for case_id in case_ids:
        case = store.case(case_id)
        if case.release_sha256 != release.sha256 or case.origin != "azure":
            raise ValueError("suite case is not bound to this hosted release")
    if heldout:
        if identity(queries) != gate.config.get(
            "assistant_frozen_suite_sha256"
        ) or mode != gate.config.get("selected_retrieval_mode"):
            raise ValueError("held suite/mode differs from the approved freeze")
    gate.begin("assistant", uuid.uuid4().hex, str(operation_bound(gate.config, "assistant")))
    if heldout:
        store.reserve_assistant_suite(identity(queries), release.sha256)
    connected = clients(gate.config, gate)
    search = SearchRunbooks(
        connected["search"],
        connected["openai"],
        gate.config["embedding_deployment"],
        gate.config["embedding_dimensions"],
        corpus,
        guard=gate,
        deployment_version=gate.config["embedding_version"],
    )

    def retrieval(query, version, top_k):
        return search.query(
            query,
            version,
            top_k,
            mode=mode,
            timeout_seconds=investigator.deadline - time.monotonic(),
        )

    investigator = Investigator(store, release, corpus, retrieval, case_ids, guard=gate)
    saved = {}
    uncertain_terminal = False
    try:
        for item in queries:
            previous_trace_length = len(investigator.trace)
            brief = await investigator.run(
                item["case_id"], item["question"], connected["chat"], prompt_text(root)
            )
            saved[item["id"]] = brief.model_dump()
            if any("failure_type" in event for event in investigator.trace[previous_trace_length:]):
                # A transport timeout can leave backend work in flight. Preserve the
                # lease and partial draft instead of starting another paid case.
                uncertain_terminal = True
                break
        result = {
            "origin": "foundry",
            "release_sha256": release.sha256,
            "briefs": saved,
            "model_trace": list(investigator.trace),
            "retrieval_trace": list(search.traces),
            "human_review": False,
            "suite_sha256": identity(queries),
            "retrieval_mode": mode,
            "heldout": heldout,
            "operation_status": "attention_required" if uncertain_terminal else "terminal",
            "outcomes_sha256": identity(saved),
        }
        save_json(output, result)
        if json.loads(output.read_text()) != result:
            raise ValueError("brief saved readback failed")
        if not uncertain_terminal:
            gate.ledger.finish_execution(gate.batch_id, provider_terminal_verified=True)
    finally:
        await close_clients(connected)
    return {
        "cases": len(saved),
        "evidence_sha256": identity(result),
        "human_review": "pending",
        "operation_status": result["operation_status"],
    }


async def publish_index(gate, root, cache_path, maximum_aud):
    gate.begin("index_publish", uuid.uuid4().hex, maximum_aud)
    connected = clients(gate.config, gate)
    try:
        corpus = Corpus.load(root / "resources/runbooks")
        gate("index_write")
        definition = index_definition(
            gate.config["search_index"], gate.config["embedding_dimensions"]
        )
        connected["indexes"].create_index(definition)
        saved = connected["indexes"].get_index(definition.name)
        if saved.name != definition.name:
            raise ValueError("index schema readback failed")
        search = SearchRunbooks(
            connected["search"],
            connected["openai"],
            gate.config["embedding_deployment"],
            gate.config["embedding_dimensions"],
            corpus,
            guard=gate,
            deployment_version=gate.config["embedding_version"],
        )
        result = search.publish(cache_path)
        gate.ledger.finish_execution(gate.batch_id, provider_terminal_verified=True)
        return result
    finally:
        await close_clients(connected)


async def close_clients(connected):
    await connected["chat"].client.close()
    await connected["chat"].project_client.close()
    await connected["async_credential"].close()
    connected["openai"].close()
    connected["project"].close()
    connected["search"].close()
    connected["indexes"].close()
    connected["credential"].close()


async def execute_request(gate, action, root, request):
    """Project-specific dispatcher; this is deliberately not a general cloud client."""
    maximum = request["maximum_aud"]
    if action == "stage":
        from fleet_fault_triage.staging import stage_training

        return stage_training(gate, root, Path(request["manifest"]), maximum)
    if action in {"train", "evaluate", "score", "monitor"}:
        return launch_job(
            gate,
            action,
            maximum,
            Path(request["private_output"]),
            root=root,
            evaluation_manifest=Path(request["evaluation_manifest"])
            if action == "evaluate"
            else None,
        )
    if action == "assistant":
        return await run_assistant(
            gate,
            root,
            Path(request["state_path"]),
            Path(request["release_path"]),
            request["case_ids"],
            request["question"],
            Path(request["brief_output"]),
            suite=request.get("suite"),
            mode=request.get("mode", "hybrid"),
            heldout=request.get("heldout", False),
        )
    if action == "index_publish":
        return await publish_index(gate, root, Path(request["cache_path"]), maximum)
    if action == "judge":
        from importlib.metadata import version

        from azure.identity import AzureCliCredential

        from fleet_fault_triage.evaluation import foundry_judge

        if gate.config["evaluator_deployment"] != gate.config[
            "generation_deployment"
        ] or not gate.config.get("generation_version"):
            raise ValueError("judge must use the same pinned deployment/version as generation")
        cases = request["cases"]
        if len(cases) != 6 or identity(cases) != gate.config.get("judge_sample_sha256"):
            raise ValueError("exact six-case predeclared judge sample required")
        if any(
            sum(len(item[key].encode()) for key in ("query", "response", "context")) > 8000
            for item in cases
        ):
            raise ValueError("judge sample exceeds its request envelope")
        validate_endpoints(gate.config)
        gate.begin(action, uuid.uuid4().hex, maximum)
        credential = AzureCliCredential(tenant_id=gate.config["tenant_id"])
        model_config = {
            "azure_endpoint": gate.config["openai_endpoint"],
            "azure_deployment": gate.config["evaluator_deployment"],
            "api_version": gate.config["openai_api_version"],
        }
        result = {
            "origin": "foundry_evaluation",
            "human_review": False,
            "cases": [],
            "sample_sha256": identity(cases),
            "sdk_version": version("azure-ai-evaluation"),
            "deployment_version": gate.config["generation_version"],
            "deployment": gate.config["evaluator_deployment"],
            "maximum_attempts_reserved": 132,
        }
        try:
            for item in cases:
                scores = foundry_judge(
                    item["query"], item["response"], item["context"], model_config, credential, gate
                )
                result["cases"].append(
                    {"id": item["id"], "input_sha256": identity(item), "scores": scores}
                )
                save_json(Path(request["judge_output"]), result)
            if json.loads(Path(request["judge_output"]).read_text()) != result:
                raise ValueError("judge output readback failed")
            gate.ledger.finish_execution(gate.batch_id, provider_terminal_verified=True)
            return result
        finally:
            credential.close()
    if action == "provision":
        return provision(
            gate, root, Path(request["parameters_path"]), maximum, resume=request.get("resume")
        )
    if action == "cleanup":
        return cleanup(gate, request["inventory"], request["exported_evidence_verified"], maximum)
    if action == "promote":
        import os

        import mlflow
        from databricks.sdk import WorkspaceClient

        from fleet_fault_triage.registry import RegistryOwner, VolumeManifests

        gate.begin(action, uuid.uuid4().hex, maximum)
        validate_endpoints(gate.config)
        os.environ["DATABRICKS_HOST"] = gate.config["databricks_host"]
        os.environ["DATABRICKS_AUTH_TYPE"] = "azure-cli"
        mlflow.set_registry_uri("databricks-uc")
        release = Release.model_validate(request["release"])
        name = release.model_uri.removeprefix("models:/").rsplit("/", 1)[0]
        expected = f"{gate.config['catalog']}.{gate.config['development_schema']}.aps_classifier"
        if name != expected:
            raise ValueError("promotion outside approved project model")
        saved = CaseStore(Path(request["state_path"])).promote(
            release,
            request["expected_previous"],
            request["release_approval"],
            request["gates"],
            provider=RegistryOwner(
                mlflow.MlflowClient(),
                name,
                VolumeManifests(
                    WorkspaceClient(
                        host=gate.config["databricks_host"], auth_type="azure-cli"
                    ).files,
                    f"/Volumes/{gate.config['catalog']}/{gate.config['evidence_schema']}/releases",
                ),
            ),
        )
        gate.ledger.finish_execution(gate.batch_id, provider_terminal_verified=True)
        return {"release_sha256": saved, "alias": "champion", "readback": True}
    raise ValueError("unknown project operation")


def resource_scope(config, resource_id):
    prefix = (
        f"/subscriptions/{config['subscription_id']}/resourceGroups/{config['resource_group']}/"
    )
    if not resource_id.lower().startswith(prefix.lower()) or ".." in resource_id.split("/"):
        raise ValueError("resource outside the approved project group")


def verify_provision_recovery(config, root, resume):
    """Only dependency additions may repair the original, already-reserved topology."""
    previous = json.loads(Path(resume["previous_config"]).read_text())
    receipt = json.loads(Path(resume["previous_receipt"]).read_text())
    if (
        identity(previous) != receipt.get("config_sha256")
        or receipt.get("operation") != "Guarded first provision"
        or receipt.get("status") != "failed_requires_readback"
        or receipt.get("provision_reservation_requested_aud") != 25
    ):
        raise ValueError("original failed provision receipt/configuration mismatch")
    allowed = {"azure_template_sha256", "source_code_sha", "runtime_identity"}
    scopes = []
    for source in (previous, config):
        scope = deepcopy({k: v for k, v in source.items() if k not in allowed})
        for quote in scope.get("quotes", {}).values():
            if isinstance(quote, dict):
                quote.pop("observed_utc", None)
                for component in quote.get("components", []):
                    component.pop("rate_aud", None)
        scopes.append(scope)
    if scopes[0] != scopes[1]:
        raise ValueError("recovery cannot change resource, parameter, quote or identity scope")
    old_file = Path(resume["previous_template"])
    if digest(old_file) != previous.get("azure_template_sha256"):
        raise ValueError("original approved ARM template changed")
    old = json.loads(old_file.read_text())
    new = json.loads((root / "resources/azure/main.json").read_text())
    if len(old.get("resources", [])) != len(new.get("resources", [])):
        raise ValueError("recovery resource topology changed")
    for before, after in zip(old["resources"], new["resources"], strict=True):
        if not set(before.get("dependsOn", [])) <= set(after.get("dependsOn", [])):
            raise ValueError("recovery cannot remove deployment dependencies")
        before.pop("dependsOn", None)
        after.pop("dependsOn", None)
    if old != new:
        raise ValueError("recovery may only add resource dependencies")


def provision(gate, root, parameters_path, maximum_aud, *, resume=None):
    if resume:
        verify_provision_recovery(gate.config, root, resume)
    config = gate.config
    if not re.fullmatch(r"[a-z0-9-]{3,64}", config["resource_group"]):
        raise ValueError("invalid project resource group")
    values = json.loads(parameters_path.read_text())["parameters"]
    if identity(values) != config["azure_parameters_sha256"]:
        raise ValueError("ARM parameters differ from owner-approved configuration")
    if digest(root / "resources/azure/main.json") != config["azure_template_sha256"]:
        raise ValueError("ARM template differs from owner-approved configuration")
    if resume:
        gate.begin("provision", resume["reservation_id"], maximum_aud, resume_existing=True)
    else:
        gate.begin("provision", uuid.uuid4().hex, maximum_aud)
    subprocess.run(
        [
            "az",
            "group",
            "create",
            "--name",
            config["resource_group"],
            "--location",
            "australiaeast",
            "--subscription",
            config["subscription_id"],
            "--tags",
            "project=fleet-fault-triage",
            "--output",
            "none",
            "--only-show-errors",
        ],
        check=True,
    )
    subprocess.run(
        [
            "az",
            "deployment",
            "group",
            "create",
            "--resource-group",
            config["resource_group"],
            "--name",
            "fleet-fault-triage-proof-recovery-1" if resume else "fleet-fault-triage-proof",
            "--subscription",
            config["subscription_id"],
            "--template-file",
            str(root / "resources/azure/main.json"),
            "--parameters",
            "@" + str(parameters_path),
            "--output",
            "none",
            "--only-show-errors",
        ],
        check=True,
    )
    inventory = json.loads(
        subprocess.check_output(
            [
                "az",
                "resource",
                "list",
                "--resource-group",
                config["resource_group"],
                "--subscription",
                config["subscription_id"],
                "--output",
                "json",
                "--only-show-errors",
            ],
            text=True,
        )
    )
    roots = [x for x in inventory if len(x["id"].split("/providers/", 1)[1].split("/")) == 3]
    if not roots or any(x.get("tags", {}).get("project") != "fleet-fault-triage" for x in roots):
        raise ValueError("unexpected/unattributed resource in project group")
    gate.ledger.finish_execution(gate.batch_id, provider_terminal_verified=True)
    return inventory


def cleanup(gate, inventory, exported_evidence_verified, maximum_aud):
    if exported_evidence_verified is not True or not inventory:
        raise ValueError("verify exported evidence and exact nonempty inventory before teardown")
    gate.begin("cleanup", uuid.uuid4().hex, maximum_aud)
    current = json.loads(
        subprocess.check_output(
            [
                "az",
                "resource",
                "list",
                "--resource-group",
                gate.config["resource_group"],
                "--subscription",
                gate.config["subscription_id"],
                "--output",
                "json",
                "--only-show-errors",
            ],
            text=True,
        )
    )
    if {x["id"].lower() for x in current} != {x["id"].lower() for x in inventory}:
        raise ValueError("inventory changed; review exact teardown scope again")
    roots = [x for x in current if len(x["id"].split("/providers/", 1)[1].split("/")) == 3]
    removed = []
    for resource in roots:
        resource_scope(gate.config, resource["id"])
        if resource.get("tags", {}).get("project") != "fleet-fault-triage":
            raise ValueError("refuse teardown of another/shared resource")
        subprocess.run(
            [
                "az",
                "resource",
                "delete",
                "--ids",
                resource["id"],
                "--subscription",
                gate.config["subscription_id"],
                "--output",
                "none",
                "--only-show-errors",
            ],
            check=True,
        )
        remaining = json.loads(
            subprocess.check_output(
                [
                    "az",
                    "resource",
                    "list",
                    "--resource-group",
                    gate.config["resource_group"],
                    "--subscription",
                    gate.config["subscription_id"],
                    "--output",
                    "json",
                    "--only-show-errors",
                ],
                text=True,
            )
        )
        if any(x["id"].lower() == resource["id"].lower() for x in remaining):
            raise ValueError("resource deletion not verified")
        removed.append(identity(resource["id"]))
    managed = gate.config["managed_resource_group"]
    if not re.fullmatch(r"[a-z0-9-]{3,64}", managed) or managed == gate.config["resource_group"]:
        raise ValueError("exact managed resource group required for orphan readback")
    exists = subprocess.check_output(
        [
            "az",
            "group",
            "exists",
            "--name",
            managed,
            "--subscription",
            gate.config["subscription_id"],
            "--output",
            "json",
        ],
        text=True,
    )
    orphans = (
        json.loads(
            subprocess.check_output(
                [
                    "az",
                    "resource",
                    "list",
                    "--resource-group",
                    managed,
                    "--subscription",
                    gate.config["subscription_id"],
                    "--output",
                    "json",
                    "--only-show-errors",
                ],
                text=True,
            )
        )
        if json.loads(exists)
        else []
    )
    remaining = json.loads(
        subprocess.check_output(
            [
                "az",
                "resource",
                "list",
                "--resource-group",
                gate.config["resource_group"],
                "--subscription",
                gate.config["subscription_id"],
                "--output",
                "json",
                "--only-show-errors",
            ],
            text=True,
        )
    )
    if remaining or orphans:
        raise ValueError("teardown incomplete; preserve inventory, lease and cleanup reserve")
    gate.ledger.finish_execution(gate.batch_id, provider_terminal_verified=True)
    return {
        "removed_resource_keys": removed,
        "billing_reconciliation": "pending",
        "project_resources_remaining": 0,
        "managed_resources_remaining": 0,
        "inventory_sha256": identity(inventory),
    }
