"""Synthetic storage/auth fixtures; no cloud credentials, data or paid calls."""

import io
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from fleet_fault_triage.data import digest
from fleet_fault_triage.staging import limited_client, stage_training, training_files


@pytest.mark.parametrize("fault", ["transient", "persistent", "other", "wrong_identity"])
def test_fresh_secret_auth_retries_are_bounded_read_only_and_key_preserving(monkeypatch, fault):
    attempts = []
    sleep = Mock()
    config = {"databricks_host": "SYNTHETIC_HOST", "tenant_id": "SYNTHETIC_TENANT"}
    client = NS(
        current_user=NS(me=lambda: NS(user_name="OTHER" if fault == "wrong_identity" else "APP"))
    )

    def factory(**kwargs):
        attempts.append(kwargs)
        if fault == "persistent" or (fault == "transient" and len(attempts) <= 2):
            raise ValueError("invalid_client: AADSTS7000215: SYNTHETIC propagation")
        if fault == "other":
            raise ValueError("SYNTHETIC unrelated error")
        return client

    monkeypatch.setattr("databricks.sdk.WorkspaceClient", factory)
    monkeypatch.setattr("fleet_fault_triage.staging.time.sleep", sleep)
    if fault == "transient":
        assert limited_client(config, "APP", "SYNTHETIC_SECRET") is client
        assert len(attempts) == 3
        assert [c.args[0] for c in sleep.call_args_list] == [2, 4]
    else:
        with pytest.raises(ValueError):
            limited_client(config, "APP", "SYNTHETIC_SECRET")
        assert len(attempts) == (5 if fault == "persistent" else 1)
        assert [c.args[0] for c in sleep.call_args_list] == (
            [2, 4, 8, 16] if fault == "persistent" else []
        )
    assert all(x == attempts[0] for x in attempts)
    assert attempts[0]["azure_client_secret"] == "SYNTHETIC_SECRET"


@pytest.fixture
def stage_input(tmp_path):
    base = "/Volumes/fleet_fault_triage"
    config = {
        "catalog": "fleet_fault_triage",
        "development_schema": "development",
        "evaluation_schema": "evaluation",
        "evidence_schema": "evidence",
        "runtime_config_path": base + "/evidence/control/config.json",
        "permit_dir": base + "/evidence/control/permits",
        "databricks_host": "https://adb-synthetic.azuredatabricks.net",
        "tenant_id": "SYNTHETIC_TENANT",
        "owner_identity": "SYNTHETIC_OWNER",
        "storage_root": "abfss://SYNTHETIC/",
        "source_code_sha": "SYNTHETIC_SOURCE",
        "dependency_lock_sha256": "SYNTHETIC_LOCK",
        "training_data_dir": base + "/development/inputs/data",
        "recipe_dir": base + "/development/inputs/recipe",
        "principals": {
            r: "SYNTHETIC_" + r for r in ("trainer", "evaluator", "scorer", "model_owner")
        },
    }
    destinations = {
        "training": config["training_data_dir"] + "/prepared/training.parquet",
        "recipe": config["recipe_dir"] + "/recipe.json",
        "parity_input": config["recipe_dir"] + "/parity-input.parquet",
        "parity_expected": config["recipe_dir"] + "/parity-expected.parquet",
        "wheel": base + "/evidence/control/package/fleet_fault_triage-0.1.0-py3-none-any.whl",
        "requirements": base + "/evidence/control/requirements-cloud.txt",
    }
    manifest = {}
    for key, destination in destinations.items():
        local = tmp_path / key
        local.write_bytes(b"SYNTHETIC " + key.encode())
        manifest[key] = {"local": local.name, "destination": destination, "sha256": digest(local)}
    recipe = {
        "code_sha": config["source_code_sha"],
        "lock_sha256": config["dependency_lock_sha256"],
        **{
            field: manifest[key]["sha256"]
            for key, field in (
                ("training", "training_sha256"),
                ("parity_input", "parity_input_sha256"),
                ("parity_expected", "parity_expected_sha256"),
            )
        },
    }
    (tmp_path / "recipe").write_text(json.dumps(recipe))
    manifest["recipe"]["sha256"] = digest(tmp_path / "recipe")
    file = tmp_path / "manifest.json"
    file.write_text(json.dumps(manifest))
    config["staging_manifest_sha256"] = digest(file)
    return config, file


@pytest.mark.parametrize("fault", ["manifest", "bytes", "path", "destination", "recipe", "sealed"])
def test_changed_or_unscoped_training_inputs_fail_before_reservation(
    stage_input, monkeypatch, fault
):
    config, file = stage_input
    manifest = json.loads(file.read_text())
    if fault == "manifest":
        config["staging_manifest_sha256"] = "UNAPPROVED"
    elif fault == "bytes":
        (file.parent / "training").write_text("CHANGED")
    elif fault == "path":
        manifest["wheel"]["local"] = "../outside-project"
    elif fault == "destination":
        manifest["wheel"]["destination"] = "/Volumes/OTHER/secret"
    elif fault == "recipe":
        value = json.loads((file.parent / "recipe").read_text())
        value["code_sha"] = "OTHER_SOURCE"
        (file.parent / "recipe").write_text(json.dumps(value))
        manifest["recipe"]["sha256"] = digest(file.parent / "recipe")
    else:
        config["training_data_dir"] = "/Volumes/fleet_fault_triage/evaluation/sealed"
    if fault != "manifest":
        file.write_text(json.dumps(manifest))
        config["staging_manifest_sha256"] = digest(file)
    gate = Mock(config=config)
    monkeypatch.setattr("fleet_fault_triage.staging.validate_endpoints", lambda _: None)
    with pytest.raises(ValueError):
        stage_training(gate, file.parent, file, "5")
    gate.begin.assert_not_called()


@pytest.fixture
def provider(stage_input, monkeypatch):
    from databricks.sdk.errors import PermissionDenied
    from databricks.sdk.service.catalog import VolumeType

    config, file = stage_input
    owner = Mock()
    owner.external_locations.get.return_value = NS(url=config["storage_root"])
    owner.current_user.me.return_value = NS(user_name=config["owner_identity"])
    owner.catalogs.get.return_value = NS(
        storage_root=config["storage_root"] + "managed", owner=config["owner_identity"]
    )
    owner.schemas.get.return_value = NS(
        storage_root=None,
        storage_location=config["storage_root"] + "managed/schema",
        owner=config["owner_identity"],
    )
    owner.volumes.read.side_effect = lambda name: NS(
        **dict(zip(("catalog_name", "schema_name", "name"), name.split("."), strict=True)),
        volume_type=VolumeType.MANAGED,
        owner=config["owner_identity"],
        storage_location=config["storage_root"] + "managed/volumes/" + name,
    )
    grants, saved, roles = {}, {}, {}

    def update(kind, name, *, changes):
        for change in changes:
            grants.setdefault((kind, name, change.principal), set()).update(change.add or [])

    def read_grants(kind, name):
        entries = [
            NS(principal=p, privileges=list(v))
            for (k, n, p), v in grants.items()
            if (k, n) == (kind, name)
        ]
        return NS(privilege_assignments=entries, as_dict=lambda: {"synthetic": True})

    def upload(name, body, **_):
        saved[name] = body.read()

    owner.grants.update.side_effect = update
    owner.grants.get.side_effect = read_grants

    def effective(kind, name, *, principal):
        values = grants.get((kind, name, principal), set())
        return NS(
            privilege_assignments=[
                NS(principal=principal, privileges=[NS(privilege=p) for p in values])
            ],
            next_page_token=None,
            as_dict=lambda: {"synthetic_effective": True},
        )

    owner.grants.get_effective.side_effect = effective
    owner.files.upload.side_effect = upload
    owner.files.download.side_effect = lambda name: NS(contents=io.BytesIO(saved[name]))
    for role in ("trainer", "evaluator", "scorer"):
        monkeypatch.setenv("FFT_STAGE_" + role.upper() + "_SECRET", "SYNTHETIC_SECRET")
        limited = Mock()
        limited.current_user.me.return_value = NS(user_name=config["principals"][role])

        def read(name, role=role):
            if "/evaluation/sealed/" in name and role != "evaluator":
                raise PermissionDenied("synthetic sealed denial")
            return NS(contents=io.BytesIO(saved[name]))

        def write(name, body, **kwargs):
            if "/evidence/control/" in name or "/development/inputs/" in name:
                raise PermissionDenied("synthetic control denial")
            upload(name, body, **kwargs)

        limited.files.download.side_effect = read
        limited.files.upload.side_effect = write
        roles[role] = limited
    monkeypatch.setattr("fleet_fault_triage.staging.validate_endpoints", lambda _: None)
    monkeypatch.setattr(
        "databricks.sdk.WorkspaceClient",
        lambda **kwargs: (
            owner
            if kwargs["auth_type"] == "azure-cli"
            else roles[kwargs["azure_client_id"].split("SYNTHETIC_")[1]]
        ),
    )
    gate = Mock(config=config, batch_id="SYNTHETIC_STAGE", started=False)
    return config, file, gate, owner, roles, saved


def test_staging_reads_back_artifacts_with_the_actual_sdk_stream(provider):
    config, file, gate, owner, roles, saved = provider
    from databricks.sdk._base_client import _StreamingResponse

    def provider_stream(name):
        body = saved[name]
        return NS(
            contents=_StreamingResponse(
                NS(iter_content=lambda **_: iter([body[:3], body[3:]]), close=Mock())
            )
        )

    owner.files.download.side_effect = provider_stream
    files = training_files(config, file.parent, file)
    assert len(files) == 6
    result = stage_training(gate, file.parent, file, "5")
    assert set(result["access_probes"]) == set(roles)
    assert result["operator_credential_cleanup_required"] is True
    assert result["compute_started"] == 0 and result["external_test_uploaded"] is False
    assert "SYNTHETIC_SECRET" not in json.dumps(result)
    owner.token_management.create_obo_token.assert_not_called()
    gate.ledger.finish_execution.assert_not_called()
    assert result["execution_lease_release_pending_cleanup"] is True
    gate.begin.assert_called_once()
    assert json.loads(saved[config["runtime_config_path"]]) == config


@pytest.mark.parametrize("fault", ["effective_write", "owner", "unknown_principal"])
def test_surplus_access_or_ownership_blocks_frozen_input_upload(provider, fault):
    from databricks.sdk.service.catalog import Privilege

    config, file, gate, owner, _, _ = provider
    if fault == "owner":
        owner.schemas.get.return_value.owner = config["principals"]["trainer"]
    elif fault == "effective_write":
        original = owner.grants.get_effective.side_effect

        def effective(kind, name, *, principal):
            result = original(kind, name, principal=principal)
            if (
                name == "fleet_fault_triage.development.inputs"
                and principal == config["principals"]["trainer"]
            ):
                result.privilege_assignments[0].privileges.append(
                    NS(privilege=Privilege.WRITE_VOLUME)
                )
            return result

        owner.grants.get_effective.side_effect = effective
    else:
        original = owner.grants.get.side_effect

        def direct(kind, name):
            result = original(kind, name)
            if kind == "volume":
                result.privilege_assignments.append(
                    NS(principal="UNAPPROVED", privileges=[Privilege.READ_VOLUME])
                )
            return result

        owner.grants.get.side_effect = direct
    with pytest.raises(ValueError, match="owned|privileges"):
        stage_training(gate, file.parent, file, "5")
    owner.files.upload.assert_not_called()
    gate.ledger.finish_execution.assert_not_called()


def test_unexpected_sealed_access_retains_execution_lease(provider):
    _, file, gate, owner, roles, saved = provider
    roles["trainer"].files.download.side_effect = lambda name: NS(contents=io.BytesIO(saved[name]))
    with pytest.raises(ValueError, match="sealed volume"):
        stage_training(gate, file.parent, file, "5")
    gate.ledger.finish_execution.assert_not_called()


@pytest.mark.parametrize("fault", ["external", "outside_volume", "outside_schema"])
def test_unapproved_backing_storage_blocks_grants_and_uploads(provider, fault):
    from databricks.sdk.service.catalog import VolumeType

    _, file, gate, owner, _, _ = provider
    if fault == "outside_schema":
        owner.schemas.get.return_value.storage_root = "abfss://OTHER/managed"
    else:
        original = owner.volumes.read.side_effect

        def volume(name):
            value = original(name)
            if fault == "external":
                value.volume_type = VolumeType.EXTERNAL
            else:
                value.storage_location = "abfss://OTHER/managed/volume"
            return value

        owner.volumes.read.side_effect = volume
    with pytest.raises(ValueError, match="storage escapes"):
        stage_training(gate, file.parent, file, "5")
    owner.grants.update.assert_not_called()
    owner.files.upload.assert_not_called()
    gate.ledger.finish_execution.assert_not_called()
