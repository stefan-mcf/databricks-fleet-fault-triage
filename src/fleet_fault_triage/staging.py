"""Fixed project storage/grants and training inputs; no SQL compute or job execution."""

import hashlib
import io
import json
import os
import time
import uuid
from pathlib import Path
from urllib.parse import unquote, urlsplit

from fleet_fault_triage.cloud import identifier, validate_endpoints
from fleet_fault_triage.contracts import canonical, identity
from fleet_fault_triage.data import digest


def limited_client(config, application_id, secret):
    """Retry only fresh-secret rejection; reuse the same key and perform read-only auth checks."""
    from databricks.sdk import WorkspaceClient

    for attempt in range(5):
        try:
            client = WorkspaceClient(
                host=config["databricks_host"],
                auth_type="azure-client-secret",
                azure_client_id=application_id,
                azure_client_secret=secret,
                azure_tenant_id=config["tenant_id"],
            )
            actual = client.current_user.me().user_name
        except ValueError as error:
            if "AADSTS7000215" not in str(error) or attempt == 4:
                raise
            # Entra replicas may reject a newly added key before it propagates.
            time.sleep(2 ** (attempt + 1))
            continue
        if actual != application_id:
            raise ValueError("limited-principal authentication preflight failed")
        return client


def download_bytes(client, path, maximum):
    with client.files.download(path).contents as stream:
        body = stream.read(maximum + 1)
    if len(body) > maximum:
        raise ValueError("provider file exceeds the approved byte bound")
    return body


def owned_storage(location, boundary):
    if not isinstance(location, str):
        return False
    actual, expected = urlsplit(location), urlsplit(boundary)
    path, root = unquote(actual.path), unquote(expected.path).rstrip("/")
    return (
        actual.scheme == expected.scheme == "abfss"
        and actual.netloc.lower() == expected.netloc.lower()
        and not actual.query
        and not actual.fragment
        and ".." not in path.split("/")
        and (path == root or path.startswith(root + "/"))
    )


def training_files(config, root: Path, manifest_path: Path):
    if digest(manifest_path) != config.get("staging_manifest_sha256"):
        raise ValueError("staging manifest differs from the approved configuration")
    manifest = json.loads(manifest_path.read_text())
    base = f"/Volumes/{identifier(config['catalog'])}"
    if (config["training_data_dir"], config["recipe_dir"]) != (
        base + "/development/inputs/data",
        base + "/development/inputs/recipe",
    ):
        raise ValueError("training inputs must remain in the development volume")
    paths = {
        "training": config["training_data_dir"] + "/prepared/training.parquet",
        "recipe": config["recipe_dir"] + "/recipe.json",
        "parity_input": config["recipe_dir"] + "/parity-input.parquet",
        "parity_expected": config["recipe_dir"] + "/parity-expected.parquet",
        "wheel": base + "/evidence/control/package/fleet_fault_triage-0.1.0-py3-none-any.whl",
        "requirements": base + "/evidence/control/requirements-cloud.txt",
    }
    if set(manifest) != set(paths) or config["catalog"] != "fleet_fault_triage":
        raise ValueError("exact six project training files required")
    files, size = {}, 0
    for key, target in paths.items():
        entry = manifest[key]
        local = (root / entry["local"]).resolve()
        if (
            not local.is_relative_to(root.resolve())
            or not local.is_file()
            or entry["destination"] != target
            or not target.startswith(base + "/")
            or ".." in target.split("/")
            or digest(local) != entry["sha256"]
        ):
            raise ValueError("staging path or source bytes changed")
        size += local.stat().st_size
        files[key] = (local, target, entry["sha256"])
    if size > 256 * 1024 * 1024:
        raise ValueError("staging exceeds the256MiB transfer bound")
    recipe = json.loads(files["recipe"][0].read_text())
    if (
        recipe["code_sha"] != config["source_code_sha"]
        or recipe["lock_sha256"] != config["dependency_lock_sha256"]
        or any(
            recipe[field] != digest(files[key][0])
            for key, field in (
                ("training", "training_sha256"),
                ("parity_input", "parity_input_sha256"),
                ("parity_expected", "parity_expected_sha256"),
            )
        )
    ):
        raise ValueError("staging differs from the frozen recipe/source")
    return files


def stage_training(gate, root, manifest_path, maximum_aud, *, credentials=None):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import NotFound, PermissionDenied
    from databricks.sdk.service.catalog import PermissionsChange, Privilege, VolumeType

    config = gate.config
    validate_endpoints(config)
    files = training_files(config, root, manifest_path)
    if (config["development_schema"], config["evaluation_schema"], config["evidence_schema"]) != (
        "development",
        "evaluation",
        "evidence",
    ):
        raise ValueError("staging requires the fixed project schemas")
    catalog = identifier(config["catalog"])
    expected_control = f"/Volumes/{catalog}/evidence/control/config.json"
    if config["runtime_config_path"] != expected_control:
        raise ValueError("runtime configuration outside the owned control volume")
    permit_dir = expected_control.rsplit("/", 1)[0] + "/permits"
    if config["permit_dir"] != permit_dir:
        raise ValueError("permits outside the owned control volume")
    if gate.started is not True:
        gate.begin("stage", uuid.uuid4().hex, maximum_aud)
    elif gate.action != "stage":
        raise ValueError("existing reservation is not a storage stage")
    client = WorkspaceClient(host=config["databricks_host"], auth_type="azure-cli")
    owner = client.current_user.me().user_name
    if owner.casefold() != config["owner_identity"].casefold():
        raise ValueError("unexpected storage administrator")
    limited_clients = {}
    for role in ("trainer", "evaluator", "scorer"):
        secret = (os.environ if credentials is None else credentials).get(
            "FFT_STAGE_" + role.upper() + "_SECRET"
        )
        if not secret:
            raise ValueError("temporary limited-principal credentials are required")
        limited_clients[role] = limited_client(config, config["principals"][role], secret)
    location = client.external_locations.get("fft_storage")
    if location.url.rstrip("/") != config["storage_root"].rstrip("/"):
        raise ValueError("external location differs from the approved storage")
    storage_root = config["storage_root"].rstrip("/") + "/managed"
    try:
        saved = client.catalogs.get(catalog)
    except NotFound:
        client.catalogs.create(catalog, storage_root=storage_root)
        saved = client.catalogs.get(catalog)
    if saved.storage_root.rstrip("/") != storage_root or saved.owner.casefold() != owner.casefold():
        raise ValueError("catalog managed location changed")
    grants = client.grants.get("catalog", catalog)
    for assignment in grants.privilege_assignments or []:
        if assignment.principal == "account users" and assignment.privileges:
            client.grants.update(
                "catalog",
                catalog,
                changes=[
                    PermissionsChange(principal="account users", remove=assignment.privileges)
                ],
            )
    volumes = {
        "development": ["inputs"],
        "evaluation": ["sealed"],
        "evidence": ["control", "artifacts", "releases"],
    }
    for schema, names in volumes.items():
        full = f"{catalog}.{schema}"
        try:
            client.schemas.get(full)
        except NotFound:
            client.schemas.create(schema, catalog)
        saved_schema = client.schemas.get(full)
        if saved_schema.owner.casefold() != owner.casefold():
            raise ValueError("project schema is not owned by the administrator")
        for location in (saved_schema.storage_root, saved_schema.storage_location):
            if location is not None and not owned_storage(location, storage_root):
                raise ValueError("schema storage escapes the approved managed boundary")
        for name in names:
            full_volume = full + "." + name
            try:
                client.volumes.read(full_volume)
            except NotFound:
                client.volumes.create(catalog, schema, name, VolumeType.MANAGED)
            volume = client.volumes.read(full_volume)
            if volume.owner.casefold() != owner.casefold():
                raise ValueError("project volume is not owned by the administrator")
            if (volume.catalog_name, volume.schema_name, volume.name) != (catalog, schema, name):
                raise ValueError("saved project volume identity changed")
            if volume.volume_type != VolumeType.MANAGED or not owned_storage(
                volume.storage_location, storage_root
            ):
                raise ValueError("volume storage escapes the approved managed boundary")
    grant_receipts, policy = {}, {}
    for role in ("trainer", "evaluator", "scorer", "model_owner"):
        principal = config["principals"][role]
        permissions = [
            ("catalog", catalog, ["USE_CATALOG"]),
            ("schema", f"{catalog}.development", ["USE_SCHEMA"]),
            ("schema", f"{catalog}.evidence", ["USE_SCHEMA"]),
        ]
        if role == "trainer":
            permissions += [("schema", f"{catalog}.development", ["CREATE_TABLE", "CREATE_MODEL"])]
        if role == "evaluator":
            permissions += [
                ("schema", f"{catalog}.evaluation", ["USE_SCHEMA", "CREATE_TABLE"]),
                ("volume", f"{catalog}.evaluation.sealed", ["READ_VOLUME"]),
            ]
        if role in {"trainer", "evaluator"}:
            permissions += [("volume", f"{catalog}.development.inputs", ["READ_VOLUME"])]
        if role in {"trainer", "evaluator", "scorer"}:
            permissions += [
                ("volume", f"{catalog}.evidence.control", ["READ_VOLUME"]),
                ("volume", f"{catalog}.evidence.artifacts", ["READ_VOLUME", "WRITE_VOLUME"]),
            ]
        if role == "scorer":
            permissions += [("schema", f"{catalog}.evidence", ["CREATE_TABLE"])]
        if role == "model_owner":
            permissions += [
                ("volume", f"{catalog}.evidence.releases", ["READ_VOLUME", "WRITE_VOLUME"])
            ]
        for kind, name, values in permissions:
            wanted = [Privilege(x) for x in values]
            policy.setdefault((kind, name, principal), set()).update(wanted)
            client.grants.update(
                kind, name, changes=[PermissionsChange(principal=principal, add=wanted)]
            )
            readback = client.grants.get(kind, name)
            actual = {
                p
                for a in (readback.privilege_assignments or [])
                if a.principal == principal
                for p in (a.privileges or [])
            }
            if not set(wanted) <= actual:
                raise ValueError("saved data grant differs from approved role")
            grant_receipts[f"{role}:{kind}:{name}"] = identity(readback.as_dict())
    objects = [("catalog", catalog)] + [("schema", catalog + "." + schema) for schema in volumes]
    objects += [
        ("volume", f"{catalog}.{schema}.{name}")
        for schema, names in volumes.items()
        for name in names
    ]
    principals = {
        config["principals"][role] for role in ("trainer", "evaluator", "scorer", "model_owner")
    }
    for kind, name in objects:
        direct = client.grants.get(kind, name)
        if any(
            a.privileges and a.principal not in principals | {owner}
            for a in direct.privilege_assignments or []
        ):
            raise ValueError("unapproved principal retains project privileges")
        for principal in principals:
            effective = client.grants.get_effective(kind, name, principal=principal)
            actual = {
                p.privilege
                for a in effective.privilege_assignments or []
                for p in a.privileges or []
            }
            if effective.next_page_token or actual != policy.get((kind, name, principal), set()):
                raise ValueError(
                    "effective project privileges differ from the complete role policy"
                )
            grant_receipts[f"effective:{principal}:{kind}:{name}"] = identity(effective.as_dict())
    uploaded = {}
    client.files.create_directory(permit_dir)
    for key, (local, destination, expected) in files.items():
        content = local.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("staging source changed after preflight")
        client.files.create_directory(destination.rsplit("/", 1)[0])
        client.files.upload(destination, io.BytesIO(content), overwrite=True)
        observed = hashlib.sha256(download_bytes(client, destination, len(content))).hexdigest()
        if observed != expected:
            raise ValueError("uploaded training artifact differs from local bytes")
        uploaded[key] = observed
    body = canonical(config).encode()
    client.files.upload(expected_control, io.BytesIO(body), overwrite=True)
    if download_bytes(client, expected_control, len(body)) != body:
        raise ValueError("saved runtime configuration changed")
    probe = canonical({"synthetic_access_probe": True, "reservation": gate.batch_id}).encode()
    prefix = f"/Volumes/{catalog}"
    inputs, sealed, artifacts, control = (
        prefix + x
        for x in (
            "/development/inputs",
            "/evaluation/sealed",
            "/evidence/artifacts",
            "/evidence/control",
        )
    )
    for directory in (inputs, sealed, artifacts):
        client.files.upload(directory + "/access-probe.json", io.BytesIO(probe), overwrite=True)
    results = {}
    for role in ("trainer", "evaluator", "scorer"):
        limited = limited_clients[role]
        allowed = artifacts if role == "scorer" else inputs
        if download_bytes(limited, allowed + "/access-probe.json", len(probe)) != probe:
            raise ValueError("allowed resource readback changed")
        if role == "evaluator":
            if download_bytes(limited, sealed + "/access-probe.json", len(probe)) != probe:
                raise ValueError("evaluator canary changed")
        else:
            try:
                download_bytes(limited, sealed + "/access-probe.json", len(probe))
            except PermissionDenied:
                pass
            else:
                raise ValueError("limited identity can read the sealed volume")
        try:
            limited.files.upload(control + "/denied-" + role + ".json", io.BytesIO(probe))
        except PermissionDenied:
            pass
        else:
            raise ValueError("limited identity can write owner-only control")
        try:
            limited.files.upload(inputs + "/denied-" + role + ".json", io.BytesIO(probe))
        except PermissionDenied:
            pass
        else:
            raise ValueError("limited identity can overwrite frozen training inputs")
        destination = artifacts + "/access-probe-" + role + ".json"
        limited.files.upload(destination, io.BytesIO(probe), overwrite=True)
        if download_bytes(client, destination, len(probe)) != probe:
            raise ValueError("allowed limited-identity write was not saved")
        results[role] = {
            "principal_verified": True,
            "allowed_read_write": True,
            "control_write_denied": True,
            "training_input_write_denied": True,
            "sealed_read": role == "evaluator",
        }
    return {
        "operation": "stage",
        "reservation_id": gate.batch_id,
        "config_sha256": identity(config),
        "uploaded_sha256": uploaded,
        "grant_readback_sha256": grant_receipts,
        "access_probes": results,
        "authentication": "azure-client-secret",
        "operator_credential_cleanup_required": True,
        "provider_mutations_complete": True,
        "execution_lease_release_pending_cleanup": True,
        "compute_started": 0,
        "external_test_uploaded": False,
        "cost_reconciled": False,
    }
