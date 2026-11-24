"""Cloud clients must never be constructed from incomplete/no-spend configuration."""

import json
from pathlib import Path

import pytest

from fleet_fault_triage.cloud import CloudGate, identifier, validate_endpoints
from fleet_fault_triage.cloud_tasks import validate_permit, verify_workspace_identity
from fleet_fault_triage.finops import CostLedger

ROOT = Path(__file__).resolve().parents[1]


def test_default_cloud_configuration_blocks_before_identity_or_network(tmp_path, monkeypatch):
    config = json.loads((ROOT / "resources/cloud.example.json").read_text())
    approval = json.loads((ROOT / "resources/approval.example.json").read_text())

    def unexpected(*args, **kwargs):
        raise AssertionError("blocked cloud operation reached an external command")

    monkeypatch.setattr("subprocess.check_output", unexpected)
    gate = CloudGate(config, approval, CostLedger(tmp_path / "cost.sqlite"), {})
    with pytest.raises(ValueError):
        gate.begin("assistant", "blocked", 1)
    with pytest.raises(ValueError):
        gate("inference")


@pytest.mark.parametrize("proof", [{"subscription_id": "OTHER_SUBSCRIPTION"}, []])
@pytest.mark.parametrize("basis", ["initial_unused_subscription", "held_envelope_audit"])
def test_incomplete_cost_basis_cannot_cross_azure_identity(tmp_path, monkeypatch, proof, basis):
    from datetime import datetime, timezone

    from fleet_fault_triage.contracts import identity

    config = {
        "subscription_id": "SYNTHETIC_SUBSCRIPTION",
        "tenant_id": "SYNTHETIC_TENANT",
        "owner_identity": "SYNTHETIC_OWNER",
        "prices_verified": True,
        "entitlements_verified": True,
        "cloud_spend_authorized": True,
        "quotes": {
            "provision": {
                "currency": "AUD",
                "observed_utc": datetime.now(timezone.utc).isoformat(),
                "components": [{"meter": "SYNTHETIC", "rate_aud": "1", "maximum_quantity": "1"}],
                "tax_and_uncertainty_multiplier": "1.25",
            }
        },
    }
    approval = {
        "synthetic": False,
        "config_sha256": identity(config),
        "actions": ["provision", "cleanup"],
    }
    context = {
        "id": config["subscription_id"],
        "tenantId": config["tenant_id"],
        "user": {"name": config["owner_identity"]},
    }
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: json.dumps(context))
    gate = CloudGate(
        config,
        approval,
        CostLedger(tmp_path / "cost.sqlite"),
        {basis: proof},
    )
    with pytest.raises(ValueError, match="another Azure"):
        gate.begin("provision", "blocked", "25")


def test_complete_reconciliation_ignores_retired_snapshot_for_role_principal(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone

    from fleet_fault_triage.contracts import identity

    now = datetime.now(timezone.utc)
    config = {
        "subscription_id": "SYNTHETIC_SUBSCRIPTION",
        "tenant_id": "SYNTHETIC_TENANT",
        "owner_identity": "SYNTHETIC_OWNER",
        "principals": {"investigator": "SYNTHETIC_INVESTIGATOR"},
        "prices_verified": True,
        "entitlements_verified": True,
        "cloud_spend_authorized": True,
        "quotes": {
            "assistant": {
                "currency": "AUD",
                "observed_utc": now.isoformat(),
                "components": [{"meter": "SYNTHETIC", "rate_aud": "1", "maximum_quantity": "1"}],
                "tax_and_uncertainty_multiplier": "1.25",
            }
        },
    }
    approval = {
        "synthetic": False,
        "approved": True,
        "project": "fleet-fault-triage",
        "source": "SYNTHETIC TEST",
        "config_sha256": identity(config),
        "actions": ["assistant"],
        "currency": "AUD",
        "budget_aud": "100",
        "cleanup_reserve_aud": "20",
        "expires_utc": (now + timedelta(hours=1)).isoformat(),
    }
    context = {
        "id": config["subscription_id"],
        "tenantId": config["tenant_id"],
        "user": {"name": "SYNTHETIC_INVESTIGATOR"},
    }
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: json.dumps(context))
    reconciled = {
        "currency": "AUD",
        "observed_utc": now.isoformat(),
        "complete_through_observed_time": True,
        "accrued_aud": "10",
        "initial_unused_subscription": {"owner_identity": "SYNTHETIC_OWNER"},
    }
    gate = CloudGate(config, approval, CostLedger(tmp_path / "cost.sqlite"), reconciled)
    gate.begin("assistant", "reconciled", "2")
    assert gate.started is True


@pytest.mark.parametrize("task", ["train", "evaluate", "score", "monitor"])
def test_cloud_task_defaults_cannot_run_locally(task):
    with pytest.raises(ValueError, match="blocked"):
        validate_permit({"cloud_spend_authorized": False}, {}, task)


@pytest.mark.parametrize("expected", [None, True, 123, "", "0", "-1", "1.0", "01"])
def test_workspace_identity_requires_an_approved_numeric_string(expected):
    from types import SimpleNamespace

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid configuration reached the provider")

    with pytest.raises(ValueError, match="provider workspace ID"):
        verify_workspace_identity(
            SimpleNamespace(api_client=SimpleNamespace(do=unexpected)),
            {"databricks_workspace_id": expected},
        )


@pytest.mark.parametrize("actual", [True, None, 123, "0", "124"])
def test_runtime_workspace_mismatch_blocks_even_with_the_same_host(actual):
    from types import SimpleNamespace

    workspace = SimpleNamespace(
        config=SimpleNamespace(host="https://synthetic.azuredatabricks.net"),
        api_client=SimpleNamespace(do=lambda *args, **kwargs: {"X-Databricks-Org-Id": actual}),
    )
    with pytest.raises(ValueError, match="wrong workspace"):
        verify_workspace_identity(
            workspace,
            {"databricks_workspace_id": "123", "databricks_host": workspace.config.host},
        )


def test_runtime_workspace_identity_allows_a_provider_url_alias():
    from types import SimpleNamespace

    from fleet_fault_triage.contracts import identity

    workspace = SimpleNamespace(
        config=SimpleNamespace(host="https://runtime-alias.azuredatabricks.net"),
        api_client=SimpleNamespace(do=lambda *args, **kwargs: {"X-Databricks-Org-Id": "123"}),
    )
    assert verify_workspace_identity(
        workspace,
        {"databricks_workspace_id": "123", "databricks_host": "https://public.azuredatabricks.net"},
    ) == identity("123")


def test_runtime_workspace_identity_ignores_a_stale_sdk_cached_id(monkeypatch):
    from types import SimpleNamespace

    import databricks.sdk.config
    from databricks.sdk import WorkspaceClient

    monkeypatch.setattr(
        databricks.sdk.config,
        "get_host_metadata",
        lambda *args, **kwargs: SimpleNamespace(
            account_id=None,
            workspace_id=None,
            oidc_endpoint=None,
            cloud=None,
            token_federation_default_oidc_audiences=[],
        ),
    )
    workspace = WorkspaceClient(
        host="https://fixture.azuredatabricks.net",
        token="SYNTHETIC_NOT_A_SECRET",
        auth_type="pat",
        workspace_id="123",
    )
    requests = []

    def observed(method, path, **kwargs):
        requests.append((method, path, kwargs))
        return {"X-Databricks-Org-Id": "124"}

    monkeypatch.setattr(workspace.api_client, "do", observed)
    assert workspace.get_workspace_id() == 123 and requests == []
    with pytest.raises(ValueError, match="wrong workspace"):
        verify_workspace_identity(workspace, {"databricks_workspace_id": "123"})
    assert requests == [
        ("GET", "/api/2.0/preview/scim/v2/Me", {"response_headers": ["X-Databricks-Org-Id"]})
    ]


def test_endpoints_cannot_exfiltrate_tokens_to_arbitrary_hosts():
    config = json.loads((ROOT / "resources/cloud.example.json").read_text())
    validate_endpoints(config)
    for value in ("http://example.test", "https://example.test", "https://evil.test/?token=secret"):
        with pytest.raises(ValueError):
            validate_endpoints({**config, "foundry_endpoint": value})
    with pytest.raises(ValueError):
        identifier("schema; DROP TABLE data")


def test_price_quote_requires_current_timezone_and_real_zero_price_evidence():
    from datetime import datetime, timezone

    from fleet_fault_triage.cloud import operation_bound

    quote = {
        "currency": "AUD",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "components": [{"meter": "synthetic", "rate_aud": "2", "maximum_quantity": "3"}],
        "tax_and_uncertainty_multiplier": "1.10",
    }
    from decimal import Decimal

    assert operation_bound({"quotes": {"train": quote}}, "train") == Decimal("6.60")
    quote["observed_utc"] = "2026-10-25T00:00:00"
    with pytest.raises(ValueError, match="timezone"):
        operation_bound({"quotes": {"train": quote}}, "train")
    quote["observed_utc"] = datetime.now(timezone.utc).isoformat()
    quote["components"][0]["rate_aud"] = "0"
    with pytest.raises(ValueError, match="unverified"):
        operation_bound({"quotes": {"train": quote}}, "train")


def test_runtime_identity_refuses_different_installed_bytes():
    from fleet_fault_triage.runtime import runtime_identity, verify_runtime

    current = runtime_identity()
    verify_runtime({"runtime_identity": current})
    current["dependencies"]["numpy"] = "SYNTHETIC_WRONG_VERSION"
    with pytest.raises(ValueError, match="differ"):
        verify_runtime({"runtime_identity": current})
