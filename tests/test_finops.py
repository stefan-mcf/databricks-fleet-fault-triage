"""Money/usage fixtures are synthetic and cannot authorise a cloud action."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from fleet_fault_triage.contracts import identity
from fleet_fault_triage.finops import CostLedger, money


@pytest.fixture
def ledger_inputs(tmp_path):
    now = datetime.now(timezone.utc)
    approval = {
        "approved": True,
        "project": "fleet-fault-triage",
        "source": "SYNTHETIC_TEST",
        "currency": "AUD",
        "budget_aud": "100",
        "cleanup_reserve_aud": "20",
        "expires_utc": (now + timedelta(hours=1)).isoformat(),
    }
    reconciliation = {
        "currency": "AUD",
        "observed_utc": now.isoformat(),
        "complete_through_observed_time": True,
        "accrued_aud": "10",
    }
    return CostLedger(tmp_path / "costs.sqlite"), approval, reconciliation, now


def test_pending_failed_attempts_and_cleanup_reserve_cannot_be_spent_twice(ledger_inputs):
    ledger, approval, reconciled, now = ledger_inputs
    ledger.reserve("first", "60", approval, reconciled, now=now)
    with pytest.raises(ValueError, match="budget"):
        ledger.reserve("second", "11", approval, reconciled, now=now)
    with pytest.raises(ValueError, match="already"):
        ledger.reserve("first", "1", approval, reconciled, now=now)
    ledger.reconcile(
        "first",
        {"batch_id": "first", "provider_usage_complete": True, "currency": "AUD", "cost_aud": "55"},
    )
    ledger.reserve("second", "10", approval, {**reconciled, "accrued_aud": "65"}, now=now)
    with pytest.raises(ValueError):
        ledger.reserve("third", None, approval, reconciled, now=now)


@pytest.mark.parametrize(
    "fault", ["approval", "currency", "stale", "future", "incomplete", "unknown_cost"]
)
def test_cost_preflight_fails_closed(ledger_inputs, fault):
    ledger, approval, reconciled, now = ledger_inputs
    if fault == "approval":
        approval["approved"] = False
    elif fault == "currency":
        reconciled["currency"] = "USD"
    elif fault == "stale":
        reconciled["observed_utc"] = (now - timedelta(days=2)).isoformat()
    elif fault == "future":
        reconciled["observed_utc"] = (now + timedelta(minutes=1)).isoformat()
    elif fault == "incomplete":
        reconciled["complete_through_observed_time"] = False
    else:
        reconciled["accrued_aud"] = None
    with pytest.raises(ValueError):
        ledger.reserve("blocked", "1", approval, reconciled, now=now)


def test_corrections_reconcile_and_zero_denominator_is_undefined(ledger_inputs):
    ledger, *_ = ledger_inputs
    for record_id, kind, cost in (
        ("original", "ORIGINAL", "3"),
        ("retract", "RETRACTION", "-3"),
        ("restate", "RESTATEMENT", "2"),
    ):
        record = {
            "record_id": record_id,
            "component": "brief",
            "run_id": "synthetic",
            "record_type": kind,
            "currency": "AUD",
            "cost_aud": cost,
        }
        ledger.record_usage(record)
    with pytest.raises(sqlite3.IntegrityError):
        ledger.record_usage(record)
    report = ledger.report({"brief": 0}, "1")
    assert Decimal(report["project_total_aud"]) == 3
    assert report["unit_costs"][0]["fully_allocated_cost_per_unit_aud"] is None
    assert report["invoice_reconciled"] is False


@pytest.mark.parametrize("value", [None, True, "NaN", "Infinity"])
def test_unknown_amounts_cannot_be_silently_zero(value):
    with pytest.raises(ValueError):
        money(value)


def test_global_lease_run_count_and_failed_request_reservations(ledger_inputs):
    ledger, approval, reconciled, now = ledger_inputs
    for i in range(4):
        ledger.reserve(str(i), "1", approval, reconciled, now=now, execution_kind="train")
        with pytest.raises(ValueError, match="active"):
            ledger.reserve("overlap", "1", approval, reconciled, now=now, execution_kind="score")
        ledger.consume_request(str(i), "inference", 4)
        ledger.finish_execution(str(i), provider_terminal_verified=True)
    with pytest.raises(ValueError, match="four-run"):
        ledger.reserve("fifth", "1", approval, reconciled, now=now, execution_kind="evaluate")
    ledger.reserve("assistant", "1", approval, reconciled, now=now, execution_kind="assistant")
    with pytest.raises(ValueError, match="allowance"):
        ledger.consume_request("assistant", "inference", 4)
    ledger.reserve_embedding_tokens("assistant", 500_000)
    with pytest.raises(ValueError, match="token allowance"):
        ledger.reserve_embedding_tokens("assistant", 1)


def test_cleanup_can_use_reserved_remainder_and_stale_accrual_cannot_release_it(ledger_inputs):
    ledger, approval, reconciled, now = ledger_inputs
    ledger.reserve("spent", "60", approval, reconciled, now=now)
    ledger.reconcile(
        "spent",
        {"batch_id": "spent", "provider_usage_complete": True, "currency": "AUD", "cost_aud": "60"},
    )
    with pytest.raises(ValueError, match="below"):
        ledger.reserve("stale", "1", approval, reconciled, now=now)
    ledger.reserve(
        "cleanup",
        "30",
        approval,
        {**reconciled, "accrued_aud": "70"},
        now=now,
        execution_kind="cleanup",
    )


@pytest.fixture
def unused_inputs(ledger_inputs):
    ledger, approval, reconciled, now = ledger_inputs
    proof = {
        "subscription_id": "SYNTHETIC_SUBSCRIPTION",
        "tenant_id": "SYNTHETIC_TENANT",
        "owner_identity": "SYNTHETIC_OWNER",
        "resource_count": 0,
        "deployment_count": 0,
        "cost_rows": 0,
        "billing_subscription_count": 1,
        "cost_query_succeeded": True,
        "history_truncated": False,
        "billing_active": True,
        "billing_currency": "AUD",
        "billing_subscription_matches_expected": True,
        "billing_product_category": "UsageBased",
        "subscription_state": "Enabled",
        "subscription_quota_id": "PayAsYouGo_2014-09-01",
        "spending_limit": "Off",
        "cost_query_has_more_pages": False,
        "activity_event_count": 2,
        "activity_operations": [
            "Microsoft.Subscription/Subscriptions/write",
            "Microsoft.Resources/subscriptions/resourcegroups/write",
        ],
        "receipt_sha256": {
            key: "a" * 64
            for key in ("account", "cost", "inventory", "activity", "deployments", "billing")
        },
        "observed_utc": now.isoformat(),
        "history_start_utc": (now - timedelta(days=2)).isoformat(),
        "history_end_utc": now.isoformat(),
        "subscription_purchase_utc": (now - timedelta(days=1)).isoformat(),
    }
    approval.update(
        actions=["provision", "cleanup"],
        initial_baseline_sha256=identity(proof),
        expires_utc=(now + timedelta(days=7)).isoformat(),
    )
    reconciled.update(
        complete_through_observed_time=False, accrued_aud=None, initial_unused_subscription=proof
    )
    return ledger, approval, reconciled, now


def test_unused_first_provision_holds_full_envelope_without_claiming_actuals(unused_inputs):
    ledger, approval, reconciled, now = unused_inputs
    result = ledger.reserve(
        "bootstrap", "25", approval, reconciled, now=now, execution_kind="provision"
    )
    assert result["budget_basis"] == "verified_unused_subscription"
    assert result["provider_accrued_verified"] is False
    assert reconciled["accrued_aud"] is None
    report = ledger.report({})
    assert report["project_total_aud"] is None
    assert report["project_cost_complete"] is False
    with sqlite3.connect(ledger.path) as db:
        assert db.execute("SELECT maximum,state FROM reservations").fetchone() == ("25", "pending")
        assert (
            db.execute("SELECT snapshot_sha256 FROM bootstrap_baseline").fetchone()[0]
            == approval["initial_baseline_sha256"]
        )
    ledger.finish_execution("bootstrap", provider_terminal_verified=True)
    with pytest.raises(ValueError, match="first provision"):
        ledger.reserve("second", "1", approval, reconciled, now=now, execution_kind="provision")
    with pytest.raises(ValueError, match="unused-subscription"):
        ledger.reserve("train", "1", approval, reconciled, now=now, execution_kind="train")


@pytest.mark.parametrize(
    "fault",
    [
        "resource",
        "boolean_count",
        "cost_rows",
        "history",
        "truncated",
        "receipt",
        "identity",
        "stale",
        "uncovered",
        "billing",
        "missing_category",
        "wrong_category",
        "old_subscription",
        "missing_events",
        "more_cost_pages",
    ],
)
def test_unused_basis_rejects_unsafe_or_missing_proof(unused_inputs, fault):
    ledger, approval, reconciled, now = unused_inputs
    proof = reconciled["initial_unused_subscription"]
    if fault == "resource":
        proof["resource_count"] = 1
    elif fault == "boolean_count":
        proof["resource_count"] = False
    elif fault == "cost_rows":
        proof["cost_rows"] = 1
    elif fault == "history":
        proof["activity_operations"].append("Microsoft.Compute/virtualMachines/write")
    elif fault == "truncated":
        proof["history_truncated"] = True
    elif fault == "receipt":
        proof["receipt_sha256"].pop("cost")
    elif fault == "identity":
        proof.pop("owner_identity")
    elif fault == "stale":
        proof["observed_utc"] = (now - timedelta(hours=2)).isoformat()
    elif fault == "uncovered":
        proof["history_start_utc"] = now.isoformat()
    elif fault == "billing":
        proof["billing_subscription_matches_expected"] = False
    elif fault == "missing_category":
        proof.pop("billing_product_category")
    elif fault == "wrong_category":
        proof["billing_product_category"] = "SeatBased"
    elif fault == "old_subscription":
        proof["subscription_purchase_utc"] = (now - timedelta(days=100)).isoformat()
        proof["history_start_utc"] = (now - timedelta(days=101)).isoformat()
    elif fault == "missing_events":
        proof["activity_event_count"] = 0
    elif fault == "more_cost_pages":
        proof["cost_query_has_more_pages"] = True
    approval["initial_baseline_sha256"] = identity(proof)
    with pytest.raises(ValueError):
        ledger.reserve("blocked", "25", approval, reconciled, now=now, execution_kind="provision")


def test_unused_basis_requires_exact_hash_and_empty_ledger(unused_inputs):
    ledger, approval, reconciled, now = unused_inputs
    with pytest.raises(ValueError, match="exact"):
        ledger.reserve(
            "changed",
            "25",
            {**approval, "initial_baseline_sha256": "b" * 64},
            reconciled,
            now=now,
            execution_kind="provision",
        )
    with sqlite3.connect(ledger.path) as db:
        db.execute("INSERT INTO requests(batch_id,kind) VALUES ('SYNTHETIC_PRIOR','inference')")
    with pytest.raises(ValueError, match="first provision"):
        ledger.reserve("prior", "25", approval, reconciled, now=now, execution_kind="provision")


def test_unused_basis_cleanup_keeps_pending_costs_and_cannot_spend_reserve_twice(unused_inputs):
    ledger, approval, reconciled, now = unused_inputs
    with pytest.raises(ValueError, match="original"):
        ledger.reserve("orphan", "1", approval, reconciled, now=now, execution_kind="cleanup")
    with pytest.raises(ValueError, match="budget"):
        ledger.reserve("oversize", "81", approval, reconciled, now=now, execution_kind="provision")
    ledger.reserve("bootstrap", "25", approval, reconciled, now=now, execution_kind="provision")
    ledger.finish_execution("bootstrap", provider_terminal_verified=True)
    later = now + timedelta(days=2)
    current = {**reconciled, "observed_utc": later.isoformat()}
    with pytest.raises(ValueError, match="budget"):
        ledger.reserve("too_large", "76", approval, current, now=later, execution_kind="cleanup")
    ledger.reserve("cleanup", "75", approval, current, now=later, execution_kind="cleanup")
    ledger.finish_execution("cleanup", provider_terminal_verified=True)
    with pytest.raises(ValueError, match="budget"):
        ledger.reserve("again", "1", approval, current, now=later, execution_kind="cleanup")


@pytest.mark.parametrize("corrected", [False, True])
def test_bootstrap_cleanup_charges_partial_overruns_without_releasing_held_spend(
    unused_inputs, corrected
):
    ledger, approval, reconciled, now = unused_inputs
    ledger.reserve("bootstrap", "25", approval, reconciled, now=now, execution_kind="provision")
    ledger.finish_execution("bootstrap", provider_terminal_verified=True)
    record = {
        "record_id": "SYNTHETIC_PARTIAL",
        "component": "bootstrap",
        "run_id": "SYNTHETIC_RUN",
        "currency": "AUD",
        "cost_aud": "40",
        "record_type": "ORIGINAL",
    }
    ledger.record_usage(record)
    if corrected:
        ledger.record_usage(
            {
                **record,
                "record_id": "SYNTHETIC_RETRACTION",
                "cost_aud": "-40",
                "record_type": "RETRACTION",
            }
        )
    with pytest.raises(ValueError, match="budget"):
        ledger.reserve("overrun", "75", approval, reconciled, now=now, execution_kind="cleanup")
    ledger.reserve("bounded_cleanup", "35", approval, reconciled, now=now, execution_kind="cleanup")
    with sqlite3.connect(ledger.path) as db:
        assert (
            db.execute("SELECT state FROM reservations WHERE batch_id='bootstrap'").fetchone()[0]
            == "pending"
        )


def test_original_provision_recovery_reuses_unchanged_held_reservation(unused_inputs):
    ledger, approval, reconciled, now = unused_inputs
    with pytest.raises(ValueError, match="unchanged active"):
        ledger.reserve(
            "missing",
            "25",
            approval,
            reconciled,
            now=now,
            execution_kind="provision",
            resume_existing=True,
        )
    ledger.reserve("bootstrap", "25", approval, reconciled, now=now, execution_kind="provision")
    later = now + timedelta(hours=2)
    current = {**reconciled, "observed_utc": later.isoformat()}
    result = ledger.reserve(
        "bootstrap",
        "25",
        approval,
        current,
        now=later,
        execution_kind="provision",
        resume_existing=True,
    )
    assert result["resumed_existing_reservation"] is True
    with sqlite3.connect(ledger.path) as db:
        assert db.execute("SELECT count(*) FROM reservations").fetchone()[0] == 1
        assert db.execute("SELECT maximum,state FROM reservations").fetchone() == ("25", "pending")
    with pytest.raises(ValueError, match="unchanged active"):
        ledger.reserve(
            "bootstrap",
            "30",
            approval,
            current,
            now=later,
            execution_kind="provision",
            resume_existing=True,
        )
    with pytest.raises(ValueError, match="only an existing"):
        ledger.reserve(
            "bootstrap",
            "25",
            approval,
            current,
            now=later,
            execution_kind="train",
            resume_existing=True,
        )
    ledger.finish_execution("bootstrap", provider_terminal_verified=True)
    with pytest.raises(ValueError, match="unchanged active"):
        ledger.reserve(
            "bootstrap",
            "25",
            approval,
            current,
            now=later,
            execution_kind="provision",
            resume_existing=True,
        )


def test_recovery_keeps_known_partial_costs_in_budget(unused_inputs):
    ledger, approval, reconciled, now = unused_inputs
    ledger.reserve("bootstrap", "25", approval, reconciled, now=now, execution_kind="provision")
    ledger.record_usage(
        {
            "record_id": "SYNTHETIC_OVERRUN",
            "component": "bootstrap",
            "run_id": "SYNTHETIC_RUN",
            "currency": "AUD",
            "cost_aud": "60",
            "record_type": "ORIGINAL",
        }
    )
    with pytest.raises(ValueError, match="budget"):
        ledger.reserve(
            "bootstrap",
            "25",
            approval,
            reconciled,
            now=now,
            execution_kind="provision",
            resume_existing=True,
        )


def test_empty_batch_id_does_not_bypass_normal_global_lease(ledger_inputs):
    ledger, approval, reconciled, now = ledger_inputs
    ledger.reserve("", "1", approval, reconciled, now=now, execution_kind="train")
    with pytest.raises(ValueError, match="active"):
        ledger.reserve("second", "1", approval, reconciled, now=now, execution_kind="score")


def test_provision_recovery_is_claimed_once_atomically(unused_inputs):
    ledger, approval, reconciled, now = unused_inputs
    ledger.reserve("bootstrap", "25", approval, reconciled, now=now, execution_kind="provision")

    def claim(_):
        try:
            ledger.reserve(
                "bootstrap",
                "25",
                approval,
                reconciled,
                now=now,
                execution_kind="provision",
                resume_existing=True,
            )
            return "claimed"
        except ValueError as exc:
            assert "already claimed" in str(exc)
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(claim, range(2))) == ["claimed", "rejected"]
    with sqlite3.connect(ledger.path) as db:
        assert db.execute("SELECT count(*) FROM provision_recoveries").fetchone()[0] == 1
        assert db.execute("SELECT maximum,state FROM reservations").fetchone() == ("25", "pending")
        assert db.execute("SELECT kind,state FROM executions").fetchone() == ("provision", "active")


@pytest.fixture
def held_inputs(unused_inputs):
    ledger, approval, reconciled, now = unused_inputs
    ledger.reserve("bootstrap", "25", approval, reconciled, now=now, execution_kind="provision")
    ledger.finish_execution("bootstrap", provider_terminal_verified=True)
    baseline = reconciled.pop("initial_unused_subscription")
    audit = {
        "observed_utc": now.isoformat(),
        **{key: baseline[key] for key in ("subscription_id", "tenant_id", "owner_identity")},
        "initial_baseline_sha256": identity(baseline),
        "ledger_sha256": ledger.snapshot_sha256(),
        "all_chargeable_activity_covered": True,
        "history_truncated": False,
        "unexpected_resources": 0,
        "active_jobs": 0,
        "active_clusters": 0,
        "cost_query_succeeded": True,
        "cost_query_has_more_pages": False,
        "reported_positive_cost_aud": None,
        "fixed_costs_covered_until_utc": (now + timedelta(days=6)).isoformat(),
        "receipt_sha256": {
            key: "b" * 64 for key in ("account", "cost", "inventory", "activity", "workloads")
        },
    }
    approval.update(
        actions=["train", "evaluate", "cleanup"],
        expires_utc=(now + timedelta(days=5)).isoformat(),
        held_envelope_audit_sha256=identity(audit),
    )
    reconciled["held_envelope_audit"] = audit
    return ledger, approval, reconciled, now


def test_held_envelopes_preserve_unknown_actuals_and_every_previous_hold(held_inputs):
    ledger, approval, reconciled, now = held_inputs
    result = ledger.reserve("train", "55", approval, reconciled, now=now, execution_kind="train")
    assert result["budget_basis"] == "held_priced_envelopes"
    assert result["provider_accrued_verified"] is False
    assert ledger.report({})["project_cost_complete"] is False
    with sqlite3.connect(ledger.path) as db:
        assert db.execute(
            "SELECT maximum,state FROM reservations ORDER BY batch_id"
        ).fetchall() == [("25", "pending"), ("55", "pending")]
    ledger.finish_execution("train", provider_terminal_verified=True)
    with pytest.raises(ValueError, match="ledger state changed"):
        ledger.reserve("reuse", "1", approval, reconciled, now=now, execution_kind="evaluate")


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "identity",
        "stale",
        "retimestamped",
        "deadline",
        "baseline",
        "uncovered",
        "history",
        "active",
        "missing_receipt",
        "released",
        "reported_cost",
        "usage",
        "unapproved",
        "provision",
    ],
)
def test_held_envelope_invalid_or_over_budget_requests_do_not_mutate_ledger(held_inputs, fault):
    ledger, approval, reconciled, now = held_inputs
    audit = reconciled["held_envelope_audit"]
    action = "train"
    if fault == "identity":
        audit["tenant_id"] = "OTHER"
    elif fault == "stale":
        reconciled["observed_utc"] = (now - timedelta(minutes=6)).isoformat()
        audit["observed_utc"] = reconciled["observed_utc"]
    elif fault == "retimestamped":
        audit["observed_utc"] = (now - timedelta(minutes=6)).isoformat()
    elif fault == "deadline":
        audit["fixed_costs_covered_until_utc"] = (now + timedelta(hours=1)).isoformat()
    elif fault == "baseline":
        audit["initial_baseline_sha256"] = "c" * 64
    elif fault == "uncovered":
        audit["all_chargeable_activity_covered"] = False
    elif fault == "history":
        audit["history_truncated"] = True
    elif fault == "active":
        audit["active_clusters"] = 1
    elif fault == "missing_receipt":
        del audit["receipt_sha256"]["activity"]
    elif fault == "released":
        ledger.reconcile(
            "bootstrap",
            {
                "batch_id": "bootstrap",
                "provider_usage_complete": True,
                "currency": "AUD",
                "cost_aud": "1",
            },
        )
        audit["ledger_sha256"] = ledger.snapshot_sha256()
    elif fault == "reported_cost":
        audit["reported_positive_cost_aud"] = "51"
    elif fault == "usage":
        ledger.record_usage(
            {
                "record_id": "overrun",
                "component": "network",
                "run_id": "bootstrap",
                "currency": "AUD",
                "cost_aud": "51",
                "record_type": "ORIGINAL",
            }
        )
        audit["ledger_sha256"] = ledger.snapshot_sha256()
    elif fault == "unapproved":
        approval["actions"] = ["cleanup"]
    elif fault == "provision":
        action = "provision"
        approval["actions"].append(action)
    approval["held_envelope_audit_sha256"] = "c" * 64 if fault == "hash" else identity(audit)
    before = ledger.snapshot_sha256()
    with pytest.raises(ValueError):
        ledger.reserve("blocked", "5", approval, reconciled, now=now, execution_kind=action)
    if fault == "reported_cost":
        # A rejected operation cannot erase a newly observed positive cost.
        assert ledger.snapshot_sha256() != before
        with sqlite3.connect(ledger.path) as db:
            assert db.execute("SELECT positive_cost FROM held_cost_high_watermark").fetchone() == (
                "51",
            )
            assert not db.execute("SELECT 1 FROM reservations WHERE batch_id='blocked'").fetchone()
    else:
        assert ledger.snapshot_sha256() == before


def test_held_audit_cannot_be_spent_twice_by_concurrent_launchers(held_inputs):
    ledger, approval, reconciled, now = held_inputs

    def launch(i):
        try:
            ledger.reserve(str(i), "5", approval, reconciled, now=now, execution_kind="train")
            return "reserved"
        except ValueError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(launch, range(2))) == ["rejected", "reserved"]


@pytest.mark.parametrize("first_maximum", ["5", "6"])
def test_known_positive_cost_cannot_disappear_after_success_or_budget_rejection(
    held_inputs, first_maximum
):
    ledger, approval, reconciled, now = held_inputs
    audit = reconciled["held_envelope_audit"]
    audit["reported_positive_cost_aud"] = "50"
    approval["held_envelope_audit_sha256"] = identity(audit)
    if first_maximum == "5":
        ledger.reserve(
            "first", first_maximum, approval, reconciled, now=now, execution_kind="train"
        )
        ledger.finish_execution("first", provider_terminal_verified=True)
    else:
        with pytest.raises(ValueError, match="budget"):
            ledger.reserve(
                "first", first_maximum, approval, reconciled, now=now, execution_kind="train"
            )
    audit["reported_positive_cost_aud"] = None
    audit["ledger_sha256"] = ledger.snapshot_sha256()
    approval["held_envelope_audit_sha256"] = identity(audit)
    with pytest.raises(ValueError, match="budget"):
        ledger.reserve("forget", "50", approval, reconciled, now=now, execution_kind="evaluate")
    with sqlite3.connect(ledger.path) as db:
        assert db.execute("SELECT positive_cost FROM held_cost_high_watermark").fetchone() == (
            "50",
        )


def test_bootstrap_cleanup_also_retains_positive_cost_from_a_rejected_held_operation(held_inputs):
    ledger, approval, reconciled, now = held_inputs
    audit = reconciled["held_envelope_audit"]
    audit["reported_positive_cost_aud"] = "50"
    approval["held_envelope_audit_sha256"] = identity(audit)
    with pytest.raises(ValueError, match="budget"):
        ledger.reserve("blocked", "6", approval, reconciled, now=now, execution_kind="train")
    with sqlite3.connect(ledger.path) as db:
        baseline = json.loads(db.execute("SELECT evidence FROM bootstrap_baseline").fetchone()[0])
    reconciled.pop("held_envelope_audit")
    reconciled["initial_unused_subscription"] = baseline
    approval["actions"] = ["provision", "cleanup"]
    with pytest.raises(ValueError, match="budget"):
        ledger.reserve("cleanup", "26", approval, reconciled, now=now, execution_kind="cleanup")


def test_stage_holds_cost_without_consuming_a_training_attempt(held_inputs):
    ledger, approval, reconciled, now = held_inputs
    approval["actions"].append("stage")
    ledger.reserve("stage", "5", approval, reconciled, now=now, execution_kind="stage")
    with sqlite3.connect(ledger.path) as db:
        assert db.execute("SELECT count(*) FROM executions WHERE kind='train'").fetchone()[0] == 0
        assert db.execute(
            "SELECT kind,state FROM executions WHERE batch_id='stage'"
        ).fetchone() == ("stage", "active")
        assert sum(float(x[0]) for x in db.execute("SELECT maximum FROM reservations")) == 30
