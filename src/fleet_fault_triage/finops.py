"""Project-only reservations and unit economics; never a provider billing cap."""

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from fleet_fault_triage.contracts import canonical, identity, utcnow

FREE_BOOTSTRAP_OPERATIONS = {
    "microsoft.subscription/subscriptions/write",
    "microsoft.resources/subscriptions/resourcegroups/write",
    "microsoft.resources/deployments/validate/action",
    "microsoft.resources/deployments/whatif/action",
    "microsoft.consumption/budgets/write",
    "microsoft.quota/quotas/write",
    *{
        f"{provider}/register/action"
        for provider in (
            "microsoft.compute",
            "microsoft.network",
            "microsoft.managedidentity",
            "microsoft.storage",
            "microsoft.databricks",
            "microsoft.search",
            "microsoft.cognitiveservices",
            "microsoft.quota",
        )
    },
}


def unused_subscription_basis(reconciliation, approval, action, now):
    """A zero prior-usage budget basis; never substitute it for provider actuals."""
    proof = reconciliation.get("initial_unused_subscription")
    if (
        action not in {"provision", "cleanup"}
        or approval.get("actions") != ["provision", "cleanup"]
        or reconciliation.get("complete_through_observed_time") is not False
        or "accrued_aud" not in reconciliation
        or reconciliation["accrued_aud"] is not None
        or not isinstance(proof, dict)
        or identity(proof) != approval.get("initial_baseline_sha256")
    ):
        raise ValueError("unreconciled costs require an exact unused-subscription bootstrap basis")
    expected = {
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
    }
    if any(type(proof.get(k)) is not type(v) or proof[k] != v for k, v in expected.items()):
        raise ValueError("unused-subscription evidence is incomplete or not empty")
    if any(not proof.get(k) for k in ("subscription_id", "tenant_id", "owner_identity")):
        raise ValueError("unused-subscription identity is incomplete")
    if (
        type(proof.get("activity_event_count")) is not int
        or not 0 < proof["activity_event_count"] < 1000
    ):
        raise ValueError("bootstrap requires nonempty, untruncated administrative history")
    operations = proof.get("activity_operations")
    if (
        not isinstance(operations, list)
        or not operations
        or any(
            not isinstance(x, str) or x.lower() not in FREE_BOOTSTRAP_OPERATIONS for x in operations
        )
        or "microsoft.subscription/subscriptions/write" not in {x.lower() for x in operations}
    ):
        raise ValueError("unknown or billable subscription history blocks bootstrap")
    hashes = proof.get("receipt_sha256")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != {"account", "cost", "inventory", "activity", "deployments", "billing"}
        or any(not re.fullmatch(r"[0-9a-f]{64}", str(x)) for x in hashes.values())
    ):
        raise ValueError("bootstrap source receipts are incomplete")
    try:
        observed, start, end, purchased = (
            datetime.fromisoformat(proof[k])
            for k in (
                "observed_utc",
                "history_start_utc",
                "history_end_utc",
                "subscription_purchase_utc",
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("bootstrap history timestamps are incomplete") from exc
    maximum_age = timedelta(hours=1 if action == "provision" else 168)
    if (
        any(x.tzinfo is None for x in (observed, start, end, purchased))
        or not timedelta(0) <= now - observed <= maximum_age
        or not start <= purchased <= end <= observed
        or observed - purchased > timedelta(days=7)
        or not timedelta(0) <= observed - end <= timedelta(minutes=5)
    ):
        raise ValueError(
            "bootstrap history is stale, future or does not cover subscription creation"
        )
    return proof


def money(value):
    if isinstance(value, bool) or value is None:
        raise ValueError("missing/invalid amount is not zero")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("nonfinite money")
    return result


class CostLedger:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with sqlite3.connect(path) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS reservations (
                    batch_id TEXT PRIMARY KEY, maximum TEXT NOT NULL, state TEXT NOT NULL,
                    created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS usage (
                    record_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS requests (
                    sequence INTEGER PRIMARY KEY, batch_id TEXT NOT NULL, kind TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS executions (
                    batch_id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS embedding_bounds (
                    batch_id TEXT NOT NULL, tokens INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS bootstrap_baseline (
                    snapshot_sha256 TEXT PRIMARY KEY, evidence TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS provision_recoveries (
                    batch_id TEXT PRIMARY KEY, claimed_utc TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS held_cost_high_watermark (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    positive_cost TEXT NOT NULL, audit_sha256 TEXT NOT NULL);
            """)

    @staticmethod
    def _snapshot(db):
        """Bind an exposure audit to all saved reservations, usage and execution state."""
        return {
            table: sorted(db.execute(f"SELECT * FROM {table}").fetchall())
            for table in (
                "reservations",
                "usage",
                "requests",
                "executions",
                "embedding_bounds",
                "bootstrap_baseline",
                "provision_recoveries",
                "held_cost_high_watermark",
            )
        }

    def snapshot_sha256(self):
        with sqlite3.connect(self.path) as db:
            db.execute("BEGIN")
            return identity(self._snapshot(db))

    def _held_envelope_basis(self, db, reconciliation, approval, action, now):
        audit = reconciliation["held_envelope_audit"]
        if (
            not isinstance(audit, dict)
            or identity(audit) != approval.get("held_envelope_audit_sha256")
            or action
            not in {
                "stage",
                "train",
                "evaluate",
                "score",
                "monitor",
                "assistant",
                "index_publish",
                "judge",
                "promote",
                "cleanup",
            }
            or action not in approval.get("actions", [])
            or reconciliation.get("accrued_aud", "missing") is not None
            or reconciliation.get("complete_through_observed_time") is not False
            or audit.get("observed_utc") != reconciliation.get("observed_utc")
            or "reported_positive_cost_aud" not in audit
        ):
            raise ValueError("held-envelope audit or exact operation approval is invalid")
        expected = {
            "all_chargeable_activity_covered": True,
            "history_truncated": False,
            "unexpected_resources": 0,
            "active_jobs": 0,
            "active_clusters": 0,
            "cost_query_succeeded": True,
            "cost_query_has_more_pages": False,
        }
        if any(type(audit.get(k)) is not type(v) or audit[k] != v for k, v in expected.items()):
            raise ValueError("uncovered activity or incomplete held-envelope audit")
        hashes = audit.get("receipt_sha256")
        if (
            not isinstance(hashes, dict)
            or set(hashes) != {"account", "cost", "inventory", "activity", "workloads"}
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(v)) for v in hashes.values())
            or audit.get("ledger_sha256") != identity(self._snapshot(db))
        ):
            raise ValueError("held-envelope receipts or ledger state changed")
        baselines = db.execute("SELECT snapshot_sha256,evidence FROM bootstrap_baseline").fetchall()
        if len(baselines) != 1 or baselines[0][0] != audit.get("initial_baseline_sha256"):
            raise ValueError("held envelopes require the original unused-subscription baseline")
        baseline = json.loads(baselines[0][1])
        if any(
            audit.get(k) != baseline[k] for k in ("subscription_id", "tenant_id", "owner_identity")
        ):
            raise ValueError("held-envelope audit belongs to another account")
        observed = datetime.fromisoformat(reconciliation["observed_utc"])
        covered_until = datetime.fromisoformat(audit["fixed_costs_covered_until_utc"])
        baseline_time = datetime.fromisoformat(baseline["observed_utc"])
        expiry = datetime.fromisoformat(approval["expires_utc"])
        if (
            covered_until.tzinfo is None
            or not timedelta(0) <= now - observed <= timedelta(minutes=5)
            or not timedelta(0) <= now - baseline_time <= timedelta(hours=168)
            or not now < expiry <= covered_until <= baseline_time + timedelta(hours=168)
            or db.execute("SELECT 1 FROM reservations WHERE state!='pending'").fetchone()
            or db.execute(
                "SELECT count(*) FROM executions WHERE kind='provision' AND state='terminal'"
            ).fetchone()[0]
            != 1
            or not db.execute("SELECT 1 FROM reservations").fetchone()
        ):
            raise ValueError("held envelopes expired, released or lack completed infrastructure")
        return audit

    def reserve(
        self,
        batch_id,
        maximum,
        approval,
        reconciliation,
        *,
        now=None,
        execution_kind=None,
        resume_existing=False,
    ):
        now = now or datetime.now(timezone.utc)
        if approval.get("approved") is not True or not approval.get("source"):
            raise ValueError("cloud spending is not approved")
        if approval.get("currency") != "AUD" or reconciliation.get("currency") != "AUD":
            raise ValueError("currency mismatch")
        if approval.get("project") != "fleet-fault-triage":
            raise ValueError("approval is for a different project")
        expiry = datetime.fromisoformat(approval["expires_utc"])
        observed = datetime.fromisoformat(reconciliation["observed_utc"])
        if expiry.tzinfo is None or observed.tzinfo is None or now >= expiry:
            raise ValueError("approval expired or timestamps lack timezone")
        if not timedelta(0) <= now - observed <= timedelta(hours=24):
            raise ValueError("stale/future reconciliation")
        incomplete = reconciliation.get("complete_through_observed_time") is not True
        held = incomplete and "held_envelope_audit" in reconciliation
        bootstrap = incomplete and not held
        if type(resume_existing) is not bool or (
            resume_existing and (execution_kind != "provision" or not bootstrap)
        ):
            raise ValueError("only an existing bootstrap provision can be resumed")
        proof = (
            unused_subscription_basis(
                reconciliation, approval, "cleanup" if resume_existing else execution_kind, now
            )
            if bootstrap
            else None
        )
        limit, reserve = money(approval["budget_aud"]), money(approval["cleanup_reserve_aud"])
        cost = Decimal(0) if incomplete else money(reconciliation.get("accrued_aud"))
        maximum = money(maximum)
        if not limit > reserve >= 0 or maximum <= 0 or cost < 0:
            raise ValueError("invalid budget, reserve or marginal upper bound")
        with sqlite3.connect(self.path, isolation_level=None) as db:
            db.execute("BEGIN IMMEDIATE")
            observation_saved = False
            try:
                if held:
                    audit = self._held_envelope_basis(
                        db, reconciliation, approval, execution_kind, now
                    )
                    reported = audit.get("reported_positive_cost_aud")
                    cost = Decimal(0) if reported is None else money(reported)
                    if cost < 0:
                        raise ValueError("reported positive cost cannot be negative")
                    previous = db.execute(
                        "SELECT positive_cost FROM held_cost_high_watermark WHERE singleton=1"
                    ).fetchone()
                    previous = money(previous[0]) if previous else Decimal(0)
                    cost = max(cost, previous)
                    if cost > previous:
                        db.execute(
                            "INSERT OR REPLACE INTO held_cost_high_watermark VALUES (1,?,?)",
                            (str(cost), identity(audit)),
                        )
                        # A failed operation must not erase an observed positive cost.
                        db.execute("SAVEPOINT operation")
                        observation_saved = True
                if resume_existing:
                    reservation = db.execute(
                        "SELECT maximum,state FROM reservations WHERE batch_id=?", (batch_id,)
                    ).fetchone()
                    execution = db.execute(
                        "SELECT kind,state FROM executions WHERE batch_id=?", (batch_id,)
                    ).fetchone()
                    if (
                        not reservation
                        or reservation[1] != "pending"
                        or money(reservation[0]) != maximum
                        or execution != ("provision", "active")
                        or not db.execute(
                            "SELECT 1 FROM bootstrap_baseline WHERE snapshot_sha256=?",
                            (identity(proof),),
                        ).fetchone()
                    ):
                        raise ValueError(
                            "recovery requires the unchanged active provision reservation"
                        )
                    if db.execute(
                        "SELECT 1 FROM provision_recoveries WHERE batch_id=?", (batch_id,)
                    ).fetchone():
                        raise ValueError("provision recovery already claimed")
                if bootstrap:
                    tables = (
                        "reservations",
                        "usage",
                        "requests",
                        "executions",
                        "embedding_bounds",
                        "bootstrap_baseline",
                        "provision_recoveries",
                        "held_cost_high_watermark",
                    )
                    if execution_kind == "provision" and not resume_existing:
                        if any(
                            db.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
                            for table in tables
                        ):
                            raise ValueError(
                                "unused-subscription basis is only for the first provision"
                            )
                    elif execution_kind == "cleanup" and (
                        not db.execute(
                            "SELECT 1 FROM bootstrap_baseline WHERE snapshot_sha256=?",
                            (identity(proof),),
                        ).fetchone()
                        or not db.execute(
                            "SELECT 1 FROM executions WHERE kind='provision'"
                        ).fetchone()
                        or db.execute(
                            "SELECT 1 FROM executions WHERE kind NOT IN ('provision','cleanup')"
                        ).fetchone()
                        or db.execute(
                            "SELECT 1 FROM reservations WHERE state!='pending'"
                        ).fetchone()
                    ):
                        raise ValueError(
                            "cleanup requires the original held bootstrap reservations"
                        )
                if (
                    not resume_existing
                    and db.execute(
                        "SELECT 1 FROM reservations WHERE batch_id=?", (batch_id,)
                    ).fetchone()
                ):
                    raise ValueError("batch already reserved; reconcile its existing reservation")
                if execution_kind:
                    active = (
                        db.execute(
                            "SELECT 1 FROM executions WHERE state='active' AND batch_id!=?",
                            (batch_id,),
                        ).fetchone()
                        if resume_existing
                        else db.execute("SELECT 1 FROM executions WHERE state='active'").fetchone()
                    )
                    if active:
                        raise ValueError(
                            "another cloud operation is active; verify terminal state first"
                        )
                    group = (
                        ("train", "evaluate")
                        if execution_kind in {"train", "evaluate"}
                        else ("score", "monitor")
                    )
                    if (
                        execution_kind in group
                        and db.execute(
                            "SELECT count(*) FROM executions WHERE kind IN (?,?)", group
                        ).fetchone()[0]
                        >= 4
                    ):
                        raise ValueError("four-run cloud allowance exhausted, including failures")
                pending = sum(
                    (
                        money(x[0])
                        for x in db.execute(
                            "SELECT maximum FROM reservations WHERE state='pending'"
                        )
                    ),
                    Decimal(0),
                )
                recorded_cost = sum(
                    (
                        money(json.loads(row[0])["cost_aud"])
                        for row in db.execute(
                            "SELECT state FROM reservations WHERE state!='pending'"
                        )
                    ),
                    Decimal(0),
                )
                if incomplete:
                    # Charge known positive usage in addition to held envelopes. Partial
                    # overruns and corrections must not release unreconciled reservations.
                    floor = db.execute(
                        "SELECT positive_cost FROM held_cost_high_watermark WHERE singleton=1"
                    ).fetchone()
                    cost = max(
                        cost,
                        money(floor[0]) if floor else Decimal(0),
                        sum(
                            (
                                max(money(json.loads(row[0])["cost_aud"]), Decimal(0))
                                for row in db.execute("SELECT payload FROM usage")
                            ),
                            Decimal(0),
                        ),
                    )
                if cost < recorded_cost:
                    raise ValueError("accrued cost is below already reconciled operations")
                ceiling = limit if execution_kind == "cleanup" else limit - reserve
                if cost + pending + (Decimal(0) if resume_existing else maximum) > ceiling:
                    raise ValueError("next batch cannot fit budget and cleanup reserve")
                if not resume_existing:
                    db.execute(
                        "INSERT INTO reservations VALUES (?,?,?,?)",
                        (
                            batch_id,
                            str(maximum),
                            "pending",
                            utcnow(),
                        ),
                    )
                if execution_kind and not resume_existing:
                    db.execute(
                        "INSERT INTO executions VALUES (?,?,?)",
                        (batch_id, execution_kind, "active"),
                    )
                if bootstrap and execution_kind == "provision" and not resume_existing:
                    db.execute(
                        "INSERT INTO bootstrap_baseline VALUES (?,?)",
                        (identity(proof), canonical(proof)),
                    )
                if resume_existing:
                    db.execute(
                        "INSERT INTO provision_recoveries VALUES (?,?)", (batch_id, utcnow())
                    )
                db.execute("COMMIT")
            except BaseException:
                if observation_saved:
                    db.execute("ROLLBACK TO operation")
                    db.execute("COMMIT")
                else:
                    db.execute("ROLLBACK")
                raise
        return {
            "batch_id": batch_id,
            "reserved_aud": str(maximum),
            "provider_hard_cap": False,
            "budget_basis": "held_priced_envelopes"
            if held
            else "verified_unused_subscription"
            if bootstrap
            else "provider_reconciliation",
            "provider_accrued_verified": not incomplete,
            "resumed_existing_reservation": resume_existing,
        }

    def finish_execution(self, batch_id, *, provider_terminal_verified):
        if provider_terminal_verified is not True:
            raise ValueError("verify provider terminal state before releasing the lease")
        with sqlite3.connect(self.path) as db:
            if (
                db.execute(
                    "UPDATE executions SET state='terminal' WHERE batch_id=? AND state='active'",
                    (batch_id,),
                ).rowcount
                != 1
            ):
                raise ValueError("active operation not found")

    def reserve_embedding_tokens(self, batch_id, upper_bound):
        if type(upper_bound) is not int or upper_bound <= 0:
            raise ValueError("invalid embedding token bound")
        with sqlite3.connect(self.path, isolation_level=None) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                if not db.execute(
                    "SELECT 1 FROM executions WHERE batch_id=? AND state='active'", (batch_id,)
                ).fetchone():
                    raise ValueError("embedding lacks an active reservation")
                total = db.execute(
                    "SELECT coalesce(sum(tokens),0) FROM embedding_bounds"
                ).fetchone()[0]
                if total + upper_bound > 500_000:
                    raise ValueError("embedding token allowance exhausted, including failures")
                db.execute("INSERT INTO embedding_bounds VALUES (?,?)", (batch_id, upper_bound))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def consume_request(self, batch_id, kind, kind_limit, total_model_limit=750):
        with sqlite3.connect(self.path, isolation_level=None) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                active = db.execute(
                    "SELECT state FROM executions WHERE batch_id=?", (batch_id,)
                ).fetchone()
                if not active or active[0] != "active":
                    raise ValueError("request lacks an active reserved operation")
                used = db.execute("SELECT count(*) FROM requests WHERE kind=?", (kind,)).fetchone()[
                    0
                ]
                models = db.execute(
                    "SELECT count(*) FROM requests WHERE kind IN ('inference','embedding','judge')"
                ).fetchone()[0]
                if used >= kind_limit or (
                    kind in {"inference", "embedding", "judge"} and models >= total_model_limit
                ):
                    raise ValueError("project request allowance exhausted, including failures")
                db.execute("INSERT INTO requests(batch_id,kind) VALUES (?,?)", (batch_id, kind))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def reconcile(self, batch_id, receipt):
        if (
            receipt.get("batch_id") != batch_id
            or receipt.get("provider_usage_complete") is not True
        ):
            raise ValueError("provider usage proof must reconcile the exact reservation")
        if receipt.get("currency") != "AUD" or money(receipt.get("cost_aud")) < 0:
            raise ValueError("invalid reconciliation amount/currency")
        with sqlite3.connect(self.path) as db:
            changed = db.execute(
                "UPDATE reservations SET state=? WHERE batch_id=? AND state='pending'",
                (canonical(receipt), batch_id),
            ).rowcount
            if changed != 1:
                raise ValueError("pending reservation not found")

    def record_usage(self, record):
        required = {"record_id", "component", "run_id", "currency", "cost_aud", "record_type"}
        if not required <= record.keys() or record["currency"] != "AUD":
            raise ValueError("incomplete or mixed-currency usage")
        amount = money(record["cost_aud"])
        kind = record["record_type"]
        if kind not in {"ORIGINAL", "RETRACTION", "RESTATEMENT"}:
            raise ValueError("unknown provider record type")
        if (kind == "RETRACTION" and amount >= 0) or (kind != "RETRACTION" and amount < 0):
            raise ValueError("correction sign does not match record type")
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO usage VALUES (?,?)", (record["record_id"], canonical(record)))

    def report(self, successful_units, shared_overhead_aud=0):
        overhead = money(shared_overhead_aud)
        if overhead < 0:
            raise ValueError("negative shared overhead")
        with sqlite3.connect(self.path) as db:
            records = [json.loads(x[0]) for x in db.execute("SELECT payload FROM usage")]
            pending = db.execute(
                "SELECT count(*) FROM reservations WHERE state='pending'"
            ).fetchone()[0]
            # A funding checkpoint preserves unresolved historical accounting in
            # its own archive. Settling new work cannot make that history complete.
            historical = 0
            if db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='historical_reservations'"
            ).fetchone():
                historical = db.execute(
                    "SELECT count(*) FROM historical_reservations WHERE state='pending'"
                ).fetchone()[0]
        totals = {}
        for record in records:
            key = record["component"]
            totals[key] = totals.get(key, Decimal(0)) + money(record["cost_aud"])
        if any(value < 0 for value in totals.values()):
            raise ValueError("corrections are incomplete; negative component total")
        total = sum(totals.values(), Decimal(0))
        rows = []
        for component, cost in sorted(totals.items()):
            count = successful_units.get(component)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("successful-output denominator missing/invalid")
            # Allocate lifetime overhead proportionally to direct cost, not to successful rows.
            allocated = cost + (overhead * cost / total if total else Decimal(0))
            rows.append(
                {
                    "component": component,
                    "direct_aud": str(cost),
                    "fully_allocated_aud": str(allocated),
                    "successful_units": count,
                    "direct_cost_per_unit_aud": str(cost / count) if count else None,
                    "fully_allocated_cost_per_unit_aud": str(allocated / count) if count else None,
                }
            )
        return {
            "currency": "AUD",
            "evidence_class": "usage_accrual_not_invoice",
            "direct_total_aud": str(total),
            "shared_overhead_aud": str(overhead),
            "unallocated_aud": str(overhead if total == 0 else Decimal(0)),
            "recorded_usage_total_aud": str(total + overhead),
            "project_total_aud": None if pending or historical else str(total + overhead),
            "project_cost_complete": pending == 0 and historical == 0,
            "pending_reservations": pending,
            "historical_pending_reservations": historical,
            "allocation_policy": "shared overhead proportional to direct component cost",
            "unit_costs": rows,
            "invoice_reconciled": False,
        }
