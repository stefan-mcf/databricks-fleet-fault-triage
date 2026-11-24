"""Atomic batches, append-only human review and serial complete-release promotion."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from fleet_fault_triage.contracts import Release, Score, canonical, identity, utcnow


class CaseStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY, input_hash TEXT NOT NULL,
                    release_hash TEXT NOT NULL, output_hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reviews (
                    event_hash TEXT PRIMARY KEY, case_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL, payload TEXT NOT NULL,
                    UNIQUE(case_id, sequence));
                CREATE TABLE IF NOT EXISTS releases (
                    release_hash TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS aliases (
                    name TEXT PRIMARY KEY, release_hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS promotions (
                    sequence INTEGER PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS evaluations (
                    release_hash TEXT PRIMARY KEY, status TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS consumed_tests (
                    test_hash TEXT PRIMARY KEY, release_hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS assistant_suites (
                    suite_hash TEXT PRIMARY KEY, release_hash TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def commit_batch(self, batch_id, input_hash, release: Release, scores: list[Score]):
        rows = [x.model_dump() for x in scores]
        if not rows or len({x.case_id for x in scores}) != len(rows):
            raise ValueError("batch must have nonempty unique cases")
        if any(x.batch_id != batch_id or x.release_sha256 != release.sha256 for x in scores):
            raise ValueError("case identity differs from batch/release")
        output_hash = identity(rows)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT input_hash, release_hash, output_hash FROM batches WHERE batch_id=?",
                (batch_id,),
            ).fetchone()
            expected = (input_hash, release.sha256, output_hash)
            if existing:
                if existing != expected:
                    raise ValueError("batch replay identity/output mismatch")
            else:
                db.execute("INSERT INTO batches VALUES (?,?,?,?)", (batch_id, *expected))
                db.executemany(
                    "INSERT INTO cases VALUES (?,?,?)",
                    [(x.case_id, batch_id, canonical(x.model_dump())) for x in scores],
                )
                db.execute(
                    "INSERT OR IGNORE INTO releases VALUES (?,?)",
                    (release.sha256, canonical(release.model_dump())),
                )
        saved = self.batch(batch_id)
        if identity(saved) != output_hash:
            raise ValueError("saved batch readback mismatch")
        return saved

    def batch(self, batch_id):
        with self.connect() as db:
            return [
                json.loads(x[0])
                for x in db.execute(
                    "SELECT payload FROM cases WHERE batch_id=? ORDER BY rowid", (batch_id,)
                )
            ]

    def case(self, case_id):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM cases WHERE case_id=?", (case_id,)).fetchone()
        if row is None:
            raise KeyError("case not in authorised store")
        return Score.model_validate_json(row[0])

    def reviews(self, case_id):
        with self.connect() as db:
            events = [
                json.loads(x[0])
                for x in db.execute(
                    "SELECT payload FROM reviews WHERE case_id=? ORDER BY sequence", (case_id,)
                )
            ]
        previous = None
        for event in events:
            content = {k: v for k, v in event.items() if k != "event_hash"}
            if event["previous_event_hash"] != previous or identity(content) != event["event_hash"]:
                raise ValueError("review hash chain is corrupt")
            previous = event["event_hash"]
        return events

    def review(
        self, case_id, reviewer, disposition, reason, previous_event_hash, *, synthetic=False
    ):
        if disposition not in {"aps_review", "other_fault_review", "insufficient_evidence"}:
            raise ValueError("invalid review disposition")
        if not reviewer.strip() or not reason.strip():
            raise ValueError("attributable reviewer and reason are required")
        case = self.case(case_id)
        if case.origin == "synthetic_rehearsal" and not synthetic:
            raise ValueError("synthetic cases require explicitly synthetic review events")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            last = db.execute(
                "SELECT event_hash,sequence FROM reviews WHERE case_id=? ORDER BY sequence DESC LIMIT 1",
                (case_id,),
            ).fetchone()
            if previous_event_hash != (last[0] if last else None):
                raise ValueError("stale review; re-read and append a correction")
            event = {
                "case_id": case_id,
                "release_sha256": case.release_sha256,
                "reviewer": reviewer,
                "disposition": disposition,
                "reason": reason,
                "previous_event_hash": previous_event_hash,
                "created_utc": utcnow(),
                "synthetic": synthetic,
                "sequence": last[1] + 1 if last else 1,
            }
            event["event_hash"] = identity(event)
            db.execute(
                "INSERT INTO reviews VALUES (?,?,?,?)",
                (
                    event["event_hash"],
                    case_id,
                    event["sequence"],
                    canonical(event),
                ),
            )
        return self.reviews(case_id)[-1]

    def alias(self, name="champion"):
        with self.connect() as db:
            row = db.execute("SELECT release_hash FROM aliases WHERE name=?", (name,)).fetchone()
        return row[0] if row else None

    def promote(self, release, expected_previous, approval, gates, *, provider=None):
        required = {"model_quality", "assistant_quality", "permissions", "reproducibility"}
        if set(gates) != required or any(value is not True for value in gates.values()):
            raise ValueError("release quality/security evidence is incomplete or rejected")
        if approval.get("release_sha256") != release.sha256 or not all(
            approval.get(k) for k in ("reviewer", "source", "approved_utc")
        ):
            raise ValueError("exact attributable release approval required")
        if (release.origin == "synthetic_rehearsal") != (approval.get("synthetic") is True):
            raise ValueError("synthetic approval cannot stand in for real approval")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT release_hash FROM aliases WHERE name='champion'").fetchone()
            if (row[0] if row else None) != expected_previous:
                raise ValueError("champion changed concurrently")
            # The sole model-owner process holds this transaction through provider readback.
            if provider is not None:
                provider.promote(release, expected_previous)
            db.execute(
                "INSERT OR IGNORE INTO releases VALUES (?,?)",
                (
                    release.sha256,
                    canonical(release.model_dump()),
                ),
            )
            db.execute("INSERT OR REPLACE INTO aliases VALUES ('champion',?)", (release.sha256,))
            db.execute(
                "INSERT INTO promotions(payload) VALUES (?)",
                (
                    canonical(
                        {
                            "previous": expected_previous,
                            "next": release.sha256,
                            "approval": approval,
                            "gates": gates,
                            "created_utc": utcnow(),
                        }
                    ),
                ),
            )
        if self.alias() != release.sha256:
            raise ValueError("promotion persisted readback failed")
        return release.sha256

    def reserve_evaluation(self, release: Release, test_hash=None):
        if release.origin != "azure":
            raise ValueError("external evaluation requires a frozen Azure release")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not test_hash or len(test_hash) != 64:
                raise ValueError("exact sealed test hash required")
            db.execute("INSERT INTO consumed_tests VALUES (?,?)", (test_hash, release.sha256))
            # A failed/interrupted evaluation remains consumed; no silent rerun.
            db.execute(
                "INSERT INTO evaluations VALUES (?,?,?)",
                (
                    release.sha256,
                    "consumed_before_read",
                    canonical({"started_utc": utcnow()}),
                ),
            )

    def reserve_assistant_suite(self, suite_hash, release_hash):
        with self.connect() as db:
            db.execute("INSERT INTO assistant_suites VALUES (?,?)", (suite_hash, release_hash))

    def finish_evaluation(self, release, report):
        with self.connect() as db:
            changed = db.execute(
                "UPDATE evaluations SET status=?,payload=? WHERE release_hash=? AND status=?",
                ("completed", canonical(report), release.sha256, "consumed_before_read"),
            ).rowcount
            if changed != 1:
                raise ValueError("evaluation reservation missing/already completed")
