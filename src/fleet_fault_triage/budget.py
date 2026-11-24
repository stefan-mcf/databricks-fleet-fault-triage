"""Persist attempted local fits, including failures, under the frozen 100-fit ceiling."""

import fcntl
import json
import uuid
from datetime import datetime, timezone

from fleet_fault_triage.data import digest, save_json


class FitBudget:
    def __init__(self, data_dir, expected, operation):
        self.root = data_dir / "prepared"
        self.expected = expected
        self.operation = operation
        self.operation_id = uuid.uuid4().hex
        self.ledger_path = self.root / "fit-ledger.json"

    def __enter__(self):
        self.lock = (self.root / "fit-ledger.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX)
            manifest_hash = digest(self.root / "manifest.json")
            self.ledger = (
                json.loads(self.ledger_path.read_text())
                if self.ledger_path.exists()
                else {
                    "schema_version": 1,
                    "data_manifest_sha256": manifest_hash,
                    "attempts": [],
                }
            )
            if self.ledger["data_manifest_sha256"] != manifest_hash:
                raise ValueError("fit allowance belongs to a different frozen manifest")
            self.total = sum(attempt["count"] for attempt in self.ledger["attempts"])
            if self.total + self.expected > 100:
                raise ValueError(
                    f"100-fit allowance exceeded: {self.total} used, {self.expected} requested"
                )
            self.started = 0
            return self
        except BaseException:
            self.lock.close()
            raise

    def fit(self, model, features, labels):
        if self.started >= self.expected:
            raise ValueError("operation attempted more fits than declared")
        self.started += 1
        self.total += 1
        self.ledger["attempts"].append(
            {
                "count": 1,
                "operation": self.operation,
                "operation_id": self.operation_id,
                "attempted_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        # Persist before calling fit, so a failed or interrupted fit still consumes allowance.
        save_json(self.ledger_path, self.ledger)
        self.ledger_sha256 = digest(self.ledger_path)
        return model.fit(features, labels)

    def __exit__(self, *exc):
        self.lock.close()
