"""Versioned application contracts; synthetic rehearsals never imply hosted evidence."""

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Release(Contract):
    schema_version: Literal[1] = 1
    origin: Literal["synthetic_rehearsal", "azure"]
    code_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    features: list[str] = Field(min_length=1)
    model_uri: str = Field(
        pattern=r"^models:/[A-Za-z_][\w]*\.[A-Za-z_][\w]*\.[A-Za-z_][\w]*/[1-9]\d*$"
    )
    model_run_id: str = Field(min_length=1, max_length=100)
    threshold: float = Field(ge=0, le=1)
    missing_fraction_limit: float = Field(ge=0, lt=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_version: str = Field(min_length=1)
    embedding_version: str = Field(min_length=1)
    generation_version: str = Field(min_length=1)

    @field_validator("features")
    @classmethod
    def feature_allowlist(cls, value):
        if len(set(value)) != len(value) or any(
            not re.fullmatch(r"[a-z][a-z0-9_]*", x)
            or x in {"label", "class", "record_id", "partition", "timestamp"}
            for x in value
        ):
            raise ValueError("features must be a unique numeric-predictor allowlist")
        return value

    @property
    def sha256(self):
        return identity(self.model_dump())


class Score(Contract):
    case_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_id: str = Field(min_length=1, max_length=128)
    batch_id: str = Field(min_length=1, max_length=100)
    release_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    origin: Literal["synthetic_rehearsal", "azure"]
    probability: float | None = Field(default=None, ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    route: Literal["aps_review", "other_fault_review", "insufficient_evidence"]
    quality_reason: str | None = None


class Brief(Contract):
    case_id: str
    release_sha256: str
    status: Literal["draft", "incomplete"] = "draft"
    facts: list[str]
    estimates: list[str]
    missing_evidence: list[str]
    proposed_checks: list[str]
    citations: list[str]
    origin: Literal["deterministic_report", "synthetic_rehearsal", "foundry"]
    failure_reason: str | None = None


SCOPE = (
    "Recorded APS versus other recorded failures in the Scania benchmark; negatives are not "
    "healthy trucks. Educational software/data investigation only. No future-failure, causal, "
    "repair, safety, field-validation or autonomous-haulage authority."
)
