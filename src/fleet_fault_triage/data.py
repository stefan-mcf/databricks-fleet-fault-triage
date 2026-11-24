"""Training-only acquisition, validation and duplicate-aware partitions."""

import csv
import hashlib
import json
import shutil
import sqlite3
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SOURCE = "https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip"
TRAIN = "aps_failure_training_set.csv"
TEST = "aps_failure_test_set.csv"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def acquire(root: Path) -> dict:
    """Preserve source bytes. Test data is only copied/hashed, never parsed."""
    raw = root / "raw"
    manifest_path = raw / "source.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if digest(raw / "source.zip") != manifest["archive_sha256"]:
            raise ValueError("cached source archive changed")
        for name, expected in manifest["files"].items():
            if digest(raw / name) != expected["sha256"]:
                raise ValueError(f"cached source changed: {name}")
        return manifest
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "source.zip"
    if not archive.exists():
        temporary = raw / "download.tmp"
        with urllib.request.urlopen(SOURCE, timeout=90) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(archive)
    files = {}
    with zipfile.ZipFile(archive) as bundle:
        for name in (TRAIN, TEST, "aps_failure_description.txt"):
            matches = [n for n in bundle.namelist() if Path(n).name == name]
            if len(matches) != 1:
                raise ValueError(f"unexpected upstream archive layout: {name}")
            with bundle.open(matches[0]) as source, (raw / name).open("wb") as out:
                shutil.copyfileobj(source, out)
            files[name] = {"sha256": digest(raw / name), "bytes": (raw / name).stat().st_size}
    manifest = {
        "schema_version": 1,
        "source": SOURCE,
        "doi": "10.24432/C51S51",
        "archive_sha256": digest(archive),
        "retrieved_utc": datetime.fromtimestamp(archive.stat().st_mtime, timezone.utc).isoformat(),
        "files": files,
        "notices": ["UCI lists CC BY 4.0", "Donor notice reproduces GPL-3.0-or-later"],
        "external_test": "sealed: bytes/hash only; not parsed or used in local development",
    }
    save_json(manifest_path, manifest)
    return manifest


def read_training(path: Path) -> pd.DataFrame:
    if path.name != TRAIN:
        raise ValueError("only the supplied training file may enter preparation")
    with path.open() as stream:
        for skip, line in enumerate(stream):
            if line.startswith("class,"):
                header = next(csv.reader([line]))
                if len(set(header)) != len(header):
                    raise ValueError("duplicate source columns")
                break
        else:
            raise ValueError("training CSV header not found")
    frame = pd.read_csv(path, skiprows=skip, dtype=str, keep_default_na=False)
    if frame.columns[0] != "class" or len(frame.columns) != 171:
        raise ValueError("expected class plus 170 source predictors")
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("duplicate source columns")
    return frame


def clean(frame: pd.DataFrame, source_hash: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "class" not in frame or frame.columns[0] != "class":
        raise ValueError("class must be the first source column")
    features = frame.drop(columns="class")
    numeric = features.replace("na", np.nan).apply(pd.to_numeric, errors="coerce")
    malformed = ((numeric.isna() & features.ne("na")) | np.isinf(numeric)).any(axis=1)
    reasons = pd.Series("", index=frame.index, dtype=str)
    reasons.loc[malformed] = "invalid_numeric"
    reasons.loc[numeric.isna().all(axis=1)] = "all_missing"
    reasons.loc[~frame["class"].isin(["pos", "neg"])] = "invalid_label"
    numeric["record_id"] = [
        hashlib.sha256(f"{source_hash}:{ordinal}".encode()).hexdigest()
        for ordinal in range(len(frame))
    ]
    numeric["label"] = frame["class"].map({"neg": 0, "pos": 1})
    # Canonical floats identify exact feature duplicates, never vehicle identities.
    numeric["group_id"] = [
        hashlib.sha256(np.asarray(row, dtype="<f8").tobytes()).hexdigest()
        for row in numeric[features.columns].to_numpy()
    ]
    eligible = numeric.loc[reasons.eq("")]
    conflicts = eligible.groupby("group_id")["label"].nunique()
    reasons.loc[numeric["group_id"].isin(conflicts[conflicts > 1].index)] = "conflicting_labels"
    quarantine = numeric.loc[reasons.ne(""), ["record_id"]].copy()
    quarantine["reason"] = reasons.loc[reasons.ne("")]
    accepted = numeric.loc[reasons.eq("")].copy()
    accepted["label"] = accepted["label"].astype(int)
    return accepted, quarantine


def partition(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    result["partition"] = "development"
    for fold, (_, indices) in enumerate(splitter.split(frame, frame.label, groups=frame.group_id)):
        if fold < 3:
            result.loc[result.index[indices], "partition"] = (
                "calibration" if fold == 0 else "decision"
            )
    if result.groupby("group_id").partition.nunique().max() != 1:
        raise ValueError("duplicate-group leakage across partitions")
    return result


def prepare(root: Path) -> dict:
    source = acquire(root)
    frame = read_training(root / "raw" / TRAIN)
    if len(frame) != 60_000 or frame["class"].eq("pos").sum() != 1_000:
        raise ValueError("supplied training population differs from the registered source")
    accepted, quarantine = clean(frame, source["files"][TRAIN]["sha256"])
    prepared = partition(accepted)
    features = list(frame.columns[1:])
    development = prepared.loc[prepared.partition.eq("development"), features]
    cutoff = float(development.isna().mean(axis=1).quantile(0.99))
    prepared["eligible"] = prepared[features].isna().mean(axis=1).le(cutoff)
    destination = root / "prepared"
    destination.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(destination / "training.parquet", index=False)
    quarantine.to_parquet(destination / "quarantine.parquet", index=False)
    with sqlite3.connect(destination / "quality.sqlite") as connection:
        prepared[["record_id", "partition", "label", "eligible"]].to_sql(
            "records", connection, if_exists="replace", index=False
        )
        quarantine.to_sql("quarantine", connection, if_exists="replace", index=False)
        counts = connection.execute(
            "SELECT partition, count(*), sum(label), sum(eligible) "
            "FROM records GROUP BY partition ORDER BY partition"
        ).fetchall()
        reconciled = connection.execute(
            "SELECT (SELECT count(*) FROM records) + (SELECT count(*) FROM quarantine)"
        ).fetchone()[0]
    if reconciled != len(frame):
        raise ValueError("input conservation failed")
    summary = {
        "schema_version": 1,
        "origin": "UCI training only; local preparation",
        "source_archive_sha256": source["archive_sha256"],
        "training_sha256": source["files"][TRAIN]["sha256"],
        "prepared_sha256": digest(destination / "training.parquet"),
        "features": features,
        "feature_count": len(features),
        "seed": 42,
        "input_rows": len(frame),
        "accepted_rows": len(prepared),
        "quarantined_rows": len(quarantine),
        "quarantine_reasons": quarantine.reason.value_counts().to_dict(),
        "missing_fraction_limit": cutoff,
        "partitions": [
            dict(zip(["name", "rows", "positives", "eligible"], row, strict=True)) for row in counts
        ],
        "split_sha256": hashlib.sha256(
            prepared[["record_id", "partition"]].to_csv(index=False).encode()
        ).hexdigest(),
        "exact_duplicate_groups_crossing_partitions": 0,
        "external_test_opened_for_analysis": False,
    }
    save_json(destination / "manifest.json", summary)
    return summary
