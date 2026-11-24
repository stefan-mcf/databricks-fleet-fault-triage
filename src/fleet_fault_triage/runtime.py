"""Bind installed wheel sources and critical numerical dependencies to an approval."""

from importlib.metadata import version
from pathlib import Path

from fleet_fault_triage.data import digest

DEPENDENCIES = ("numpy", "pandas", "scipy", "scikit-learn", "mlflow", "databricks-sdk", "pydantic")


def runtime_identity():
    return {
        "package_sources": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "dependencies": {name: version(name) for name in DEPENDENCIES},
    }


def verify_runtime(config):
    if runtime_identity() != config.get("runtime_identity"):
        raise ValueError("installed package/dependencies differ from approved runtime identity")
