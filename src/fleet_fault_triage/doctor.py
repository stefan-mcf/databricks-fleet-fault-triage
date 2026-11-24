"""Read-only, redacted readiness checks. Never starts compute or invokes a model."""

import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys

PACKAGES = {
    "mlflow": "mlflow",
    "scikit-learn": "sklearn",
    "databricks-sdk": "databricks.sdk",
    "azure-identity": "azure.identity",
    "azure-ai-projects": "azure.ai.projects",
    "azure-search-documents": "azure.search.documents",
    "agent-framework-foundry": "agent_framework.foundry",
}


def doctor(check_auth: bool = False) -> dict:
    packages = {}
    for package, module in PACKAGES.items():
        try:
            importlib.import_module(module)
            packages[package] = {"version": importlib.metadata.version(package), "importable": True}
        except ImportError:
            packages[package] = {"version": None, "importable": False}
    tools = {}
    for name, args in {
        "az": ["version", "-o", "json"],
        "databricks": ["--version"],
        "mlflow": ["--version"],
        "uv": ["--version"],
        "git": ["--version"],
        "gh": ["--version"],
    }.items():
        executable = shutil.which(name)
        if not executable:
            tools[name] = {"installed": False}
            continue
        result = subprocess.run([executable, *args], capture_output=True, text=True, timeout=30)
        output = result.stdout
        if name == "az" and result.returncode == 0:
            output = json.loads(output)["azure-cli"]
        tools[name] = {
            "installed": True,
            "usable": result.returncode == 0,
            "version": output.splitlines()[0] if output else None,
        }
    azure_auth = "not_checked"
    if check_auth and tools["az"]["installed"]:
        result = subprocess.run(
            ["az", "account", "get-access-token", "--query", "expires_on", "-o", "tsv"],
            capture_output=True,
            text=True,
            timeout=45,
        )
        azure_auth = "token_refresh_succeeded" if result.returncode == 0 else "login_required"
    return {
        "schema_version": 1,
        "scope": "local tools and imports; no cloud workload invoked",
        "python": sys.version.split()[0],
        "tools": tools,
        "packages": packages,
        "local_ready": all(t.get("usable", False) for t in tools.values())
        and all(p["importable"] for p in packages.values()),
        "azure_auth": azure_auth,
        "cloud_ready": False,
        "cloud_pending": [
            "SM Systems identity/subscription selection",
            "workspace and service access",
            "local acceptance",
            "priced allowance approval",
        ],
    }
