"""Run the original synthetic rehearsal in a new output directory every time."""

import json
from datetime import datetime, timezone
from pathlib import Path

from fleet_fault_triage.rehearsal import rehearse

ROOT = Path(__file__).resolve().parents[1]


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "outputs" / f"demo-{stamp}"
    result = rehearse(ROOT, destination)
    print(json.dumps({"output": str(destination.relative_to(ROOT)), "result": result}, indent=2))


if __name__ == "__main__":
    main()
