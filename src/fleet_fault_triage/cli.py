"""Small project CLI. Platform operations stay with official provider clients."""

import argparse
import json
from pathlib import Path

from fleet_fault_triage.data import acquire, prepare, save_json
from fleet_fault_triage.doctor import doctor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="Verify local tools/imports; no cloud compute")
    check.add_argument(
        "--check-auth", action="store_true", help="Try token refresh, never print token"
    )
    check.add_argument("--output", type=Path)
    repeat = commands.add_parser(
        "reproduce", help="Refit the selected local recipe, without searching"
    )
    repeat.add_argument("--data-dir", type=Path, default=Path("data"))
    repeat.add_argument("--previous-dir", type=Path, default=Path("artifacts/local"))
    repeat.add_argument("--output-dir", type=Path, default=Path("artifacts/reproduction"))
    demo = commands.add_parser("rehearse", help="Synthetic end-to-end rehearsal; zero network/fits")
    demo.add_argument("--output-dir", type=Path, required=True)
    demo.add_argument("--root", type=Path, default=Path.cwd())
    verify = commands.add_parser("verify-evidence", help="Verify saved evidence hashes")
    verify.add_argument("--root", type=Path, required=True)
    review = commands.add_parser("review", help="Append a human disposition; never alter a score")
    review.add_argument("--state", type=Path, required=True)
    review.add_argument("--case-id", required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--disposition", required=True)
    review.add_argument("--reason", required=True)
    review.add_argument("--previous-event-hash")
    review.add_argument("--synthetic", action="store_true")
    freeze = commands.add_parser(
        "freeze-inputs", help="Export frozen recipe/parity; no new fits or test analysis"
    )
    freeze.add_argument("--data-dir", type=Path, default=Path("data"))
    freeze.add_argument("--previous-dir", type=Path, default=Path("artifacts/local"))
    freeze.add_argument("--output-dir", type=Path, required=True)
    task = commands.add_parser(
        "cloud-task", help="Approved Databricks wheel task; blocked locally by default"
    )
    task.add_argument("--task", choices=["train", "evaluate", "score", "monitor"], required=True)
    task.add_argument("--config", type=Path, required=True)
    task.add_argument("--permit", type=Path, required=True)
    identity_command = commands.add_parser(
        "runtime-identity", help="Print exact installed source/dependency hashes"
    )
    identity_command.add_argument("--output", type=Path, required=True)
    cloud = commands.add_parser(
        "cloud", help="Manual approved project operation; defaults refuse spend"
    )
    cloud.add_argument(
        "--action",
        required=True,
        choices=[
            "stage",
            "train",
            "evaluate",
            "score",
            "monitor",
            "assistant",
            "judge",
            "index_publish",
            "provision",
            "cleanup",
            "promote",
        ],
    )
    for name in ("config", "approval", "reconciliation", "ledger", "request", "output"):
        cloud.add_argument("--" + name, type=Path, required=True)
    cloud.add_argument("--root", type=Path, default=Path.cwd())
    imported = commands.add_parser(
        "import-batch", help="Import identity-verified cloud scoring results for review"
    )
    imported.add_argument("--result", type=Path, required=True)
    imported.add_argument("--release", type=Path, required=True)
    imported.add_argument("--state", type=Path, required=True)
    costs = commands.add_parser(
        "cost-report", help="Import provider usage and calculate attributable unit costs"
    )
    costs.add_argument("--ledger", type=Path, required=True)
    costs.add_argument("--input", type=Path, required=True)
    costs.add_argument("--output", type=Path, required=True)
    rubric = commands.add_parser(
        "evaluate-briefs", help="Evaluate frozen outcomes against actual human rubric"
    )
    rubric.add_argument("--input", type=Path, required=True)
    rubric.add_argument("--output", type=Path, required=True)
    for name in ("acquire", "prepare", "train"):
        command = commands.add_parser(name)
        command.add_argument("--data-dir", type=Path, default=Path("data"))
        if name == "train":
            command.add_argument("--output-dir", type=Path, default=Path("artifacts/local"))
            command.add_argument("--resume-cv", type=Path, help="Reuse a complete saved comparison")
    args = parser.parse_args()
    if args.command == "runtime-identity":
        from fleet_fault_triage.runtime import runtime_identity

        save_json(args.output, runtime_identity())
        return 0
    if args.command == "cloud":
        import asyncio

        from fleet_fault_triage.operations import execute_request, gate_from_files

        gate = gate_from_files(args.config, args.approval, args.reconciliation, args.ledger)
        result = asyncio.run(
            execute_request(gate, args.action, args.root, json.loads(args.request.read_text()))
        )
        save_json(args.output, result)
        print(json.dumps({"saved": str(args.output), "human_review": "required"}))
        return 0
    if args.command == "import-batch":
        from fleet_fault_triage.contracts import Release, Score
        from fleet_fault_triage.lifecycle import CaseStore

        release = Release.model_validate_json(args.release.read_text())
        result = json.loads(args.result.read_text())
        if (
            release.origin != "azure"
            or result.get("origin") != "azure"
            or result.get("release_sha256") != release.sha256
        ):
            raise ValueError("cloud scoring/release identity mismatch")
        rows = [Score.model_validate(row) for row in result["scores"]]
        if any(row.origin != "azure" for row in rows):
            raise ValueError("synthetic score cannot be imported as cloud evidence")
        CaseStore(args.state).commit_batch(rows[0].batch_id, result["input_sha256"], release, rows)
        print(json.dumps({"cases": len(rows), "review": "pending"}))
        return 0
    if args.command == "cost-report":
        from fleet_fault_triage.finops import CostLedger

        inputs = json.loads(args.input.read_text())
        ledger = CostLedger(args.ledger)
        for record in inputs["usage_records"]:
            ledger.record_usage(record)
        save_json(
            args.output, ledger.report(inputs["successful_units"], inputs["shared_overhead_aud"])
        )
        return 0
    if args.command == "evaluate-briefs":
        from fleet_fault_triage.evaluation import evaluate_briefs

        result = evaluate_briefs(**json.loads(args.input.read_text()))
        save_json(args.output, result)
        return 0 if result.get("release_allowed") else 2
    if args.command == "rehearse":
        from fleet_fault_triage.rehearsal import rehearse

        print(json.dumps(rehearse(args.root.resolve(), args.output_dir.resolve()), indent=2))
        return 0
    if args.command == "verify-evidence":
        from fleet_fault_triage.evidence import verify_manifest

        print(json.dumps(verify_manifest(args.root), indent=2))
        return 0
    if args.command == "review":
        from fleet_fault_triage.lifecycle import CaseStore

        event = CaseStore(args.state).review(
            args.case_id,
            args.reviewer,
            args.disposition,
            args.reason,
            args.previous_event_hash,
            synthetic=args.synthetic,
        )
        print(json.dumps(event, indent=2))
        return 0
    if args.command == "freeze-inputs":
        from fleet_fault_triage.cloud import freeze_inputs
        from fleet_fault_triage.model import source_identity

        source = source_identity()
        if source["git_dirty"]:
            raise ValueError("freeze inputs only from a committed, clean source revision")
        report = freeze_inputs(
            args.data_dir,
            args.previous_dir,
            args.output_dir,
            code_sha=source["git_commit"],
            lock_sha256=source["lock_sha256"],
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "cloud-task":
        from fleet_fault_triage.cloud_tasks import run_task

        print(json.dumps(run_task(args.config, args.permit, args.task), indent=2))
        return 0
    if args.command == "doctor":
        report = doctor(args.check_auth)
        if args.output:
            save_json(args.output, report)
        print(json.dumps(report, indent=2))
        return 0 if report["local_ready"] else 1
    if args.command == "acquire":
        report = acquire(args.data_dir)
    elif args.command == "prepare":
        report = prepare(args.data_dir)
    elif args.command == "reproduce":
        from fleet_fault_triage.model import reproduce

        report = reproduce(args.data_dir, args.previous_dir, args.output_dir)
    else:
        from fleet_fault_triage.model import train

        report = train(args.data_dir, args.output_dir, args.resume_cv)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report.get("development_gate_passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
