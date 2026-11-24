"""Hash-bound private evidence packages and a small read-only case report."""

import html
import json
import re
from pathlib import Path

from fleet_fault_triage.contracts import SCOPE, identity, utcnow
from fleet_fault_triage.data import digest, save_json


def export_manifest(root: Path, names, source_sha, evidence_class):
    if evidence_class not in {"synthetic_rehearsal", "local", "azure"}:
        raise ValueError("evidence class required")
    rows = []
    for name in names:
        original = root / name
        if any(p.is_symlink() for p in (original, *original.parents) if p.is_relative_to(root)):
            raise ValueError("evidence symlinks are excluded")
        target = (root / name).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file() or target.is_symlink():
            raise ValueError("evidence path outside the approved package")
        if target.suffix not in {".json", ".md", ".html"}:
            raise ValueError("raw/model/credential artifact excluded from review package")
        text = target.read_text()
        if re.search(
            r"(?i)(bearer\s+[\w.-]{12,}|[?&](?:sig|token|key)=|-----BEGIN .*PRIVATE KEY)", text
        ):
            raise ValueError("potential secret in evidence export")
        rows.append({"path": name, "sha256": digest(target), "bytes": target.stat().st_size})
    report = {
        "schema_version": 1,
        "created_utc": utcnow(),
        "source_sha": source_sha,
        "evidence_class": evidence_class,
        "scope": SCOPE,
        "artifacts": rows,
        "manifest_sha256": identity(rows),
    }
    save_json(root / "evidence-manifest.json", report)
    verify_manifest(root)
    return report


def verify_manifest(root):
    manifest = json.loads((root / "evidence-manifest.json").read_text())
    if identity(manifest["artifacts"]) != manifest["manifest_sha256"]:
        raise ValueError("evidence manifest changed")
    for row in manifest["artifacts"]:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != row["sha256"]:
            raise ValueError("evidence file hash/path mismatch")
    return manifest


def case_report(root: Path, scores, briefs, reviews, origin):
    escape = html.escape
    cards = []
    for score in scores:
        brief = briefs[score["case_id"]]
        decisions = reviews.get(score["case_id"], [])
        content = ""
        for label, values in (
            ("Recorded facts", [x for x in brief["facts"] if x != SCOPE]),
            (
                "Model estimate",
                brief["estimates"] or ["No score: input quality policy rejected this record."],
            ),
            ("Missing evidence", brief["missing_evidence"]),
            ("Proposed verification", brief["proposed_checks"]),
        ):
            content += (
                f"<h3>{label}</h3><ul>"
                + "".join(f"<li>{escape(line)}</li>" for line in values)
                + "</ul>"
            )
        content += (
            "<p class='label'>Citations: "
            + escape(", ".join(brief["citations"]) or "None — deterministic report")
            + "</p>"
        )
        history = "".join(
            f"<li>{escape(x['disposition'])}: {escape(x['reason'])}</li>" for x in decisions
        )
        cards.append(
            f"<article><p class='label'>{escape(score['route'].replace('_', ' ').capitalize())} · {escape(brief['status'])}</p>"
            f"<h2>Record {escape(score['record_id'])}</h2>{content}"
            f"<h3>Review history</h3><ul>{history or '<li>Human review pending</li>'}</ul></article>"
        )
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>Fleet Fault Triage — case review</title><style>
body{margin:0;background:#f6f5f1;color:#16242a;font:16px/1.6 system-ui,sans-serif}
main{max-width:960px;margin:56px auto;padding:0 24px}.label{font-size:13px;color:#315c60}
h1{font-size:38px;line-height:1.15}h2{font-size:23px}h3{font-size:16px}
article{background:white;padding:24px 28px;margin:24px 0;border:1px solid #d8dfdd}
.scope{border-left:3px solid #315c60;padding-left:18px}li{margin:8px 0}
</style><main><p class="label">FLEET FAULT TRIAGE · READ-ONLY REVIEW</p>
<h1>Recorded faults, traceable decisions.</h1>"""
    page += f"<p>Evidence class: <strong>{escape(origin)}</strong></p><p class='scope'>{escape(SCOPE)}</p>"
    page += "".join(cards) + "</main></html>"
    root.mkdir(parents=True, exist_ok=True)
    (root / "case-review.html").write_text(page)
    return root / "case-review.html"
