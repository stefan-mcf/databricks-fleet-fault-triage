"""Check the public package's links, JSON, provenance and publication exclusions."""

import ast
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOTS = {
    ".agent-handoffs",
    ".azure",
    ".databricks",
    ".venv",
    "artifacts",
    "data",
    "datasets",
    "models",
    "outputs",
    "research",
}
PRIVATE_SUFFIXES = {".csv", ".parquet", ".pkl", ".pickle", ".joblib", ".skops", ".onnx", ".zip"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".yml", ".yaml", ".toml", ".sql", ".svg", ".txt", ".cff"}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "owner-only Drive link": re.compile(r"https?://(?:drive|docs)\.google\.com/[^\s<>\"]+"),
    "local user path": re.compile(r"(?:/(?:Users|home)/[^/\s]+/|[A-Z]:\\Users\\[^\\\s]+\\)"),
    "signed URL": re.compile(r"[?&](?:sig|X-Amz-Signature)=[A-Za-z0-9%+/=_-]{16,}", re.I),
}
GUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
PUBLIC_PROVIDER_IDS = set(
    json.loads((Path(__file__).parent / "public_provider_identifiers.json").read_text())[
        "public_guids"
    ]
)
EMAIL = re.compile(r"\b[A-Za-z0-9_.+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
CREDENTIAL_LITERAL = re.compile(
    r"""(?i)["']?(?:client_secret|api_key|access_token|refresh_token|password)["']?\s*[:=]\s*["']([^"'\s]{20,})["']"""
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]\n]*\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)")
HTML_LINK = re.compile(r"\b(?:href|src)=[\"']([^\"']+)[\"']")
EVIDENCE_EXTRACTS = {
    "evidence/classifier-benchmark.json",
    "evidence/databricks-delivery.json",
    "evidence/delta-scoring.json",
    "evidence/assistant-status.json",
    "evidence/preservation.json",
}
AST_CANONICALIZATION = {
    "python_major_minor": "3.12",
    "format": "ast.dump(include_attributes=False)",
}
EXECUTION_FUNCTIONS = {
    (
        "src/fleet_fault_triage/cloud.py",
        "train_registered",
    ): "13577cf4ebf3bc11d443982d9f223540f8e9dc10",
    ("src/fleet_fault_triage/cloud.py", "score_delta"): "6e444e89be2f73cbbdd8f3c0901be1cddb88a621",
}


def public_files(root):
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
    )
    return sorted({Path(p.decode()) for p in output.split(b"\0") if p})


def content_errors(text):
    errors = [label for label, pattern in SECRET_PATTERNS.items() if pattern.search(text)]
    if any(
        not value.startswith(("SYNTHETIC_", "EXAMPLE_", "PLACEHOLDER_", "REPLACE_", "YOUR_"))
        for value in CREDENTIAL_LITERAL.findall(text)
    ):
        errors.append("credential literal")
    # Repeated-digit GUIDs and reserved example domains are explicit synthetic identities.
    if any(
        len(set(value.replace("-", "").lower())) != 1 and value.lower() not in PUBLIC_PROVIDER_IDS
        for value in GUID.findall(text)
    ):
        errors.append("non-placeholder identity GUID")
    reserved = ("example.com", "example.org", "example.net", ".invalid", ".test")
    if any(
        not (domain in reserved[:3] or domain.endswith(reserved[3:]))
        for domain in EMAIL.findall(text)
    ):
        errors.append("non-example email address")
    return errors


def link_errors(root, relative, text):
    errors = []
    # Do not interpret example links inside fenced code as document navigation.
    prose = re.sub(r"^\s*(`{3,}|~{3,}).*?^\s*\1\s*$", "", text, flags=re.M | re.S)
    links = [m.group(1).strip("<>") for m in MARKDOWN_LINK.finditer(prose)]
    links.extend(HTML_LINK.findall(prose))
    for link in links:
        parsed = urlsplit(link)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        destination = (root / relative.parent / unquote(parsed.path)).resolve()
        if not destination.is_relative_to(root.resolve()):
            errors.append("local link leaves repository")
            continue
        resolved = destination.relative_to(root.resolve())
        if resolved.parts and resolved.parts[0] in PRIVATE_ROOTS:
            errors.append(f"local link points to excluded material: {resolved}")
        elif not destination.exists():
            errors.append(f"missing local link: {resolved}")
    return errors


def provenance_errors(root):
    manifest = root / "evidence/source-provenance.json"
    if not manifest.exists():
        return ["missing evidence/source-provenance.json"]
    try:
        document = json.loads(manifest.read_text())
        entries = document["files"]
        if not isinstance(entries, list) or not entries:
            return ["source provenance files must be a nonempty list"]
        paths = [Path(entry["path"]) for entry in entries]
        if any(not relative.parts for relative in paths):
            return ["source provenance paths must name files"]
        if len(paths) != len(set(paths)):
            return ["duplicate source provenance path"]
        declared_runtime = {
            relative
            for relative in paths
            if relative.parts[0] == "src" and relative.suffix == ".py"
        }
        published_runtime = {
            relative
            for relative in public_files(root)
            if relative.parts[0] == "src" and relative.suffix == ".py"
        }
        errors = []
        errors.extend(
            f"unmanifested published runtime source: {relative}"
            for relative in sorted(published_runtime - declared_runtime)
        )
        errors.extend(
            f"manifest runtime source is not published: {relative}"
            for relative in sorted(declared_runtime - published_runtime)
        )
        for entry in entries:
            relative = Path(entry["path"])
            destination = (root / relative).resolve()
            if relative.is_absolute() or not destination.is_relative_to(root.resolve()):
                errors.append("source provenance path leaves repository")
            elif not destination.is_file():
                errors.append(f"source provenance file missing: {relative}")
            elif hashlib.sha256(destination.read_bytes()).hexdigest() != entry["sha256"]:
                errors.append(f"source provenance hash mismatch: {relative}")
            if (
                type(entry["byte_identical"]) is not bool
                or entry["byte_identical"] != (entry["source_sha256"] == entry["sha256"])
                or (relative.parts[0] == "src" and entry["byte_identical"] is not True)
            ):
                errors.append(f"inconsistent source byte-identity declaration: {relative}")
        runtime = [entry for entry in entries if Path(entry["path"]) in declared_runtime]
        expected = {
            "copied_files": len(entries),
            "byte_identical_files": sum(entry["byte_identical"] is True for entry in entries),
            "runtime_source_files": len(runtime),
            "runtime_source_files_byte_identical": all(
                entry["byte_identical"] for entry in runtime
            ),
        }
        for name, value in expected.items():
            if type(document[name]) is not type(value) or document[name] != value:
                errors.append(f"inconsistent source provenance summary: {name}")
        return errors
    except (KeyError, TypeError, ValueError, OSError) as error:
        return [f"invalid source provenance: {type(error).__name__}"]


def execution_function_errors(root):
    """Recompute public function syntax; historical Git sources remain withheld."""
    if sys.version_info[:2] != (3, 12):
        return ["execution function AST validation requires Python 3.12"]
    try:
        document = json.loads((root / "evidence/source-provenance.json").read_text())
        if document.get("ast_canonicalization") != AST_CANONICALIZATION:
            return ["execution function AST canonicalization must declare the Python 3.12 format"]
        entries = document["historical_execution_function_comparisons"]
        if not isinstance(entries, list) or len(entries) != len(EXECUTION_FUNCTIONS):
            return ["execution function provenance must contain the two expected comparisons"]
        keys = [(entry["public_path"], entry["function"]) for entry in entries]
        if len(keys) != len(set(keys)):
            return ["duplicate execution function comparison"]
        if set(keys) != set(EXECUTION_FUNCTIONS):
            return [
                "unexpected execution function comparison; expected cloud.py train_registered and score_delta"
            ]
        errors = []
        for entry, key in zip(entries, keys, strict=True):
            relative, name = key
            label = f"{relative}:{name}"
            if entry["historical_execution_revision"] != EXECUTION_FUNCTIONS[key]:
                errors.append(f"unexpected historical execution revision: {label}")
                continue
            if entry["function_ast_identical"] is not True:
                errors.append(f"function AST identity flag must be literal true: {label}")
                continue
            public_hash = entry["public_function_ast_sha256"]
            historical_hash = entry["historical_function_ast_sha256"]
            if any(
                type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in (public_hash, historical_hash)
            ):
                errors.append(
                    f"function AST hashes must be 64 lowercase hexadecimal characters: {label}"
                )
                continue
            functions = [
                node
                for node in ast.parse((root / relative).read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name
            ]
            if len(functions) != 1:
                errors.append(f"public function missing or ambiguous: {label}")
                continue
            actual = hashlib.sha256(
                ast.dump(functions[0], include_attributes=False).encode("utf-8")
            ).hexdigest()
            if public_hash != actual:
                errors.append(f"public function AST hash mismatch: {label}")
            elif historical_hash != actual:
                errors.append(
                    f"declared historical function AST hash differs from public function: {label}"
                )
        return errors
    except (KeyError, TypeError, ValueError, OSError, SyntaxError):
        return ["invalid execution function provenance"]


def benchmark_errors(document):
    try:
        evaluation = document["evaluation"]
        count_names = ("input_rows", "scorable_rows")
        if any(type(evaluation[name]) is not int or evaluation[name] < 0 for name in count_names):
            return ["benchmark row counts must be nonnegative integers"]
        counts = evaluation["confusion_matrix"]
        if set(counts) != {"tp", "tn", "fp", "fn"} or any(
            type(value) is not int or value < 0 for value in counts.values()
        ):
            return ["benchmark confusion counts must be four nonnegative integers"]
        inputs, scored = (evaluation[name] for name in count_names)
        if inputs == 0 or scored > inputs or sum(counts.values()) != scored:
            return ["benchmark confusion counts and scored/input rows disagree"]
        unscored = document["unscored_rows_derived"]
        if type(unscored) is not int or unscored != inputs - scored:
            return ["benchmark unscored rows do not equal input minus scored"]
        denominators = {
            "recall": counts["tp"] + counts["fn"],
            "precision": counts["tp"] + counts["fp"],
        }
        if any(value == 0 for value in denominators.values()):
            return ["benchmark recall or precision denominator is zero"]
        expected = {name: counts["tp"] / total for name, total in denominators.items()}
        expected["coverage"] = scored / inputs
        for name, value in expected.items():
            observed = evaluation[name]
            if (
                type(observed) not in (int, float)
                or not math.isfinite(observed)
                or not math.isclose(observed, value, rel_tol=0, abs_tol=1e-12)
            ):
                return [f"benchmark {name} disagrees with aggregate counts"]
        # AP requires prediction scores; only its recorded numeric range can be checked here.
        ap = evaluation["average_precision"]
        if type(ap) not in (int, float) or not math.isfinite(ap) or not 0 <= ap <= 1:
            return ["benchmark recorded average precision must be finite and in [0,1]"]
        return []
    except (KeyError, TypeError, ValueError):
        return ["benchmark evidence has an invalid shape"]


def _local_schema_refs(value):
    if isinstance(value, dict):
        if "$ref" in value and (
            not isinstance(value["$ref"], str) or not value["$ref"].startswith("#/")
        ):
            return False
        return all(_local_schema_refs(item) for item in value.values())
    if isinstance(value, list):
        return all(_local_schema_refs(item) for item in value)
    return True


def _schema_errors(document, schema, contract, label):
    schema = {"$schema": contract["$schema"], "$defs": contract["$defs"], **schema}
    problems = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda error: "/".join(map(str, error.absolute_path)),
    )
    return [
        f"{label}: semantic schema {error.validator} at /{'/'.join(map(str, error.absolute_path))}"
        for error in problems
    ]


def _source_refs(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "source_ref":
                yield item
            elif key == "source_refs":
                yield from item
            else:
                yield from _source_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _source_refs(item)


def _evidence_reference_errors(root, manifest, documents):
    sources = manifest["sources"]
    if len({row["source_id"] for row in sources}) != len(sources):
        return ["duplicate provenance source_id"]
    catalog = {row["source_id"]: row for row in sources}
    errors = []
    refs = set(_source_refs(documents)) | set(_source_refs(manifest))
    for link in manifest["source_integrity_links_verified_during_packaging"]:
        refs.update((link["source_id"], link["referenced_by"]))
    for output in manifest["outputs"]:
        projected = set(output["source_field_projection"])
        refs.update(projected)
        if not projected <= set(documents[output["path"]]["source_refs"]):
            errors.append(f"source-field projection is not declared by extract: {output['path']}")
    errors.extend(
        f"unknown provenance source reference: {ref}" for ref in sorted(refs - catalog.keys())
    )
    source_manifest = json.loads((root / "evidence/source-provenance.json").read_text())
    for source in sources:
        identifier = source["source_id"]
        expected_class = (
            "shipped_source_code_scope_interpretation"
            if identifier == "C1"
            else "coordinator_saved_state_verification"
            if identifier == "V1"
            else "preserved_databricks_sdk_response"
            if identifier.startswith("N")
            else "coordinator_authored_summary"
        )
        if source["evidence_class"] != expected_class:
            errors.append(f"unexpected provenance source evidence class: {identifier}")
        public = identifier == "C1"
        if source["original_bytes_in_public_package"] is not public:
            errors.append(f"unsupported provenance source publication state: {identifier}")
        if not public:
            if "public_path" in source:
                errors.append(f"withheld source declares a public file: {identifier}")
            if identifier.startswith("S"):
                if source["original_repository_commit"] != manifest[
                    "original_source_snapshot_commit"
                ] or not source.get("original_repository_path"):
                    errors.append(
                        f"withheld coordinator source metadata is incomplete: {identifier}"
                    )
            elif (
                source["original_repository_commit"] is not None
                or source.get("private_capture_path") != "withheld"
                or not source.get("original_source_basename")
                or "/" in source["original_source_basename"]
            ):
                errors.append(f"withheld native source metadata is incomplete: {identifier}")
            # Private original paths are metadata, not locally resolvable public links.
            continue
        relative = Path(source.get("public_path", ""))
        destination = (root / relative).resolve()
        if (
            not relative.parts
            or relative.is_absolute()
            or not destination.is_relative_to(root.resolve())
        ):
            errors.append(f"public provenance source path leaves repository: {identifier}")
            continue
        if not destination.is_file():
            errors.append(f"public provenance file missing: {relative}")
            continue
        if hashlib.sha256(destination.read_bytes()).hexdigest() != source["original_raw_sha256"]:
            errors.append(f"public provenance source hash mismatch: {identifier}")
        matches = [row for row in source_manifest["files"] if row["path"] == str(relative)]
        if (
            source.get("verified_against_public_source_manifest")
            != "evidence/source-provenance.json"
            or len(matches) != 1
            or matches[0]["source_sha256"] != source["original_raw_sha256"]
            or matches[0]["source_commit"] != source["original_repository_commit"]
        ):
            errors.append(f"public source differs from declared source manifest: {identifier}")
        functions = [
            node
            for node in ast.parse(destination.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == source.get("function")
        ]
        if len(functions) != 1 or not all(
            functions[0].lineno <= line <= functions[0].end_lineno
            for line in source.get("scope_lines", [])
        ):
            errors.append(f"public source function or scope lines are invalid: {identifier}")
    for derivation in manifest["scope_derivations"]:
        source = catalog.get(derivation["source_ref"], {})
        if (
            source.get("public_path") != derivation["public_source_path"]
            or source.get("scope_lines") != derivation["source_lines"]
        ):
            errors.append("scope derivation does not match its declared public source")
        for pointer in derivation["output_fields"]:
            value = documents[derivation["output_path"]]
            try:
                if not pointer.startswith("/"):
                    raise KeyError(pointer)
                for key in pointer[1:].split("/"):
                    value = value[key.replace("~1", "/").replace("~0", "~")]
            except (KeyError, TypeError):
                errors.append(f"scope derivation output field is missing: {pointer}")
    return errors


def _evidence_relationship_errors(documents):
    benchmark = documents["evidence/classifier-benchmark.json"]
    delivery = documents["evidence/databricks-delivery.json"]
    scoring = documents["evidence/delta-scoring.json"]
    assistant = documents["evidence/assistant-status.json"]
    preservation = documents["evidence/preservation.json"]
    errors = benchmark_errors(benchmark)
    training = delivery["training_summary"]
    difference, tolerance = training["parity_max_absolute_difference"], training["parity_tolerance"]
    if not math.isfinite(difference) or not 0 <= difference <= tolerance:
        errors.append("Databricks parity difference is nonfinite or exceeds tolerance")
    if training["local_fits_including_failures"] + training["cloud_fits"] != training["total_fits"]:
        errors.append("Databricks fit totals are inconsistent")
    summary = scoring["summary"]
    if (
        summary["scored_rows"] + summary["quality_abstentions"] != summary["case_rows"]
        or summary["local_cases_saved_and_read_back"] != summary["case_rows"]
    ):
        errors.append("Delta scoring counts or persisted readback are inconsistent")
    table = scoring["native_response_projections"]["scoring_table"]
    if table["column_count_derived"] != len(table["schema"]):
        errors.append("Delta scoring table column count differs from its schema")
    workload = benchmark["workload_commit"]
    if (
        delivery["training_workload_commit"] != workload
        or summary["model_source_commit"] != workload
    ):
        errors.append("classifier and Databricks/scoring workload identities differ")
    release = benchmark["scoring_release_sha256"]
    if any(
        value != release
        for value in (
            summary["release_sha256"],
            preservation["scoring_release_sha256"],
            assistant["remaining_acceptance"]["classifier_sha256_unchanged"],
        )
    ):
        errors.append("evidence extracts disagree on the frozen classifier identity")
    evaluation = benchmark["evaluation"]
    if benchmark["overlap_scope"]["rows_checked"] != evaluation["scorable_rows"]:
        errors.append("benchmark overlap population differs from scored rows")
    for metric in ("recall", "precision"):
        lower, upper = evaluation[f"{metric}_interval_95"]
        if not all(math.isfinite(value) for value in (lower, upper)) or not (
            0 <= lower <= evaluation[metric] <= upper <= 1
        ):
            errors.append(f"benchmark {metric} interval is inconsistent")
    inventory = assistant["review_inventory_summary"]
    for reviewer, modes in assistant["independent_ai_reviews"].items():
        for method, review in modes.items():
            if (
                review["source_supported_facts"] != review["all_facts"]
                or review["supported_cited_facts"] != review["all_cited_facts"]
            ):
                errors.append(f"assistant support inventory differs: {reviewer}/{method}")
            if not math.isclose(
                100 * review["completed_answerable"] / review["answerable"],
                inventory["completion_percent_per_method_in_each_inventory"],
            ):
                errors.append(f"assistant completion rate differs: {reviewer}/{method}")
    exposure = assistant["control_exposure"]
    returned = sum(
        exposure[name] for name in ("model_response", "application_fallback", "provider_block")
    )
    if any(count != returned for count in assistant["saved_outcomes"].values()):
        errors.append("assistant outcome counts differ from response-origin inventory")
    return errors


def evidence_semantic_errors(root, manifest, documents):
    contract = json.loads((Path(__file__).parent / "evidence-contract.json").read_text())
    if not _local_schema_refs(contract):
        return ["evidence contract must use local schema references only"]
    errors = _schema_errors(manifest, contract["provenance"], contract, "evidence/provenance.json")
    for relative, document in documents.items():
        errors.extend(_schema_errors(document, contract["extracts"][relative], contract, relative))
    if errors:
        return errors
    errors.extend(_evidence_reference_errors(root, manifest, documents))
    errors.extend(_evidence_relationship_errors(documents))
    return errors


def evidence_errors(root):
    manifest = root / "evidence/provenance.json"
    if not manifest.exists():
        return ["missing evidence/provenance.json"]
    try:
        document = json.loads(manifest.read_text())
        entries = document["outputs"]
        if not isinstance(entries, list) or len(entries) != len(EVIDENCE_EXTRACTS):
            return ["evidence provenance must list the five current extracts"]
        if {entry["path"] for entry in entries} != EVIDENCE_EXTRACTS:
            return ["evidence provenance extract paths differ from the five current extracts"]
        errors = []
        documents = {}
        for entry in entries:
            file = root / entry["path"]
            if not file.is_file():
                errors.append(f"evidence extract missing: {entry['path']}")
            elif hashlib.sha256(file.read_bytes()).hexdigest() != entry["sha256"]:
                errors.append(f"evidence extract hash mismatch: {entry['path']}")
            if file.is_file():
                documents[entry["path"]] = json.loads(file.read_text())
        if len(documents) == len(EVIDENCE_EXTRACTS):
            errors.extend(evidence_semantic_errors(root, document, documents))
        return errors
    except (KeyError, TypeError, ValueError, OSError):
        return ["invalid evidence provenance"]


def validate(root=ROOT):
    errors = []
    files = public_files(root)
    for relative in files:
        first = relative.parts[0]
        if (
            first in PRIVATE_ROOTS
            or relative.name == "SHOWCASE-BRIEF.md"
            or relative.name.startswith(".env")
            or relative.suffix in PRIVATE_SUFFIXES
            or ".sqlite" in relative.name
            or relative.suffix in {".db", ".pem", ".key"}
        ):
            errors.append(f"excluded publication file: {relative}")
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"publication file missing: {relative}")
            continue
        if relative.suffix in TEXT_SUFFIXES or relative.name in {"Makefile", ".gitignore"}:
            text = path.read_text()
            errors.extend(f"{relative}: {error}" for error in content_errors(text))
            if relative.suffix == ".json":
                try:
                    json.loads(text)
                except ValueError:
                    errors.append(f"{relative}: invalid JSON")
            if relative.suffix == ".md":
                errors.extend(f"{relative}: {error}" for error in link_errors(root, relative, text))
    errors.extend(provenance_errors(root))
    errors.extend(execution_function_errors(root))
    errors.extend(evidence_errors(root))
    return files, errors


def main():
    files, errors = validate()
    print(
        json.dumps(
            {"public_files_checked": len(files), "errors": errors, "passed": not errors}, indent=2
        )
    )
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
