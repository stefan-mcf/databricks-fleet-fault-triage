import importlib.util
import json
import subprocess
import uuid
from copy import deepcopy
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "tools/verify_publication.py"
SPEC = importlib.util.spec_from_file_location("publication", MODULE)
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


def test_publication_rejects_nonplaceholder_identity_and_private_links():
    assert publication.content_errors(str(uuid.UUID(int=123456789123456789)))
    assert publication.content_errors("https://" + "drive.google.com/file/d/private-id/view")
    assert publication.content_errors("https://blob.example.net/data?sig=" + "a" * 32)


def test_publication_allows_hashes_and_explicit_synthetic_identities():
    assert not publication.content_errors("a" * 64)
    assert not publication.content_errors("11111111-1111-1111-1111-111111111111")
    assert not publication.content_errors("reviewer@example.invalid")


def test_publication_rejects_a_credential_literal_but_allows_marked_placeholders():
    assert publication.content_errors('api_key="' + "abc123" * 6 + '"')
    assert not publication.content_errors('api_key="SYNTHETIC_PLACEHOLDER_NOT_A_CREDENTIAL"')


def test_publication_links_cannot_escape_or_reference_private_outputs(tmp_path):
    errors = publication.link_errors(tmp_path, Path("README.md"), "[bad](../outside.md)")
    assert errors == ["local link leaves repository"]
    errors = publication.link_errors(tmp_path, Path("README.md"), "[bad](outputs/result.json)")
    assert errors == ["local link points to excluded material: outputs/result.json"]


def test_publication_requires_local_links_but_skips_fenced_examples(tmp_path):
    errors = publication.link_errors(tmp_path, Path("README.md"), "[missing](docs/missing.md)")
    assert errors == ["missing local link: docs/missing.md"]
    assert not publication.link_errors(
        tmp_path, Path("README.md"), "```md\n[example](not-real.md)\n```\n"
    )


def test_publication_provenance_is_required_and_binds_file_bytes(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    assert publication.provenance_errors(tmp_path) == ["missing evidence/source-provenance.json"]
    destination = tmp_path / "evidence/source-provenance.json"
    destination.parent.mkdir()
    entry = {"path": "code.py", "sha256": "wrong", "source_sha256": "wrong", "byte_identical": True}
    document = {
        "files": [entry],
        "copied_files": 1,
        "byte_identical_files": 1,
        "runtime_source_files": 0,
        "runtime_source_files_byte_identical": True,
    }
    destination.write_text(json.dumps(document))
    assert publication.provenance_errors(tmp_path) == ["source provenance file missing: code.py"]
    (tmp_path / "code.py").write_text("original")
    assert publication.provenance_errors(tmp_path) == ["source provenance hash mismatch: code.py"]
    destination.write_text(json.dumps({**document, "files": [{**entry, "path": "../escape"}]}))
    assert publication.provenance_errors(tmp_path) == ["source provenance path leaves repository"]


def runtime_provenance_fixture(root):
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    source = root / "src/package/core.py"
    source.parent.mkdir(parents=True)
    source.write_text("SYNTHETIC_SOURCE = True\n")
    digest = publication.hashlib.sha256(source.read_bytes()).hexdigest()
    document = {
        "files": [
            {
                "path": "src/package/core.py",
                "sha256": digest,
                "source_sha256": digest,
                "byte_identical": True,
            }
        ],
        "copied_files": 1,
        "byte_identical_files": 1,
        "runtime_source_files": 1,
        "runtime_source_files_byte_identical": True,
    }
    manifest = root / "evidence/source-provenance.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps(document))
    assert not publication.provenance_errors(root)
    return manifest, document


def test_source_provenance_rejects_an_added_unlisted_runtime_module(tmp_path):
    runtime_provenance_fixture(tmp_path)
    (tmp_path / "src/package/extra.py").write_text("SYNTHETIC_EXTRA = True\n")
    assert publication.provenance_errors(tmp_path) == [
        "unmanifested published runtime source: src/package/extra.py"
    ]


def test_source_provenance_rejects_duplicate_manifest_entries(tmp_path):
    manifest, document = runtime_provenance_fixture(tmp_path)
    document["files"].append(dict(document["files"][0]))
    manifest.write_text(json.dumps(document))
    assert publication.provenance_errors(tmp_path) == ["duplicate source provenance path"]


def execution_function_fixture(root):
    source = root / "src/fleet_fault_triage/cloud.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        'def train_registered():\n    return "SYNTHETIC_TRAIN"\n\n'
        'def score_delta():\n    return "SYNTHETIC_SCORE"\n'
    )
    functions = {node.name: node for node in publication.ast.parse(source.read_text()).body}
    comparisons = []
    for (relative, name), revision in publication.EXECUTION_FUNCTIONS.items():
        digest = publication.hashlib.sha256(
            publication.ast.dump(functions[name], include_attributes=False).encode("utf-8")
        ).hexdigest()
        comparisons.append(
            {
                "public_path": relative,
                "function": name,
                "historical_execution_revision": revision,
                "historical_function_ast_sha256": digest,
                "public_function_ast_sha256": digest,
                "function_ast_identical": True,
            }
        )
    document = {
        "ast_canonicalization": dict(publication.AST_CANONICALIZATION),
        "historical_execution_function_comparisons": comparisons,
    }
    manifest = root / "evidence/source-provenance.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps(document))
    assert not publication.execution_function_errors(root)
    return manifest, document, source


def test_execution_function_comparisons_use_recomputed_public_ast(tmp_path):
    execution_function_fixture(tmp_path)
    assert not publication.execution_function_errors(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("public_function_ast_sha256", "0" * 64, "public function AST hash mismatch"),
        (
            "historical_function_ast_sha256",
            "0" * 64,
            "declared historical function AST hash differs from public function",
        ),
        ("function_ast_identical", False, "function AST identity flag must be literal true"),
        ("function_ast_identical", "true", "function AST identity flag must be literal true"),
    ],
)
def test_execution_function_comparisons_reject_tampered_hashes_and_identity_flags(
    tmp_path, field, value, message
):
    manifest, document, _ = execution_function_fixture(tmp_path)
    document["historical_execution_function_comparisons"][0][field] = value
    manifest.write_text(json.dumps(document))
    assert publication.execution_function_errors(tmp_path) == [
        f"{message}: src/fleet_fault_triage/cloud.py:train_registered"
    ]


def test_execution_function_comparisons_reject_an_unknown_function(tmp_path):
    manifest, document, _ = execution_function_fixture(tmp_path)
    document["historical_execution_function_comparisons"][0]["function"] = "unknown"
    manifest.write_text(json.dumps(document))
    assert publication.execution_function_errors(tmp_path) == [
        "unexpected execution function comparison; expected cloud.py train_registered and score_delta"
    ]


def test_execution_function_comparisons_reject_changed_public_function_syntax(tmp_path):
    _, _, source = execution_function_fixture(tmp_path)
    source.write_text(source.read_text().replace("SYNTHETIC_TRAIN", "SYNTHETIC_CHANGED"))
    assert publication.execution_function_errors(tmp_path) == [
        "public function AST hash mismatch: src/fleet_fault_triage/cloud.py:train_registered"
    ]


def test_execution_function_comparisons_require_unique_pairs_and_current_canonicalization(tmp_path):
    manifest, document, _ = execution_function_fixture(tmp_path)
    document["historical_execution_function_comparisons"][1] = dict(
        document["historical_execution_function_comparisons"][0]
    )
    manifest.write_text(json.dumps(document))
    assert publication.execution_function_errors(tmp_path) == [
        "duplicate execution function comparison"
    ]
    document["ast_canonicalization"]["python_major_minor"] = "3.14"
    manifest.write_text(json.dumps(document))
    assert publication.execution_function_errors(tmp_path) == [
        "execution function AST canonicalization must declare the Python 3.12 format"
    ]


def benchmark_fixture():
    return {
        "evaluation": {
            "input_rows": 10,
            "scorable_rows": 8,
            "confusion_matrix": {"tp": 2, "fn": 1, "fp": 1, "tn": 4},
            "recall": 2 / 3,
            "precision": 2 / 3,
            "coverage": 0.8,
            "average_precision": 0.75,
        },
        "unscored_rows_derived": 2,
    }


def test_benchmark_aggregate_checks_reject_tampered_metrics_counts_and_types():
    original = benchmark_fixture()
    assert not publication.benchmark_errors(original)
    for field, value in (("recall", 1), ("precision", 1), ("coverage", 1), ("input_rows", True)):
        changed = deepcopy(original)
        changed["evaluation"][field] = value
        assert publication.benchmark_errors(changed)
    changed = deepcopy(original)
    changed["evaluation"]["confusion_matrix"]["tn"] += 1
    assert publication.benchmark_errors(changed)
    changed = deepcopy(original)
    changed["unscored_rows_derived"] = 0
    assert publication.benchmark_errors(changed)


def public_evidence_fixture(root):
    public_root = MODULE.parents[1]
    relatives = publication.EVIDENCE_EXTRACTS | {
        "evidence/provenance.json",
        "evidence/source-provenance.json",
    }
    manifest = json.loads((public_root / "evidence/provenance.json").read_text())
    relatives |= {
        source["public_path"]
        for source in manifest["sources"]
        if source["original_bytes_in_public_package"]
    }
    for relative in sorted(relatives):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((public_root / relative).read_bytes())
    assert not publication.evidence_errors(root)
    return manifest


def rehash_public_extract(root, relative, document):
    destination = root / relative
    destination.write_text(json.dumps(document, indent=2))
    path = root / "evidence/provenance.json"
    manifest = json.loads(path.read_text())
    entry = next(row for row in manifest["outputs"] if row["path"] == relative)
    entry["sha256"] = publication.hashlib.sha256(destination.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))


def update_evidence_field(root, relative, keys, value):
    document = json.loads((root / relative).read_text())
    target = document
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    rehash_public_extract(root, relative, document)


def assert_semantic_rejection(root, fragment="semantic schema"):
    errors = publication.evidence_errors(root)
    assert errors
    assert not any("evidence extract hash mismatch" in error for error in errors)
    assert any(fragment in error for error in errors), errors


def test_evidence_manifest_requires_five_exact_extracts_and_binds_saved_bytes(tmp_path):
    assert publication.evidence_errors(tmp_path) == ["missing evidence/provenance.json"]
    manifest = public_evidence_fixture(tmp_path)
    (tmp_path / "evidence/delta-scoring.json").write_text('{"tampered":true}')
    assert (
        "evidence extract hash mismatch: evidence/delta-scoring.json"
        in publication.evidence_errors(tmp_path)
    )
    manifest["outputs"] = manifest["outputs"][:-1]
    (tmp_path / "evidence/provenance.json").write_text(json.dumps(manifest))
    assert publication.evidence_errors(tmp_path) == [
        "evidence provenance must list the five current extracts"
    ]


@pytest.mark.parametrize("relative", sorted(publication.EVIDENCE_EXTRACTS))
def test_every_extract_rejects_an_empty_object_even_with_a_new_matching_digest(tmp_path, relative):
    public_evidence_fixture(tmp_path)
    rehash_public_extract(tmp_path, relative, {})
    assert_semantic_rejection(tmp_path)


@pytest.mark.parametrize(
    ("relative", "keys"),
    [
        ("evidence/classifier-benchmark.json", ("evaluation", "coverage")),
        ("evidence/databricks-delivery.json", ("training_summary", "parity_rows")),
        ("evidence/delta-scoring.json", ("summary", "native_job_success")),
        (
            "evidence/assistant-status.json",
            ("remaining_acceptance", "human_disposition_and_output_acceptance"),
        ),
        (
            "evidence/preservation.json",
            ("native_export_summary", "complete_inventory_and_each_saved_member_hash_verified"),
        ),
    ],
)
def test_every_extract_requires_its_critical_fields_after_digest_update(tmp_path, relative, keys):
    public_evidence_fixture(tmp_path)
    document = json.loads((tmp_path / relative).read_text())
    target = document
    for key in keys[:-1]:
        target = target[key]
    target.pop(keys[-1])
    rehash_public_extract(tmp_path, relative, document)
    assert_semantic_rejection(tmp_path, "semantic schema required")


@pytest.mark.parametrize(
    ("relative", "keys", "value"),
    [
        ("evidence/classifier-benchmark.json", ("evaluation", "input_rows"), "16000"),
        ("evidence/databricks-delivery.json", ("training_summary", "parity_rows"), True),
        ("evidence/delta-scoring.json", ("summary", "native_job_success"), 1),
        ("evidence/assistant-status.json", ("saved_outcomes", "keyword"), "40"),
        ("evidence/preservation.json", ("native_export_summary", "bytes"), "3250122"),
    ],
)
def test_every_extract_rejects_wrong_typed_critical_facts_after_digest_update(
    tmp_path, relative, keys, value
):
    public_evidence_fixture(tmp_path)
    update_evidence_field(tmp_path, relative, keys, value)
    assert_semantic_rejection(tmp_path, "semantic schema type")


@pytest.mark.parametrize(
    ("relative", "keys", "value"),
    [
        ("evidence/classifier-benchmark.json", ("evaluation", "confusion_matrix", "tp"), 347),
        ("evidence/classifier-benchmark.json", ("overlap_scope", "rows_checked"), 16000),
        ("evidence/databricks-delivery.json", ("training_summary", "total_fits"), 85),
        ("evidence/databricks-delivery.json", ("model_signature_summary", "feature_count"), 169),
        (
            "evidence/databricks-delivery.json",
            ("native_response_projections", "training_job", "state", "result_state"),
            "FAILED",
        ),
        (
            "evidence/databricks-delivery.json",
            ("native_response_projections", "unity_catalog_model_version", "version"),
            2,
        ),
        (
            "evidence/databricks-delivery.json",
            ("delta_lineage_summary", "explicit_version_as_of_pinning_verified"),
            True,
        ),
        ("evidence/databricks-delivery.json", ("governance_summary", "aliases_created"), True),
        ("evidence/delta-scoring.json", ("summary", "scored_rows"), 120),
        ("evidence/delta-scoring.json", ("summary", "new_fits"), 1),
        ("evidence/delta-scoring.json", ("summary", "model_alias_or_promotion_changed"), True),
        ("evidence/delta-scoring.json", ("summary", "delta_commit_version"), 0),
        (
            "evidence/delta-scoring.json",
            ("native_response_projections", "scoring_table", "column_count_derived"),
            4,
        ),
        (
            "evidence/assistant-status.json",
            ("independent_ai_reviews", "A", "keyword", "critical_required_response_failures"),
            0,
        ),
        (
            "evidence/assistant-status.json",
            ("review_inventory_summary", "completion_percent_per_method_in_each_inventory"),
            100,
        ),
        (
            "evidence/assistant-status.json",
            ("review_inventory_summary", "methods_passed"),
            ["hybrid"],
        ),
        (
            "evidence/assistant-status.json",
            ("remaining_acceptance", "current_candidate_SEALED_eligible"),
            True,
        ),
        (
            "evidence/assistant-status.json",
            ("remaining_acceptance", "human_disposition_and_output_acceptance"),
            True,
        ),
        ("evidence/assistant-status.json", ("remaining_acceptance", "G3_or_final_G4_pass"), True),
        ("evidence/preservation.json", ("native_export_summary", "files"), 13),
        ("evidence/preservation.json", ("native_export_summary", "new_fits_or_inference"), 1),
        ("evidence/preservation.json", ("private_backup_summary", "shared"), True),
        (
            "evidence/preservation.json",
            ("private_backup_summary", "public_redistribution_authorised"),
            True,
        ),
    ],
)
def test_frozen_counts_states_and_nonacceptance_cannot_be_laundered_with_a_new_digest(
    tmp_path, relative, keys, value
):
    public_evidence_fixture(tmp_path)
    update_evidence_field(tmp_path, relative, keys, value)
    assert_semantic_rejection(tmp_path)


@pytest.mark.parametrize("difference", [0.0, 5e-7, 2e-6, float("nan"), float("inf")])
def test_recorded_parity_cannot_change_even_within_tolerance(tmp_path, difference):
    public_evidence_fixture(tmp_path)
    update_evidence_field(
        tmp_path,
        "evidence/databricks-delivery.json",
        ("training_summary", "parity_max_absolute_difference"),
        difference,
    )
    assert_semantic_rejection(tmp_path, "semantic schema const")


@pytest.mark.parametrize(
    ("relative", "keys", "value"),
    [
        (
            "evidence/databricks-delivery.json",
            ("native_response_projections", "training_job", "source_ref"),
            "N6",
        ),
        (
            "evidence/delta-scoring.json",
            ("native_response_projections", "scoring_table", "source_ref"),
            "N1",
        ),
        ("evidence/assistant-status.json", ("source_refs",), ["S1"]),
        ("evidence/preservation.json", ("limits",), []),
        ("evidence/preservation.json", ("limits",), [" "] * 4),
        ("evidence/classifier-benchmark.json", ("interpretation",), []),
    ],
)
def test_extract_provenance_labels_and_limitations_are_required_after_digest_update(
    tmp_path, relative, keys, value
):
    public_evidence_fixture(tmp_path)
    update_evidence_field(tmp_path, relative, keys, value)
    assert_semantic_rejection(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    [
        ("dangling", "unknown provenance source reference: N1"),
        ("duplicate", "duplicate provenance source_id"),
        ("hollow", "semantic schema required"),
        ("wrong_hash_type", "semantic schema type"),
        ("wrong_publication_type", "semantic schema type"),
        ("wrong_evidence_class", "unexpected provenance source evidence class: N1"),
        ("hash_newline", "semantic schema maxLength"),
        ("id_newline", "semantic schema enum"),
    ],
)
def test_provenance_source_graph_rejects_missing_duplicate_and_hollow_sources(
    tmp_path, mutation, fragment
):
    manifest = public_evidence_fixture(tmp_path)
    row = next(row for row in manifest["sources"] if row["source_id"] == "N1")
    if mutation == "dangling":
        manifest["sources"].remove(row)
    elif mutation == "duplicate":
        manifest["sources"].append(dict(row))
    elif mutation == "hollow":
        row.clear()
        row["source_id"] = "N1"
    elif mutation == "wrong_hash_type":
        row["original_raw_sha256"] = 0
    elif mutation == "wrong_publication_type":
        row["original_bytes_in_public_package"] = "false"
    elif mutation == "hash_newline":
        row["original_raw_sha256"] += "\n"
    elif mutation == "id_newline":
        row["source_id"] += "\n"
    else:
        row["evidence_class"] = "coordinator_authored_summary"
    (tmp_path / "evidence/provenance.json").write_text(json.dumps(manifest))
    assert_semantic_rejection(tmp_path, fragment)


def test_provenance_links_and_shipped_source_files_must_exist_and_match(tmp_path):
    public_evidence_fixture(tmp_path)
    source = tmp_path / "src/fleet_fault_triage/evaluation.py"
    source.write_text(source.read_text() + "\n# SYNTHETIC_TAMPER\n")
    assert_semantic_rejection(tmp_path, "public provenance source hash mismatch: C1")
    source.unlink()
    assert_semantic_rejection(tmp_path, "public provenance file missing")


def test_evidence_schemas_cannot_request_remote_or_invalid_reference_retrieval():
    assert publication._local_schema_refs({"items": [{"$ref": "#/$defs/example"}]})
    assert not publication._local_schema_refs(
        {"items": [{"$ref": "https://example.invalid/schema"}]}
    )
    assert not publication._local_schema_refs({"$ref": None})
