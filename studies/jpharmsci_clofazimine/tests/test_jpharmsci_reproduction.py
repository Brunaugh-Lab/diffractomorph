from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import manuscript_arm_b_ld_response as arm_b


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "reproduce.py"
SPEC = importlib.util.spec_from_file_location("reproduce_jpharmsci", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_reproduction_manifest_is_restricted_and_path_safe():
    manifest = MODULE.load_manifest(ROOT / "manifest.json")
    assert manifest["status"] == "restricted-local-draft"
    assert manifest["publication_boundary"]["redistributable"] is False
    assert {item["class"] for item in manifest["reference_artifacts"]} >= {
        "pipeline-generated", "writing-repo-assembly"
    }
    for recipe in manifest["recipes"]:
        assert not Path(recipe["script"]).is_absolute()
        assert recipe["expected_outputs"]
    for artifact in manifest["reference_artifacts"]:
        assert not Path(artifact["path"]).is_absolute()
        assert len(artifact["sha256"]) == 64
    state = MODULE.code_state(ROOT / "manifest.json", manifest)
    assert state["declared_pipeline_commit"] == manifest["pipeline_commit"]
    assert state["declared_commit_matches_head"] is (
        state["actual_pipeline_commit"] == manifest["pipeline_commit"]
    )
    assert state["diffractomorph_pipeline_version"] == "0.1.0"
    assert "studies/jpharmsci_clofazimine/analysis/psd_evolution_common.py" in state["file_sha256"]
    assert "src/diffractomorph_pipeline/study/manifest.py" in state["file_sha256"]
    assert "studies/jpharmsci_clofazimine/reproduce.py" in state["file_sha256"]


def test_manifest_rejects_unsafe_recipe_path(tmp_path):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    manifest["recipes"][0]["script"] = "../private.py"
    candidate = tmp_path / "manifest.json"
    candidate.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe recipe script path"):
        MODULE.load_manifest(candidate)


def test_code_state_fails_closed_for_a_stale_declared_commit():
    manifest_path = ROOT / "manifest.json"
    manifest = MODULE.load_manifest(manifest_path)
    manifest["pipeline_commit"] = "a" * 40
    state = MODULE.code_state(manifest_path, manifest)
    assert state["declared_commit_matches_head"] is False


def test_manifest_rejects_recipe_symlink_escape(tmp_path, monkeypatch):
    fake_root = tmp_path / "pipeline"
    (fake_root / "analysis").mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n")
    (fake_root / "analysis" / "recipe.py").symlink_to(outside)
    manifest = {
        "schema_version": 1,
        "bundle_id": "symlink-test",
        "status": "restricted-local-draft",
        "pipeline_commit": "a" * 40,
        "manuscript_commit": "b" * 40,
        "migration_source": {"science_state": MODULE.FROZEN_ARM_B_SCIENCE_STATE},
        "data_layout": {"study": "study"},
        "recipes": [{
            "id": "synthetic",
            "script": "analysis/recipe.py",
            "arguments": ["--output-dir", "{recipe_output}"],
            "expected_outputs": ["result.txt"],
        }],
        "reference_artifacts": [],
    }
    candidate = tmp_path / "symlink-manifest.json"
    candidate.write_text(json.dumps(manifest))
    monkeypatch.setattr(MODULE, "STUDY_ROOT", fake_root)
    with pytest.raises(ValueError, match="resolves outside study root"):
        MODULE.load_manifest(candidate)


def test_manifest_rejects_unsafe_data_layout_and_unknown_generator(tmp_path):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    manifest["data_layout"]["ph_study"] = "../private"
    candidate = tmp_path / "unsafe-layout.json"
    candidate.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe data-layout path"):
        MODULE.load_manifest(candidate)

    manifest = json.loads((ROOT / "manifest.json").read_text())
    manifest["reference_artifacts"][-1]["class"] = "pipeline-generated-inputs-withheld"
    manifest["reference_artifacts"][-1]["generator"] = (
        "diffractomorph_pipeline.missing_parent.missing_module"
    )
    candidate = tmp_path / "unknown-generator.json"
    candidate.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unresolvable module generator"):
        MODULE.load_manifest(candidate)

    manifest = json.loads((ROOT / "manifest.json").read_text())
    manifest["reference_artifacts"][-1]["class"] = "pipeline-generated-inputs-withheld"
    manifest["reference_artifacts"][-1]["generator"] = ".absolute.escape"
    candidate = tmp_path / "unsafe-generator.json"
    candidate.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe module generator"):
        MODULE.load_manifest(candidate)

    manifest = json.loads((ROOT / "manifest.json").read_text())
    renamed = next(
        item for item in manifest["reference_artifacts"]
        if item["class"] == "pipeline-generated-renamed"
    )
    renamed["generated_path"] = "not-a-declared-output.pdf"
    candidate = tmp_path / "bad-generated-path.json"
    candidate.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="is not an output"):
        MODULE.load_manifest(candidate)


def test_manifest_rejects_unknown_artifact_class_and_bad_component(tmp_path):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    manifest["reference_artifacts"][0]["class"] = "pipeline-generatd"
    candidate = tmp_path / "bad-class.json"
    candidate.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported artifact class"):
        MODULE.load_manifest(candidate)

    manifest = json.loads((ROOT / "manifest.json").read_text())
    assembled = next(item for item in manifest["reference_artifacts"] if item.get("components"))
    assembled["components"][0]["generated_path"] = "missing-component.pdf"
    candidate = tmp_path / "bad-component.json"
    candidate.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="component path .* is not an output"):
        MODULE.load_manifest(candidate)


def test_manifest_enforces_frozen_arm_b_state_and_complete_source_closure(tmp_path):
    manifest = MODULE.load_manifest(ROOT / "manifest.json")
    assert manifest["migration_source"]["science_state"] == MODULE.FROZEN_ARM_B_SCIENCE_STATE
    recipe = next(item for item in manifest["recipes"] if item["id"] == "medium_polysorbate_response")
    prefix = Path(manifest["data_layout"]["arm_b"])
    declared = {Path(value) for value in recipe["required_inputs"]}
    actual = {prefix / value for value in arm_b.SOURCES.values()}
    assert declared == actual
    assert any("filtered_48h_pipeline_copt" in str(path) for path in declared)

    altered = json.loads((ROOT / "manifest.json").read_text())
    altered["migration_source"]["science_state"] = "summed-intensity-revision"
    candidate = tmp_path / "changed-science.json"
    candidate.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="must preserve"):
        MODULE.load_manifest(candidate)


def test_preflight_rejects_layout_without_declared_input_closure(tmp_path):
    manifest = MODULE.load_manifest(ROOT / "manifest.json")
    for relative in manifest["data_layout"].values():
        (tmp_path / relative).mkdir(parents=True)
    checks = MODULE.preflight(manifest, tmp_path)
    assert checks
    assert all(item["ok"] for item in checks if item["check"].startswith("data_layout:"))
    assert any(not item["ok"] for item in checks if item["check"].startswith("input:"))


def test_reference_verification_detects_change(tmp_path):
    path = tmp_path / "figure.pdf"
    path.write_bytes(b"reference")
    manifest = {"reference_artifacts": [{
        "path": "figure.pdf", "sha256": MODULE.sha256(path), "class": "writing-repo-assembly"
    }]}
    assert MODULE.verify_reference(manifest, tmp_path)[0]["ok"]
    path.write_bytes(b"changed")
    assert not MODULE.verify_reference(manifest, tmp_path)[0]["ok"]


def test_render_uses_argument_vector_and_confined_recipe_output(tmp_path, monkeypatch):
    script = ROOT / "analysis" / "_test_reproduction_recipe.py"
    manifest = {"recipes": [{
        "id": "synthetic",
        "script": "analysis/_test_reproduction_recipe.py",
        "arguments": ["--output-dir", "{recipe_output}"],
        "expected_outputs": ["result.txt"],
    }]}

    class Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "result.txt").write_text("ok")
        return Completed()

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    results = MODULE.render_recipes(manifest, tmp_path, tmp_path / "out")
    assert results[0]["ok"]
    assert seen["command"][0] == MODULE.sys.executable
    assert (tmp_path / "out" / "synthetic" / "result.txt").is_file()
    assert not script.exists()


@pytest.mark.parametrize(("returncode", "write_output"), [(1, True), (0, False)])
def test_render_failure_or_missing_output_cannot_pass(tmp_path, monkeypatch, returncode, write_output):
    manifest = {"recipes": [{
        "id": "synthetic", "script": "analysis/recipe.py",
        "arguments": ["--output-dir", "{recipe_output}"],
        "expected_outputs": ["result.txt"],
    }]}

    class Completed:
        stdout = ""
        stderr = ""

        def __init__(self, code):
            self.returncode = code

    def fake_run(command, **kwargs):
        if write_output:
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "result.txt").write_text("result")
        return Completed(returncode)

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    assert not MODULE.render_recipes(manifest, tmp_path, tmp_path / "out")[0]["ok"]


def test_generated_text_is_portable_and_disclosure_scanned(tmp_path):
    data_root = tmp_path / "private-data"
    output = tmp_path / "out"
    output.mkdir()
    provenance = output / "provenance.json"
    provenance.write_text(json.dumps({"source": str(data_root / "study" / "input.csv")}))
    assert MODULE._portable_text_outputs(output, data_root) == []
    assert "${DFM_DATA_ROOT}" in provenance.read_text()
    leaked = output / "leaked.txt"
    leaked.write_text("/" + "Users/example/private.csv")
    assert MODULE._portable_text_outputs(output, data_root) == []
    assert leaked.read_text() == "${USER_HOME}/private.csv"
    linux = output / "linux.txt"
    linux.write_text("/" + "home/example/private.csv")
    windows = output / "windows.txt"
    windows.write_text("C:\\" + r"Users\example\private.csv")
    assert MODULE._portable_text_outputs(output, data_root) == []
    assert linux.read_text() == "${USER_HOME}/private.csv"
    assert windows.read_text() == r"${USER_HOME}\private.csv"


def test_unreadable_pdf_fails_disclosure_review(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "broken.pdf").write_bytes(b"not a PDF")
    findings = MODULE._portable_text_outputs(output, tmp_path / "data")
    assert findings[0]["path"] == "broken.pdf"
    assert findings[0]["markers"][0].startswith("PDF text scan failed:")


def test_preflight_report_is_explicitly_partial(tmp_path, monkeypatch):
    manifest = {
        "schema_version": 1,
        "bundle_id": "test-bundle",
        "status": "restricted-local-draft",
        "pipeline_commit": MODULE.code_state(
            ROOT / "manifest.json",
            MODULE.load_manifest(ROOT / "manifest.json"),
        )["actual_pipeline_commit"],
        "manuscript_commit": json.loads(
            (ROOT / "manifest.json").read_text()
        )["manuscript_commit"],
        "migration_source": {"science_state": MODULE.FROZEN_ARM_B_SCIENCE_STATE},
        "data_layout": {"study": "study"},
        "recipes": [{
            "id": "synthetic", "script": "analysis/manuscript_figures.py",
            "arguments": ["--output-dir", "{recipe_output}"],
            "expected_outputs": ["result.txt"],
        }],
        "reference_artifacts": [],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    original_code_state = MODULE.code_state
    monkeypatch.setattr(
        MODULE, "code_state",
        lambda path, value: {**original_code_state(path, value), "worktree_clean": True},
    )
    (tmp_path / "data" / "study").mkdir(parents=True)
    out = tmp_path / "out"
    code = MODULE.main([
        "--manifest", str(manifest_path), "--data-root", str(tmp_path / "data"),
        "--output-dir", str(out), "--mode", "preflight",
    ])
    reports = list(out.glob("reproduction_report_preflight_partial_all_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert code == 0
    assert report["requested_checks_ok"] is True
    assert report["ok"] is False
    assert report["scope"] == "partial"
    assert report["mode"] == "preflight"
    assert report["pipeline_root"] == "${PIPELINE_ROOT}"
    assert len(report["code_state"]["digest"]) == 64
