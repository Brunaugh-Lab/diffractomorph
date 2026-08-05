from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

import diffractomorph_pipeline as dfm
from diffractomorph_pipeline.assay import AssayCalibration
from diffractomorph_pipeline.processing import AggregateKWWConfig
from diffractomorph_pipeline.study import bundled_example_manifest, load_manifest


def test_public_version_and_generic_profile_contract():
    assert dfm.__version__ == "0.1.0"
    profile = AssayCalibration.from_mapping({
        "curves": {"7.0": {"300": [0.05, 0.0]}},
        "blank": {"300": 0.02},
        "filter_offset_ugml": {"7.0": 0.0},
        "dilution": {"sample_uL": 100, "dmso_uL": 0},
        "meta": {"calibration_id": "public-synthetic"},
    })
    assert profile.curve(7.0, 300).concentration(0.52, blank=0.02) == 10.0


def test_synthetic_manifest_is_self_contained():
    project = load_manifest(bundled_example_manifest())
    run = project.read_run("compound-x-prep-a-run-1")
    assert run.provenance.sample_id == "compound-x"
    assert run.signal.shape[1] == 4
    assert np.isfinite(run.signal).all()


def test_paqxos_intensity_template_is_complete_and_parser_aligned():
    root = Path(__file__).resolve().parents[2]
    template = (
        root / "docs" / "templates" / "paqxos_raw_intensity_report_template.txt"
    ).read_text()
    rows = [line for line in template.splitlines() if line[:1].isdigit()]
    assert len(rows) == 31
    assert rows[0] == "1, @I.REF(1), @I.NORM(1)"
    assert rows[-1] == "31, @I.REF(31), @I.NORM(31)"
    assert "Measurement Time: @MTIME" in template
    assert "Optical Concentration: @C.OPT" in template


def test_public_analysis_profile_is_explicit():
    project = load_manifest(bundled_example_manifest())
    config = AggregateKWWConfig.from_profile(project.require_profile("analysis").parameters)
    assert config.channel_ids is None
    assert config.reference_mode == "raw_measured"


def test_unimplemented_figure_command_is_not_exported():
    metadata = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    assert "dfm-figure" not in metadata["project"]["scripts"]


def test_public_urls_never_reuse_private_repository_slug():
    root = Path(__file__).resolve().parents[2]
    public_slug = "Brunaugh-Lab/diffractomorph"
    private_slug = "Brunaugh-Lab/diffractomorph_pipeline"
    for relative in (
        "pyproject.toml", "CITATION.cff", "README.md",
        ".github/ISSUE_TEMPLATE/config.yml",
    ):
        text = (root / relative).read_text()
        assert public_slug in text
        assert private_slug not in text


@pytest.mark.parametrize("entrypoint", [
    "manifest_main", "run_main", "ingest_main", "noise_filter_main",
    "build_kernel_main", "qc_main", "noise_floor_main", "noise_surface_main",
    "extract_main", "aggregate_kww_main",
    "diagnostics_main",
])
def test_every_advertised_command_loads_and_displays_help(entrypoint):
    from diffractomorph_pipeline import cli

    with pytest.raises(SystemExit) as stopped:
        getattr(cli, entrypoint)(["--help"])
    assert stopped.value.code == 0


def test_commands_do_not_claim_withheld_qc_defaults():
    from diffractomorph_pipeline import cli

    with pytest.raises(SystemExit) as missing_kernel_inputs:
        cli.build_kernel_main([])
    assert missing_kernel_inputs.value.code == 2

    with pytest.raises(SystemExit) as missing_qc_reference:
        cli.qc_main(["measurement.rtf"])
    assert missing_qc_reference.value.code == 2


def test_diagnostic_bundle_excludes_paths_and_research_values(tmp_path):
    from diffractomorph_pipeline import cli

    output = tmp_path / "diagnostic.json"
    cli.diagnostics_main([
        str(bundled_example_manifest()), "--output", str(output), "--inspect-runs",
    ])
    payload = json.loads(output.read_text())
    assert payload["privacy"] == {
        "contains_local_paths": False,
        "contains_raw_data": False,
        "contains_sample_or_independent_unit_ids": False,
        "contains_signal_values": False,
    }
    text = output.read_text()
    assert "/" + "Users" + "/" not in text
    assert "compound-x-prep-a-run-1" not in text
    assert "data_root" not in text


def test_missing_optional_legacy_profiles_fail_clearly(monkeypatch, tmp_path):
    from diffractomorph_pipeline import noise_surface, solubility
    from diffractomorph_pipeline.assay import calibration
    from diffractomorph_pipeline.assay import suspension
    from diffractomorph_pipeline.optics import mie

    absent = tmp_path / "not-installed.json"
    monkeypatch.setattr(solubility, "default_path", lambda: absent)
    with pytest.raises(FileNotFoundError, match="optional CFZ solubility"):
        solubility.load_default()

    monkeypatch.setattr(noise_surface, "_surface_path", lambda: absent)
    with pytest.raises(FileNotFoundError, match="optional CFZ noise surface"):
        noise_surface.load_surface()

    monkeypatch.setattr(calibration, "SUSPENSION", {})
    with pytest.raises(FileNotFoundError, match="optional CFZ suspension calibration"):
        suspension.suspension_conc_mgml(0.5)

    with pytest.raises(FileNotFoundError, match="optional optical kernel"):
        mie.load_kernel(absent)


def test_snapshot_builder_is_exact_and_manifest_complete(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "snapshot"
    subprocess.run(
        [sys.executable, str(root / "scripts" / "build_public_snapshot.py"), str(output)],
        check=True,
    )
    manifest_path = output / "PUBLIC_SNAPSHOT_MANIFEST.sha256"
    entries = {}
    for line in manifest_path.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path != manifest_path
    }
    assert actual == set(entries)
    for relative, digest in entries.items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == digest
    subprocess.run(
        [sys.executable, str(root / "scripts" / "build_public_snapshot.py"),
         "--verify-existing", str(output)],
        check=True,
    )

    policy = json.loads((output / "release" / "public_snapshot_policy.json").read_text())
    assert "examples" not in policy["directory_roots"]
    assert not (output / "src" / "diffractomorph_pipeline" / "data" / "standards").exists()
    exported_template = (
        output / "docs" / "templates" / "paqxos_raw_intensity_report_template.txt"
    )
    assert exported_template.is_file()
    assert exported_template.read_text() == (
        root / "docs" / "templates" / "paqxos_raw_intensity_report_template.txt"
    ).read_text()
    assert "/" + "Users" + "/" not in exported_template.read_text()

    spec = importlib.util.spec_from_file_location(
        "build_public_snapshot", root / "scripts" / "build_public_snapshot.py",
    )
    builder = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(builder)
    disclosure = tmp_path / "disclosure.txt"
    disclosure.write_text(
        "contact person@med." + "umich.edu from /" + "home/person/project",
    )
    findings = builder._audit_text(disclosure, Path("disclosure.txt"))
    assert any("institutional email" in item for item in findings)
    assert any("Linux user path" in item for item in findings)

    workflow = (root / ".github" / "workflows" / "ci.yml").read_text()
    verify_at = workflow.index("--verify-existing")
    copy_at = workflow.index("public-build")
    wheel_at = workflow.index("pip wheel")
    assert verify_at < copy_at < wheel_at
    assert 'pip wheel --no-deps --wheel-dir dist "${RUNNER_TEMP}/public-build"' in workflow


def test_unknown_material_requires_explicit_forward_parameters():
    from diffractomorph_pipeline.forward import PSD, predict

    psd = PSD.from_q3([1.0, 2.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="unknown material"):
        predict(psd, ph=6.0, dose_mg=1.0, drug="compound-y", t_end=1.0, n_eval=2)


def test_kernel_build_api_requires_external_registry_and_calibration_date(tmp_path):
    from diffractomorph_pipeline.optics import mie_build

    with pytest.raises(ValueError, match="cal_date is required"):
        mie_build.build_kernel_from_files(
            "missing.rtf", "missing.csv", "compound-y", registry=tmp_path / "registry.yaml",
        )
    with pytest.raises(ValueError, match="outside the installed"):
        mie_build.build_kernel_from_files(
            "missing.rtf", "missing.csv", "compound-y", cal_date="2026-08-04",
            registry=Path(mie_build.__file__).parent / "registry.yaml",
        )
    with pytest.raises(ValueError, match="registry is required"):
        mie_build.resolve_kernel_for_day("2026-08-04", "compound-y")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        mie_build.build_kernel_from_files(
            "missing.rtf", "missing.csv", "compound-y", cal_date="20260804",
            registry=tmp_path / "registry.yaml",
        )
