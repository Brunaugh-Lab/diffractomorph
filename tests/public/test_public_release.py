from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import diffractomorph_pipeline as dfm
from diffractomorph_pipeline.assay import AssayCalibration
from diffractomorph_pipeline.model import Run, RunProvenance
from diffractomorph_pipeline.processing import (
    AggregateKWWConfig, fit_aggregate_kww, select_aggregate_start,
)
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
    assert config.start_policy == "first_frame"


def _synthetic_run(signal, copt, time_min=None):
    signal = np.asarray(signal, dtype=float)
    if time_min is None:
        time_min = np.arange(signal.shape[0], dtype=float)
    return Run(
        signal=signal,
        channel_ids=tuple(f"ch{index + 1}" for index in range(signal.shape[1])),
        time_min=np.asarray(time_min, dtype=float),
        acquisition={"copt": np.asarray(copt, dtype=float)},
        provenance=RunProvenance(
            run_id="synthetic-start-test", source_path="synthetic.csv",
            adapter="tidy_csv", sample_id="compound-x",
        ),
        run_kind="measurement",
    )


def test_concordant_early_maximum_selects_and_rezeros_shape_preserving_rise():
    pattern = np.array([4.0, 3.0, 2.0, 1.0])
    scales = np.array([1.0, 1.5, 1.35, 1.0, 0.7, 0.5, 0.4])
    run = _synthetic_run(scales[:, None] * pattern, [1.0, 1.45, 1.3, 1.0, 0.8, 0.6, 0.5])
    config = AggregateKWWConfig(
        start_policy="concordant_early_maximum",
        start_search_frames=3,
        start_maximum_time_min=2.0,
        start_minimum_relative_increase=0.2,
        start_minimum_spectral_cosine=0.995,
    )
    result = fit_aggregate_kww(run, config)
    assert result.start_index == 1
    assert result.start_reason == "concordant_early_maximum"
    assert result.time_min[0] == 0.0
    assert result.selected_elapsed_time_min == 1.0


def test_concordant_early_maximum_rejects_shape_changing_rise():
    run = _synthetic_run(
        [[4, 3, 2, 1], [1, 2, 3, 8], [3, 2, 1.5, 0.8], [2, 1.5, 1, 0.5],
         [1.5, 1, 0.8, 0.4], [1.0, 0.8, 0.6, 0.3], [0.8, 0.6, 0.4, 0.2]],
        [1.0, 1.5, 1.3, 1.0, 0.8, 0.6, 0.5],
    )
    config = AggregateKWWConfig(
        start_policy="concordant_early_maximum",
        start_search_frames=3,
        start_maximum_time_min=2.0,
        start_minimum_relative_increase=0.2,
        start_minimum_spectral_cosine=0.995,
    )
    aggregate = run.signal.sum(axis=1)
    index, reason = select_aggregate_start(run, aggregate, run.channel_ids, config)
    assert index == 0
    assert reason == "early_maximum_changed_angular_pattern"


def test_first_frame_policy_preserves_released_start_behavior():
    pattern = np.array([4.0, 3.0, 2.0, 1.0])
    scales = np.array([1.0, 1.5, 1.35, 1.0, 0.7, 0.5, 0.4])
    run = _synthetic_run(scales[:, None] * pattern, scales)
    result = fit_aggregate_kww(run, AggregateKWWConfig(start_policy="first_frame"))
    assert result.start_index == 0
    assert result.selected_elapsed_time_min == 0.0
    assert result.start_reason == "first_frame"


def test_concordant_start_requires_declared_acquisition_variable():
    run = _synthetic_run([[4, 3], [6, 4.5], [5, 4]], [1.0, 1.5, 1.3])
    config = AggregateKWWConfig(
        start_policy="concordant_early_maximum",
        start_acquisition_variable="transmission",
        start_maximum_time_min=2.0,
        start_minimum_relative_increase=0.2,
        start_minimum_spectral_cosine=0.995,
    )
    with pytest.raises(KeyError, match="transmission"):
        fit_aggregate_kww(run, config)


def test_concordant_start_rejects_nonfinite_early_acquisition():
    run = _synthetic_run([[4, 3], [6, 4.5], [5, 4]], [1.0, np.nan, 1.3])
    config = AggregateKWWConfig(
        start_policy="concordant_early_maximum",
        start_maximum_time_min=2.0,
        start_minimum_relative_increase=0.2,
        start_minimum_spectral_cosine=0.995,
    )
    aggregate = run.signal.sum(axis=1)
    index, reason = select_aggregate_start(run, aggregate, run.channel_ids, config)
    assert index == 0
    assert reason == "nonfinite_early_start_variable"


def test_concordant_start_profile_requires_complete_explicit_contract():
    project = load_manifest(bundled_example_manifest())
    parameters = dict(project.require_profile("analysis").parameters)
    parameters["start_boundary"] = {
        "policy": "concordant_early_maximum",
        "acquisition_variable": "copt",
        "search_frames": 3,
        "maximum_time_min": 1.0,
        "minimum_relative_increase": 0.2,
        "minimum_spectral_cosine": 0.995,
    }
    config = AggregateKWWConfig.from_profile(parameters)
    assert config.start_policy == "concordant_early_maximum"
    assert config.start_maximum_time_min == 1.0
    assert config.start_minimum_relative_increase == 0.2
    assert config.start_minimum_spectral_cosine == 0.995

    incomplete = dict(parameters)
    incomplete["start_boundary"] = dict(parameters["start_boundary"])
    incomplete["start_boundary"].pop("maximum_time_min")
    with pytest.raises(ValueError, match="maximum_time_min"):
        AggregateKWWConfig.from_profile(incomplete)

    missing_boundary = dict(parameters)
    missing_boundary.pop("start_boundary")
    with pytest.raises(ValueError, match="start_boundary"):
        AggregateKWWConfig.from_profile(missing_boundary)

    invalid = dict(parameters)
    invalid["start_boundary"] = dict(parameters["start_boundary"])
    invalid["start_boundary"]["maximum_time_min"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        AggregateKWWConfig.from_profile(invalid)


def test_concordant_start_rejects_candidate_after_startup_interval():
    pattern = np.array([4.0, 3.0, 2.0, 1.0])
    scales = np.array([1.0, 1.5, 1.3])
    run = _synthetic_run(
        scales[:, None] * pattern,
        [1.0, 1.5, 1.3],
        time_min=[0.0, 1.2, 1.4],
    )
    config = AggregateKWWConfig(
        start_policy="concordant_early_maximum",
        start_search_frames=3,
        start_maximum_time_min=1.0,
        start_minimum_relative_increase=0.2,
        start_minimum_spectral_cosine=0.995,
    )
    aggregate = run.signal.sum(axis=1)
    index, reason = select_aggregate_start(run, aggregate, run.channel_ids, config)
    assert index == 0
    assert reason == "early_maximum_after_startup_interval"


def test_start_selection_rejects_misaligned_aggregate():
    run = _synthetic_run([[4, 3], [6, 4.5], [5, 4]], [1.0, 1.5, 1.3])
    config = AggregateKWWConfig(
        start_policy="concordant_early_maximum",
        start_search_frames=3,
        start_maximum_time_min=1.0,
        start_minimum_relative_increase=0.2,
        start_minimum_spectral_cosine=0.995,
    )
    with pytest.raises(ValueError, match="one value per run frame"):
        select_aggregate_start(run, np.array([7.0, 10.5]), run.channel_ids, config)


def test_python_310_public_config_entrypoint_imports():
    from diffractomorph_pipeline.config import load_project

    assert load_project(bundled_example_manifest()).project_id == "synthetic-compound-x"


def test_unimplemented_figure_command_is_not_exported():
    metadata = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    assert "dfm-figure" not in metadata


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

    with pytest.raises(SystemExit) as implicit_kernel_identity:
        cli.build_kernel_main([
            "--cal-date", "2026-08-04", "--registry", "registry.yaml",
            "--nist-intensity", "nist.rtf", "--nist-psd", "nist.csv",
        ])
    assert implicit_kernel_identity.value.code == 2

    with pytest.raises(SystemExit) as implicit_noise_output:
        cli.noise_surface_main(["calibration.rtf"])
    assert implicit_noise_output.value.code == 2

    with pytest.raises(SystemExit) as missing_qc_reference:
        cli.qc_main(["measurement.rtf"])
    assert missing_qc_reference.value.code == 2


def test_aggregate_cli_writes_start_provenance_and_summary_counts(tmp_path):
    from diffractomorph_pipeline import cli

    output = tmp_path / "aggregate"
    cli.aggregate_kww_main([
        str(bundled_example_manifest()), "--output-dir", str(output),
        "--artifact-correction", "off",
    ])
    by_run = pd.read_csv(output / "aggregate_kww_by_run.csv")
    by_unit = pd.read_csv(output / "aggregate_kww_by_independent_unit.csv")
    by_condition = pd.read_csv(output / "aggregate_kww_by_condition.csv")
    assert {
        "start_policy", "start_index", "selected_elapsed_time_min", "start_reason",
    }.issubset(by_run.columns)
    assert by_run.loc[0, "start_index"] == 0
    assert by_unit.loc[0, "n_runs_started_late"] == 0
    assert by_condition.loc[0, "n_runs_started_late"] == 0
    assert by_condition.loc[0, "n_independent_units_started_late"] == 0


def test_aggregate_cli_rejects_two_startup_treatments(tmp_path):
    from diffractomorph_pipeline import cli

    source_manifest = bundled_example_manifest()
    payload = yaml.safe_load(source_manifest.read_text())
    payload["data_root"] = str(source_manifest.parent)
    analysis = payload["profiles"]["analysis"]["parameters"]
    analysis["start_boundary"] = {
        "policy": "concordant_early_maximum",
        "acquisition_variable": "copt",
        "search_frames": 3,
        "maximum_time_min": 1.0,
        "minimum_relative_increase": 0.2,
        "minimum_spectral_cosine": 0.995,
    }
    analysis["artifact_correction"] = {
        "acquisition_variable": "copt",
        "synchronized_intensity_z": 4.0,
        "synchronized_acquisition_z": 1.5,
        "synchronized_half_window": 2,
        "isolated_spike_mad": 5.0,
        "gap_threshold_min": 2.0,
    }
    manifest = tmp_path / "two-startup-treatments.yaml"
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(SystemExit) as stopped:
        cli.aggregate_kww_main([
            str(manifest), "--output-dir", str(tmp_path / "output"),
        ])
    assert stopped.value.code == 2


def test_aggregate_cli_reports_missing_start_acquisition_cleanly(tmp_path):
    from diffractomorph_pipeline import cli

    source_manifest = bundled_example_manifest()
    payload = yaml.safe_load(source_manifest.read_text())
    payload["data_root"] = str(source_manifest.parent)
    start = payload["profiles"]["analysis"]["parameters"]["start_boundary"]
    start.update({
        "policy": "concordant_early_maximum",
        "acquisition_variable": "transmission",
        "search_frames": 3,
        "maximum_time_min": 1.0,
        "minimum_relative_increase": 0.2,
        "minimum_spectral_cosine": 0.995,
    })
    manifest = tmp_path / "missing-start-acquisition.yaml"
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(SystemExit) as stopped:
        cli.aggregate_kww_main([
            str(manifest), "--output-dir", str(tmp_path / "output"),
            "--artifact-correction", "off",
        ])
    assert stopped.value.code == 2


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


def test_snapshot_builder_is_exact_and_manifest_complete(tmp_path, monkeypatch):
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
    forbidden_study_suffixes = {".csv", ".tsv", ".xlsx", ".xls", ".npz", ".rtf"}
    assert not [
        path for path in (output / "studies").rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_study_suffixes
    ]
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
    assert builder._study_file_allowed(
        Path("studies/jpharmsci_clofazimine/analysis.py"), policy
    )
    assert not builder._study_file_allowed(
        Path("studies/jpharmsci_clofazimine/derived.csv"), policy
    )
    fake_root = tmp_path / "fake-repo"
    leaked_study_file = fake_root / "studies" / "jpharmsci_clofazimine" / "derived.csv"
    leaked_study_file.parent.mkdir(parents=True)
    leaked_study_file.write_text("real,data\n")
    monkeypatch.setattr(builder, "ROOT", fake_root)
    assert "unapproved study-layer file" in builder._study_tree_findings(policy)[0]
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
    from diffractomorph_pipeline.forward import Parameters, PSD, predict, predict_from_snapshot

    psd = PSD.from_q3([1.0, 2.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="material parameters are required"):
        predict(psd, ph=6.0, dose_mg=1.0, t_end=1.0, n_eval=2)
    with pytest.raises(ValueError, match="unknown material"):
        predict(psd, ph=6.0, dose_mg=1.0, drug="compound-y", t_end=1.0, n_eval=2)
    with pytest.raises(ValueError, match="must agree"):
        predict_from_snapshot(
            psd, ph=6.0, injected_mg=1.0, conc_ugml=1.0, volume_mL=20.0,
            params=Parameters(mw=100.0, pka_bh=5.0, s0_uM=1.0, v_diss_mL=20.0),
            v_diss_mL=40.0, t_end=1.0, n_eval=2,
        )


def test_kernel_build_api_requires_external_registry_and_calibration_date(tmp_path):
    from diffractomorph_pipeline.optics import mie_build

    with pytest.raises(ValueError, match="cal_date is required"):
        mie_build.build_kernel_from_files(
            "missing.rtf", "missing.csv", "compound-y", lens="R3",
            registry=tmp_path / "registry.yaml",
        )
    with pytest.raises(ValueError, match="outside the installed"):
        mie_build.build_kernel_from_files(
            "missing.rtf", "missing.csv", "compound-y", lens="R3", cal_date="2026-08-04",
            registry=Path(mie_build.__file__).parent / "registry.yaml",
        )
    with pytest.raises(ValueError, match="registry is required"):
        mie_build.resolve_kernel_for_day("2026-08-04", "compound-y", lens="R3")
    with pytest.raises(ValueError, match="lens is required"):
        mie_build.resolve_kernel_for_day(
            "2026-08-04", "compound-y", registry=tmp_path / "registry.yaml",
        )
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        mie_build.build_kernel_from_files(
            "missing.rtf", "missing.csv", "compound-y", lens="R3", cal_date="20260804",
            registry=tmp_path / "registry.yaml",
        )
