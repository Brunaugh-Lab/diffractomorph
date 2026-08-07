"""Guardrails for the Section 3.4.2 main-text starting-particle-loading figure.

The figure makes one claim that is easy to overstate — a frozen model predicted a loading trend
that the UV measurements followed — and three facts qualify it: the ladder came from ONE
suspension preparation with technical replicates, nothing was refitted, and the UV ordering
reverses below an offset the calibration cannot bound. These tests pin all three alongside the
arithmetic, and pin the scope: the manuscript reports the pH 4.5 sub-study, so no other sub-study
may reach the figure, the source data, the caption or the provenance sidecar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))

import manuscript_loading_common as mlc              # noqa: E402
import manuscript_loading_response as fig            # noqa: E402


def _have_sources() -> bool:
    try:
        return mlc.sources_present()
    except Exception:
        return False


corpus = pytest.mark.skipif(not _have_sources(), reason="copt_loading artifacts not present")


# 1 ── three panels, one row ──────────────────────────────────────────────────────────────────

@corpus
def test_exactly_three_panels_are_produced():
    table, _ = fig.build_source_data()
    assert set(table["panel"]) == {"A", "B", "C"}


def test_renderer_creates_a_single_row_of_three_axes():
    import inspect
    body = inspect.getsource(fig.render)
    assert "add_gridspec(1, 3" in body
    assert '(axA, "A"), (axB, "B"), (axC, "C")' in body


# 2 ── scope: the manuscript sub-study only, everywhere a reader looks ────────────────────────

@corpus
def test_only_the_manuscript_substudy_reaches_the_source_data():
    table, meta = fig.build_source_data()
    assert set(table["substudy"]) == {mlc.MANUSCRIPT_SUBSTUDY} == {"pH 4.5"}
    assert meta["design"]["preparation"] == [mlc.MANUSCRIPT_PREP]


@corpus
def test_the_out_of_scope_substudy_is_absent_from_every_written_artifact(tmp_path):
    fig.main(["--output-dir", str(tmp_path), "--formats", "pdf"])
    for suffix in ("_source_data.csv", "_caption.md", "_provenance.json"):
        text = (tmp_path / f"{fig.STEM}{suffix}").read_text()
        assert not mlc.find_foreign_tokens(text, mlc.OUT_OF_SCOPE_SUBSTUDY_TOKENS), suffix
        for literal in ("pH 4.0", "pH4.0", "pH = 4.0", "20260623"):
            assert literal not in text, (suffix, literal)


@corpus
def test_the_reader_never_sees_an_out_of_scope_row_even_though_the_artifact_carries_them():
    # the admission happens once, in read_scoped, and the raw artifact is genuinely larger
    from diffractomorph_pipeline.config import data_root
    raw = pd.read_csv(data_root() / mlc.SOURCES["runs"])
    scoped = mlc.read_runs()
    assert len(scoped) < len(raw)
    assert set(scoped["substudy"]) == {mlc.MANUSCRIPT_SUBSTUDY}


def test_read_scoped_rejects_a_table_that_keeps_out_of_scope_rows(monkeypatch):
    frame = pd.DataFrame({"substudy": ["pH 4.5"], "x": [1]})
    monkeypatch.setattr(mlc, "read", lambda key: frame.assign(substudy="other"))
    with pytest.raises(ValueError, match="carries no pH 4.5 rows"):
        mlc.read_scoped("runs")


# 3 ── design: one preparation, three TECHNICAL replicates ────────────────────────────────────

@corpus
def test_one_preparation_and_three_technical_replicates_per_level():
    table, meta = fig.build_source_data()
    design = meta["design"]
    assert design["n_preparations"] == 1
    assert design["technical_reps_per_level"] == [3]
    assert design["n_levels"] == 3 and design["n_runs"] == 9
    assert design["replicates_are_technical"] is True
    assert design["loading_levels_are_aliquots_of_one_suspension"] is True
    runs = table[table["kind"].eq("run_point")]
    assert runs.groupby("level_pct")["rep"].nunique().tolist() == [3, 3, 3]


@corpus
def test_error_bars_are_declared_as_technical_replicate_scatter():
    assert "TECHNICAL" in mlc.AGGREGATIONS["level_sd"]
    assert "not a preparation-level error term" in mlc.AGGREGATIONS["level_sd"]


# 4 ── the frozen model is frozen ─────────────────────────────────────────────────────────────

@corpus
def test_rate_scale_source_and_column_are_the_frozen_historical_fit():
    _, meta = fig.build_source_data()
    frozen = meta["frozen_model"]
    assert frozen["rate_scale"] == 2.197
    assert frozen["source_file"] == "selected_rate_only_fit_summary.csv"
    assert frozen["source_column"] == "rate_scale_datebalanced"
    assert frozen["refitted_here"] is False


@corpus
def test_the_shared_qc_psd_is_primary_and_the_per_run_q0_is_not_plotted():
    table, meta = fig.build_source_data()
    frozen = meta["frozen_model"]
    assert "shared suspension QC" in frozen["psd_primary"]
    assert "model_pct_end_per_run_q0" in frozen["psd_sensitivity_not_plotted"]
    assert not table["source_column"].astype(str).str.contains("per_run_q0").any()
    model = table[table["kind"].eq("model_level")]
    assert set(model["source_column"]) == {"model_pct_end"}


@corpus
def test_a_refitted_provenance_is_rejected(monkeypatch):
    real = mlc.read_provenance

    def fake():
        record = json.loads(json.dumps(real()))
        record["frozen_model"]["refitted_here"] = True
        return record

    monkeypatch.setattr(mlc, "read_provenance", fake)
    with pytest.raises(ValueError, match="refitted_here = true"):
        mlc.frozen_model_record(mlc.read_runs())


# 5 ── the displayed numbers reproduce the authoritative artifacts ────────────────────────────

@corpus
def test_displayed_values_match_the_established_results():
    table, meta = fig.build_source_data()
    checks = fig.validate(table, meta)
    for level, expected in ((12, (133.0, 9.9, 96.7, 99.0, 7.7)),
                            (18, (177.4, 12.0, 94.8, 92.9, 2.7)),
                            (24, (221.7, 14.2, 91.5, 87.2, 6.1))):
        got = checks["displayed_values"][level]
        assert got["ok"], got
        assert (got["dose_ug"], got["copt0_pct"], got["model_pct"],
                got["uv_pct"], got["uv_sd_pp"]) == expected


@corpus
def test_level_means_agree_with_the_tabulated_level_mean_artifact():
    table, meta = fig.build_source_data()
    checks = fig.validate(table, meta)
    assert checks["agrees_with_level_mean_artifact"]["ok"]
    assert checks["agrees_with_level_mean_artifact"]["max_abs_deviation"] < 1e-9


@corpus
def test_the_two_orderings_the_figure_asserts():
    table, meta = fig.build_source_data()
    checks = fig.validate(table, meta)["orderings"]
    assert checks["copt_ladder_monotonic_increasing"]
    assert checks["model_monotonic_decreasing"]
    assert checks["uv_monotonic_decreasing"]


# 6 ── panel C: the conditionality, and no uncertainty band ───────────────────────────────────

@corpus
def test_panel_c_plots_one_contrast_not_three_recovery_curves():
    table, meta = fig.build_source_data()
    c = table[table["panel"].eq("C") & table["kind"].eq("offset_contrast")]
    assert set(c["y_quantity"]) == {"low_minus_high_load_recovery_pp"}
    assert set(c["series"]) == {fig.CONTRAST_SERIES}
    assert (fig.LOW_LEVEL, fig.HIGH_LEVEL) == (12, 24)
    # one value per tabulated offset, not one per level per offset
    offsets = mlc.read("offsets")
    assert len(c) == offsets["offset_ugml"].nunique()


@corpus
def test_the_crossover_is_0_59_and_about_40_percent_of_the_calibrated_offset():
    table, meta = fig.build_source_data()
    checks = fig.validate(table, meta)["offset_conditionality"]
    assert round(checks["crossover_offset_ugml"], 2) == 0.59
    assert checks["calibrated_offset_ugml"] == 1.48
    assert checks["fraction_displayed_pct"] == 40
    assert checks["agrees_with_provenance"]
    assert checks["contrast_negative_below_crossover"]
    assert checks["contrast_positive_above_crossover"]


@corpus
def test_no_uncertainty_band_is_drawn_because_none_exists():
    _, meta = fig.build_source_data()
    c = meta["panel_c"]
    assert c["uncertainty_interval_available"] is False
    assert c["uncertainty_band_drawn"] is False
    assert "no offset uncertainty interval" in c["uncertainty_band_reason"]
    assert meta["panel_b"]["model_uncertainty_band_drawn"] is False


@corpus
def test_the_contrast_is_exactly_linear_in_the_offset():
    # the correction is additive, so a curved contrast would mean the artifact changed shape
    _, meta = fig.build_source_data()
    assert meta["panel_c"]["linear_in_offset"]["max_abs_residual_pp"] < 1e-9


# 7 ── caption wording ────────────────────────────────────────────────────────────────────────

@corpus
def test_caption_states_the_boundaries_and_avoids_overstatement(tmp_path):
    fig.main(["--output-dir", str(tmp_path), "--formats", "pdf"])
    caption = " ".join((tmp_path / f"{fig.STEM}_caption.md").read_text().split())
    low = caption.lower()
    assert "not a preparation-level validation" in low
    assert "conditional" in low
    assert "technical replicates" in low
    assert "no preparation-level error term" in low
    assert "no model parameter was refitted" in low.replace("**", "")
    assert "do not independently confirm" in low
    for banned in ("external validation", "robust loading trend", "cleanest validation",
                   "validates the model", "independent confirmation"):
        assert banned not in low, banned
    # "preparation-level validation" may appear only as a denial
    assert low.count("preparation-level validation") == low.count(
        "not a preparation-level validation")


def test_check_wording_rejects_an_unnegated_validation_claim():
    result = mlc.check_wording("This is a preparation-level validation of the model.",
                               allow_negated=("preparation-level validation",))
    assert not result["ok"]
    assert "preparation-level validation" in result["banned_phrases_found"]


def test_check_wording_does_not_read_between_as_a_polysorbate_token():
    assert mlc.check_wording("the contrast between the ends of the ladder")["ok"]


# 8 ── outputs and provenance ─────────────────────────────────────────────────────────────────

@corpus
def test_all_six_outputs_are_written(tmp_path):
    fig.main(["--output-dir", str(tmp_path), "--formats", "pdf,png,svg"])
    for suffix in (".pdf", ".png", ".svg", "_source_data.csv",
                   "_provenance.json", "_caption.md"):
        path = tmp_path / f"{fig.STEM}{suffix}"
        assert path.exists() and path.stat().st_size > 0, suffix


@corpus
def test_source_data_carries_every_plotted_value_with_its_artifact(tmp_path):
    fig.main(["--output-dir", str(tmp_path), "--formats", "pdf"])
    table = pd.read_csv(tmp_path / f"{fig.STEM}_source_data.csv")
    # the two reference offsets are vertical lines: they carry an x position and no y value
    marks = table[table["kind"].eq("reference_offset")]
    assert len(marks) == 2 and marks["x_value"].notna().all() and marks["y_value"].isna().all()
    assert table[table["kind"].ne("reference_offset")]["y_value"].notna().all()
    assert table["x_value"].notna().all()
    assert table["source_artifact"].notna().all()
    assert set(table["source_artifact"]) <= {
        "copt_loading_runs.csv", "copt_loading_filter_offset_sensitivity.csv", "provenance.json"}


@corpus
def test_provenance_records_sources_frozen_model_and_claim_boundaries(tmp_path):
    fig.main(["--output-dir", str(tmp_path), "--formats", "pdf"])
    prov = json.loads((tmp_path / f"{fig.STEM}_provenance.json").read_text())
    assert set(prov["sources"]) == set(mlc.SOURCES)
    assert prov["frozen_model"]["rate_scale"] == 2.197
    assert prov["frozen_model"]["refitted_here"] is False
    assert prov["scope"]["substudy"] == mlc.MANUSCRIPT_SUBSTUDY
    assert prov["career_artifacts_used"] is False
    assert prov["filter_offset"]["uncertainty_interval_available"] is False
    assert prov["git_commit"]
    joined = " ".join(prov["scope"]["claim_boundaries"]).lower()
    for boundary in ("technical", "frozen at 2.197", "conditional", "no uncertainty interval",
                     "not used as confirmation"):
        assert boundary in joined, boundary


@corpus
def test_the_analysis_figure_is_not_overwritten(tmp_path):
    fig.main(["--output-dir", str(tmp_path), "--formats", "pdf"])
    assert not (tmp_path / "copt_loading.png").exists()
    prov = json.loads((tmp_path / f"{fig.STEM}_provenance.json").read_text())
    assert "copt_loading" in prov["does_not_overwrite"]


def test_missing_source_raises_rather_than_reconstructing(tmp_path, monkeypatch):
    monkeypatch.setattr(mlc, "data_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="do not reconstruct"):
        mlc.read("runs")


# 9 ── axis limits ────────────────────────────────────────────────────────────────────────────

@corpus
def test_no_panel_clips_a_drawn_value_or_error_bar():
    table, meta = fig.build_source_data()
    checks = fig.validate(table, meta)["axis_limits_do_not_clip"]
    for panel, result in checks.items():
        assert result["ok"], (panel, result)


@corpus
def test_dose_axis_covers_every_delivered_dose():
    table, _ = fig.build_source_data()
    doses = table[table["x_quantity"].eq("delivered_dose_ug")]["x_value"]
    assert fig.DOSE_LIM[0] < float(doses.min()) and float(doses.max()) < fig.DOSE_LIM[1]
    assert np.allclose(sorted(fig.DOSE_TICKS), sorted(fig.DOSE_TICKS))
