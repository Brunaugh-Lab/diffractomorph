"""Guardrails for the Section 3.4.2 Supporting Information optical-diagnostics figure.

This figure is the one a reader is most likely to over-read: fractional Copt loss rises across the
loading ladder while the UV dissolved-mass fraction falls, and q3 D50 paths separate slightly.
Neither is a mass measurement and neither is independent of the laser-diffraction acquisition, so
the tests pin the descriptive framing — no confirmation of the mass result, no invariance or
difference verdict on the size paths — alongside the arithmetic and the scope.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))

import manuscript_loading_common as mlc                 # noqa: E402
import manuscript_si_loading_diagnostics as si          # noqa: E402


def _have_sources() -> bool:
    try:
        return mlc.sources_present()
    except Exception:
        return False


corpus = pytest.mark.skipif(not _have_sources(), reason="copt_loading artifacts not present")


# 1 ── exactly two panels ─────────────────────────────────────────────────────────────────────

@corpus
def test_exactly_two_panels_are_produced():
    table, _ = si.build_source_data()
    assert set(table["panel"]) == {"A", "B"}


def test_renderer_creates_a_single_row_of_two_axes():
    import inspect
    body = inspect.getsource(si.render)
    assert "add_gridspec(1, 2" in body
    assert '(axA, "A"), (axB, "B")' in body


@corpus
def test_only_the_two_optical_quantities_are_plotted():
    table, _ = si.build_source_data()
    assert set(table["y_quantity"]) == {"copt_fractional_loss_pct", "q3_d50_um"}
    assert set(table["x_quantity"]) == {"delivered_dose_ug", "matched_optical_extent_g"}


# 2 ── scope: the manuscript sub-study only ───────────────────────────────────────────────────

@corpus
def test_only_the_manuscript_substudy_reaches_the_source_data():
    table, meta = si.build_source_data()
    assert set(table["substudy"]) == {mlc.MANUSCRIPT_SUBSTUDY} == {"pH 4.5"}
    assert meta["design"]["preparation"] == [mlc.MANUSCRIPT_PREP]


@corpus
def test_the_out_of_scope_substudy_is_absent_from_every_written_artifact(tmp_path):
    si.main(["--output-dir", str(tmp_path), "--formats", "pdf"])
    for suffix in ("_source_data.csv", "_caption.md", "_provenance.json"):
        text = (tmp_path / f"{si.STEM}{suffix}").read_text()
        assert not mlc.find_foreign_tokens(text, mlc.OUT_OF_SCOPE_SUBSTUDY_TOKENS), suffix
        for literal in ("pH 4.0", "pH4.0", "pH = 4.0", "20260623"):
            assert literal not in text, (suffix, literal)


@corpus
def test_the_q3_artifacts_carry_more_than_the_figure_admits():
    # the frame-level artifact spans both sub-studies; only the in-scope rows are read
    from diffractomorph_pipeline.config import data_root
    raw = pd.read_csv(data_root() / mlc.SOURCES["q3_frames"])
    assert raw["substudy"].nunique() > 1
    assert set(mlc.read_scoped("q3_frames")["substudy"]) == {mlc.MANUSCRIPT_SUBSTUDY}


# 3 ── design ─────────────────────────────────────────────────────────────────────────────────

@corpus
def test_one_preparation_and_three_technical_replicates_per_level():
    _, meta = si.build_source_data()
    design = meta["design"]
    assert design["n_preparations"] == 1
    assert design["technical_reps_per_level"] == [3]
    assert design["replicates_are_technical"] is True


# 4 ── panel A: the optical extent moves OPPOSITE to the mass fraction ────────────────────────

@corpus
def test_copt_loss_values_reproduce_the_artifact():
    table, meta = si.build_source_data()
    checks = si.validate(table, meta)["panel_a_values"]
    assert checks[12]["copt_loss_pct"] == 58.0
    assert checks[18]["copt_loss_pct"] == 72.0
    assert checks[24]["copt_loss_pct"] == 78.3
    for level in (12, 18, 24):
        assert checks[level]["ok"], checks[level]


@corpus
def test_copt_loss_rises_while_the_uv_mass_fraction_falls():
    table, meta = si.build_source_data()
    checks = si.validate(table, meta)["optical_opposes_mass"]
    assert checks["copt_loss_rises_with_loading"]
    assert checks["uv_mass_fraction_falls_with_loading"]
    assert checks["uv_matches_expected"]
    assert checks["used_as_confirmation"] is False


@corpus
def test_copt_loss_is_never_labelled_a_dose_fraction():
    table, meta = si.build_source_data()
    assert meta["panel_a"]["used_as_confirmation_of_mass_result"] is False
    assert "not a dissolved-dose fraction" in meta["panel_a"]["interpretation"]
    a = table[table["panel"].eq("A")]
    assert set(a["y_quantity"]) == {"copt_fractional_loss_pct"}


# 5 ── panel B: gated, descriptive, no verdict ────────────────────────────────────────────────

@corpus
def test_every_plotted_size_comes_from_the_gated_matched_extent_artifact():
    table, meta = si.build_source_data()
    checks = si.validate(table, meta)["panel_b_gate_and_spread"]
    assert checks["ok"], checks
    assert checks["n_plotted_points"] == checks["n_matched_to_artifact"]
    assert checks["max_abs_deviation_um"] < 1e-12
    matched = mlc.read_scoped("q3_matched")
    assert checks["n_plotted_points"] == len(matched) == 54     # 3 levels x 3 reps x 6 extents


@corpus
def test_the_reliability_gate_is_the_established_one():
    _, meta = si.build_source_data()
    gate = meta["panel_b"]["gate"]
    assert mlc.TAIL_MAX_PCT == 5.0
    assert "15 um" in gate["rule"] and "5 %" in gate["rule"]
    assert "copt_loading.TAIL_MAX_PCT" in gate["source"]
    # the tabulated per-frame flag really is that rule
    frames = mlc.read_scoped("q3_frames")
    assert (frames["q3_frame_reliable"]
            == (frames["tail_pct_above_15um"] <= mlc.TAIL_MAX_PCT)).all()


@corpus
def test_the_descriptive_spread_reproduces_the_tabulated_summary():
    table, meta = si.build_source_data()
    checks = si.validate(table, meta)["panel_b_gate_and_spread"]
    assert (checks["d50_range_um_min"], checks["d50_range_um_max"]) == (0.02, 0.18)
    assert checks["ratio_at_largest_comparison"] == 1.5
    spread = mlc.read_scoped("q3_spread")
    assert round(float(spread["d50_range_across_loadings_um"].max()), 2) == 0.18


@corpus
def test_no_invariance_or_difference_verdict_is_issued():
    _, meta = si.build_source_data()
    pb = meta["panel_b"]
    assert pb["verdict_issued"] is False
    assert "not particle mass" in pb["interpretation"]
    assert "not independent of the" in pb["interpretation"]
    assert "no error term" in pb["interpretation"]


# 6 ── caption wording ────────────────────────────────────────────────────────────────────────

@corpus
def test_caption_is_descriptive_and_disclaims_the_optical_coordinates(tmp_path):
    si.main(["--output-dir", str(tmp_path), "--formats", "pdf"])
    caption = " ".join((tmp_path / f"{si.STEM}_caption.md").read_text().split())
    low = caption.lower()
    assert "descriptive" in low
    assert "technical replicates" in low
    assert "not used as confirmation of it" in low
    assert "scattering-weighted particle coordinate" in low
    assert "paqxos-inverted relative composition" in low
    assert "not particle mass" in low
    assert "not an independent modality" in low
    assert "neither panel is an independent check" in low
    for banned in si.BANNED_VERDICTS:
        assert banned not in low, banned
    for banned in ("external validation", "confirms the model", "independent confirmation",
                   "robust"):
        assert banned not in low, banned


@corpus
def test_a_verdict_word_in_the_caption_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "CAPTION",
                        si.CAPTION + "\nThe size path was invariant with loading.\n")
    with pytest.raises(ValueError, match="issues a verdict"):
        si.main(["--output-dir", str(tmp_path), "--formats", "pdf"])


# 7 ── outputs and provenance ─────────────────────────────────────────────────────────────────

@corpus
def test_all_six_outputs_are_written(tmp_path):
    si.main(["--output-dir", str(tmp_path), "--formats", "pdf,png,svg"])
    for suffix in (".pdf", ".png", ".svg", "_source_data.csv",
                   "_provenance.json", "_caption.md"):
        path = tmp_path / f"{si.STEM}{suffix}"
        assert path.exists() and path.stat().st_size > 0, suffix


@corpus
def test_provenance_records_the_gate_and_the_scope_flags(tmp_path):
    si.main(["--output-dir", str(tmp_path), "--formats", "pdf"])
    prov = json.loads((tmp_path / f"{si.STEM}_provenance.json").read_text())
    assert set(prov["sources"]) == set(mlc.SOURCES)
    assert prov["scope"]["substudy"] == mlc.MANUSCRIPT_SUBSTUDY
    assert prov["scope"]["is_mass_measurement"] is False
    assert prov["scope"]["independent_of_ld_acquisition"] is False
    assert prov["scope"]["used_as_confirmation_of_mass_result"] is False
    assert prov["scope"]["verdict_issued_on_size_paths"] is False
    assert prov["career_artifacts_used"] is False
    assert "TAIL_MAX_PCT" in prov["reliability_gate"]["source"]
    assert prov["git_commit"]


@corpus
def test_the_analysis_figure_is_not_overwritten(tmp_path):
    si.main(["--output-dir", str(tmp_path), "--formats", "pdf"])
    assert not (tmp_path / "copt_loading.png").exists()
    prov = json.loads((tmp_path / f"{si.STEM}_provenance.json").read_text())
    assert "copt_loading" in prov["does_not_overwrite"]


@corpus
def test_source_data_carries_every_plotted_value_with_its_artifact(tmp_path):
    si.main(["--output-dir", str(tmp_path), "--formats", "pdf"])
    table = pd.read_csv(tmp_path / f"{si.STEM}_source_data.csv")
    assert table["y_value"].notna().all()
    assert set(table["source_artifact"]) == {"copt_loading_runs.csv",
                                             "copt_loading_q3_matched_extent.csv"}


@corpus
def test_no_panel_clips_a_drawn_value_or_error_bar():
    table, meta = si.build_source_data()
    for panel, result in si.validate(table, meta)["axis_limits_do_not_clip"].items():
        assert result["ok"], (panel, result)
