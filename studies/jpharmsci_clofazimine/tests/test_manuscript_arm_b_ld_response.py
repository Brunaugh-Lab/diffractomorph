"""Guardrails for the main dissolution-medium polysorbate laser-diffraction figure.

This figure makes LD the primary result rather than a corroboration of UV, so the tests pin two
things hard. First, the arithmetic: every displayed summary must reproduce the published Arm B
tables, and panel A must reproduce the frozen ΣI export, which in turn must reproduce the
published KWW fits. Second, the boundary: four independent preparations per condition with
technical runs nested inside, spreads that are between-preparation SDs, no UV / forward-model /
solubility / normalised-rate field anywhere in the artifact, no coarse tail above 15 µm, and no
mass-equivalence or q3 verdict in the caption.
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

import manuscript_arm_b_ld_response as fig            # noqa: E402


def _base():
    try:
        return fig.arm_b_root()
    except Exception:
        return None


BASE = _base()


def _have_sources() -> bool:
    return BASE is not None and all((BASE / v).exists() for v in fig.SOURCES.values())


corpus = pytest.mark.skipif(not _have_sources(), reason="Arm B analysis outputs not present")


@pytest.fixture(scope="module")
def built():
    return fig.build_source_data(BASE)


@pytest.fixture(scope="module")
def checked(built):
    table, meta = built
    return table, meta, fig.validate(table, meta, BASE)


# 1 ── four panels, 2x2 ───────────────────────────────────────────────────────────────────────

@corpus
def test_exactly_four_panels_are_produced(built):
    table, _ = built
    assert set(table["panel"]) == {"A", "B", "C", "D"}


def test_renderer_creates_a_two_by_two_layout():
    import inspect
    body = inspect.getsource(fig.render)
    assert "add_gridspec(2, 2" in body
    assert '(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")' in body


# 2 ── four independent preparations per condition, technical runs nested ────────────────────

@corpus
def test_every_panel_rests_on_four_preparations_per_condition(checked):
    _, _, checks = checked
    result = checks["four_preparations_per_condition"]
    assert result["ok"], result
    for panel, per_condition in result["per_panel"].items():
        assert set(per_condition.values()) == {4}, (panel, per_condition)


@corpus
def test_technical_runs_are_nested_not_independent(built):
    table, meta = built
    assert meta["aggregation"]["independent_unit"] == "suspension preparation"
    assert meta["aggregation"]["technical_runs_are_not_independent_replicates"] is True
    # the nesting is real and uneven: preparations carry 1, 2 or 3 technical runs, so pooling
    # runs as independent observations would silently weight the preparations unequally
    prep_rows = table[table["kind"].isin(["prep_trajectory", "prep_point", "prep_path"])]
    assert sorted(int(v) for v in prep_rows["n_technical_reps"].unique()) == [1, 2, 3]
    # condition rows record the nine runs behind them, but the summary is over four preparations
    condition_rows = table[table["kind"].eq("condition_mean") & table["panel"].eq("B")]
    assert set(int(v) for v in condition_rows["n_technical_reps"]) == {9}
    assert set(int(v) for v in condition_rows["n_preps"]) == {4}


@corpus
def test_every_drawn_spread_is_a_between_preparation_sd(checked):
    _, _, checks = checked
    result = checks["spread_is_between_preparation_sd"]
    assert result["ok"], result
    assert result["n_preps_behind_each_sd"] == [4]
    assert "never a pooled run SD and never a SEM" in result["definition"]


@corpus
def test_no_spread_is_attached_to_a_preparation_level_row(built):
    table, _ = built
    prep_rows = table[table["kind"].isin(["prep_trajectory", "prep_point", "prep_path"])]
    assert prep_rows["y_sd"].isna().all()


# 3 ── panel A reproduces the frozen export and the published fits ───────────────────────────

@corpus
def test_panel_a_condition_mean_is_the_equal_weight_mean_of_four_preparations(checked):
    _, _, checks = checked
    result = checks["panel_a_equal_weight_over_preparations"]
    assert result["ok"], result
    assert result["max_abs_mean_deviation_pp"] < 1e-9
    assert result["max_abs_sd_deviation_pp"] < 1e-9


@corpus
def test_the_frozen_trajectory_export_reproduces_the_published_kww_fits(built):
    _, meta = built
    cross = meta["panel_a"]["upstream_cross_check"]
    assert cross["ok"]
    assert cross["n_runs_checked"] == 27
    assert cross["max_abs_tau_deviation_min"] < 1e-9


@corpus
def test_panel_a_signal_is_the_raw_unsubtracted_angular_sum(built):
    _, meta = built
    definition = meta["panel_a"]["signal_definition"].lower()
    assert "measured value" in definition
    assert "not background" in definition and "σref" in definition
    assert meta["panel_a"]["is_dissolved_mass"] is False


@corpus
def test_panel_a_is_normalized_to_the_fitted_back_extrapolated_start(built):
    table, meta = built
    assert "back-extrapolated starting signal" in meta["panel_a"]["normalization"]
    assert set(table[table["panel"].eq("A")]["y_quantity"]) == {
        "angular_signal_pct_of_fitted_start"}


@corpus
def test_the_display_window_is_declared_with_the_omitted_regions_measured_range(built):
    table, meta = built
    a = meta["panel_a"]
    assert a["display_window_min"] == [0.0, fig.DISPLAY_WINDOW_MIN]
    assert a["acquisition_window_min"][1] > fig.DISPLAY_WINDOW_MIN
    omitted = a["omitted_region"]
    # the omitted region is described by a measured range, never asserted to be flat
    assert round(omitted["max_condition_mean_range_pp"], 1) == 4.6
    assert "NOT a claim that the region is flat" in omitted["described_as"]
    assert "flat" not in " ".join(str(v) for v in a.values()).lower().replace(
        "not a claim that the region is flat", "")
    # the omitted record is still in the source data, flagged
    panel_a = table[table["panel"].eq("A")]
    assert panel_a["displayed_in_figure"].any() and not panel_a["displayed_in_figure"].all()


# 4 ── panels B, C, D reproduce the published endpoints ──────────────────────────────────────

@corpus
def test_published_endpoints_are_reproduced_exactly(checked):
    _, _, checks = checked
    for condition, result in checks["published_endpoints"].items():
        assert result["ok"], (condition, result)


@corpus
def test_the_cross_check_values(checked):
    _, _, checks = checked
    endpoints = checks["published_endpoints"]
    for condition, expected in (("0.5x CMC", (2.1384, 0.2999)), ("1.0x CMC", (2.0222, 0.2634)),
                                ("10x CMC", (1.4909, 0.0981))):
        got = (endpoints[condition]["mean_relax_min"], endpoints[condition]["mean_relax_sd_min"])
        assert got == expected, (condition, got)
    assert [endpoints[c]["copt_loss_pct"] for c in fig.CONDITIONS] == [57.5, 63.9, 71.1]
    assert (endpoints["0.5x CMC"]["d50_um_g0.2"], endpoints["0.5x CMC"]["d50_um_g0.8"]) == (3.3317, 3.1342)
    assert (endpoints["1.0x CMC"]["d50_um_g0.2"], endpoints["1.0x CMC"]["d50_um_g0.8"]) == (3.3937, 3.1616)
    assert (endpoints["10x CMC"]["d50_um_g0.2"], endpoints["10x CMC"]["d50_um_g0.8"]) == (3.3486, 3.1583)


@corpus
def test_panel_b_condition_summary_is_preparation_first(checked):
    _, _, checks = checked
    assert checks["panel_b_summary_is_prep_first"]["ok"]


@corpus
def test_panel_b_shows_the_prespecified_mean_relaxation_time_not_tau(built, checked):
    table, meta, checks = *built, checked[2]
    assert checks["panel_b_shows_the_prespecified_descriptor"]["ok"]
    assert meta["panel_b"]["descriptor_is_prespecified_primary"] is True
    assert meta["panel_b"]["descriptor"].startswith("mean relaxation time")
    b = table[table["panel"].eq("B")]
    assert set(b["y_quantity"]) == {"mean_relaxation_time_min"}
    assert set(b["source_column"]) == {"mean_relax_min",
                                       "mean_relax_min_mean,mean_relax_min_sd"}
    # tau is available upstream and deliberately not drawn
    published = pd.read_csv(BASE / fig.SOURCES["kww_by_condition"])
    assert "tau_min_mean" in published.columns
    assert not any("tau" in str(v) for v in b["source_column"])
    assert "tau" not in " ".join(str(v) for v in b["y_quantity"]).lower()


@corpus
def test_panel_c_uses_the_primary_filtered_48h_pipeline_copt_path(built):
    table, meta = built
    assert meta["panel_c"]["partition_path"].startswith("filtered_48h_pipeline_copt")
    assert "filtered_48h_pipeline_copt" in fig.SOURCES["partition_preps"]
    assert set(table[table["panel"].eq("C")]["y_quantity"]) == {"copt_fractional_loss_pct"}


@corpus
def test_panel_c_matches_the_published_condition_table(checked):
    _, _, checks = checked
    assert checks["panel_c_matches_published_condition_table"]["ok"]


@corpus
def test_panel_d_preparation_paths_reproduce_the_balanced_table(built):
    table, _ = built
    runs = pd.read_csv(BASE / fig.SOURCES["q3_runs"])
    balanced = pd.read_csv(BASE / fig.SOURCES["q3_balanced"])
    derived = (table[table["kind"].eq("prep_path")]
               .groupby(["condition", "x_value"], as_index=False)
               .agg(mean=("y_value", "mean"), sd=("y_value", "std")))
    merged = derived.merge(balanced, left_on=["condition", "x_value"],
                           right_on=["condition", "extent_g"])
    assert len(merged) == len(balanced)
    assert (merged["mean"] - merged["d50_um_mean"]).abs().max() < 1e-9
    assert (merged["sd"] - merged["d50_um_sd"]).abs().max() < 1e-9
    assert runs.groupby("condition")["unit"].nunique().eq(4).all()


@corpus
def test_panel_d_g_is_the_fraction_of_copt_lost(built):
    table, meta = built
    assert "fraction of Copt lost" in meta["panel_d"]["g_definition"]
    d = table[table["panel"].eq("D")]
    assert sorted(set(np.round(d["x_value"], 3))) == list(fig.G_GRID)
    assert set(d["x_quantity"]) == {"fraction_of_copt_lost_g"}


# 5 ── scope: no validation or interpretation quantity, no coarse tail ───────────────────────

@corpus
def test_no_uv_model_solubility_or_normalized_rate_field_enters_the_artifact(checked, tmp_path):
    _, _, checks = checked
    assert checks["scope"]["forbidden_fields_found"] == []
    fig.main(["--output-dir", str(tmp_path), "--formats", "pdf"])
    written = pd.read_csv(tmp_path / f"{fig.STEM}_source_data.csv")
    blob = (" ".join(str(v) for v in written.to_dict("list").values())
            + " " + " ".join(written.columns)).lower()
    assert [f for f in fig.FORBIDDEN_FIELDS if f in blob] == []


@corpus
def test_no_coarse_tail_field_or_value_above_15_um_is_plotted(checked):
    _, _, checks = checked
    assert checks["scope"]["coarse_tail_fields_found"] == []
    assert checks["scope"]["all_plotted_sizes_below_boundary"]
    assert checks["scope"]["max_plotted_q3_um"] < 15.0


@corpus
def test_the_coarse_tail_exists_upstream_and_is_deliberately_dropped(built):
    _, meta = built
    runs = pd.read_csv(BASE / fig.SOURCES["q3_runs"])
    assert "tail_frac_above_15um" in runs.columns      # the upstream table does carry it
    assert meta["panel_d"]["coarse_tail_displayed"] is False
    assert "reliable inversion range" in meta["panel_d"]["coarse_tail_reason"]


@corpus
def test_no_inferential_test_and_no_q3_verdict(built):
    _, meta = built
    assert meta["panel_b"]["inferential_test_shown"] is False
    assert meta["panel_d"]["verdict_issued"] is False


# 5b ── the claim is narrowed to the contrasts four preparations separate ────────────────────

@corpus
def test_only_the_10x_relaxation_contrast_is_claimed_as_separated(checked):
    _, _, checks = checked
    sc = checks["supported_contrasts"]
    assert sc["relaxation_10x_separated_from_both_lower_levels"]
    assert sc["relaxation_lower_two_overlap"]


@corpus
def test_copt_loss_means_are_ordered_but_the_lower_two_are_not_resolved(checked):
    _, _, checks = checked
    sc = checks["supported_contrasts"]
    assert sc["copt_loss_means_ordered"]
    assert sc["copt_loss_lower_two_overlap"]
    assert sc["copt_loss_pairs"]["0.5x CMC vs 1.0x CMC"] is False


@corpus
def test_separation_is_declared_descriptive_not_inferential(built):
    _, meta = built
    for panel in ("panel_b", "panel_c"):
        separation = meta[panel]["separation"]
        assert "not an inferential test" in separation["basis"]
        assert "not evidence that the levels do not differ" in separation["caveat"]


# 6 ── caption wording ───────────────────────────────────────────────────────────────────────

@corpus
def test_caption_states_the_design_and_the_optical_boundary(tmp_path):
    fig.main(["--output-dir", str(tmp_path), "--formats", "pdf"])
    caption = " ".join((tmp_path / f"{fig.STEM}_caption.md").read_text().split())
    low = caption.lower()
    assert "four independent suspension preparations per condition" in low
    assert "averaged within preparation" in low
    assert "between-preparation sd" in low
    assert "never a pooled run sd and never a standard error" in low
    assert "not dissolved drug mass" in low
    assert "instrument-inverted relative composition" in low
    assert "not an independent modality" in low
    assert "no verdict of equivalence and no verdict of difference" in low
    assert "no inferential test is applied" in low
    # the narrowed claim, stated and bounded
    assert "earlier aggregate angular-signal disappearance than either lower level" in low
    assert "lie entirely below both lower levels" in low
    assert "means were ordered across the ladder" in low
    assert "overlap and that contrast is not resolved" in low
    assert "a limit of resolution rather than evidence that the levels do not differ" in low
    # despiking is stated, and stated as distinct from background subtraction
    assert "upward-despiked but **not** background-subtracted" in low
    assert "spans at most" in low and "percentage points" in low
    assert "flat" not in low
    assert "neither displayed nor interpreted" in low
    assert fig.check_wording(caption)["ok"]


def test_check_wording_rejects_a_bare_equivalence_claim():
    assert not fig.check_wording("The q3 paths were equivalent across conditions.")["ok"]
    assert not fig.check_wording("Copt loss is the dissolved fraction.")["ok"]


def test_check_wording_allows_the_same_words_inside_a_denial():
    assert fig.check_wording("This is fractional Copt loss and is not a dissolved fraction "
                             "or a particle-mass loss.")["ok"]
    assert fig.check_wording("No verdict of equivalence is made between the paths.")["ok"]


# 7 ── outputs and provenance ────────────────────────────────────────────────────────────────

@corpus
def test_all_six_outputs_are_written(tmp_path):
    fig.main(["--output-dir", str(tmp_path), "--formats", "pdf,png,svg"])
    for suffix in (".pdf", ".png", ".svg", "_source_data.csv",
                   "_provenance.json", "_caption.md"):
        path = tmp_path / f"{fig.STEM}{suffix}"
        assert path.exists() and path.stat().st_size > 0, suffix


@corpus
def test_source_data_carries_every_displayed_point_with_its_artifact(tmp_path):
    fig.main(["--output-dir", str(tmp_path), "--formats", "pdf"])
    table = pd.read_csv(tmp_path / f"{fig.STEM}_source_data.csv")
    assert table["y_value"].notna().all() and table["x_value"].notna().all()
    assert table["source_artifact"].notna().all()
    assert set(table["source_artifact"]) == {
        "arm_b_angular_trajectories_preps.csv", "arm_b_angular_trajectories_conditions.csv",
        "angular_kww_by_date_condition.csv", "angular_kww_by_condition.csv",
        "arm_b_partition_preps.csv", "q3_matched_extent_runs.csv",
        "q3_matched_extent_balanced.csv"}
    # preparation-level values are present, not only condition summaries
    assert table["prep"].notna().sum() > table["prep"].isna().sum()


@corpus
def test_provenance_identifies_every_upstream_file_rule_and_scope_flag(tmp_path):
    fig.main(["--output-dir", str(tmp_path), "--formats", "pdf"])
    prov = json.loads((tmp_path / f"{fig.STEM}_provenance.json").read_text())
    assert set(prov["sources"]) == set(fig.SOURCES)
    assert set(prov["panel_sources"]) == {"A", "B", "C", "D"}
    assert prov["aggregation"]["independent_unit"] == "suspension preparation"
    assert prov["reliability_rules"]["q3_coarse_tail_displayed"] is False
    assert "filtered_48h_pipeline_copt" in prov["reliability_rules"]["partition_path"]
    kww_input = prov["reliability_rules"]["kww_input"]
    assert "upward-despiked" in kww_input and "NOT background-subtracted" in kww_input
    # despiking and background subtraction are recorded as distinct operations, and the
    # contradictory "cleaned: false / none — raw channels" block is gone
    conditioning = prov["signal_conditioning"]
    assert conditioning["upward_despiking"]["applied"] is True
    assert conditioning["background_subtraction"]["applied"] is False
    assert "despiking removes transient" in conditioning["distinction"]
    assert "optical" not in prov, "the coarse cleaned/not-cleaned block must not be emitted"
    assert "raw Copt and raw channels" not in json.dumps(prov)
    assert prov["scope"]["is_dissolved_mass"] is False
    assert prov["scope"]["verdict_issued_on_q3_paths"] is False
    assert prov["scope"]["inferential_test_shown"] is False
    for excluded in ("UV dissolved mass", "forward model", "saturation solubility",
                     "solubility-normalized coefficient", "coarse tail above 15 µm"):
        assert excluded in prov["scope"]["excluded"], excluded
    assert prov["git_commit"]


@corpus
def test_upstream_analysis_outputs_are_not_overwritten(tmp_path):
    before = {p: p.stat().st_mtime_ns for p in (BASE / "analysis").rglob("*.csv")}
    fig.main(["--output-dir", str(tmp_path), "--formats", "pdf"])
    after = {p: p.stat().st_mtime_ns for p in (BASE / "analysis").rglob("*.csv")}
    assert before == after


def test_missing_source_raises_rather_than_reconstructing(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not reconstruct"):
        fig._read("kww_by_condition", tmp_path)


# 8 ── axis limits ───────────────────────────────────────────────────────────────────────────

@corpus
def test_no_panel_clips_a_drawn_value_or_error_bar(checked):
    _, _, checks = checked
    for panel, result in checks["axis_limits_do_not_clip"].items():
        assert result["ok"], (panel, result)
