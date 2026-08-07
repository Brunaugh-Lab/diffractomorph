"""Guardrails for the revised Section 3.4.1 starting-suspension portability figure.

The revision's whole point is that the plotted trajectories carry the same evidential status as
the error bars beside them, so the tests pin that first: every drawn curve must be an out-of-fold
prediction, its fold parameters must match the leave-one-date-out artifact, and the all-data
artifact must be demonstrably NOT what is drawn. The remaining tests pin scope, aggregation, the
four displayed RMSE values, the narrative direction, and the absence of the extent candidates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))

import manuscript_starting_suspension_portability_revised as rv  # noqa: E402


def _have_sources() -> bool:
    try:
        from diffractomorph_pipeline.config import data_root
        return all((data_root() / v).exists() for v in rv.SOURCES.values())
    except Exception:
        return False


corpus = pytest.mark.skipif(not _have_sources(), reason="authoritative outputs not present")


# ── three panels ─────────────────────────────────────────────────────────────────────────────

@corpus
def test_the_source_data_carries_exactly_three_panels():
    table, _ = rv.build_source_data()
    assert set(table["panel"]) == {"0.01% w/v polysorbate 80", "0.03% w/v polysorbate 80",
                                   "held-out error"}


@corpus
def test_the_rendered_figure_has_exactly_three_axes(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(rv.plt, "close", lambda fig: captured.append(fig))
    table, meta = rv.build_source_data()
    rv.render(table, meta, tmp_path, formats=("pdf",))
    fig = captured[0]
    assert len(fig.axes) == 3
    assert [ax.get_title() for ax in fig.axes[:2]] == list(rv.PANEL.values())
    assert fig.axes[2].get_title() == "Held-out prediction error"


@corpus
def test_the_revised_stem_does_not_overwrite_the_original(tmp_path):
    assert rv.STEM == "Figure_starting_suspension_model_portability_revised"
    rv._emit(tmp_path, ("pdf",))
    assert (tmp_path / f"{rv.STEM}.pdf").exists()
    assert not (tmp_path / "Figure_starting_suspension_model_portability.pdf").exists()


# ── trajectories are out-of-fold, and say so ─────────────────────────────────────────────────

@corpus
def test_every_plotted_trajectory_is_an_out_of_fold_prediction():
    table, _ = rv.build_source_data()
    curves = table[table["kind"].eq("model_curve")]
    assert not curves.empty
    assert set(curves["prediction_basis"]) == {"out_of_fold"}
    # and the measured series are labelled as measurements, not predictions
    measured = table[table["kind"].isin(["date_mean", "cohort_mean"])]
    assert set(measured["prediction_basis"]) == {"measured"}


@corpus
def test_the_prediction_basis_of_every_row_is_explicitly_recorded():
    table, _ = rv.build_source_data()
    assert table["prediction_basis"].notna().all()
    assert set(table["prediction_basis"]) <= {"measured", "out_of_fold"}
    assert "all_data" not in set(table["prediction_basis"])


@corpus
def test_plotted_series_carry_the_fold_parameters_of_their_declared_model():
    from diffractomorph_pipeline.config import data_root
    oof = pd.read_csv(data_root() / rv.SOURCES["out_of_fold"])
    oof = oof[oof["cohort"].eq(rv.COHORT) & oof["prediction_basis"].eq("out_of_fold")]
    table, meta = rv.build_source_data()
    checks = rv.validate(table, meta)
    per_model = checks["trajectories_are_out_of_fold"]["per_model"]
    for model in rv.SHOWN_IN_TRAJECTORY:
        assert per_model[model]["match"], model
        artifact = sorted({round(float(v), 6) for v in oof[oof["model"].eq(model)]["rate_scale"]})
        assert per_model[model]["fold_rate_scales_artifact"] == artifact


@corpus
def test_the_selected_model_really_varies_between_folds_and_the_others_do_not():
    table, meta = rv.build_source_data()
    checks = rv.validate(table, meta)
    assert checks["selected_model_varies_between_folds"]["ok"]
    rates = checks["selected_model_varies_between_folds"]["fold_rate_scales"]
    assert len(rates) == 3 and min(rates) < 1.0 and max(rates) < 1.0   # all slower than base
    per_model = checks["trajectories_are_out_of_fold"]["per_model"]
    for model in ("injection_base", "injection_selected_rate"):
        assert per_model[model]["n_distinct_fold_values"] == 1, model
        assert per_model[model]["parameter_free"], model


@corpus
def test_the_all_data_artifact_is_not_what_is_plotted():
    """The published predictions artifact holds all-data fits; the figure must not draw them."""
    from diffractomorph_pipeline.config import data_root
    all_data = pd.read_csv(data_root() / rv.ARM_A
                           / "analysis/rate_refinement/rate_refinement_predictions.csv")
    all_data = all_data[all_data["cohort"].eq(rv.COHORT)
                        & all_data["model"].eq(rv.SELECTED_MODEL)]
    # one rate for every row => fitted on all dates at once
    assert all_data["rate_scale"].nunique() == 1
    single = float(all_data["rate_scale"].iloc[0])
    table, meta = rv.build_source_data()
    folds = rv.validate(table, meta)["selected_model_varies_between_folds"]["fold_rate_scales"]
    assert len(folds) > 1
    assert set(folds) != {round(single, 6)}
    # the figure never names the all-data artifact as a source
    assert "rate_refinement_predictions.csv" not in {Path(p).name for p in rv.SOURCES.values()}


@corpus
def test_the_out_of_fold_generator_reproduces_the_published_all_data_predictions():
    """The regeneration is auditable: its all-data arm must match the published artifact."""
    from diffractomorph_pipeline.config import data_root
    import arm_a_out_of_fold_predictions as gen
    oof = pd.read_csv(data_root() / rv.SOURCES["out_of_fold"])
    published = pd.read_csv(data_root() / rv.ARM_A
                            / "analysis/rate_refinement/rate_refinement_predictions.csv")
    checks = gen.verify_against_all_data_artifact(oof, published)
    assert checks["ok"]
    assert checks["max_abs_predicted_diff_pct"] < 1e-6
    assert checks["n_rows_compared"] == 4 * 18 * 7      # 4 models x 18 runs x 7 UV times


# ── panel C values and narrative direction ───────────────────────────────────────────────────

@corpus
def test_the_four_displayed_rmse_values_reproduce_the_artifact():
    from diffractomorph_pipeline.config import data_root
    cand = pd.read_csv(data_root() / rv.SOURCES["candidates"])
    cand = cand[cand["cohort"].eq(rv.COHORT)].set_index("model")["lodo_rmse_pct"]
    table, _ = rv.build_source_data()
    shown = table[table["kind"].eq("heldout_summary")].set_index("model_key")["value_pct"]
    assert len(shown) == 4
    for key, _, _ in rv.MODELS:
        assert np.isclose(float(shown[key]), float(cand[key])), key
        assert round(float(shown[key]), 1) == rv.EXPECTED_LODO[key], key
    assert rv.EXPECTED_LODO == {"injection_base": 8.1, "injection_selected_rate": 13.7,
                                "first_uv_anchor": 7.8, "anchored_rate": 5.5}


@corpus
def test_the_historical_correction_is_worse_than_the_unmodified_injection_model():
    table, meta = rv.build_source_data()
    checks = rv.validate(table, meta)
    assert checks["narrative"]["historical_correction_worse_than_injection_base"]
    shown = table[table["kind"].eq("heldout_summary")].set_index("model_key")["value_pct"]
    assert float(shown["injection_selected_rate"]) > float(shown["injection_base"])


@corpus
def test_the_slower_first_uv_aligned_model_has_the_lowest_held_out_error():
    table, meta = rv.build_source_data()
    checks = rv.validate(table, meta)
    assert checks["narrative"]["selected_is_lowest_of_the_four"]
    assert checks["narrative"]["slower_rate_beats_aligned_base_rate"]
    shown = table[table["kind"].eq("heldout_summary")].set_index("model_key")["value_pct"]
    assert float(shown["anchored_rate"]) == min(float(v) for v in shown)
    selected = table[table["kind"].eq("heldout_summary") & table["is_selected"].eq(True)]
    assert len(selected) == 1 and selected["model_key"].iloc[0] == "anchored_rate"


@corpus
def test_each_value_is_the_rms_over_its_held_out_dates():
    from diffractomorph_pipeline.config import data_root
    lodo = pd.read_csv(data_root() / rv.SOURCES["lodo"])
    lodo = lodo[lodo["cohort"].eq(rv.COHORT)]
    table, meta = rv.build_source_data()
    checks = rv.validate(table, meta)
    for key, _, _ in rv.MODELS:
        folds = lodo[lodo["model"].eq(key)]["held_post_first_rmse_pct"].dropna()
        assert checks[f"heldout_rmse_{key}"]["rms_matches_summary"], key
        assert checks[f"heldout_rmse_{key}"]["n_folds_used"] == 3, key
        assert not np.isclose(float(checks[f"heldout_rmse_{key}"]["value"]),
                              folds.mean(), atol=1e-6), key


@corpus
def test_the_two_timing_groups_are_assigned_correctly():
    table, _ = rv.build_source_data()
    fam = table[table["kind"].eq("heldout_summary")].set_index("model_key")["timing_family"]
    assert fam["injection_base"] == fam["injection_selected_rate"] == rv.INJECTION
    assert fam["first_uv_anchor"] == fam["anchored_rate"] == rv.ALIGNED


@corpus
def test_validate_raises_when_a_displayed_value_drifts():
    table, meta = rv.build_source_data()
    broken = table.copy()
    broken.loc[broken["model_key"].eq("injection_base")
               & broken["kind"].eq("heldout_summary"), "value_pct"] = 99.9
    with pytest.raises(ValueError, match="does not reproduce"):
        rv.validate(broken, meta)


# ── extent candidates absent ─────────────────────────────────────────────────────────────────

@corpus
def test_no_extent_model_series_appears_anywhere_in_the_figure():
    table, meta = rv.build_source_data()
    checks = rv.validate(table, meta)
    assert checks["extent_models_absent"]["ok"]
    present = set(table["model_key"].dropna())
    for excluded in rv.EXCLUDED_FROM_FIGURE:
        assert excluded not in present, excluded
    blob = " ".join(str(v) for v in table.to_dict("list").values()).lower()
    for token in ("participation", "extent"):
        assert token not in blob, token
    # ...while the artifacts still carry them, for Table S8
    from diffractomorph_pipeline.config import data_root
    cand = pd.read_csv(data_root() / rv.SOURCES["candidates"])
    assert set(rv.EXCLUDED_FROM_FIGURE) <= set(cand["model"])


@corpus
def test_the_caption_mentions_the_extent_result_without_plotting_it(tmp_path):
    rv._emit(tmp_path, ("pdf",))
    caption = " ".join((tmp_path / f"{rv.STEM}_caption.md").read_text().split())
    assert "a separate terminal extent did not improve held-out prediction" in caption
    assert "Table S8" in caption


# ── scope ────────────────────────────────────────────────────────────────────────────────────

def test_every_source_path_lives_inside_the_arm_a_study_tree():
    assert len(rv.SOURCES) == 4
    for key, rel in rv.SOURCES.items():
        assert str(rel).startswith(str(rv.ARM_A)), (key, rel)


def test_no_source_path_names_a_foreign_artifact():
    for key, rel in rv.SOURCES.items():
        low = str(rel).lower()
        for token in ("career", "objective1", "nist", "ph_dependent", "arm_b",
                      "copt", "q3", "angular", "psd_evolution"):
            assert token not in low, (key, token)


def test_the_module_reads_no_file_outside_its_declared_sources():
    src = Path(rv.__file__).read_text()
    assert src.count("read_csv") == 1
    assert "data_root() / SOURCES[key]" in src
    for banned in ("objective1_feasibility", "_ph_dependent_runs", "PH_STUDY",
                   "arm_b_cs", "arm_b_optical", "psd_", "forward_intensity"):
        assert banned not in src, banned


@corpus
def test_only_the_arm_a_starting_suspension_cohort_is_included():
    from diffractomorph_pipeline.config import data_root
    raw = pd.read_csv(data_root() / rv.SOURCES["candidates"])
    assert raw["cohort"].nunique() > 1
    assert "historical pH 4.0/4.5 fit cohort" in set(raw["cohort"])
    assert rv.COHORT == "antisolvent Tween 80 concentration transfer"
    table, meta = rv.build_source_data()
    assert set(table["study"]) == {"tween80_suspension_wetting_arm_a"}
    checks = rv.validate(table, meta)
    assert checks["no_foreign_input"]["ok"]
    assert checks["no_foreign_input"]["forbidden_tokens_found"] == []


@corpus
def test_no_optical_or_foreign_token_survives_into_the_source_data():
    table, _ = rv.build_source_data()
    blob = " ".join(str(v) for v in table.to_dict("list").values()).lower()
    for token in rv.FORBIDDEN_TOKENS:
        assert token not in blob, token
    for optical in ("kww", "relax", "channel", "mie", "scatter", "intensity"):
        assert optical not in blob, optical


@corpus
def test_validate_raises_when_a_foreign_row_reaches_the_table():
    table, meta = rv.build_source_data()
    foreign = pd.DataFrame([{"panel": "held-out error", "kind": "note",
                             "series": "arm_b micelle cohort",
                             "study": "tween80_suspension_wetting_arm_a"}])
    with pytest.raises(ValueError, match="foreign"):
        rv.validate(pd.concat([table, foreign], ignore_index=True), meta)


# ── aggregation ──────────────────────────────────────────────────────────────────────────────

def test_preparation_dates_are_weighted_equally_regardless_of_run_count():
    frame = pd.DataFrame({"model": "m", "time_min": 2.0,
                          "date": ["d1"] * 9 + ["d2"], "v": [10.0] * 9 + [20.0]})
    out = rv._equal_weight_over_dates(frame, "v", ["model", "time_min"])
    assert np.isclose(out.loc[0, "value_pct"], 15.0)     # not 11.0
    assert out.loc[0, "n_dates"] == 2


@corpus
def test_three_preparation_dates_and_three_nested_runs_per_condition():
    table, meta = rv.build_source_data()
    checks = rv.validate(table, meta)
    assert checks["preparation_dates"]["n_dates_per_condition"] == {"0.01": 3, "0.03": 3}
    assert checks["preparation_dates"]["nested_runs_per_date"] == [3]
    assert meta["trajectories"]["nested_runs_per_date"] == [3]
    dates = table[table["kind"].eq("date_mean")]
    assert len(dates) == 2 * 3 * 7


@corpus
def test_cohort_means_and_model_curves_are_both_date_first_aggregations():
    table, meta = rv.build_source_data()
    checks = rv.validate(table, meta)
    assert checks["date_first_aggregation"]["ok"]
    assert checks["date_first_aggregation"]["run_weighting_used"] is False
    # trajectories are averaged over the same three preparation dates
    curves = table[table["kind"].eq("model_curve")]
    assert (curves["n_dates"] == 3).all()
    assert "within preparation date" in meta["trajectories"]["aggregation"]


@corpus
def test_the_two_conditions_did_not_differ_in_mean_dissolved_fraction():
    table, meta = rv.build_source_data()
    checks = rv.validate(table, meta)
    similar = checks["condition_means_are_similar"]
    assert similar["abs_difference_pct"] < 3.0
    # the difference that IS real is between-date variability
    sd = similar["between_date_sd_pct"]
    assert sd["0.03"] > sd["0.01"]


@corpus
def test_caption_does_not_claim_a_condition_difference_in_mean_dissolved_fraction(tmp_path):
    rv._emit(tmp_path, ("pdf",))
    caption = " ".join((tmp_path / f"{rv.STEM}_caption.md").read_text().split())
    assert "similar cohort-average dissolved fraction" in caption
    assert "what differed was between-date variability" in caption
    for banned in ("dissolved more", "dissolved less", "higher dissolved fraction",
                   "increased dissolution", "reduced dissolution"):
        assert banned not in caption.lower(), banned


# ── clipping and outputs ─────────────────────────────────────────────────────────────────────

@corpus
def test_no_plotted_value_is_clipped_by_the_common_mass_axis():
    table, meta = rv.build_source_data()
    checks = rv.validate(table, meta)
    assert checks["axis_limits_do_not_clip"]["ok"]
    lo, hi = rv._limits(table)
    drawn = table[table["panel"].isin(rv.PANEL.values())]["value_pct"]
    assert drawn.min() >= lo and drawn.max() <= hi
    assert lo < drawn.min() and drawn.max() < hi          # genuine padding, not a touching edge


@corpus
def test_both_trajectory_panels_share_one_mass_axis(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(rv.plt, "close", lambda fig: captured.append(fig))
    table, meta = rv.build_source_data()
    rv.render(table, meta, tmp_path, formats=("pdf",))
    axA, axB = captured[0].axes[0], captured[0].axes[1]
    assert axA.get_ylim() == axB.get_ylim()
    assert axA.get_xlim() == axB.get_xlim()


@corpus
def test_all_six_artifacts_are_written_under_the_revised_stem(tmp_path):
    rv._emit(tmp_path, ("pdf", "png", "svg"))
    for suffix in (".pdf", ".png", ".svg", "_source_data.csv",
                   "_provenance.json", "_caption.md"):
        assert (tmp_path / f"{rv.STEM}{suffix}").stat().st_size > 0
    csv = pd.read_csv(tmp_path / f"{rv.STEM}_source_data.csv")
    assert set(csv["kind"]) == {"date_mean", "cohort_mean", "model_curve", "heldout_summary"}


@corpus
def test_provenance_records_the_out_of_fold_basis_and_the_scope(tmp_path):
    rv._emit(tmp_path, ("pdf",))
    prov = pd.read_json(tmp_path / f"{rv.STEM}_provenance.json", typ="series")
    basis = prov["prediction_basis"]
    assert basis["trajectories"].startswith("out_of_fold")
    assert basis["generator"] == "arm_a_out_of_fold_predictions.py"
    assert "NOT drawn" in basis["all_data_artifact_not_plotted"]
    assert len(basis["fold_rate_scales"]["anchored_rate"]) == 3
    assert prov["career_artifacts_used"] is False
    assert prov["design"]["n_panels"] == 3
    assert prov["design"]["models_excluded_from_figure"] == list(rv.EXCLUDED_FROM_FIGURE)
    assert prov["aggregation_hierarchy"]["independent_unit"] == "preparation date"
    assert prov["git_commit"]


# ── claim boundary ───────────────────────────────────────────────────────────────────────────

@corpus
def test_caption_leads_with_the_result_and_keeps_the_boundaries(tmp_path):
    rv._emit(tmp_path, ("pdf",))
    caption = " ".join((tmp_path / f"{rv.STEM}_caption.md").read_text().split())
    assert ("Changing the starting-suspension preparation reversed the direction of the "
            "empirical correction required by the forward model") in caption
    assert "All three trajectories are out-of-fold predictions" in caption
    assert "not external validation" in caption
    assert "rather than a universal rate constant" in caption
    assert ("not attributed here to polysorbate concentration, wetting, surface area, "
            "deaggregation, or a participating particle fraction") in caption
    assert "No laser-diffraction or other optical measurement entered this figure" in caption


def test_module_never_affirms_external_validation_or_a_universal_constant():
    src = " ".join(Path(rv.__file__).read_text().split()).lower()
    for phrase in ("external validation", "universal rate constant", "independently validated"):
        start = 0
        while (i := src.find(phrase, start)) != -1:
            window = src[max(0, i - 70):i]
            assert any(n in window for n in ("not ", "never ", "no ", "rather than ", "cannot")), \
                src[max(0, i - 70):i + len(phrase) + 20]
            start = i + 1


def test_panels_carry_no_mechanistic_explanation_text():
    import inspect
    body = inspect.getsource(rv._trajectory_axis) + inspect.getsource(rv._heldout_axis)
    for banned in ("wetting", "surface area", "deaggregation", "participating particle",
                   "micelle", "aicc", "significant", "p ="):
        assert banned.lower() not in body.lower(), banned
    assert "suptitle" not in inspect.getsource(rv.render)


# ── artifact discipline ──────────────────────────────────────────────────────────────────────

def test_missing_source_raises_with_a_pointer_to_the_generator(tmp_path, monkeypatch):
    monkeypatch.setattr(rv, "data_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="arm_a_out_of_fold_predictions"):
        rv._read("out_of_fold")


def test_empty_source_raises(tmp_path, monkeypatch):
    target = tmp_path / rv.SOURCES["candidates"]
    target.parent.mkdir(parents=True)
    target.write_text("cohort,model,lodo_rmse_pct\n")
    monkeypatch.setattr(rv, "data_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="is empty"):
        rv._read("candidates")


# ── panel C: paired two-row comparison ───────────────────────────────────────────────────────

def _panel_c(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(rv.plt, "close", lambda fig: captured.append(fig))
    table, meta = rv.build_source_data()
    rv.render(table, meta, tmp_path, formats=("pdf",))
    return captured[0], captured[0].axes[2]


@corpus
def test_panel_c_has_two_rows_labelled_by_where_the_prediction_starts(tmp_path, monkeypatch):
    _, axC = _panel_c(tmp_path, monkeypatch)
    ticks = [t.get_text() for t in axC.get_yticklabels()]
    assert len(ticks) == 2
    assert [" ".join(t.split()) for t in ticks] == ["Prediction starts at injection",
                                                    "Prediction aligned to first UV"]
    assert len(rv.PAIRS) == 2


@corpus
def test_panel_c_carries_no_free_floating_group_headings(tmp_path, monkeypatch):
    _, axC = _panel_c(tmp_path, monkeypatch)
    # the empty-text entries are the two connector annotations, not labels
    inside = [" ".join(t.get_text().split()) for t in axC.texts if t.get_text()]
    # exactly the four endpoint labels, nothing else
    assert len(inside) == 4
    for text in inside:
        assert not text.startswith("Prediction "), text
    assert not hasattr(rv, "HEADER_Y")


@corpus
def test_panel_c_labels_each_endpoint_with_its_model_name_and_one_decimal_value(
        tmp_path, monkeypatch):
    _, axC = _panel_c(tmp_path, monkeypatch)
    inside = [" ".join(t.get_text().split()) for t in axC.texts]
    for key, expected in rv.EXPECTED_LODO.items():
        name = rv.ENDPOINT[key]["name"]
        hit = [t for t in inside if name in t]
        assert len(hit) == 1, (name, inside)
        assert f"{expected:.1f}" in hit[0], hit


@corpus
def test_panel_c_arrows_run_base_to_modified_in_opposite_directions(tmp_path, monkeypatch):
    """The panel's whole point: one change increases error, the other decreases it."""
    _, axC = _panel_c(tmp_path, monkeypatch)
    arrows = [a for a in axC.texts + list(axC.artists) if False]      # annotations live below
    anns = [c for c in axC.get_children()
            if isinstance(c, rv.matplotlib.text.Annotation)]
    assert len(anns) == 2
    spans = sorted((float(a.xyann[0]), float(a.xy[0])) for a in anns)
    (aligned_from, aligned_to), (inj_from, inj_to) = spans
    assert inj_to > inj_from       # injection row: historical correction increases error
    assert aligned_to < aligned_from   # aligned row: slower rate decreases error
    assert round(inj_from, 1) == 8.1 and round(inj_to, 1) == 13.7
    assert round(aligned_from, 1) == 7.8 and round(aligned_to, 1) == 5.5


@corpus
def test_panel_c_uses_the_established_colours(tmp_path, monkeypatch):
    assert rv.ENDPOINT["injection_base"]["color"] == rv.DARK
    assert rv.ENDPOINT["first_uv_anchor"]["color"] == rv.DARK
    assert rv.ENDPOINT["injection_selected_rate"]["color"] == rv.HISTORICAL == "#D55E00"
    assert rv.ENDPOINT["anchored_rate"]["color"] == rv.ACCENT == "#0072B2"
    # the trajectory panels use the same two accent colours for the same two models
    assert rv.MODEL_STYLE["injection_selected_rate"]["color"] == rv.HISTORICAL
    assert rv.MODEL_STYLE["anchored_rate"]["color"] == rv.ACCENT
    # only the selected model is emphasised
    assert rv.ENDPOINT["anchored_rate"]["size"] > rv.ENDPOINT["first_uv_anchor"]["size"]


@corpus
def test_panel_c_keeps_lower_is_better_in_the_axis_label(tmp_path, monkeypatch):
    _, axC = _panel_c(tmp_path, monkeypatch)
    label = " ".join(axC.get_xlabel().split())
    assert "lower is better" in label
    assert "Leave-one-preparation-date-out RMSE" in label


@corpus
def test_panel_c_gridlines_are_thin(tmp_path, monkeypatch):
    _, axC = _panel_c(tmp_path, monkeypatch)
    widths = {round(float(g.get_linewidth()), 2) for g in axC.get_xgridlines()
              if g.get_visible()}
    assert widths and max(widths) <= 0.5, widths
    assert not any(g.get_visible() for g in axC.get_ygridlines())


# ── the cropping guard ───────────────────────────────────────────────────────────────────────

@corpus
def test_no_label_is_cropped_by_the_tight_bounding_box(tmp_path, monkeypatch):
    fig, _ = _panel_c(tmp_path, monkeypatch)
    rv._assert_no_text_is_cropped(fig)        # must not raise for the shipped geometry


@corpus
def test_the_guard_catches_a_panel_too_narrow_for_its_axis_label(tmp_path, monkeypatch):
    """The guard fails when the save bounding box would omit rendered text.

    Matplotlib versions differ in whether ``get_tightbbox`` automatically expands around a
    narrowed axes, so inject the clipped save box directly and test the invariant itself.
    """
    from matplotlib.transforms import Bbox
    fig, axC = _panel_c(tmp_path, monkeypatch)
    axC.set_position([0.80, 0.42, 0.06, 0.50])       # far too narrow for the two-line label
    monkeypatch.setattr(fig, "get_tightbbox", lambda renderer: Bbox.from_extents(0, 0, 0.5, 4))
    with pytest.raises(ValueError, match="cropped"):
        rv._assert_no_text_is_cropped(fig)


# ── page layout: even spacing and a shared A/B legend ────────────────────────────────────────

def _laid_out(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(rv.plt, "close", lambda fig: captured.append(fig))
    table, meta = rv.build_source_data()
    rv.render(table, meta, tmp_path, formats=("pdf",))
    fig = captured[0]
    fig.canvas.draw()
    return fig


@corpus
def test_the_three_panels_are_evenly_spaced_across_the_page(tmp_path, monkeypatch):
    fig = _laid_out(tmp_path, monkeypatch)
    a, b, c = [ax.get_position() for ax in fig.axes]
    gap_ab = (b.x0 - a.x1) * rv.FIG_W
    gap_bc = (c.x0 - b.x1) * rv.FIG_W
    assert abs(gap_ab - gap_bc) < 0.02, (gap_ab, gap_bc)   # inches
    assert gap_ab > 0.8, gap_ab                            # a real gutter, not a hairline
    # the two trajectory panels stay the same size as each other
    assert abs(a.width - b.width) < 1e-9


@corpus
def test_panel_c_is_wide_enough_for_its_two_line_axis_label(tmp_path, monkeypatch):
    fig = _laid_out(tmp_path, monkeypatch)
    axC = fig.axes[2]
    renderer = fig.canvas.get_renderer()
    label = axC.xaxis.label.get_window_extent(renderer)
    axes_px = axC.get_position().width * fig.bbox.x1
    assert label.x1 - label.x0 <= axes_px + 1, (label.x1 - label.x0, axes_px)


@corpus
def test_the_shared_legend_is_centred_under_panels_a_and_b(tmp_path, monkeypatch):
    fig = _laid_out(tmp_path, monkeypatch)
    a, b, c = [ax.get_position() for ax in fig.axes]
    assert len(fig.legends) == 1                    # one key, not one per panel
    box = fig.legends[0].get_window_extent(
        fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    centre = (box.x0 + box.x1) / 2
    assert abs(centre - (a.x0 + b.x1) / 2) * rv.FIG_W < 0.05      # inches
    assert box.x1 < c.x0                            # never runs under panel C
    # and it is not centred on the whole figure, which would read as a figure-wide key
    assert abs(centre - 0.5) > 0.05


@corpus
def test_the_legend_covers_every_series_drawn_in_the_trajectory_panels(tmp_path, monkeypatch):
    fig = _laid_out(tmp_path, monkeypatch)
    labels = {t.get_text() for t in fig.legends[0].get_texts()}
    for model in rv.SHOWN_IN_TRAJECTORY:
        assert rv.TRAJECTORY_LABEL[model] in labels, model
    assert "UV measurement (mean of dates)" in labels
    assert "Individual preparation date" in labels
    assert len(labels) == len(rv.SHOWN_IN_TRAJECTORY) + 2


@corpus
def test_the_legend_guard_rejects_an_off_centre_legend(tmp_path, monkeypatch):
    from matplotlib.transforms import Bbox
    fig = _laid_out(tmp_path, monkeypatch)
    a, b, c = [ax.get_position() for ax in fig.axes]
    shifted_a = Bbox.from_extents(a.x0 + 0.20, a.y0, a.x1 + 0.20, a.y1)
    shifted_b = Bbox.from_extents(b.x0 + 0.20, b.y0, b.x1 + 0.20, b.y1)
    with pytest.raises(ValueError, match="off the A/B centre line"):
        rv._assert_legend_is_centred_under(fig, fig.legends[0], shifted_a, shifted_b, c)


@corpus
def test_the_legend_guard_rejects_a_legend_running_under_panel_c(tmp_path, monkeypatch):
    from matplotlib.transforms import Bbox
    fig = _laid_out(tmp_path, monkeypatch)
    a, b, c = [ax.get_position() for ax in fig.axes]
    encroached = Bbox.from_extents(0.10, c.y0, c.x1, c.y1)     # C starts left of the legend
    with pytest.raises(ValueError, match="runs under panel C"):
        rv._assert_legend_is_centred_under(fig, fig.legends[0], a, b, encroached)
