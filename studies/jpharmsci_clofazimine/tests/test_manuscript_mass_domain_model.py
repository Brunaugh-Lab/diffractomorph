"""Guardrails for the Section 3.2 mass-domain manuscript figure.

This figure carries claims that are easy to overstate, so the tests pin the claim boundary as
well as the arithmetic: only the pH-dependent study may appear, aggregation must be
preparation-date-first with dates weighted equally, the aligning observation must not enter the
residuals, the 2.2x multiplier must stay frozen for pH 5.0, and neither figure nor caption may
call pH 5.0 external validation or 2.2 a transferable kinetic constant.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))

import manuscript_mass_domain_model as mm  # noqa: E402


def _have_sources() -> bool:
    try:
        from diffractomorph_pipeline.config import data_root
        return all((data_root() / v).exists() for v in mm.SOURCES.values())
    except Exception:
        return False


corpus = pytest.mark.skipif(not _have_sources(), reason="authoritative outputs not present")


# ── scope: pH-dependent study only ───────────────────────────────────────────────────────────

@corpus
def test_only_the_ph_dependent_study_appears_in_the_source_data():
    table, _ = mm.build_source_data()
    assert set(table["study"]) == {"ph_dependent_dissolution_study"}
    assert set(table["panel"]) == {"pH 4.0", "pH 4.5", "pH 5.0", "held-out RMSE"}
    conditions = set(table["condition"].dropna())
    assert conditions <= {"pH 4.0", "pH 4.5", "pH 5.0"}


@corpus
def test_antisolvent_medium_and_loading_cohorts_are_absent():
    table, _ = mm.build_source_data()
    blob = " ".join(str(v) for v in table.to_dict("list").values()).lower()
    for foreign in ("antisolvent", "tween", "cmc", "micelle", "copt", "loading",
                    "0.01", "0.03", "arm_a", "arm_b"):
        assert foreign not in blob, foreign


@corpus
def test_only_the_ph_fit_cohort_is_admitted_from_the_shared_artifacts():
    # rate_refinement carries a Tween cohort in the same files; it must never be drawn on.
    from diffractomorph_pipeline.config import data_root
    raw = pd.read_csv(data_root() / mm.SOURCES["predictions"])
    assert raw["cohort"].nunique() > 1              # the artifact really does mix cohorts
    assert mm.FIT_COHORT == "historical pH 4.0/4.5 fit cohort"
    table, meta = mm.build_source_data()
    assert meta["trajectories"]["conditions"] == ["pH 4.0", "pH 4.5"]


# ── aggregation ──────────────────────────────────────────────────────────────────────────────

def test_technical_runs_are_averaged_within_date_before_condition_aggregation():
    # Two dates, unequal run counts. Date means are 10 and 20, so the equally weighted
    # condition mean is 15. A run-weighted mean would be 12.5 and must not appear.
    frame = pd.DataFrame({
        "condition": "pH 4.0", "time_min": 2.0,
        "date": ["d1", "d1", "d1", "d2"], "rep": [1, 2, 3, 1],
        "v": [8.0, 10.0, 12.0, 20.0]})
    out = mm._date_balanced(frame, "v", ["condition", "time_min"])
    assert np.isclose(out.loc[0, "value_pct"], 15.0)
    assert not np.isclose(out.loc[0, "value_pct"], 12.5)
    assert out.loc[0, "n_dates"] == 2 and out.loc[0, "n_runs"] == 4


def test_preparation_dates_receive_equal_weight_regardless_of_run_count():
    many = pd.DataFrame({"condition": "c", "time_min": 1.0,
                         "date": ["d1"] * 9 + ["d2"], "rep": list(range(9)) + [1],
                         "v": [0.0] * 9 + [100.0]})
    out = mm._date_balanced(many, "v", ["condition", "time_min"])
    assert np.isclose(out.loc[0, "value_pct"], 50.0)     # not 10.0


@corpus
def test_variability_is_between_preparation_date_means():
    table, _ = mm.build_source_data()
    obs = table[table["kind"].eq("observation") & table["panel"].eq("pH 4.0")]
    assert (obs["n_dates"] == 3).all()
    assert (obs["n_runs"] == 9).all()
    assert obs["sd_pct"].notna().all() and (obs["sd_pct"] > 0).all()


# ── panel C ──────────────────────────────────────────────────────────────────────────────────

@corpus
def test_the_four_heldout_summaries_reproduce_the_authoritative_outputs():
    table, meta = mm.build_source_data()
    checks = mm.validate(table, meta)
    for key, expected in mm.EXPECTED_LODO.items():
        assert checks[f"heldout_rmse_{key}"]["value"] == expected
        assert checks[f"heldout_rmse_{key}"]["match"]


@corpus
def test_the_two_aligned_strategies_are_indistinguishable_at_one_decimal():
    table, _ = mm.build_source_data()
    summ = table[table["kind"].eq("heldout_summary")].set_index("series")["value_pct"]
    aligned = round(float(summ[mm.MODELS[2][1]]), 1)
    aligned_rate = round(float(summ[mm.MODELS[3][1]]), 1)
    assert aligned == aligned_rate == 7.0        # the "no material improvement" claim


@corpus
def test_the_first_aligning_observation_is_excluded_from_residuals():
    # Verified against the artifact, not assumed: the summary equals the RMS over held-out dates
    # of the POST-FIRST residual column, and differs from the all-observation column.
    from diffractomorph_pipeline.config import data_root
    lodo = pd.read_csv(data_root() / mm.SOURCES["lodo"])
    cand = pd.read_csv(data_root() / mm.SOURCES["candidates"])
    lodo = lodo[lodo["cohort"].eq(mm.FIT_COHORT)]
    cand = cand[cand["cohort"].eq(mm.FIT_COHORT)]
    for key, _ in mm.MODELS:
        post = lodo[lodo["model"].eq(key)]["held_post_first_rmse_pct"].dropna()
        allo = lodo[lodo["model"].eq(key)]["held_all_obs_rmse_pct"].dropna()
        summary = float(cand[cand["model"].eq(key)]["lodo_rmse_pct"].iloc[0])
        assert np.isclose(np.sqrt((post ** 2).mean()), summary), key
        assert not np.isclose(np.sqrt((allo ** 2).mean()), summary), key
    table, meta = mm.build_source_data()
    assert "post_first" in meta["held_out"]["residual_basis"]


# ── pH 5.0 handling ──────────────────────────────────────────────────────────────────────────

@corpus
def test_the_multiplier_is_frozen_for_ph5_and_not_refitted():
    from diffractomorph_pipeline.config import data_root
    fs = pd.read_csv(data_root() / mm.SOURCES["fit_summary"]).iloc[0]
    table, meta = mm.build_source_data()
    frozen = meta["transfer"]["frozen_rate_scale"]
    assert np.isclose(frozen, float(fs["selected_rate_scale_datebalanced"]))
    curve = table[table["kind"].eq("model_curve") & table["panel"].eq("pH 5.0")]
    assert np.isclose(curve["rate_scale"].unique()[0], frozen)
    # the pH 5.0-only value is present but held separately as a subordinate diagnostic
    post_hoc = table[table["kind"].eq("model_curve_subordinate")]
    assert not np.isclose(post_hoc["rate_scale"].unique()[0], frozen)


@corpus
def test_ph5_is_not_included_in_the_ph40_45_parameter_estimation():
    table, meta = mm.build_source_data()
    assert meta["trajectories"]["conditions"] == ["pH 4.0", "pH 4.5"]
    checks = mm.validate(table, meta)
    assert checks["ph5_frozen_not_refitted"]["ph5_in_estimation"] is False
    assert checks["ph5_frozen_not_refitted"]["estimated_on"] == "pH 4.0 and pH 4.5 only"
    # and the artifact itself records that pH 5.0 shared dates with training
    assert meta["transfer"]["heldout_is_independent_external"] is False
    assert meta["transfer"]["shared_dates_with_training"]


# ── claim boundary ───────────────────────────────────────────────────────────────────────────

NEGATORS = r"(?:not|never|rather than|instead of|cannot|does not|did not|is not|no )"


def _affirmative(text: str, phrase: str) -> list[str]:
    """Occurrences of ``phrase`` NOT governed by a nearby negator.

    The correct wording necessarily *mentions* these phrases in order to disclaim them, so a
    bare substring search would flag the very sentences that establish the boundary. This looks
    for affirmative use only.
    """
    import re
    hits = []
    for m in re.finditer(re.escape(phrase), text):
        window = text[max(0, m.start() - 60):m.start()]
        if not re.search(NEGATORS + r"[^.]{0,60}$", window):
            hits.append(text[max(0, m.start() - 60):m.end() + 20])
    return hits


def test_module_does_not_call_ph5_external_validation_or_2point2_a_kinetic_constant():
    src = " ".join(Path(mm.__file__).read_text().split()).lower()
    for phrase in ("external validation", "independently validated",
                   "selected rate scale", "shared effective rate",
                   "transferable kinetic constant", "2.2-fold faster"):
        assert not _affirmative(src, phrase), (phrase, _affirmative(src, phrase))
    assert "condition transfer" in src
    assert "aligned at first uv" in src
    # and the disclaimers really are present, in negated form
    assert "never external validation" in src



# ── artifact discipline ──────────────────────────────────────────────────────────────────────

def test_missing_source_raises_instead_of_falling_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "data_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="does not substitute"):
        mm._read("candidates")


def test_empty_source_raises(tmp_path, monkeypatch):
    target = tmp_path / mm.SOURCES["candidates"]
    target.parent.mkdir(parents=True)
    target.write_text("cohort,model,lodo_rmse_pct\n")
    monkeypatch.setattr(mm, "data_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="is empty"):
        mm._read("candidates")


@corpus
def test_validate_rejects_a_table_whose_heldout_numbers_drifted():
    table, meta = mm.build_source_data()
    broken = table.copy()
    mask = broken["kind"].eq("heldout_summary") & broken["series"].eq(mm.MODELS[0][1])
    broken.loc[mask, "value_pct"] = 99.9
    with pytest.raises(ValueError, match="does not reproduce"):
        mm.validate(broken, meta)


@corpus
def test_first_uv_time_is_read_from_the_data_not_hard_coded():
    _, meta = mm.build_source_data()
    src = Path(mm.__file__).read_text()
    assert meta["trajectories"]["first_uv_time_min"] == 2.0      # what the data happens to say
    assert "first_uv_time_min = 2" not in src and "FIRST_UV = 2" not in src


@corpus
def test_rendered_outputs_and_source_csv_cover_every_plotted_series(tmp_path):
    mm._emit(tmp_path, ("pdf", "png", "svg"))
    for ext in ("pdf", "png", "svg"):
        assert (tmp_path / f"{mm.STEM}.{ext}").stat().st_size > 0
    csv = pd.read_csv(tmp_path / f"{mm.STEM}_source_data.csv")
    # the retired 2x2 design carried a pH 5.0 subordinate curve; the three-panel figure does not
    assert set(csv["kind"]) == {"observation", "model_curve",
                                "heldout_point", "heldout_summary"}
    prov = pd.read_json(tmp_path / f"{mm.STEM}_provenance.json", typ="series")
    assert "Copt" in prov["scope"]["excluded"] and "q3" in prov["scope"]["excluded"]
    assert "pH 5.0 transfer panel" in prov["scope"]["excluded"]
    assert prov["git_commit"]


# ── corrections: pH-specific calibration, artifact-sourced pH5 RMSE, wording ─────────────────

def test_provenance_records_ph_specific_curves_and_offsets():
    from diffractomorph_pipeline.assay import calibration as calibration_module
    if calibration_module._DEFAULT_PROFILE is None:
        pytest.skip("study assay profile is supplied by the external data bundle")
    # The filter offset is 1.48 ug/mL at pH 4.0 and 4.5 but 1.02 at pH 5.0, and the standard
    # curves differ per pH. A figure spanning all three must not record only one condition's
    # constants.
    from arm_b_provenance import provenance_record
    rec = provenance_record("probe", uv_ph_values=(4.0, 4.5, 5.0))
    cal = rec["uv_calibration"]
    assert cal["ph_values"] == [4.0, 4.5, 5.0]
    assert cal["filter_offset_ugml"] == {"4.0": 1.48, "4.5": 1.48, "5.0": 1.02}
    for ph in ("4.0", "4.5", "5.0"):
        for wl in ("280", "490"):
            assert f"{ph}_{wl}nm" in cal["standard_curves"]
    # the three pH values genuinely differ, so recording one would have been wrong
    slopes = {k: v["slope"] for k, v in cal["standard_curves"].items() if k.endswith("280nm")}
    assert len(set(slopes.values())) == 3


def test_provenance_default_stays_single_ph_for_existing_callers():
    from arm_b_provenance import provenance_record
    rec = provenance_record("probe")
    assert rec["uv_calibration"]["ph_values"] == [4.5]


@corpus
def test_figure_provenance_records_the_ph_conditions_it_actually_spans(tmp_path):
    # the three-panel figure covers pH 4.0 and 4.5 only; pH 5.0 moved to Results/Table S3
    mm._emit(tmp_path, ("pdf",))
    prov = pd.read_json(tmp_path / f"{mm.STEM}_provenance.json", typ="series")
    assert prov["uv_calibration"]["ph_values"] == [4.0, 4.5]
    assert prov["uv_calibration"]["filter_offset_ugml"] == {"4.0": 1.48, "4.5": 1.48}


@corpus
def test_both_ph5_rmse_values_come_from_the_transfer_artifact():
    from diffractomorph_pipeline.config import data_root
    art = pd.read_csv(data_root() / mm.SOURCES["ph5_transfer"])
    frozen = art[art["scope"].eq("overall_frozen")].iloc[0]
    own = art[art["scope"].eq("pH5_own_optimum")].iloc[0]
    _, meta = mm.build_source_data()
    t = meta["transfer"]
    assert t["ph5_transfer_rmse_pct_artifact"] == float(frozen["rmse_pct"]) == 10.772
    assert t["ph5_own_rmse_pct_artifact"] == float(own["rmse_pct"]) == 7.097
    assert round(t["ph5_own_rmse_pct_artifact"], 1) == 7.1
    assert t["ph5_rmse_source"] == "selected_rate_only_pH5_transfer.csv"


@corpus
def test_both_plotted_ph5_curves_are_validated_against_the_artifact():
    table, meta = mm.build_source_data()
    checks = mm.validate(table, meta)
    for name in ("ph5_frozen_transfer_rmse", "ph5_own_optimum_rmse"):
        assert checks[name]["match_to_1dp"], name
        assert checks[name]["source"] == "selected_rate_only_pH5_transfer.csv"
    assert checks["ph5_frozen_transfer_rmse"]["rate_scale"] == 2.197
    assert checks["ph5_own_optimum_rmse"]["rate_scale"] == 0.672




@corpus
def test_panel_c_summary_really_is_an_rms_not_a_mean():
    # The distinction the label now makes must be true of the numbers.
    from diffractomorph_pipeline.config import data_root
    lodo = pd.read_csv(data_root() / mm.SOURCES["lodo"])
    lodo = lodo[lodo["cohort"].eq(mm.FIT_COHORT) & lodo["model"].eq("injection_base")]
    v = lodo["held_post_first_rmse_pct"].dropna()
    table, _ = mm.build_source_data()
    shown = float(table[table["kind"].eq("heldout_summary")
                        & table["series"].eq(mm.MODELS[0][1])]["value_pct"].iloc[0])
    assert np.isclose(shown, np.sqrt((v ** 2).mean()))
    assert not np.isclose(shown, v.mean())        # RMS and mean genuinely differ here


# ── revised three-panel design ───────────────────────────────────────────────────────────────

@corpus
def test_revised_source_data_has_exactly_the_three_intended_panels():
    table, _ = mm.figure_source_data()
    assert set(table["panel"]) == {"pH 4.0", "pH 4.5", "held-out RMSE"}


@corpus
def test_revised_figure_contains_no_ph5_panel_or_rows():
    table, _ = mm.figure_source_data()
    assert "pH 5.0" not in set(table["panel"])
    assert "pH 5.0" not in set(table["condition"].dropna())
    assert table["kind"].ne("model_curve_subordinate").all()      # the post hoc pH 5.0 curve
    blob = " ".join(str(v) for v in table.to_dict("list").values())
    assert "pH 5.0" not in blob
    # ...while the full builder still produces it, for Results/Table S3
    full, _ = mm.build_source_data()
    assert "pH 5.0" in set(full["panel"])


@corpus
def test_revised_trajectory_panels_show_only_the_two_intended_strategies():
    table, _ = mm.figure_source_data()
    for panel in ("pH 4.0", "pH 4.5"):
        curves = set(table[table["panel"].eq(panel)
                           & table["kind"].eq("model_curve")]["series"])
        assert curves == set(mm.SHOWN_IN_TRAJECTORY)
        assert len(curves) == 2
        # the comparison models are deliberately absent from A/B
        assert mm.MODELS[1][1] not in curves      # original 2.2x correction
        assert mm.MODELS[3][1] not in curves      # aligned + fitted rate


@corpus
def test_revised_panel_c_still_carries_all_four_strategies_exactly():
    table, meta = mm.figure_source_data()
    summ = table[table["kind"].eq("heldout_summary")].set_index("series")["value_pct"]
    assert len(summ) == 4
    for key, label in mm.MODELS:
        assert round(float(summ[label]), 1) == mm.EXPECTED_LODO[key]
    # and the aligned pair remains equal, which is the panel's point
    assert round(float(summ[mm.MODELS[2][1]]), 1) == round(float(summ[mm.MODELS[3][1]]), 1) == 7.0


@corpus
def test_revised_first_uv_time_comes_from_the_artifact():
    _, meta = mm.figure_source_data()
    assert meta["trajectories"]["first_uv_time_min"] == 2.0
    src = Path(mm.__file__).read_text()
    for literal in ("first_uv = 2", "FIRST_UV = 2", "axvspan(0, 2"):
        assert literal not in src, literal


def test_revised_panels_carry_no_in_panel_annotation_text():
    """The redesign moved every explanatory string to the caption."""
    import inspect
    body = inspect.getsource(mm._trajectory_axis) + inspect.getsource(mm._heldout_axis)
    for banned in ("not observed", "by UV", "preparation dates ·", "no material",
                   "improvement", "overlapped", "diamond =", "RMS across"):
        assert banned not in body, banned
    # short panel titles only
    assert 'ax.set_title(condition' in body
    assert '"Held-out prediction error"' in body


def test_revised_uses_a_compact_three_entry_legend_in_the_trajectory_panels():
    import inspect
    body = inspect.getsource(mm._trajectory_axis)
    for entry in ("UV measurement", "Injection-start model", "First-UV-aligned model"):
        assert entry in body, entry
    assert "fig.legend" not in inspect.getsource(mm.render)   # no global legend


@corpus
def test_revised_y_axis_spans_40_to_100_without_clipping_any_drawn_series():
    table, _ = mm.figure_source_data()
    lo, hi = mm.YLIM
    assert (lo, hi) == (40.0, 100.0)
    drawn = table[table["panel"].isin(["pH 4.0", "pH 4.5"])]
    assert drawn["value_pct"].min() >= lo, drawn["value_pct"].min()
    assert drawn["value_pct"].max() <= hi


@corpus
def test_outputs_use_the_canonical_stem_with_no_revised_suffix(tmp_path):
    mm._emit(tmp_path, ("pdf", "png", "svg"))
    for suffix in (".pdf", ".png", ".svg", "_source_data.csv",
                   "_provenance.json", "_caption.md"):
        assert (tmp_path / f"{mm.STEM}{suffix}").stat().st_size > 0
    # the superseded 2x2 design is retired: no _revised stem, no variant flag
    assert "_revised" not in mm.STEM
    assert not (tmp_path / f"{mm.STEM}_revised.pdf").exists()



@corpus
def test_per_condition_rmse_shows_alignment_helps_ph45_and_slightly_hurts_ph40():
    pc = mm.per_condition_rmse()
    get = lambda c, m: float(pc[pc["condition"].eq(c) & pc["model"].eq(m)]
                             ["within_condition_rmse_pct"].iloc[0])
    assert round(get("pH 4.5", "injection_base"), 1) == 15.5
    assert round(get("pH 4.5", "first_uv_anchor"), 1) == 6.1
    assert round(get("pH 4.0", "injection_base"), 1) == 8.4
    assert round(get("pH 4.0", "first_uv_anchor"), 1) == 9.0
    # the direction differs by condition — that is the qualification the caption must carry
    assert get("pH 4.5", "first_uv_anchor") < get("pH 4.5", "injection_base")
    assert get("pH 4.0", "first_uv_anchor") > get("pH 4.0", "injection_base")


@corpus
def test_per_condition_basis_is_labelled_as_not_held_out():
    pc = mm.per_condition_rmse()
    assert pc["basis"].str.contains("not held out", case=False).all()
    # and it must genuinely differ from the pooled held-out numbers, so the two cannot be
    # silently reconciled by a reader
    table, _ = mm.figure_source_data()
    pooled = float(table[table["kind"].eq("heldout_summary")
                         & table["series"].eq(mm.MODELS[2][1])]["value_pct"].iloc[0])
    per = pc[pc["model"].eq("first_uv_anchor")]["within_condition_rmse_pct"].tolist()
    assert all(not np.isclose(pooled, v, atol=0.05) for v in per)


@corpus
def test_revised_caption_heading_does_not_claim_the_mismatch_is_resolved(tmp_path):
    mm._emit(tmp_path, ("pdf",))
    caption = " ".join((tmp_path / f"{mm.STEM}_caption.md").read_text().split())
    assert "Mass-domain evaluation separates first-observation alignment from post-capture kinetics" in caption
    assert "alignment resolves the early mass-domain mismatch" not in caption


@corpus
def test_caption_describes_the_panel_c_summary_as_an_rms_not_an_average(tmp_path):
    mm._emit(tmp_path, ("pdf",))
    caption = " ".join((tmp_path / f"{mm.STEM}_caption.md").read_text().split())
    assert "root-mean-square across them" in caption
    assert "diamonds are the average" not in caption


@corpus
def test_revised_caption_carries_the_condition_dependence_with_its_basis(tmp_path):
    mm._emit(tmp_path, ("pdf",))
    caption = " ".join((tmp_path / f"{mm.STEM}_caption.md").read_text().split())
    assert "the improvement was condition-dependent" in caption
    assert "15.5 to 6.1 percentage points at pH 4.5" in caption
    assert "8.4 to 9.0 percentage points at pH 4.0" in caption
    # the basis distinction must survive editing, or the numbers cannot be reconciled with panel C
    assert "a different basis from the pooled held-out value" in caption
    assert "pooled held-out error" in caption
