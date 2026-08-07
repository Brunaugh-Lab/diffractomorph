"""Arm B discovery and replication-structure tests.

The part worth pinning is what counts as an independent replicate: the suspension prep, keyed by
INJECTION DATE rather than folder name, with the cuvette runs sharing a prep being technical
repeats. The day-1 folder spans two preps, so a condition is n=4, never n=3 and never n=9. These
cover the QC-to-replicate routing, the volume-log parsing that supplies prep identity, the
replication summary, and the within-prep paired contrast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))

import arm_b_common as ab  # noqa: E402
from arm_b_injected_mass import build_dose_table  # noqa: E402
from arm_b_partition import condition_means, paired_contrasts, prep_means  # noqa: E402


# ── QC → replicate routing ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name, expected", [
    ("CFZ QC 20260702 Rep 1.rtf", {1}),
    ("CFZ_QC_20260702_For_Rep1.xlsx", {1}),
    ("CFZ QC 20260707 Reps 2 and 3.rtf", {2, 3}),
    ("CFZ_QC_20260707_For_Reps2_and_3.xlsx", {2, 3}),
    ("CFZ QC 20260722.rtf", {1, 2, 3}),          # no replicate token → covers the whole cell
    ("CFZ UV QC 20260723.xlsx", {1, 2, 3}),
])
def test_qc_filename_maps_to_the_replicates_it_covers(name, expected):
    assert ab._reps_for_qc_name(name) == expected


def test_qc_date_in_filename_is_not_read_as_a_replicate():
    # "20260702" contains digits 1-3; the replicate parser must not pick them up.
    assert ab._reps_for_qc_name("CFZ QC 20260723.rtf") == {1, 2, 3}


@pytest.mark.parametrize("stem, expected", [
    ("CFZ_0.5x_Tween_Spike_pH4.5_rep1", 1),           # day-1 files carry no day token
    ("CFZ_0.5x_Tween_Spike_pH4.5_day2_rep3", 2),
    ("CFZ_10x_Tween_Spike_pH4.5_day3_rep2", 3),
])
def test_day_token_defaults_to_one_when_absent(stem, expected):
    assert ab._day_token(stem) == expected


# ── replication bookkeeping ──────────────────────────────────────────────────────────────────

def _runs_frame():
    """The real Arm B layout: four independent preps per condition (the day-1 folder is two).

    Preps carry 1, 2, 3 and 3 technical repeats; ``folder`` keeps the on-disk grouping so the
    prep/folder distinction stays testable.
    """
    later = {"0.5x CMC": ("20260722", "20260723"),
             "1.0x CMC": ("20260722", "20260724"),
             "10x CMC": ("20260723", "20260724")}
    rows = []
    for condition, tail in later.items():
        plan = [("20260702", "20260702", [1]), ("20260707", "20260702", [2, 3])]
        plan += [(d, d, [1, 2, 3]) for d in tail]
        for i, (prep, folder, reps) in enumerate(plan, 1):
            for r in reps:
                rows.append({"condition": condition, "xcmc": ab.CONDITIONS[condition],
                             "prep": prep, "prep_index": i, "folder": folder, "rep": r,
                             "run_id": f"{condition}_{prep}_rep{r}"})
    return pd.DataFrame(rows)


def _synthetic(runs, *, k):
    """Attach the per-run metrics the audited aggregation expects, so fixtures stay minimal."""
    runs = runs.copy()
    runs["k"] = k
    for col in ("tau_size_slope_qc", "tau_size_slope_unfiltered", "copt_thalf_min",
                "copt_floor", "copt_frac_loss", "C_end_ugml", "n_channels_qc_pass"):
        runs[col] = 1.0
    runs["run_has_bound_hit"] = False
    return runs


def test_each_condition_has_four_independent_preps_not_three_or_nine():
    summary = ab.replication_summary(_runs_frame()).set_index("condition")
    assert set(summary["n_preps"]) == {4}                     # injection dates, the true n
    assert set(summary["n_folders"]) == {3}                   # folders under-count preps
    assert set(summary["n_runs"]) == {9}                      # cuvette runs, deliberately not an n
    assert set(summary["reps_per_prep"]) == {"1 | 2 | 3 | 3"}


def test_condition_n_is_the_prep_count_after_collapsing_technical_repeats():
    preps = prep_means(_synthetic(_runs_frame(), k=1.0))
    assert len(preps) == 12                                   # 3 conditions x 4 preps
    assert sorted(preps["n_technical_reps"].unique()) == [1, 2, 3]
    assert set(condition_means(preps)["n_preps"]) == {4}


def test_prep_means_are_combined_unweighted_not_by_run_count():
    # One prep backed by a single run, three backed by 2-3. An unweighted mean of prep means is
    # 2.5; a run-count-weighted mean would be pulled toward the many-repeat preps (2.78).
    runs = _runs_frame()
    runs = _synthetic(runs, k=[1.0 if r.prep_index == 1 else 3.0 for r in runs.itertuples()])
    out = condition_means(prep_means(runs))
    assert np.allclose(out["k_mean"], 2.5)


def test_prep_to_prep_spread_is_the_error_term_not_technical_scatter():
    # Technical repeats vary wildly within each prep, but every prep mean is identical → the
    # reported SD must be zero. An SD taken over runs instead of preps would be large.
    runs = _runs_frame()
    scatter = {1: [5.0], 2: [1.0, 9.0], 3: [1.0, 5.0, 9.0]}      # every prep mean is 5.0
    runs = _synthetic(runs, k=[scatter[len(g)][i]
                               for _, g in runs.groupby(["condition", "prep"], sort=False)
                               for i in range(len(g))])
    out = condition_means(prep_means(runs))
    assert np.allclose(out["k_mean"], 5.0)
    assert np.allclose(out["k_sd"], 0.0)
    assert np.allclose(out["k_sem"], 0.0)


def test_shared_preps_let_a_pair_be_contrasted_within_one_suspension():
    pairs = ab.condition_pairs_sharing_preps(_runs_frame())
    assert len(pairs) == 3
    # every pair shares 07-02, 07-07 and one later prep
    assert set(pairs["n_shared_preps"]) == {3}


def test_within_prep_contrast_cancels_prep_to_prep_variation():
    # Large per-prep offsets with a fixed 2x condition effect: differencing within a shared prep
    # must recover the effect exactly, which is why the paired contrast is worth reporting.
    offsets = {"20260702": 1.0, "20260707": 3.0, "20260722": 10.0,
               "20260723": 100.0, "20260724": 1000.0}
    effect = {"0.5x CMC": 1.0, "1.0x CMC": 2.0, "10x CMC": 4.0}
    runs = _runs_frame()
    runs = _synthetic(runs, k=[offsets[r.prep] * effect[r.condition] for r in runs.itertuples()])
    contrasts = paired_contrasts(prep_means(runs), ab.condition_pairs_sharing_preps(runs))
    ratios = {(r.condition_a, r.condition_b): r.k_ratio_b_over_a for r in contrasts.itertuples()}
    assert np.isclose(ratios[("0.5x CMC", "1.0x CMC")], 2.0)
    assert np.isclose(ratios[("1.0x CMC", "10x CMC")], 2.0)
    assert np.isclose(ratios[("0.5x CMC", "10x CMC")], 4.0)


def test_condition_means_carry_the_centrifuge_filter_diagnostic():
    preps = prep_means(_synthetic(_runs_frame(), k=1.0))
    out = condition_means(preps).set_index("condition")
    # a diagnostic of particulate carryover in the centrifuged read, not a caveat on the
    # selected filtered Cs
    assert bool(out.loc["0.5x CMC", "centrifuge_filter_disagreement"]) is True
    assert bool(out.loc["10x CMC", "centrifuge_filter_disagreement"]) is False
    assert "k_per_Cs" not in out.columns          # Cs enters via the driving force, not a 2nd divide


# ── injection-volume log parsing ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("label, level, day", [
    ("4.5 0.02% w/v Tween (0.5x)", "0.5x", None),
    ("4.5 0.02% w/v Tween (1.0x - Day 2)", "1.0x", "2"),
    ("4.5 0.02% w/v Tween (0.5x  - Day 3)", "0.5x", "3"),     # irregular double space
    ("4.5 0.02% w/v Tween (10x  - Day 2)", "10x", "2"),
])
def test_arm_b_volume_labels_parse(label, level, day):
    m = ab._LABEL.search(label)
    assert m and m.group("level") == level and m.group("day") == day


@pytest.mark.parametrize("label", [
    "4.5 0.01% w/v Tween",                    # Arm A concentration study
    "4.5 0.03% w/v Tween",
    "4.5 0.02% w/v Tween (12% Copt)",         # the Copt ladder, same 0.02% stock
    "4.0 0.02% w/v",
])
def test_other_studies_in_the_shared_log_are_not_claimed_as_arm_b(label):
    assert ab._LABEL.search(label) is None


@pytest.mark.parametrize("cell, expected", [
    ("40 uL", 40.0), ("45 uL ", 45.0), ("30uL", 30.0), (40, 40.0), (None, None), ("n/a", None),
])
def test_volume_cell_parsing(cell, expected):
    assert ab._volume_uL(cell) == expected


# ── real-corpus discovery (skipped when the data tree is absent) ─────────────────────────────

def _study_root():
    try:
        return ab.default_study_root()
    except RuntimeError:
        return Path("/__dfm_data_root_unset__")


corpus = pytest.mark.skipif(not _study_root().is_dir(), reason="Arm B corpus not present")


@corpus
def test_discovery_finds_four_preps_per_condition():
    runs = ab.discover_runs()
    assert len(runs) == 27
    assert set(runs["condition"]) == set(ab.CONDITIONS)
    assert (runs.groupby("condition")["prep"].nunique() == 4).all()
    assert (runs.groupby("condition")["folder"].nunique() == 3).all()
    assert (runs.groupby(["condition", "folder"]).size() == 3).all()
    for column in ("rtf", "uv_file", "q3_dir", "qc_rtf", "qc_xlsx"):
        assert runs[column].map(lambda p: Path(p).exists()).all(), column


@corpus
def test_split_qc_day_routes_reps_two_and_three_together():
    runs = ab.discover_runs()
    cell = runs[(runs["folder"] == "20260702") & (runs["condition"] == "1.0x CMC")]
    by_rep = dict(zip(cell["rep"], cell["qc_rtf"]))
    assert by_rep[1] != by_rep[2]
    assert by_rep[2] == by_rep[3]


@corpus
def test_every_run_gets_a_volume_and_a_dose():
    dose = build_dose_table()
    assert len(dose) == 27
    assert dose["injected_uL"].notna().all()
    assert (dose["injected_mass_mg"] > 0).all()
    # dose is resolved per run: same volume, different QC → different delivered mass
    day1 = dose[dose["folder"] == "20260702"]
    assert day1["injected_uL"].nunique() == 1
    assert day1["injected_mass_mg"].nunique() > 1


@corpus
def test_day_one_folder_holds_two_preps():
    # Replicate 1 was injected on 07-02; replicates 2-3 on 07-07 off a fresh suspension. The
    # folder name flattens that, so prep identity must come from the injection date.
    dose = build_dose_table()
    day1 = dose[dose["folder"] == "20260702"]
    assert set(day1["prep"]) == {"20260702", "20260707"}
    assert day1.loc[day1["rep"].eq(1), "prep"].eq("20260702").all()
    assert day1.loc[day1["rep"].ne(1), "prep"].eq("20260707").all()
    assert day1["prep_differs_from_folder"].sum() == 6
    # ...and every other folder is one prep
    rest = dose[dose["folder"] != "20260702"]
    assert not rest["prep_differs_from_folder"].any()


@corpus
def test_conditions_sharing_a_prep_share_one_suspension_qc():
    # The evidence that a prep is shared across conditions: same-prep condition folders carry
    # byte-identical suspension QC exports.
    runs = ab.discover_runs()
    for prep, group in runs.groupby("prep"):
        digests = {c: g.sort_values("rep")["qc_rtf"].map(lambda p: Path(p).read_bytes()).tolist()
                   for c, g in group.groupby("condition")}
        reference = next(iter(digests.values()))
        assert all(v == reference for v in digests.values()), prep


# ── UV mass balance and model transfer ───────────────────────────────────────────────────────

def test_uv_prep_means_collapse_repeats_before_conditions():
    import arm_b_uv_timecourse as uv
    rows = []
    for run in _runs_frame().itertuples():
        for t in (2.0, 10.0):
            # technical repeats disagree wildly; every prep mean is 0.5
            offset = {1: 0.0, 2: -0.4, 3: 0.4}[run.rep] if run.prep_index == 4 else 0.0
            rows.append({"condition": run.condition, "xcmc": run.xcmc, "prep": run.prep,
                         "prep_index": run.prep_index, "rep": run.rep, "time_min": t,
                         "conc_ugml": 1.0, "pct_injected": 50.0,
                         "cumulative_dissolved_ug": 100.0,
                         "recovery_mass_fraction": 0.5 + offset, "qc_any": False})
    long = pd.DataFrame(rows)
    preps = uv.prep_means(long)
    assert np.allclose(preps["recovery_mass_fraction"], 0.5)
    conditions = uv.condition_means(preps)
    assert set(conditions["n_preps"]) == {4}
    assert np.allclose(conditions["recovery_mass_fraction_sd"], 0.0)


def test_model_transfer_residual_is_model_minus_measured():
    import arm_b_model_transfer as mt
    rows = []
    for prep in ("20260702", "20260707", "20260722", "20260723"):
        rows.append({"model": "frozen_selected_rate_scale", "condition": "0.5x CMC", "xcmc": 0.5,
                     "prep": prep, "rep": 1, "time_min": 2.0,
                     "residual_model_minus_measured_pct": 4.0, "model_saturated": False})
    by_prep, overall = mt.residual_summaries(pd.DataFrame(rows))
    assert len(by_prep) == 4                                  # one row per prep, not per run
    assert overall.loc[0, "n_preps"] == 4
    assert np.isclose(overall.loc[0, "prep_balanced_rmse_pct"], 4.0)
    assert np.isclose(overall.loc[0, "mean_signed_resid_pct"], 4.0)   # signed: model above measured


def test_saturated_late_observations_are_flagged_and_split_from_the_early_window():
    # The frozen model pins near 100% after a few minutes, so a pooled residual measures the
    # ceiling. Early and late must be reported separately, with saturation counted.
    import arm_b_model_transfer as mt
    rows = []
    for prep in ("20260702", "20260707", "20260722", "20260723"):
        for t, resid, sat in [(2.0, 1.0, False), (4.0, 1.0, False), (20.0, 9.0, True)]:
            rows.append({"model": "frozen_selected_rate_scale", "condition": "10x CMC",
                         "xcmc": 10.0, "prep": prep, "rep": 1, "time_min": t,
                         "residual_model_minus_measured_pct": resid, "model_saturated": sat})
    by_prep, overall = mt.residual_summaries(pd.DataFrame(rows))
    assert np.isclose(overall.loc[0, "early_prep_balanced_rmse_pct"], 1.0)   # clean early window
    assert np.isclose(overall.loc[0, "prep_balanced_rmse_pct"], np.sqrt(83.0 / 3))  # pooled is worse
    assert np.isclose(overall.loc[0, "frac_obs_unsaturated"], 2 / 3)


@corpus
def test_model_transfer_supplies_the_tween_level_cs_not_the_aqueous_one():
    # Each condition must be predicted against its own Arm C ladder Cs, otherwise the transfer
    # would credit the micellar solubility boost to kinetics.
    import arm_b_cs
    import arm_b_model_transfer as mt
    expected_cs = arm_b_cs.cs_map("filtered_48h")
    path = (mt.default_study_root() / "analysis" / "model_transfer" / "filtered_48h"
            / "arm_b_model_transfer_predictions.csv")
    if not path.exists():
        pytest.skip("model transfer outputs not generated")
    preds = pd.read_csv(path)
    for condition, expected in expected_cs.items():
        got = preds.loc[preds["condition"].eq(condition), "cs_ugml"].unique()
        assert len(got) == 1 and np.isclose(got[0], expected), (condition, got)


# ── audit coverage: Cs ladders, despike provenance, fit QC, double normalization ─────────────

def test_filtered_48h_is_the_primary_ladder_and_the_default():
    import arm_b_cs
    assert arm_b_cs.PRIMARY_LADDER == "filtered_48h"
    assert arm_b_cs.DEFAULT_LADDER == arm_b_cs.PRIMARY_LADDER
    assert set(arm_b_cs.SECONDARY_LADDERS) == {"centrifuged_48h", "centrifuged_24h"}
    prov = arm_b_cs.provenance()
    assert prov["cs_is_primary"] and prov["cs_role"] == "primary"
    assert "prospectively" in prov["cs_selection"] and "9ed2cb9" in prov["cs_selection"]


def test_centrifuged_ladders_are_labelled_secondary_method_sensitivities():
    import arm_b_cs
    for name in arm_b_cs.SECONDARY_LADDERS:
        prov = arm_b_cs.provenance(name)
        assert not prov["cs_is_primary"]
        assert prov["cs_role"] == "secondary_method_sensitivity"
        assert "not the operational Cs" in prov["cs_selection"]


def test_primary_ladder_matches_the_recorded_arm_c_values():
    import arm_b_cs
    cs = arm_b_cs.cs_map()                       # defaults to the primary
    assert np.isclose(cs["0.5x CMC"], 7.77, atol=0.01)
    assert np.isclose(cs["1.0x CMC"], 8.39, atol=0.01)
    assert np.isclose(cs["10x CMC"], 12.97, atol=0.01)


def test_three_cs_ladders_are_exposed_with_provenance():
    import arm_b_cs
    t = arm_b_cs.ladders()
    assert set(t["ladder"]) == set(arm_b_cs.LADDER_NAMES)
    assert set(t["condition"]) == set(ab.CONDITIONS)
    assert set(t.loc[t["ladder"].eq(arm_b_cs.PRIMARY_LADDER), "role"]) == {"primary"}
    for name in arm_b_cs.LADDER_NAMES:
        prov = arm_b_cs.provenance(name)
        assert prov["cs_ladder"] == name and prov["cs_source_file"] and prov["cs_description"]
        assert set(prov["cs_ugml"]) == set(ab.CONDITIONS)


def test_below_cmc_anchor_is_retained_in_every_ladder():
    # 0.5x CMC is the deliberate monomer-only anchor; it must never be dropped.
    import arm_b_cs
    t = arm_b_cs.ladders()
    for name in arm_b_cs.LADDER_NAMES:
        assert "0.5x CMC" in set(t.loc[t["ladder"].eq(name), "condition"])


def test_centrifuge_filter_gap_is_a_particulate_diagnostic():
    import arm_b_cs
    gap = arm_b_cs.centrifuge_filter_gap().set_index("condition")
    # large where a short spin cannot clear particulate, small where the methods agree
    assert bool(gap.loc["0.5x CMC", "particulate_carryover_suspected"])
    assert bool(gap.loc["1.0x CMC", "particulate_carryover_suspected"])
    assert not bool(gap.loc["10x CMC", "particulate_carryover_suspected"])


def test_cs_ladders_disagree_most_below_cmc_which_is_why_they_are_all_reported():
    import arm_b_cs
    t = arm_b_cs.ladders().pivot(index="condition", columns="ladder", values="cs_ugml")
    gap = (t["centrifuged_48h"] - t["filtered_48h"]).abs()
    # where the short spin retains particulate, vs where the two preparations agree
    assert gap["0.5x CMC"] > 3.0 and gap["1.0x CMC"] > 3.0
    assert gap["10x CMC"] < 1.0


def test_unknown_cs_ladder_is_rejected_rather_than_silently_defaulted():
    import arm_b_cs
    with pytest.raises(ValueError):
        arm_b_cs.cs_map("whatever_ladder")


def test_no_second_cs_normalization_anywhere_in_the_partition_outputs():
    # k already contains Cs in its (Cs - C) denominator. Dividing the k ratio by the Cs ratio
    # again was the audited defect; it must not reappear in any column or in the source.
    import arm_b_partition as bp
    src = Path(bp.__file__).read_text()
    body = src.split('"""', 2)[2]                 # skip the module docstring explaining the removal
    assert "k_ratio_over_cs_ratio" not in body
    preps = prep_means(_synthetic(_runs_frame(), k=1.0))
    contrasts = paired_contrasts(preps, ab.condition_pairs_sharing_preps(_runs_frame()))
    assert "k_ratio_over_cs_ratio" not in contrasts.columns
    assert "cs_ratio_b_over_a" not in contrasts.columns


PROVENANCE_KEYS = ("mode", "cleaned", "n_frames_input", "n_frames_retained", "n_lead_dropped",
                   "n_interior_interpolated", "spike_frames", "n_copt_repaired",
                   "copt_repaired_frames", "n_dropped_copt_ceiling", "n_dropped_pre_gap",
                   "n_channels_admitted", "admission")


@corpus
@pytest.mark.parametrize("mode", ["raw", "pipeline", "pipeline+copt"])
def test_optical_modes_record_full_frame_provenance(mode):
    """Every dropped, interpolated and retained frame must be accounted for, per mode."""
    from arm_b_optical import optical_run
    run = ab.discover_runs().iloc[0]
    o = optical_run(run["rtf"], clean=mode)
    prov = o.provenance
    for key in PROVENANCE_KEYS:
        assert key in prov, key
    assert prov["mode"] == mode
    assert prov["cleaned"] is (mode != "raw")
    # frames retained must equal the array actually returned, and the accounting must balance
    assert prov["n_frames_retained"] == o.n_frames == o.copt.size == o.I.shape[0]
    assert prov["n_frames_input"] >= prov["n_frames_retained"]
    assert prov["n_lead_dropped"] >= 0 and prov["n_interior_interpolated"] >= 0
    if mode == "raw":
        assert prov["n_lead_dropped"] == prov["n_interior_interpolated"] == 0
        assert prov["n_copt_repaired"] == 0 and prov["admission"].startswith("none")
        assert o.admitted.all()
    else:
        assert prov["admission"].startswith("noise-surface")
        assert prov["n_channels_admitted"] == int(o.admitted.sum()) <= o.channels.size
        assert len(prov["copt_repaired_frames"]) == prov["n_copt_repaired"]
        assert (prov["n_copt_repaired"] > 0) is (mode == "pipeline+copt"
                                                 and prov["n_copt_repaired"] > 0)
    if mode != "pipeline+copt":
        assert prov["n_copt_repaired"] == 0        # the Copt repair is opt-in, never implicit


@corpus
def test_copt_repair_is_the_only_difference_between_the_two_cleaned_modes():
    from arm_b_optical import optical_run
    run = ab.discover_runs().iloc[0]
    a = optical_run(run["rtf"], clean="pipeline")
    b = optical_run(run["rtf"], clean="pipeline+copt")
    assert np.allclose(a.I, b.I)                    # intensities untouched by the Copt repair
    assert np.allclose(a.t_min, b.t_min)
    assert a.provenance["n_copt_repaired"] == 0


def test_unknown_optical_mode_is_rejected():
    import arm_b_optical
    with pytest.raises(ValueError):
        arm_b_optical.optical_run("nonexistent.rtf", clean="scrub_it_hard")


def test_copt_repair_removes_an_isolated_spike_and_reports_which_frame():
    import arm_b_optical
    t = np.arange(40, dtype=float)
    copt = np.full(40, 10.0)
    copt[17] = 40.0                                   # one isolated excursion
    fixed, flagged = arm_b_optical._copt_repair(t, copt)
    assert flagged == [17]
    assert np.isclose(fixed[17], 10.0)
    assert np.allclose(np.delete(fixed, 17), np.delete(copt, 17))   # nothing else touched


def test_boundary_capped_tau_fits_are_flagged_and_excluded_from_the_slope():
    import arm_b_partition as bp
    t = np.linspace(0, 20, 40)
    eff = np.geomspace(2.0, 40.0, 12)
    # channels 0-5 decay properly; 6-11 barely move → the fit rails at the upper tau bound
    cols = []
    for i in range(12):
        cols.append(bp._decay(t, 0.2, 1.5) if i < 6 else bp._decay(t, 0.0, 50.0))
    I = np.stack(cols, axis=1)
    table, slope, slope_unfiltered = bp.channel_taus(t, I, eff, np.ones(12, bool))
    railed = table[table["at_bound"]]
    assert len(railed) > 0
    assert railed["qc_reason"].str.contains("tau_at_bound").all()
    assert not table.loc[table["at_bound"], "qc_pass"].any()
    assert table["r2"].notna().any() and "decay_depth" in table


def test_shallow_channels_are_excluded_even_when_the_fit_converges():
    import arm_b_partition as bp
    t = np.linspace(0, 20, 40)
    eff = np.geomspace(2.0, 40.0, 8)
    I = np.stack([1.0 - 0.01 * (t / t.max()) for _ in range(8)], axis=1)   # ~flat
    table, slope, _ = bp.channel_taus(t, I, eff, np.ones(8, bool))
    assert table["qc_reason"].str.contains("shallow_decay").any()
    assert not table["qc_pass"].any()
    assert np.isnan(slope)


def test_uv_observation_flags_mark_but_never_drop():
    import arm_b_uv_timecourse as uv
    rows = []
    for i, rec in enumerate([1.00, 1.05, 0.90, 1.40]):     # rise, >1, big drop, extreme
        rows.append({"run_id": "r1", "time_min": float(i), "recovery_mass_fraction": rec})
    flagged = uv._flag(pd.DataFrame(rows))
    assert len(flagged) == len(rows)                        # nothing deleted
    assert flagged["qc_recovery_above_1"].sum() == 2      # 1.05 and 1.40, strictly > 1
    assert flagged["qc_large_decrease"].sum() == 1
    assert flagged["qc_isolated_extreme"].sum() == 1        # the 1.40, neighbours below threshold


def test_uv_variants_are_subsets_and_the_inclusive_one_is_complete():
    import arm_b_uv_timecourse as uv
    long = uv._flag(pd.DataFrame([
        {"run_id": "r1", "time_min": float(i), "recovery_mass_fraction": v}
        for i, v in enumerate([0.5, 0.9, 1.4, 0.95])]))
    v = uv.variants(long)
    assert len(v["inclusive"]) == len(long)
    assert len(v["excl_isolated_extreme"]) <= len(long)
    assert len(v["excl_all_flagged"]) <= len(v["excl_isolated_extreme"])


def test_provenance_record_carries_every_conditional_choice():
    import arm_b_provenance as prov
    r = prov.provenance_record("unit-test", cs_ladder="centrifuged_48h", optical_cleaned=True,
                               study_root="/tmp/x")
    for key in ("analysis", "generated_utc", "pipeline_version", "git_commit",
                "uv_calibration", "optical", "solubility"):
        assert key in r, key
    assert r["solubility"]["cs_ladder"] == "centrifuged_48h"
    # calibration is recorded per pH; Arm B is a pH-4.5 study so it records that condition only
    assert r["uv_calibration"]["ph_values"] == [4.5]
    assert r["uv_calibration"]["filter_offset_ugml"] == {"4.5": 1.48}
    assert "not calibrated" in r["uv_calibration"]["note"].lower()


def test_stability_verdict_is_scoped_to_the_primary_ladder():
    import arm_b_cs
    import arm_b_sensitivity as sens
    rows = []
    for mode, vals in (("raw", [3.0, 2.0, 1.0]), ("pipeline", [3.0, 2.0, 1.0]),
                       ("pipeline+copt", [3.0, 2.0, 1.0])):
        rows += [{"cs_ladder": arm_b_cs.PRIMARY_LADDER, "optical": mode, "condition": c,
                  "k_mean": v, "k_sem": 0.01} for c, v in zip(sens.ORDER, vals)]
    for ladder in arm_b_cs.SECONDARY_LADDERS:            # the particulate-containing pattern
        rows += [{"cs_ladder": ladder, "optical": "raw", "condition": c, "k_mean": v,
                  "k_sem": 0.01} for c, v in zip(sens.ORDER, [1.4, 1.3, 1.5])]
    contrasts = pd.DataFrame([{"cs_ladder": arm_b_cs.PRIMARY_LADDER, "optical": m, "k_ratio": 0.5}
                              for m in ("raw", "pipeline", "pipeline+copt")])
    v = sens.stability(pd.DataFrame(rows), contrasts)

    primary = v[v["scope"].eq("primary")]
    assert len(primary) == 3 and primary["cs_ladder"].eq(arm_b_cs.PRIMARY_LADDER).all()
    assert primary["survives_all"].all()                  # 3/3 optical modes
    assert set(primary["n_cells"]) == {3}                 # scoped to modes, not ladders

    secondary = v[v["scope"].eq("secondary_method_sensitivity")]
    assert set(secondary["cs_ladder"]) == set(arm_b_cs.SECONDARY_LADDERS)
    assert secondary["survives_all"].isna().all()         # reported, never adjudicating


@corpus
def test_prep_balanced_condition_means_use_four_preps_in_every_cell():
    import arm_b_sensitivity as sens
    path = ab.default_study_root() / "analysis" / "sensitivity" / "arm_b_k_sensitivity.csv"
    if not path.exists():
        pytest.skip("sensitivity sweep not generated")
    k = pd.read_csv(path)
    assert (k["n_preps"] == 4).all()
    assert len(k) == 3 * 3 * 3          # 3 ladders x 3 optical modes x 3 conditions


@corpus
def test_generated_summaries_are_reproducible_from_committed_code():
    # Re-running the analysis must reproduce the committed condition-level k table.
    import arm_b_partition as bp
    path = (ab.default_study_root() / "analysis" / "partition"
            / "filtered_48h_pipeline_copt" / "arm_b_partition_conditions.csv")
    if not path.exists():
        pytest.skip("partition outputs not generated")
    stored = pd.read_csv(path).set_index("condition")["k_mean"]
    fresh = bp.run_analysis(ladder="filtered_48h", clean="pipeline+copt")["conditions"] \
        .set_index("condition")["k_mean"]
    assert np.allclose(stored.loc[fresh.index], fresh, rtol=1e-9)


# ── q3 size-space analysis (media-diagnostic arms) ───────────────────────────────────────────

def test_extent_coordinate_is_monotone_and_bounded():
    import media_diagnostic_q3 as q3
    copt = np.array([20.0, 18.0, 19.0, 12.0, 8.0, 5.0, 5.0, 5.0, 5.0, 5.0])   # note the bump
    g = q3._extent(copt)
    assert g[0] == 0.0
    assert np.all(np.diff(g) >= 0)              # monotone envelope: matching stays single-valued
    assert g.min() >= 0.0 and g.max() <= 1.0


def test_extent_is_flat_when_there_is_no_optical_loss():
    import media_diagnostic_q3 as q3
    assert np.allclose(q3._extent(np.full(10, 7.0)), 0.0)


def test_percentiles_use_the_restricted_cumulative_not_the_raw_tail():
    # A distribution with mass beyond the inversion's reliable range must not have that mass
    # drag D50 upward; psd.restrict_cumulative is what prevents it.
    import media_diagnostic_q3 as q3
    from diffractomorph_pipeline import psd
    from diffractomorph_pipeline.optics.standards import GRID
    grid = np.asarray(GRID, float)
    dq = np.zeros((1, grid.size))
    fine, coarse = grid < 5.0, grid > psd.VALID_SIZE_MAX_UM
    dq[0, fine] = 0.7 / fine.sum()
    dq[0, coarse] = 0.3 / coarse.sum()
    traj = psd.Q3Trajectory(grid_um=grid, dQ3=dq, layout="synthetic", source="unit-test")
    out = q3._percentile_frames(traj)
    assert out.loc[0, "tail_frac_above_15um"] > 0        # the coarse mass is reported...
    assert out.loc[0, "d50_um"] < psd.VALID_SIZE_MAX_UM  # ...but does not set D50


def test_matched_extent_balancing_weights_units_not_runs():
    import media_diagnostic_q3 as q3
    rows = []
    for unit, n_reps, d50 in (("A", 1, 3.0), ("B", 3, 5.0)):
        for rep in range(1, n_reps + 1):
            rows.append({"condition": "X", "unit": unit, "rep": rep, "extent_g": 0.5,
                         "d10_um": 1.0, "d50_um": d50, "d90_um": 9.0,
                         "tail_frac_above_15um": 0.1})
    out = q3.balanced(pd.DataFrame(rows))
    assert out.loc[0, "n_units"] == 2
    assert np.isclose(out.loc[0, "d50_um_mean"], 4.0)    # unweighted (3+5)/2, not run-weighted 4.5


def test_separation_reports_an_equivalence_bound_not_just_a_difference():
    import media_diagnostic_q3 as q3
    tbl = pd.DataFrame([
        {"condition": "lo", "extent_g": 0.5, "d50_um_mean": 3.0, "d50_um_sd": 0.2, "n_units": 4},
        {"condition": "hi", "extent_g": 0.5, "d50_um_mean": 3.02, "d50_um_sd": 0.2, "n_units": 4},
    ])
    sep = q3.separation_at_extent(tbl, "lo", "hi")
    assert "detectable_diff_um" in sep                    # the bound, so a null is not overread
    assert np.isclose(sep.loc[0, "d50_diff_um"], 0.02, atol=1e-9)
    assert sep.loc[0, "detectable_diff_um"] > abs(sep.loc[0, "d50_diff_um"])


@corpus
def test_arm_b_forward_prediction_covers_every_run_with_the_primary_cs():
    import arm_b_cs
    from forward_predict import _arm_b_runs
    runs, base, _ = _arm_b_runs()
    assert len(runs) == 27
    cs = arm_b_cs.cs_map()
    assert {r["cs_ugml"] for r in runs} == set(cs.values())
    assert all(r["dose_mg"] > 0 for r in runs)
    assert len({r["prep"] for r in runs}) == 5      # 07-02, 07-07, 07-22, 07-23, 07-24
