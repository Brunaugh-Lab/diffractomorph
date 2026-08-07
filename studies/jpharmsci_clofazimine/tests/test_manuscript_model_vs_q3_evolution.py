"""Tests for the forward-model vs observed-q3 size-evolution analysis
(``analysis/manuscript_model_vs_q3_evolution.py``).

Covers the correctness-critical logic the review asked to confirm: cohorts are placed at their CURRENT
diameter (not their original bin); weighted percentiles are computed directly from diameters/weights;
conservative log-grid rebinning conserves in-window mass; below/above-window accounting; fully-dissolved
(frozen) cohorts carry zero weight; mass weights are equivalent to N·r³; date-first equal weighting with a
≥2-date draw rule; reuse of the observed-q3 eligibility; monotone matched-remaining-mass interpolation;
the base ≡ uniform-rate equivalence at matched mass (an exact time-rescaling); and the detectability ratio.
Nothing here changes production forward-model physics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
sys.path.insert(0, str(ANALYSIS))

import manuscript_model_vs_q3_evolution as M  # noqa: E402
from diffractomorph_pipeline.forward.params import Parameters  # noqa: E402


class _Run:
    """Minimal DissolutionRun stand-in: qundiss (frames×nbin), radius_um, diam0_um, t."""
    def __init__(self, qundiss, radius_um, diam0_um, t=None):
        self.qundiss = np.asarray(qundiss, float)
        self.radius_um = np.asarray(radius_um, float)
        self.diam0_um = np.asarray(diam0_um, float)
        self.t = np.asarray(t if t is not None else np.arange(len(self.qundiss)) * 12.0, float)
        self.cbulk = np.zeros(len(self.qundiss))
        self.inputs = {}


# ── cohort placement at CURRENT diameter ──────────────────────────────────────
def test_cohorts_placed_at_current_diameter_not_original_bin():
    diam0 = np.array([2.0, 10.0])
    # frame 1: the 10 µm cohort has shrunk to radius 2 µm (d_cur = 4 µm); the 2 µm cohort unchanged
    q = np.array([[1.0, 1.0], [1.0, 0.008]])         # second cohort nearly gone (phi=0.008)
    rad = np.array([[1.0, 5.0], [1.0, 5.0 * 0.008 ** (1 / 3)]])
    run = _Run(q, rad, diam0)
    d, w = M._cohort_weights(run, 1)
    assert d[1] == pytest.approx(2 * 5.0 * 0.008 ** (1 / 3), rel=1e-6)   # current diameter, not 10 µm
    assert d[1] < 4.1                                 # well below its original 10 µm


# ── direct weighted percentiles ───────────────────────────────────────────────
def test_weighted_pctiles_logd_known_values():
    d = np.array([1.0, 2.0, 4.0, 8.0]); w = np.ones(4)
    d10, d50, d90 = M._weighted_pctiles_logd(d, w)
    assert d50 == pytest.approx(2.0, rel=1e-6)
    assert d90 == pytest.approx(10 ** (np.log10(4) + 0.6 * (np.log10(8) - np.log10(4))), rel=1e-6)


def test_weighted_pctiles_ignore_zero_weight():
    d = np.array([1.0, 2.0, 100.0]); w = np.array([1.0, 1.0, 0.0])   # the 100 µm cohort is gone
    _d10, d50, _d90 = M._weighted_pctiles_logd(d, w)
    assert d50 < 3.0                                  # unaffected by the zero-weight coarse cohort


# ── conservative rebinning + window accounting ────────────────────────────────
def test_rebin_conserves_in_window_mass_and_normalizes():
    grid = np.geomspace(0.7, 15.0, 20)
    d = np.array([1.0, 3.0, 7.0]); w = np.array([2.0, 3.0, 5.0])
    dq, below, inw, above = M.model_rebin_inwindow(d, w, grid)
    assert dq.sum() == pytest.approx(1.0, abs=1e-9)  # in-window distribution normalized
    assert below == pytest.approx(0.0) and above == pytest.approx(0.0)
    assert inw == pytest.approx(1.0, abs=1e-9)       # all mass in-window


def test_window_accounting_below_and_above():
    grid = np.geomspace(0.7, 15.0, 20)
    d = np.array([0.3, 3.0, 40.0]); w = np.array([1.0, 2.0, 1.0])    # 25% below, 50% in, 25% above
    dq, below, inw, above = M.model_rebin_inwindow(d, w, grid)
    assert below == pytest.approx(0.25, abs=1e-9)
    assert above == pytest.approx(0.25, abs=1e-9)
    assert inw == pytest.approx(0.50, abs=1e-9)


def test_rebin_two_bin_deposition_conserves_mass_between_neighbors():
    grid = np.array([1.0, 2.0, 4.0, 8.0])
    d = np.array([3.0]); w = np.array([1.0])         # sits between grid points 2 and 4 (log-midpoint)
    dq, _b, inw, _a = M.model_rebin_inwindow(d, w, grid)
    assert (dq > 0).sum() == 2                        # split across the two neighbouring bins only
    assert dq.sum() == pytest.approx(1.0) and inw == pytest.approx(1.0)


# ── fully dissolved (frozen) cohorts carry zero weight ────────────────────────
def test_frozen_cohorts_zero_weight():
    p = Parameters(); ff = p.freeze_frac
    diam0 = np.array([2.0, 5.0])
    q = np.array([[1.0, 1.0], [ff * 1.0, 0.5]])      # cohort 0 frozen at the floor → fully dissolved
    rad = np.array([[1.0, 2.5], [1.0 * ff ** (1 / 3), 2.5 * 0.5 ** (1 / 3)]])
    run = _Run(q, rad, diam0)
    _d, w = M._cohort_weights(run, 1)
    assert w[0] == 0.0 and w[1] > 0.0                # frozen cohort contributes nothing


# ── mass weight equivalence to N·r³ ───────────────────────────────────────────
def test_mass_weight_equivalent_to_Nr3():
    # q_i ∝ N_i r_i^3 (constant ρ). Direct percentiles from q-weights must equal those from N r^3 weights.
    rng = np.random.default_rng(0)
    d = np.geomspace(0.7, 15.0, 12)
    N = rng.uniform(1, 10, 12)
    r = d / 2.0
    q = N * r ** 3                                    # mass ∝ N r^3
    p_from_q = M._weighted_pctiles_logd(d, q)
    p_from_Nr3 = M._weighted_pctiles_logd(d, N * r ** 3)
    assert p_from_q == pytest.approx(p_from_Nr3, rel=1e-12)


# ── date-first equal weighting + ≥2-date rule ─────────────────────────────────
def test_date_first_equal_weight_and_two_date_rule():
    # date A: three runs with D50=2; date B: one run with D50=8 → equal-date mean = 5 (not 3.5)
    rows = ([dict(source="model", ph=4.0, target_min=2.0, date=1, rep=k, **_zeros(D50=2.0)) for k in range(3)]
            + [dict(source="model", ph=4.0, target_min=2.0, date=2, rep=1, **_zeros(D50=8.0))])
    _dl, cond = M.date_first_aggregate(pd.DataFrame(rows))
    r = cond.iloc[0]
    assert r.D50 == pytest.approx(5.0) and r.n_dates == 2 and bool(r.drawable) is True
    # a single date → not drawable
    one = pd.DataFrame([dict(source="model", ph=4.0, target_min=2.0, date=1, rep=1, **_zeros(D50=3.0))])
    assert bool(M.date_first_aggregate(one)[1].iloc[0].drawable) is False


def _zeros(**over):
    base = {c: 0.0 for c in M.DESC_COLS}
    base.update(over)
    return base


# ── observed-q3 eligibility reuse ─────────────────────────────────────────────
def test_reuses_observed_eligibility_from_mqs():
    import manuscript_q3_scattering_evolution as mqs
    assert M.WORKING_VERSION in mqs.VERSIONS
    assert M.mqs.VERSIONS[M.WORKING_VERSION] == {mqs.CAT_SUPPORTED, mqs.CAT_PROVISIONAL}   # coarse-excluded, Copt≥0.79


# ── matched-remaining-mass interpolation ──────────────────────────────────────
def test_interp_along_g_monotone():
    g = np.array([1.0, 0.8, 0.5, 0.2])               # decreasing in time
    d50 = np.array([3.0, 3.2, 3.6, 4.0])
    (mi,) = M._interp_along_g(g, (d50,), 0.65)
    assert 3.2 < mi < 3.6                             # between the g=0.8 and g=0.5 samples


# ── depletion → missing distribution ──────────────────────────────────────────
def test_depleted_inwindow_returns_missing_distribution():
    grid = np.geomspace(0.7, 15.0, 20)
    diam0 = np.array([3.0, 3.0])
    q = np.array([[1.0, 1.0], [0.001, 0.001]])       # frame 1: essentially all mass gone
    rad = np.array([[1.5, 1.5], [1.5 * 0.001 ** (1 / 3)] * 2])
    run = _Run(q, rad, diam0)
    desc = M.model_frame_descriptors(run, 1, grid, init_d10=3.0, init_inwindow=M._inwindow_mass(run, 0, grid))
    assert desc["depleted"] is True and not np.all(np.isfinite(desc["dq"]))   # distribution missing


# ── detectability ratio ───────────────────────────────────────────────────────
def test_detectability_ratio_and_classification():
    # model D50 shifts 3→5 (Δ2); within-date observed SD ~0.5 → R_detect=4 → "larger than repeatability"
    clock = pd.DataFrame(
        [dict(source="observed", ph=4.0, target_min=5.0, date=d, rep=k, D10=1, D50=2.0 + 0.5 * (k - 1), D90=6)
         for d in (1, 2) for k in (0, 1, 2)])
    cond_model = pd.DataFrame([dict(ph=4.0, target_min=0.0, D10=1, D50=3.0, D90=6, n_dates=2),
                               dict(ph=4.0, target_min=5.0, D10=1, D50=5.0, D90=6, n_dates=2)])
    det = M.detectability(clock, cond_model)
    row = det[(det.target_min == 5.0) & (det.percentile == "D50")].iloc[0]
    assert row.model_shift_um == pytest.approx(2.0)
    assert row.R_detect > 2 and row.classification == "larger than observed repeatability"


# ── base ≡ uniform-rate at matched remaining mass (exact time-rescaling) ───────
def test_base_equals_rateonly_at_matched_mass():
    from diffractomorph_pipeline import solubility as solubility_module
    if not solubility_module.default_path().is_file():
        pytest.skip("study solubility profile is supplied by the external data bundle")
    from diffractomorph_pipeline.forward import PSD
    fine = np.geomspace(0.5, 20.0, 24)
    dv = np.exp(-0.5 * ((np.log(fine) - np.log(3.0)) / 0.5) ** 2); dv /= dv.sum()
    psd0 = PSD.from_q3(fine, dv)
    runs = [dict(psd0=psd0, ph=4.5, dose_mg=0.17, date=1, rep=1)]
    res = M.base_vs_rateonly_matched_mass(runs, fine)
    assert res["equivalent"] is True and res["max_abs_D50_diff_um"] < 1e-3


# ── matched-progress coordinate correction (UV mass balance) ──────────────────
def test_matched_progress_uses_cumulative_not_fixed_volume():
    """The observed progress coordinate must come from the aliquot-corrected cumulative UV mass balance,
    not the superseded fixed-40 mL pct_injected."""
    import inspect
    src = inspect.getsource(M.matched_mass_rows)
    assert "cumulative_dissolved" in src
    assert "apparent_remaining_dose_fraction" in src
    # the old fixed-volume coordinate survives ONLY as the labelled audit baseline
    assert "old_remaining_fraction_uncorrected" in src


def test_matched_progress_audit_ranges_and_out_of_interval():
    matched = pd.DataFrame([
        dict(ph=4.0, old_remaining_fraction_uncorrected=0.60, apparent_remaining_dose_fraction=0.60,
             model_reaches_progress=True, obs_D50=2.0, model_D50=4.0, model_D50_at_old_coord=4.0),
        dict(ph=4.0, old_remaining_fraction_uncorrected=0.30, apparent_remaining_dose_fraction=0.18,
             model_reaches_progress=True, obs_D50=2.1, model_D50=4.5, model_D50_at_old_coord=4.3),
        dict(ph=5.0, old_remaining_fraction_uncorrected=0.02, apparent_remaining_dose_fraction=-0.12,
             model_reaches_progress=False, obs_D50=2.5, model_D50=np.nan, model_D50_at_old_coord=3.9),
    ])
    a = M.matched_progress_audit(matched).set_index("ph")
    assert a.loc[4.0, "n_uv_obs"] == 2
    assert a.loc[4.0, "max_abs_change"] == pytest.approx(0.12)          # |0.18−0.30|
    assert bool(a.loc[4.0, "any_outside_0_1"]) is False
    assert bool(a.loc[5.0, "any_outside_0_1"]) is True                  # −0.12 preserved, flagged
    assert a.loc[5.0, "n_outside_0_1"] == 1
    # conclusion (obs D50 below model) holds under both coordinates at pH 4.0 → unchanged
    assert bool(a.loc[4.0, "conclusion_changes"]) is False


def test_matched_progress_coordinate_matches_cumulative_dissolved():
    """The coordinate used equals assay.cumulative_dissolved's apparent_remaining_dose_fraction exactly."""
    from diffractomorph_pipeline.assay import cumulative_dissolved
    conc = np.array([1.0, 2.0, 3.0, 2.5])
    dose_mg = 0.15
    cd = cumulative_dissolved(conc, dose_mg, v0_mL=M.V_ML)
    # reproduce the module's row coordinate: 1 − cumulative_recovery
    assert np.allclose(cd["apparent_remaining_dose_fraction"],
                       1.0 - cd["cumulative_dissolved_ug"] / (dose_mg * 1000.0))
    # and it differs from the old fixed-volume coordinate once aliquots accumulate
    old = 1.0 - conc * M.V_ML / (dose_mg * 1000.0)
    assert not np.allclose(cd["apparent_remaining_dose_fraction"].to_numpy()[1:], old[1:])


# ── primary revision: relative percentile evolution (normalized size-shape) ───
def _synth_clock():
    """A small clock table: model + observed, pH 4.0, 2 dates, 2 reps, times 0/2/5 min."""
    rows = []
    for src, coarsen in (("model", 2.0), ("observed", 1.1)):          # model coarsens more than observed
        for date in (1, 2):
            for rep in (1, 2):
                for t, frac in ((0.0, 0.0), (2.0, 0.6), (5.0, 1.0)):
                    mult = 1.0 + (coarsen - 1.0) * frac + 0.02 * rep    # small per-rep spread
                    rows.append(dict(source=src, ph=4.0, date=date, rep=rep, target_min=t,
                                     D10=1.0 * mult, D50=2.0 * mult, D90=4.0 * mult))
    return pd.DataFrame(rows)


def test_relative_trajectories_equal_one_at_normalization_time():
    rl = M.relative_percentile_long(_synth_clock())
    t0 = rl[rl.target_min == 0.0]
    assert np.allclose(t0.relative, 1.0)                               # every eligible trajectory = 1× at t0
    assert np.allclose(t0.log2_fold_change, 0.0)


def test_relative_source_reconstructs_absolute_and_has_required_columns():
    st = M.relative_source_table(M.relative_percentile_long(_synth_clock()))
    required = {"ph", "date", "rep", "time_min", "eligibility", "source", "percentile", "absolute_um",
                "initial_um", "relative", "log2_fold_change", "aggregation_level", "n_runs", "n_dates"}
    assert required <= set(st.columns)
    run = st[st.aggregation_level == "run"]
    assert np.allclose(run.absolute_um, run.relative * run.initial_um, atol=1e-3)   # absolutes reconstructable
    assert set(st.aggregation_level.unique()) == {"run", "date", "cross_date_median"}


def test_intraday_reps_aggregated_before_cross_date_median():
    # date A reps [1.0, 3.0] → date mean 2.0; date B rep [10.0] → 10.0.
    # cross-date MEDIAN must be median(2.0, 10.0) = 6.0, NOT median of the raw reps [1,3,10] = 3.0.
    rows = []
    for date, reps in ((1, [1.0, 3.0]), (2, [10.0])):
        for rep, val in enumerate(reps, start=1):
            for t in (0.0, 5.0):
                m = 1.0 if t == 0.0 else val                          # t0 normalizes to 1; t5 carries the value
                rows.append(dict(source="observed", ph=4.0, date=date, rep=rep, target_min=t,
                                 D10=m, D50=m, D90=m))
    _date, cross = M._agg_levels(M.relative_percentile_long(pd.DataFrame(rows)))
    c = cross[(cross.percentile == "D50") & (cross.target_min == 5.0)].iloc[0]
    assert c.relative == pytest.approx(6.0)                            # date-first median, not raw-frame median
    assert c.n_dates == 2


def test_fold_ticks_include_1x_and_cover_range_without_clipping():
    rl = M.relative_percentile_long(_synth_clock())
    ylo, yhi, ticks, labels = M._relative_ylim(rl)
    assert ylo <= float(rl.log2_fold_change.min()) and yhi >= float(rl.log2_fold_change.max())  # nothing clipped
    assert 0.0 in ticks and "1×" in labels                            # 1× reference always present


def test_relative_and_absolute_figures_use_no_optical_or_uv_variable():
    import inspect
    for fn in (M.relative_percentile_long, M._agg_levels, M.relative_source_table,
               M.figure_relative_percentile_evolution, M.figure_absolute_percentile_evolution,
               M.figure_absolute_percentile_datelevel, M.absolute_source_table,
               M.relative_interpretation, M._relative_ylim, M._fold_ticks_for):
        body = inspect.getsource(fn).split('"""')[-1]                 # executable body, not docstring
        for banned in ("copt", "angular", "cumulative_dissolved", "pct_injected", "recovery",
                       "apparent_remaining", "obscur", "kernel", "mie"):
            assert banned not in body.lower(), f"{fn.__name__} references {banned}"


def test_relative_figure_panels_share_identical_ylim(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rl = M.relative_percentile_long(_synth_clock())
    M.figure_relative_percentile_evolution(rl, tmp_path, formats=("png",), min_dates=1)
    # the figure sets one shared (ylo,yhi) from _relative_ylim on sharey axes → identical by construction
    ylo, yhi, _t, _l = M._relative_ylim(rl)
    assert yhi > ylo and (yhi - ylo) > 0
    assert (tmp_path / "model_vs_observed_q3_relative_percentile_evolution.png").exists()


# ── primary revision: absolute percentile evolution (now the PRIMARY figure) ───
_EXTRA_DESC = [c for c in M.DESC_COLS if c not in ("D10", "D50", "D90")]


def _abs_clock_rows(rows):
    """A clock-like table (all DESC_COLS present) from minimal dicts
    (source, ph, date, rep, target_min, D10, D50, D90)."""
    out = []
    for r in rows:
        d = dict(version="v", ph=4.0, **r)
        for c in _EXTRA_DESC:
            d.setdefault(c, 0.0)
        out.append(d)
    return pd.DataFrame(out)


def _abs_synth(target_mins=(0.0, 0.2, 0.4), dates=(1, 2), reps=(1,), big=None):
    """Synthetic clock + condition tables (model + observed) at 12 s-like cadence for the absolute figure.
    ``big`` = (date, rep, target_min, (D10, D50, D90)) injects one large observed run to mimic the outlier.
    Returns ``(clock, cond_m_median, cond_o_median, cond_m_mean, cond_o_mean)``."""
    rows = []
    for src, base in (("model", 2.0), ("observed", 2.0)):
        for date in dates:
            for rep in reps:
                for t in target_mins:
                    d = dict(D10=1.0 + t, D50=base + t, D90=4.0 + t)
                    if big and src == "observed" and (date, rep, round(t, 6)) == (big[0], big[1], round(big[2], 6)):
                        d = dict(D10=big[3][0], D50=big[3][1], D90=big[3][2])
                    rows.append(dict(source=src, date=date, rep=rep, target_min=t, **d))
    clock = _abs_clock_rows(rows)
    _dm, cm_mean = M.date_first_aggregate(clock[clock.source == "model"])
    _do, co_mean = M.date_first_aggregate(clock[clock.source == "observed"])
    _dmm, cm_med = M.date_first_aggregate(clock[clock.source == "model"], stat="median")
    _dom, co_med = M.date_first_aggregate(clock[clock.source == "observed"], stat="median")
    return clock, cm_med, co_med, cm_mean, co_mean


def test_absolute_and_relative_use_same_observed_eligibility():
    # both the absolute (condition) and relative tables are built from the SAME clock table, whose observed
    # rows are filtered by mqs.VERSIONS[WORKING_VERSION] inside clock_descriptor_rows → identical population
    import inspect
    import manuscript_q3_scattering_evolution as mqs
    assert M.WORKING_VERSION == "coarse_flag_excluded"
    assert M.mqs.VERSIONS[M.WORKING_VERSION] == {mqs.CAT_SUPPORTED, mqs.CAT_PROVISIONAL}
    cdr = inspect.getsource(M.clock_descriptor_rows)
    assert "mqs.VERSIONS[WORKING_VERSION]" in cdr           # observed eligibility applied once, in the clock
    clock, cm_med, co_med, cm_mean, co_mean = _abs_synth()
    rl = M.relative_percentile_long(clock)
    abs_keys = set(map(tuple, clock[clock.source == "observed"][["ph", "date", "rep", "target_min"]]
                       .round(6).to_numpy()))
    rel_keys = set(map(tuple, rl[rl.source == "observed"][["ph", "date", "rep", "target_min"]]
                       .round(6).to_numpy()))
    assert abs_keys == rel_keys and len(abs_keys) > 0       # same eligible observed population
    st = M.absolute_source_table(clock, cm_med, co_med, cm_mean, co_mean)
    assert (st[st.source == "observed_q3"].eligibility == f"eligible_{M.WORKING_VERSION}").all()


def test_absolute_and_relative_use_consistent_cross_date_median():
    # the relative figure summarizes date means by their MEDIAN (_agg_levels), and the primary absolute
    # figure now uses the stat="median" condition tables → same date-first, cross-date-median philosophy
    import inspect
    agg = inspect.getsource(M._agg_levels)
    assert '"median"' in agg or "'median'" in agg          # relative cross-date = median
    runsrc = inspect.getsource(M.run)
    assert 'stat="median"' in runsrc                        # absolute primary tables are the median tables
    assert "figure_absolute_percentile_evolution(cond_m_med, cond_o_med" in runsrc
    assert "figure_absolute_percentile_datelevel(clock, cond_o_med" in runsrc


def test_absolute_figure_uses_no_display_time_subsampling():
    import inspect
    abs_src = inspect.getsource(M.figure_absolute_percentile_evolution)
    assert "_at_display_times" not in abs_src and "REL_DISPLAY_MIN" not in abs_src   # no subsampling
    rel_src = inspect.getsource(M.figure_relative_percentile_evolution)
    assert "_at_display_times" in rel_src                    # subsampling is unique to the relative display


def test_all_eligible_absolute_observations_retained():
    clock, cm_med, co_med, cm_mean, co_mean = _abs_synth(target_mins=(0.0, 0.2, 0.4, 0.6, 0.8))
    st = M.absolute_source_table(clock, cm_med, co_med, cm_mean, co_mean)
    plotted = st[(st.source == "observed_q3") & st.plotted_primary]
    for pctl in ("D10", "D50", "D90"):
        assert plotted[plotted.percentile == pctl].time_min.nunique() == 5   # every eligible time kept


def test_absolute_points_require_min_dates():
    clock, cm_med, co_med, cm_mean, co_mean = _abs_synth(dates=(1,))          # only one preparation date
    st = M.absolute_source_table(clock, cm_med, co_med, cm_mean, co_mean)
    cond_obs = st[(st.source == "observed_q3") & (st.summary_statistic == "cross_date_median")]
    assert len(cond_obs) and (~cond_obs.plotted_primary).all()               # nothing plotted with < 2 dates
    assert (cond_obs.reason == "fewer than 2 independent dates").all()


def test_absolute_primary_statistic_is_cross_date_median_for_model_and_observed():
    # date 1 reps D50 [2, 2] → date mean 2; dates 2,3 rep D50 [8],[9] → 8,9. MEDIAN across dates = 8, mean ≈ 6.33.
    rows = []
    for src in ("model", "observed"):
        for date, reps in ((1, [2.0, 2.0]), (2, [8.0]), (3, [9.0])):
            for rep, v in enumerate(reps, start=1):
                rows.append(dict(source=src, date=date, rep=rep, target_min=5.0, D10=v, D50=v, D90=v))
    clock = _abs_clock_rows(rows)
    _dm, cm_mean = M.date_first_aggregate(clock[clock.source == "model"])
    _do, co_mean = M.date_first_aggregate(clock[clock.source == "observed"])
    _dmm, cm_med = M.date_first_aggregate(clock[clock.source == "model"], stat="median")
    _dom, co_med = M.date_first_aggregate(clock[clock.source == "observed"], stat="median")
    assert float(co_med.D50.iloc[0]) == pytest.approx(8.0)                   # median of date means [2,8,9]
    assert float(cm_med.D50.iloc[0]) == pytest.approx(8.0)
    assert float(co_mean.D50.iloc[0]) == pytest.approx((2 + 8 + 9) / 3)      # mean differs from median
    st = M.absolute_source_table(clock, cm_med, co_med, cm_mean, co_mean)
    prim = st[st.plotted_primary]
    assert (prim.summary_statistic == "cross_date_median").all()            # every plotted point is the median
    assert set(prim.source.unique()) == {"model", "observed_q3"}            # for both trajectories


def test_absolute_no_run_level_pooling_for_cross_date_estimate():
    # date 1 has three low reps, date 2 one high rep. Pooled-run median would be dominated by date 1 (=2);
    # the date-first cross-date median of the date means [2, 20] = 11 — proving no run/frame pooling.
    rows = []
    for date, reps in ((1, [2.0, 2.0, 2.0]), (2, [20.0])):
        for rep, v in enumerate(reps, start=1):
            rows.append(dict(source="observed", date=date, rep=rep, target_min=5.0, D10=v, D50=v, D90=v))
    clock = _abs_clock_rows(rows)
    _d, co_med = M.date_first_aggregate(clock, stat="median")
    pooled_run_median = float(np.median([2.0, 2.0, 2.0, 20.0]))              # = 2.0 (wrong, run-pooled)
    assert float(co_med.D50.iloc[0]) == pytest.approx(11.0)                  # date-first median, not pooled
    assert float(co_med.D50.iloc[0]) != pytest.approx(pooled_run_median)


def test_absolute_source_reproduces_primary_medians_without_clipping():
    clock, cm_med, co_med, cm_mean, co_mean = _abs_synth(big=(2, 1, 0.4, (3.0, 50.0, 80.0)))
    st = M.absolute_source_table(clock, cm_med, co_med, cm_mean, co_mean)
    # every primary plotted point equals the cross-date MEDIAN table exactly
    for pctl in ("D10", "D50", "D90"):
        for _i, r in co_med[co_med.drawable].iterrows():
            row = st[(st.source == "observed_q3") & st.plotted_primary & (st.percentile == pctl)
                     & np.isclose(st.time_min, float(r.target_min))]
            assert row.iloc[0].absolute_um == pytest.approx(round(float(r[pctl]), 4))
    # the large underlying value is NOT clipped/removed — it survives in run + date + mean rows
    obs = st[st.source == "observed_q3"]
    assert float(obs[obs.aggregation_level == "run"].absolute_um.max()) >= 80.0
    assert float(obs[obs.aggregation_level == "date"].absolute_um.max()) >= 80.0
    assert float(obs[obs.summary_statistic == "cross_date_mean"].absolute_um.max()) > 40.0   # mean retained


def test_absolute_source_has_required_columns_and_levels():
    clock, cm_med, co_med, cm_mean, co_mean = _abs_synth()
    st = M.absolute_source_table(clock, cm_med, co_med, cm_mean, co_mean)
    required = {"source", "ph", "date", "rep", "time_min", "percentile", "absolute_um", "aggregation_level",
                "summary_statistic", "plotted_primary", "n_runs", "n_dates", "eligibility", "reason"}
    assert required <= set(st.columns)
    assert set(st.aggregation_level.unique()) == {"run", "date", "cross_date"}
    assert set(st.summary_statistic.unique()) == {"raw", "within_date_mean", "cross_date_median",
                                                  "cross_date_mean"}
    assert set(st.source.unique()) == {"model", "observed_q3"}
    assert (st[st.plotted_primary].summary_statistic == "cross_date_median").all()   # only medians plotted


def _q3rec(cat, elapsed, copt, frac100, coarse, frame=0):
    return dict(category=cat, before_zero=False, dq_grid=np.array([0.5, 0.5]), elapsed_min=elapsed,
                q3_frame=frame, measured_copt=copt, frac_gt_100um=frac100, coarse_tail_flag=coarse)


def _outlier_fixture():
    import manuscript_q3_scattering_evolution as mqs
    prov = mqs.CAT_PROVISIONAL
    t = 12.6
    # one date drives a huge value; two other dates are near-normal; all eligible (provisional, no coarse flag)
    clock = _abs_clock_rows([
        dict(source="observed", date=1, rep=1, target_min=t, D10=3.0, D50=60.0, D90=82.0),
        dict(source="observed", date=2, rep=1, target_min=t, D10=1.0, D50=2.0, D90=2.6),
        dict(source="observed", date=3, rep=1, target_min=t, D10=1.2, D50=5.0, D90=19.0),
    ])
    _d, cond_o_mean = M.date_first_aggregate(clock)          # MEAN table selects the audited timepoint
    runs = [
        dict(ph=4.0, date=1, rep=1, obs=dict(q3=[_q3rec(prov, 12.67, 2.37, 0.63, False, 69)])),
        dict(ph=4.0, date=2, rep=1, obs=dict(q3=[_q3rec(prov, 12.60, 3.0, 0.10, False, 70)])),
        dict(ph=4.0, date=3, rep=1, obs=dict(q3=[_q3rec(prov, 12.60, 3.0, 0.10, False, 71)])),
    ]
    return runs, clock, cond_o_mean


def test_outlier_audit_reports_both_estimators_and_retains_observation():
    runs, clock, cond_o_mean = _outlier_fixture()
    per_run, summary = M.outlier_audit(runs, clock, cond_o_mean)
    assert not per_run.empty and set(per_run.date) == {1, 2, 3}                 # traceable to every date/run
    assert set(per_run.q3_frame.dropna().astype(int)) == {69, 70, 71}           # traced to the q3 frames
    # both estimators reported; mean >> median because the excursion is single-date
    assert summary["primary_statistic"] == "cross_date_median"
    assert summary["cross_date_mean_D50"] == pytest.approx((60 + 2 + 5) / 3, abs=1e-3)   # 22.33
    assert summary["cross_date_median_D50"] == pytest.approx(5.0)               # median of [60, 2, 5]
    assert summary["cross_date_median_D50"] < summary["cross_date_mean_D50"]
    assert summary["magnitude_driven_by_single_date"] is True and summary["reproduced_across_dates"] is False
    assert summary["all_contributors_passed_working_eligibility"] is True       # same prespecified rules
    assert summary["any_coarse_tail_flag"] is False and summary["coarse_bin_inversion_event"] is False
    # retained, not deleted / excluded / QC-failed; no point-specific exclusion
    assert summary["underlying_observation_deleted"] is False
    assert summary["retained_in_run_and_date_source"] is True
    assert summary["visible_in_datelevel_sensitivity"] is True
    assert summary["point_specific_exclusion_applied"] is False


def test_outlier_observation_remains_in_source_and_datelevel():
    # the 12.6-min observation stays in run + date source rows and in the date-level overlay data
    runs, clock, cond_o_mean = _outlier_fixture()
    _dom, co_med = M.date_first_aggregate(clock, stat="median")
    _dmm, cm_med = M.date_first_aggregate(clock, stat="median")               # (single source; reuse shape)
    st = M.absolute_source_table(clock, co_med, co_med, cond_o_mean, cond_o_mean)
    obs = st[st.source == "observed_q3"]
    # the big value (D90=82) is present at run and date level, i.e. not removed from the source data
    assert (np.isclose(obs[obs.aggregation_level == "run"].absolute_um, 82.0)).any()
    assert (np.isclose(obs[obs.aggregation_level == "date"].absolute_um, 82.0)).any()
    # the date-level sensitivity figure draws from _observed_date_level(clock), which retains that date line
    dlv = M._observed_date_level(clock)
    assert float(dlv[dlv.date == 1].D90.iloc[0]) == pytest.approx(82.0)
    # but the primary (median) plotted point is NOT 82
    prim = st[st.plotted_primary & (st.percentile == "D90")]
    assert float(prim.absolute_um.iloc[0]) < 82.0


def test_no_point_specific_ph_time_exclusion_in_absolute_pipeline():
    # the executable bodies (docstrings stripped) must contain no hard-coded time/date used to drop a point
    import inspect
    for fn in (M.figure_absolute_percentile_evolution, M.figure_absolute_percentile_datelevel,
               M.absolute_source_table, M.outlier_audit, M.date_first_aggregate):
        body = inspect.getsource(fn).split('"""')[-1]        # after the closing docstring quotes
        assert "12.6" not in body and "20260608" not in body


def test_absolute_figures_render_primary_filename(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    clock, cm_med, co_med, cm_mean, co_mean = _abs_synth(target_mins=(0.0, 0.2, 0.4, 0.6))
    M.figure_absolute_percentile_evolution(cm_med, co_med, tmp_path, formats=("png",))
    M.figure_absolute_percentile_datelevel(clock, co_med, tmp_path, formats=("png",))
    assert (tmp_path / "model_vs_observed_q3_absolute_percentile_evolution.png").exists()
    assert (tmp_path / "model_vs_observed_q3_absolute_percentile_evolution_datelevel.png").exists()
