"""Tests for the manuscript figure builders (:mod:`diffractomorph_pipeline.figures.manuscript`).

Two tiers:

* **Pure / synthetic** (always run): the shared ``r_c`` statistic, the deeper-``g`` guard, the
  nesting-before-condition-summary rule, early-only UV exclusion, and that every renderer writes
  PNG + PDF + a source CSV for every numerical panel — all on hand-built data, no corpus needed.
* **Corpus regression** (skipped when the data root is unconfigured): Figure 2's KWW values and
  Figure 3's plateau recovery reproduce the canonical CSVs, and matched-``g`` coverage reports three
  independent dates at ``g = 0.8``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from diffractomorph_pipeline.config import corpus
from figures import manuscript as M
import manuscript_figures as DRIVER
from ph_aggregate import CONFIG as PH_AGGREGATE_CONFIG, build_tables as build_ph_aggregate_tables

STUDY = corpus("disso_experiments", "ph_dependent_dissolution_study")
requires_corpus = pytest.mark.skipif(not STUDY.exists(), reason="pH-study corpus not present")


# ── pure: the shared r_c statistic (same equation for both Figure-5 panels) ──
def test_r_c_sums_to_zero_and_matches_formula():
    rng = np.random.default_rng(0)
    anchor = rng.uniform(1, 10, size=31)
    x = rng.uniform(0, 10, size=31)
    s0 = float(anchor.sum())
    g, r = M._r_c(x, anchor, s0)
    assert g == pytest.approx(x.sum() / s0)
    assert np.allclose(r, (x - g * anchor) / s0)
    assert r.sum() == pytest.approx(0.0, abs=1e-12)   # Σ_c r_c = 0 by construction


def test_endpoint_and_matched_use_the_same_r_c():
    # both Figure-5 panel builders route through _r_c; a constant matrix has zero residual everywhere
    X = np.ones((10, 31)) * 3.0
    ev = M._endpoint_vector(X)
    assert ev is not None and np.allclose(ev["r"], 0.0)
    # a matrix that decays proportionally (scalar × anchor) also has r_c ≈ 0 at every g
    anchor = np.linspace(1, 5, 31)
    X2 = np.vstack([anchor * s for s in np.linspace(1.0, 0.5, 12)])
    ev2 = M._endpoint_vector(X2)
    assert np.max(np.abs(ev2["r"])) < 1e-9


# ── pure: the deeper-g guard refuses a three-condition comparison below g = 0.8 ──
def test_deeper_g_comparison_refused():
    data = M.Figure5Data(
        endpoint_run=pd.DataFrame(), endpoint_date=pd.DataFrame(), endpoint_cond=pd.DataFrame(),
        endpoint_gend_date=pd.DataFrame(), endpoint_gend_cond=pd.DataFrame(),
        matched_run=pd.DataFrame(), matched_date=pd.DataFrame(), matched_cond=pd.DataFrame(),
        matched_gend_cond=pd.DataFrame(), coverage=pd.DataFrame(), conditions=["pH 4.0"])
    with pytest.raises(ValueError, match="only common-support extent"):
        M.build_deeper_g_comparison(data)


# ── pure: nested runs are averaged within date before the condition summary ──
def test_agg_profiles_averages_within_date_first():
    # two dates for one condition; date A has 2 runs, date B has 1. The condition mean must weight
    # the two DATE means equally (not the three runs).
    recs = [
        dict(condition="pH 4.0", date=1, rep=1, g=0.5, r=np.full(31, 0.0)),
        dict(condition="pH 4.0", date=1, rep=2, g=0.5, r=np.full(31, 2.0)),   # date-1 mean = 1.0
        dict(condition="pH 4.0", date=2, rep=1, g=0.5, r=np.full(31, 4.0)),   # date-2 mean = 4.0
    ]
    date_df, cond_df, run_df, gend_date, gend_cond = M._agg_profiles(recs)
    d1 = date_df[(date_df.date == 1) & (date_df.channel == 1)].r_c.iloc[0]
    assert d1 == pytest.approx(1.0)                    # nested runs averaged within the date
    cmean = cond_df[cond_df.channel == 1].r_c_mean.iloc[0]
    assert cmean == pytest.approx((1.0 + 4.0) / 2)     # equal-weight over the two DATE means, not 2.0
    assert int(cond_df.n_dates.iloc[0]) == 2
    assert int(gend_cond.n_dates.iloc[0]) == 2


# ── pure: early-only UV records cannot enter plateau recovery ──
def test_plateau_gate_excludes_early_only():
    assert M.PLATEAU_BASIS == "plateau(t>=10)"         # the eligibility flag Figure 3B filters on
    rec = pd.DataFrame({
        "basis": ["plateau(t>=10)", "single(~2min)", "plateau(t>=10)"],
        "recovery_corrected": [0.90, 0.10, 0.94],
    })
    eligible = rec[rec.basis == M.PLATEAU_BASIS]
    assert len(eligible) == 2 and 0.10 not in eligible.recovery_corrected.values
    # the excluded early-only value would drag the mean down if it leaked in
    assert eligible.recovery_corrected.mean() == pytest.approx(0.92)


# ── synthetic render: both formats + every numerical panel's source CSV ──
def _synthetic_fig2() -> M.Figure2Data:
    grid = np.linspace(0, 20, 21)
    conds = ["pH 4.0", "pH 4.5", "pH 5.0"]
    runs, date, cond, bcd_date = [], [], [], []
    for c in conds:
        for d in (1, 2, 3):
            for t, v in zip(grid, np.linspace(100, 20, len(grid))):
                date.append(dict(level="date_mean", condition=c, date=d, time_min=t, sigma_i_pct=v))
            bcd_date.append(dict(condition=c, date=d, mean_relax_min=2.0 + d * 0.1,
                                 optical_decay_depth_pct=80.0 - d, beta=0.9))
        for t, v in zip(grid, np.linspace(100, 20, len(grid))):
            cond.append(dict(level="condition", condition=c, time_min=t, mean_pct=v, sd_pct=2.0,
                             n_dates=3))
        runs.append(dict(level="run", condition=c, date=1, rep=1, id=f"{c}_1_R1",
                         time_min=0.0, sigma_i_pct=100.0))
    bcd = pd.DataFrame(bcd_date)
    bcd_cond = []
    for c in conds:
        sub = bcd[bcd.condition == c]
        row = {"condition": c, "n_dates": 3}
        for m in ("mean_relax_min", "optical_decay_depth_pct", "beta"):
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_sd"] = sub[m].std(ddof=1)
        bcd_cond.append(row)
    return M.Figure2Data(grid_t=grid, panelA_runs=pd.DataFrame(runs), panelA_date=pd.DataFrame(date),
                         panelA_cond=pd.DataFrame(cond), panelBCD_date=bcd,
                         panelBCD_cond=pd.DataFrame(bcd_cond), conditions=conds)


def test_render_figure2_writes_png_pdf_and_all_panel_sources(tmp_path):
    figs, srcs = tmp_path / "figures", tmp_path / "source_data"
    M.render_figure2(_synthetic_fig2(), figs, srcs, formats=("png", "pdf"))
    stem = "Figure_pH_angular_scattering_kinetics"
    for ext in ("png", "pdf"):
        assert (figs / f"{stem}.{ext}").exists()
    for panel in ("panelA_angular_trajectories", "panelB_mean_relaxation_time",
                  "panelC_optical_decay_depth", "panelD_stretch_exponent"):
        assert (srcs / f"{stem}_{panel}.csv").exists()


def _synthetic_fig3() -> M.Figure3Data:
    conds = ["pH 4.0", "pH 4.5", "pH 5.0"]
    phs = [4.0, 4.5, 5.0]
    a_runs, a_date, a_cond, b_date, b_cond, c_rows, t2, cs = [], [], [], [], [], [], [], []
    for c, ph in zip(conds, phs):
        for d in (1, 2, 3):
            for t in (2, 5, 10, 20):
                a_runs.append(dict(ph=ph, date=d, rep=1, time_min=t, dissolved_pct=50.0))
                a_date.append(dict(ph=ph, date=d, time_min=t, dissolved_pct=50.0))
            b_date.append(dict(ph=ph, date=d, recovery_pct=90.0, recovery_sd_within_date=1.0,
                               n_eligible_runs=3))
            c_rows.append(dict(ph=ph, date=d, optical_decay_depth_pct=80.0, uv_recovery_pct=90.0))
            t2.append(dict(condition=c, date=d, nested_run_count=3, mean_relax_min=2.0, beta=0.9,
                           optical_decay_depth_pct=80.0, n_plateau_eligible_uv_runs=3,
                           uv_plateau_recovery_mean=90.0, uv_plateau_recovery_sd_within_date=1.0,
                           eligibility_notes="3/3"))
        for t in (2, 5, 10, 20):
            a_cond.append(dict(ph=ph, time_min=t, mean_pct=50.0, sd_pct=2.0, n_dates=3))
        b_cond.append(dict(ph=ph, recovery_mean=90.0, recovery_sd=1.0, n_dates=3))
        cs.append(dict(condition=c, n_dates=3))
    return M.Figure3Data(panelA_runs=pd.DataFrame(a_runs), panelA_date=pd.DataFrame(a_date),
                         panelA_cond=pd.DataFrame(a_cond), panelB_date=pd.DataFrame(b_date),
                         panelB_cond=pd.DataFrame(b_cond), panelC=pd.DataFrame(c_rows),
                         table2=pd.DataFrame(t2), cond_summary=pd.DataFrame(cs), conditions=conds)


def test_render_figure3_writes_png_pdf_sources_tables_and_supplement(tmp_path):
    figs, srcs, tbls = tmp_path / "figures", tmp_path / "source_data", tmp_path / "tables"
    M.render_figure3(_synthetic_fig3(), figs, srcs, tbls, formats=("png", "pdf"))
    stem = "Figure_UV_dissolved_mass_and_recovery"
    for ext in ("png", "pdf"):
        assert (figs / f"{stem}.{ext}").exists()
        assert (figs / f"Figure_optical_decay_vs_uv_recovery.{ext}").exists()   # supplement
    for panel in ("panelA_uv_trajectories", "panelB_plateau_recovery", "panelC_optical_vs_uv"):
        assert (srcs / f"{stem}_{panel}.csv").exists()
    assert (tbls / "Table_date_level_endpoints.csv").exists()


def test_render_wires_one_recomputed_optical_table_into_figures_2_and_3(tmp_path, monkeypatch):
    fits = pd.DataFrame({"source": ["raw-rtf"]})
    by_date = pd.DataFrame({"source": ["current-circulation-start"]})
    by_condition = pd.DataFrame({"source": ["current-circulation-start"]})
    monkeypatch.setattr(
        DRIVER, "build_ph_aggregate_tables", lambda _root: (fits, by_date, by_condition)
    )
    seen = {}

    def fake_figure2(_root, *, fits, by_date):
        seen["figure2_fits"] = fits
        seen["figure2_by_date"] = by_date
        return SimpleNamespace(panelBCD_cond=pd.DataFrame({
            "condition": ["pH 4.0", "pH 4.5", "pH 5.0"],
            "mean_relax_min_mean": [1.13, 3.06, 2.27],
            "beta_mean": [0.845, 0.765, 0.952],
            "optical_decay_depth_pct_mean": [82.5, 85.8, 25.2],
        }))

    def fake_figure3(_root, *, by_date):
        seen["figure3_by_date"] = by_date
        return SimpleNamespace(panelB_cond=pd.DataFrame({
            "ph": [4.0, 4.5, 5.0], "recovery_mean": [89.7, 87.0, 33.4]
        }))

    monkeypatch.setattr(M, "figure2_data", fake_figure2)
    monkeypatch.setattr(M, "figure3_data", fake_figure3)
    monkeypatch.setattr(M, "render_figure2", lambda *args, **kwargs: {})
    monkeypatch.setattr(M, "render_figure3", lambda *args, **kwargs: {})
    figures_dir, tables_dir, source_dir = (
        tmp_path / "figures", tmp_path / "tables", tmp_path / "source_data"
    )
    for directory in (figures_dir, tables_dir, source_dir):
        directory.mkdir()
    _out, discrepancies = DRIVER._render(
        [2, 3], tmp_path, figures_dir, tables_dir, source_dir, ("pdf",)
    )
    assert discrepancies == []
    assert seen["figure2_fits"] is fits
    assert seen["figure2_by_date"] is by_date
    assert seen["figure3_by_date"] is by_date
    assert (tables_dir / "aggregate_kww_by_run.csv").is_file()
    assert (tables_dir / "aggregate_kww_by_independent_unit.csv").is_file()
    assert (tables_dir / "aggregate_kww_by_condition.csv").is_file()


def test_manuscript_loaders_require_current_optical_tables():
    with pytest.raises(TypeError):
        M.figure2_data("unused")
    with pytest.raises(TypeError):
        M.figure3_data("unused")


def test_figure5_provenance_lists_metadata_coverage_and_raw_rtfs(tmp_path):
    metadata = tmp_path / "summary" / "run_metadata.csv"
    coverage = tmp_path / "psd_evolution" / "redistribution_matched_g" / "matched_g_coverage.csv"
    raw = tmp_path / "ph_4.0" / "20260608_pH4" / "example measurement Rep 1.rtf"
    for path in (metadata, coverage, raw):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    inputs = DRIVER._input_files(tmp_path, [5])
    assert metadata in inputs
    assert coverage in inputs
    assert raw in inputs
    assert not any("angular_kww" in str(path) for path in inputs)


# ── corpus regression ────────────────────────────────────────────────────────
@requires_corpus
def test_figure2_reproduces_canonical_kww_values():
    fits, by_date, _ = build_ph_aggregate_tables(STUDY)
    cc = M.figure2_data(STUDY, fits=fits, by_date=by_date).panelBCD_cond.set_index("condition")
    assert cc.loc["pH 4.0", "mean_relax_min_mean"] == pytest.approx(1.13, abs=0.02)
    assert cc.loc["pH 4.5", "mean_relax_min_mean"] == pytest.approx(3.06, abs=0.02)
    assert cc.loc["pH 5.0", "mean_relax_min_mean"] == pytest.approx(2.27, abs=0.02)
    assert cc.loc["pH 4.0", "beta_mean"] == pytest.approx(0.845, abs=0.01)
    assert cc.loc["pH 5.0", "optical_decay_depth_pct_mean"] == pytest.approx(25.0, abs=1.5)
    assert all(cc["n_dates"] == 3)                    # between-date SD is over n = 3 dates


@requires_corpus
def test_figure2_condition_summary_is_between_date_not_pooled_runs():
    fits, by_date, _ = build_ph_aggregate_tables(STUDY)
    d = M.figure2_data(STUDY, fits=fits, by_date=by_date)
    sub = d.panelBCD_date[d.panelBCD_date.condition == "pH 4.0"]["mean_relax_min"]
    cc = d.panelBCD_cond.set_index("condition").loc["pH 4.0"]
    # mean over the three DATE values, and SD is the between-date SD (ddof=1, n=3)
    assert cc["mean_relax_min_mean"] == pytest.approx(float(sub.mean()), abs=1e-4)
    assert cc["mean_relax_min_sd"] == pytest.approx(float(sub.std(ddof=1)), abs=1e-4)


@requires_corpus
def test_figure3_plateau_recovery_reproduces_expected_all_runs_eligible():
    # After the UV kinetic-table parser fix (recovery_assay.read_timecourse) every run has a full
    # timecourse, so all 27 runs are plateau-eligible (no early-only exclusions) and the date-balanced
    # plateau recovery reflects the complete 9/9/9 design.
    _fits, by_date, _by_condition = build_ph_aggregate_tables(STUDY)
    d = M.figure3_data(STUDY, by_date=by_date)
    pb = d.panelB_cond.set_index("ph")
    assert pb.loc[4.0, "recovery_mean"] == pytest.approx(89.7, abs=2.0)
    assert pb.loc[4.5, "recovery_mean"] == pytest.approx(87.0, abs=2.0)
    assert pb.loc[5.0, "recovery_mean"] == pytest.approx(33.4, abs=2.0)
    t2 = d.table2
    assert (t2.n_plateau_eligible_uv_runs == t2.nested_run_count).all()   # none early-only
    assert (t2.nested_run_count == 3).all()                              # complete 3 reps per cell
    assert all(pb["n_dates"] == 3)


def test_ph_aggregate_declares_current_circulation_start_policy():
    assert PH_AGGREGATE_CONFIG.start_policy == "concordant_early_maximum"
    assert PH_AGGREGATE_CONFIG.start_acquisition_variable == "copt"
    assert PH_AGGREGATE_CONFIG.start_search_frames == 3
    assert PH_AGGREGATE_CONFIG.start_maximum_time_min == pytest.approx(1.0)
    assert PH_AGGREGATE_CONFIG.start_minimum_relative_increase == pytest.approx(0.20)
    assert PH_AGGREGATE_CONFIG.start_minimum_spectral_cosine == pytest.approx(0.995)


@requires_corpus
def test_current_ph_aggregate_selects_two_late_starts_and_matches_manuscript():
    by_run, _by_date, by_condition = build_ph_aggregate_tables(STUDY)
    assert int(by_run.start_index.gt(0).sum()) == 2
    selected = by_run[by_run.start_index.gt(0)]
    assert set(selected.condition) == {"pH 4.5"}
    cc = by_condition.set_index("condition")
    assert cc.loc["pH 4.5", "mean_relax_min"] == pytest.approx(3.061645, abs=1e-5)
    assert cc.loc["pH 4.5", "beta"] == pytest.approx(0.765130, abs=1e-5)
    assert cc.loc["pH 4.5", "optical_decay_depth_pct"] == pytest.approx(85.815161, abs=1e-5)


@requires_corpus
def test_figure5_run_universe_comes_from_run_metadata():
    endpoint, _matched = M._figure5_run_records(STUDY)
    metadata = pd.read_csv(M._paths(STUDY)["run_metadata"])
    expected = {
        (f"pH {float(row.ph):.1f}", int(row.date_i), int(row.rep))
        for row in metadata.itertuples()
    }
    observed = {(row["condition"], row["date"], row["rep"]) for row in endpoint}
    assert observed == expected


@requires_corpus
def test_matched_g_coverage_is_three_dates_at_g08_not_five_or_nine():
    cov = pd.read_csv(M._paths(STUDY)["mg_coverage"])
    g08 = cov[np.isclose(cov.target_g, 0.8)].set_index("condition")
    for cond in ("pH 4.0", "pH 4.5", "pH 5.0"):
        assert int(g08.loc[cond, "n_days"]) == 3        # three INDEPENDENT dates, not 5 or 9 units
    assert int(g08.loc["pH 5.0", "n_runs"]) == 5        # pH 5.0 spans 3 dates with 5 qualifying runs
    # deeper targets have no pH 5.0 support
    deep = cov[np.isclose(cov.target_g, 0.6) & (cov.condition == "pH 5.0")]
    assert int(deep.n_days.iloc[0]) == 0
