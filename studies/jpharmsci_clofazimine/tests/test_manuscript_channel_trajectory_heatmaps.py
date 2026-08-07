"""Tests for the revised detector-channel trajectory analysis
(``analysis/manuscript_channel_trajectory_heatmaps.py``).

Confirms the revision: the calibrated noise surface is passed to ``noise_filter``; no Copt eligibility
cutoff is used (Copt only via the despike API); correlations are computed from the cleaned
reference-adjusted trajectories (``G_X``), NOT the normalized ratios; zero/near-zero initial channels are
never divided by their initial value; low-initial channels (1–4) remain in the separate absolute-signal
audit; the primary heatmap uses one common eligible channel set; date-first aggregation is preserved;
deliberately-excluded channels are NOT represented as failed frame-reliability QC; and non-estimable
``t50`` values stay missing (never extrapolated). Nothing here modifies production noise_filter/
noise_surface.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
sys.path.insert(0, str(ANALYSIS))

import manuscript_channel_trajectory_heatmaps as H  # noqa: E402

NCH = H.N_CH


def _synth_run(ph, date, rep, GX, reliable):
    """Minimal run dict matching build_run's revised output. ``GX`` is the grid×31 reference-adjusted
    trajectory; ``reliable`` = channels with a reliable initial value (X0 from GX[0])."""
    GX = np.asarray(GX, float)
    ng = GX.shape[0]
    X0 = np.where(np.isin(np.arange(1, NCH + 1), list(reliable)),
                  np.nan_to_num(GX[0], nan=0.0), 0.0)
    norm_reliable = np.isin(np.arange(1, NCH + 1), list(reliable))
    metrics = [dict(channel=c, X0_initial=float(X0[c - 1]), max_post_initial=np.nan, t_max_min=np.nan,
                    final=float(GX[-1, c - 1]) if np.isfinite(GX[-1, c - 1]) else np.nan, auc_post=np.nan,
                    frac_frame_reliable=1.0, norm_reliable=bool(norm_reliable[c - 1]),
                    directional=bool(c in reliable), trajectory_class=H.TC_DECAY) for c in H.CHANNELS]
    return dict(ph=float(ph), date=int(date), rep=int(rep), X=GX, X0=X0, t=np.arange(ng) * 0.2,
                G_X=GX, G_raw=GX, G_clean=GX, G_frel=np.isfinite(GX), norm_reliable=norm_reliable,
                directional=sorted(reliable), metrics=metrics, frame_reliable_frac=1.0,
                noise_surface_params={"z_thresh": 4.0}, spike_frames=[46], spike_times=[8.7],
                event_spikes=[8.7], n_lead_dropped=2, n_directional=len(reliable))


def _stub_ingest(monkeypatch, I, ref, copt, t_min):
    monkeypatch.setattr(H.ingest, "extract_run",
                        lambda _p: types.SimpleNamespace(I=I, ref=ref, copt=copt, t_min=t_min,
                                                         channels=list(range(1, 32))))


# ── calibrated surface used; Copt only via despike API ────────────────────────
def test_calibrated_surface_passed_and_no_copt_cutoff(monkeypatch):
    from diffractomorph_pipeline.noise_surface import load_surface
    from diffractomorph_pipeline.noise_surface import _surface_path
    if not _surface_path().is_file():
        pytest.skip("study noise surface is supplied by the external data bundle")
    surface = load_surface()
    T = 20
    I = np.full((T, NCH), 0.5)
    for c in range(6, NCH):
        I[:, c] = 8.0 * np.exp(-np.arange(T) / 5.0) + 0.5
    _stub_ingest(monkeypatch, I, np.full(NCH, 0.5), np.linspace(20, 5, T), np.arange(T) * 0.2)
    r = H.build_run("rtf", 4.0, 1, 1, H.mqs._common_time_grid(), surface)
    assert r["noise_surface_params"] is not None and "z_thresh" in r["noise_surface_params"]
    import inspect
    body = H._strip = inspect.getsource(H.build_run)
    assert "noise_surface=surface" in body                     # calibrated path
    assert "copt=np.asarray(run.copt" in body                  # copt only as despike API arg
    assert "copt >" not in body and "copt <" not in body and "copt_floor" not in body


# ── correlation from cleaned reference-adjusted trajectories, not normalized ──
def test_correlation_uses_reference_adjusted_not_normalized():
    import inspect
    src = inspect.getsource(H.condition_correlation)
    assert 'r["G_X"]' in src                                   # correlates the reference-adjusted trajectory
    assert "Y_normalized" not in src and "/ r[\"X0\"]" not in src
    # numerically: correlation is amplitude-free (scaling a channel does not change r)
    grid = np.arange(0, 2.01, 0.2)
    base = np.exp(-grid)
    GX = np.full((len(grid), NCH), np.nan)
    GX[:, 9] = base; GX[:, 10] = 5.0 * base + 1e-3 * np.arange(len(grid))   # different amplitude
    runs = [_synth_run(4.0, d, 1, GX, {10, 11}) for d in (1, 2)]
    cc = H.condition_correlation(runs, [10, 11], "pearson")
    assert cc["n_dates"] == 2 and cc["r"][9, 10] > 0.9         # high r despite 5× amplitude difference


# ── zero/near-zero initial channels never divided by initial ──────────────────
def test_zero_initial_channel_never_divided():
    grid = np.array([0.0, 1.0, 2.0])
    GX = np.full((3, NCH), np.nan)
    GX[:, 9] = [2.0, 1.0, 0.5]                                  # channel 10 reliable
    GX[:, 0] = [0.0, 0.8, 0.4]                                  # channel 1 starts at ZERO, later positive
    run = _synth_run(4.0, 1, 1, GX, {10})                      # channel 1 NOT reliable-initial
    Gn = H._normalized_G(run, [1, 10])                         # eligible list includes ch1 (adversarial)
    assert np.all(np.isnan(Gn[:, 0]))                          # ch1 never normalized (no divide-by-zero)
    assert not np.any(np.isinf(Gn))
    assert Gn[0, 9] == pytest.approx(1.0)                      # ch10 normalized to its initial


# ── low-initial channels remain in the separate absolute audit ────────────────
def test_low_initial_channels_audited_separately():
    assert H.LOW_INIT_CHANNELS == [1, 2, 3, 4]
    runs = {ph: [_synth_run(ph, d, 1, np.tile(np.linspace(1, 0, 3)[:, None], (1, NCH)), {10})
                 for d in (1, 2)] for ph in H.CONDITIONS}
    run_df = H.low_initial_audit_run_level(runs)
    assert set(run_df.channel.unique()) == {1, 2, 3, 4}       # exactly channels 1–4
    assert "initial_ref_adjusted" in run_df and "trajectory_class" in run_df
    # date + summary levels build without error and keep the reproducibility flag
    summ = H.low_initial_audit_summary(H.low_initial_audit_date_level(run_df))
    assert "reproducible_late_emerging_low_angle_signal" in summ.columns


# ── primary heatmap uses one common eligible channel set ──────────────────────
def test_common_eligible_channel_set_is_condition_blind_intersection():
    # channel 10 reliable in all conditions/dates; channel 5 reliable only at pH 4.5/5.0 (not 4.0)
    runs = {}
    for ph in H.CONDITIONS:
        rel = {10, 11} | ({5} if ph in (4.5, 5.0) else set())
        runs[ph] = [_synth_run(ph, d, 1, np.ones((3, NCH)), rel) for d in (1, 2)]
    info = H.common_eligible_channels(runs)
    assert 10 in info["common"] and 11 in info["common"]
    assert 5 not in info["common"]                            # fails pH 4.0 → excluded from the common set
    assert 5 in info["per_condition"][4.5]                    # but eligible within pH 4.5
    assert isinstance(info["rule"], str) and "condition" in info["rule"].lower()


# ── date-first aggregation + excluded ≠ failed QC ─────────────────────────────
def test_date_first_aggregation_and_excluded_not_failed_qc():
    grid = np.array([0.0, 5.0])
    def GX(v):
        m = np.full((2, NCH), np.nan); m[:, 9] = [1.0, v]; return m
    runs = [_synth_run(4.0, 1, 1, GX(0.2), {10}), _synth_run(4.0, 1, 2, GX(0.4), {10}),
            _synth_run(4.0, 2, 1, GX(1.0), {10})]
    agg = H.aggregate_condition(runs, [10], grid)             # normalized: 0.2/0.4→date mean 0.3; 1.0
    assert agg["median"][1, 9] == pytest.approx(np.median([0.3, 1.0]))   # date-first (not raw-frame)
    assert agg["n_dates"][1, 9] == 2
    # source table: excluded low-initial channels are labelled excluded_low_initial, NOT below_frame_noise
    runs_by_ph = {4.0: runs, 4.5: [], 5.0: []}
    cond = {ph: (H.aggregate_condition(rs, [10], grid) if rs else
                 dict(median=np.full((2, NCH), np.nan), n_dates=np.zeros((2, NCH), int),
                      n_runs=np.zeros((2, NCH), int), date_mats={}, dates=[])) for ph, rs in runs_by_ph.items()}
    st = H.source_table(runs_by_ph, cond, [10], grid)
    ch1 = st[(st.aggregation_level == "run") & (st.channel == 1)]
    assert set(ch1.display_status.unique()) <= {"excluded_low_initial"}
    assert H.ST_NOISE not in set(ch1.display_status.unique())  # not mislabeled as failed frame-reliability


# ── source reproduces the displayed condition matrix ──────────────────────────
def test_source_reproduces_condition_matrix():
    grid = np.array([0.0, 5.0, 10.0])
    def GX(vals):
        m = np.full((3, NCH), np.nan); m[:, 9] = vals; return m
    runs = [_synth_run(4.0, 1, 1, GX([1.0, 0.6, 0.3]), {10}), _synth_run(4.0, 2, 1, GX([1.0, 0.8, 0.5]), {10})]
    rbp = {4.0: runs, 4.5: [], 5.0: []}
    cond = {ph: (H.aggregate_condition(rs, [10], grid) if rs else
                 dict(median=np.full((3, NCH), np.nan), n_dates=np.zeros((3, NCH), int),
                      n_runs=np.zeros((3, NCH), int), date_mats={}, dates=[])) for ph, rs in rbp.items()}
    st = H.source_table(rbp, cond, [10], grid)
    cd = st[(st.aggregation_level == "cross_date_median") & (st.ph == 4.0) & (st.channel == 10)]
    recon = cd.set_index("time_min").Y_normalized.to_dict()
    for gi, t in enumerate(grid):
        v = cond[4.0]["median"][gi, 9]
        if np.isfinite(v):
            assert recon[round(float(t), 3)] == pytest.approx(round(float(v), 5))


# ── t50: interpolated crossing; non-estimable stays missing ───────────────────
def test_t50_crossing_and_non_estimable_missing():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    # decays 1.0→0.25 crossing 0.5 between t=1 (0.8) and t=2 (0.4): 0.5 at 1 + (0.5-0.8)/(0.4-0.8) = 1.75
    x = np.array([1.0, 0.8, 0.4, 0.25]); x0 = 1.0
    assert H._t50(x, t, x0) == pytest.approx(1.75, abs=1e-6)
    # never crosses 50 % (pH-5.0-like) → NaN, NOT extrapolated
    flat = np.array([1.0, 0.95, 0.9, 0.85])
    assert np.isnan(H._t50(flat, t, x0))
    assert np.isnan(H._t50(x, t, 0.0))                        # zero initial → non-estimable


def test_t50_summary_keeps_non_crossing_missing():
    grid = H.mqs._common_time_grid()
    flat = np.tile(np.linspace(1.0, 0.9, len(grid))[:, None], (1, NCH))   # never crosses 50 %
    runs = {ph: [_synth_run(ph, d, 1, flat, {10, 11}) for d in (1, 2)] for ph in H.CONDITIONS}
    run_df, cond_df = H.t50_summary(runs, [10, 11])
    assert bool((~run_df.estimable).all())                    # none estimable
    assert cond_df.t50_median_min.isna().all()                # stays missing, not a large number
    assert (cond_df.n_channel_runs_never_crossed > 0).all()


# ── no size mapping / band routing ────────────────────────────────────────────
def test_no_size_mapping_or_band_routing():
    import inspect
    src = inspect.getsource(H)
    for banned in ("char_size", "channel_size_map", "diam_um", "particle_diameter", "size_um"):
        assert banned not in src
    assert "import band_routing" not in src and "route_channels" not in src


# ── FigureS low-initial absolute y-axis: shared limit must bound EVERY plotted trajectory ──
def _low_init_runs(grid, tall_ph=4.0, tall_value=2.4, base_value=0.5):
    """runs_by_ph where, for ``tall_ph``, one preparation date's channel-1 date-level trajectory reaches
    ``tall_value`` while the other two dates stay at ``base_value`` — so the DATE-LEVEL maximum exceeds the
    across-date-MEDIAN maximum (the exact condition the old median-only ymax clipped)."""
    ng = len(grid)
    runs_by_ph = {}
    for ph in H.CONDITIONS:
        runs = []
        for di, date in enumerate((1, 2, 3)):
            GX = np.zeros((ng, NCH))
            GX[:, 0] = tall_value if (ph == tall_ph and di == 0) else base_value   # channel 1
            GX[:, 1] = 0.3; GX[:, 2] = 0.2; GX[:, 3] = 0.1                          # channels 2–4 (small)
            runs.append(_synth_run(ph, date, 1, GX, {10, 11}))
        runs_by_ph[ph] = runs
    return runs_by_ph


def test_figureS_low_initial_yaxis_bounds_every_plotted_trajectory(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = H.mqs._common_time_grid()
    runs_by_ph = _low_init_runs(grid, tall_ph=4.0, tall_value=2.4, base_value=0.5)

    # capture each panel's y-limits at render time (the function closes the figure before returning)
    captured = {}
    real_close = plt.close
    def _capture(fig):
        captured["ylims"] = [tuple(ax.get_ylim()) for ax in fig.axes]
        return real_close(fig)
    monkeypatch.setattr(plt, "close", _capture)

    written, val = H.figure_low_initial_absolute(runs_by_ph, grid, tmp_path, ("png",))

    # (1) the constructed date-level maximum genuinely exceeds the cross-date-median maximum,
    #     and the OLD median-only ceiling (max_cross * 1.08) would have clipped it
    assert val["max_datelevel"] == pytest.approx(2.4)
    assert val["max_datelevel"] > val["max_cross"]
    assert val["max_datelevel"] > val["max_cross"] * 1.08          # old ymax would clip this date-level curve
    assert val["max_plotted"] == pytest.approx(2.4)

    # (3) shared upper limit strictly exceeds every finite plotted value; zero clipped
    assert val["y_upper"] > val["max_plotted"]
    assert val["y_upper"] == pytest.approx(2.5)                    # data-driven nice ceiling of 2.4
    assert val["n_clipped"] == 0

    # (2)/(4)/(5) rendered panels: all three identical, zero lower bound, upper bounds the max
    ylims = captured["ylims"]
    assert len(ylims) == 3                                         # three pH panels
    assert len(set(ylims)) == 1                                    # identical shared y-limits
    assert ylims[0][0] == 0.0                                      # zero lower bound retained
    assert ylims[0][1] == pytest.approx(val["y_upper"]) and ylims[0][1] > val["max_plotted"]
    assert (tmp_path / "FigureS_low_initial_channels_absolute_trajectories.png").exists()


def test_nice_ceiling_is_data_driven_and_strictly_above():
    for v, expect in [(2.37, 2.5), (2.4, 2.5), (1.6, 2.0), (0.42, 0.5), (2.5, 3.0)]:
        c = H._nice_ceiling(v)
        assert c == pytest.approx(expect)
        assert c > v                                              # always strictly above the data max
