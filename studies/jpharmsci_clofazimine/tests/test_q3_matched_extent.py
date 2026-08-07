"""Synthetic tests for the Track-1 q3 matched-extent pipeline.

Covers the correctness invariants the reviewer required: timestamp matching, cumulative-Q3 percentile
recovery, matched-g interpolation, per-target rejection reasons, run/rep provenance, the coarse-tail
instability flag + D90 invalidation, and day-level aggregation. Pure-logic tests use hand-built
synthetic inputs (no raw data). One data-gated test asserts the detector g coordinate is byte-identical
to the pre-inversion channel module's preprocessing.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from diffractomorph_pipeline import psd

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
sys.path.insert(0, str(ANALYSIS))

qme = pytest.importorskip("q3_matched_extent")


def test_manuscript_pairing_tolerance_is_half_nominal_cadence():
    assert qme.MATCH_TOL_S == pytest.approx(6.0)


# ── package pure helpers ────────────────────────────────────────────────────────────
def test_timestamp_matching_exact_and_unmatched():
    # detector at 0,11,22,33 s; q3 offset +0.5 s, plus one stray q3 frame far away
    t_det = np.array([0.0, 11.0, 22.0, 33.0])
    t_q3 = np.array([0.5, 11.5, 22.5, 33.5, 500.0])
    pairs, unmatched_q3, unmatched_det = psd.match_frames_by_time(t_q3, t_det, tol_s=3.0)
    assert len(pairs) == 4                       # 1:1 within tolerance
    assert unmatched_q3 == [4]                   # the stray frame has no detector partner
    assert unmatched_det == []
    assert all(abs(dt - 0.5) < 1e-9 for _, _, dt in pairs)   # signed t_q3 - t_det
    # tightening tolerance below the offset rejects everything
    p2, uq2, ud2 = psd.match_frames_by_time(t_q3, t_det, tol_s=0.1)
    assert p2 == [] and len(ud2) == 4


def test_q3_percentile_recovery_from_cumulative():
    xo = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    cum = np.array([0.0, 25.0, 50.0, 75.0, 100.0])           # D50 sits exactly on xo=4
    d10, d50, d90 = psd.q3_percentiles(xo, cum)
    assert d50 == pytest.approx(4.0, rel=1e-9)
    assert 1.0 < d10 < 2.0 and 8.0 < d90 < 16.0
    # a percentile the cumulative cannot bracket (first boundary already above p) → NaN, not clamped
    cum_hi = np.array([15.0, 40.0, 60.0, 80.0, 100.0])       # 10th pct is below the finest boundary
    assert np.isnan(psd.q3_percentiles(xo, cum_hi, (10.0,))[0])


def test_tail_fraction_and_restricted_renormalization():
    xo = np.array([1.0, 4.0, 15.0, 60.0, 120.0])
    cum = np.array([0.0, 40.0, 70.0, 90.0, 100.0])           # 30 % of volume sits above 15 µm
    assert psd.q3_tail_fraction(xo, cum, size_max=15.0) == pytest.approx(30.0, abs=1e-6)
    xr, cr = psd.restrict_cumulative(xo, cum, size_max=15.0)
    assert xr[-1] == pytest.approx(15.0) and cr[-1] == pytest.approx(100.0)   # renormalized to 100 %
    assert xr.max() <= 15.0 + 1e-9


def test_interp_cumulative_at_g():
    g_env = np.array([1.0, 0.7, 0.5])
    cum = np.array([[0, 50, 100.0], [0, 40, 100.0], [0, 30, 100.0]])
    res = psd.interp_cumulative_at_g(g_env, cum, target=0.6)
    assert res is not None
    g_at, cum_at, j, frac = res
    assert j == 2 and frac == pytest.approx(0.5)             # halfway between g=0.7 and g=0.5
    assert cum_at[1] == pytest.approx(35.0)                  # mean of 40 and 30
    assert psd.interp_cumulative_at_g(np.array([1.0, 0.9]), cum[:2], target=0.5) is None  # never reached


# ── driver: synthetic run builders ──────────────────────────────────────────────────
def _frames(n, cum_row, xo, t0=0.0, dt=11.0):
    """A Q3Frames with the same cumulative shape every frame (so anchor == shape → TV=0, Δlog D50=0)."""
    Q3 = np.tile(np.asarray(cum_row, float), (n, 1))
    return psd.Q3Frames(t_epoch=t0 + dt * np.arange(n), xo=np.asarray(xo, float),
                        Q3_cum=Q3, xm=np.asarray(xo, float), dQ3=np.diff(Q3, axis=1, prepend=0.0) / 100.0,
                        source="synthetic")


def _det(n, g_end=0.4, s0=1000.0, sigma=1.0, t0=0.0, dt=11.0, copt=5.0):
    g_env = np.linspace(1.0, g_end, n)
    return dict(epoch=t0 + dt * np.arange(n), total=g_env * s0, g_env=g_env, s0=s0, sigma=sigma,
                copt=np.full(n, copt), n=n)


def test_reason_ok_and_metrics():
    xo = [1.0, 2.0, 4.0, 8.0, 16.0]
    cum = [0.0, 25.0, 50.0, 75.0, 100.0]
    frames, det = _frames(10, cum, xo), _det(10, g_end=0.4)
    res = qme.matched_extent_run(frames, det, copt_floor_absolute=0.79)
    assert res[0.8]["reason"] == "ok"
    assert res[0.8]["D50"] == pytest.approx(4.0, rel=1e-6)
    assert res[0.8]["tv_distance"] == pytest.approx(0.0, abs=1e-9)     # shape constant over time
    assert res[0.8]["dlog10_D50"] == pytest.approx(0.0, abs=1e-9)


def test_reason_not_reached_vs_below_noise():
    xo = [1.0, 2.0, 4.0, 8.0, 16.0]
    cum = [0.0, 25.0, 50.0, 75.0, 100.0]
    # g only descends to 0.75 → deep targets are NOT reached (the shallow-loss / pH-5 case)
    shallow = qme.matched_extent_run(_frames(10, cum, xo), _det(10, g_end=0.75), copt_floor_absolute=0.79)
    assert shallow[0.6]["reason"] == "not_reached"
    assert shallow[0.4]["reason"] == "not_reached"
    # huge noise floor → even a reached target is rejected as below_noise (distinct reason)
    noisy = qme.matched_extent_run(_frames(10, cum, xo), _det(10, g_end=0.2, sigma=1e6),
                                   copt_floor_absolute=0.79)
    assert noisy[0.8]["reason"] == "below_noise"


def test_reason_q3_unreliable_below_floor():
    xo = [1.0, 2.0, 4.0, 8.0, 16.0]
    cum = [0.0, 25.0, 50.0, 75.0, 100.0]
    det = _det(12, g_end=0.2)
    det["copt"] = np.concatenate([np.full(4, 10.0), np.full(8, 1.0)])   # signal collapses after frame 4
    res = qme.matched_extent_run(_frames(12, cum, xo), det, copt_floor_absolute=3.0)
    # g=0.8 is crossed while copt still high (reliable); a deep target is crossed after the collapse
    assert res[0.8]["reason"] == "ok"
    assert res[0.2]["reason"] == "q3_unreliable"


def test_tail_instability_flag_and_d90_invalidation():
    xo = [1.0, 4.0, 15.0, 60.0, 120.0]
    cum = [0.0, 40.0, 70.0, 90.0, 100.0]                     # 30 % above 15 µm, D90 in the tail
    result = qme.matched_extent_run(
        _frames(10, cum, xo), _det(10), copt_floor_absolute=0.79,
    )
    m = result[0.8]
    assert m["tail_unstable"] is True
    assert m["coarse_review"] is True
    assert np.isnan(m["D90"]) and np.isnan(m["span"])        # D90/span invalidated in the tail
    assert np.isfinite(m["D50_restricted"])                  # restricted-range view still reported


def test_provenance_rep_and_run_id_in_rows():
    xo = [1.0, 2.0, 4.0, 8.0, 16.0]
    cum = [0.0, 25.0, 50.0, 75.0, 100.0]
    runs = [("pH 4.0", 20260608, 2, "pH4.0_20260608_R2", _frames(10, cum, xo), _det(10))]
    rows = qme._evaluate(runs, floor=0.79)
    assert rows and all(r["rep"] == 2 and r["run_id"] == "pH4.0_20260608_R2" for r in rows)


def test_day_level_aggregation_collapses_reps():
    # two reps same date → one per-day value (their mean); a second date → second value
    ok = [dict(date=20260608, condition="pH 4.0", D50=2.0),
          dict(date=20260608, condition="pH 4.0", D50=4.0),
          dict(date=20260609, condition="pH 4.0", D50=3.0)]
    vals = qme._day_values(ok, "D50")
    assert sorted(vals.tolist()) == [3.0, 3.0]               # (2+4)/2 = 3 for day 1; 3 for day 2


def test_condition_test_uses_shared_dates_only():
    # pH4 on days 1,2,3 ; pH4.5 on days 2,3,4 → shared blocks are days 2,3 only
    ok = []
    for d, v in [(1, 1.0), (2, 1.1), (3, 1.2)]:
        ok.append(dict(condition="pH 4.0", date=d, dlog10_D50=v))
    for d, v in [(2, 2.0), (3, 2.1), (4, 2.2)]:
        ok.append(dict(condition="pH 4.5", date=d, dlog10_D50=v))
    res = qme._condition_test(ok, "dlog10_D50")
    assert res is not None
    # date × condition units (NOT independent days): days 2,3 × 2 conds = 4; unshared dates dropped
    assert res["n_date_condition_units"] == 4
    assert res["n_unique_date_blocks"] == 2
    assert res["n_conds"] == 2


def test_condition_test_permutation_is_exact_diagnostic():
    # 2 shared blocks (days 2,3), each with 2 conditions → 2! * 2! = 4 distinct arrangements
    ok = []
    for d, v in [(2, 1.0), (3, 1.2)]:
        ok.append(dict(condition="pH 4.0", date=d, dlog10_D50=v))
    for d, v in [(2, 2.0), (3, 2.1)]:
        ok.append(dict(condition="pH 4.5", date=d, dlog10_D50=v))
    res = qme._condition_test(ok, "dlog10_D50")
    assert res["diagnostic_only"] is True
    assert res["n_distinct_perms"] == 4                      # exact enumeration, not Monte-Carlo
    assert res["min_attainable_p"] == pytest.approx(0.25)    # 1/4
    # exact p is a multiple of 1/N (enumerated, deterministic — no RNG jitter)
    assert res["exact_perm_p_provisional"] in (0.25, 0.5, 0.75, 1.0)


# ── data-gated: detector g coordinate identical to the channel matched-g module ───────
def test_detector_g_identical_to_channel_module():
    from psd_evolution_common import iter_runs
    prm = pytest.importorskip("psd_redistribution_matched_g")
    first = next(iter(iter_runs()), None)
    if first is None:
        pytest.skip("pH-study data not present")
    rtf = first[3]
    det = qme.detector_series(rtf)
    X, _t = prm._run_matrix(rtf)                             # channel module's own preprocessing
    anchor = np.nanmedian(X[:qme.ANCHOR_N], axis=0)
    g_ref = np.minimum.accumulate(X.sum(1) / anchor.sum())
    assert det is not None
    np.testing.assert_allclose(det["g_env"], g_ref, rtol=1e-9, atol=1e-9)
