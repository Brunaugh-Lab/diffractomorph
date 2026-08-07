"""Tests for the q3 / angular-scattering evolution manuscript figure module
(``analysis/manuscript_q3_scattering_evolution.py``).

Covers the correctness-critical logic the review asked to confirm: the five eligibility categories
(supported / provisionally_supported / review_required / outside_validation_range / pairing_or_qc_failure)
with the coarse-tail flag taken from the shared audit utility; date-first equal-weight aggregation with a
≥ 2-date draw rule (a one-date summary is never emitted as a condition mean); q3 differential-row
normalization before aggregation; the scattering normalization that preserves total-signal decline (fixed
initial-total denominator, NOT per-frame renormalization); that q3 eligibility never removes an otherwise
valid scattering frame; and the version filtering + sensitivity comparison. Nothing here touches
production QC, ``frame_mask``, the forward model, or the Mie operator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
sys.path.insert(0, str(ANALYSIS))

import manuscript_q3_scattering_evolution as E  # noqa: E402


# ── eligibility categories ────────────────────────────────────────────────────
def test_category_tiers_and_precedence():
    ok = dict(finite_ok=True, paired_ok=True, glitch=False)
    assert E._category(10.0, False, **ok) == E.CAT_SUPPORTED
    assert E._category(2.0, False, **ok) == E.CAT_PROVISIONAL
    assert E._category(10.0, True, **ok) == E.CAT_REVIEW            # coarse flag overrides the Copt tier
    assert E._category(0.5, False, **ok) == E.CAT_OUTSIDE           # Copt < 0.79
    # a coarse frame that is ALSO outside/failed is not "review" — outside/fail take precedence
    assert E._category(0.5, True, **ok) == E.CAT_OUTSIDE
    assert E._category(10.0, True, finite_ok=True, paired_ok=False, glitch=False) == E.CAT_FAIL
    assert E._category(10.0, False, finite_ok=True, paired_ok=True, glitch=True) == E.CAT_FAIL
    assert E._category(10.0, False, finite_ok=False, paired_ok=True, glitch=False) == E.CAT_FAIL


def test_versions_partition_coarse_only_between_intended_figures():
    assert E.CAT_REVIEW in E.VERSIONS["inclusive"]                  # coarse events kept
    assert E.CAT_REVIEW not in E.VERSIONS["coarse_flag_excluded"]   # coarse events dropped
    assert E.VERSIONS["stringent_copt4"] == {E.CAT_SUPPORTED}       # supported only
    # provisional (low-Copt) frames are in the first two versions but not the stringent one
    assert E.CAT_PROVISIONAL in E.VERSIONS["inclusive"]
    assert E.CAT_PROVISIONAL in E.VERSIONS["coarse_flag_excluded"]
    assert E.CAT_PROVISIONAL not in E.VERSIONS["stringent_copt4"]
    # outside / pairing-failure are excluded from EVERY version
    for cats in E.VERSIONS.values():
        assert E.CAT_OUTSIDE not in cats and E.CAT_FAIL not in cats


# ── date-first equal-weight aggregation + ≥2-date rule ────────────────────────
def test_date_first_equal_weight_not_frame_weighted():
    # date A: 3 reps at 0; date B: 1 rep at 6. Frame-pooling → 0.75; equal-date → 3.0.
    aligned = [("A", {0: np.array([0.0])}), ("A", {0: np.array([0.0])}),
               ("A", {0: np.array([0.0])}), ("B", {0: np.array([6.0])})]
    out = E._date_first_mean(aligned, min_dates=2)
    assert out[0]["mean"][0] == pytest.approx(3.0)
    assert out[0]["n_dates"] == 2 and out[0]["n_runs"] == 4


def test_single_date_never_drawn_as_condition_mean():
    out = E._date_first_mean([("A", {0: np.array([1.0])}), ("A", {0: np.array([3.0])})], min_dates=2)
    assert 0 not in out                                            # only one date → no condition mean


# ── grid alignment: nearest within tolerance, no extrapolation ────────────────
def test_align_matches_within_tolerance_only():
    grid = np.array([0.0, 0.2, 0.4])
    times = np.array([0.01, 0.55])                                 # 0.55 is > tol from any grid point
    vecs = [np.array([1.0]), np.array([2.0])]
    out = E._align_pairs(times, vecs, grid, tol=E.TOL_MIN)
    assert 0 in out and out[0][0] == 1.0                           # 0.01 ↔ grid 0.0
    assert 2 not in out                                            # 0.4 has no frame within tol (nearest 0.55)


# ── integration: build_run categories, scattering decline, no q3→scattering loss ─
class _StubRun:
    def __init__(self, xo, Q3_cum, copt, totals, t0):
        T = len(copt)
        self.copt = np.asarray(copt, float)
        self.t_min = np.arange(T) * (E.FRAME_S / 60.0)
        self.t0 = t0
        self.ref = np.zeros(E.N_CHANNELS)
        # build I so that Σ(I − ref) at frame k equals totals[k] (put it all in one channel)
        I = np.zeros((T, E.N_CHANNELS))
        I[:, 20] = np.asarray(totals, float)
        self.I = I


class _StubQ3:
    def __init__(self, xo, Q3_cum, t_epoch):
        self.xo = np.asarray(xo, float)
        self.Q3_cum = np.asarray(Q3_cum, float)
        self.dQ3 = np.diff(self.Q3_cum, axis=1, prepend=0.0) / 100.0
        self.xm = self.xo
        self.t_epoch = np.asarray(t_epoch, float)


def _fine_cum(xo):
    # fully undersize by ~15 µm (no >100 µm tail): supported/fine composition
    c = np.clip((np.log10(xo) - np.log10(xo[0])) / (np.log10(15.0) - np.log10(xo[0])) * 100.0, 0, 100)
    return c


def _coarse_cum(xo):
    # ~10 % of the volume sits above 100 µm → coarse-tail flag (> 1 %)
    c = _fine_cum(xo) * 0.9
    c[xo > 100.0] = 100.0
    return c


def _patch(monkeypatch, run, q3):
    monkeypatch.setattr(E.ingest, "extract_run", lambda _p: run)
    monkeypatch.setattr(E.psd, "read_q3_frames", lambda _p: q3)
    monkeypatch.setattr(E, "despike_frames", lambda I, t, c: (I, t, c, {"spike_frames": []}))


def test_build_run_categories_scattering_decline_and_independence(monkeypatch):
    xo = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 120.0, 240.0])
    t0 = pd.Timestamp("2026-06-08T10:00:00")
    copt = [10.0, 2.0, 0.5, 10.0]                                  # supported, provisional, outside, review(coarse)
    totals = [100.0, 80.0, 40.0, 20.0]                            # total angular signal declines over time
    Q3 = np.vstack([_fine_cum(xo), _fine_cum(xo), _fine_cum(xo), _coarse_cum(xo)])
    run = _StubRun(xo, Q3, copt, totals, t0)
    t_epoch = np.array([(t0 + pd.Timedelta(minutes=float(m))).timestamp() for m in run.t_min])
    q3 = _StubQ3(xo, Q3, t_epoch)
    _patch(monkeypatch, run, q3)

    out = E.build_run(4.0, 20260608, 1, "rtf", "fo", common_xo=xo)
    cats = [r["category"] for r in out["q3"]]
    assert cats == [E.CAT_SUPPORTED, E.CAT_PROVISIONAL, E.CAT_OUTSIDE, E.CAT_REVIEW]
    assert out["q3"][3]["coarse_tail_flag"] is True and out["q3"][0]["coarse_tail_flag"] is False

    # every q3 differential row is normalized (sums ~1) before aggregation
    for r in out["q3"]:
        assert abs(np.nansum(r["dq_grid"]) - 1.0) < 0.05

    # scattering preserves the total-signal decline: S sums track totals/total0 (1.0, 0.8, 0.4, 0.2)
    sums = [float(np.sum(f["S"])) / 100.0 for f in out["scattering"]]        # S is in %
    assert sums == pytest.approx([1.0, 0.8, 0.4, 0.2], abs=1e-6)

    # q3 eligibility (outside/review) must NOT remove any scattering frame
    assert len(out["scattering"]) == 4


# ── sensitivity comparison primitives ─────────────────────────────────────────
def test_cum_from_mean_dq_and_zero_self_distance():
    dq = np.array([0.1, 0.3, 0.4, 0.2])
    cum = E._cum_from_mean_dq(dq)
    assert cum[-1] == pytest.approx(100.0)                          # normalized fraction → 100 %
    xo = np.array([1.0, 2.0, 4.0, 8.0])
    assert E.psd.q3_wasserstein_log(xo, cum, cum) == pytest.approx(0.0, abs=1e-9)
