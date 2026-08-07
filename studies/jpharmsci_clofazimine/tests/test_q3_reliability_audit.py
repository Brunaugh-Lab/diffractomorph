"""Tests for the corrected q3 signal-reliability audit and the packaged log-diameter Wasserstein
primitive (``psd.q3_wasserstein_log``).

Covers the correction-pass logic: contiguous segment identification (time-gap + Copt stationarity),
transition preservation, plateau-first equal-weight reference, catastrophic-tail event and contiguous
episode counting, dissolution run-specific 30%-of-peak translation, the raw-vs-background-subtracted
total-signal distinction, the relabelled stability ratio, and a regression guard that nonzero
frame-level >100 µm tails cannot be hidden by a zero segment median. Nothing here touches the
production ``frame_mask`` or any default QC.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
sys.path.insert(0, str(ANALYSIS))

from diffractomorph_pipeline import psd, kinetics  # noqa: E402
import q3_signal_reliability_audit as A  # noqa: E402

XO = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 120.0])
CUM = np.array([0.0, 5.0, 25.0, 60.0, 88.0, 97.0, 99.5, 100.0, 100.0])


class _Q3Stub:
    """Minimal stand-in for a Q3Frames with a single frame on grid ``xo``."""
    def __init__(self, xo, cum):
        self.xo = np.asarray(xo, float)
        self.Q3_cum = np.asarray([cum], float)


def _rec(ts, copt, cum=CUM, sess="1760 rpm", nom=5):
    return dict(session=sess, nominal_copt=nom, ts=float(ts), actual_copt=float(copt),
                q3=_Q3Stub(XO, cum), i_q3=0, run=None, j_det=0)


# ── packaged primitive ────────────────────────────────────────────────────────
def test_wasserstein_zero_and_hand_integral():
    assert psd.q3_wasserstein_log(XO, CUM, CUM) == pytest.approx(0.0, abs=1e-12)
    xo = np.array([1.0, 10.0, 100.0])
    a, b = np.array([0.0, 100.0, 100.0]), np.array([0.0, 0.0, 100.0])
    assert psd.q3_wasserstein_log(xo, a, b) == pytest.approx(1.0, abs=1e-9)


def test_tail_fractions_full_range_not_truncated():
    assert psd.q3_tail_fraction(XO, CUM, 15.0) == pytest.approx(3.0, abs=0.5)
    assert psd.q3_tail_fraction(XO, CUM, 100.0) == pytest.approx(0.0, abs=0.3)
    coarse = np.array([0.0, 1, 2, 4, 6, 9, 12, 25, 100.0])      # mass shoved into the coarse tail
    tail = psd.q3_tail_fraction(XO, coarse, 100.0)
    assert tail > 15.0 and tail > psd.q3_tail_fraction(XO, CUM, 100.0) + 10   # reported, never hidden


# ── segmentation: time-gap split + Copt stationarity + transition preservation ─
def test_time_gap_splits_into_contiguous_segments():
    recs = [_rec(0 + 12 * k, 5.0) for k in range(8)]             # block A, 8 stable frames
    recs += [_rec(2000 + 12 * k, 5.0) for k in range(6)]         # big gap, block B, 6 stable frames
    segs = A.segment_session(recs, gap_s=60.0, cv_stable=0.10, min_frames=5)
    assert len(segs) == 2                                        # gap split, not pooled
    assert all(s["classification"] == "stable" for s in segs)
    assert [s["n_frames"] for s in segs] == [8, 6]


def test_copt_drift_block_is_a_transition_and_is_preserved():
    recs = [_rec(12 * k, c) for k, c in enumerate(np.linspace(2.0, 12.0, 10))]   # ramp within one block
    segs = A.segment_session(recs, gap_s=60.0, cv_stable=0.10, min_frames=5)
    assert len(segs) == 1 and segs[0]["classification"] == "transition"          # not forced stable
    assert segs[0]["copt_min"] < 3 and segs[0]["copt_max"] > 11                   # range preserved


def test_small_stable_block_below_min_frames_is_transition():
    recs = [_rec(12 * k, 5.0) for k in range(3)]                 # only 3 frames
    segs = A.segment_session(recs, gap_s=60.0, cv_stable=0.10, min_frames=5)
    assert segs[0]["classification"] == "transition"


def test_segmentation_sensitivity_grid_shape():
    recs = [_rec(12 * k, 5.0) for k in range(10)]
    sens = A.segmentation_sensitivity(recs)
    assert len(sens) == len(A.GAP_SENS) * len(A.CV_SENS)
    assert {"gap_s", "copt_cv_stable", "n_stable", "n_transition"} <= set(sens.columns)


# ── plateau-first equal-weight reference ──────────────────────────────────────
def test_reference_weights_segments_equally_not_frames():
    fine = CUM
    coarse = np.array([0.0, 1, 3, 8, 20, 40, 70, 90, 100.0])     # a different (coarser) distribution
    # segment A: 30 fine frames; segment B: 3 coarse frames. Frame-pooling would be ~all-fine;
    # plateau-first (segment medians, equal weight) must sit halfway between fine and coarse.
    segA = dict(session="1760 rpm", nominal_copt=18, classification="stable",
                frames=[_rec(k, 20, fine, nom=18) for k in range(30)])
    segB = dict(session="1760 rpm", nominal_copt=25, classification="stable",
                frames=[_rec(1000 + k, 26, coarse, nom=25) for k in range(3)])
    ref = A.build_reference([segA, segB], XO, "1760 rpm", (18, 25))
    assert np.allclose(ref, 0.5 * (fine + coarse))
    assert not np.allclose(ref, fine)                            # would be the frame-weighted answer


# ── catastrophic-tail events + contiguous episode counting ────────────────────
def _fail_frame_df(pattern, sid="S", sess="1760 rpm"):
    rows = []
    for k, fail in enumerate(pattern):
        rows.append(dict(session=sess, nominal_copt=5, segment_id=sid, segment_class="stable",
                         q3_timestamp=pd.Timestamp(1000 + 12 * k, unit="s").isoformat(),
                         actual_copt=2.0, frac_gt_100um=(50.0 if fail else 0.0),
                         x50_um=(140.0 if fail else 6.5), last_bin_frac=(50.0 if fail else 0.0)))
    return pd.DataFrame(rows)


def test_failure_episode_counting_is_contiguous():
    df = _fail_frame_df([1, 1, 0, 1, 0, 0, 1, 1, 1])            # episodes: [0,1], [3], [6,7,8]
    seg = dict(session="1760 rpm", nominal_copt=5, segment_id="S", classification="stable")
    ev = A.failure_events(df, [seg], definition="frac100_gt1")
    r = ev.iloc[0]
    assert r.n_fail == 6 and r.n_episodes == 3 and r.max_episode_frames == 3


def test_failure_definition_sensitivity_counts_by_session():
    df = pd.concat([_fail_frame_df([1, 0, 1], sid="a", sess="1760 rpm"),
                    _fail_frame_df([0, 0, 0], sid="b", sess="1500 rpm")], ignore_index=True)
    fs = A.failure_definition_sensitivity(df).set_index("definition")
    assert fs.loc["frac100_gt1", "n_fail"] == 2
    assert fs.loc["frac100_gt1", "n_fail_1760"] == 2 and fs.loc["frac100_gt1", "n_fail_1500"] == 0


def test_regression_nonzero_frame_tail_not_hidden_by_zero_median():
    # a segment whose MEDIAN >100 µm is 0 but that contains real failing frames — must NOT read "absent"
    df = _fail_frame_df([0] * 38 + [1, 1, 1])                    # 3/41 fail; median frac100 = 0
    assert df.frac_gt_100um.median() == 0.0
    assert int((df.frac_gt_100um > 0).sum()) == 3               # frame-level presence survives
    assert int(A._flag(df, "frac100_gt1").sum()) == 3           # failure detector still flags them


# ── raw vs background-subtracted total signal; stability-ratio labelling ───────
def test_raw_and_bgsub_totals_are_distinct_observables():
    I = np.array([[10.0] * 31])
    ref = np.full(31, 3.0)
    raw = float(kinetics.total_signal(I)[0])                    # pipeline ΣI on raw I
    bg = float(np.clip(I - ref[None, :], 0, None).sum())        # Σ clip(I − Ref)
    assert raw == pytest.approx(310.0)
    assert bg == pytest.approx(217.0)
    assert raw != bg                                            # never compared as the same observable


def test_stability_ratio_not_labelled_snr():
    src = (ANALYSIS / "q3_signal_reliability_audit.py").read_text()
    assert "angular_stability_ratio" in src                     # relabelled
    assert "angular_snr" not in src                             # detector-SNR claim removed


# ── dissolution run-specific 30%-of-peak translation ──────────────────────────
def test_dissolution_translation_is_per_run_peak(tmp_path, monkeypatch):
    # stub ingest.extract_run to return runs with known peak Copt, and a matching folder tree
    base = tmp_path / "disso_experiments" / "ph_dependent_dissolution_study" / "ph_4.0" / "20260608_pH4"
    base.mkdir(parents=True)
    (base / "ameasurementRep1.rtf").write_text("x")
    (base / "ameasurementRep2.rtf").write_text("x")
    peaks = iter([20.0, 25.0])

    class _R:
        copt = None

    def fake(_p):
        r = _R(); r.copt = np.array([1.0, next(peaks), 5.0]); return r
    monkeypatch.setattr(A.ingest, "extract_run", fake)
    df = A.dissolution_peak_translation(tmp_path)
    assert set(df.copt_at_30pct.round(1)) == {6.0, 7.5}          # 30% of 20 and 25, per-run
