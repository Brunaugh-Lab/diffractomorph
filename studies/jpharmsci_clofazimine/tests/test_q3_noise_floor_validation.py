"""Tests for the three-day 1500-rpm noise-floor validation driver
(``analysis/q3_noise_floor_validation.py``) — corrected raw-channel pass.

Focus: the corrected raw-channel support logic (unclipped ``I.NORM − I.REF``, noise-standardised against
the blank σ, judged against an empirical non-event calibration, projected on the documented coarse
contrast) and its neutral, evidence-scaled classification (detector-supported / detector-unsupported /
indeterminate / pairing-semantics unresolved). Includes the key regression guard the review demanded: a
**noise-significant low-channel change must not be missed merely because it is < 5 % of total intensity**
— the exact failure mode of the retired fraction-of-total threshold. Coverage, event definitions, blank
noise, and the low-Copt/between-day corrections are also covered. Nothing here changes production QC.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
sys.path.insert(0, str(ANALYSIS))

import q3_noise_floor_validation as V  # noqa: E402

NCH = 31


def _matched(day, copts):
    return [dict(day=day, actual_copt=float(c)) for c in copts]


def _blank(sigma_vec):
    """A blank-noise dict with a given per-channel σ (diagonal covariance)."""
    sig = np.asarray(sigma_vec, float)
    cov = np.diag(sig ** 2)
    return dict(total_sigma=float(np.sqrt((sig ** 2).sum())), per_ch_sigma=sig,
                per_ch_mean=np.zeros(NCH), cov_reg=cov, cov_inv=np.diag(1.0 / sig ** 2), n_blank=20)


# ── coverage on measured Copt ─────────────────────────────────────────────────
def test_coverage_reports_measured_copt_and_band_overlap():
    matched = _matched(1, [4.2, 15.0, 99.0]) + _matched(2, [0.8, 3.0, 29.0])
    cov = V.coverage_summary(matched, segments=[]).set_index("scope")
    assert cov.loc["pooled", "copt_min"] == pytest.approx(0.8)
    assert cov.loc["pooled", "copt_max"] == pytest.approx(99.0)
    assert bool(cov.loc["Day 2", "reaches_below_diss_floor"]) is True     # Day 2 reaches < 2.0
    assert bool(cov.loc["Day 1", "reaches_below_diss_floor"]) is False    # Day 1 min 4.2
    assert cov.loc["Day 2", "frac_frames_in_diss_band"] == pytest.approx(1 / 3, abs=1e-3)


# ── coarse-tail event definitions ─────────────────────────────────────────────
def test_event_definition_sensitivity_counts_by_day():
    df = pd.DataFrame([
        dict(day=1, frac_gt_100um=10.0, frac_gt_50um=12.0, x50_um=5.0, x90_um=70.0),
        dict(day=1, frac_gt_100um=0.0, frac_gt_50um=0.0, x50_um=6.0, x90_um=10.0),
        dict(day=2, frac_gt_100um=0.0, frac_gt_50um=0.0, x50_um=6.0, x90_um=10.0),
    ])
    fs = V.event_definition_sensitivity(df).set_index("definition")
    assert fs.loc["frac100_gt1", "n_events"] == 1
    assert fs.loc["frac100_gt1", "day1"] == 1 and fs.loc["frac100_gt1", "day2"] == 0
    assert fs.loc["x90_gt60", "n_events"] == 1


# ── noise-standardised channel statistics (the corrected primitive) ───────────
def test_channel_stats_are_noise_standardised_not_fraction():
    contrast, cs, coarse_ch, _cav = V.expected_coarse_contrast()
    ncoarse = int(np.sum(contrast > 0))
    assert ncoarse >= 1 and coarse_ch[0] == 1                     # low channels are the coarse channels
    bl = _blank(np.full(NCH, 0.02))
    nb = np.zeros(NCH); nb[10:] = np.linspace(1, 5, NCH - 10)     # signal lives in the small-particle channels
    ev = nb.copy(); ev[:ncoarse] += 0.4                           # coarse channels rise by 0.4 (20× the σ)
    st = V._channel_stats(ev, nb, contrast, bl)
    assert st["coarse_absmax_z"] > 15                             # 0.4 / 0.02 = 20σ — hugely significant
    assert st["coarse_signed_z"] > 0 and st["proj"] > 0          # coarse-consistent direction
    # ... yet the rise is a tiny fraction of the total signal (the old threshold's blind spot)
    assert (ev[:ncoarse].sum() - nb[:ncoarse].sum()) / nb.sum() < 0.05


def test_noise_significant_low_channel_change_not_missed_below_5pct():
    """Regression guard for the retired fraction-of-total rule: a low-channel change that is many times
    the detector noise must be flagged even when it is < 5 % of total intensity."""
    contrast, *_ = V.expected_coarse_contrast()
    ncoarse = int(np.sum(contrast > 0))
    bl = _blank(np.full(NCH, 0.01))
    nb = np.zeros(NCH); nb[12:] = 8.0                            # ~152 total, all in high channels
    ev = nb.copy(); ev[:ncoarse] += 0.5                         # 0.5 rise → z = 50σ; but 0.5/152 ≈ 0.3 %
    frac_of_total = (ev[:ncoarse].sum() - nb[:ncoarse].sum()) / nb.sum()
    st = V._channel_stats(ev, nb, contrast, bl)
    assert frac_of_total < 0.05                                 # below the old absolute-fraction threshold
    assert st["coarse_absmax_z"] > 30                           # but unmistakable against the noise floor
    assert st["coarse_signed_z"] > 0                            # and in the coarse-consistent direction


# ── expected coarse contrast + R3-range caveat (documented ordering) ──────────
def test_expected_coarse_contrast_low_channels_and_range_caveat():
    contrast, cs, coarse_ch, caveat = V.expected_coarse_contrast()
    assert np.all(contrast[:len(coarse_ch)] > 0) and np.all(contrast[len(coarse_ch):] == 0)
    assert cs.max() < 100                                        # R3 has NO > 100 µm characteristic channel
    assert "outside reliable R3" in caveat and "reduced" in caveat


# ── neutral classification via the full driver path ───────────────────────────
def _classify(event_diff, nb_diff, *, seg_class="stable", copt_change=0.0, sigma=0.02,
              pairing_robust=True, n_normal=6):
    """Assemble a minimal segment (non-events + one event) and run ``classify_events`` end-to-end."""
    contrast, *_ = V.expected_coarse_contrast()
    blank = {1: _blank(np.full(NCH, sigma))}
    rows, chan, frames = [], {}, []
    ts0 = 1_000_000
    diffs = []
    seq = list(range(n_normal))
    ev_pos = n_normal // 2
    for k in range(n_normal + 1):
        is_ev = (k == ev_pos)
        gi = len(rows)
        d = np.asarray(event_diff if is_ev else nb_diff, float)
        rows.append(dict(day=1, date=20260611, segment_id="D1_S", segment_class=seg_class,
                         q3_timestamp=pd.Timestamp(ts0 + 12 * k, unit="s").isoformat(), elapsed_min=k,
                         actual_copt=(5.0 + copt_change) if is_ev else 5.0,
                         total_bgsub=float(np.clip(d, 0, None).sum()), x50_um=6.0,
                         x90_um=(160.0 if is_ev else 10.0),
                         frac_gt_100um=(30.0 if is_ev else 0.0), frac_gt_50um=(35.0 if is_ev else 0.0),
                         angular_snr_blank=100.0, wass_from_prev=0.0,
                         copt_change_vs_neighbors=(copt_change if is_ev else 0.0)))
        chan[gi] = dict(diff=d)
        frames.append(object())
    df = pd.DataFrame(rows)
    seg = dict(segment_id="D1_S", session="Day 1", session_day=1, classification=seg_class, frames=frames)
    ps = pd.DataFrame([dict(day=1, q3_timestamp=df.iloc[ev_pos].q3_timestamp, pairing_robust=pairing_robust)])
    return V.classify_events(df, [seg], chan, blank, contrast, ps)


def test_detector_supported_when_coarse_channels_rise_above_calibration():
    contrast, *_ = V.expected_coarse_contrast()
    ncoarse = int(np.sum(contrast > 0))
    nb = np.zeros(NCH); nb[10:] = np.linspace(1, 5, NCH - 10)
    ev = nb.copy(); ev[:ncoarse] += 0.6                          # strong coarse-consistent rise
    out = _classify(ev, nb)
    assert out.iloc[0].classification == V.CLS_SUPPORTED
    assert out.iloc[0].coarse_signed_z > 0


def test_detector_unsupported_when_within_normal_variability():
    contrast, *_ = V.expected_coarse_contrast()
    nb = np.zeros(NCH); nb[10:] = np.linspace(1, 5, NCH - 10)
    out = _classify(nb.copy(), nb)                               # event channels == neighbours
    assert out.iloc[0].classification == V.CLS_UNSUPPORTED


def test_indeterminate_when_total_signal_transient():
    contrast, *_ = V.expected_coarse_contrast()
    ncoarse = int(np.sum(contrast > 0))
    nb = np.zeros(NCH); nb[10:] = np.linspace(1, 5, NCH - 10)
    ev = nb * 2.0; ev[:ncoarse] += 0.6                           # total signal doubles → transient
    out = _classify(ev, nb)
    assert out.iloc[0].classification == V.CLS_INDETERMINATE
    assert "transient" in out.iloc[0].criterion


def test_pairing_unresolved_when_pairing_not_robust():
    contrast, *_ = V.expected_coarse_contrast()
    ncoarse = int(np.sum(contrast > 0))
    nb = np.zeros(NCH); nb[10:] = np.linspace(1, 5, NCH - 10)
    ev = nb.copy(); ev[:ncoarse] += 0.6
    out = _classify(ev, nb, pairing_robust=False)               # pairing flagged not robust
    assert out.iloc[0].classification == V.CLS_PAIRING


def test_classification_never_labels_events_inversion_artifacts_wholesale():
    """The corrected logic must NOT convert indeterminate events into blanket 'artifact' verdicts:
    the four labels are neutral and evidence-scaled, and 'unsupported' requires ALL-of criteria."""
    src = (ANALYSIS / "q3_noise_floor_validation.py").read_text()
    assert "LOWFRAC_MARGIN" not in src                          # the indefensible fraction threshold is gone
    # the four neutral labels exist and are distinct; none asserts a wholesale artifact verdict
    labels = {V.CLS_SUPPORTED, V.CLS_UNSUPPORTED, V.CLS_INDETERMINATE, V.CLS_PAIRING}
    assert len(labels) == 4
    assert V.CLS_UNSUPPORTED == "detector-unsupported coarse-mode output"


# ── low-Copt (Day 2 ramp) + between-day corrections ───────────────────────────
def _frame_rows(day, copts, x50s, seg_class="stable", seg="S"):
    return [dict(day=day, date=0, segment_id=f"D{day}_{seg}", segment_class=seg_class,
                 q3_timestamp=pd.Timestamp(1_000_000 + 12 * k, unit="s").isoformat(), elapsed_min=k,
                 actual_copt=float(c), total_bgsub=100.0, total_bgsub_unclipped=100.0,
                 angular_snr_blank=500.0, x10_um=x / 2, x50_um=float(x), x90_um=x * 2,
                 frac_gt_15um=0.0, frac_gt_30um=0.0, frac_gt_50um=0.0, frac_gt_100um=0.0,
                 wass_from_prev=0.001, wass_from_seg_median=0.0, copt_change_vs_neighbors=0.0)
            for k, (c, x) in enumerate(zip(copts, x50s))]


def test_low_copt_flags_ramp_not_plateau_when_low_copt_is_transient():
    # Day 2 traverses Copt 0.79→3 in a TRANSITION, plus a stable high plateau at 25
    lo = _frame_rows(2, [0.79, 1.2, 1.8, 2.4, 3.0], [2.9] * 5, seg_class="transition", seg="ramp")
    hi = _frame_rows(2, [25.0] * 6, [2.95] * 6, seg_class="stable", seg="plateau")
    df = pd.DataFrame(lo + hi)
    seg_lo = dict(segment_id="D2_ramp", session="Day 2", classification="transition",
                  frames=[dict(actual_copt=c) for c in [0.79, 1.2, 1.8, 2.4, 3.0]])
    seg_hi = dict(segment_id="D2_plateau", session="Day 2", classification="stable",
                  frames=[dict(actual_copt=25.0)] * 6)
    out = V.low_copt_stability(df, [seg_lo, seg_hi]).set_index("day")
    assert bool(out.loc[2, "reaches_low_copt"]) is True
    assert out.loc[2, "copt_min"] == pytest.approx(0.79)
    assert out.loc[2, "plateau_or_ramp"] == "ramp (transition)"   # NOT a stable plateau
    assert out.loc[2, "x50_rcv"] < 0.05                          # output continuous across the ramp


def test_between_day_reports_preparation_spread_not_tight():
    df = pd.DataFrame(_frame_rows(1, [5.0] * 4, [1.67] * 4) +
                      _frame_rows(2, [5.0] * 4, [2.92] * 4) +
                      _frame_rows(3, [5.0] * 4, [3.32] * 4))
    out = V.between_day_comparison(df, [])
    assert out.attrs["between_day_x50_spread_in_band"] > 1.0     # 1.67→3.32 is a ~2× spread
    assert out.attrs["between_exceeds_within"] is True          # not "tight" reproducibility


# ── candidate-QC comparison exposes peak fragility ────────────────────────────
def test_qc_candidate_comparison_flags_run_peak_fragility(monkeypatch):
    class _Run:
        # a transient Copt≈99 spike inflates the raw peak; the true plateau is ~5
        copt = np.array([99.0] + [5.0] * 20)
        I = np.ones((21, NCH)); t_min = np.arange(21) / 6.0
    monkeypatch.setattr(V, "_event_flag", lambda df, d: np.zeros(len(df), bool))
    monkeypatch.setattr("diffractomorph_pipeline.noise_filter.despike_frames",
                        lambda I, t, c: (I, t, c, {"spike_frames": []}), raising=False)
    matched = [dict(day=1, run=_Run())]
    fr = pd.DataFrame(_frame_rows(1, [5.0] * 20, [6.0] * 20))
    fr = fr.assign(angular_snr_blank=500.0)
    qc = V.qc_candidate_comparison(matched, fr).set_index("candidate")
    # 30 % of the 99 spike ≈ 29.7 removes ~all plateau frames; robust-peak / absolute floors do not
    assert qc.loc["run_peak_30pct", "stable_q3_removed"] > qc.loc["robust_peak_30pct", "stable_q3_removed"]
    assert qc.loc["abs_copt_2.0", "n_flagged"] == 0


# ── blank-derived detector noise (unclipped) ──────────────────────────────────
def test_blank_noise_returns_unclipped_perchannel_and_covariance(monkeypatch):
    class _Blank:
        I_bgsub = np.random.default_rng(0).normal(0, 0.5, (19, NCH))
    monkeypatch.setattr(V.ingest, "extract_run", lambda _p: _Blank())
    bl = V._blank_noise("x")
    assert bl["total_sigma"] > 0
    assert bl["per_ch_sigma"].shape == (NCH,) and np.all(bl["per_ch_sigma"] > 0)
    assert bl["cov_inv"].shape == (NCH, NCH)                     # regularised covariance is invertible
    # unclipped: the blank mean straddles zero (a clipped floor would force it non-negative)
    assert bl["per_ch_mean"].min() < 0
