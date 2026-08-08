"""Tests for the Objective-1 operator-feasibility analysis + CAREER figure.

Two tiers: a vault-free pure-core tier that exercises the certified PSD, fixed manual geometry,
shape metrics, and resolution diagnostics; and a vault-gated tier
that reproduces the headline audit numbers from the real NIST measurement files (skipped in CI).
"""
import numpy as np
import pytest

from figures import objective1_feasibility as o1
from diffractomorph_pipeline.optics.standards import GRID

# One modest-resolution operator, built once, reused across the pure-core tests.
A_TEST = o1.build_physical_operator(n_quad=32)
N_CERT = o1.certified_number_psd()


# ── pure core: certified distribution ──────────────────────────────────────────────────────────

def test_certified_psd_normalized_and_confined():
    assert abs(N_CERT.sum() - 1.0) < 1e-9
    assert np.all(N_CERT >= 0)
    # The explicit 0--5 % extrapolated tail begins near 1.7 um; no submicron mass is injected.
    assert N_CERT[GRID < 1.5].sum() < 1e-6
    # essentially all number-weighted mass falls in the grid bins overlapping the certified band
    # (the 1.99 um bin, edges 1.8-2.2, legitimately catches the certified 2.1 um material; the d^-3
    # number weighting concentrates the PSD toward the small-diameter end of that band)
    assert N_CERT[(GRID >= 1.6) & (GRID <= 20.0)].sum() > 0.99
    assert N_CERT[GRID > 20.0].sum() < 1e-6


def test_certified_cdf_preserves_every_certificate_knot():
    assert np.allclose(o1.certified_cumulative(o1.CERT_D), o1.CERT_PCT, atol=1e-10)
    low, high = o1.certified_tail_anchors()
    assert low < o1.CERT_D[0] and high > o1.CERT_D[-1]
    assert o1.certified_cumulative(np.array([low, high])).tolist() == pytest.approx([0.0, 100.0])


def test_certified_cdf_tails_are_linear_in_log_diameter():
    low, high = o1.certified_tail_anchors()
    geometric_midpoints = np.sqrt([low * o1.CERT_D[0], o1.CERT_D[-1] * high])
    assert o1.certified_cumulative(geometric_midpoints).tolist() == pytest.approx([2.5, 95.0])


# ── pure core: operator + prediction ───────────────────────────────────────────────────────────

def test_operator_shape_and_nonneg():
    assert A_TEST.shape == (31, len(GRID))
    assert np.all(A_TEST >= 0)


def test_predicted_shape_normalized_nonneg():
    pred = o1.predict_channel_shape(A_TEST, N_CERT)
    assert abs(pred.sum() - 1.0) < 1e-9
    assert np.all(pred >= 0)


def test_shape_metrics_identity():
    v = np.abs(np.random.default_rng(0).random(31)) + 0.01
    v /= v.sum()
    m = o1.shape_metrics(v, v)
    assert m["cosine"] == pytest.approx(1.0, abs=1e-9)
    assert m["tv"] == pytest.approx(0.0, abs=1e-9)


def test_geometry_is_fixed_from_manual_edges_not_a_fit_parameter():
    rings = o1.detector_rings()
    assert len(rings) == 31
    assert rings.is_monotonic
    assert rings.theta_lo[0] == pytest.approx(0.0)
    assert rings.theta_hi[0] == pytest.approx(0.189904, abs=1e-5)
    assert rings.theta_hi[-1] == pytest.approx(40.126286, abs=1e-5)
    assert not hasattr(o1, "fit_theta_max")


# ── pure core: calibration-boundary diagnostics ─────────────────────────────────────────────────

def test_resolution_diagnostics_ranges():
    d = o1.resolution_diagnostics(A_TEST)
    assert 1.0 <= d["eff_rank_abs"] <= d["n_size_columns"]
    assert 1.0 <= d["eff_rank_colnorm"] <= d["n_size_columns"]
    # the operator is low-rank / strongly overlapping over the certified band
    assert d["eff_rank_colnorm"] < d["n_size_columns"]          # not full rank
    assert 0.0 < d["adjacent_col_cosine_median"] <= 1.0
    assert d["adjacent_col_cosine_median"] > 0.5                # neighbouring sizes overlap substantially
    assert d["condition_abs"] >= 1.0


def test_monodisperse_distributed_and_overlapping():
    mono = o1.monodisperse_responses(n_quad=64)
    P = mono["profiles"]
    assert P.shape == (31, len(o1.MONODISPERSE_UM))
    assert np.allclose(P.sum(0), 1.0)
    # every monodisperse diameter is distributed across multiple channels — never a single channel
    assert np.all(np.asarray(mono["fwhm_ch"]) >= 2)
    # neighbouring example diameters share substantial signal
    assert all(0.0 <= c <= 1.0 for c in mono["adjacent_pair_cosine"])
    assert max(mono["adjacent_pair_cosine"]) > 0.4


# ── vault-gated: reproduce the headline audit numbers from the real measurement files ────────────

_HAVE_VAULT = o1.DEFAULT_NIST_RTF.exists() and o1.DEFAULT_HELDOUT_RTF.exists()
vault = pytest.mark.skipif(not _HAVE_VAULT, reason="NIST vault measurement files not present")


@vault
def test_reproduces_audit_headline():
    r = o1.compute_objective1()
    f = r.feasibility
    # certified PSD → measured aggregate profile, C_sca-inclusive physical operator
    assert r.assumptions["geometry_fit_to_nist"] is False
    assert 0.90 <= f["cosine"] <= 0.98
    assert 0.88 <= f["second_session_cosine"] <= 0.98
    assert f["second_session_negative_channel_frame_fraction"] > f["negative_channel_frame_fraction"]
    # limited independent size information over 2-15 um; strong neighbour overlap
    b = r.boundary
    assert 5.5 <= b["eff_rank_colnorm"] <= 8.0
    assert 0.8 <= b["adjacent_col_cosine_median"] <= 0.97
