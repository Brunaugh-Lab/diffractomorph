"""Tests for the Objective-1 operator-feasibility analysis + CAREER figure.

Two tiers: a vault-free pure-core tier that exercises the certified PSD, operator, shape metrics,
geometry fit, and calibration-boundary diagnostics on the candidate operator; and a vault-gated tier
that reproduces the headline audit numbers from the real NIST measurement files (skipped in CI).
"""
import numpy as np
import pytest

from figures import objective1_feasibility as o1
from diffractomorph_pipeline.optics.standards import GRID

# One modest-resolution operator, built once, reused across the pure-core tests.
A_TEST = o1.build_physical_operator(9.88, n_quad=32)
N_CERT = o1.certified_number_psd()


# ── pure core: certified distribution ──────────────────────────────────────────────────────────

def test_certified_psd_normalized_and_confined():
    assert abs(N_CERT.sum() - 1.0) < 1e-9
    assert np.all(N_CERT >= 0)
    # certificate spans ~2.1-12.9 um; no submicron mass injected below the certified span
    assert N_CERT[GRID < 1.8].sum() < 1e-6
    # essentially all number-weighted mass falls in the grid bins overlapping the certified band
    # (the 1.99 um bin, edges 1.8-2.2, legitimately catches the certified 2.1 um material; the d^-3
    # number weighting concentrates the PSD toward the small-diameter end of that band)
    assert N_CERT[(GRID >= 1.9) & (GRID <= 14.0)].sum() > 0.98
    assert N_CERT[GRID > 15.0].sum() < 1e-6


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


def test_fit_theta_max_recovers_planted_geometry():
    # obs generated at a known theta_max in a tiny grid → the fit must select that theta_max
    grid = np.array([8.0, 9.88, 20.0])
    A_true = o1.build_physical_operator(9.88, n_quad=32)
    obs = o1.predict_channel_shape(A_true, N_CERT)
    fit = o1.fit_theta_max(obs, N_CERT, mode="physical", grid=grid)
    assert fit["theta_max_deg"] == pytest.approx(9.88, abs=1e-6)
    assert fit["cosine"] == pytest.approx(1.0, abs=1e-4)


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
    mono = o1.monodisperse_responses(9.88, n_quad=64)
    P = mono["profiles"]
    assert P.shape == (31, len(o1.MONODISPERSE_UM))
    assert np.allclose(P.sum(0), 1.0)
    # every monodisperse diameter is distributed across multiple channels — never a single channel
    assert np.all(np.asarray(mono["fwhm_ch"]) >= 2)
    # neighbouring example diameters share substantial signal
    assert all(0.0 <= c <= 1.0 for c in mono["adjacent_pair_cosine"])
    assert max(mono["adjacent_pair_cosine"]) > 0.4


def test_current_operator_differs_from_physical():
    # the no-C_sca shortcut is a genuinely different operator (large-particle weighting missing)
    cur = o1._current_operator(9.88)
    assert cur.shape == A_TEST.shape
    pc = o1.predict_channel_shape(cur, N_CERT)
    pp = o1.predict_channel_shape(A_TEST, N_CERT)
    assert not np.allclose(pc, pp, atol=1e-3)


# ── vault-gated: reproduce the headline audit numbers from the real measurement files ────────────

_HAVE_VAULT = o1.DEFAULT_NIST_RTF.exists() and o1.DEFAULT_HELDOUT_RTF.exists()
vault = pytest.mark.skipif(not _HAVE_VAULT, reason="NIST vault measurement files not present")


@vault
def test_reproduces_audit_headline():
    r = o1.compute_objective1()
    f = r.feasibility
    # certified PSD → measured aggregate profile, C_sca-inclusive physical operator
    assert r.theta_max_deg == pytest.approx(9.9, abs=1.0)
    assert f["cosine"] >= 0.999                                  # ~0.9997
    assert f["csca_cosine_gain"] > 0                             # C_sca is load-bearing (0.9928 → 0.9997)
    assert f["physical_at_paqxos_theta_max_cosine"] < 0.96       # old PAQXOS 39.2° angular scale is wrong
    assert f["held_out_transfer_cosine"] >= 0.997                # ~0.9989 second same-standard session
    # limited independent size information over 2-15 um (~5 directions; strong neighbour overlap)
    b = r.boundary
    assert 4.0 <= b["eff_rank_colnorm"] <= 7.0
    assert 0.8 <= b["adjacent_col_cosine_median"] <= 0.97
