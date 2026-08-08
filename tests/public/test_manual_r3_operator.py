"""Public-contract tests for the manual-derived HELOS R3 optical operator."""

import numpy as np
import pytest

from diffractomorph_pipeline.optics import mie_candidate as mc
from diffractomorph_pipeline.optics.standards import EDGES


def test_manual_radii_are_exact_fraunhofer_transforms():
    radii = mc.r3_manual_radii(EDGES)
    expected = np.sort(np.r_[0.0, 1.22 * mc.LAMBDA0_UM * mc.R3_FOCAL_MM / EDGES])
    assert radii.shape == (32,)
    assert np.array_equal(radii, expected)
    assert np.all(np.diff(radii) > 0)


def test_manual_medium_angles_are_contiguous_and_fixed():
    rings = mc.r3_manual_rings(EDGES)
    assert len(rings) == 31
    assert rings.is_monotonic
    assert np.array_equal(rings.theta_hi[:-1], rings.theta_lo[1:])
    assert rings.theta_lo[0] == pytest.approx(0.0)
    assert rings.theta_hi[0] == pytest.approx(0.189904, abs=1e-5)
    assert rings.theta_hi[-1] == pytest.approx(40.126286, abs=1e-5)


def test_reversed_or_unsorted_radii_are_rejected():
    radii = mc.r3_manual_radii(EDGES)
    with pytest.raises(ValueError, match="strictly increasing"):
        mc.rings_from_radii(radii[::-1])


def test_ring_integral_converges_and_is_bounded_by_total_scattering():
    rings = mc.r3_manual_rings(EDGES)
    response_64 = mc.ring_response(5.0, rings, mc.GLASS_RI, n_quad=64)
    response_128 = mc.ring_response(5.0, rings, mc.GLASS_RI, n_quad=128)
    assert np.all(response_64 >= 0)
    assert np.allclose(response_64, response_128, rtol=1e-6, atol=1e-10)
    assert response_128.sum() < mc.csca_um2(5.0, mc.GLASS_RI)
