"""External study profiles must be selectable without modifying the installed wheel."""
from __future__ import annotations

import numpy as np

from diffractomorph_pipeline import noise_surface, solubility
from diffractomorph_pipeline.assay import calibration
from diffractomorph_pipeline.optics import mie


def test_profile_paths_can_be_selected_by_the_study_runner(tmp_path, monkeypatch):
    assay = tmp_path / "assay.json"
    sol = tmp_path / "solubility.json"
    noise = tmp_path / "noise.json"
    monkeypatch.setenv("DFM_ASSAY_PROFILE", str(assay))
    monkeypatch.setenv("DFM_SOLUBILITY_PROFILE", str(sol))
    monkeypatch.setenv("DFM_NOISE_SURFACE", str(noise))
    assert calibration.default_path() == assay
    assert solubility.default_path() == sol
    assert noise_surface._surface_path() == noise


def test_optical_kernel_can_be_selected_by_the_study_runner(tmp_path, monkeypatch):
    path = tmp_path / "kernel.npz"
    expected = mie.MieKernel(
        A=np.ones((2, 3)), xm=np.array([1.0, 2.0, 3.0]), theta=np.array([0.1, 0.2]),
        char_size=np.array([3.0, 1.0]), n_cfz=1.71, meta={"id": "synthetic"},
    )
    expected.save(path)
    monkeypatch.setenv("DFM_OPTICAL_KERNEL", str(path))
    observed = mie.load_kernel()
    assert observed.meta["id"] == "synthetic"
    assert np.array_equal(observed.A, expected.A)
