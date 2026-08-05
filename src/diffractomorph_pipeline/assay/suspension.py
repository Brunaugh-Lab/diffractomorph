"""CFZ dosing-suspension concentration assay (the pH-7 antisolvent stock).

Each run day, the antisolvent dosing suspension is diluted in DMSO and read at 453 nm.
The packaged calibration (assay ``calibration.json`` → ``suspension`` block) converts that
absorbance to the undiluted suspension concentration (mg/mL): the 453 nm DMSO standard curve
times the dilution factor. The suspension assay was not filtered, so no filter correction is
applied by default. Multiplying by the per-run injected volume gives
the delivered CFZ mass — the dose record the forward model needs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from diffractomorph_pipeline.assay import calibration as cal
from diffractomorph_pipeline.assay.plates import read_qc


def _suspension_profile():
    if not cal.SUSPENSION:
        raise FileNotFoundError(
            "the optional CFZ suspension calibration is not installed; supply an explicit "
            "assay profile and perform the concentration conversion from that profile"
        )
    return cal.SUSPENSION


def suspension_conc_mgml(abs_bs, *, filter_corrected=False):
    """Blank-subtracted 453 nm absorbance → undiluted suspension conc (mg/mL).

    Applies the packaged DMSO standard curve then ×DF. ``filter_corrected=True`` is an
    explicit legacy sensitivity only; the manuscript dosing-suspension samples were unfiltered.
    """
    s = _suspension_profile()
    slope, intercept = s["curve_dmso"]
    base = (np.asarray(abs_bs, float) - intercept) / slope           # mg/mL at read level
    if filter_corrected:
        base = base + s["filter_offset_ugml"] / 1e3
    return base * s["dilution_factor"]


def suspension_conc_from_qc(qc_path, *, filter_corrected=False):
    """The day's suspension conc (mg/mL) read straight from a QC plate export.

    Convenience for the common one-suspension-per-file case: locates the 453 nm grid via
    :func:`~diffractomorph_pipeline.assay.plates.read_qc` and converts. Raises if the file holds
    more than one read — then call ``read_qc`` and map each read to its condition yourself.
    """
    profile = _suspension_profile()
    reads = read_qc(qc_path, wavelength=profile["wavelength_nm"])
    if len(reads) != 1:
        raise ValueError(f"{Path(qc_path).name}: expected 1 suspension read, found {len(reads)}; "
                         "use assay.read_qc() and map reads to conditions")
    return float(suspension_conc_mgml(reads[0].abs_bs, filter_corrected=filter_corrected))


def injected_mass_mg(susp_mgml, volume_uL):
    """Delivered CFZ mass (mg) = suspension conc (mg/mL) × injected volume (µL)."""
    return np.asarray(susp_mgml, float) * np.asarray(volume_uL, float) / 1e3
