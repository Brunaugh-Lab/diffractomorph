"""Dissolved concentration C(t) from a BioTek UV plate export.

Defaults are the DISSOLUTION assay: DMSO dilution 300/270 and the additive syringe-filter
offset. Other assays (e.g. suspension-conc) used a different dilution — override ``dilution``.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from diffractomorph_pipeline.assay import calibration as cal
from diffractomorph_pipeline.assay.curve import StandardCurve
from diffractomorph_pipeline.assay.plates import read_plate, read_plate_wavelengths


def uv_timecourse(path, ph, *, dilution=None, filter_offset_ugml=None, cs_ugml=None,
                  injected_mg=None, volume_mL=40.0, calibration=None) -> pd.DataFrame:
    """C(t) (µg/mL) vs time from a plate export, via the packaged CFZ calibration.

    ``dilution`` defaults to the dissolution scheme (300/270); ``filter_offset_ugml`` to the
    additive syringe-filter offset for ``ph``. Override both for non-dissolution assays.

    Also carries the dissolution driving force: ``cs_ugml`` (saturation solubility at ``ph``,
    from the packaged CFZ Cs(pH) model unless overridden), ``driving_ugml`` = Cs − C(t), and
    ``pct_cs`` = 100·C(t)/Cs.

    Pass ``injected_mg`` (the delivered dose, e.g. from ``forward.injected_mass``) to also get
    ``pct_injected`` = 100·C(t)·V / injected — the measured fraction of the injected dose in
    solution, directly comparable to the forward model's ``pct_dissolved``. ``volume_mL`` is the
    dissolution-cell volume.
    """
    ph = float(ph)
    pl = read_plate(path)
    profile = calibration or cal._DEFAULT_PROFILE  # None is the documented legacy CFZ path.
    if profile is None:
        raise FileNotFoundError(
            "the optional CFZ assay artifact is not installed; pass calibration explicitly"
        )
    dilution = profile.dilution if dilution is None else dilution
    offset = profile.filter_offset.get(ph, 0.0) if filter_offset_ugml is None else filter_offset_ugml
    b280 = profile.blank[280] if np.isnan(pl.blank280) else pl.blank280
    b490 = profile.blank[490] if np.isnan(pl.blank490) else pl.blank490
    c280 = profile.curve(ph, 280).concentration(pl.a280, blank=b280)
    c490 = profile.curve(ph, 490).concentration(pl.a490, blank=b490)
    well = (c280 + c490) / 2.0
    conc = (well + offset) * dilution
    cs = _cs_for_ph(ph) if cs_ugml is None else float(cs_ugml)
    out = pd.DataFrame({"time_min": pl.times_min, "conc_ugml": conc,
                        "cs_ugml": cs, "driving_ugml": cs - conc, "pct_cs": 100.0 * conc / cs,
                        "conc_well_ugml": well, "conc280_ugml": c280, "conc490_ugml": c490,
                        "abs280": pl.a280, "abs490": pl.a490, "blank280": b280, "blank490": b490,
                        "source": pl.source})
    if injected_mg is not None:                    # fraction of the injected dose in solution
        out.insert(5, "pct_injected", 100.0 * conc * float(volume_mL) / (float(injected_mg) * 1e3))
        out.insert(6, "injected_mg", float(injected_mg))
    return out


def uv_timecourse_profiled(path, condition, *, calibration, cs_ugml, wavelengths_nm,
                           filter_offset_ugml=None, injected_mg=None, volume_mL=40.0) -> pd.DataFrame:
    """Generic UV timecourse requiring explicit assay and solubility inputs."""
    if calibration is None:
        raise ValueError("calibration is required")
    if cs_ugml is None:
        raise ValueError("cs_ugml is required; no material solubility is selected implicitly")
    condition = float(condition)
    wavelengths = tuple(int(value) for value in wavelengths_nm)
    plate = read_plate_wavelengths(path, wavelengths)
    if filter_offset_ugml is None and condition not in calibration.filter_offset:
        raise KeyError(
            f"assay profile {calibration.calibration_id!r} has no filter offset for {condition}"
        )
    concentrations = {}
    blanks = {}
    for wavelength in wavelengths:
        blank = plate.blank[wavelength]
        if np.isnan(blank):
            if wavelength not in calibration.blank:
                raise KeyError(
                    f"assay profile {calibration.calibration_id!r} has no blank for {wavelength} nm"
                )
            blank = calibration.blank[wavelength]
        blanks[wavelength] = blank
        concentrations[wavelength] = calibration.curve(condition, wavelength).concentration(
            plate.absorbance[wavelength], blank=blank,
        )
    well = np.mean(np.vstack(list(concentrations.values())), axis=0)
    offset = (calibration.filter_offset[condition]
              if filter_offset_ugml is None else float(filter_offset_ugml))
    concentration = (well + offset) * calibration.dilution
    cs = float(cs_ugml)
    data = {
        "time_min": plate.times_min,
        "conc_ugml": concentration,
        "cs_ugml": cs,
        "driving_ugml": cs - concentration,
        "pct_cs": 100.0 * concentration / cs,
        "conc_well_ugml": well,
        "source": plate.source,
    }
    for wavelength in wavelengths:
        data[f"conc{wavelength}_ugml"] = concentrations[wavelength]
        data[f"abs{wavelength}"] = plate.absorbance[wavelength]
        data[f"blank{wavelength}"] = blanks[wavelength]
    out = pd.DataFrame(data)
    if injected_mg is not None:
        out.insert(5, "pct_injected", 100.0 * concentration * float(volume_mL) / (float(injected_mg) * 1e3))
        out.insert(6, "injected_mg", float(injected_mg))
    return out


def cumulative_dissolved(conc_ugml, dose_mg, *, aliquot_mL=0.400, v0_mL=40.0) -> pd.DataFrame:
    """Cumulative dissolved CFZ and the **UV-derived apparent remaining dose fraction**, correcting for
    UNREPLACED aliquot removal at each UV sample (a UV **mass-balance** coordinate — not an optical mass
    estimate).

    A ``aliquot_mL`` sample is withdrawn and NOT replaced at every UV timepoint, so the vessel volume
    shrinks and already-dissolved drug leaves with each aliquot. Starting from the already calibration-,
    dilution-, and additive-filter-corrected vessel concentration ``conc_ugml`` (µg/mL), for zero-indexed
    sample ``i``::

        V_i           = v0_mL − aliquot_mL·i                              [mL]
        m_dissolved_i = C_i·V_i + Σ_{j<i} C_j·aliquot_mL                  [µg]
        recovery_mass_fraction_i           = m_dissolved_i / (dose_mg·1000)
        apparent_remaining_dose_fraction_i = 1 − recovery_mass_fraction_i

    (C in µg/mL × V in mL → mass in µg; the dose is converted mg → µg by ×1000.) Under a constant
    concentration the removed mass exactly compensates the shrinking volume, so the cumulative dissolved
    mass is constant (= C·v0).

    Values are **not** clipped to ``[0, 1]``; QC flag columns mark out-of-interval samples so assay
    variability stays visible. **No optical variable** (Copt, total angular signal, or q3 magnitude)
    enters. Interpreting the apparent remaining dose fraction as suspended undissolved *particle* mass
    assumes no important deposition, precipitation, or other unrecovered drug compartment — notably
    uncertain at pH 5.0, where recovery is incomplete.

    Returns a DataFrame (one row per sample, input order) with ``sample_index``, ``vessel_mL``,
    ``cumulative_dissolved_ug``, ``recovery_mass_fraction``, ``apparent_remaining_dose_fraction``, and
    the ``qc_remaining_below_0`` / ``qc_remaining_above_1`` / ``qc_vessel_nonpositive`` flags.
    """
    C = np.asarray(conc_ugml, float)
    i = np.arange(C.size)
    V = float(v0_mL) - float(aliquot_mL) * i
    removed_prior = (np.concatenate([[0.0], np.cumsum(C[:-1] * float(aliquot_mL))])
                     if C.size else np.zeros(0))            # Σ_{j<i} C_j·aliquot (drug already withdrawn)
    m_dissolved = C * V + removed_prior                     # µg
    rec = m_dissolved / (float(dose_mg) * 1000.0)
    rem = 1.0 - rec
    return pd.DataFrame({
        "sample_index": i, "vessel_mL": V,
        "cumulative_dissolved_ug": m_dissolved,
        "recovery_mass_fraction": rec,
        "apparent_remaining_dose_fraction": rem,
        "qc_remaining_below_0": rem < 0.0,
        "qc_remaining_above_1": rem > 1.0,
        "qc_vessel_nonpositive": V <= 0.0,
    })


def _cs_for_ph(ph):
    """Saturation solubility Cs(pH) in µg/mL from the packaged CFZ model (lazy import)."""
    from diffractomorph_pipeline.solubility import load_default
    return load_default().cs_for_ph(ph)


def _ph_from_name(name: str):
    m = re.search(r"pH[=_ ]?([0-9]+(?:\.[0-9]+)?)", name, re.IGNORECASE)
    return float(m.group(1)) if m else None


def timecourse_folder(folder, ph=None, out_csv=None, out_png=None, **kw) -> pd.DataFrame:
    """Every plate in ``folder`` → one long (time, conc) table; optional CSV + figure.

    ``ph`` is inferred per file from the filename when not given.
    """
    folder = Path(folder)
    frames = []
    for f in sorted(folder.glob("*.xlsx")):
        p = ph if ph is not None else _ph_from_name(f.name)
        if p is None:
            continue
        df = uv_timecourse(f, p, **kw)
        df.insert(0, "sample", f.stem)
        df.insert(1, "ph", p)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out_csv is not None:
        out.to_csv(out_csv, index=False)
    if out_png is not None and not out.empty:
        _plot(out, out_png)
    return out


def _plot(df: pd.DataFrame, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    has_pct = "pct_injected" in df.columns
    nrows = 3 if has_pct else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(6, 3.5 * nrows), sharex=True)
    ax_c, ax_d = axes[0], axes[1]
    ax_p = axes[2] if has_pct else None
    for sample, g in df.groupby("sample"):
        line, = ax_c.plot(g["time_min"], g["conc_ugml"], "-o", ms=3, label=sample)
        ax_c.axhline(g["cs_ugml"].iloc[0], ls="--", lw=0.8, color=line.get_color(), alpha=0.6)
        ax_d.plot(g["time_min"], g["driving_ugml"], "-o", ms=3, color=line.get_color())
        if ax_p is not None:
            ax_p.plot(g["time_min"], g["pct_injected"], "-o", ms=3, color=line.get_color())
    ax_c.set_ylabel("dissolved C (µg/mL)")
    ax_c.legend(fontsize=6, ncol=2, title="dashed = Cs", title_fontsize=6)
    ax_d.set_ylabel("driving force  Cs − C (µg/mL)")
    ax_d.axhline(0.0, ls=":", lw=0.8, color="k", alpha=0.5)
    if ax_p is not None:
        ax_p.set_ylabel("% of injected dose dissolved")
        ax_p.axhline(100.0, ls=":", lw=0.8, color="k", alpha=0.5)
    axes[-1].set_xlabel("time (min)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
