"""Predict dissolution from lab inputs and a laser-diffraction PSD.

Reads the suspension's q3 from a Sympatec export, uses an explicit parameter set
(or the optional legacy CFZ profile), and runs
:func:`simulate_dissolution`. Any :class:`Parameters` field can be overridden (e.g.
``s0_uM=…`` to ignore the artifact, or ``regime``/``pka_bh`` to explore).

    from diffractomorph_pipeline.forward import predict
    run = predict(psd="…/CFZ QC Q0 20260609/", ph=4.5, dose_mg=0.17, drug="CFZ")
    run = predict(psd=my_PSD, ph=4.5, conc_ugml=4.4, volume_mL=40,
                  params=my_material_parameters)
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from diffractomorph_pipeline.forward.params import PSD, Parameters
from diffractomorph_pipeline.forward.surface_ode import DissolutionRun, simulate
from diffractomorph_pipeline.forward.morphology import MorphologyParams
from diffractomorph_pipeline.forward.morphology import simulate as simulate_morphology

# drugs with a packaged Cs(pH) artifact + their constants
_DRUGS = {"CFZ": {"mw": 473.4, "pka_bh": 6.08}}


def _resolve_cs_s0_ugml(drug: str, ph: float):
    """S0 (µg/mL) reproducing the measured Cs at this pH, or None if no artifact."""
    if drug.upper() != "CFZ":
        return None
    from diffractomorph_pipeline import solubility
    return solubility.load_default().s0_for_ph_ugml(ph)


def predict(psd, ph: float, *, dose_mg: float | None = None,
            conc_ugml: float | None = None, volume_mL: float | None = None,
            drug: str | None = None, params: Parameters | None = None,
            morph: MorphologyParams | None = None,
            t_end: float = 1800.0, n_eval: int = 181, c0_ugml: float = 0.0,
            **overrides) -> DissolutionRun:
    """Predict a dissolution run from lab inputs.

    Parameters
    ----------
    psd
        A :class:`PSD`, or a path to a Sympatec PSD export (read via ``PSD.from_sympatec``).
    ph
        Bulk medium pH.
    dose_mg
        Dosed mass; or give ``conc_ugml`` + ``volume_mL`` (dose = conc·vol).
    drug
        Explicit legacy material identifier. ``drug="CFZ"`` selects the optional CFZ
        profile. Generic callers instead pass a complete ``params`` object.
    params
        Base :class:`Parameters` (defaults used if omitted).
    morph
        Optional :class:`~forward.morphology.MorphologyParams` to run the exploratory
        independent-exponent extension (``a_area``/``b_size``/``rate_scale``; deprecated ``w_por`` /
        ``f_acc``) instead of the base model. ``None`` (default) = base Nernst–Brunner.
    **overrides
        Any ``Parameters`` field to override (``s0_uM``, ``regime``, ``pka_bh``, …);
        an explicit ``s0_uml``/``s0_uM`` wins over the measured-Cs lookup.
    """
    psd = psd if isinstance(psd, PSD) else PSD.from_sympatec(Path(psd))

    if dose_mg is None:
        if conc_ugml is None or volume_mL is None:
            raise ValueError("give dose_mg, or both conc_ugml and volume_mL")
        dose_mg = conc_ugml * volume_mL / 1000.0

    if params is None and drug is None:
        raise ValueError(
            "material parameters are required; pass an explicit Parameters object or "
            "select the optional legacy profile with drug='CFZ'"
        )

    p = params or Parameters()
    d = _DRUGS.get(drug.upper()) if drug is not None else None
    if d:
        p = replace(p, mw=d["mw"], pka_bh=d["pka_bh"])
    elif params is None:
        raise ValueError(
            f"unknown material {drug!r}; pass an explicit Parameters object containing "
            "material-specific mw, pka_bh, and s0_uM values"
        )

    if volume_mL is not None:
        if "v_diss_mL" in overrides and float(overrides["v_diss_mL"]) != float(volume_mL):
            raise ValueError("volume_mL and v_diss_mL must agree when both are supplied")
        p = replace(p, v_diss_mL=float(volume_mL))

    if "s0_uM" not in overrides:                       # measured Cs(pH) unless overridden
        s0_ugml = _resolve_cs_s0_ugml(drug, ph) if drug is not None else None
        if s0_ugml is not None:
            p = p.with_s0_ugml(s0_ugml)
        elif drug is not None and drug.upper() == "CFZ":
            raise FileNotFoundError(
                "the optional CFZ solubility artifact is not installed; pass s0_uM explicitly"
            )
    if overrides:
        p = replace(p, **overrides)

    if morph is not None:                              # opt-in exploratory morphology (a_area/b_size/rate_scale)
        return simulate_morphology(p, psd, dose_mg=dose_mg, ph_bulk=ph, morph=morph,
                                   t_end=t_end, n_eval=n_eval, c0_ugml=c0_ugml)
    return simulate(p, psd, dose_mg=dose_mg, ph_bulk=ph, t_end=t_end, n_eval=n_eval,
                    c0_ugml=c0_ugml)


def predict_from_snapshot(psd, ph: float, *, injected_mg: float, conc_ugml: float,
                          volume_mL: float, drug: str | None = None,
                          params: Parameters | None = None, morph: MorphologyParams | None = None,
                          t_end: float = 1800.0, n_eval: int = 181, **overrides) -> DissolutionRun:
    """Predict a run **from the first grounded UV+LD snapshot** instead of from injection.

    Mode 2: anchor the model at the earliest timepoint where both the dissolved concentration
    (UV) and the particle size (LD) are measured. Mass balance sets how much is still solid; the
    already-dissolved drug seeds the bulk (so the driving force starts at the measured ``Cs − C``).

    Parameters
    ----------
    psd
        The **snapshot** PSD — the eroded first LD frame (not the QC/injected suspension); a
        :class:`PSD` or a Sympatec export path. Describes the still-solid material at the anchor.
    ph
        Bulk medium pH.
    injected_mg
        Total injected dose (e.g. from ``forward.injected_mass``); the still-solid mass is
        ``injected_mg − conc_ugml·volume_mL`` by mass balance.
    conc_ugml
        Measured dissolved concentration at the snapshot (the first UV timepoint).
    volume_mL
        Dissolution-cell volume; used for both the mass balance and the ODE (kept consistent).
    **overrides
        Any :class:`Parameters` field (``s0_uM``, ``regime``, …); as in :func:`predict`.

    Returns
    -------
    DissolutionRun
        Trajectory with time measured **from the snapshot** (``t=0`` is the anchor), the bulk
        seeded at ``conc_ugml``. ``pct_dissolved`` is normalized to the **injected** dose
        (still-solid + pre-dissolved reassemble to ``injected_mg``), so it starts at the already-
        dissolved fraction ``C·V/injected`` and is directly comparable to Mode 1 and the UV.
    """
    dissolved_mg = conc_ugml * volume_mL / 1000.0
    still_solid_mg = injected_mg - dissolved_mg
    if still_solid_mg <= 0.0:
        raise ValueError(
            f"snapshot anchor dissolves everything: C·V = {conc_ugml}×{volume_mL}/1000 = "
            f"{dissolved_mg:.4f} mg ≥ injected {injected_mg} mg — check the dose or the UV concentration")
    overrides.setdefault("v_diss_mL", volume_mL)       # ODE cell volume == mass-balance volume
    return predict(psd, ph, dose_mg=still_solid_mg, c0_ugml=conc_ugml, drug=drug,
                   params=params, morph=morph, t_end=t_end, n_eval=n_eval, **overrides)
