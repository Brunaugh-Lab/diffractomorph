"""Size-resolved dissolution of a weak base, with surface-pH self-buffering.

Couples a transport-limited Nernst–Brunner dissolution rate to a self-consistent surface pH.
The surface-pH treatment follows the reversible-non-equilibrium (RNE) dissolution framework of
Al-Gousous et al. [1,2]; the diffusion length uses the stagnant-film approximation h ≈ r
(Sherwood number → 2 for small particles) supported by Al-Gousous et al. [3]. The bulk and
interfacial proton balances are solved in :mod:`buffer` and :mod:`surface`.

State
-----
- ``q_i`` : undissolved mass in each size bin (one per :class:`PSD` bin).
- ``C``   : bulk dissolved concentration.

Rate law
--------
Each bin is a shrinking sphere. For a single particle, transport-limited Nernst–Brunner with a
diffusion layer ``h ≈ r`` gives ``dm/dt = −4πD·r·(Cs − C)``. Conserving particle number within a
bin and writing ``r_i(t) = φ_i^{1/3}·r_{0,i}`` (mass ∝ r³, with ``φ_i = q_i/q_i(0)``):

    dq_i/dt = −3·D·q_i(0)·φ_i^{1/3} / (ρ·r_{0,i}²) · max(0, Cs − C)
    dC/dt   = −(1/V) · Σ_i dq_i/dt

``D`` = drug diffusivity, ``ρ`` = solid molar density, ``V`` = medium volume, ``r_{0,i}`` = initial
bin radius. The ``1/r_{0,i}²`` factor makes the dissolution time scale as r² — a direct consequence
of the ``h ≈ r`` choice. ``Cs`` is the SURFACE solubility ``Cs = S0·(1 + [H+]_s / Ka_BH)``, set by
the surface pH the two proton balances solve for. The driving force is clamped ≥ 0: at or above
saturation, dissolution stops (no re-growth). A bin freezes once ``q_i ≤ freeze_frac·q_i(0)``, a
non-singular guard as ``r → 0``.

This module is the **base** model (pure transport-limited Nernst–Brunner). The exploratory
generalizations — independent rate-law exponents ``a_area`` (reactive-area evolution) and ``b_size``
(initial-size dependence), one global ``rate_scale`` (k_tr = dd·rate_scale), and the deprecated
``f_acc`` dissolvable-fraction cap — live in :mod:`forward.morphology`, which reuses :func:`_integrate`
here. The starting distribution is :class:`PSD`; the tunable knobs are :class:`Parameters`.

Model formulated by J. Al-Gousous; implemented by J. Al-Gousous and A. Brunaugh.

References
----------
[1] Al-Gousous, J. et al. (2019) Mol. Pharm. 16(6), 2626–2635.
[2] Al-Gousous, J. et al. (2025) J. Pharm. Sci. 114, 103702.
[3] Al-Gousous, J. et al. (2026) Ind. Eng. Chem. Res.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from diffractomorph_pipeline.forward.buffer import solve_bulk_H, spectator_for_pH
from diffractomorph_pipeline.forward.params import PSD, Parameters
from diffractomorph_pipeline.forward.surface import solve_surface_H, surface_solubility


@dataclass
class DissolutionRun:
    t: np.ndarray                # seconds
    pct_dissolved: np.ndarray    # % of total initial mass
    cbulk: np.ndarray            # mol/L
    ph_bulk: np.ndarray
    ph_surf: np.ndarray
    qundiss: np.ndarray          # (frames × nbin) mmol remaining per fraction
    radius_um: np.ndarray        # (frames × nbin) per-fraction radius over time
    diam0_um: np.ndarray         # starting representative diameters
    inputs: dict = field(default_factory=dict)

    @property
    def frac_dissolved_final(self) -> float:
        return float(self.pct_dissolved[-1] / 100.0)


def _integrate(p: Parameters, psd: PSD, dose_mg: float, ph_bulk: float, shape,
               t_end: float = 1800.0, n_eval: int = 181, method: str = "BDF",
               c0_ugml: float = 0.0, extra_inputs: dict | None = None) -> DissolutionRun:
    """Integrate the coupled dissolution ODE for one run (shared core).

    ``shape(rini, q0, diam_um, live)`` returns the per-bin rate-law pieces
    ``(a_exp, geom, floor_q)``: ``a_exp`` = the φ exponent, ``geom`` = the per-bin geometric
    factor, ``floor_q`` = the freeze floor. The base model (:func:`simulate`) and the morphology
    extension (:func:`forward.morphology.simulate`) differ *only* in this callback.

    ``c0_ugml`` seeds the bulk with an already-dissolved concentration (a partly-filled sink,
    e.g. pre-frame-0 dissolution) — it lowers the initial driving force ``(Cs − C)``; pass the
    still-solid mass as ``dose_mg``.
    """
    if p.regime != "transport":
        raise NotImplementedError(
            "only the transport-limited (τ∝r²) Nernst–Brunner regime is ported; "
            f"got regime={p.regime!r}")
    fw = psd.volfrac_norm
    diam_um = psd.diam_um
    dose = dose_mg / p.mw                       # mmol
    rini = diam_um * 1e-4 / 2.0                 # cm
    q0 = fw * dose
    live = q0 > 0
    total_mass = q0.sum()
    dd, rho, Vdiss = p.diffusivities.dd, p.rho_mol, p.v_diss_mL
    m = spectator_for_pH(p, ph_bulk)            # spectator Na+ fixed by the initial pH

    a_exp, geom, floor_q = shape(rini, q0, diam_um, live)

    def protons(cbulk):
        hb = solve_bulk_H(p, m, c_drug=max(cbulk, 0.0))
        return hb, solve_surface_H(hb, max(cbulk, 0.0), p)

    def rhs(_t, y):
        q = y[:-1]; cbulk = y[-1]
        _, hs = protons(cbulk)
        cs_surf = surface_solubility(hs, p)
        drive = max(0.0, cs_surf - cbulk)       # clamp: no dissolution above saturation
        phi = np.zeros_like(q)
        phi[live] = np.clip(q[live] / q0[live], 0.0, None)
        dq = -3.0 * dd * fw * dose * phi ** a_exp * geom / rho * drive
        dq = np.where(q <= floor_q, 0.0, dq)
        return np.concatenate([dq, [-dq.sum() / Vdiss]])

    c0_mol = c0_ugml * 1e-3 / p.mw          # µg/mL → mol/L (g/L ÷ g/mol)
    y0 = np.concatenate([q0, [c0_mol]])
    t_eval = np.linspace(0.0, t_end, n_eval)
    sol = solve_ivp(rhs, (0.0, t_end), y0, t_eval=t_eval, method=method, rtol=1e-8, atol=1e-12)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    q_t = sol.y[:-1].T
    cbulk_t = sol.y[-1]
    phi_t = np.zeros_like(q_t)
    phi_t[:, live] = np.clip(q_t[:, live] / q0[live], 0.0, None)
    radius_um = phi_t ** (1.0 / 3.0) * diam_um / 2.0

    ph_b, ph_s = [], []
    for c in cbulk_t:
        hb, hs = protons(c)
        ph_b.append(-np.log10(hb)); ph_s.append(-np.log10(hs))

    true_total = total_mass + c0_mol * Vdiss          # solid + pre-dissolved (mass balance)
    return DissolutionRun(
        t=sol.t, pct_dissolved=100.0 * cbulk_t * Vdiss / true_total, cbulk=cbulk_t,
        ph_bulk=np.array(ph_b), ph_surf=np.array(ph_s), qundiss=q_t, radius_um=radius_um,
        diam0_um=diam_um,
        inputs={"dose_mg": dose_mg, "ph_bulk": ph_bulk, "total_mass_mmol": total_mass,
                "s0_uM": p.s0_uM, "regime": p.regime, "n_bins": int(live.sum()),
                **(extra_inputs or {})})


def _base_shape(p: Parameters):
    """Pure transport-limited Nernst–Brunner: τ∝r² (φ^{1/3}·r₀⁻²) + the uniform freeze-floor guard."""
    def shape(rini, q0, diam_um, live):
        geom = np.zeros_like(rini)
        geom[live] = 1.0 / rini[live] ** 2
        return 1.0 / 3.0, geom, p.freeze_frac * q0
    return shape


def simulate(p: Parameters, psd: PSD, dose_mg: float, ph_bulk: float,
             t_end: float = 1800.0, n_eval: int = 181, method: str = "BDF",
             c0_ugml: float = 0.0) -> DissolutionRun:
    """Integrate the **base** transport-limited Nernst–Brunner dissolution ODE for one run.

    ``p`` = tunable parameters; ``psd`` = starting q3 distribution; ``dose_mg`` = dosed
    (still-solid) mass; ``ph_bulk`` = the buffer's bulk pH; ``c0_ugml`` = pre-dissolved bulk
    concentration (partly-filled sink). Returns the full trajectory.

    This is the pure base model (smooth-sphere shrinking core). The exploratory independent
    exponents (``a_area``/``b_size``), the global ``rate_scale``, and the deprecated ``f_acc`` depth
    cap are in :func:`forward.morphology.simulate`.
    """
    return _integrate(p, psd, dose_mg, ph_bulk, _base_shape(p),
                      t_end=t_end, n_eval=n_eval, method=method, c0_ugml=c0_ugml)


def volume_psd(run: DissolutionRun, denom: str = "initial") -> dict:
    """Evolving volume distribution per frame. ``denom`` — ``"initial"`` (default, absolute
    fraction of injected dose, artifact-free) or ``"frame"`` (per-frame q3 density; large
    bins inflate as fines leave — the Sympatec relative convention). Returns dict of
    (frames × nbin) arrays: ``d_cur`` (µm), ``fV`` (fraction)."""
    q = run.qundiss
    d_cur = 2.0 * run.radius_um
    if denom == "initial":
        total = np.clip(q[0].sum(), 1e-300, None)
    elif denom == "frame":
        total = np.clip(q.sum(axis=1, keepdims=True), 1e-300, None)
    else:
        raise ValueError(f"denom must be 'initial' or 'frame', got {denom!r}")
    return {"d_cur": d_cur, "fV": q / total}
