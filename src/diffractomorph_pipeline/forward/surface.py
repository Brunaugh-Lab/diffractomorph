"""Surface-pH solve — the weak-base self-buffering at the particle surface.

The dissolution driving force needs the **surface** proton activity [H+]_s, set by the
interfacial proton flux / charge balance. The surface buffer totals (``cas``/``cbors``/``cps``)
are closed forms in [H+]_s and [H+]_bulk, so the balance reduces to one equation in one unknown
[H+]_s. The Sherwood/radius factor is common to all species at a given particle and cancels, so
the surface pH is **size-independent** — solved once per timestep. All constants from
:class:`Parameters`. Part of the surface-pH dissolution model — see
:mod:`diffractomorph_pipeline.forward.surface_ode`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from diffractomorph_pipeline.forward.params import Parameters

# field order of the interfacial flux terms: see Parameters.diffusivities


def _phos_poly(h, kap):
    k1, k2, k3 = kap
    return h ** 3 + k1 * h ** 2 + k1 * k2 * h + k1 * k2 * k3


def _phos_flux(h, kap, D):
    k1, k2, k3 = kap
    return (D.dpo4 * k1 * k2 * k3 + D.dhpo4 * k1 * k2 * h
            + D.dh2po4 * k1 * h ** 2 + D.dh3po4 * h ** 3) / _phos_poly(h, kap)


def surface_residual(hs, hbulk, cbulk, p: Parameters) -> float:
    """Interfacial charge-balance residual (LHS − RHS) as a function of surface [H+] = hs (mol/L)."""
    D = p.diffusivities
    kw, kad, kaa, kab, kap = p.kw, p.kad, p.kaa, p.kab, p.kap
    ca, cbor, cp, s0 = p.acetate_M, p.borate_M, p.phosphate_M, p.s0_mol

    # bulk species (fixed for this solve)
    drugh_b = hbulk * cbulk / (hbulk + kad)
    oh_b = kw / hbulk
    ac_b = kaa * ca / (hbulk + kaa)
    bor_b = kab * cbor / (hbulk + kab)
    pden_b = _phos_poly(hbulk, kap)
    h2po4_b = kap[0] * hbulk ** 2 * cp / pden_b
    hpo4_b = kap[0] * kap[1] * hbulk * cp / pden_b
    po4_b = kap[0] * kap[1] * kap[2] * cp / pden_b

    # surface buffer totals (closed-form) + surface species at hs
    cas = ((ca * D.da * kaa + ca * D.dha * hbulk) * (kaa + hs)
           / ((kaa + hbulk) * (D.da * kaa + D.dha * hs)))
    cbors = ((cbor * D.dbor * kab + cbor * D.dborh * hbulk) * (kab + hs)
             / ((kab + hbulk) * (D.dbor * kab + D.dborh * hs)))
    cps = cp * _phos_flux(hbulk, kap, D) / _phos_flux(hs, kap, D)
    oh_s = kw / hs
    ac_s = kaa * cas / (hs + kaa)
    bor_s = kab * cbors / (hs + kab)
    pden_s = _phos_poly(hs, kap)
    h2po4_s = kap[0] * hs ** 2 * cps / pden_s
    hpo4_s = kap[0] * kap[1] * hs * cps / pden_s
    po4_s = kap[0] * kap[1] * kap[2] * cps / pden_s
    drugh_s = s0 * hs / kad                    # surface DrugH+ pinned to surface solubility

    lhs = D.dh * (hs - hbulk) + D.dd * (drugh_s - drugh_b)
    rhs = (D.doh * (oh_s - oh_b) + D.da * (ac_s - ac_b) + D.dbor * (bor_s - bor_b)
           + D.dh2po4 * (h2po4_s - h2po4_b) + 2 * D.dhpo4 * (hpo4_s - hpo4_b)
           + 3 * D.dpo4 * (po4_s - po4_b))
    return lhs - rhs


def solve_surface_H(hbulk, cbulk, p: Parameters) -> float:
    """Surface proton activity [H+]_s (mol/L). A dissolving base alkalizes its surface,
    so hs ≤ hbulk; the root lies in (0, hbulk].

    Robustness guard: when the bulk approaches/exceeds surface saturation the balance root
    leaves the bracket — there is no net flux, so the surface relaxes to the bulk; return
    hbulk rather than failing (the rate clamp in ``surface_ode`` then gives zero driving
    force). Keeps parameter sweeps from crashing at the supersaturation edge.
    """
    f = lambda hs: surface_residual(hs, hbulk, cbulk, p)
    f_hi = f(hbulk)
    if abs(f_hi) < 1e-30:
        return hbulk
    f_lo = f(hbulk * 1e-9)
    if np.sign(f_lo) == np.sign(f_hi):         # no root in (0, hbulk] → saturated edge
        return hbulk
    return brentq(f, hbulk * 1e-9, hbulk, xtol=1e-22, rtol=1e-12)


def surface_solubility(hs, p: Parameters) -> float:
    """Surface solubility C_s^surf = S0·(1 + [H+]_s/Ka_BH) (mol/L)."""
    return p.s0_mol * (1.0 + hs / p.kad)
