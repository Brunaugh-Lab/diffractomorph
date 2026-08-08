"""Manual-derived physical Mie + detector forward operator — HELOS R3 audit (isolated).

This module is deliberately **separate from the production operator** (:mod:`optics.mie` /
:mod:`optics.mie_build`) and does not replace it. It exists to test — during the NIST glass-bead
audit — whether a physically complete forward model maps the *certified* SRM-1021 size distribution
to the observed 31-ring detector pattern, without the two shortcuts the production build takes
(no scattering cross-section; centre-angle × θ² instead of an annular integral).

Per-ring per-particle response
------------------------------
For particle diameter ``d`` and detector ring ``j`` spanning ``[θ_lo, θ_hi]``::

    A_j(d) = C_sca(d) · ∫_{θ_lo}^{θ_hi} p(θ | d, m, λ) · 2π sinθ dθ
    C_sca(d) = π (d/2)² Q_sca(d)

- ``Q_sca`` from ``miepython.efficiencies`` (reliable) — the cross-section the production build omits.
- ``p(θ)`` is the unpolarized phase function ``miepython.i_unpolarized(..., norm='one')``, verified to
  integrate to 1 over 4π (so ``C_sca`` must be applied explicitly — it is NOT already in ``p``).
- The ring integral uses Gauss–Legendre quadrature in θ and is convergence-tested
  (:func:`ring_response` with increasing ``n_quad``), not a single centre angle.

Detector geometry
-----------------
The manufacturer manual gives the R3 focal length, 31 measuring-class limits, and the Fraunhofer
first-minimum relation. :func:`r3_manual_radii` uses those quantities to reconstruct 32 nominal
optical-equivalent boundaries (beam centre plus 31 transformed class limits), and
:func:`r3_manual_rings` maps them to medium scattering angles. This is traceable to the manual and
independent of the NIST measurements, but it is not mechanical detector metrology: the manual does
not supply a detector drawing, ring gaps/centre elements, channel gains, aberration, or blur.

:func:`log_rings` remains only a generic sensitivity helper. It assumes log-spaced angular
boundaries between supplied endpoints and must not be described as hardware geometry.

Optical constants (fixed; optics are not free parameters during morphology work):
He–Ne ``λ₀ = 0.6328 µm``; soda-lime glass ``n = 1.52 + 0i`` (SRM 1021 certificate value). The
measurement medium was **pH 7 Britton–Robinson buffer** (dilute aqueous); its index is not measured,
so ``n_med = 1.331`` is a nominal water value carried as a bounded-sensitivity input, not a fact.
"""
from __future__ import annotations

from dataclasses import dataclass

import miepython as mp
import numpy as np
from numpy.polynomial.legendre import leggauss

LAMBDA0_UM = 0.6328          # He–Ne vacuum wavelength (µm)
N_MED = 1.331                # NOMINAL medium index (water); real medium = pH 7 Britton–Robinson
                             # buffer, unmeasured → carry as a bounded-sensitivity input, not a fact
GLASS_RI = 1.52              # soda-lime glass, SRM 1021 certificate value (n = 1.52 + 0i)
R3_FOCAL_MM = 100.0          # HELOS R3 focal length
R3_THETA_MIN_DEG = 0.19      # generic log_rings sensitivity default; not used by manual geometry
N_CHANNELS = 31


@dataclass
class DetectorRings:
    """Angular boundaries of the 31 detector rings (degrees)."""
    theta_lo: np.ndarray
    theta_hi: np.ndarray

    def __len__(self) -> int:
        return len(self.theta_lo)

    @property
    def theta_center(self) -> np.ndarray:
        """Geometric-mean centre angle of each ring (deg)."""
        return np.sqrt(self.theta_lo * self.theta_hi)

    @property
    def is_monotonic(self) -> bool:
        return bool(np.all(np.diff(self.theta_lo) > 0) and np.all(self.theta_hi > self.theta_lo))


def log_rings(theta_min_deg: float = R3_THETA_MIN_DEG, theta_max_deg: float = 30.0,
              n: int = N_CHANNELS) -> DetectorRings:
    """``n`` contiguous log-spaced rings between two ASSUMED angular endpoints (deg).

    **Not a physical geometry.** This assumes Fraunhofer-style log-spaced angular boundaries; it does
    NOT use ``θ = arctan(r/f)`` because no ring radii are supplied. Both ``theta_min_deg`` and
    ``theta_max_deg`` are bounded-sensitivity inputs — sweep them, do not treat either as known, and
    do not adopt the old PAQXOS-fitted ``θ_max ≈ 39.19°``. For real geometry use :func:`rings_from_radii`.
    """
    edges = np.logspace(np.log10(theta_min_deg), np.log10(theta_max_deg), n + 1)
    return DetectorRings(edges[:-1], edges[1:])


def rings_from_radii(radii_mm, focal_mm: float = R3_FOCAL_MM, n_med: float = N_MED,
                     mapping: str = "medium") -> DetectorRings:
    """Ring boundaries from a physical detector-plane radius table.

    The **radius→scattering-angle** mapping matters and is NOT ``arctan(r/f)``: Mie theory uses the
    angle **in the dispersion medium**, and a sine-condition Fourier lens with liquid→air refraction
    gives ``r = f·sin(θ_air)`` with Snell ``n_med·sin(θ_med) = sin(θ_air)``, so:

        ``mapping="medium"`` (PRIMARY):  θ_med = arcsin(r / (n_med · f))
        ``mapping="arctan"``  (approx):  θ    = arctan(r / f)      — small-angle-distorting, diagnostic only

    ``radii_mm`` (length n+1, inner→outer) **must be strictly increasing** — unsorted input is rejected
    (silent sorting could hide a reversed channel mapping).
    """
    r = np.asarray(radii_mm, float)
    if not np.all(np.diff(r) > 0):
        raise ValueError("radii_mm must be strictly increasing (inner→outer); refusing to sort silently")
    if mapping == "medium":
        s = r / (n_med * focal_mm)
        if np.any(s > 1.0):
            raise ValueError("r/(n_med·f) > 1: radii exceed the collectible aperture for this n_med/f")
        th = np.degrees(np.arcsin(s))
    elif mapping == "arctan":
        th = np.degrees(np.arctan(r / focal_mm))
    else:
        raise ValueError(f"mapping must be 'medium' or 'arctan', got {mapping!r}")
    return DetectorRings(th[:-1], th[1:])


# HELOS R3 detector-plane geometry from the manual's measuring-range table (page 14) + Fraunhofer
# r = 1.22·λ·f/x (page 55): for R3 (f=100 mm, λ=632.8 nm) → r[mm] = 77.2016 / x[µm]. The RADII are
# manual-supported; the ANGLES depend on the optical mapping (see rings_from_radii). With the
# sine-condition/Snell medium mapping the inner edge → 0.190° (independently recovering the hardware
# θ_min) and the outer → 40.1°. NOT a fitted θmin/θmax.
R3_FRAUNHOFER_MM_UM = 1.22 * LAMBDA0_UM * R3_FOCAL_MM      # = 77.2016 mm·µm


def r3_manual_radii(size_edges_um) -> np.ndarray:
    """Frozen R3 detector-plane boundary RADII (mm) from the manual size boundaries (do NOT fit).
    Innermost boundary is the beam centre (r=0). ``size_edges_um`` = the 31 R3 class upper edges."""
    return np.sort(np.r_[0.0, R3_FRAUNHOFER_MM_UM / np.asarray(size_edges_um, float)])


def r3_manual_rings(size_edges_um, n_med: float = N_MED, mapping: str = "medium") -> DetectorRings:
    """Frozen R3 annuli: manual radii → medium scattering angles (default) via :func:`rings_from_radii`."""
    return rings_from_radii(r3_manual_radii(size_edges_um), R3_FOCAL_MM, n_med, mapping)


def scattering_efficiency(d_um: float, n_particle: float, n_med: float = N_MED,
                          lam0: float = LAMBDA0_UM) -> float:
    """``Q_sca`` for one sphere (``miepython.efficiencies``; real d and λ, medium n_env).

    ``miepython.efficiencies`` forms the relative index internally as ``m / n_env``, so it must be
    given the **absolute** particle index here — NOT ``n_particle / n_med`` (that would divide by the
    medium index twice, under-estimating Q_sca by ~26 % at 1 µm down to ~5 % at 10 µm).
    """
    _qext, qsca, _qback, _g = mp.efficiencies(complex(n_particle, 0.0), d_um, lam0, n_env=n_med)
    return float(qsca)


def csca_um2(d_um: float, n_particle: float, n_med: float = N_MED, lam0: float = LAMBDA0_UM) -> float:
    """Scattering cross-section ``C_sca = π (d/2)² Q_sca`` (µm²) — omitted by the production build."""
    return float(np.pi * (d_um / 2.0) ** 2 * scattering_efficiency(d_um, n_particle, n_med, lam0))


def ring_response(d_um: float, rings: DetectorRings, n_particle: float, n_med: float = N_MED,
                  lam0: float = LAMBDA0_UM, n_quad: int = 64) -> np.ndarray:
    """Per-particle collected power in each ring: ``C_sca · ∫ p(θ) 2π sinθ dθ``. Returns ``(C,)``.

    Annular integration by ``n_quad``-node Gauss–Legendre per ring (convergence-tested). Non-negative
    and distributed across rings — never collapsed to a single characteristic channel.
    """
    m = complex(n_particle, 0.0) / n_med
    x = np.pi * d_um * n_med / lam0
    cs = csca_um2(d_um, n_particle, n_med, lam0)
    nodes, wts = leggauss(n_quad)
    out = np.empty(len(rings))
    for j in range(len(rings)):
        lo, hi = np.deg2rad(rings.theta_lo[j]), np.deg2rad(rings.theta_hi[j])
        th = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wts
        p = mp.i_unpolarized(m, x, np.cos(th), norm="one")
        out[j] = cs * float(np.sum(w * p * 2.0 * np.pi * np.sin(th)))
    return out


def response_matrix(diams_um, rings: DetectorRings, n_particle: float, n_med: float = N_MED,
                    lam0: float = LAMBDA0_UM, n_quad: int = 64) -> np.ndarray:
    """Diameter→channel response matrix ``A`` (C × D): column ``i`` = :func:`ring_response` for
    ``diams_um[i]``. This is the audit's principal physical output."""
    diams = np.asarray(diams_um, float)
    return np.stack([ring_response(d, rings, n_particle, n_med, lam0, n_quad) for d in diams], axis=1)


def total_scattering_check(d_um: float, n_particle: float, n_med: float = N_MED,
                           lam0: float = LAMBDA0_UM, n_quad: int = 2000) -> tuple[float, float]:
    """Integrate ``C_sca · p(θ) 2π sinθ`` over the FULL sphere and compare to ``C_sca``.

    Returns ``(integrated, csca)``; they should agree once quadrature resolves the forward peak —
    the normalization/cross-section sanity check (miepython ``p`` integrates to 1, so the full-sphere
    integral of ``C_sca·p`` must return ``C_sca``)."""
    m = complex(n_particle, 0.0) / n_med
    x = np.pi * d_um * n_med / lam0
    cs = csca_um2(d_um, n_particle, n_med, lam0)
    nodes, wts = leggauss(n_quad)                       # over μ = cosθ ∈ [-1, 1]
    p = mp.i_unpolarized(m, x, nodes, norm="one")
    integrated = cs * float(np.sum(wts * p * 2.0 * np.pi))
    return integrated, cs
