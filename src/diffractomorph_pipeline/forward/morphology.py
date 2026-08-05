"""Exploratory morphology extensions to the base Nernst–Brunner model.

Generalizes the pure transport-limited base model (:func:`forward.surface_ode.simulate`) with three
**independent, empirical** knobs, all reproducing the base model at their defaults:

- ``a_area`` (default 1/3) — the φ exponent: how each bin's reactive area evolves as solid
  disappears. 1/3 = smooth shrinking sphere (area ∝ φ^{1/3}); → 1 = first-order in remaining mass.
- ``b_size`` (default 2) — the r₀ exponent: the initial-size dependence of the rate. 2 = τ∝r²
  (stagnant film h ≈ r); → 0 = size-independent.
- ``rate_scale`` (default 1) — one dimensionless global multiplier on the drug diffusivity ``dd``,
  i.e. the effective transport coefficient ``k_tr = dd · rate_scale``. It absorbs the (unmeasured,
  Wilke–Chang) uncertainty on ``dd`` (see :func:`forward.params.default_diffusivities`) and any
  BET-accessible-area rate enhancement. **Do not fit ``rate_scale`` and ``dd`` jointly** — they are
  degenerate; ``dd`` is the physical anchor, ``rate_scale`` the single fitted rate multiplier. Its
  ``b_size≠2`` meaning is normalized against a **fixed** reference radius ``r_ref_um`` (default 1 µm,
  run-independent) — never a per-run PSD statistic — so a "global" ``rate_scale`` is comparable
  across conditions. ``r_ref`` cancels at ``b_size=2``, so the base model does not depend on it.

``a_area`` and ``b_size`` are *diagnostic* exponents, not independent mechanisms: under a simple
self-similar particle they are linked. Separating them lets the initial size → reactive-area map
(rugosity/accessibility, ``b_size``) differ from how reactive area evolves during dissolution
(``a_area``). That is plausible for this rugose, heterogeneously-wetted solid — but it is an
empirical decoupling, and should be stated as such, not read as two independently validated laws.

These three map to three independent observables (the identifiability the old ``w_por`` lacked):
``rate_scale`` → absolute UV / total-angular time scale; ``a_area`` → aggregate trajectory
curvature; ``b_size`` → forward-kernel channel-rate size slope (estimate it through the full Mie
operator, not raw channel labels).

Deprecated
----------
- ``w_por`` — the old *single* dial that moved ``a_area`` and ``b_size`` together
  (``a = 1/3 + 2w/3``, ``b = 2(1−w)``). Kept only as a compatibility mapping that emits
  ``DeprecationWarning``; pass ``a_area`` / ``b_size`` directly. Because it carried no free rate
  scale, ``w_por`` confounded the per-bin size-shape with the ensemble timescale — ``rate_scale``
  is the knob it was missing.
- ``f_acc`` — a size-dependent cap on each bin's dissolvable fraction (a permanently inert core).
  The size-law is **not empirically validated** (demoted in V5) and is off by default; it is not the
  recommended path and is *not* replaced by another depth cap. No depth cap is applied unless
  independent residual-solid evidence requires one.

Both reuse the shared integrator :func:`forward.surface_ode._integrate`; the base model is unchanged
at the defaults (``MorphologyParams()`` reproduces :func:`forward.surface_ode.simulate`).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from diffractomorph_pipeline.forward.params import PSD, Parameters
from diffractomorph_pipeline.forward.surface_ode import DissolutionRun, _integrate

# Physically reasonable ranges for an external fitter (NOT enforced at construction, so diagnostic
# runs can go outside them). ``rate_scale`` is strictly > 0; that bound *is* enforced.
PARAM_BOUNDS = {"a_area": (0.0, 1.2), "b_size": (0.0, 2.5), "rate_scale": (0.0, float("inf"))}

_A_BASE, _B_BASE = 1.0 / 3.0, 2.0        # base transport-limited exponents (τ∝r² shrinking core)


@dataclass
class MorphologyParams:
    """Independent per-bin rate-law exponents + one global rate scale (defaults = base model)."""
    a_area: float = _A_BASE           # φ exponent (reactive-area evolution); 1/3 = shrinking sphere
    b_size: float = _B_BASE           # r₀ exponent (initial-size dependence); 2 = τ∝r²
    rate_scale: float = 1.0           # dimensionless multiplier on dd: k_tr = dd·rate_scale
    r_ref_um: float = 1.0             # FIXED reference radius (µm) for the geom normalization —
                                      # run-independent, so rate_scale means the same across PSDs.
                                      # Cancels exactly at b_size=2, so the base model is unaffected.
    w_por: float | None = None        # DEPRECATED single dial → (a_area, b_size); use those instead
    acc_enabled: bool = False         # enable the (unvalidated) size-dependent dissolvable-frac cap
    acc_f_base: float = 1.0           # f_acc intercept
    acc_f_scale: float = 0.0          # coupling to the high-energy area fraction φ_hi
    acc_phi0: float = 0.04            # φ₀: size-independent high-energy area fraction (IGC fit)
    acc_kappa_um: float = 0.159       # κ (µm): 1/size coefficient of φ_hi = φ₀ + κ/L

    def __post_init__(self):
        if self.w_por is not None:                     # deprecated coupled mapping
            if self.a_area != _A_BASE or self.b_size != _B_BASE:
                raise ValueError("pass either w_por (deprecated) or a_area/b_size, not both")
            warnings.warn(
                "MorphologyParams.w_por is deprecated; pass a_area/b_size directly "
                "(w_por=w maps to a_area=1/3+2w/3, b_size=2(1-w)).",
                DeprecationWarning, stacklevel=2)
            self.a_area = _A_BASE + 2.0 / 3.0 * self.w_por
            self.b_size = _B_BASE * (1.0 - self.w_por)
        if self.rate_scale <= 0.0:
            raise ValueError(f"rate_scale must be > 0, got {self.rate_scale!r}")
        if self.r_ref_um <= 0.0:
            raise ValueError(f"r_ref_um must be > 0, got {self.r_ref_um!r}")

    # morph_a / morph_b retained as aliases (used by _morph_shape and existing callers/tests).
    @property
    def morph_a(self) -> float:
        """φ exponent of the per-bin rate (== ``a_area``)."""
        return self.a_area

    @property
    def morph_b(self) -> float:
        """r₀ exponent of the per-bin rate (== ``b_size``)."""
        return self.b_size

    def accessible_fraction(self, diam_um) -> np.ndarray:
        """Per-bin dissolvable fraction f_acc(L). Ones when ``acc_enabled`` is False (then the
        uniform ``Parameters.freeze_frac`` floor applies instead)."""
        d = np.asarray(diam_um, float)
        if not self.acc_enabled:
            return np.ones_like(d)
        phi_hi = self.acc_phi0 + self.acc_kappa_um / np.maximum(d, 1e-6)
        return np.clip(self.acc_f_base + self.acc_f_scale * phi_hi, 0.0, 1.0)


def _morph_shape(p: Parameters, morph: MorphologyParams):
    """Rate-law shape for the morphology extension (reduces to the base model at the defaults).

    ``geom = rate_scale · (r/r_ref)^{-b}/r_ref²`` reduces to ``1/r²`` at ``b_size=2`` and to a
    size-independent ``rate_scale/r_ref²`` at ``b_size=0``. ``r_ref`` is a **fixed** reference radius
    (``morph.r_ref_um``, default 1 µm) — run-independent, so ``rate_scale`` means the same thing
    across every PSD (essential for a cross-run "global" fit). It cancels exactly at ``b_size=2``, so
    the base model is independent of ``r_ref``; only ``b_size≠2`` runs feel it.
    """
    a_exp, b_exp, k = morph.a_area, morph.b_size, morph.rate_scale
    r_ref = morph.r_ref_um * 1e-4                          # µm → cm (rini is in cm); fixed, per-study

    def shape(rini, q0, diam_um, live):
        geom = np.zeros_like(rini)
        geom[live] = k * (rini[live] / r_ref) ** (-b_exp) / r_ref ** 2
        facc = morph.accessible_fraction(diam_um)
        floor_q = (1.0 - facc) * q0 if morph.acc_enabled else p.freeze_frac * q0
        return a_exp, geom, floor_q

    return shape


def simulate(p: Parameters, psd: PSD, dose_mg: float, ph_bulk: float,
             morph: MorphologyParams, t_end: float = 1800.0, n_eval: int = 181,
             method: str = "BDF", c0_ugml: float = 0.0) -> DissolutionRun:
    """Base Nernst–Brunner + the independent-exponent morphology terms (``a_area``, ``b_size``,
    ``rate_scale``; optional ``f_acc``).

    Identical to :func:`forward.surface_ode.simulate` when ``morph = MorphologyParams()``
    (``a_area=1/3``, ``b_size=2``, ``rate_scale=1``, ``acc_enabled=False``). See the module
    docstring for the caveats. ``a_area``/``b_size``/``rate_scale`` are recorded in
    ``DissolutionRun.inputs`` for provenance.
    """
    return _integrate(p, psd, dose_mg, ph_bulk, _morph_shape(p, morph),
                      t_end=t_end, n_eval=n_eval, method=method, c0_ugml=c0_ugml,
                      extra_inputs={"a_area": morph.a_area, "b_size": morph.b_size,
                                    "rate_scale": morph.rate_scale, "r_ref_um": morph.r_ref_um,
                                    "acc_enabled": morph.acc_enabled, "w_por": morph.w_por})
