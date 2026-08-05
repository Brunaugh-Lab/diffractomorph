"""Size-resolved Noyes–Whitney forward simulation + Mie mapping + rate fit.

Population carried as number ``n_i`` on the kernel size grid ``xm`` (µm diameter).
Single-particle radius evolves under a power-law dissolution time, with the bulk
driving force ``(1 − C(t)/Cs)`` from a mass balance with a deposition sink:

- **transport-limited** (τ ∝ r²):  ``dr/dt = −(G/r)·(1 − C/Cs)``
- **surface-limited**  (τ ∝ r):   ``dr/dt = −G·(1 − C/Cs)``

The fittable scalar ``G`` (≈ ``D·Cs/ρ``) is a starting-condition-controlled rate
that **is** comparable across runs (PSD, mass, Cs are inputs). The observable is
the Mie-forward scattering magnitude ``a(t) = Σ_c forward(kernel, n(t))``, matched
to the observed ``a(t)`` in fractional form.

The ensemble physics maps a log-normal PSD width σ_g to a KWW stretching exponent β
via :func:`ensemble_beta` (narrow PSD → β→1; broader PSD → lower β).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, minimize_scalar

OBSTRUCTION_COPT = 40.0
FLOOR_UM = 0.9          # R3 number-resolution floor (sub-floor bins are unreliable)
V_DEFAULT_ML = 40.0
REGIMES = {"transport": 2, "surface": 1}


# ── Ensemble σ_g → β for a log-normal PSD ────────────────────────────────────

def ensemble_beta(sigma_g: float, regime: str = "transport", nbin: int = 2000) -> float:
    """Predicted KWW β for a log-normal PSD of width ``sigma_g`` (full-decay fit).

    A narrow PSD gives β→1; broadening lowers β (e.g. transport regime, σ_g=1.86 → β≈0.78).
    """
    n = REGIMES[regime]
    s = np.log(sigma_g)
    lr = np.linspace(-4 * s, 4 * s, nbin)
    r0 = np.exp(lr)
    wnum = np.exp(-lr ** 2 / (2 * s ** 2)); wnum /= wnum.sum()
    vol0 = wnum * r0 ** 3
    tau = r0 ** n
    t = np.logspace(np.log10(tau.min() * 1e-2), np.log10(tau.max()), 400)
    M = np.array([(vol0 * np.clip(1 - tt / tau, 0, None) ** (3.0 / n)).sum() for tt in t])
    M /= M[0]
    return float(_fit_kww(t, M)[1])


# ── Output contract ──────────────────────────────────────────────────────────

@dataclass
class ForwardRun:
    t: np.ndarray
    a_pred: np.ndarray          # predicted scattering magnitude (normalized a/a0)
    r_t: np.ndarray             # (frames × grid) radius (µm) of each initial bin over time
    G_fit: float
    G_se: float
    beta_implied: float
    frac_diss_pred: float
    regime: str
    inputs: dict
    resid: np.ndarray | None = None
    resid_significant: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def save(self, path: Path | str) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, t=self.t, a_pred=self.a_pred, r_t=self.r_t,
                 resid=(self.resid if self.resid is not None else np.array([])))
        path.with_suffix(".json").write_text(json.dumps({
            "G_fit": self.G_fit, "G_se": self.G_se, "beta_implied": self.beta_implied,
            "frac_diss_pred": self.frac_diss_pred, "regime": self.regime,
            "inputs": self.inputs, "meta": self.meta}, indent=2, default=float))
        return path


# ── Core simulation ──────────────────────────────────────────────────────────

def _w_of(kernel):
    """Per-particle total scattering vs diameter (interpolator over the kernel grid)."""
    w = kernel.A.sum(axis=0)
    xm = kernel.xm
    return lambda d: np.interp(d, xm, w, left=0.0)


def simulate(n0, t, G, kernel, mass_mg, Cs_ugml, f_dep=0.5, V_ml=V_DEFAULT_ML,
             regime="transport", rate_mod=None) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the size-resolved population; return ``(a_pred_norm, r_t)``.

    ``n0`` is the number fraction on the kernel grid (raw daily Q0); ``G`` is the
    rate scalar; ``Cs_ugml`` the effective solubility; ``mass_mg`` the dosed mass.
    """
    n = REGIMES[regime]
    xm = np.asarray(kernel.xm, dtype=float)
    n0 = np.asarray(n0, dtype=float)
    r0 = xm / 2.0
    t = np.asarray(t, dtype=float)
    loaded = mass_mg * (1.0 - f_dep) / V_ml * 1000.0      # µg/mL suspended solid
    vol0 = (n0 * r0 ** 3).sum()

    # Integrate a single scalar "extent" E (∫ driving force dt): for transport
    # r²(t)=r0²−2G·E, for surface r(t)=r0−G·E. This is non-singular (the per-radius
    # dr/dt=−G/r blows up as r→0; the r²-form does not) and identical physics.
    m = 1.0 if rate_mod is None else np.asarray(rate_mod, dtype=float)

    def radii(E):
        if n == 2:
            return np.sqrt(np.clip(r0 ** 2 - 2.0 * G * m * E, 0.0, None))
        return np.clip(r0 - G * m * E, 0.0, None)

    def drive(E):
        C = loaded * (1.0 - (n0 * radii(E) ** 3).sum() / vol0)   # dissolved (mass balance)
        return max(0.0, 1.0 - C / Cs_ugml) if Cs_ugml > 0 else 1.0

    sol = solve_ivp(lambda _t, E: [drive(E[0])], (t[0], t[-1]), [0.0],
                    t_eval=t, method="RK45", rtol=1e-6, atol=1e-9)
    E_t = np.interp(t, sol.t, sol.y[0]) if sol.t.size != t.size else sol.y[0]
    r_t = np.array([radii(E) for E in E_t])               # (frames × grid)
    w = _w_of(kernel)
    a = np.array([(n0 * w(2.0 * r_t[k])).sum() for k in range(r_t.shape[0])])
    a = a / a[0] if a[0] > 0 else a
    return a, r_t


def _fit_kww(t, frac):
    def kww(t, k, b, c): return c + (1 - c) * np.exp(-(k * t) ** b)
    p, _ = curve_fit(kww, t, frac, p0=[1.0, 0.8, float(np.min(frac))],
                     bounds=([1e-4, 0.3, 0], [50, 2, 0.95]), maxfev=40000)
    return p


# ── Fit the mechanistic rate to an observed run ──────────────────────────────

def fit_rate(n0, t, observed, kernel, mass_mg, Cs_ugml, f_dep=0.5, V_ml=V_DEFAULT_ML,
             regime="transport", rate_mod=None) -> ForwardRun:
    """Fit the single rate ``G`` matching simulated → observed (fractional ``a``)."""
    obs = np.asarray(observed, dtype=float)
    obs = obs / obs[0]

    def loss(logG):
        a, _ = simulate(n0, t, 10 ** logG, kernel, mass_mg, Cs_ugml, f_dep, V_ml, regime, rate_mod)
        return float(np.sum((a - obs) ** 2))

    res = minimize_scalar(loss, bounds=(-4, 1.5), method="bounded")
    G = float(10 ** res.x)
    a, r_t = simulate(n0, t, G, kernel, mass_mg, Cs_ugml, f_dep, V_ml, regime, rate_mod)
    # crude SE from local curvature of the loss
    d = 0.05
    curv = (loss(res.x + d) - 2 * loss(res.x) + loss(res.x - d)) / d ** 2
    se = float(G * np.log(10) * np.sqrt(1.0 / curv)) if curv > 0 else float("nan")
    try:
        beta = float(_fit_kww(t - t[0], a)[1])
    except Exception:
        beta = float("nan")
    return ForwardRun(
        t=np.asarray(t), a_pred=a, r_t=r_t, G_fit=G, G_se=se, beta_implied=beta,
        frac_diss_pred=float(1 - a[-1]), regime=regime,
        inputs={"mass_mg": mass_mg, "Cs_ugml": Cs_ugml, "f_dep": f_dep, "V_ml": V_ml},
        meta={"rss": float(res.fun)},
    )


def residual(forward_run: ForwardRun, observed, noise_floor=None) -> ForwardRun:
    """Attach the observed − predicted residual and (optionally) its significance."""
    obs = np.asarray(observed, dtype=float)
    obs = obs / obs[0]
    forward_run.resid = obs - forward_run.a_pred
    if noise_floor is not None:
        scale = np.std(np.diff(obs)) or 1.0
        forward_run.resid_significant = np.abs(forward_run.resid) > noise_floor * scale
    return forward_run
