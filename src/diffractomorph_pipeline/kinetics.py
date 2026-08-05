"""Scattering-intensity dissolution kinetics — total angular signal ΣI(t) → KWW fit + shape-robust rate.

The **overall scattering intensity** workflow (the LD dissolution readout the project settled on): sum
the ring-channel intensities into one signal ``ΣI(t)`` (total angular scattering, ∝ particle area), clean
upward optical glitches (bubbles), and fit the free-amplitude KWW / stretched exponential that a
polydisperse population's superposed single-particle decays produce (see :mod:`empirical_fit`).

The **meaningful readouts** (per the optics analysis — the ΣI β is a scattering-weighted, uniformity-
flattered number, and raw τ is coupled to β):

- ``mean_relax_min`` = ``⟨t⟩ = (τ/β)·Γ(1/β)`` — the β-decoupled "how fast" timescale (or ``t50``),
- ``beta`` — decay **heterogeneity** (β→1 uniform/single-exponential; β<1 a spread of dissolution times),
- ``depth`` — extent of the decay,
- ``i0`` — back-extrapolated t=0 signal (recovers the start before the ~30 s injection→first-frame delay).

Compare rates across conditions with ``⟨t⟩`` / ``t50``, **not** raw ``τ``.

    from diffractomorph_pipeline import ingest, kinetics
    run = ingest.extract_run(rtf)
    t, sig = kinetics.despike_upward(run.t_min, kinetics.total_signal(run.I))
    fit = kinetics.fit_signal(t, sig)   # {tau_min, beta, mean_relax_min, t50_min, depth, i0, r2, flag}
"""
from __future__ import annotations

import numpy as np
from scipy.special import gamma

from .empirical_fit import fit_channel_rate


def total_signal(I) -> np.ndarray:
    """Total angular scattering ``ΣI(t)`` — sum of the ring-channel intensities per frame (``I`` is T×C).
    Summing over channels integrates out the Mie cross-channel smear (a redistribution *between* channels
    a sum is invariant to), leaving a cleaner signal than any single channel."""
    return np.asarray(I, dtype=float).sum(axis=1)


def despike_upward(t_min, y, *, z=4.0, half=3, return_mask=False):
    """Remove UPWARD glitch spikes from a monotone-ish decaying LD signal (``Copt`` or the total
    angular ``ΣI``), on the full absolute time grid.

    Bubbles / obscuration hits raise the signal above the local trend, but dissolution only *lowers*
    the undissolved material — so an upward excursion is never real. A Hampel filter (rolling-median
    baseline over ``±half`` frames, robust MAD, **upward tail only**), flagged frames interpolated from
    their good neighbours so time stays aligned with UV. A flagged point must also be **locally rising**
    (a contiguous above-trend run must *contain* a rising frame): the steep monotone *start* of a fast
    run sits above the local median without being a spike, and the rising-edge guard keeps that leading
    drop from being clipped flat. Returns ``(t_min, y_clean)``.
    """
    t = np.asarray(t_min, dtype=float)
    y = np.asarray(y, dtype=float).copy()
    n = y.size
    if n < 2 * half + 1:
        return (t, y, np.zeros(n, dtype=bool)) if return_mask else (t, y)
    base = np.array([np.median(y[max(0, k - half):k + half + 1]) for k in range(n)])
    resid = y - base
    mad = float(np.median(np.abs(resid - np.median(resid)))) or 1e-9
    above = (resid > z * 1.4826 * mad) & np.isfinite(y)        # upward excursions above the local trend
    rising = y > np.r_[y[0], y[:-1]]                            # locally rising (frame 0 → False)
    # flag each contiguous above-trend run that CONTAINS a rising entry (a bubble); the steep monotone
    # start is an above-trend run with no rising frame, so it is left intact.
    spike = np.zeros(n, dtype=bool)
    k = 0
    while k < n:
        if above[k]:
            j = k
            while j < n and above[j]:
                j += 1
            if rising[k:j].any():
                spike[k:j] = True
            k = j
        else:
            k += 1
    good = (~spike) & np.isfinite(y)
    if spike.any() and good.sum() >= 2:
        y[spike] = np.interp(t[spike], t[good], y[good])
    return (t, y, spike) if return_mask else (t, y)


def mean_relaxation(tau, beta) -> float:
    """KWW mean relaxation time ``⟨t⟩ = (τ/β)·Γ(1/β)`` — the β-decoupled characteristic timescale (the
    area under the normalized stretched-exponential decay). Prefer this (or :func:`half_time`) over raw
    ``τ`` when comparing rates, because ``τ`` and ``β`` are coupled in the fit."""
    if not (np.isfinite(tau) and np.isfinite(beta) and beta > 0):
        return float("nan")
    return float((tau / beta) * gamma(1.0 / beta))


def half_time(tau, beta) -> float:
    """KWW half-time ``t50 = τ·(ln 2)^(1/β)`` — time for the decaying part to fall to half."""
    if not (np.isfinite(tau) and np.isfinite(beta) and beta > 0):
        return float("nan")
    return float(tau * np.log(2.0) ** (1.0 / beta))


def fit_signal(t_min, signal, *, tau_bounds=(0.05, 500.0), beta_bounds=(0.2, 3.0)) -> dict:
    """Fit a decaying signal (e.g. ``ΣI(t)``) to the free-amplitude KWW and return the meaningful
    readouts: ``tau_min, beta, floor, depth, i0, r2, flag`` (from :func:`empirical_fit.fit_channel_rate`)
    plus the derived shape-robust rate ``mean_relax_min`` = ⟨t⟩ and ``t50_min``. Use ⟨t⟩ / t50 (not raw
    ``τ``) to compare rates across conditions."""
    f = dict(fit_channel_rate(t_min, signal, tau_bounds=tau_bounds, beta_bounds=beta_bounds))
    f["mean_relax_min"] = mean_relaxation(f["tau_min"], f["beta"])
    f["t50_min"] = half_time(f["tau_min"], f["beta"])
    return f
