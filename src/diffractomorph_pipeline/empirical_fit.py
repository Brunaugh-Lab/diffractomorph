"""Empirical dissolution-kinetics fits to the laser-diffraction signal.

**Per-channel stretched-exponential (KWW) decay, free amplitude.** Each detector channel's raw
intensity ``I(c,t)`` is fit directly to

    ``I(t) = plateau + amp·exp(−(t/τ)^β)``            (Kohlrausch–Williams–Watts, offset)

giving, per channel: ``τ`` (a decay time; the RATE scale), ``β`` (the stretch/SHAPE exponent),
``floor = plateau/(plateau+amp)`` (relative plateau) and ``depth = 1 − floor`` (how far it fell).
The fitted start ``I₀ = plateau + amp`` is a **free parameter** — we do NOT normalize by a
start-frame window. That removes a real bias: at fast pH the signal already decays within the
first frames, so normalizing by ``mean(first 3 frames)`` underestimates the true start, inflates
``I/I₀`` above 1, and pushes the fit to ``β>1`` with distorted ``τ``. Fitting a free amplitude lets
the data set the start.

This is a **data-only** fit, indexed by detector channel — a channel→size mapping is a *kernel*
interpretation (the thing the optical smear puts in question) and is deferred to model comparison.

**Why KWW, not a plain exponential (data-driven, not assumed).** A channel is
``I(c,t) = Σ_size A[c,size]·n(size,t)`` — a kernel-weighted **mixture** of many single-particle
decays, so a first-order fit (``β = 1``) is the wrong shape: on real channels it leaves strongly
**autocorrelated residuals** and a free-``β`` fit lands at ``β < 1`` (polydispersity). We impose no
shape — ``β`` floats. See ``test_kww_beats_first_order_on_a_mixture``. Still a descriptor, not a
mechanism (for that, the forward model computes ``I=A·n`` directly).

Each fit carries a ``flag`` (``low_r2`` / ``no_decay`` / ``beta_bound`` / ``tau_bound`` /
``fit_failed``) so ill-behaved channels are **marked**, not silently dropped.

    from diffractomorph_pipeline.empirical_fit import fit_channel_rates
    df = fit_channel_rates(nf.clean_t, nf.clean_I, channels, active=nf.active_channels)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def kww_decay(t, plateau, amp, tau, beta):
    """``I(t) = plateau + amp·exp(−(t/τ)^β)`` — absolute KWW decay (β=1 → first-order)."""
    return plateau + amp * np.exp(-(np.asarray(t, dtype=float) / tau) ** beta)


def stretched_decay(t, floor, tau, beta):
    """Normalized KWW shape ``floor + (1−floor)·exp(−(t/τ)^β)`` (β=1 → first-order). Reference /
    for generating normalized test signals; the channel fit uses the free-amplitude :func:`kww_decay`."""
    return floor + (1.0 - floor) * np.exp(-(np.asarray(t, dtype=float) / tau) ** beta)


def first_order_decay(t, floor, tau):
    """``floor + (1−floor)·exp(−t/τ)`` — the β=1 special case; the reference form the channel data
    rejects (see the module docstring)."""
    return floor + (1.0 - floor) * np.exp(-np.asarray(t, dtype=float) / tau)


def fit_channel_rate(t_min, signal, tau_bounds=(0.05, 500.0), beta_bounds=(0.2, 3.0),
                     r2_min=0.8, depth_min=0.05) -> dict:
    """Fit one channel's absolute ``I(t)`` to :func:`kww_decay` (free amplitude).

    Returns a dict: ``tau_min, beta, floor, depth, i0, r2, flag`` — ``flag`` is a comma-joined set
    of warnings (``few_frames``, ``low_r2``, ``no_decay``, ``beta_bound``, ``tau_bound``,
    ``fit_failed``) and is
    empty for a clean fit.
    """
    t = np.asarray(t_min, dtype=float)
    s = np.asarray(signal, dtype=float)
    finite = np.isfinite(t) & np.isfinite(s)
    t, s = t[finite], s[finite]
    if t.size < 5:
        nan = float("nan")
        return {"tau_min": nan, "beta": nan, "floor": nan, "depth": nan, "i0": nan,
                "r2": nan, "flag": "few_frames"}
    lo, hi = float(np.min(s)), float(np.max(s))
    span = max(hi - lo, 1e-9)
    cap = max(abs(hi), 1e-6) * 3.0
    nan = float("nan")
    try:
        p, _ = curve_fit(kww_decay, t, s, p0=[max(lo, 0.0), span, 2.0, 1.0],
                         bounds=([0.0, 0.0, tau_bounds[0], beta_bounds[0]],
                                 [cap, cap, tau_bounds[1], beta_bounds[1]]), maxfev=20000)
    except Exception:
        return {"tau_min": nan, "beta": nan, "floor": nan, "depth": nan, "i0": nan,
                "r2": nan, "flag": "fit_failed"}
    plateau, amp, tau, beta = (float(v) for v in p)
    i0 = plateau + amp
    floor = plateau / i0 if i0 > 0 else nan
    depth = amp / i0 if i0 > 0 else nan
    yhat = kww_decay(t, *p)
    ss_tot = float(np.sum((s - s.mean()) ** 2))
    r2 = 1.0 - float(np.sum((s - yhat) ** 2)) / ss_tot if ss_tot > 0 else nan
    flags = []
    if not np.isfinite(r2) or r2 < r2_min:
        flags.append("low_r2")
    if not np.isfinite(depth) or depth < depth_min:
        flags.append("no_decay")
    if beta <= beta_bounds[0] * 1.02 or beta >= beta_bounds[1] * 0.98:
        flags.append("beta_bound")
    if tau <= tau_bounds[0] * 1.02 or tau >= tau_bounds[1] * 0.98:
        flags.append("tau_bound")
    return {"tau_min": tau, "beta": beta, "floor": floor, "depth": depth, "i0": i0,
            "r2": r2, "flag": ",".join(flags)}


def fit_channel_rates(t_min, I, channels, *, active=None, char_size=None,
                      reliable=None, min_frames=8) -> pd.DataFrame:
    """Per-channel free-amplitude KWW fit over one run's cleaned signal ``I`` (T×C).

    One row per fit channel: ``channel[, char_size_um], tau_min, beta, floor, depth_pct, i0, r2,
    n_frames, flag``. ``active`` restricts to the noise-filter-admitted channels (pass
    ``noise_filter(...).active_channels``); ``char_size`` (kernel channel→size) is an optional
    geometric annotation, off by default.

    ``reliable`` is an optional per-frame trust mask (T×C, e.g. from
    :meth:`~diffractomorph_pipeline.noise_surface.NoiseSurface.frame_reliable`): each channel is
    then fit only over the frames where it sits above its own noise floor, so an emerging channel
    is fit from where it lights up and a channel that dissolves into the noise is not fit on its
    dead tail. Channels left with fewer than ``min_frames`` reliable frames are flagged
    ``"few_frames"``.
    """
    I = np.asarray(I, dtype=float)
    t_min = np.asarray(t_min, dtype=float)
    reliable = np.asarray(reliable, dtype=bool) if reliable is not None else None
    channels = list(channels)
    keep = set(active) if active is not None else set(channels)
    rows = []
    for k, ch in enumerate(channels):
        if ch not in keep:
            continue
        if reliable is not None:
            m = reliable[:, k]
            tt, yy = t_min[m], I[m, k]
        else:
            tt, yy = t_min, I[:, k]
        f = fit_channel_rate(tt, yy)
        flag = f["flag"]
        if reliable is not None and int(yy.size) < min_frames:
            flag = ",".join([flag, "few_frames"]) if flag else "few_frames"
        row = {"channel": int(ch)}
        if char_size is not None:
            row["char_size_um"] = round(float(char_size[k]), 3)
        row.update({
            "tau_min": round(f["tau_min"], 3) if np.isfinite(f["tau_min"]) else np.nan,
            "beta": round(f["beta"], 3) if np.isfinite(f["beta"]) else np.nan,
            "floor": round(f["floor"], 4) if np.isfinite(f["floor"]) else np.nan,
            "depth_pct": round(100.0 * f["depth"], 2) if np.isfinite(f["depth"]) else np.nan,
            "i0": round(f["i0"], 2) if np.isfinite(f["i0"]) else np.nan,
            "r2": round(f["r2"], 4) if np.isfinite(f["r2"]) else np.nan,
            "n_frames": int(yy.size),
            "flag": flag,
        })
        rows.append(row)
    return pd.DataFrame(rows)
