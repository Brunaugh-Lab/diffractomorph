"""Predicted q3 (volume) size-distribution evolution from a dissolution run.

The model's particle bins keep their identity but shrink as drug dissolves, so their *sizes*
move over time. :func:`q3_evolution` re-histograms the still-solid volume onto a FIXED size
grid at each frame, so every snapshot shares one size axis — the Sympatec q3 view — letting you
watch the distribution shift to smaller sizes across time.

    from diffractomorph_pipeline.forward import predict, q3_evolution
    run = predict(psd, ph=4.5, dose_mg=0.2, t_end=1200, n_eval=101)   # 20 min, every 12 s
    q3 = q3_evolution(run)          # tidy [time_s, time_min, size_um, q3_pct]
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def q3_evolution(run, grid_um=None, normalize="frame") -> pd.DataFrame:
    """Model q3 (volume %) on a fixed size grid over time — the evolving PSD.

    Parameters
    ----------
    run
        A :class:`~forward.surface_ode.DissolutionRun` (from ``predict`` / ``simulate``).
    grid_um
        Fixed size grid to bin onto (µm); defaults to the run's initial diameters, so ``t=0``
        reproduces the injected q3 exactly.
    normalize
        ``"frame"`` — per-frame renormalized q3 (the instrument convention; each snapshot sums
        to 100 %, showing the shape shift), or ``"initial"`` — fraction of the injected volume
        (the total shrinks as drug dissolves).

    Returns
    -------
    DataFrame
        Tidy ``[time_s, time_min, size_um, q3_pct]`` — one row per (frame, size class).
    """
    q = np.asarray(run.qundiss, float)                  # frames × nbin, undissolved amount (∝ volume)
    d_cur = 2.0 * np.asarray(run.radius_um, float)      # frames × nbin, current diameters
    grid = np.asarray(run.diam0_um if grid_um is None else grid_um, float)
    n_frame, n_grid = q.shape[0], grid.size
    lg = np.log(grid)
    binned = np.zeros((n_frame, n_grid))
    for f in range(n_frame):                            # spread each shrinking bin across its two
        pos = np.interp(np.log(np.clip(d_cur[f], grid[0] * 1e-6, None)), lg, np.arange(n_grid))
        lo = np.clip(np.floor(pos).astype(int), 0, n_grid - 1)   # nearest grid centres (linear in log-size)
        hi = np.clip(lo + 1, 0, n_grid - 1)
        frac = pos - lo
        np.add.at(binned[f], lo, q[f] * (1.0 - frac))
        np.add.at(binned[f], hi, q[f] * frac)
    if normalize == "frame":
        total = np.clip(binned.sum(axis=1, keepdims=True), 1e-300, None)
    elif normalize == "initial":
        total = np.clip(binned[0].sum(), 1e-300, None)
    else:
        raise ValueError(f"normalize must be 'frame' or 'initial', got {normalize!r}")
    q3 = 100.0 * binned / total
    t_s = np.asarray(run.t, float)
    return pd.DataFrame({
        "time_s": np.repeat(np.round(t_s, 1), n_grid),
        "time_min": np.repeat(np.round(t_s / 60.0, 3), n_grid),
        "size_um": np.tile(grid, n_frame),
        "q3_pct": q3.ravel(),
    })


def compare_bucket_kinetics(bk_a, bk_b, labels=("a", "b"), tol=0.15) -> pd.DataFrame:
    """Align two :func:`bucket_kinetics` tables by nearest size → per-bucket rate comparison.

    The two runs may sit on slightly different size grids (e.g. a QC-anchored run vs a
    snapshot-anchored one), so buckets are matched within ``tol`` µm. ``tau_ratio`` is
    ``τ_b / τ_a`` (>1 ⇒ run *b* dissolves that bucket slower). Returns one row per matched
    bucket with a finite τ in both.
    """
    la, lb = labels
    a = bk_a.sort_values("size_um"); b = bk_b.sort_values("size_um")
    m = pd.merge_asof(b, a, on="size_um", direction="nearest", tolerance=tol, suffixes=(f"_{lb}", f"_{la}"))
    m = m.dropna(subset=[f"tau_r2_min_{la}", f"tau_r2_min_{lb}"])
    return pd.DataFrame({
        "size_um": m["size_um"].round(3),
        f"tau_{la}_min": m[f"tau_r2_min_{la}"], f"tau_{lb}_min": m[f"tau_r2_min_{lb}"],
        "tau_ratio": (m[f"tau_r2_min_{lb}"] / m[f"tau_r2_min_{la}"]).round(3),
        f"t50_{la}_min": m[f"t50_min_{la}"], f"t50_{lb}_min": m[f"t50_min_{lb}"],
    }).reset_index(drop=True)


def _cross_time(frac, t_min, target):
    """First time (min) the (monotone-decreasing) remaining fraction reaches ``target``."""
    if frac[-1] > target:
        return float("nan")                             # never gets there within the run
    return round(float(np.interp(-target, -frac, t_min)), 3)   # frac ↓ ⇒ −frac ↑ (interp needs ascending x)


def bucket_kinetics(run, active=(0.03, 0.97)) -> pd.DataFrame:
    """Per-size-bucket dissolution parameters from a run (the native model bins).

    Each populated bin dissolves by the transport-limited shrinking-core **r²-law**
    ``remaining = (1 − t/τ)^{3/2}``, so its characteristic parameter is the dissolution time
    ``τ`` — recovered as ``−1/slope`` of ``(r/r₀)² = remaining^{2/3}`` vs time over the
    actively-dissolving window ``active`` (fraction remaining between the two bounds). Also
    reports the model-free ``t50``/``t90`` (time to 50 %/90 % dissolved), the fraction of the
    dose in the bin, and the final % dissolved. Times in **minutes** (the model integrates in
    seconds). One row per populated bucket.
    """
    q = np.asarray(run.qundiss, float)
    q0 = q[0]
    diam0 = np.asarray(run.diam0_um, float)
    t_min = np.asarray(run.t, float) / 60.0
    lo, hi = active
    rows = []
    for i in np.where(q0 > q0.max() * 1e-4)[0]:         # populated buckets only
        frac = np.clip(q[:, i] / q0[i], 1e-9, 1.0)
        m = (frac > lo) & (frac < hi)                   # active-dissolution window
        tau = float(-1.0 / np.polyfit(t_min[m], frac[m] ** (2.0 / 3.0), 1)[0]) if m.sum() >= 2 else float("nan")
        rows.append({
            "size_um": round(float(diam0[i]), 3),
            "dose_frac_pct": round(100.0 * float(q0[i] / q0.sum()), 2),
            "tau_r2_min": round(tau, 3) if np.isfinite(tau) else float("nan"),
            "t50_min": _cross_time(frac, t_min, 0.5),
            "t90_min": _cross_time(frac, t_min, 0.1),
            "pct_diss_final": round(100.0 * (1.0 - float(frac[-1])), 2),
        })
    return pd.DataFrame(rows)
