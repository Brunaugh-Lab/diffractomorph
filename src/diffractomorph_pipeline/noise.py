"""Noise-floor characterization — the significance threshold for shape change.

A non-dissolving standard (NIST glass beads) titrated across optical-concentration
(Copt) levels gives, at each level, a set of frame-to-frame Wasserstein-1 (W₁)
distances on the channel distribution. Those are **pure noise** (the PSD isn't
changing), so their **95th percentile is the noise floor at that Copt** — the
shape change that noise alone exceeds only ~5% of the time. A bootstrap CI puts
uncertainty on it. See ``Noise_Determination_Explainer`` (CDO).

Because Copt sweeps continuously during a real dissolution run, the floor is a
**curve indexed by Copt** (:class:`NoiseFloorCurve`), looked up per frame by its
instantaneous Copt.

Plateau detection is **data-driven** (finds stable Copt segments automatically),
so re-collected or extended-Copt titrations need no configuration. W₁-as-noise-floor
is kept here even though W₁-as-dissolution-readout was retired.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Drug-relevant channel window (PAQXOS 15–25; 0-indexed slice [14:25]).
DEFAULT_CH_START, DEFAULT_CH_END = 14, 25


# ── Wasserstein-1 on the normalized cumulative distribution ──────────────────

def compute_w1_cdf(I, indices=None, ch_start=DEFAULT_CH_START, ch_end=DEFAULT_CH_END):
    """Frame-to-frame W₁ on a channel window (normalized cumulative intensity).

    Each frame's intensity over the window is normalized to sum 1 (a distribution
    over scattering angle), converted to cumulative form; W₁ is the L1 norm of the
    difference of successive cumulatives. Returns one value per consecutive pair.
    """
    sub = I[indices, ch_start:ch_end] if indices is not None else I[:, ch_start:ch_end]
    if sub.shape[0] < 2:
        return np.array([])
    sums = sub.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    F = np.cumsum(sub / sums, axis=1)
    return np.array([np.sum(np.abs(F[i + 1] - F[i])) for i in range(F.shape[0] - 1)])


# ── Data-driven plateau detection ────────────────────────────────────────────

@dataclass
class Plateau:
    indices: np.ndarray
    n_frames: int
    copt_mean: float
    copt_std: float
    t_start_min: float
    t_end_min: float


def detect_plateaus(copt, t_min, tolerance=0.6, min_stable_frames=20,
                    skip_equilibration_frames=5, window=9):
    """Segment a continuous Copt(t) trace into stable plateaus (no preset levels).

    A frame is "stable" if the local rolling std of Copt (over ``window`` frames)
    is below ``tolerance`` (%). Maximal contiguous stable runs of at least
    ``min_stable_frames`` become plateaus, after trimming the leading
    post-aliquot equilibration. Works for any set of Copt levels — so extended /
    re-collected titrations need no reconfiguration.
    """
    copt = np.asarray(copt, dtype=float)
    t_min = np.asarray(t_min, dtype=float)
    n = copt.size
    if n < min_stable_frames:
        return []
    half = window // 2
    stable = np.array([copt[max(0, i - half):min(n, i + half + 1)].std() < tolerance
                       for i in range(n)])

    plateaus, i = [], 0
    while i < n:
        if not stable[i]:
            i += 1
            continue
        j = i
        while j < n and stable[j]:
            j += 1
        seg = np.arange(i, j)
        if seg.size >= min_stable_frames:
            trimmed = seg[skip_equilibration_frames:] if seg.size > skip_equilibration_frames + 10 else seg
            plateaus.append(Plateau(
                indices=trimmed, n_frames=int(trimmed.size),
                copt_mean=float(copt[trimmed].mean()), copt_std=float(copt[trimmed].std()),
                t_start_min=float(t_min[trimmed[0]]), t_end_min=float(t_min[trimmed[-1]]),
            ))
        i = j
    return plateaus


# ── Bootstrap CI on the 95th percentile ──────────────────────────────────────

def bootstrap_p95_ci(w1_values, n_boot=2000, ci_level=0.95, seed=42):
    """Bootstrap CI on the 95th percentile of W₁ values → (point, lo, hi)."""
    w1 = np.asarray(w1_values, dtype=float)
    if w1.size < 5:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    resamples = rng.choice(w1, size=(n_boot, w1.size), replace=True)
    dist = np.percentile(resamples, 95, axis=1)
    alpha = (1 - ci_level) / 2
    return (float(np.percentile(w1, 95)),
            float(np.percentile(dist, 100 * alpha)),
            float(np.percentile(dist, 100 * (1 - alpha))))


# ── Noise-floor curve ────────────────────────────────────────────────────────

@dataclass
class NoiseFloorPoint:
    copt: float                # plateau mean Copt (%)
    w1_p95: float              # 95th-percentile W₁ (the floor at this Copt)
    ci_lo: float
    ci_hi: float
    n_pairs: int
    source: str = ""           # which run this came from


@dataclass
class NoiseFloorCurve:
    """Noise floor as a function of Copt; ``floor_at`` interpolates."""
    points: list[NoiseFloorPoint]
    channel_window: tuple[int, int]
    meta: dict

    def floor_at(self, copt) -> float:
        """Interpolated 95th-percentile W₁ noise floor at a given Copt (clamped at ends)."""
        if not self.points:
            return float("nan")
        xs = np.array([p.copt for p in self.points])
        ys = np.array([p.w1_p95 for p in self.points])
        ux, inv = np.unique(np.round(xs, 2), return_inverse=True)
        uy = np.array([ys[inv == k].mean() for k in range(ux.size)])
        return float(np.interp(copt, ux, uy))

    def is_significant(self, observed_w1, copt) -> bool:
        """Is an observed frame-to-frame W₁ above the noise floor at this Copt?"""
        return float(observed_w1) > self.floor_at(copt)

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame([p.__dict__ for p in self.points]).sort_values("copt")

    def save(self, path):
        from pathlib import Path
        import json
        path = Path(path)
        self.to_frame().to_csv(path, index=False)
        path.with_name(path.stem + "_meta.json").write_text(json.dumps(
            {"channel_window": list(self.channel_window), **self.meta}, indent=2))
        return path


def characterize_noise_floor(runs, ch_start=DEFAULT_CH_START, ch_end=DEFAULT_CH_END,
                             n_boot=2000, seed=42, **plateau_kw) -> NoiseFloorCurve:
    """Build a :class:`NoiseFloorCurve` from one or more glass-bead titration runs.

    ``runs`` is a list of ``(label, FrameArrays-like)`` where each item exposes
    ``.I``, ``.copt``, ``.t_min`` (e.g. an ingest ``RawRun``). Fouled runs should
    be excluded before calling (see :func:`ref_level` / the CLI).
    """
    points: list[NoiseFloorPoint] = []
    for label, run in runs:
        for pl in detect_plateaus(run.copt, run.t_min, **plateau_kw):
            w1 = compute_w1_cdf(run.I, pl.indices, ch_start, ch_end)
            if w1.size < 5:
                continue
            p95, lo, hi = bootstrap_p95_ci(w1, n_boot=n_boot, seed=seed)
            points.append(NoiseFloorPoint(pl.copt_mean, p95, lo, hi, int(w1.size), label))
    return NoiseFloorCurve(points=points, channel_window=(ch_start, ch_end),
                           meta={"n_sources": len({p.source for p in points}),
                                 "n_points": len(points)})


# ── Optics-fouling guard ─────────────────────────────────────────────────────

def ref_level(run) -> float:
    """Reference-background level (Σ of the static reference spectrum).

    Elevated values flag fouled optics — exclude such runs from the noise floor.
    """
    return float(np.asarray(run.ref).sum())


def flag_fouled(runs, factor=2.5, baseline=None):
    """Return labels whose reference background looks fouled.

    A run is flagged if its ``ref_level`` exceeds ``factor`` × the cleanest run's
    level (or an explicit ``baseline``, e.g. a known clean week). Baselining on the
    minimum is robust when an outlier would otherwise drag the median up.
    """
    levels = {label: ref_level(run) for label, run in runs}
    if not levels:
        return []
    base = baseline if baseline is not None else min(levels.values())
    return [label for label, lv in levels.items() if base > 0 and lv > factor * base]
