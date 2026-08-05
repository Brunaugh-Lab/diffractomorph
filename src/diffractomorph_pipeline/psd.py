"""The Sympatec **q3 size distribution** as it evolves in time — reader, de-normalization, figures.

The size-space companion to :mod:`diffractomorph_pipeline.empirical_fit` (which fits each detector
*channel's* raw intensity). Here we work on the instrument's own inverted **q3** — the volume-weighted
size distribution PAQXOS reports per frame — taken as given (its inversion, not our Mie kernel), so
this is a **data-only** view of the distribution over time.

Two layers, plus the two standard figures:

1. **Reader** — :func:`read_q3` parses the two on-disk PAQXOS layouts into one :class:`Q3Trajectory`
   ``(grid_um, dQ3)``:
     * *folder* (one CSV per frame; ``dQ3`` = ``diff`` of the cumulative ``Q₃``), and
     * *stacked* (all frames in one CSV, ``dQ3`` column given, header rows delimiting frames).
   Each row of ``dQ3`` is a per-frame **relative volume fraction** (sums to 1).

2. **De-normalization** — :func:`to_absolute`. q3 is renormalized to 100 % every frame, so a class can
   *gain share* purely because others drained faster. To see real change over time you must scale each
   frame by an absolute total ``total(t)`` — ``Copt(t)`` (an *area* proxy from the ``.rtf``) or the
   mass/volume-conserving undissolved-solid fraction ``1 − f_dissolved(t)`` from UV-Vis.

Figures:
   * :func:`plot_layered` — the whole distribution over time, coloured by time, as **% of the initial
     total volume** (open system: every class depletes toward 0 as the total falls).
   * :func:`plot_buckets` — a small-multiples grid, one populated size class each, showing **% of its
     own initial** still present (each starts at 100 % and declines) — the per-bucket depletion view.

The parametric analysis of these trajectories — distribution shape-change / loss-of-signal, and the
per-bucket rate & depth of decline — is intentionally **not** here yet; it's the next design step.

**Reliability ceiling.** The instrument's inversion tail above ~15 µm
(``optics.mie.VALID_SIZE_MAX_UM``) is ill-constrained; the reader keeps every class, and the figures
take an optional ``size_max`` to confine the view to the trustworthy range.

    from diffractomorph_pipeline import ingest, psd
    traj = psd.read_q3(q3_path)                          # folder or stacked, auto-detected
    run = ingest.extract_run(rtf)                        # paired .rtf for Copt(t) + time
    psd.plot_layered(traj, run.copt, "layered.png", title=rid, t_min=run.t_min)
    psd.plot_buckets(traj, run.copt, "buckets.png", title=rid, t_min=run.t_min)
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from diffractomorph_pipeline.optics.mie import VALID_SIZE_MAX_UM

FRAME_S = 12.0          # default LD frame interval (s); the PAQXOS export cadence


# ── reader ────────────────────────────────────────────────────────────────────
@dataclass
class Q3Trajectory:
    """One run's inverted q3 over time.

    ``grid_um`` (B,): ascending size-class representatives (the ``xm`` column).
    ``dQ3`` (T×B): per-frame **relative** volume fraction — each row sums to ~1 (the PAQXOS per-frame
    renormalization; :func:`to_absolute` undoes it). ``layout`` is ``"folder"`` or ``"stacked"``.
    """
    grid_um: np.ndarray
    dQ3: np.ndarray
    layout: str
    source: str

    @property
    def n_frames(self) -> int:
        return int(self.dQ3.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.dQ3.shape[1])

    def times(self, frame_s: float = FRAME_S) -> np.ndarray:
        """Standalone minute grid ``arange(T)·frame_s/60`` when no paired ``.rtf`` time is used."""
        return np.arange(self.n_frames) * frame_s / 60.0

    def to_absolute(self, total) -> np.ndarray:
        """De-normalize by a per-frame total; see :func:`to_absolute`."""
        return to_absolute(self.dQ3, total)


def _frames_to_grid(frames):
    """``frames`` = list of (xm, dQ3%) arrays (possibly ragged) → ``(grid_um, M[T×B])`` in **%**.
    Bins are matched on ``xm`` rounded to 0.01 µm so ragged frames share one grid."""
    keys = sorted({round(float(x), 2) for xm, _ in frames for x in xm})
    idx = {k: j for j, k in enumerate(keys)}
    M = np.zeros((len(frames), len(keys)))
    for t, (xm, dq) in enumerate(frames):
        for x, v in zip(xm, dq):
            M[t, idx[round(float(x), 2)]] += float(v)
    return np.array(keys), M


def _load_q3_folder(folder):
    """folder layout: one CSV per frame, 2 metadata rows above the ``xo,Q₃,xm,…`` header. Per-class
    volume = ``diff`` of the cumulative ``Q₃`` (col 1); size = ``xm`` (col 2)."""
    frames = []
    for f in sorted(glob.glob(str(Path(folder) / "*.csv"))):
        df = pd.read_csv(f, skiprows=2, encoding="utf-8-sig")
        Q3 = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
        xm = pd.to_numeric(df.iloc[:, 2], errors="coerce").values
        dq = np.diff(Q3, prepend=0.0)
        ok = np.isfinite(xm) & np.isfinite(dq)
        frames.append((xm[ok], dq[ok]))
    return _frames_to_grid(frames)


def _load_q3_stacked(csv):
    """stacked layout: all frames in one CSV (``xo,Q3,R3,dQ3,xm,q3*``), each frame a contiguous
    numeric block preceded by its header row (non-numeric ``xo`` → block boundary). ``dQ3`` is col 3,
    size ``xm`` col 4."""
    df = pd.read_csv(csv, header=None, dtype=str, encoding="utf-8-sig")
    xo = pd.to_numeric(df.iloc[:, 0], errors="coerce").values     # header rows → NaN (boundaries)
    dq = pd.to_numeric(df.iloc[:, 3], errors="coerce").values
    xm = pd.to_numeric(df.iloc[:, 4], errors="coerce").values
    frames, i, n = [], 0, len(xo)
    while i < n:
        if not np.isfinite(xo[i]):
            i += 1
            continue
        j = i
        while j < n and np.isfinite(xo[j]):
            j += 1
        fx, fd = xm[i:j], dq[i:j]
        ok = np.isfinite(fx) & np.isfinite(fd)
        if ok.sum() >= 5:
            frames.append((fx[ok], fd[ok]))
        i = j
    return _frames_to_grid(frames)


def read_q3(path, *, layout: str = "auto") -> Q3Trajectory:
    """Read a run's q3 export into a :class:`Q3Trajectory` (``dQ3`` as per-frame fraction, rows→1).

    ``layout="auto"`` picks *folder* when ``path`` is a directory (one CSV per frame) and *stacked*
    when it is a single ``.csv`` (all frames stacked). Pass ``"folder"``/``"stacked"`` to force it.
    """
    p = Path(path)
    if layout == "auto":
        layout = "folder" if p.is_dir() else "stacked"
    if layout == "folder":
        grid, M = _load_q3_folder(p)
    elif layout == "stacked":
        grid, M = _load_q3_stacked(p)
    else:
        raise ValueError(f"layout must be 'auto', 'folder', or 'stacked' (got {layout!r})")
    row = M.sum(axis=1, keepdims=True)
    dQ3 = np.divide(M, row, out=np.zeros_like(M), where=row > 0)   # % → per-frame fraction
    return Q3Trajectory(grid_um=grid, dQ3=dQ3, layout=layout, source=str(p))


# ── timestamped cumulative reader + size-percentile primitives ─────────────────
# The distribution-shape drivers (matched-extent) need three things the shape-SVD reader above does
# not expose: (a) each frame's absolute acquisition time (to timestamp-match q3 frames to the paired
# detector frames rather than index-align them), and (b) the **cumulative** ``Q₃(xo)`` at the class
# *boundaries* — the quantity percentiles must be read off directly. Reconstructing per-``xm`` bin
# fractions and interpolating on the class *means* systematically shifts D10/D50/D90; the vendor already
# reports the cumulative, so read it as given.

_Q3_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")


def _parse_q3_time(s: str) -> datetime:
    """Parse the PAQXOS ``Time`` metadata cell (``2026-06-08 10:43:55.4613``) to a naive-local
    ``datetime`` — the same clock as the paired ``.rtf`` ``t0`` (:mod:`ingest`), so the two match
    directly. Trailing timezone offsets, if present, are dropped (both sides are the one local clock)."""
    s = str(s).strip().split(" -")[0].split(" +")[0]        # drop a trailing " -04:00"-style offset
    for fmt in _Q3_TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(s)                         # last resort (handles ISO 'T' + offset)


@dataclass
class Q3Frames:
    """One run's q3 export with the pieces size-percentile / timestamp work needs (folder layout).

    ``t_epoch`` (T,): per-frame absolute acquisition time as POSIX seconds (from the ``Time`` cell).
    ``xo`` (B,): ascending class **lower-boundary** grid (µm) — the cumulative's support.
    ``Q3_cum`` (T×B): cumulative volume % undersize at each ``xo`` (monotone 0→100 along size).
    ``xm`` (B,): class-mean grid (µm), kept for reference. ``dQ3`` (T×B): per-class volume **fraction**
    (rows sum ~1) = ``diff(Q3_cum)/100`` — the shape vector for distances."""
    t_epoch: np.ndarray
    xo: np.ndarray
    Q3_cum: np.ndarray
    xm: np.ndarray
    dQ3: np.ndarray
    source: str

    @property
    def n_frames(self) -> int:
        return int(self.Q3_cum.shape[0])


def read_q3_frames(folder) -> Q3Frames:
    """Read a folder-layout run (one CSV per frame) preserving per-frame timestamp + cumulative ``Q₃``.

    Each CSV carries a two-row metadata block (the second row's ``Time`` cell is the acquisition time),
    then an ``xo,Q₃,xm,…`` table. Frames share the instrument's fixed size grid; frames are ordered by
    timestamp. Use :func:`read_q3` instead when you only need the relative shape (SVD/figures)."""
    files = sorted(glob.glob(str(Path(folder) / "*.csv")))
    if not files:
        raise FileNotFoundError(f"no q3 CSV frames in {folder}")
    times, rows = [], []
    grid_keys = None
    for f in files:
        meta = pd.read_csv(f, nrows=1, encoding="utf-8-sig")
        times.append(_parse_q3_time(meta["Time"].iloc[0]))
        df = pd.read_csv(f, skiprows=2, encoding="utf-8-sig")
        xo = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
        Q3 = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
        xm = pd.to_numeric(df.iloc[:, 2], errors="coerce").values
        ok = np.isfinite(xo) & np.isfinite(Q3)
        rows.append((xo[ok], Q3[ok], xm[ok]))
        grid_keys = sorted({round(float(x), 4) for x in xo[ok]}) if grid_keys is None \
            else grid_keys
    keys = np.array(sorted({round(float(x), 4) for xo, _, _ in rows for x in xo}))
    idx = {k: j for j, k in enumerate(keys)}
    Q3_cum = np.full((len(rows), keys.size), np.nan)
    xm_grid = np.full(keys.size, np.nan)
    for t, (xo, Q3, xm) in enumerate(rows):
        for x, q, m in zip(xo, Q3, xm):
            j = idx[round(float(x), 4)]
            Q3_cum[t, j] = q
            if np.isfinite(m):
                xm_grid[j] = m
    Q3_cum = _ffill_cumulative(Q3_cum)                      # a frame missing a boundary → carry cumulative
    dQ3 = np.diff(Q3_cum, axis=1, prepend=0.0) / 100.0
    order = np.argsort([t.timestamp() for t in times])
    t_epoch = np.array([times[i].timestamp() for i in order], dtype=float)
    return Q3Frames(t_epoch=t_epoch, xo=keys, Q3_cum=Q3_cum[order], xm=xm_grid,
                    dQ3=dQ3[order], source=str(folder))


def _ffill_cumulative(Q):
    """Forward-fill NaNs along the size axis (cumulative is monotone, so a missing boundary carries the
    prior cumulative), leading NaNs → 0."""
    Q = np.asarray(Q, float).copy()
    for r in range(Q.shape[0]):
        last = 0.0
        for c in range(Q.shape[1]):
            if np.isfinite(Q[r, c]):
                last = Q[r, c]
            else:
                Q[r, c] = last
    return Q


def q3_percentiles(xo, Q3_cum_row, pcts=(10.0, 50.0, 90.0)):
    """Volume-weighted percentile diameters read directly off the cumulative ``Q₃(xo)`` (in %), by
    linear interpolation in log10-diameter. Returns NaN for any percentile the cumulative does not
    bracket (e.g. D10 when the finest boundary already sits above 10 %) — never a clamped edge value."""
    x = np.asarray(xo, float)
    c = np.asarray(Q3_cum_row, float)
    m = np.isfinite(x) & np.isfinite(c) & (x > 0)
    x, c = x[m], c[m]
    o = np.argsort(x)
    x, c = x[o], c[o]
    lx = np.log10(x)
    out = []
    for p in pcts:
        if x.size < 2 or p < c[0] or p > c[-1]:
            out.append(np.nan)
            continue
        j = int(np.searchsorted(c, p))
        j = min(max(j, 1), c.size - 1)
        c0, c1 = c[j - 1], c[j]
        w = 0.0 if c1 == c0 else (p - c0) / (c1 - c0)
        out.append(float(10.0 ** (lx[j - 1] + w * (lx[j] - lx[j - 1]))))
    return out


def q3_tail_fraction(xo, Q3_cum_row, size_max: float = VALID_SIZE_MAX_UM) -> float:
    """Volume fraction (%) above ``size_max`` = ``100 − Q₃(size_max)`` (cumulative interpolated at
    ``size_max``). Large values flag the ill-constrained coarse inversion tail (``VALID_SIZE_MAX_UM``),
    where D90 and span become inversion artifacts rather than dissolution morphology."""
    x = np.asarray(xo, float)
    c = np.asarray(Q3_cum_row, float)
    m = np.isfinite(x) & np.isfinite(c)
    x, c = x[m], c[m]
    o = np.argsort(x)
    x, c = x[o], c[o]
    if x.size == 0:
        return float("nan")
    if size_max <= x[0]:
        return float(100.0 - c[0])
    if size_max >= x[-1]:
        return float(max(0.0, 100.0 - c[-1]))
    q_at = float(np.interp(np.log10(size_max), np.log10(x), c))
    return float(max(0.0, 100.0 - q_at))


def q3_wasserstein_log(xo, Q3_cum_a, Q3_cum_b) -> float:
    """1-D Wasserstein-1 distance between two q3 distributions in **log₁₀-diameter** space, computed
    directly from their cumulative ``Q₃`` (%) on a shared class-boundary grid ``xo``.

    For 1-D distributions the earth-mover distance equals ``∫ |F_a − F_b| dx`` of the CDFs; here the
    support is ``x = log₁₀(diameter)`` and ``F = Q₃/100`` (a fraction in ``[0, 1]``). Units are
    log₁₀-µm (decades). Both cumulatives must be sampled on the same ``xo`` grid (rebin/interpolate the
    cumulative onto a common grid first if they differ). NaN class boundaries are dropped pairwise.
    """
    x = np.log10(np.asarray(xo, float))
    Fa = np.asarray(Q3_cum_a, float) / 100.0
    Fb = np.asarray(Q3_cum_b, float) / 100.0
    m = np.isfinite(x) & np.isfinite(Fa) & np.isfinite(Fb)
    if m.sum() < 2:
        return float("nan")
    x, d = x[m], np.abs(Fa[m] - Fb[m])
    o = np.argsort(x)
    x, d = x[o], d[o]
    return float(np.sum(0.5 * (d[:-1] + d[1:]) * np.diff(x)))      # ∫|F_a−F_b| dlog₁₀d (trapezoid)


def restrict_cumulative(xo, Q3_cum_row, size_max: float = VALID_SIZE_MAX_UM):
    """Truncate the distribution to ``≤ size_max`` and renormalize its cumulative to 100 % — the
    prespecified restricted-range view for the coarse-tail sensitivity check. Returns ``(xo_r,
    Q3_cum_r)`` (both include an interpolated endpoint exactly at ``size_max``)."""
    x = np.asarray(xo, float)
    c = np.asarray(Q3_cum_row, float)
    m = np.isfinite(x) & np.isfinite(c)
    x, c = x[m], c[m]
    o = np.argsort(x)
    x, c = x[o], c[o]
    keep = x <= size_max
    xr = x[keep]
    cr = c[keep]
    if xr.size and xr[-1] < size_max < x[-1]:               # add the exact size_max endpoint
        q_end = float(np.interp(np.log10(size_max), np.log10(x), c))
        xr = np.append(xr, size_max)
        cr = np.append(cr, q_end)
    top = cr[-1] if cr.size else 0.0
    cr = 100.0 * cr / top if top > 0 else cr
    return xr, cr


def match_frames_by_time(t_q3, t_det, tol_s: float = 3.0):
    """Nearest-time frame match between two timestamp sequences (POSIX seconds), each within ``tol_s``.

    Returns ``(pairs, unmatched_q3, unmatched_det)`` where ``pairs`` is a list of ``(i_q3, j_det, dt_s)``
    (signed ``t_q3 − t_det``). Greedy nearest, each detector frame used once — q3 and detector come from
    the same acquisition at the same cadence, so the map is ~1:1; frames with no partner within
    tolerance are reported, not silently dropped (audit step 1)."""
    tq = np.asarray(t_q3, float)
    td = np.asarray(t_det, float)
    used_det = set()
    pairs = []
    for i in np.argsort(tq):
        if td.size == 0:
            break
        d = np.abs(td - tq[i])
        for j in np.argsort(d):
            if d[j] > tol_s:
                break
            if j not in used_det:
                used_det.add(int(j))
                pairs.append((int(i), int(j), float(tq[i] - td[j])))
                break
    matched_q3 = {p[0] for p in pairs}
    matched_det = {p[1] for p in pairs}
    unmatched_q3 = [int(i) for i in range(tq.size) if i not in matched_q3]
    unmatched_det = [int(j) for j in range(td.size) if j not in matched_det]
    pairs.sort(key=lambda p: p[1])                           # detector (time) order
    return pairs, unmatched_q3, unmatched_det


def interp_cumulative_at_g(g_env, cum_rows, target):
    """Interpolate a cumulative-``Q₃`` row at the first crossing of progress ``g = target``.

    ``g_env`` (n,) is a monotone non-increasing remaining-signal fraction; ``cum_rows`` (n×B) the paired
    per-frame cumulative ``Q₃``. Returns ``(g, cum_at_g, j, frac)`` where ``j`` is the lower bracketing
    frame and ``frac`` the interpolation weight, or ``None`` if ``target`` is never reached."""
    g = np.asarray(g_env, float)
    C = np.asarray(cum_rows, float)
    below = np.where(g <= target)[0]
    if below.size == 0 or below[0] == 0:
        return None
    j = int(below[0])
    g0, g1 = g[j - 1], g[j]
    frac = 0.0 if g0 == g1 else (g0 - target) / (g0 - g1)
    cum_at = C[j - 1] + frac * (C[j] - C[j - 1])
    g_at = float(g0 + frac * (g1 - g0))
    return g_at, cum_at, j, float(frac)


# ── de-normalization ──────────────────────────────────────────────────────────
def to_absolute(dQ3, total) -> np.ndarray:
    """Scale each per-frame-relative q3 row by an absolute total → absolute volume ``(T×B)``.

    ``total`` (T,) is the absolute amount of material each frame — ``Copt(t)`` (area proxy) or
    ``1 − f_dissolved(t)`` (UV, mass/volume-conserving). Truncates to the shorter of the two so a
    q3 trajectory and its paired ``.rtf`` can be aligned by length.
    """
    dQ3 = np.asarray(dQ3, dtype=float)
    total = np.asarray(total, dtype=float)
    n = min(dQ3.shape[0], total.shape[0])
    return dQ3[:n] * total[:n, None]


# ── shape-change SVD ──────────────────────────────────────────────────────────
@dataclass
class ShapeSVD:
    """Rank-1 decomposition of how the q3 **shape** (relative distribution) changes over time.

    ``grid_um`` (B,): the populated size classes used. ``mean_shape`` (B,): the time-mean relative q3.
    ``mode1_size`` (B,) = ``v₁``: the dominant shape-change direction — which sizes gain (+) vs lose (−)
    share together. ``mode1_time`` (T,) = ``σ₁·u₁``: its amplitude over time (oriented to grow with
    time). ``rank1_var`` = ``σ₁²/Σσₖ²`` of the *centered* shape matrix — how one-dimensional the shape
    change is (→1: a single axis; lower: multi-mode rearrangement)."""
    grid_um: np.ndarray
    t_min: np.ndarray
    mean_shape: np.ndarray
    mode1_size: np.ndarray
    mode1_time: np.ndarray
    rank1_var: float
    singular_values: np.ndarray


def shape_svd(traj: Q3Trajectory, t_min=None, *, frame_s: float = FRAME_S) -> ShapeSVD:
    """SVD of the **relative** q3 shape over time, centered on its time-mean shape.

    Works on ``dQ3`` (each frame a shape summing to 1), so it is independent of the magnitude
    (loss-of-mass) envelope — pass an already frame-masked trajectory so dead-inversion frames don't
    pollute it. Centering is essential: an uncentered rows-sum-to-1 matrix is trivially rank-1 (the
    mean shape), so we subtract the time-mean and decompose the *variation*. The q3-size-space analog
    of :func:`~diffractomorph_pipeline.band_routing._rank1_null_pvalue`.
    """
    grid = np.asarray(traj.grid_um, dtype=float)
    S = np.asarray(traj.dQ3, dtype=float)
    t = np.asarray(t_min, dtype=float) if t_min is not None else traj.times(frame_s)
    t = t[:S.shape[0]]
    pop = S.sum(axis=0) > 0
    grid, S = grid[pop], S[:, pop]
    mean_shape = S.mean(axis=0)
    nan_v = np.full(grid.size, np.nan)
    if S.shape[0] < 2 or grid.size < 2:
        return ShapeSVD(grid, t, mean_shape, nan_v, np.full(S.shape[0], np.nan),
                        float("nan"), np.array([]))
    Sc = S - mean_shape
    U, s, Vt = np.linalg.svd(Sc, full_matrices=False)
    ssq = float(np.sum(s ** 2))
    rank1_var = float(s[0] ** 2 / ssq) if ssq > 0 else float("nan")
    v1, u1 = Vt[0].copy(), U[:, 0] * s[0]
    if np.polyfit(np.arange(u1.size), u1, 1)[0] < 0:       # orient so the mode grows with time
        v1, u1 = -v1, -u1
    return ShapeSVD(grid, t, mean_shape, v1, u1, rank1_var, s)


def plot_shape_svd(svd: ShapeSVD, out_png, *, title=""):
    """Two panels: (A) the mean shape + the dominant shape-change direction ``v₁(size)``, and
    (B) its amplitude over time ``u₁(t)``. Returns the output path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))
    axA.plot(svd.grid_um, 100.0 * svd.mean_shape, "-o", ms=3, color="#555555", label="mean shape")
    axA.set_xscale("log")
    axA.set_xlabel("size class xm (µm)")
    axA.set_ylabel("mean q3 (%)")
    ax2 = axA.twinx()
    ax2.plot(svd.grid_um, svd.mode1_size, "-s", ms=3, color="#b2182b",
             label=r"mode-1 Δshape ($v_1$)")
    ax2.axhline(0.0, color="#b2182b", lw=0.5, ls=":")
    ax2.set_ylabel(r"mode-1 shape-change $v_1$  (+ gains share, − loses)", color="#b2182b")
    axA.set_title(f"Shape: mean + dominant change  (rank1_var = {svd.rank1_var:.2f})", fontsize=10)
    axB.plot(svd.t_min, svd.mode1_time, "-", color="#2166ac", lw=1.6)
    axB.axhline(0.0, color="#888888", lw=0.5, ls=":")
    axB.set_xlabel("time (min)")
    axB.set_ylabel(r"mode-1 amplitude $u_1(t)$")
    axB.set_title("Shape-change progression over time", fontsize=10)
    axB.grid(alpha=0.3)
    fig.suptitle(title or "q3 shape-change SVD", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return Path(out_png)


# ── frame masking ─────────────────────────────────────────────────────────────
def frame_mask(copt, *, copt_floor_frac: float = 0.0, drop_frames=None) -> np.ndarray:
    """Per-frame keep-mask (True = trustworthy) for a q3 trajectory — two independent cuts.

    * ``drop_frames`` — explicit frame indices to exclude. Pass the synchronized-glitch / startup
      frames that :func:`~diffractomorph_pipeline.noise_filter.despike_frames` finds on the paired
      ``.rtf``: they are bad in the shared acquisition, so the q3 inversion at those same frames is
      equally suspect.
    * ``copt_floor_frac`` — drop frames whose scattering signal ``copt`` falls below
      ``copt_floor_frac · max(copt)``. Below that the total signal is too weak for the instrument's q3
      inversion to be trustworthy (it dumps the residual into a low size bin — the end-of-run spike).
      ``copt`` is the **scattering** signal regardless of how the trajectory is later de-normalized
      (Copt vs UV) — inversion reliability is a signal property, not a basis choice. ``0`` disables it.

    Non-finite ``copt`` frames are always excluded.
    """
    copt = np.asarray(copt, dtype=float)
    keep = np.isfinite(copt)
    if copt_floor_frac and copt_floor_frac > 0:
        ref = np.nanmax(copt) if np.isfinite(copt).any() else np.nan
        if np.isfinite(ref) and ref > 0:
            keep &= copt >= copt_floor_frac * ref
    if drop_frames is not None:
        idx = np.asarray(list(drop_frames), dtype=int)
        idx = idx[(idx >= 0) & (idx < keep.size)]
        keep[idx] = False
    return keep


def apply_frame_mask(traj: Q3Trajectory, keep) -> Q3Trajectory:
    """Return a new :class:`Q3Trajectory` keeping only the frames where ``keep`` is True (a bool array
    of length ``traj.n_frames``). Grid unchanged; ``dQ3`` subset to the kept rows."""
    keep = np.asarray(keep, dtype=bool)
    return Q3Trajectory(grid_um=traj.grid_um, dQ3=traj.dQ3[keep], layout=traj.layout, source=traj.source)


# ── figures ───────────────────────────────────────────────────────────────────
def _align_time(traj, n, t_min, frame_s):
    t = np.asarray(t_min, dtype=float) if t_min is not None else traj.times(frame_s)
    return t[:n]


def plot_layered(traj: Q3Trajectory, total, out_png, *, title="", t_min=None,
                 size_max=None, frame_s: float = FRAME_S):
    """Layered q3 distribution over time as **% of the initial total volume** (de-normalized by
    ``total`` = Copt or UV). Every frame is a line coloured by time (viridis + colorbar); the total
    falls as material dissolves, so the whole set of classes depletes toward 0 (an *open system*).

    ``size_max`` (µm) clips the size axis to the trustworthy range. Returns the output path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    grid = np.asarray(traj.grid_um, dtype=float)
    abs_q3 = to_absolute(traj.dQ3, total)
    n = abs_q3.shape[0]
    t = _align_time(traj, n, t_min, frame_s)
    init_total = float(abs_q3[0].sum())
    pct = 100.0 * abs_q3 / init_total if init_total > 0 else abs_q3 * np.nan
    keep = grid <= size_max if size_max is not None else np.ones(grid.size, bool)

    norm = Normalize(float(t.min()), float(t.max()) if t.max() > t.min() else float(t.min()) + 1.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(n):
        ax.plot(grid[keep], pct[k][keep], "-", lw=0.8, alpha=0.7, color=plt.cm.viridis(norm(t[k])))
    ax.set_xscale("log")
    ax.set_xlabel("size class xm (µm)")
    ax.set_ylabel("% of INITIAL total volume")
    ax.set_title(title or "q3 size distribution over time (% of initial, ×total)", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
    cb.set_label("time (min)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return Path(out_png)


def plot_buckets(traj: Q3Trajectory, total, out_png, *, title="", t_min=None,
                 size_max=None, ncol=4, frame_s: float = FRAME_S):
    """Per-size-bucket depletion: for each populated class, **% of its own initial** volume still
    present vs time — a small-multiples grid (each panel starts at 100 % and declines). De-normalized
    by ``total`` so a flat-share class still falls as the total drops.

    Only classes populated at ``t=0`` (``q₀ > max·1e-4``) are drawn; ``size_max`` (µm) further caps
    the grid to the trustworthy range. Returns the output path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.asarray(traj.grid_um, dtype=float)
    abs_q3 = to_absolute(traj.dQ3, total)
    n = abs_q3.shape[0]
    t = _align_time(traj, n, t_min, frame_s)
    q0 = abs_q3[0]
    populated = q0 > (q0.max() * 1e-4 if q0.max() > 0 else np.inf)
    within = grid <= size_max if size_max is not None else np.ones(grid.size, bool)
    live = np.where(populated & within)[0]
    if live.size == 0:
        raise ValueError("no populated size classes to plot")

    nrow = int(np.ceil(len(live) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.1 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for k, b in enumerate(live):
        ax = axes[k]
        ax.plot(t, 100.0 * abs_q3[:, b] / q0[b], color="#2166ac", lw=1.5)
        ax.axhline(50, ls=":", color="gray", lw=0.6)
        ax.set_title(f"{grid[b]:.2f} µm", fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
    for k in range(len(live), len(axes)):
        axes[k].axis("off")
    for k in range(len(axes)):
        if k >= len(axes) - ncol:
            axes[k].set_xlabel("time (min)", fontsize=8)
        if k % ncol == 0:
            axes[k].set_ylabel("% remaining", fontsize=8)
    fig.suptitle(title or "per-size-bucket q3 depletion", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return Path(out_png)


def plot_total_mass(t_min, series, out_png, *, title="",
                    ylabel="total signal mass (% of initial)", normalize=True):
    """Overlay one or more total-mass-vs-time curves (the loss-of-signal / magnitude view).

    ``series`` is an iterable of dicts ``{"label": str, "y": array, "color": str|None,
    "pts": (t, y)|None}`` — one line per basis (e.g. Copt area vs UV mass); ``pts`` optionally overlays
    the raw measured samples as markers. With ``normalize`` each curve is shown as % of its own first
    finite non-zero value. Returns the output path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(t_min, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for s in series:
        y = np.asarray(s["y"], dtype=float)
        n = min(t.size, y.size)
        yy = y[:n].astype(float)
        ref = 1.0
        if normalize:
            finite = yy[np.isfinite(yy) & (yy != 0)]
            ref = float(finite[0]) if finite.size else float("nan")
            yy = 100.0 * yy / ref
        ax.plot(t[:n], yy, "-", lw=2, color=s.get("color"), label=s["label"])
        if s.get("pts") is not None:
            pt, py = s["pts"]
            py = np.asarray(py, dtype=float)
            ax.plot(pt, (100.0 * py / ref) if normalize else py, "o", ms=5, color=s.get("color"))
    ax.set_xlabel("time (min)")
    ax.set_ylabel(ylabel)
    ax.set_title(title or "total signal mass vs time", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return Path(out_png)


def plot_uv_vs_copt(t_copt, copt, uv_t, uv_remaining, out_png, *, title="", forward=None,
                    copt_label="Copt(t)  — LD area", uv_label="UV remaining solid  — mass"):
    """Compare the *shape* of the LD obscuration ``Copt(t)`` (dense line) against the UV-derived
    remaining undissolved solid (sparse measured points + dashed connector), both as **% of their
    ``t=0`` value**, and optionally the forward model's predicted remaining.

    ``uv_t`` / ``uv_remaining`` must already include the ``t=0`` anchor at the injected amount (so the
    UV curve starts at 100 % of injected, before the assay's first ~2-min sample). Copt is normalized
    to its own first finite non-zero value. ``forward``, if given, is ``(t_min, remaining_pct)`` — the
    predicted remaining undissolved in % (i.e. ``100 − pct_dissolved``), already on the injected basis.
    Returns the output path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_copt = np.asarray(t_copt, dtype=float)
    copt = np.asarray(copt, dtype=float)
    finite = copt[np.isfinite(copt) & (copt != 0)]
    cref = float(finite[0]) if finite.size else float("nan")
    uv_t = np.asarray(uv_t, dtype=float)
    uv_remaining = np.asarray(uv_remaining, dtype=float)
    uref = float(uv_remaining[0]) if uv_remaining.size else float("nan")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(t_copt, 100.0 * copt / cref, "-", lw=2, color="#2166ac", label=copt_label)
    if forward is not None:
        ft, fpct = np.asarray(forward[0], dtype=float), np.asarray(forward[1], dtype=float)
        ax.plot(ft, fpct, "-", lw=1.5, color="#1b7837", alpha=0.85, label="forward model (N–B)")
    ax.plot(uv_t, 100.0 * uv_remaining / uref, "--o", lw=1.2, ms=6, color="#d95f02", label=uv_label)
    ax.plot([0], [100.0], "D", ms=8, color="#d95f02", zorder=5, label="injected (T0 anchor)")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("remaining undissolved (% of injected)")
    ax.set_title(title or "UV remaining solid vs Copt — shape comparison", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return Path(out_png)


def plot_mass_and_copt(uv_t, uv_remaining, forward, t_copt, copt, out_png, *, title="",
                       signal_ylabel="Copt — optical concentration (%)",
                       signal_title="B — LD area signal (raw Copt)"):
    """Two-panel per-run figure. **(A) mass basis** — the UV remaining undissolved solid (as % of
    injected: ``uv_t``/``uv_remaining`` include the ``t=0`` anchor at the injected amount) overlaid on
    the forward N–B prediction ``forward=(t_min, remaining_pct)``; these are directly comparable, both
    mass. **(B) an LD signal** — ``(t_copt, copt)`` on its own axis (a different measurand); relabel via
    ``signal_ylabel`` / ``signal_title`` (defaults are raw Copt; e.g. the total angular signal ΣI).
    Shared time axis. Returns the output path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uv_t = np.asarray(uv_t, dtype=float)
    uv_remaining = np.asarray(uv_remaining, dtype=float)
    uref = float(uv_remaining[0]) if uv_remaining.size else float("nan")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    if forward is not None:
        axA.plot(np.asarray(forward[0], dtype=float), np.asarray(forward[1], dtype=float),
                 "-", lw=1.6, color="#1b7837", alpha=0.9, label="forward model (N–B)")
    axA.plot(uv_t, 100.0 * uv_remaining / uref, "--o", lw=1.3, ms=6, color="#d95f02",
             label="UV remaining solid")
    axA.plot([0], [100.0], "D", ms=8, color="#d95f02", zorder=5, label="injected (T0 anchor)")
    axA.set_xlabel("time (min)")
    axA.set_ylabel("remaining undissolved (% of injected)")
    axA.set_title("A — mass basis: UV vs forward", fontsize=10)
    axA.grid(alpha=0.3)
    axA.legend(fontsize=8)

    axB.plot(np.asarray(t_copt, dtype=float), np.asarray(copt, dtype=float), "-", lw=1.8, color="#2166ac")
    axB.set_xlabel("time (min)")
    axB.set_ylabel(signal_ylabel)
    axB.set_title(signal_title, fontsize=10)
    axB.grid(alpha=0.3)

    fig.suptitle(title or "mass (UV vs forward)  |  raw Copt", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return Path(out_png)


def plot_uv_vs_copt_mean(t_copt, copt_mean, copt_sd, uv_t, uv_mean, uv_sd, out_png, *,
                         title="", forward=None, n_reps=None,
                         copt_label="Copt(t)  — LD area (mean±SD)",
                         uv_label="UV remaining solid  — mass (mean±SD)"):
    """Replicate-averaged :func:`plot_uv_vs_copt`. All inputs are **already in % of injected/T0**:
    Copt is a mean line with a ±SD band; UV is mean points with SD error bars + dashed connector
    (``uv_*`` must include the ``t=0`` anchor row, mean 100, sd 0). ``forward`` = ``(t_min,
    remaining_pct)``; ``n_reps`` annotates the title. Returns the output path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_copt = np.asarray(t_copt, dtype=float)
    cm, csd = np.asarray(copt_mean, dtype=float), np.asarray(copt_sd, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(t_copt, cm, "-", lw=2, color="#2166ac", label=copt_label)
    ax.fill_between(t_copt, cm - csd, cm + csd, color="#2166ac", alpha=0.18, lw=0)
    if forward is not None:
        ax.plot(np.asarray(forward[0], dtype=float), np.asarray(forward[1], dtype=float),
                "-", lw=1.5, color="#1b7837", alpha=0.85, label="forward model (N–B)")
    ax.errorbar(np.asarray(uv_t, dtype=float), np.asarray(uv_mean, dtype=float),
                yerr=np.asarray(uv_sd, dtype=float), fmt="--o", lw=1.2, ms=6, color="#d95f02",
                capsize=3, label=uv_label)
    ax.plot([0], [100.0], "D", ms=8, color="#d95f02", zorder=5, label="injected (T0 anchor)")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("remaining undissolved (% of injected)")
    ttl = title or "UV remaining solid vs Copt — replicate mean"
    if n_reps:
        ttl += f"  (n={n_reps})"
    ax.set_title(ttl, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return Path(out_png)
