"""Noise filter — per-channel admission via the CFZ-pH-7 noise surface.

Decide **which channels carry real directional change** (above each channel's own noise
floor) and hand that admitted set to the overlay / downstream views. The output is the
per-channel filter: ``active_channels`` / ``masked_channels`` + the per-channel ``drift_*``
diagnostics; the ``verdict`` is left ``"unrouted"``.

Steps (§4.1–4.2): mask noise-floor channels first (dividing noise by noise in the normalized
trajectory manufactures spurious anti-correlation — the 2026-06-09 splinter bug), despike
synchronized frame artifacts, re-zero any acquisition gap, then admit channels whose directional
trend exceeds the CFZ-pH-7 noise surface (or the legacy static mask).

Classifying the admitted channels into single-mode vs. multi-band is a **separate** step:
:func:`diffractomorph_pipeline.band_routing.route_channels` (only the kinetics extractor uses it).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ── Output contract (§5) ─────────────────────────────────────────────────────

@dataclass
class ChannelTriage:
    """Channel-analysis result. :func:`noise_filter` fills the admission half (active/masked
    channels + drift + cleaned arrays, ``verdict="unrouted"``);
    :func:`diffractomorph_pipeline.band_routing.route_channels` fills the routing half
    (``verdict`` / ``bands`` / ``r_min`` / null test)."""
    verdict: str                     # "single_mode" | "multi_band"
    r_min: float                     # minimum off-diagonal correlation (the gate value)
    active_channels: list[int]
    masked_channels: list[tuple[int, str]]   # (channel, reason): "ch1"|"noise_floor"|"sub_floor"|"spike"
    k: int                           # band count (1 if single-mode)
    silhouette: float | None
    bands: list                      # band_routing.Band list (empty until routed)
    gap_rezeroed: bool
    gap_min: float                   # size of the removed gap (0.0 if none)
    correlation_matrix: np.ndarray   # active-channel C_ij (for the heatmap)
    params: dict                     # thresholds actually used (provenance)
    flags: list[str] = field(default_factory=list)  # "low_signal" | "low_confidence"
    # Per-run null test (§4.4). p_real = P(noise r_min ≤ observed); None when the
    # null was not run (too few active channels / too few frames).
    p_real: float | None = None
    r_min_floor: float = -0.4        # effect-size floor the verdict actually used
    alpha: float = 0.05              # significance level the verdict actually used
    s2_over_s1: float = float("nan")  # σ₂/σ₁ of the active-trajectory SVD
    rank1_var: float = float("nan")   # σ₁²/Σσₖ² (variance explained by one population)
    # Super-floor (noise-surface) per-channel directional-drift diagnostics, aligned
    # to the full input ``channels`` order. None when the static mask was used.
    drift_channels: np.ndarray | None = None   # the channel labels these align to
    drift_zmax: np.ndarray | None = None        # per-channel max windowed-trend |slope|/SE
    drift_sign: np.ndarray | None = None        # slope sign (−1 dissolution, +1 growth)
    drift_z_thresh: float | None = None         # admission threshold used
    # Despiked arrays the admission ran on (synchronized frame artifacts removed,
    # leading startup dropped) + provenance — figures plot these so outputs match.
    clean_I: np.ndarray | None = None
    clean_t: np.ndarray | None = None
    n_lead_dropped: int = 0            # leading startup frames dropped
    spike_frames: list = field(default_factory=list)  # frame indices flagged as artifacts


# ── Internals ────────────────────────────────────────────────────────────────

def _rezero_gap(t_min: np.ndarray, I: np.ndarray, gap_threshold_min: float):
    """Drop pre-gap frames and re-zero time at the restart (§4.2).

    Detects the largest inter-frame time jump; if it exceeds the threshold, keeps
    only the frames after it and re-zeros ``t_min``. Returns
    ``(t_min, I, rezeroed, gap_min)``.
    """
    if t_min.shape[0] < 2:
        return t_min, I, False, 0.0
    dt = np.diff(t_min)
    j = int(np.argmax(dt))
    gap = float(dt[j])
    if gap <= gap_threshold_min:
        return t_min, I, False, 0.0
    keep = slice(j + 1, None)
    t2 = t_min[keep] - t_min[j + 1]
    return t2, I[keep], True, gap


def _mask_spikes(I: np.ndarray) -> np.ndarray:
    """Replace single-frame optical hits with the local median, per channel (§4.2).

    Conservative: only points that deviate from the 3-frame median by more than
    5× the channel's median-absolute-deviation are replaced. No-ops for short runs.
    """
    if I.shape[0] < 5:
        return I
    out = I.copy()
    for c in range(I.shape[1]):
        x = I[:, c]
        med = np.median(np.stack([np.r_[x[0], x[:-1]], x, np.r_[x[1:], x[-1]]]), axis=0)
        resid = np.abs(x - med)
        mad = np.median(np.abs(x - np.median(x))) or 1.0
        spikes = resid > 5.0 * mad
        out[spikes, c] = med[spikes]
    return out


def _robust_z(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Median/MAD z-score (1.4826·MAD ≈ σ for a Gaussian).

    ``mask`` selects the frames used for the median/MAD baseline (e.g. excluding
    already-detected spikes), but the z-score is returned for every frame.
    """
    base = x if mask is None else x[mask]
    med = np.median(base)
    mad = np.median(np.abs(base - med)) or 1e-6
    return (x - med) / (1.4826 * mad)


def despike_frames(I, t_min, copt, z_int: float = 4.0, z_copt_corrob: float = 1.5,
                   w: int = 2):
    """Remove *synchronized* frame artifacts: whole-spectrum + Copt jumps together.

    Per-channel ``_mask_spikes`` misses these — a glitch where every channel jumps
    one frame is dwarfed by each dissolving channel's own dynamic range. Instead
    detect at the frame level: the synchronized factor ``rI[k] = median_c(I[k,c] /
    neighbor-median_c)`` (≈1 normally) spikes on a bad frame. To avoid touching
    real dissolution features, require **Copt to corroborate** — move the same
    direction (a true optical glitch dims/brightens the whole detector, nudging
    Copt; real size change does not). Intensity is the primary detector
    (``|z| > z_int``); Copt only needs to agree in sign at a low bar
    (``|z| > z_copt_corrob``).

    Leading flagged frames (a contiguous run from frame 0 — the obscuration/laser
    startup transient) are **dropped** and ``t`` re-zeroed; interior flagged frames
    are replaced by per-channel linear interpolation from good neighbors.

    Returns ``(I, t_min, copt, info)`` with ``info`` =
    ``{n_lead_dropped, interior_fixed, spike_frames}``.
    """
    I = np.asarray(I, dtype=float)
    t_min = np.asarray(t_min, dtype=float)
    copt = np.asarray(copt, dtype=float)
    n = I.shape[0]
    info = {"n_lead_dropped": 0, "interior_fixed": [], "spike_frames": []}
    if n < 5:
        return I, t_min, copt, info

    # Copt may carry NaN frames (flagged at ingest); interpolate for detection only.
    coptf = copt.copy()
    if np.isnan(coptf).any():
        good = ~np.isnan(coptf)
        if good.sum() < 2:
            return I, t_min, copt, info
        coptf[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), coptf[good])

    def nb(k, excl):
        idx = [j for j in range(max(0, k - w), min(n, k + w + 1)) if j != k and j not in excl]
        if len(idx) >= 2:
            return idx
        return [j for j in range(max(0, k - w), min(n, k + w + 1)) if j != k]

    nbc = np.stack([np.array([np.median(I[nb(k, set()), c]) for c in range(I.shape[1])])
                    for k in range(n)])
    rI = np.array([np.median(I[k] / np.maximum(nbc[k], 1e-9)) for k in range(n)])
    rC = coptf / np.maximum(np.array([np.median(coptf[nb(k, set())]) for k in range(n)]), 1e-9)
    # Intensity is the primary detector; Copt only has to corroborate (same-sign jump
    # at a low bar) so real dissolution features — no optical glitch — are never flagged.
    zI, zC = _robust_z(rI), _robust_z(rC)
    flag = (np.abs(zI) > z_int) & (np.sign(zI) == np.sign(zC)) & (np.abs(zC) > z_copt_corrob)
    flagged = list(np.flatnonzero(flag))
    if not flagged:
        return I, t_min, copt, info
    info["spike_frames"] = [int(k) for k in flagged]

    lead = []
    k = 0
    while k in flagged:
        lead.append(k)
        k += 1
    interior = [f for f in flagged if f not in lead]

    I2, copt2 = I.copy(), copt.copy()
    if interior:
        keep = np.array([j for j in range(n) if j not in flagged])
        ti = t_min[interior]
        for c in range(I.shape[1]):
            I2[interior, c] = np.interp(ti, t_min[keep], I[keep, c])
        copt2[interior] = np.interp(ti, t_min[keep], copt2[keep])
        info["interior_fixed"] = [int(f) for f in interior]

    if lead:
        sl = slice(len(lead), None)
        I2, copt2 = I2[sl], copt2[sl]
        t2 = t_min[sl] - t_min[len(lead)]
        info["n_lead_dropped"] = len(lead)
    else:
        t2 = t_min
    return I2, t2, copt2, info


def _initial_intensity(I: np.ndarray, init_window_frames: int) -> np.ndarray:
    """Per-channel start intensity = mean of the first ``init_window_frames``."""
    n = min(init_window_frames, I.shape[0])
    return I[:n].mean(axis=0)


def _active_mask(
    I0: np.ndarray,
    channels: list[int],
    abs_floor_counts: float,
    rel_floor_frac: float,
    drop_ch1: bool,
    noise_floor: float | None,
):
    """Compute the active-channel mask and the list of (channel, reason) exclusions (§4.1)."""
    peak = float(I0.max()) if I0.size else 0.0
    if noise_floor is not None:
        floor = float(noise_floor)          # external floor preferred over heuristic
    else:
        floor = max(abs_floor_counts, rel_floor_frac * peak)

    active, masked = [], []
    for idx, ch in enumerate(channels):
        if drop_ch1 and ch == 1:
            masked.append((ch, "ch1"))
            continue
        if I0[idx] >= floor:
            active.append(idx)
        else:
            masked.append((ch, "noise_floor"))
    return active, masked, floor


# ── Public API (§9) ──────────────────────────────────────────────────────────

def noise_filter(
    I: np.ndarray,
    t_min: np.ndarray,
    channels: list[int],
    noise_floor: float | None = None,
    abs_floor_counts: float = 1.0,
    rel_floor_frac: float = 0.05,
    drop_ch1: bool = True,
    init_window_frames: int = 3,
    gap_threshold_min: float = 2.0,
    noise_surface=None,
    copt=None,
    z_thresh: float = 4.0,
    despike: bool = True,
) -> ChannelTriage:
    """Apply the per-channel noise filter to a run — channel admission.

    Preprocess (despike, gap re-zero), then keep only the channels that carry real
    *directional* change above their own noise floor (the CFZ-pH-7 noise surface, or the
    legacy static mask). The result's ``active_channels`` / ``masked_channels`` and per-channel
    ``drift_*`` fields are that filter; ``verdict`` is ``"unrouted"``. To classify the admitted
    channels into single-mode vs. multi-band, pass the result to
    :func:`diffractomorph_pipeline.band_routing.route_channels`.

    Channel admission is a whole-run decision; *per-frame* trust (which frames of an admitted
    channel sit above its own absolute noise floor as it fills or empties) is a separate,
    time-resolved question handled at the fit stage via
    :meth:`diffractomorph_pipeline.noise_surface.NoiseSurface.frame_reliable`.

    Parameters
    ----------
    I
        (T × C) per-frame channel intensities.
    t_min
        (T,) elapsed minutes from frame 0.
    channels
        Channel indices (Sympatec uses 1…31 on the R3 lens).
    noise_floor
        Optional externally supplied per-channel floor (counts). If given, it is
        preferred over the per-run heuristic.

    Remaining parameters are the thresholds from spec §6; all are exposed and
    recorded in the returned ``params`` for provenance.
    """
    I = np.asarray(I, dtype=float)
    t_min = np.asarray(t_min, dtype=float)
    params = {
        "abs_floor_counts": abs_floor_counts, "rel_floor_frac": rel_floor_frac,
        "drop_ch1": drop_ch1, "init_window_frames": init_window_frames,
        "gap_threshold_min": gap_threshold_min, "noise_floor": noise_floor,
    }

    # Despike synchronized frame artifacts (whole-spectrum + Copt jump together):
    # drops the leading obscuration/laser startup transient and interpolates interior
    # glitches. Needs Copt for the corroboration test, so it's skipped without it.
    despike_info = {"n_lead_dropped": 0, "spike_frames": []}
    if despike and copt is not None:
        I, t_min, copt, despike_info = despike_frames(I, t_min, np.asarray(copt, dtype=float))

    # Super-floor channel selection (preferred): a channel enters the gate only if
    # it shows real *directional* change above the CFZ-pH-7 noise surface — the
    # principled replacement for the static drop-ch1–4 mask. Computed on the run as
    # passed (the directional test smooths internally and is robust to the gap trim).
    super_floor = None
    if noise_surface is not None:
        if copt is None:
            raise ValueError("noise_filter: `copt` is required when `noise_surface` is given.")
        super_floor = noise_surface.is_significant(I, np.asarray(copt, dtype=float),
                                                   t_min, z_thresh=z_thresh)

    # §4.2 preprocessing — clean spikes, then re-zero any acquisition gap. Done
    # before the start value is computed so the mask and normalization share a
    # consistent frame 0.
    I = _mask_spikes(I)
    t_min, I, gap_rezeroed, gap_min = _rezero_gap(t_min, I, gap_threshold_min)

    # §4.1 active-channel selection — super-floor (noise surface) or the static mask.
    I0 = _initial_intensity(I, init_window_frames)
    if super_floor is not None:
        active = [i for i in range(len(channels)) if super_floor.real[i]]
        masked = [(channels[i], "sub_floor") for i in range(len(channels)) if not super_floor.real[i]]
        floor_used = None
        params["noise_surface"] = {"z_thresh": z_thresh, "k": noise_surface.k,
                                   "p": noise_surface.p, "rho": noise_surface.rho,
                                   "n_super_floor": len(active)}
    else:
        active, masked, floor_used = _active_mask(
            I0, channels, abs_floor_counts, rel_floor_frac, drop_ch1, noise_floor
        )
    params["floor_used"] = floor_used
    active_channels = [channels[i] for i in active]

    # Carried on every result: the despiked arrays the admission ran on (so figures
    # show the same cleaned data) + the despike provenance, plus the per-channel
    # drift diagnostics (super-floor path only — each channel's max windowed-trend z
    # and slope sign, aligned to the full input ``channels`` order).
    drift_fields = dict(
        clean_I=I, clean_t=t_min,
        n_lead_dropped=int(despike_info.get("n_lead_dropped", 0)),
        spike_frames=list(despike_info.get("spike_frames", [])),
    )
    if super_floor is not None:
        drift_fields.update(
            drift_channels=np.asarray(channels),
            drift_zmax=np.asarray(super_floor.zmax, dtype=float),
            drift_sign=np.asarray(super_floor.slope_sign, dtype=float),
            drift_z_thresh=float(super_floor.z_thresh),
        )

    return ChannelTriage(
        verdict="unrouted", r_min=float("nan"), active_channels=active_channels,
        masked_channels=masked, k=1, silhouette=None, bands=[],
        gap_rezeroed=gap_rezeroed, gap_min=gap_min,
        correlation_matrix=np.zeros((len(active), len(active))),
        params=params, flags=[], **drift_fields,
    )
