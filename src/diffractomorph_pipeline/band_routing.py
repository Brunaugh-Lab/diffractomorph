"""Channel band routing — single population vs. multiple anti-correlated size-bands.

The classifier that used to live inside ``noise_filter`` (its ``route_modes`` sub-routine).
Given a run the noise filter has already admitted (``active_channels`` + the cleaned arrays),
decide whether the admitted channels evolve as ONE dissolving population (``single_mode``) or as
SEVERAL bands (``multi_band``), via a per-run rank-1 + circular-shift surrogate null on the
minimum off-diagonal correlation ``r_min``. Only the kinetics extractor
(:mod:`diffractomorph_pipeline.extract`) needs this; ``dfm-run`` never routes.

    from diffractomorph_pipeline.noise_filter import noise_filter
    from diffractomorph_pipeline.band_routing import route_channels
    adm    = noise_filter(I, t_min, channels, noise_surface=surface, copt=copt)   # admission
    routed = route_channels(adm, channels)          # fills verdict / bands / r_min / null test

``triage_channels`` runs both in one call (the old combined API).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from diffractomorph_pipeline.noise_filter import ChannelTriage, _initial_intensity, noise_filter


@dataclass
class Band:
    """One channel band (multi-band runs only)."""
    id: str                          # "C1" (largest particles) … "Ck" (smallest)
    channel_range: tuple[int, int]   # (min, max) active channel index; ±1-ch tolerance
    role: str                        # "growth" | "crossover" | "dissolution" | "unassigned"
    mean_trajectory: np.ndarray      # band-mean normalized trajectory over time


def _assign_role(traj: np.ndarray) -> str:
    """Provisional mechanism role from band-mean trajectory shape (§4.6).

    Trajectory is normalized (starts ≈ 1). Rising → growth; monotonic decay →
    dissolution; dip-then-recover (interior minimum below both ends) → crossover.
    """
    if traj.size < 2:
        return "unassigned"
    start, end = float(traj[0]), float(traj[-1])
    net = end - start
    interior_min = float(traj[1:-1].min()) if traj.size > 2 else min(start, end)
    if interior_min < min(start, end) - 0.05 and end > interior_min + 0.05:
        return "crossover"
    if net > 0.05:
        return "growth"
    if net < -0.05:
        return "dissolution"
    return "unassigned"


def _rank1_null_pvalue(
    traj: np.ndarray,
    observed_r_min: float,
    n_surrogates: int = 2000,
    rng: np.random.Generator | None = None,
    method: str = "circular_shift",
) -> tuple[float, float, float]:
    """Per-run rank-1 + shuffled-residual null for ``r_min`` (audit §The test).

    Fits the rank-1 SVD model ``M1 = σ₁u₁v₁ᵀ`` (one decaying population), takes the
    residual ``R = traj − M1``, and builds ``n_surrogates`` surrogates of the form
    ``M1 + R'`` where each channel's residual column is **independently circularly
    time-shifted** (preserves its autocorrelation, destroys genuine cross-channel
    coupling). For each surrogate it recomputes ``r_min``; the one-sided p-value is

        ``p_real = (#{null r_min ≤ observed} + 1) / (n_surrogates + 1)``.

    Returns ``(p_real, σ₂/σ₁, rank1_var)``. The whole surrogate ensemble is
    vectorized — no Python loop over draws — so 2000 draws stay sub-second.
    """
    if rng is None:
        rng = np.random.default_rng()
    if method != "circular_shift":
        raise ValueError(f"unknown null_method: {method!r}")
    T, n = traj.shape
    U, s, Vt = np.linalg.svd(traj, full_matrices=False)
    M1 = s[0] * np.outer(U[:, 0], Vt[0])
    R = traj - M1
    ssq = float(np.sum(s ** 2))
    s2_over_s1 = float(s[1] / s[0]) if s.size > 1 and s[0] > 0 else 0.0
    rank1_var = float(s[0] ** 2 / ssq) if ssq > 0 else float("nan")

    offsets = rng.integers(1, T, size=(n_surrogates, n))                  # (N, n)
    gidx = (np.arange(T)[None, :, None] - offsets[:, None, :]) % T        # (N, T, n)
    Rs = R[gidx, np.arange(n)[None, None, :]]                            # (N, T, n)
    surr = M1[None, :, :] + Rs                                            # (N, T, n)

    sc = surr - surr.mean(axis=1, keepdims=True)
    std = np.sqrt(np.einsum("stc,stc->sc", sc, sc))                       # (N, n) ×√T
    cov = np.einsum("stc,std->scd", sc, sc)                               # (N, n, n) ×T
    denom = std[:, :, None] * std[:, None, :]
    corr = np.where(denom > 0, cov / np.where(denom > 0, denom, 1.0), 0.0)
    corr[:, np.eye(n, dtype=bool)] = np.inf                               # ignore diagonal
    null_r_min = corr.min(axis=(1, 2))

    p_real = float((np.sum(null_r_min <= observed_r_min) + 1) / (n_surrogates + 1))
    return p_real, s2_over_s1, rank1_var


def route_channels(admission: ChannelTriage, channels: list[int], *,
                   init_window_frames: int = 3, k_range: tuple[int, int] = (2, 5),
                   null_method: str = "circular_shift", n_surrogates: int = 2000,
                   alpha: float = 0.05, r_min_floor: float = -0.4,
                   random_state=None) -> ChannelTriage:
    """Route the admitted channels into single-mode vs. multi-band (§4.3–4.6).

    Takes the admission result from :func:`~diffractomorph_pipeline.noise_filter.noise_filter`
    (uses its ``clean_I`` + ``active_channels``) and returns a copy with the routing fields
    filled: ``verdict``, ``r_min``, ``k``, ``silhouette``, ``bands``, ``correlation_matrix``,
    and the per-run null diagnostics. Multi-band requires BOTH a significant ``p_real`` under the
    rank-1 + circular-shift null AND a meaningful effect size (``r_min < r_min_floor``).
    """
    I = np.asarray(admission.clean_I, dtype=float)
    channels = list(channels)
    active_channels = list(admission.active_channels)
    active = [channels.index(ch) for ch in active_channels]
    params = {**admission.params, "null_method": null_method, "n_surrogates": n_surrogates,
              "alpha": alpha, "r_min_floor": r_min_floor, "k_range": tuple(k_range),
              "random_state": random_state}

    # Edge case (§7): too few active channels / frames to compare → single-mode + warning.
    if len(active) < 2 or I.shape[0] < 2:
        return replace(admission, verdict="single_mode", k=1, bands=[], params=params,
                       flags=list(admission.flags) + ["low_signal"])

    # §4.3 normalized trajectories on active channels.
    I0 = _initial_intensity(I, init_window_frames)
    I0act = np.where(I0[active] == 0, 1.0, I0[active])
    traj = I[:, active] / I0act               # (T × n_active)

    C = np.nan_to_num(np.corrcoef(traj, rowvar=False), nan=0.0)
    np.fill_diagonal(C, 1.0)
    r_min = float(C[~np.eye(C.shape[0], dtype=bool)].min())

    rng = np.random.default_rng(random_state)
    p_real, s2_over_s1, rank1_var = _rank1_null_pvalue(
        traj, r_min, n_surrogates=n_surrogates, rng=rng, method=null_method)
    is_multi = (p_real < alpha) and (r_min < r_min_floor)
    null = dict(p_real=p_real, r_min_floor=r_min_floor, alpha=alpha,
                s2_over_s1=s2_over_s1, rank1_var=rank1_var)

    def _one_band():
        m = traj.mean(axis=1)
        return [Band("C1", (min(active_channels), max(active_channels)), _assign_role(m), m)]

    if not is_multi:                          # single-mode — do NOT force clusters (§7)
        return replace(admission, verdict="single_mode", r_min=r_min, k=1, silhouette=None,
                       bands=_one_band(), correlation_matrix=C, params=params, flags=[], **null)

    # §4.5 multi-band — cluster, pick k by best silhouette.
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score

    D = np.clip(((1.0 - C) + (1.0 - C).T) / 2.0, 0.0, None)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    kmin, kmax = k_range
    kmax = min(kmax, len(active))
    best_k, best_sil, best_labels = None, -np.inf, None
    for k in range(max(2, kmin), kmax + 1):
        labels = fcluster(Z, k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        sil = float(silhouette_score(D, labels, metric="precomputed"))
        if sil > best_sil:
            best_k, best_sil, best_labels = k, sil, labels

    if best_labels is None:                   # degenerate; fall back to single-mode
        return replace(admission, verdict="single_mode", r_min=r_min, k=1, silhouette=None,
                       bands=_one_band(), correlation_matrix=C, params=params,
                       flags=["low_confidence"], **null)

    # Relabel clusters by mean channel index: C1 = largest particles (low channel).
    ch_arr = np.array(active_channels)
    cluster_ids = sorted(set(best_labels), key=lambda cl: ch_arr[best_labels == cl].mean())
    bands: list[Band] = []
    for i, cl in enumerate(cluster_ids, start=1):
        member = best_labels == cl
        band_traj = traj[:, member].mean(axis=1)
        chs = ch_arr[member]
        bands.append(Band(f"C{i}", (int(chs.min()), int(chs.max())),
                          _assign_role(band_traj), band_traj))

    flags = []
    if r_min > r_min_floor - 0.1 or p_real > alpha / 5.0 or best_sil < 0.5:
        flags.append("low_confidence")
    return replace(admission, verdict="multi_band", r_min=r_min, k=best_k, silhouette=best_sil,
                   bands=bands, correlation_matrix=C, params=params, flags=flags, **null)


def triage_channels(I, t_min, channels, *, route_modes=True, k_range=(2, 5),
                    null_method="circular_shift", n_surrogates=2000, alpha=0.05,
                    r_min_floor=-0.4, random_state=None, init_window_frames=3,
                    **admission_kwargs) -> ChannelTriage:
    """Admission + routing in one call (the old combined API): run
    :func:`~diffractomorph_pipeline.noise_filter.noise_filter`, then — unless
    ``route_modes=False`` — :func:`route_channels`."""
    adm = noise_filter(I, t_min, channels, init_window_frames=init_window_frames, **admission_kwargs)
    if not route_modes:
        return adm
    return route_channels(adm, channels, init_window_frames=init_window_frames, k_range=k_range,
                          null_method=null_method, n_surrogates=n_surrogates, alpha=alpha,
                          r_min_floor=r_min_floor, random_state=random_state)
