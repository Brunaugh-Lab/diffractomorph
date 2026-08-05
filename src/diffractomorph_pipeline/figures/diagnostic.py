"""Diagnostic / QC figures — quick-look plots for lab members.

The headline diagnostic is :func:`plot_triage_diagnostic` (spec §10): a
one-glance, four-panel view of *why* a run routed the way it did.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from diffractomorph_pipeline.plot_styles import apply_lab_style, get_color, setup_axes


def plot_triage_diagnostic(triage, I, t_min, channels, out_path: Path | str,
                           init_window_frames: int = 3):
    """Four-panel triage diagnostic (spec §10).

    Panels, in pipeline order: (a) per-channel directional-drift significance
    (which channels are admitted), (b) correlation heatmap of the admitted
    channels, (c) normalized trajectories colored by band (single color if
    single-mode), (d) headline r_min + verdict.

    ``triage`` is the :class:`~diffractomorph_pipeline.noise_filter.ChannelTriage`
    result after routing (``band_routing.route_channels``); ``I``/``t_min``/``channels``
    are the same arrays passed to the noise filter.
    """
    import matplotlib.pyplot as plt

    apply_lab_style()
    I = np.asarray(I, dtype=float)
    n0 = min(init_window_frames, I.shape[0])
    I0 = I[:n0].mean(axis=0)
    active = triage.active_channels
    floor = triage.params.get("floor_used", np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7.6))
    # Read in pipeline order: channel admission (drift) → correlation on the
    # admitted channels → band trajectories → verdict.
    ax_drift, ax_corr, ax_traj, ax_text = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # (b) correlation heatmap (active channels)
    C = triage.correlation_matrix
    if C.size:
        im = ax_corr.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax_corr.set_xticks(range(len(active)))
        ax_corr.set_yticks(range(len(active)))
        ax_corr.set_xticklabels(active, fontsize=6, rotation=90)
        ax_corr.set_yticklabels(active, fontsize=6)
        fig.colorbar(im, ax=ax_corr, fraction=0.046, label="Pearson r")
    ax_corr.set_title("Channel × channel correlation", fontsize=9)

    # (a) per-channel directional-drift significance — the actual super-floor
    # evidence. Each channel's max windowed-trend z = |slope|/SE vs the noise
    # surface; a channel is admitted iff its z clears z_thresh. Bars colored by
    # the channel's *net* direction of change over the run (start vs end window),
    # not the max-z-window slope sign — so a trough channel that ends lower than
    # it started reads as decreasing, not "growth". Markers hollow below threshold.
    channels = np.asarray(channels)
    z = triage.drift_zmax
    if z is not None:
        z = np.asarray(z, dtype=float)
        net = I[-n0:].mean(axis=0) - I0       # net change over the run
        zt = triage.drift_z_thresh or 4.0
        zc = np.clip(z, 0.3, None)            # floor for the log axis
        admitted = z > zt
        groups = [(net <= 0, "#2C2C2A", "decreasing (↓)"),
                  (net > 0, "#D85A30", "increasing (↑)")]
        for mask, col, lab in groups:
            if not mask.any():
                continue
            ax_drift.vlines(channels[mask], 0.3, zc[mask], color=col, lw=1.4, alpha=0.85)
            adm, rej = mask & admitted, mask & ~admitted
            if adm.any():
                ax_drift.plot(channels[adm], zc[adm], "o", ms=4, color=col)
            if rej.any():
                ax_drift.plot(channels[rej], zc[rej], "o", ms=4, mfc="white",
                              mec=col, mew=1.2)
        ax_drift.axhline(zt, color="0.4", ls="--", lw=1)
        ax_drift.set_yscale("log")
        ax_drift.set_xlabel("channel"); ax_drift.set_ylabel("drift z  (|slope| / SE)")
        n_adm = int(admitted.sum())
        ax_drift.set_title(f"Per-channel drift significance\n"
                           f"{n_adm}/{len(channels)} admitted (super-floor)", fontsize=9)
        # Consolidated legend below the panel (2 cols) so it never covers stems.
        from matplotlib.lines import Line2D
        handles = [
            Line2D([], [], marker="o", ls="none", color="#2C2C2A", label="decreasing (↓)"),
            Line2D([], [], marker="o", ls="none", color="#D85A30", label="increasing (↑)"),
            Line2D([], [], marker="o", ls="none", mfc="white", mec="0.4", mew=1.2,
                   label="sub-floor (hollow)"),
            Line2D([], [], ls="--", color="0.4", label=f"z_thresh = {zt:g}"),
        ]
        ax_drift.legend(handles=handles, fontsize=6, ncol=2, loc="upper center",
                        bbox_to_anchor=(0.5, -0.20), frameon=False,
                        columnspacing=1.3, handletextpad=0.4)
    else:
        # Static-mask path (no noise surface): fall back to the start-intensity
        # profile with the scalar floor drawn.
        I0_clipped = np.clip(I0, 1e-3, None)
        ax_drift.semilogy(channels, I0_clipped, "o-", ms=3, color="#2C2C2A")
        if floor is not None and np.isfinite(floor):
            ax_drift.axhline(floor, color="#D85A30", ls="--", lw=1, label=f"floor={floor:.2g}")
            ax_drift.legend(fontsize=7)
        ax_drift.set_xlabel("channel"); ax_drift.set_ylabel("start intensity")
        ax_drift.set_title("Start-intensity profile\nstatic-mask selection", fontsize=9)
    setup_axes(ax_drift)

    # (c) normalized trajectories colored by band
    for bi, band in enumerate(triage.bands):
        ax_traj.plot(t_min, band.mean_trajectory, lw=2, color=get_color(bi),
                     label=f"{band.id} ch{band.channel_range[0]}–{band.channel_range[1]} ({band.role})")
    ax_traj.axhline(1.0, color="0.7", lw=0.6)
    ax_traj.set_xlabel("time (min)"); ax_traj.set_ylabel("normalized I(c,t) / I(c,0)")
    ax_traj.set_title("Band-mean trajectories", fontsize=9)
    if triage.bands:
        ax_traj.legend(fontsize=7)
    setup_axes(ax_traj)

    # (d) headline text
    ax_text.axis("off")
    p_str = f"{triage.p_real:.4f}" if triage.p_real is not None else "n/a"
    lines = [
        f"VERDICT: {triage.verdict.upper()}",
        f"r_min = {triage.r_min:.3f}   (floor: {triage.r_min_floor})",
        f"p_real = {p_str}   (alpha: {triage.alpha})",
        f"k = {triage.k}" + (f"   silhouette = {triage.silhouette:.2f}" if triage.silhouette is not None else ""),
        f"active: {len(active)} ch   masked: {len(triage.masked_channels)} ch",
    ]
    if triage.gap_rezeroed:
        lines.append(f"gap re-zeroed: {triage.gap_min:.1f} min")
    if triage.flags:
        lines.append("flags: " + ", ".join(triage.flags))
    ax_text.text(0.02, 0.95, "\n".join(lines), va="top", ha="left",
                 family="monospace", fontsize=11, transform=ax_text.transAxes)

    fig.tight_layout(h_pad=3.0)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def plot_channel_noise_grid(triage, I, t_min, channels, surface, out_path: Path | str,
                            ncols: int = 6, init_window_frames: int = 3):
    """Small-multiples grid: every channel's trajectory vs a no-trend noise null.

    One panel per channel. The grey band is the **no-trend noise null** — where the
    signal *would* wander if it were only bouncing (not dissolving): a bootstrap of
    AR(1) fluctuations with the noise surface's per-frame magnitude at the channel's
    start level (``surface.sigma``, ``surface.rho``), anchored at that start level
    (the central 95% of the simulated flat-null trajectories). The colored line is
    the **observed** trajectory. The question each panel answers: does the real
    signal pull *away* from the band (directional change ⇒ admitted) or stay *inside*
    it (just noise ⇒ sub-floor)? Colored by the super-floor verdict — admitted in the
    net-change direction (decreasing dark / increasing orange), sub-floor grey. The
    per-channel evidence behind the collapsed drift panel of
    :func:`plot_triage_diagnostic`.
    """
    import matplotlib.pyplot as plt

    apply_lab_style()
    I = np.asarray(I, dtype=float)
    t_min = np.asarray(t_min, dtype=float)
    channels = np.asarray(channels)
    n_ch = len(channels)
    n0 = min(init_window_frames, I.shape[0])
    net = I[-n0:].mean(axis=0) - I[:n0].mean(axis=0)

    z = triage.drift_zmax
    zt = triage.drift_z_thresh or 4.0
    if z is None:                       # no surface used → no admission info
        z = np.full(n_ch, np.nan)

    # No-trend bootstrap null: unit AR(1) fluctuations, scaled per channel by the
    # measured noise magnitude. Generated once (fixed seed) and reused per channel.
    n_boot, T = 300, I.shape[0]
    rho = float(np.clip(surface.rho, -0.99, 0.99))
    rng = np.random.default_rng(0)
    white = rng.standard_normal((n_boot, T))
    e = np.empty((n_boot, T))
    e[:, 0] = white[:, 0]
    a = np.sqrt(1.0 - rho ** 2)
    for kk in range(1, T):
        e[:, kk] = rho * e[:, kk - 1] + a * white[:, kk]   # unit-variance AR(1)

    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 1.7 * nrows),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()

    DARK, ORANGE, GREY = "#2C2C2A", "#D85A30", "0.6"
    n_adm = 0
    for c in range(n_ch):
        ax = axes[c]
        admitted = np.isfinite(z[c]) and z[c] > zt
        n_adm += int(admitted)
        col = ORANGE if (admitted and net[c] > 0) else (DARK if admitted else GREY)
        # No-trend null anchored at the channel's start level.
        base = float(I[:n0, c].mean())
        sig = float(surface.sigma(max(base, 1e-6)))
        sims = base + sig * e
        lo, hi = np.percentile(sims, [2.5, 97.5], axis=0)
        ax.fill_between(t_min, lo, hi, color=GREY, alpha=0.28, lw=0, zorder=1)
        ax.axhline(base, color=GREY, lw=0.5, ls=":", zorder=1)
        # Observed trajectory.
        ax.plot(t_min, I[:, c], "-", lw=1.2, color=col, zorder=3)
        ztxt = f"z={z[c]:.0f}" if np.isfinite(z[c]) else ""
        mark = "" if admitted else " (sub)"
        ax.set_title(f"ch{channels[c]}  {ztxt}{mark}", fontsize=7, color=col, pad=2)
        ax.tick_params(labelsize=5)
        setup_axes(ax)
    for c in range(n_ch, len(axes)):
        axes[c].axis("off")

    # Shared axis labels on the outer frame.
    for c in range(n_ch):
        if c // ncols == nrows - 1 or c + ncols >= n_ch:
            axes[c].set_xlabel("time (min)", fontsize=6)
        if c % ncols == 0:
            axes[c].set_ylabel("intensity", fontsize=6)

    fig.suptitle(
        f"Observed trajectory vs no-trend noise null  "
        f"({n_adm}/{n_ch} channels admitted, z_thresh={zt:g}; grey = 95% of "
        f"bootstrapped flat-null fluctuation, anchored at start)",
        fontsize=10, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_channel_overlay(triage, I, t_min, channels, out_path: Path | str,
                         init_window_frames: int = 3):
    """All admitted channels superimposed — raw and start-normalized.

    Overlays only the channels that passed the noise threshold
    (``triage.active_channels``), colored by channel index (small → large size).
    Left: raw intensity. Right: ``I(c,t)/I(c,0)``. The normalized curves do **not**
    collapse — they share a rough timescale but fan out in *extent*: large channels
    empty (~90% drop) while small channels barely move or rise, because mass
    cascades down through size bins as particles shrink (Nernst-Brunner: small bins
    are fed by larger particles shrinking into them). The size-graded extent is the
    expected single-population signature, not multiple modes.
    """
    import matplotlib.pyplot as plt

    apply_lab_style()
    I = np.asarray(I, dtype=float)
    t_min = np.asarray(t_min, dtype=float)
    channels = np.asarray(channels)
    n0 = min(init_window_frames, I.shape[0])

    active = set(triage.active_channels)
    idx = [i for i, c in enumerate(channels) if c in active]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.viridis
    norm = plt.Normalize(int(channels.min()), int(channels.max()))
    for i in idx:
        col = cmap(norm(channels[i]))
        base = max(I[:n0, i].mean(), 1e-9)
        axes[0].plot(t_min, I[:, i], lw=1, color=col, alpha=0.9)
        axes[1].plot(t_min, I[:, i] / base, lw=1, color=col, alpha=0.9)
    axes[0].set_title("Admitted channels — raw intensity", fontsize=10)
    axes[0].set_xlabel("time (min)"); axes[0].set_ylabel("intensity")
    axes[1].set_title("Admitted channels — normalized I(c,t)/I(c,0)", fontsize=10)
    axes[1].set_xlabel("time (min)"); axes[1].set_ylabel("normalized")
    axes[1].axhline(1.0, color="0.7", lw=0.6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    for ax in axes:
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02
                     ).set_label("channel (small → large size)", fontsize=8)
        setup_axes(ax)

    fig.suptitle(
        f"All admitted channels superimposed  "
        f"({len(idx)}/{len(channels)} channels passed the noise threshold)",
        fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_day_overlay(reps, channels, out_path: Path | str, label: str = "",
                     n_grid: int = 120, init_window_frames: int = 3):
    """Replicate-averaged overlay for one day (folder of reps).

    ``reps`` is a list of ``(I, t)`` arrays — the *despiked* per-rep data (e.g.
    ``tri.clean_I`` / ``tri.clean_t``). Each rep is interpolated onto a common time
    grid, then per-channel averaged across reps. Left: mean raw intensity. Right:
    mean ``I(c,t)/I(c,0)`` with a ±1 SD band (across reps) so the per-channel
    reproducibility is visible — tight for the large dissolution channels, wide for
    the small cascade-fed ones. Colored by channel index (small → large size).
    """
    import matplotlib.pyplot as plt

    apply_lab_style()
    channels = np.asarray(channels)
    C = len(channels)
    reps = [(np.asarray(I, float), np.asarray(t, float)) for I, t in reps]
    tmax = min(t[-1] for _, t in reps)
    tg = np.linspace(0.0, tmax, n_grid)

    raw = np.zeros((len(reps), n_grid, C))
    norm = np.zeros_like(raw)
    for ri, (I, t) in enumerate(reps):
        n0 = min(init_window_frames, I.shape[0])
        I0 = np.maximum(I[:n0].mean(axis=0), 1e-9)
        for c in range(C):
            raw[ri, :, c] = np.interp(tg, t, I[:, c])
            norm[ri, :, c] = np.interp(tg, t, I[:, c] / I0[c])
    raw_m = raw.mean(0)
    norm_m, norm_sd = norm.mean(0), norm.std(0)

    fig, (axr, axn) = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.viridis
    cnorm = plt.Normalize(int(channels.min()), int(channels.max()))
    for c in range(C):
        col = cmap(cnorm(channels[c]))
        axr.plot(tg, raw_m[:, c], lw=1, color=col, alpha=0.9)
        axn.plot(tg, norm_m[:, c], lw=1, color=col, alpha=0.9)
        axn.fill_between(tg, norm_m[:, c] - norm_sd[:, c], norm_m[:, c] + norm_sd[:, c],
                         color=col, alpha=0.10, lw=0)
    axr.set_title("Replicate-mean — raw intensity", fontsize=10)
    axr.set_xlabel("time (min)"); axr.set_ylabel("intensity")
    axn.set_title("Replicate-mean — normalized I(c,t)/I(c,0)  (±SD band)", fontsize=10)
    axn.set_xlabel("time (min)"); axn.set_ylabel("normalized")
    axn.axhline(1.0, color="0.7", lw=0.6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
    for ax in (axr, axn):
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02
                     ).set_label("channel (small → large size)", fontsize=8)
        setup_axes(ax)

    head = f"{label} — " if label else ""
    fig.suptitle(f"{head}mean of {len(reps)} replicates", fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_dissolution_vs_size(reps, channels, char_size, out_path: Path | str,
                             label: str = "", init_window_frames: int = 3):
    """Per-channel dissolution vs **physical particle size** (µm), day-averaged.

    Puts a real size axis (the Mie kernel's ``char_size``, hardware-anchored) on the
    per-channel analysis. ``reps`` is a list of ``(I, t)`` despiked arrays. Per
    channel and rep: normalized ``p(t)=I/I0``; **extent** = fraction of signal lost
    (1 − plateau), and a model-free **t½** (time to reach halfway to the plateau).

    Left — extent vs size: the clean readout; small particles dissolve nearly
    completely, large ones barely (Noyes-Whitney). Right — rate ≈ 1/t½ vs size:
    **cascade-confounded** (small-size channels are fed by larger particles
    shrinking into them, slowing their apparent rate), shown with that caveat.
    """
    import matplotlib.pyplot as plt

    apply_lab_style()
    channels = np.asarray(channels)
    char_size = np.asarray(char_size, dtype=float)
    C = len(channels)
    reps = [(np.asarray(I, float), np.asarray(t, float)) for I, t in reps]

    def per_rep(I, t):
        n0 = min(init_window_frames, I.shape[0])
        p = I / np.maximum(I[:n0].mean(0), 1e-9)
        plat = p[-max(3, len(t) // 10):].mean(0)
        extent = 1.0 - plat
        thalf = np.full(C, np.nan)
        for c in range(C):
            if extent[c] > 0.1:
                below = np.where(p[:, c] <= 1 - 0.5 * extent[c])[0]
                if below.size:
                    thalf[c] = t[below[0]]
        return extent, thalf

    ext = np.array([per_rep(I, t)[0] for I, t in reps])
    th = np.array([per_rep(I, t)[1] for I, t in reps])
    ext_m, ext_s = np.nanmean(ext, 0), np.nanstd(ext, 0)
    rate = 1.0 / np.nanmean(th, 0)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.errorbar(char_size, 100 * ext_m, yerr=100 * ext_s, fmt="o", ms=5, capsize=2,
                color="#2C6FB5")
    a1.set_xscale("log"); a1.invert_xaxis()
    a1.set_xlabel("particle size (µm)"); a1.set_ylabel("% signal lost (extent)")
    a1.set_title("Dissolution extent vs size", fontsize=10)
    setup_axes(a1)

    ok = np.isfinite(rate)
    a2.plot(char_size[ok], rate[ok], "o", ms=5, color="#D85A30")
    a2.set_xscale("log"); a2.invert_xaxis()
    a2.set_xlabel("particle size (µm)"); a2.set_ylabel("rate ≈ 1/t½  (1/min)")
    a2.set_title("Dissolution rate vs size  (cascade-confounded)", fontsize=10)
    setup_axes(a2)

    head = f"{label} — " if label else ""
    fig.suptitle(f"{head}per-channel dissolution vs physical size "
                 f"(mean of {len(reps)} reps)", fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_kinetics_diagnostic(fit, run, triage, out_path: Path | str, window=None):
    """Four-panel KWW kinetics diagnostic — *why* a run's fit looks the way it does.

    Panels: (a) observed signal mass ``a(t)`` with the fitted KWW curve, (b) fit
    residuals vs time (structure ⇒ wrong model), (c) Copt(t) and the AUC-vs-Copt
    divergence ratio ``R(t)`` (precipitation indicator), (d) headline parameters.

    ``fit`` is the :class:`~diffractomorph_pipeline.extract.KineticFit`; ``run`` and
    ``triage`` are re-used to reconstruct the exact fitted trajectory (same
    obstruction-trim / gap-rezero / ``window`` the fit used).
    """
    import matplotlib.pyplot as plt

    from diffractomorph_pipeline.extract import copt_divergence, dissolution_signal, kww

    apply_lab_style()
    t, a, copt, chans, target, band_id, _ = dissolution_signal(run, triage, window)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    ax_fit, ax_res, ax_copt, ax_text = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    have_fit = np.isfinite(fit.k) and np.isfinite(fit.beta) and t.size
    a_hat = kww(t, fit.a_inf, fit.da, fit.k, fit.beta) if have_fit else None

    # (a) observed signal mass + fitted curve
    ax_fit.plot(t, a, "o", ms=3, color="#2C2C2A", label="observed a(t) = ΣI")
    if a_hat is not None:
        ax_fit.plot(t, a_hat, "-", lw=2, color=get_color(0),
                    label=f"{fit.model} fit (β={fit.beta:.2f})")
        ax_fit.axhline(fit.a_inf, color="0.7", ls=":", lw=1)
    ax_fit.set_xlabel("time (min)"); ax_fit.set_ylabel("signal mass  ΣI(c,t)")
    ax_fit.set_title(f"Dissolution signal — {target}"
                     + (f" ({band_id})" if band_id else ""), fontsize=9)
    ax_fit.legend(fontsize=7)
    setup_axes(ax_fit)

    # (b) residuals
    if a_hat is not None:
        ax_res.plot(t, a - a_hat, "o-", ms=3, lw=0.8, color="#D85A30")
    ax_res.axhline(0.0, color="0.6", lw=0.8)
    ax_res.set_xlabel("time (min)"); ax_res.set_ylabel("observed − fit")
    ax_res.set_title("Fit residuals", fontsize=9)
    setup_axes(ax_res)

    # (c) Copt and the AUC-vs-Copt divergence ratio R(t)
    ax_copt.plot(t, copt, "o-", ms=2, lw=0.8, color="#2C2C2A", label="Copt (%)")
    ax_copt.set_xlabel("time (min)"); ax_copt.set_ylabel("Copt (%)", color="#2C2C2A")
    if t.size and a[0] > 0 and np.isfinite(copt).all() and copt[0] > 0:
        R = (a * copt[0]) / (copt * a[0])
        ax_r = ax_copt.twinx()
        ax_r.plot(t, R, "-", lw=1.5, color=get_color(2), label="R(t)")
        ax_r.axhline(1.0, color=get_color(2), ls=":", lw=1)
        ax_r.set_ylabel("R(t) = a*Copt0 / (Copt*a0)", color=get_color(2))
    ax_copt.set_title("Copt & AUC-vs-Copt divergence", fontsize=9)
    setup_axes(ax_copt)

    # (d) headline text
    ax_text.axis("off")
    hl = f"{fit.half_life:.2f}" if np.isfinite(fit.half_life) else "n/a"
    div = f"{fit.copt_divergence:.2f}" if np.isfinite(fit.copt_divergence) else "n/a"
    lines = [
        f"MODEL: {fit.model.upper()}   ({fit.verdict})",
        f"k    = {fit.k:.3f} /min   (±{fit.k_se:.3f})" if np.isfinite(fit.k) else "k    = n/a",
        f"beta = {fit.beta:.3f}      (±{fit.beta_se:.3f})" if np.isfinite(fit.beta) else "beta = n/a",
        f"R^2  = {fit.r2:.4f}   (first-order R^2 = {fit.fo_r2:.4f})",
        f"t1/2 = {hl} min     drop = {fit.signal_drop_frac:.0%}"
        if np.isfinite(fit.signal_drop_frac) else f"t1/2 = {hl} min",
        f"channels: {len(fit.channels)}   frames: {fit.n_frames}",
        f"Copt divergence (late |R-1|): {div}",
    ]
    if fit.flags:
        lines.append("flags: " + ", ".join(fit.flags))
    ax_text.text(0.02, 0.95, "\n".join(lines), va="top", ha="left",
                 family="monospace", fontsize=11, transform=ax_text.transAxes)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
