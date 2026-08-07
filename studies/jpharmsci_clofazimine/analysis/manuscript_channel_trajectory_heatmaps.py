"""Manuscript figure — condition-level detector-channel trajectory analysis (pH-dependent study).

Purpose: determine whether detector channels show **coordinated decay** and whether their **decay
timescales are similar** — kept strictly descriptive. No Mie, q3, UV, forward-model, or particle-size
claim is made; a detector channel is never equated with a particle size, and high correlation is never
read as identical dissolution rates.

Two facts the earlier "gray block" conflated are now separated:
  1. a channel passed calibrated raw-intensity QC / directional-change admission; vs.
  2. a channel had near-zero INITIAL reference-adjusted signal and so cannot support normalization by
     its initial value. This is NOT a noise-QC failure — an initially empty channel that later develops
     signal is an **emergence/redistribution** trajectory (examined separately in absolute units), not a
     decay trajectory.

Calibrated workflow (production ``noise_filter`` / ``noise_surface`` are NOT modified):

    surface = load_surface()
    tri = noise_filter(run.I, run.t_min, run.channels, noise_surface=surface, copt=run.copt, z_thresh=4)
    X_c(t) = max(clean_I_c(t) − reference_c, 0)        # particle-associated, reference-adjusted signal

``copt`` is used ONLY for the existing synchronized-despike corroboration — never as an inclusion
threshold. ``frame_reliable(clean_I, snr_min=5)`` is retained as a separate raw-measurement diagnostic.

Structure:
  * Part 1 — audit the low-initial channels (1–4): are they flat near reference, or do they develop a
    reproducible **late-emerging low-angle signal**? (run / date / summary CSVs + markdown).
  * Part 2 — primary normalized heatmap over ONE condition-blind **common eligible channel set** (channels
    with reliable initial reference-adjusted signal across independent dates in every condition). Displays
    ``Y_c(t)=X_c(t)/X_c(0)`` — eligible rows only, no gray block for deliberately-excluded low-initial
    channels, no directional strip, no channel-1 over-range row.
  * Part 3 — the low-initial channels 1–4 shown separately in **absolute** reference-adjusted intensity.
  * Part 4 — correlations computed from the cleaned reference-adjusted trajectories (NOT the normalized
    ratios): Pearson, Spearman, each channel vs the total reference-adjusted signal, and a first-
    difference sensitivity; date-first, never one concatenated series.
  * Part 5 — a robust per-channel ``t50`` (time to 50 % of the initial reference-adjusted signal) rate
    summary; pH 5.0 treated cautiously (limited decay ⇒ ``t50`` often non-estimable, kept missing).

Run with the pipeline venv::

    python analysis/manuscript_channel_trajectory_heatmaps.py --output-dir <…/figures_and_tables> \
        [--formats pdf,png,svg,tiff]
"""
from __future__ import annotations

import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from diffractomorph_pipeline import ingest, plot_styles as ps
from diffractomorph_pipeline.noise_filter import noise_filter
from diffractomorph_pipeline.noise_surface import load_surface

import manuscript_q3_scattering_evolution as mqs
from psd_evolution_common import iter_runs

CONDITIONS = [4.0, 4.5, 5.0]
CHANNELS = list(range(1, 32))
N_CH = 31
INIT_FRAMES = 3                        # the established 3-frame initial stable-frame window
Z_THRESH = 4.0                         # directional-change (super-floor) admission threshold
SNR_MIN = 5.0                          # per-frame reliability + initial-signal SNR floor
LOW_INIT_CHANNELS = [1, 2, 3, 4]       # audited/plotted separately (near-zero initial signal)
MIN_DATES_ELIGIBLE = 2                 # eligibility: reliable-initial in ≥ this many dates, every condition
LATE_SIGNAL_INITIAL_FACTOR = 3.0       # "later positive" = post-initial max ≥ this × the (tiny) initial
ILLUSTRATIVE = (4.5, 20260608, 2)      # author-prespecified illustrative run (NOT claimed representative)
EVENT_WINDOW_MIN = (8.0, 9.5)          # the shared abrupt event to report on (handled by existing despike QC)
HEATMAP_CMAP = "viridis"               # sequential: ~1 = retained initial intensity, →0 = deeper loss
OVER_COLOR = "#b2182b"                 # explicit over-range colour for the few cells above the principal cap
MISSING_COLOR = "0.8"                  # neutral gray for genuinely missing / below-noise cells
PH_COLORS = {4.0: "#0072B2", 4.5: "#E69F00", 5.0: "#009E73"}   # colorblind-safe per-pH
LOW_CH_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73", 4: "#CC79A7"}
# per-cell status codes (both channel-noise decisions preserved)
ST_DIR = "reliable_directional_change_detected"
ST_NODIR = "reliable_no_directional_change_detected"
ST_NOISE = "below_frame_noise"
ST_MISSING = "missing_or_qc_excluded"
# low-initial trajectory classes (Part 1)
TC_DECAY = "initially_measurable_and_decaying"
TC_EMERGE = "initially_unmeasurable_with_later_positive"
TC_FLAT = "consistently_near_reference"
TC_INDET = "indeterminate"


def _trapz(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 2:
        return np.nan
    yy, xx = y[m], x[m]
    return float(np.sum(0.5 * (yy[:-1] + yy[1:]) * np.diff(xx)))


# ── per-run trajectories: calibrated admission + per-frame reliability ────────
def _align_index(clean_t, grid, tol=None):
    """Nearest despiked-frame index for each grid time (within tolerance), else -1. One index reused for
    every aligned quantity so raw / cleaned / X / reliability stay consistent."""
    tol = mqs.TOL_MIN if tol is None else tol
    t = np.asarray(clean_t, float)
    idx = np.full(len(grid), -1, int)
    if not t.size:
        return idx
    for gi, gt in enumerate(grid):
        k = int(np.argmin(np.abs(t - gt)))
        if abs(t[k] - gt) <= tol + 1e-9:
            idx[gi] = k
    return idx


def build_run(rtf, ph, date, rep, grid, surface):
    """One run → despiked, reference-adjusted trajectories on the common grid, using the **calibrated**
    CFZ-pH-7 noise surface. Retains the absolute reference-adjusted signal ``X`` (for the low-initial
    audit + correlation + t50) and the per-channel decisions (directional admission; per-frame
    reliability; reliable-initial). ``copt`` reaches ``noise_filter`` ONLY for the production despike
    corroboration — never as an inclusion threshold."""
    run = ingest.extract_run(rtf)
    I_raw = np.asarray(run.I, float)
    tri = noise_filter(I_raw, np.asarray(run.t_min, float), run.channels,
                       noise_surface=surface, copt=np.asarray(run.copt, float), z_thresh=Z_THRESH)
    assert "noise_surface" in tri.params, "calibrated noise surface was not used"
    clean = np.asarray(tri.clean_I, float)
    t = np.asarray(tri.clean_t, float)
    ref = np.asarray(run.ref, float)
    X = np.clip(clean - ref[None, :], 0.0, None)                  # X_c(t) = max(clean_I − R, 0)
    n0 = min(INIT_FRAMES, X.shape[0])
    X0 = X[:n0].mean(axis=0)                                       # per-channel initial (3-frame window)
    frel = np.asarray(surface.frame_reliable(clean, snr_min=SNR_MIN), bool)   # per-cell reliability (on clean_I)
    # reliable-initial = initial reference-adjusted signal above the calibrated noise (≥ SNR_MIN·σ). This
    # decides *normalizability* (a valid initial value), and is DISTINCT from frame-reliability QC.
    sigma_init = np.asarray(surface.sigma(clean[:n0].mean(axis=0)), float)
    norm_reliable = X0 >= SNR_MIN * sigma_init
    directional = set(tri.active_channels)

    # grid alignment (one nearest-frame index reused for every quantity; no interpolation)
    idx = _align_index(t, grid)
    ng = len(grid)
    G_X = np.full((ng, N_CH), np.nan); G_raw = np.full((ng, N_CH), np.nan)
    G_clean = np.full((ng, N_CH), np.nan); G_frel = np.zeros((ng, N_CH), bool)
    nlead = int(tri.n_lead_dropped)
    for gi, k in enumerate(idx):
        if k < 0:
            continue
        G_clean[gi] = clean[k]; G_X[gi] = X[k]; G_frel[gi] = frel[k]
        rk = k + nlead
        G_raw[gi] = I_raw[rk] if rk < I_raw.shape[0] else clean[k]

    # per-channel audit metrics on the despiked trajectory (Part 1); NO fold change when X0 ≈ 0
    post = np.arange(X.shape[0]) >= n0
    metrics = []
    for ci, c in enumerate(CHANNELS):
        xc = X[:, ci]; xpost = xc[post]; tpost = t[post]
        if xpost.size and np.isfinite(xpost).any():
            kk = int(np.nanargmax(xpost)); maxp = float(xpost[kk]); tmax = float(tpost[kk])
        else:
            maxp, tmax = np.nan, np.nan
        final = float(xc[-1]) if xc.size else np.nan
        sig_at_max = SNR_MIN * float(surface.sigma(np.array([maxp + ref[ci]]))[0]) if np.isfinite(maxp) else np.nan
        late_positive = bool(np.isfinite(maxp) and maxp >= sig_at_max
                             and maxp >= LATE_SIGNAL_INITIAL_FACTOR * max(X0[ci], 1e-9))
        if norm_reliable[ci] and np.isfinite(final) and final < 0.5 * X0[ci]:
            tclass = TC_DECAY
        elif (not norm_reliable[ci]) and late_positive:
            tclass = TC_EMERGE
        elif (not norm_reliable[ci]) and not late_positive:
            tclass = TC_FLAT
        else:
            tclass = TC_INDET
        metrics.append(dict(channel=c, X0_initial=float(X0[ci]), max_post_initial=maxp, t_max_min=tmax,
                            final=final, auc_post=_trapz(xpost, tpost), frac_frame_reliable=float(frel[:, ci].mean()),
                            norm_reliable=bool(norm_reliable[ci]), directional=bool(c in directional),
                            trajectory_class=tclass))

    spike_t = [float(t[k]) for k in tri.spike_frames if 0 <= k < len(t)]
    event_hit = sorted({round(s, 2) for s in spike_t if EVENT_WINDOW_MIN[0] <= s <= EVENT_WINDOW_MIN[1]})
    return dict(ph=float(ph), date=int(date), rep=int(rep), X=X, X0=X0, t=t,
                G_X=G_X, G_raw=G_raw, G_clean=G_clean, G_frel=G_frel,
                norm_reliable=norm_reliable, directional=sorted(directional), metrics=metrics,
                frame_reliable_frac=float(frel.mean()), noise_surface_params=tri.params.get("noise_surface"),
                spike_frames=list(tri.spike_frames), spike_times=spike_t, event_spikes=event_hit,
                n_lead_dropped=nlead, n_directional=len(directional))


def legacy_admitted(rtf):
    """The LEGACY static starting-intensity admission (no noise surface) — for the QC before/after audit
    only. Not used to gate any figure."""
    run = ingest.extract_run(rtf)
    tri = noise_filter(np.asarray(run.I, float), np.asarray(run.t_min, float), run.channels,
                       copt=np.asarray(run.copt, float))
    return set(tri.active_channels), ("noise_surface" in tri.params), list(tri.spike_frames)


# ── condition-blind common eligible channel set (Part 2 rule) ─────────────────
def common_eligible_channels(runs_by_ph, min_dates=MIN_DATES_ELIGIBLE):
    """Condition-blind eligibility: a channel is eligible in a condition if its INITIAL reference-adjusted
    signal is reliable (≥ SNR_MIN·σ, majority of replicates) in ≥ ``min_dates`` independent preparation
    dates. The **common** set (used for all three panels) is the intersection over conditions. Returns the
    rule text, per-condition eligible sets, and the common set."""
    per_cond = {}
    for ph in CONDITIONS:
        eligible = []
        for ci in range(N_CH):
            by_date = defaultdict(list)
            for r in runs_by_ph[ph]:
                by_date[r["date"]].append(bool(r["norm_reliable"][ci]))
            n_dates = sum(1 for _d, v in by_date.items() if np.mean(v) >= 0.5)
            if n_dates >= min_dates:
                eligible.append(ci + 1)
        per_cond[ph] = eligible
    common = sorted(set.intersection(*[set(per_cond[ph]) for ph in CONDITIONS])) if per_cond else []
    rule = (f"A channel is eligible where its INITIAL reference-adjusted signal exceeds the calibrated "
            f"noise (X0 ≥ {SNR_MIN:g}·σ, in a majority of replicates) in ≥ {min_dates} of the 3 "
            f"preparation dates in EVERY pH condition (condition-blind intersection). This is the "
            f"normalizability criterion, NOT frame-reliability QC.")
    return dict(rule=rule, per_condition=per_cond, common=common)


# ── date-first NaN-aware aggregation ──────────────────────────────────────────
def _nanmean(stack, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stack, axis=axis)


def _nanmedian(stack, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stack, axis=axis)


def _nice_ceiling(v, headroom=1.02):
    """Smallest publication-friendly ceiling ≥ ``v * headroom`` (so it is strictly greater than ``v`` with
    a little headroom). Data-driven: picks a nice mantissa (1/1.2/1.5/2/2.5/3/4/5/6/8/10) at ``v``'s decade,
    so e.g. a plotted max of 2.37 rounds up to 2.5. No axis value is hard-coded."""
    if not np.isfinite(v) or v <= 0:
        return 0.1
    target = v * headroom
    p = 10.0 ** np.floor(np.log10(target))
    for m in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if m * p >= target:
            return round(float(m * p), 6)
    return round(float(10.0 * p), 6)


def _normalized_G(run, eligible):
    """Per-run normalized Y = X/X0 on the grid, ONLY for eligible channels with a reliable initial value;
    all other channel-time cells stay NaN (never zero). Division by a zero/near-zero initial never
    happens (guarded by the reliable-initial eligibility)."""
    G = np.full_like(run["G_X"], np.nan)
    for c in eligible:
        ci = c - 1
        if run["norm_reliable"][ci] and run["X0"][ci] > 0:
            col = run["G_X"][:, ci] / run["X0"][ci]
            G[:, ci] = np.where(run["G_frel"][:, ci], col, np.nan)   # keep only frame-reliable cells
    return G


def aggregate_condition(runs, eligible, grid, min_dates=2):
    """Date-first: replicate matrices (normalized Y for eligible channels) averaged within a preparation
    date, then the cross-date MEDIAN per (channel, time). A cell is drawable only with ≥ ``min_dates``
    dates. Non-eligible channels stay NaN throughout."""
    ng = len(grid)
    dates = sorted({r["date"] for r in runs})
    date_mats, date_runcount = {}, {}
    for d in dates:
        stk = np.stack([_normalized_G(r, eligible) for r in runs if r["date"] == d])
        date_mats[d] = _nanmean(stk, axis=0)
        date_runcount[d] = np.sum(~np.isnan(stk), axis=0)
    dstack = np.stack([date_mats[d] for d in dates]) if dates else np.full((0, ng, N_CH), np.nan)
    n_dates_cell = np.sum(~np.isnan(dstack), axis=0)
    n_runs_cell = np.sum([date_runcount[d] for d in dates], axis=0) if dates else np.zeros((ng, N_CH), int)
    cond = _nanmedian(dstack, axis=0)
    return dict(median=np.where(n_dates_cell >= min_dates, cond, np.nan), n_dates=n_dates_cell,
                n_runs=n_runs_cell, date_mats=date_mats, dates=dates)


# ── Part 1: low-initial channel audit (channels 1–4) ──────────────────────────
def low_initial_audit_run_level(runs_by_ph):
    rows = []
    for ph in CONDITIONS:
        for r in runs_by_ph[ph]:
            m = {d["channel"]: d for d in r["metrics"]}
            for c in LOW_INIT_CHANNELS:
                d = m[c]
                x0 = d["X0_initial"]
                rows.append(dict(
                    ph=ph, date=r["date"], rep=r["rep"], channel=c,
                    initial_ref_adjusted=round(x0, 4), max_post_initial=_r(d["max_post_initial"]),
                    t_max_min=_r(d["t_max_min"], 2), final_ref_adjusted=_r(d["final"]),
                    auc_post_initial=_r(d["auc_post"], 3),
                    directional_admission=("admitted" if d["directional"] else "not_admitted"),
                    frac_frame_reliable=round(d["frac_frame_reliable"], 3),
                    despike_event_times_min=",".join(f"{s:.1f}" for s in r["event_spikes"]) or "none",
                    trajectory_class=d["trajectory_class"],
                    fold_change_note=("undefined_initial_near_zero" if x0 < 1e-6 else "not_reported_here")))
    return pd.DataFrame(rows)


def low_initial_audit_date_level(run_df):
    g = (run_df.groupby(["ph", "date", "channel"]).agg(
        n_reps=("rep", "nunique"), initial_med=("initial_ref_adjusted", "median"),
        max_post_med=("max_post_initial", "median"), t_max_med=("t_max_min", "median"),
        final_med=("final_ref_adjusted", "median"), auc_post_med=("auc_post_initial", "median"),
        frac_reliable_med=("frac_frame_reliable", "median"),
        dominant_class=("trajectory_class", lambda s: s.value_counts().index[0])).reset_index())
    return g.round(4)


def low_initial_audit_summary(date_df):
    """Across-date reproducibility per (pH, channel): is a late-emerging low-angle signal present, is its
    peak time consistent, does it coincide with the ~8–9 min despike event, and does it differ by pH?"""
    rows = []
    for (ph, c), g in date_df.groupby(["ph", "channel"]):
        classes = list(g["dominant_class"])
        n_emerge = sum(cl == TC_EMERGE for cl in classes)
        tmax = g["t_max_med"].dropna()
        peak_iqr = float(tmax.quantile(.75) - tmax.quantile(.25)) if len(tmax) >= 2 else np.nan
        near_event = bool(len(tmax) and EVENT_WINDOW_MIN[0] <= float(tmax.median()) <= EVENT_WINDOW_MIN[1])
        reproducible = bool(n_emerge >= 2)                    # emergence dominant in ≥2 independent dates
        rows.append(dict(
            ph=ph, channel=c, n_dates=len(g), n_dates_late_emerging=n_emerge,
            reproducible_late_emerging_low_angle_signal=reproducible,
            peak_time_median_min=_r(float(tmax.median()) if len(tmax) else np.nan, 2),
            peak_time_iqr_min=_r(peak_iqr, 2), peak_coincides_with_8_9min_event=near_event,
            max_post_across_dates=_r(float(g["max_post_med"].median()), 3),
            dominant_class_across_dates=(pd.Series(classes).value_counts().index[0] if classes else "n/a")))
    return pd.DataFrame(rows)


def _low_initial_report(out_dir, run_df, date_df, summ):
    L = ["# Low-initial detector channels (1–4) — audit\n",
         "Channels 1–4 carry near-zero INITIAL reference-adjusted particle signal, so they cannot support "
         "within-channel normalization by their initial value. This is a normalization limitation, **not** "
         "a noise-QC failure. They are examined here in ABSOLUTE reference-adjusted units. Fold change is "
         "never computed when the initial value is zero/near-zero. No particle-size claim is made.\n",
         "## Across-date reproducibility summary\n",
         "```\n" + summ.to_string(index=False) + "\n```\n",
         "## Reading\n",
         "- A channel flagged `reproducible_late_emerging_low_angle_signal = True` develops a positive "
         "signal after starting empty, reproducibly across ≥ 2 independent preparation dates — described "
         "neutrally as a **late-emerging low-angle signal**, NOT as physical coarsening.\n",
         "- `peak_coincides_with_8_9min_event` flags whether the peak time falls in the 8–9 min window of "
         "the synchronized event already handled by the established despike QC (possible common-disturbance "
         "coincidence, reported without implying causation).\n",
         "- Date-level detail: `low_initial_channel_audit_date_level.csv`; per-run detail: "
         "`low_initial_channel_audit_run_level.csv`.\n"]
    (out_dir / "low_initial_channel_audit.md").write_text("\n".join(L))


# ── Part 4: correlations from cleaned reference-adjusted trajectories ──────────
def _corr(mat, method):
    df = pd.DataFrame(mat, columns=CHANNELS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return df.corr(method=method, min_periods=5).to_numpy()


def condition_correlation(runs, eligible, method="pearson", derivative=False, dates_required=2):
    """Date-balanced channel×channel correlation among the **cleaned reference-adjusted** trajectories
    (``G_X``) of the eligible channels — NOT the normalized ratios. Run-level r → Fisher z → mean within
    date → mean across dates → tanh. ``derivative`` correlates first differences (a sensitivity check,
    since correlations among monotonic trajectories are inflated). Never concatenates runs."""
    keep = np.array([c - 1 for c in eligible], int)
    by_date = defaultdict(list)
    for r in runs:
        M = np.diff(r["G_X"], axis=0) if derivative else r["G_X"]
        by_date[r["date"]].append(np.arctanh(np.clip(_corr(M, method), -0.9999, 0.9999)))
    if len(by_date) < dates_required:
        return dict(r=np.full((N_CH, N_CH), np.nan), n_dates=0, median_offdiag=np.nan)
    date_z = [_nanmean(np.stack(v), axis=0) for v in by_date.values()]
    r_full = np.tanh(_nanmean(np.stack(date_z), axis=0))
    r_kept = np.full((N_CH, N_CH), np.nan)
    if keep.size:
        r_kept[np.ix_(keep, keep)] = r_full[np.ix_(keep, keep)]
    off = r_kept[np.ix_(keep, keep)][~np.eye(keep.size, dtype=bool)] if keep.size > 1 else np.array([np.nan])
    return dict(r=r_kept, n_dates=len(by_date), median_offdiag=float(np.nanmedian(off)))


def channel_vs_total(runs, eligible, dates_required=2):
    """Date-balanced correlation of each eligible channel's reference-adjusted trajectory with the TOTAL
    reference-adjusted angular signal (summed over eligible channels), Fisher-z aggregated."""
    keep = np.array([c - 1 for c in eligible], int)
    by_date = defaultdict(list)
    for r in runs:
        Gx = r["G_X"][:, keep]
        total = np.nansum(Gx, axis=1)
        z = []
        for j in range(keep.size):
            x = Gx[:, j]; m = np.isfinite(x) & np.isfinite(total)
            if m.sum() >= 5 and np.std(x[m]) > 0 and np.std(total[m]) > 0:
                z.append(np.arctanh(np.clip(np.corrcoef(x[m], total[m])[0, 1], -0.9999, 0.9999)))
            else:
                z.append(np.nan)
        by_date[r["date"]].append(np.array(z))
    if len(by_date) < dates_required:
        return pd.DataFrame()
    date_z = [_nanmean(np.stack(v), axis=0) for v in by_date.values()]
    r = np.tanh(_nanmean(np.stack(date_z), axis=0))
    return pd.DataFrame(dict(channel=eligible, r_with_total_signal=np.round(r, 3)))


# ── Part 5: per-channel t50 rate summary ──────────────────────────────────────
def _t50(x, t, x0):
    """First time the reference-adjusted signal drops to 50 % of its initial value (linear interpolation).
    Returns NaN if it never crosses (kept missing — never extrapolated to a long fitted time)."""
    x = np.asarray(x, float); t = np.asarray(t, float)
    m = np.isfinite(x) & np.isfinite(t)
    x, t = x[m], t[m]
    if x.size < 2 or not np.isfinite(x0) or x0 <= 0:
        return np.nan
    target = 0.5 * x0
    below = np.where(x <= target)[0]
    if not below.size:
        return np.nan                                          # never crosses 50 %
    k = below[0]
    if k == 0:
        return float(t[0])
    x1, x2, t1, t2 = x[k - 1], x[k], t[k - 1], t[k]
    if x1 == x2:
        return float(t2)
    return float(t1 + (target - x1) * (t2 - t1) / (x2 - x1))


def t50_summary(runs_by_ph, eligible):
    """Per-run per-eligible-channel t50 → date-first summary by condition. pH 5.0 handled cautiously: a
    channel that never crosses 50 % is kept as non-estimable (missing), NOT read as a very long time."""
    run_rows, cond_rows = [], []
    for ph in CONDITIONS:
        per_run_med, never_cross, by_date = [], 0, defaultdict(list)
        for r in runs_by_ph[ph]:
            vals = []
            for c in eligible:
                ci = c - 1
                tt = _t50(r["G_X"][:, ci], np.asarray(mqs._common_time_grid()), r["X0"][ci])
                run_rows.append(dict(ph=ph, date=r["date"], rep=r["rep"], channel=c,
                                     t50_min=_r(tt, 3), estimable=bool(np.isfinite(tt))))
                if np.isfinite(tt):
                    vals.append(tt)
                else:
                    never_cross += 1
            if vals:
                per_run_med.append(np.median(vals)); by_date[r["date"]].append(np.median(vals))
        est = [v for v in per_run_med if np.isfinite(v)]
        date_meds = [np.median(v) for v in by_date.values() if v]
        cond_rows.append(dict(
            ph=ph, n_eligible_channels=len(eligible),
            n_channel_runs_with_t50=int(sum(np.isfinite([rr["t50_min"] for rr in run_rows
                                                         if rr["ph"] == ph]))),
            n_channel_runs_never_crossed=never_cross,
            t50_median_min=_r(float(np.median(est)) if est else np.nan, 3),
            t50_iqr_min=_r(float(np.percentile(est, 75) - np.percentile(est, 25)) if len(est) >= 2 else np.nan, 3),
            t50_range_min=(f"{min(est):.2f}–{max(est):.2f}" if est else "n/a"),
            between_date_sd_min=_r(float(np.std(date_meds, ddof=1)) if len(date_meds) >= 2 else np.nan, 3)))
    return pd.DataFrame(run_rows), pd.DataFrame(cond_rows)


# ── figures ────────────────────────────────────────────────────────────────────
def _color_limits(cond_by_ph, principal_hi=1.0):
    vals = np.concatenate([m["median"][~np.isnan(m["median"])].ravel() for m in cond_by_ph.values()]) \
        if cond_by_ph else np.array([1.0])
    vmax_data = float(np.nanmax(vals)) if vals.size else 1.0
    capped = vals[vals > principal_hi]
    return dict(vmin=0.0, vmax=principal_hi, vmax_data=round(vmax_data, 3), n_capped=int(capped.size),
                capped_range=(round(float(capped.min()), 3), round(float(capped.max()), 3)) if capped.size else None)


def figure_heatmaps(cond_by_ph, eligible, grid, clim, out_dir, formats):
    """Primary — 3 panels (pH 4.0/4.5/5.0), the SAME common eligible channel rows in each, one shared
    colour scale. Deliberately-excluded low-initial channels are simply not rows here (no gray block);
    genuinely missing/below-noise eligible cells are neutral gray. No directional strip, no size labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    ps.apply_manuscript_style()
    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(MISSING_COLOR); cmap.set_over(OVER_COLOR)
    norm = Normalize(clim["vmin"], clim["vmax"])
    lo, hi = eligible[0], eligible[-1]
    rows = np.array(eligible)

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.85, top=0.86, bottom=0.17, wspace=0.10)
    im = None
    for ax, ph in zip(axes, CONDITIONS):
        M = np.ma.masked_invalid(cond_by_ph[ph]["median"][:, rows - 1].T)   # (eligible channel, time)
        im = ax.pcolormesh(grid, rows, M, cmap=cmap, norm=norm, shading="nearest")
        ax.set_title(f"pH {ph}", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 20); ax.set_xlabel("Time (min)")
        ax.set_yticks([r for r in (lo, 10, 15, 20, 25, hi) if lo <= r <= hi])
        ps.setup_axes(ax)
    axes[0].set_ylabel(f"Detector channel ({lo}–{hi})")
    cax = fig.add_axes([0.865, 0.17, 0.02, 0.69])
    cb = fig.colorbar(im, cax=cax, extend=("max" if clim["n_capped"] else "neither"))
    cb.set_label("Channel intensity relative to initial value", fontsize=8.5)
    written = _save_multiformat(fig, out_dir / "pH_condition_level_channel_trajectory_heatmaps", formats)
    plt.close(fig)
    return written


def figure_low_initial_absolute(runs_by_ph, grid, out_dir, formats):
    """Part 3 — channels 1–4 in ABSOLUTE reference-adjusted intensity (not normalized). One panel per pH;
    thin date-level medians + a heavy across-date median per channel; despike/event times marked (no
    causation implied). Answers: flat near reference, or a reproducible late-emerging low-angle signal?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ps.apply_manuscript_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1), sharex=True)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.84, bottom=0.24, wspace=0.24)
    # y-limit must bound EVERY plotted trajectory, not just the heavy across-date medians: collect the
    # date-level (thin) AND across-date-median (heavy) curves as they are drawn, then size the shared axis.
    max_datelevel = 0.0     # heaviest thin date-level trajectory (was excluded from ymax → clipped)
    max_cross = 0.0         # heaviest across-date-median trajectory
    plotted = []            # every finite plotted curve, for the clipped-value count
    for ax, ph in zip(axes, CONDITIONS):
        runs = runs_by_ph[ph]
        ev = sorted({s for r in runs for s in r["event_spikes"]})
        for c in LOW_INIT_CHANNELS:
            ci = c - 1; col = LOW_CH_COLORS[c]
            by_date = defaultdict(list)
            for r in runs:
                by_date[r["date"]].append(r["G_X"][:, ci])
            date_meds = []
            for d, mats in by_date.items():
                dm = _nanmean(np.stack(mats), axis=0)
                ax.plot(grid, dm, "-", color=col, lw=0.5, alpha=0.30)      # thin date-level
                date_meds.append(dm)
                plotted.append(dm)
                if np.isfinite(dm).any():
                    max_datelevel = max(max_datelevel, float(np.nanmax(dm)))
            if len(date_meds) >= 2:
                cross = _nanmedian(np.stack(date_meds), axis=0)
                ax.plot(grid, cross, "-", color=col, lw=1.8, alpha=0.95)   # heavy across-date median
                plotted.append(cross)
                if np.isfinite(cross).any():
                    max_cross = max(max_cross, float(np.nanmax(cross)))
        for s in ev:
            ax.axvline(s, color="0.5", lw=0.6, ls=":", alpha=0.7)
        ax.set_title(f"pH {ph}", fontsize=10, fontweight="bold"); ax.set_xlim(0, 20)
        ax.set_xlabel("Time (min)"); ps.setup_axes(ax); ax.grid(True, lw=0.3, alpha=0.22)
    # shared ceiling over ALL plotted (thin + heavy, every channel, every pH); zero lower bound retained
    ymax = max(max_datelevel, max_cross)
    upper = max(_nice_ceiling(ymax), 0.1)
    n_clipped = int(sum(int(np.sum(np.asarray(t, float)[np.isfinite(t)] > upper)) for t in plotted))
    assert n_clipped == 0 and (ymax == 0.0 or ymax < upper), \
        f"y-limit {upper} does not bound every plotted value (max {ymax}, {n_clipped} clipped)"
    for ax in axes:
        ax.set_ylim(0, upper)                                 # shared y (absolute reference-adjusted units)
    axes[0].set_ylabel("Reference-adjusted\nintensity (abs. units)")
    handles = [Line2D([0], [0], color=LOW_CH_COLORS[c], lw=2, label=f"ch {c}") for c in LOW_INIT_CHANNELS]
    handles += [Line2D([0], [0], color="0.5", lw=1, ls=":", label="despike/event")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.54, -0.01))
    fig.suptitle("Low-initial channels 1–4 — absolute reference-adjusted trajectories "
                 "(thin = date medians, heavy = across-date median)", fontsize=8.5)
    written = _save_multiformat(fig, out_dir / "FigureS_low_initial_channels_absolute_trajectories", formats)
    plt.close(fig)
    validation = dict(max_datelevel=round(max_datelevel, 4), max_cross=round(max_cross, 4),
                      max_plotted=round(ymax, 4), y_lower=0.0, y_upper=round(float(upper), 4),
                      n_clipped=n_clipped)
    return written, validation


def figure_correlation(corr_by_ph, eligible, out_dir, formats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    ps.apply_manuscript_style()
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad(MISSING_COLOR)
    norm = Normalize(-1, 1)
    lo, hi = eligible[0], eligible[-1]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.82, bottom=0.15, wspace=0.12)
    im = None
    for ax, ph in zip(axes, CONDITIONS):
        M = np.ma.masked_invalid(corr_by_ph[ph]["r"][np.ix_(np.array(eligible) - 1, np.array(eligible) - 1)])
        im = ax.pcolormesh(eligible, eligible, M, cmap=cmap, norm=norm, shading="nearest")
        ax.set_title(f"pH {ph}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Channel"); ax.set_aspect("equal")
        ax.set_xticks([lo, 15, 20, 25, hi]); ax.set_yticks([lo, 15, 20, 25, hi])
    axes[0].set_ylabel("Channel")
    cax = fig.add_axes([0.895, 0.15, 0.02, 0.67])
    cb = fig.colorbar(im, cax=cax); cb.set_label("Pearson r  (date-balanced, Fisher-z)")
    fig.suptitle(f"Correlation of cleaned reference-adjusted channel trajectories, channels {lo}–{hi} "
                 f"(descriptive)", fontsize=8.5)
    written = _save_multiformat(fig, out_dir / "FigureS_channel_trajectory_correlation_matrices", formats)
    plt.close(fig)
    return written


def _save_multiformat(fig, stem, formats):
    stem = Path(stem); stem.parent.mkdir(parents=True, exist_ok=True)
    dpi = {"png": 300, "pdf": ps.DPI, "svg": ps.DPI, "tif": 600, "tiff": 600}
    out = []
    for fmt in formats:
        ext = "tiff" if fmt in ("tif", "tiff") else fmt
        p = stem.with_suffix(f".{ext}")
        kw = dict(dpi=dpi.get(fmt, ps.DPI), bbox_inches="tight")
        if ext == "tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(p, **kw); out.append(p)
    return out


def _r(x, n=4):
    return round(float(x), n) if x is not None and np.isfinite(x) else np.nan


# ── source data + summaries + audit ───────────────────────────────────────────
def source_table(runs_by_ph, cond_by_ph, eligible, grid):
    """Tidy long source data. Run-level rows carry raw + cleaned + reference-adjusted intensity, the
    initial value, the normalized value (eligible channels only; NaN otherwise), the eligibility flag, the
    directional-admission and frame-reliability status, and the final display status — so the displayed
    condition matrices are reproducible. Date / cross-date-median levels carry the aggregated normalized
    value + counts."""
    elig = set(eligible)
    rows = []
    for ph, runs in runs_by_ph.items():
        for r in runs:
            directional = set(r["directional"])
            Gn = _normalized_G(r, eligible)
            for gi, t in enumerate(grid):
                for ci, c in enumerate(CHANNELS):
                    y = Gn[gi, ci]
                    frel = bool(r["G_frel"][gi, ci])
                    if c not in elig:
                        disp = "excluded_low_initial" if c in LOW_INIT_CHANNELS else "excluded_not_eligible"
                    elif np.isfinite(y):
                        disp = ST_DIR if c in directional else ST_NODIR
                    else:
                        disp = ST_NOISE if not frel else ST_MISSING
                    rows.append(dict(
                        ph=ph, date=r["date"], rep=r["rep"], channel=c, time_min=round(float(t), 3),
                        raw_intensity=_r(r["G_raw"][gi, ci]), cleaned_intensity=_r(r["G_clean"][gi, ci]),
                        ref_adjusted_X=_r(r["G_X"][gi, ci]), X0_initial=round(float(r["X0"][ci]), 4),
                        Y_normalized=_r(y, 5), eligible=bool(c in elig),
                        directional_admission=("admitted" if c in directional else "not_admitted"),
                        frame_reliability=("reliable" if frel else "below_noise_or_unaligned"),
                        display_status=disp, displayed=bool(c in elig and np.isfinite(y)),
                        aggregation_level="run", n_runs=1, n_dates=1))
    for ph, runs in runs_by_ph.items():
        agg = cond_by_ph[ph]
        for d, mat in agg["date_mats"].items():
            for gi, t in enumerate(grid):
                for ci, c in enumerate(CHANNELS):
                    v = mat[gi, ci]
                    if np.isfinite(v):
                        rows.append(_agg_row(ph, d, np.nan, c, t, v, "date_mean", "date",
                                             int(agg["n_runs"][gi, ci]), 1))
        for gi, t in enumerate(grid):
            for ci, c in enumerate(CHANNELS):
                v = agg["median"][gi, ci]
                if np.isfinite(v):
                    rows.append(_agg_row(ph, np.nan, np.nan, c, t, v, "cross_date_median", "cross_date_median",
                                         int(agg["n_runs"][gi, ci]), int(agg["n_dates"][gi, ci])))
    return pd.DataFrame(rows)


def _agg_row(ph, date, rep, c, t, v, disp, level, nr, nd):
    return dict(ph=ph, date=date, rep=rep, channel=c, time_min=round(float(t), 3), raw_intensity=np.nan,
                cleaned_intensity=np.nan, ref_adjusted_X=np.nan, X0_initial=np.nan,
                Y_normalized=round(float(v), 5), eligible=True, directional_admission="",
                frame_reliability="", display_status=disp, displayed=True, aggregation_level=level,
                n_runs=nr, n_dates=nd)


def noise_filter_audit(runs_by_ph, legacy_by_run):
    rows = []
    for ph in CONDITIONS:
        for r in runs_by_ph[ph]:
            leg, _leg_surface, _leg_spikes = legacy_by_run[(ph, r["date"], r["rep"])]
            directional = set(r["directional"])
            rows.append(dict(
                ph=ph, date=r["date"], rep=r["rep"],
                legacy_static_admitted=len(leg), calibrated_directional_admitted=len(directional),
                n_channels_changed=len(leg ^ directional), changed_channels=",".join(map(str, sorted(leg ^ directional))) or "none",
                frac_cells_frame_reliable=round(r["frame_reliable_frac"], 4),
                noise_surface_populated=bool(r["noise_surface_params"] is not None),
                despike_frames=",".join(map(str, r["spike_frames"])) or "none",
                copt_used_for_despike_only="yes (despike corroboration only; no Copt eligibility cutoff)"))
    return pd.DataFrame(rows)


def condition_summary(runs_by_ph, cond_by_ph, corr_by_ph, corr_spearman, corr_deriv, t50_cond,
                      eligible, grid):
    rows = []
    for ph in CONDITIONS:
        runs = runs_by_ph[ph]; agg = cond_by_ph[ph]
        med = agg["median"][:, np.array(eligible) - 1]
        finals = [float(med[np.isfinite(med[:, j]), j][-1]) if np.isfinite(med[:, j]).any() else np.nan
                  for j in range(len(eligible))]
        t50r = t50_cond[t50_cond.ph == ph].iloc[0]
        rows.append(dict(
            ph=ph, n_runs=len(runs), n_dates=len(agg["dates"]),
            n_eligible_channels=len(eligible),
            directional_admitted_median=int(np.median([r["n_directional"] for r in runs])),
            median_pearson_r=_r(corr_by_ph[ph]["median_offdiag"], 3),
            median_spearman_r=_r(corr_spearman[ph]["median_offdiag"], 3),
            median_pearson_r_first_diff=_r(corr_deriv[ph]["median_offdiag"], 3),
            channelwise_final_relative_median=_r(float(np.nanmedian(finals)), 3),
            channelwise_final_relative_min=_r(float(np.nanmin(finals)), 3),
            drawable_cell_coverage=round(float(np.mean(np.isfinite(med))), 3),
            t50_median_min=t50r.t50_median_min, t50_iqr_min=t50r.t50_iqr_min,
            n_channel_runs_never_crossed_50pct=int(t50r.n_channel_runs_never_crossed),
            noise_surface_populated_all=bool(all(r["noise_surface_params"] is not None for r in runs)),
            event_8_9min_despike_frames=(", ".join(f"{s:.1f}" for s in sorted({s for r in runs for s in r["event_spikes"]}))
                                         or "none")))
    return pd.DataFrame(rows)


# ── report ────────────────────────────────────────────────────────────────────
def _report(out_dir, elig_info, summary, audit, low_summary, corr_pearson, corr_spear, corr_deriv,
            ch_total_by_ph, t50_cond, clim, surface_meta, low_yvalid=None):
    audit_cond = (audit.groupby("ph")[["legacy_static_admitted", "calibrated_directional_admitted",
                  "n_channels_changed", "frac_cells_frame_reliable"]].mean().round(2).reset_index())
    L = ["# Detector-channel trajectory analysis — report (revised)\n",
         "Descriptive analysis of whether detector channels decay in a coordinated way and on similar "
         "timescales. Calibrated CFZ-pH-7 noise surface used for admission + reliability (production "
         "`noise_filter`/`noise_surface` unchanged). NO Mie / q3 / UV / forward-model / particle-size "
         "claim; `copt` reaches `noise_filter` only for despike corroboration (no Copt cutoff).\n",
         "## Common eligible channel set (Part 2 rule)\n",
         f"- **Rule:** {elig_info['rule']}\n",
         f"- Per-condition eligible: " + "; ".join(f"pH {ph}: {_rng(elig_info['per_condition'][ph])}"
                                                   for ph in CONDITIONS) + "\n",
         f"- **Common set (all panels): channels {_rng(elig_info['common'])}** "
         f"({len(elig_info['common'])} channels). Channels 1–4 are excluded as low-initial (audited "
         f"separately); channel 5 is a borderline exclusion (reliable-initial in only 1 of 3 pH-4.0 dates).\n",
         "## Before/after QC audit (mean by condition; full per-run CSV: channel_heatmap_noise_filter_audit.csv)\n",
         "```\n" + audit_cond.to_string(index=False) + "\n```\n",
         f"- `frac_cells_frame_reliable` ≈ {audit['frac_cells_frame_reliable'].mean():.3f}: raw intensity "
         "stays above its calibrated noise floor throughout (it decays toward the baseline, not zero), so "
         "`Y→0` is real signal loss — disappearance into channel noise is NOT equated with complete "
         "dissolution. `noise_surface` populated for every run.\n",
         "## Part 1 — what channels 1–4 actually do (across dates)\n",
         "```\n" + low_summary.to_string(index=False) + "\n```\n",
         "- Channels flagged `reproducible_late_emerging_low_angle_signal = True` start empty and develop "
         "a positive low-angle signal reproducibly across ≥ 2 dates — described as a **late-emerging "
         "low-angle signal**, not physical coarsening. Whether the peak coincides with the ~8–9 min "
         "despike/common-event window is flagged (no causation implied). See "
         "`FigureS_low_initial_channels_absolute_trajectories.*` (absolute units).\n",
         (("- **`FigureS_low_initial_channels_absolute_trajectories` y-axis (shared, zero-based, "
           "no clipping):** the shared upper limit is sized from EVERY finite plotted trajectory — the thin "
           "date-level curves AND the heavy across-date medians, all channels 1–4, all three pH. Maximum "
           f"plotted date-level value = {low_yvalid['max_datelevel']}; maximum plotted across-date-median "
           f"value = {low_yvalid['max_cross']}; overall maximum plotted = {low_yvalid['max_plotted']} "
           "(absolute units). Final shared y-axis range = "
           f"[{low_yvalid['y_lower']}, {low_yvalid['y_upper']}] (data-driven nice ceiling, all three panels "
           f"identical). Clipped finite plotted values = **{low_yvalid['n_clipped']}** (verified zero: "
           "the previous ~1.6 limit, sized from the medians only, clipped the taller date-level curves).\n")
          if low_yvalid else ""),
         "## Part 4 — correlations (from cleaned reference-adjusted trajectories, NOT normalized ratios)\n",
         "```\n" + summary[["ph", "median_pearson_r", "median_spearman_r",
                            "median_pearson_r_first_diff"]].to_string(index=False) + "\n```\n",
         "- Pearson does not require prior amplitude normalization. The first-difference column is a "
         "sensitivity check: correlations among monotonic trajectories are inflated, so the drop from the "
         "level to the first-difference correlation bounds how much of the apparent coordination is merely "
         "shared monotonic decay. Each channel's correlation with the total reference-adjusted signal is in "
         "`channel_vs_total_signal_correlation.csv`. **High correlation is coordinated decay, NOT identical "
         "decay rates.**\n",
         "## Part 5 — channel-specific decay timescale (t50)\n",
         "```\n" + t50_cond.to_string(index=False) + "\n```\n",
         "- `t50` = time to 50 % of the initial reference-adjusted signal (robust, interpolated). Channels "
         "that never cross 50 % (notably at pH 5.0, whose limited decay makes `t50` largely non-estimable) "
         "are kept **missing** — NOT read as a very long fitted dissolution time. A tight `t50` "
         "distribution supports *similar channel-specific decay timescales*; coordinated behaviour, "
         "similar shape, and similar timescale are distinct claims and reported separately.\n",
         "## Interpretation limits\n",
         "- Supported (as warranted by the numbers above): **coordinated detector-channel decay** (high "
         "trajectory correlation) and, where the t50 distribution is tight, **similar channel-specific "
         "decay timescales**; **late-emerging low-angle signal** for channels 1–4 where reproducible. NOT "
         "claimed: one detector channel = one particle size; identical particle-size dissolution rates from "
         "correlation alone; noise-disappearance = complete dissolution.\n",
         "## Output paths\n",
         "- Figures: `pH_condition_level_channel_trajectory_heatmaps.*`, "
         "`FigureS_low_initial_channels_absolute_trajectories.*`, "
         "`FigureS_channel_trajectory_correlation_matrices.*`, "
         "`FigureS_per_channel_trajectories_pH4p5_20260608_R2.*`.\n",
         "- Tables: `channel_trajectory_heatmaps_source.csv`, `channel_trajectory_condition_summary.csv`, "
         "`channel_heatmap_noise_filter_audit.csv`, `low_initial_channel_audit_run_level.csv`, "
         "`low_initial_channel_audit_date_level.csv`, `low_initial_channel_audit_summary.csv`, "
         "`channel_vs_total_signal_correlation.csv`, `channel_t50_run_level.csv`, "
         "`channel_t50_condition_summary.csv`; reports `low_initial_channel_audit.md`, this file.\n",
         "## Remaining interpretation concerns\n",
         f"- Colour scale vmin 0 → vmax {clim['vmax']}; data max {clim['vmax_data']}; {clim['n_capped']} "
         f"over-range eligible cell(s) (channel-1's anomalous floor-bin behaviour is now OUTSIDE the "
         f"eligible set, so it no longer appears).\n",
         "- The late-emerging low-angle signal in channel 1 near the 8–9 min window may be a common "
         "disturbance rather than particle behaviour; it is reported neutrally and left for the raw-channel "
         "/ redistribution analyses to resolve. Correlation cannot establish equal size-dependent rates.\n"]
    (out_dir / "channel_trajectory_heatmaps_report.md").write_text("\n".join(L))


def _rng(chs):
    if not chs:
        return "none"
    chs = sorted(chs)
    return f"{chs[0]}–{chs[-1]}" if chs == list(range(chs[0], chs[-1] + 1)) else ",".join(map(str, chs))


# ── per-channel small multiples for the illustrative run (retained diagnostic) ─
def figure_per_channel(run, eligible, out_dir, formats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ps.apply_manuscript_style()
    t = run["t"]; X = run["X"]; X0 = run["X0"]
    directional = set(run["directional"]); norm_ok = run["norm_reliable"]; elig = set(eligible)
    spike_t = set(round(s, 2) for s in run["spike_times"])
    fig, axes = plt.subplots(5, 7, figsize=(7.4, 6.4), sharex=True)
    axes = axes.ravel()
    for ci, c in enumerate(CHANNELS):
        ax = axes[ci]
        if c in elig and norm_ok[ci] and X0[ci] > 0:
            Y = X[:, ci] / X0[ci]
            col = "#0072B2" if c in directional else "#009E73"
            ax.axhline(1.0, color="0.6", lw=0.6, ls=":")
            ax.plot(t, Y, "-", color=col, lw=1.0)
            for st in spike_t:
                if t.min() <= st <= t.max():
                    ax.axvline(st, color="#D55E00", lw=0.5, alpha=0.5)
            ax.set_ylim(0, 1.3)
            if c not in directional:
                ax.text(0.04, 0.06, "no sig. trend", transform=ax.transAxes, fontsize=5.0, color="#007a55")
        else:
            ax.set_facecolor("0.94"); ax.set_yticks([])
            tag = ("low-initial\n(see abs. fig.)" if c in LOW_INIT_CHANNELS else "not eligible")
            ax.text(0.5, 0.5, tag, transform=ax.transAxes, ha="center", va="center", fontsize=5.5, color="0.45")
        ax.set_title(f"ch {c}", fontsize=6.5, pad=1.5); ax.tick_params(labelsize=6)
        if ci >= 28:
            ax.set_xlabel("min", fontsize=7)
    for k in range(N_CH, len(axes)):
        axes[k].axis("off")
    fig.legend([Line2D([0], [0], color="#0072B2", lw=2), Line2D([0], [0], color="#009E73", lw=2)],
               ["directional change detected", "reliable, no significant trend"], loc="lower center",
               ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.55, -0.01))
    fig.suptitle(f"pH {run['ph']}  {run['date']}  Rep {run['rep']} — per-channel relative trajectories "
                 f"(eligible channels normalized; low-initial shown in the absolute figure)", fontsize=8.5)
    fig.supylabel("Channel intensity relative to initial value", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.09, hspace=0.5, wspace=0.35)
    written = _save_multiformat(fig, out_dir / "FigureS_per_channel_trajectories_pH4p5_20260608_R2", formats)
    plt.close(fig)
    return written


# ── driver ────────────────────────────────────────────────────────────────────
def run(output_dir, formats=("pdf", "png", "svg", "tiff")):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    grid = mqs._common_time_grid()
    surface = load_surface()

    runs_by_ph = defaultdict(list)
    legacy_by_run = {}
    illustrative = None
    for ph, date, rep, rtf, fo in iter_runs():
        if float(ph) not in CONDITIONS:
            continue
        r = build_run(rtf, ph, date, rep, grid, surface)
        runs_by_ph[float(ph)].append(r)
        legacy_by_run[(float(ph), int(date), int(rep))] = legacy_admitted(rtf)
        if (float(ph), int(date), int(rep)) == ILLUSTRATIVE:
            illustrative = r

    elig_info = common_eligible_channels(runs_by_ph)
    eligible = elig_info["common"]
    cond_by_ph = {ph: aggregate_condition(runs_by_ph[ph], eligible, grid) for ph in CONDITIONS}
    clim = _color_limits(cond_by_ph)
    corr_pearson = {ph: condition_correlation(runs_by_ph[ph], eligible, "pearson") for ph in CONDITIONS}
    corr_spear = {ph: condition_correlation(runs_by_ph[ph], eligible, "spearman") for ph in CONDITIONS}
    corr_deriv = {ph: condition_correlation(runs_by_ph[ph], eligible, "pearson", derivative=True) for ph in CONDITIONS}
    ch_total_by_ph = {ph: channel_vs_total(runs_by_ph[ph], eligible) for ph in CONDITIONS}
    t50_run, t50_cond = t50_summary(runs_by_ph, eligible)

    # Part 1 audit
    low_run = low_initial_audit_run_level(runs_by_ph)
    low_date = low_initial_audit_date_level(low_run)
    low_summary = low_initial_audit_summary(low_date)
    low_run.to_csv(output_dir / "low_initial_channel_audit_run_level.csv", index=False)
    low_date.to_csv(output_dir / "low_initial_channel_audit_date_level.csv", index=False)
    low_summary.to_csv(output_dir / "low_initial_channel_audit_summary.csv", index=False)
    _low_initial_report(output_dir, low_run, low_date, low_summary)

    audit = noise_filter_audit(runs_by_ph, legacy_by_run)
    summary = condition_summary(runs_by_ph, cond_by_ph, corr_pearson, corr_spear, corr_deriv, t50_cond,
                                eligible, grid)
    source_table(runs_by_ph, cond_by_ph, eligible, grid).to_csv(
        output_dir / "channel_trajectory_heatmaps_source.csv", index=False)
    summary.to_csv(output_dir / "channel_trajectory_condition_summary.csv", index=False)
    audit.to_csv(output_dir / "channel_heatmap_noise_filter_audit.csv", index=False)
    pd.concat([ch_total_by_ph[ph].assign(ph=ph) for ph in CONDITIONS if len(ch_total_by_ph[ph])],
              ignore_index=True).to_csv(output_dir / "channel_vs_total_signal_correlation.csv", index=False)
    t50_run.to_csv(output_dir / "channel_t50_run_level.csv", index=False)
    t50_cond.to_csv(output_dir / "channel_t50_condition_summary.csv", index=False)

    figs = {}
    figs["heatmaps"] = figure_heatmaps(cond_by_ph, eligible, grid, clim, output_dir, formats)
    figs["low_initial"], low_yvalid = figure_low_initial_absolute(runs_by_ph, grid, output_dir, formats)
    figs["correlation"] = figure_correlation(corr_pearson, eligible, output_dir, formats)
    if illustrative is not None:
        figs["per_channel"] = figure_per_channel(illustrative, eligible, output_dir, formats)
    surface_meta = dict(surface.meta) if hasattr(surface, "meta") else {}
    _report(output_dir, elig_info, summary, audit, low_summary, corr_pearson, corr_spear, corr_deriv,
            ch_total_by_ph, t50_cond, clim, surface_meta, low_yvalid)
    return dict(runs_by_ph=runs_by_ph, eligible=eligible, elig_info=elig_info, cond_by_ph=cond_by_ph,
                corr_pearson=corr_pearson, t50_cond=t50_cond, low_summary=low_summary, audit=audit,
                summary=summary, clim=clim, grid=grid, figures=figs, low_initial_yaxis=low_yvalid)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--formats", default="pdf,png,svg,tiff")
    args = ap.parse_args(argv)
    formats = tuple(s.strip() for s in args.formats.split(",") if s.strip())
    res = run(args.output_dir, formats)
    print("common eligible channels:", _rng(res["eligible"]))
    print("figures:")
    for k, paths in res["figures"].items():
        print(f"  [{k}] " + "  ".join(p.name for p in paths))
    print("\nlow-initial reproducible late-emerging signal:")
    print(res["low_summary"][["ph", "channel", "reproducible_late_emerging_low_angle_signal",
          "peak_time_median_min", "peak_coincides_with_8_9min_event"]].to_string(index=False))
    print("\nt50 by condition:")
    print(res["t50_cond"][["ph", "t50_median_min", "t50_iqr_min", "n_channel_runs_never_crossed"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
