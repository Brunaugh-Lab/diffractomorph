"""Manuscript figure — PAQXOS q3 evolution vs measured angular-scattering evolution (pH study).

A 3 × 2 comparison (rows = pH 4.0 / 4.5 / 5.0; left = PAQXOS differential q3 size composition, right =
normalized background-subtracted angular scattering) over every ~12 s timepoint through 20 min, showing
side by side that:

* the PAQXOS **q3** distribution is the *relative* particle-size composition of the population still
  present (each frame's differential distribution sums to 1 — it is NOT a mass measurement); while
* the measured **angular scattering**, normalized once by each run's initial total, retains BOTH the
  channel redistribution AND the disappearance of total particulate signal (later curves lose area).

**Eligibility comes from the corrected three-day q3 reliability audit**, not the retired
``0.30 × run-peak Copt`` cutoff. Each paired q3 frame is classified ``supported`` (Copt ≥ 4, no
coarse-tail flag), ``provisionally_supported`` (0.79 ≤ Copt < 4, no flag), ``review_required`` (a
coarse-tail flag — > 1 % q3 volume above 100 µm, via the shared audit utility; NOT called an inversion
failure, because detector support was mixed/mostly unresolved), ``outside_validation_range`` (Copt <
0.79), or ``pairing_or_qc_failure`` (timestamp-match failure, synchronized detector glitch, or nonfinite
/ malformed distribution). Three otherwise-identical figure versions differ only in which q3 categories
enter aggregation (inclusive / coarse-flag-excluded / stringent Copt ≥ 4); scattering QC is evaluated
**independently** so q3 eligibility never removes an otherwise-valid scattering frame.

Aggregation is **date-first**: eligible intraday replicate distributions are averaged within a
preparation date, then the date means are averaged with equal weight; a condition curve is drawn only
where **≥ 2 dates** contribute (a one-date summary is never shown as a condition mean).

This is figure generation + sensitivity analysis only. It reuses the established ingestion,
reference-subtraction, q3 parsing, timestamp-matching, and despiking code and the audit's coarse-tail
utility; it does not change production QC, ``frame_mask``, the forward model, or the Mie operator.

Run with the pipeline venv::

    python analysis/manuscript_q3_scattering_evolution.py \
        --output-dir <…/figures_and_tables> [--formats png,pdf,svg,tiff]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from diffractomorph_pipeline import ingest, plot_styles as ps, psd
from diffractomorph_pipeline.noise_filter import despike_frames

import q3_noise_floor_validation as audit          # shared coarse-tail definition + event metadata
from psd_evolution_common import iter_runs

# ── acquisition / display constants ───────────────────────────────────────────
FRAME_S = psd.FRAME_S                 # ~12 s PAQXOS/LD cadence
DT_MIN = FRAME_S / 60.0               # common target-grid step (min)
TMAX_MIN = 20.0
TOL_MIN = (FRAME_S / 2.0) / 60.0      # nearest-frame match tolerance = half the cadence (no interpolation)
ANCHORS_MIN = [0.0, 2.0, 5.0, 10.0, 20.0]
CONDITIONS = [4.0, 4.5, 5.0]
CMAP = "viridis"                      # perceptually-uniform, colorblind-safe time colormap
Q3_XLIM = (0.7, 15.0)                 # established manuscript diameter range (display only; no renorm)
N_CHANNELS = 31

# eligibility tiers (from the corrected audit)
COPT_SUPPORTED = 4.0
COPT_MIN = 0.79
COARSE_COL, COARSE_THR = audit.EVENT_DEFS[audit.PRIMARY_EVENT]     # ("frac_gt_100um", 1.0)
COARSE_SIZE_UM = 100.0

# categories
CAT_SUPPORTED = "supported"
CAT_PROVISIONAL = "provisionally_supported"
CAT_REVIEW = "review_required"
CAT_OUTSIDE = "outside_validation_range"
CAT_FAIL = "pairing_or_qc_failure"

# which q3 categories enter each version's aggregation
VERSIONS = {
    "inclusive": {CAT_SUPPORTED, CAT_PROVISIONAL, CAT_REVIEW},
    "coarse_flag_excluded": {CAT_SUPPORTED, CAT_PROVISIONAL},
    "stringent_copt4": {CAT_SUPPORTED},
}
VERSION_STEM = {
    "inclusive": "pH_q3_scattering_all_timepoints_inclusive",
    "coarse_flag_excluded": "pH_q3_scattering_all_timepoints_coarse_flag_excluded",
    "stringent_copt4": "pH_q3_scattering_all_timepoints_stringent_copt4",
}


def _common_time_grid() -> np.ndarray:
    n = int(round(TMAX_MIN / DT_MIN)) + 1
    return np.round(np.linspace(0.0, TMAX_MIN, n), 6)


# ── per-run ingestion, eligibility classification, and both observables ───────
def _first_stable_frame(run):
    """Index of the first stable retained detector frame (first not flagged a startup/synchronized
    glitch by the established despiker) and the glitch-frame set. 0 min is re-zeroed to this frame — it
    is the first analyzed post-stabilization measurement, NOT assumed to be the physical injection."""
    n = len(run.t_min)
    *_, info = despike_frames(np.asarray(run.I[:n], float), np.asarray(run.t_min[:n], float),
                              np.asarray(run.copt[:n], float))
    drop = set(int(k) for k in info.get("spike_frames", []))
    first = next((k for k in range(n) if k not in drop), 0)
    return first, drop


def _category(copt, coarse_flag, finite_ok, paired_ok, glitch):
    if not (paired_ok and finite_ok) or glitch:
        return CAT_FAIL
    if not np.isfinite(copt) or copt < COPT_MIN:
        return CAT_OUTSIDE
    if coarse_flag:
        return CAT_REVIEW
    if copt >= COPT_SUPPORTED:
        return CAT_SUPPORTED
    return CAT_PROVISIONAL


def build_run(ph, date, rep, rtf, fo, common_xo):
    """Parse one run and return per-frame q3 records (eligibility + differential distribution on the
    common native grid) and the independently-QC'd scattering series. q3 and detector frames are paired
    by timestamp; time is re-zeroed to the first stable detector frame."""
    run = ingest.extract_run(rtf)
    q = psd.read_q3_frames(fo)
    det_ts = np.array([(run.t0 + pd.Timedelta(minutes=float(x))).timestamp() for x in run.t_min], float)
    first, glitch = _first_stable_frame(run)
    t0 = float(run.t_min[first])
    pairs, _un_q3, _un_det = psd.match_frames_by_time(q.t_epoch, det_ts, tol_s=FRAME_S / 2.0)
    det_of = {i: (j, dt) for i, j, dt in pairs}

    iref = np.asarray(run.ref, float)
    q3_records = []
    for i in range(len(q.t_epoch)):
        paired = i in det_of
        j, dt = det_of.get(i, (None, np.nan))
        cum = np.asarray(q.Q3_cum[i], float)
        dq = np.asarray(q.dQ3[i], float)
        finite_ok = bool(np.all(np.isfinite(cum)) and np.all(np.isfinite(dq)) and abs(dq.sum() - 1.0) < 0.05)
        copt = float(run.copt[j]) if paired else np.nan
        elapsed = (float(run.t_min[j]) - t0) if paired else np.nan
        frac100 = float(psd.q3_tail_fraction(q.xo, cum, COARSE_SIZE_UM)) if finite_ok else np.nan
        coarse_flag = bool(finite_ok and frac100 > COARSE_THR)
        glitchy = bool(paired and j in glitch)
        before0 = bool(paired and j < first)
        cat = _category(copt, coarse_flag, finite_ok, paired, glitchy)
        # frames before the 0-min anchor are not plotted, but keep their category for the audit trail
        cum_grid = audit.base._cum_on_grid(q.xo, cum, common_xo) if finite_ok else None
        dq_grid = (np.diff(cum_grid, prepend=0.0) / 100.0) if cum_grid is not None else None
        q3_records.append(dict(
            ph=ph, date=date, rep=rep, q3_frame=i,
            q3_timestamp=pd.Timestamp(float(q.t_epoch[i]), unit="s").isoformat(),
            rtf_timestamp=(pd.Timestamp(float(det_ts[j]), unit="s").isoformat() if paired else ""),
            source_rtf_frame=(int(j) if paired else -1), dt_s=(round(float(dt), 3) if paired else np.nan),
            elapsed_min=(round(elapsed, 3) if paired else np.nan), measured_copt=(round(copt, 3) if paired else np.nan),
            frac_gt_100um=(round(frac100, 3) if np.isfinite(frac100) else np.nan),
            coarse_tail_flag=coarse_flag, before_zero=before0, category=cat,
            cum_grid=cum_grid, dq_grid=dq_grid))

    # scattering series (independent QC): drop startup/glitch frames, scale once by the initial total
    I_bg = np.clip(np.asarray(run.I, float) - iref[None, :], 0.0, None)
    init_total = float(I_bg[first].sum())
    sc_frames = []
    if init_total > 0:
        for j in range(len(run.t_min)):
            if j < first or j in glitch:
                continue
            vec = I_bg[j] / init_total
            if not np.all(np.isfinite(vec)):
                continue
            sc_frames.append(dict(elapsed_min=float(run.t_min[j]) - t0, source_rtf_frame=j,
                                  S=vec * 100.0))                    # % of initial total angular signal
    return dict(ph=ph, date=date, rep=rep, q3=q3_records, scattering=sc_frames,
                scatter_ok=bool(init_total > 0), first_frame=first, n_glitch=len(glitch))


# ── grid alignment + date-first equal-weight aggregation ──────────────────────
def _align(times, vectors, grid, tol=TOL_MIN):
    """Nearest-frame match of one run's (times, vectors) onto the common target ``grid`` within ``tol``
    (no extrapolation). Returns ``{grid_index: (vector, |dt|)}`` for matched grid points only."""
    times = np.asarray(times, float)
    out = {}
    if times.size == 0:
        return out
    for gi, gt in enumerate(grid):
        k = int(np.argmin(np.abs(times - gt)))
        if abs(times[k] - gt) <= tol + 1e-9:
            out[gi] = (np.asarray(vectors[k], float), abs(times[k] - gt))
    return out


def _date_first_mean(run_aligned, min_dates=2):
    """Date-first equal-weight condition mean. ``run_aligned`` = ``[(date, {gi: vector})]``. Per grid
    index: average replicate vectors within each date, then average the date means with equal weight.
    A grid index with fewer than ``min_dates`` contributing dates is dropped (never drawn as a mean)."""
    by_gi = defaultdict(lambda: defaultdict(list))                   # gi -> date -> [vectors]
    for date, aligned in run_aligned:
        for gi, vec in aligned.items():
            by_gi[gi][date].append(vec)
    out = {}
    for gi, by_date in by_gi.items():
        if len(by_date) < min_dates:
            continue
        date_means = [np.mean(np.vstack(vs), axis=0) for vs in by_date.values()]
        out[gi] = dict(mean=np.mean(np.vstack(date_means), axis=0), n_dates=len(by_date),
                       n_runs=sum(len(vs) for vs in by_date.values()))
    return out


def aggregate_q3(runs, grid, categories, min_dates=2):
    """Date-first condition mean of the differential q3 (fraction) at each target time, using only q3
    frames whose category is in ``categories`` (and that are at/after the 0-min anchor)."""
    run_aligned = []
    for run in runs:
        recs = [r for r in run["q3"] if r["category"] in categories and not r["before_zero"]
                and r["dq_grid"] is not None]
        if not recs:
            continue
        times = [r["elapsed_min"] for r in recs]
        vecs = [r["dq_grid"] for r in recs]
        run_aligned.append((run["date"], _align_pairs(times, vecs, grid)))
    return _date_first_mean(run_aligned, min_dates)


def aggregate_scattering(runs, grid, min_dates=2):
    """Date-first condition mean of the normalized scattering at each target time (independent QC)."""
    run_aligned = []
    for run in runs:
        if not run["scatter_ok"] or not run["scattering"]:
            continue
        times = [r["elapsed_min"] for r in run["scattering"]]
        vecs = [r["S"] for r in run["scattering"]]
        run_aligned.append((run["date"], _align_pairs(times, vecs, grid)))
    return _date_first_mean(run_aligned, min_dates)


def _align_pairs(times, vectors, grid, tol=TOL_MIN):
    """As :func:`_align` but returns ``{gi: vector}`` (drops the |dt|, kept only for coverage)."""
    return {gi: v for gi, (v, _dt) in _align(times, vectors, grid, tol).items()}


# ── coverage + sensitivity tables ─────────────────────────────────────────────
def per_frame_eligibility(runs):
    rows = []
    for run in runs:
        for r in run["q3"]:
            rows.append({k: r[k] for k in ("ph", "date", "rep", "q3_frame", "q3_timestamp",
                                           "rtf_timestamp", "source_rtf_frame", "dt_s", "elapsed_min",
                                           "measured_copt", "frac_gt_100um", "coarse_tail_flag",
                                           "before_zero", "category")})
    return pd.DataFrame(rows)


def run_summary(elig):
    """Frame counts by eligibility category, per pH / date / replicate."""
    if not len(elig):
        return pd.DataFrame()
    cats = [CAT_SUPPORTED, CAT_PROVISIONAL, CAT_REVIEW, CAT_OUTSIDE, CAT_FAIL]
    g = (elig.groupby(["ph", "date", "rep", "category"]).size().unstack("category", fill_value=0)
         .reindex(columns=cats, fill_value=0).reset_index())
    g["n_frames"] = g[cats].sum(axis=1)
    return g


def q3_coverage(runs, grid):
    """Per-timepoint contributing dates/runs for every version (a curve is drawn only where dates ≥ 2)."""
    rows = []
    for ph in CONDITIONS:
        ph_runs = [r for r in runs if r["ph"] == ph]
        for vname, cats in VERSIONS.items():
            agg = aggregate_q3(ph_runs, grid, cats, min_dates=1)     # min_dates=1 to report even 1-date coverage
            for gi in sorted(agg):
                rows.append(dict(ph=ph, version=vname, target_min=round(float(grid[gi]), 3),
                                 n_dates=agg[gi]["n_dates"], n_runs=agg[gi]["n_runs"],
                                 drawn=bool(agg[gi]["n_dates"] >= 2)))
    return pd.DataFrame(rows)


def scattering_coverage(runs, grid):
    rows = []
    for ph in CONDITIONS:
        ph_runs = [r for r in runs if r["ph"] == ph]
        agg = aggregate_scattering(ph_runs, grid, min_dates=1)
        for gi in sorted(agg):
            rows.append(dict(ph=ph, target_min=round(float(grid[gi]), 3), n_dates=agg[gi]["n_dates"],
                             n_runs=agg[gi]["n_runs"], drawn=bool(agg[gi]["n_dates"] >= 2)))
    return pd.DataFrame(rows)


def _cum_from_mean_dq(mean_dq):
    """Cumulative Q3 (%) from an aggregated differential fraction vector."""
    return np.cumsum(np.asarray(mean_dq, float)) * 100.0


def q3_sensitivity(runs, grid, common_xo):
    """Compare the three versions' aggregated q3 at every condition × target time: x10/x50/x90 per
    version, and pairwise (inclusive vs coarse-excluded; coarse-excluded vs stringent) log-diameter
    Wasserstein distance and max |Δ cumulative Q3|, with contributing run/date counts. Metrics are
    presented for review — no pass/fail threshold is applied."""
    aggs = {ph: {v: aggregate_q3([r for r in runs if r["ph"] == ph], grid, cats, min_dates=2)
                 for v, cats in VERSIONS.items()} for ph in CONDITIONS}
    rows = []
    for ph in CONDITIONS:
        A = aggs[ph]
        gis = set().union(*[set(A[v]) for v in VERSIONS])
        for gi in sorted(gis):
            rec = dict(ph=ph, target_min=round(float(grid[gi]), 3))
            cums = {}
            for v in VERSIONS:
                if gi in A[v]:
                    cum = _cum_from_mean_dq(A[v][gi]["mean"])
                    cums[v] = cum
                    x10, x50, x90 = psd.q3_percentiles(common_xo, cum)
                    rec[f"{v}_x10"] = round(x10, 3); rec[f"{v}_x50"] = round(x50, 3)
                    rec[f"{v}_x90"] = round(x90, 3)
                    rec[f"{v}_n_dates"] = A[v][gi]["n_dates"]; rec[f"{v}_n_runs"] = A[v][gi]["n_runs"]
            for a, b, tag in [("inclusive", "coarse_flag_excluded", "incl_vs_coarseexcl"),
                              ("coarse_flag_excluded", "stringent_copt4", "coarseexcl_vs_stringent")]:
                if a in cums and b in cums:
                    rec[f"{tag}_wasserstein_log"] = round(psd.q3_wasserstein_log(common_xo, cums[a], cums[b]), 4)
                    rec[f"{tag}_max_abs_dcumQ3"] = round(float(np.max(np.abs(cums[a] - cums[b]))), 3)
            rows.append(rec)
    return pd.DataFrame(rows)


def sensitivity_headline(sens):
    """Largest observed discrepancy per pH, and the target-time ranges with < 2 dates (per version)."""
    rows = []
    for ph in CONDITIONS:
        g = sens[sens.ph == ph]
        for tag in ("incl_vs_coarseexcl", "coarseexcl_vs_stringent"):
            col = f"{tag}_wasserstein_log"
            if col in g and g[col].notna().any():
                i = g[col].idxmax()
                rows.append(dict(ph=ph, comparison=tag, max_wasserstein_log=round(float(g.loc[i, col]), 4),
                                 at_target_min=float(g.loc[i, "target_min"]),
                                 max_abs_dcumQ3=float(g.loc[i, f"{tag}_max_abs_dcumQ3"])))
    return pd.DataFrame(rows)


# ── render one version ────────────────────────────────────────────────────────
def render_version(vname, runs, grid, common_xo, out_dir, formats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    ps.apply_manuscript_style()

    cats = VERSIONS[vname]
    q3_by_ph = {ph: aggregate_q3([r for r in runs if r["ph"] == ph], grid, cats) for ph in CONDITIONS}
    sc_by_ph = {ph: aggregate_scattering([r for r in runs if r["ph"] == ph], grid) for ph in CONDITIONS}

    norm = Normalize(0.0, TMAX_MIN)
    cmap = plt.get_cmap(CMAP)
    anchor_gis = {int(round(a / DT_MIN)) for a in ANCHORS_MIN}

    # shared y-limits per column across the three rows (drop empty)
    q3_vals = [m["mean"] * 100.0 for d in q3_by_ph.values() for m in d.values()]
    sc_vals = [m["mean"] for d in sc_by_ph.values() for m in d.values()]
    q3_ymax = max((v[(common_xo >= Q3_XLIM[0]) & (common_xo <= Q3_XLIM[1])].max() for v in q3_vals), default=1.0)
    sc_ymax = max((v.max() for v in sc_vals), default=1.0)
    ch = np.arange(1, N_CHANNELS + 1)

    # manual layout: a reserved left gutter holds the rotated pH row labels clear of the y-labels
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 8.2))
    fig.subplots_adjust(left=0.155, right=0.86, top=0.945, bottom=0.085, hspace=0.34, wspace=0.40)

    def _draw(ax, means, x, *, scale):
        for gi in sorted(means):
            thick = gi in anchor_gis
            ax.plot(x, means[gi]["mean"] * scale, color=cmap(norm(grid[gi])),
                    lw=(1.7 if thick else 0.55), alpha=(0.95 if thick else 0.32),
                    zorder=(3 if thick else 1), solid_capstyle="round")

    for r, ph in enumerate(CONDITIONS):
        axq, axs = axes[r]
        _draw(axq, q3_by_ph[ph], common_xo, scale=100.0)          # fraction → % volume per class
        axq.set_xlim(*Q3_XLIM); axq.set_ylim(0, q3_ymax * 1.05)
        _draw(axs, sc_by_ph[ph], ch, scale=1.0)                   # S already in % of initial total
        axs.set_xlim(0.5, N_CHANNELS + 0.5); axs.set_ylim(0, sc_ymax * 1.05)
        for ax in (axq, axs):
            ps.setup_axes(ax)
            ax.grid(True, lw=0.3, alpha=0.25)
        axq.set_ylabel("Volume fraction per\nsize class (%)")
        axs.set_ylabel("Channel signal (% of\ninitial total angular signal)")
        ps.panel_label(axq, "ACE"[r], x=-0.20, y=1.03)
        ps.panel_label(axs, "BDF"[r], x=-0.17, y=1.03)

    axes[0][0].set_title("PAQXOS q3 size composition", fontsize=9, fontweight="bold")
    axes[0][1].set_title("Measured angular scattering", fontsize=9, fontweight="bold")
    axes[2][0].set_xlabel("Particle diameter (µm)")
    axes[2][1].set_xlabel("Detector channel")
    # unobtrusive direction cue: low channel = larger characteristic size
    axes[2][1].annotate("larger ←  characteristic size  → smaller", xy=(0.5, -0.34),
                        xycoords="axes fraction", ha="center", va="top", fontsize=6.5, color="0.35")

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.885, 0.30, 0.018, 0.40])               # dedicated colorbar axis in the right gutter
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Time (min)"); cb.set_ticks(ANCHORS_MIN)

    # pH row labels in the reserved left gutter (clear of the y-labels)
    for r, ph in enumerate(CONDITIONS):
        pos = axes[r][0].get_position()
        fig.text(0.025, 0.5 * (pos.y0 + pos.y1), f"pH {ph}", rotation=90, ha="center", va="center",
                 fontsize=12, fontweight="bold")

    stem = out_dir / VERSION_STEM[vname]
    written = _save_multiformat(fig, stem, formats)
    plt.close(fig)
    return written


def _save_multiformat(fig, stem: Path, formats) -> list:
    """Save one figure to each requested format with a per-format dpi (600 dpi TIFF for manuscript use)."""
    stem = Path(stem); stem.parent.mkdir(parents=True, exist_ok=True)
    dpi = {"png": 300, "pdf": ps.DPI, "svg": ps.DPI, "tif": 600, "tiff": 600}
    out = []
    for fmt in formats:
        ext = "tiff" if fmt in ("tif", "tiff") else fmt
        p = stem.with_suffix(f".{ext}")
        kw = dict(dpi=dpi.get(fmt, ps.DPI), bbox_inches="tight")
        if ext == "tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(p, **kw)
        out.append(p)
    return out


# ── driver ────────────────────────────────────────────────────────────────────
def run(output_dir, formats=("png", "pdf", "svg", "tiff")):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = _common_time_grid()

    by_ph = defaultdict(list)
    for ph, date, rep, rtf, fo in iter_runs():
        by_ph[float(ph)].append((int(date), int(rep), rtf, fo))
    common_xo = np.array(sorted({round(float(x), 4)
                                 for ph in CONDITIONS for (_d, _r, _rtf, fo) in by_ph[ph]
                                 for x in psd.read_q3_frames(fo).xo}))

    runs = []
    for ph in CONDITIONS:
        for date, rep, rtf, fo in by_ph[ph]:
            runs.append(build_run(ph, date, rep, rtf, fo, common_xo))

    # tables
    elig = per_frame_eligibility(runs)
    summ = run_summary(elig)
    q3cov = q3_coverage(runs, grid)
    sccov = scattering_coverage(runs, grid)
    sens = q3_sensitivity(runs, grid, common_xo)
    head = sensitivity_headline(sens)
    elig.drop(columns=[], errors="ignore").to_csv(output_dir / "q3_scattering_per_frame_eligibility.csv", index=False)
    summ.to_csv(output_dir / "q3_scattering_run_summary.csv", index=False)
    q3cov.to_csv(output_dir / "q3_evolution_timepoint_coverage.csv", index=False)
    sccov.to_csv(output_dir / "scattering_evolution_timepoint_coverage.csv", index=False)
    sens.to_csv(output_dir / "q3_evolution_sensitivity_comparison.csv", index=False)
    head.to_csv(output_dir / "q3_evolution_sensitivity_headline.csv", index=False)

    # figures (one per version)
    figs = {v: render_version(v, runs, grid, common_xo, output_dir, formats) for v in VERSIONS}
    return dict(runs=runs, eligibility=elig, run_summary=summ, q3_coverage=q3cov,
                scattering_coverage=sccov, sensitivity=sens, headline=head, figures=figs,
                common_xo=common_xo, grid=grid)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True, help="manuscript figures_and_tables directory")
    ap.add_argument("--formats", default="png,pdf,svg,tiff", help="comma-separated output formats")
    args = ap.parse_args(argv)
    formats = tuple(s.strip() for s in args.formats.split(",") if s.strip())
    res = run(args.output_dir, formats)
    print("q3 / angular-scattering evolution figures:")
    for v, paths in res["figures"].items():
        print(f"  [{v}]")
        for p in paths:
            print(f"    {p}")
    print("\nframes by eligibility category (pooled):")
    print(res["eligibility"].category.value_counts().to_string())
    if len(res["headline"]):
        print("\nlargest q3 version discrepancy per pH:")
        print(res["headline"].to_string(index=False))


if __name__ == "__main__":
    main()
