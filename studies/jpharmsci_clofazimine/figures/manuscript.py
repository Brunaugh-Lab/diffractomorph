"""Manuscript figure builders (Figures 2, 3, 5) for the pH-dependent dissolution/diffraction study.

These are the **publication** figures, distinct from the exploratory diagnostics in
:mod:`diffractomorph_pipeline.figures.diagnostic` and the per-run analysis drivers under
``analysis/``. They read only the canonical CSV outputs (and, for Figure 2 panel A, reconstruct
the per-run total angular signal ΣI(t) straight from the measurement ``.rtf`` files), never the
diagnostic PNGs, and they never recompute the KWW / UV / matched-``g`` science — they re-express it.

Two rules govern every panel here:

* **The experimental unit is the preparation date.** The three runs inside a (date × condition)
  cell are nested replicates. Condition-level central tendency and uncertainty are built
  *day-first*: runs → (date × condition) mean → condition mean ± **between-date** SD (``n = 3
  dates``), never by pooling the nine runs as independent.
* **No mass/size over-claim.** Total angular signal ΣI, ``Copt``, and q3 are particle-side optical
  quantities, not dissolved drug mass and not a per-channel particle diameter.

The module is split into pure *loaders* (``figureN_data`` — read a study root, return plain
dict/DataFrame data) and pure *renderers* (``render_figureN`` — take that data, write PNG+PDF and a
per-panel source-data CSV for every numerical panel). The split lets the numeric loaders be
regression-tested against the canonical CSVs and the renderers be exercised on synthetic data with
no instrument files present.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from diffractomorph_pipeline import plot_styles as ps

# ── canonical layout under the study root ────────────────────────────────────
_MATCHED_G = ("psd_evolution", "redistribution_matched_g")
_SUMMARY = ("summary",)

# Panel-A common time grid (min). Runs share the instrument but differ by a frame or two in length
# and cadence, so per-run transformed trajectories are interpolated onto this documented grid before
# any nested averaging. No extrapolation: a run contributes only up to its own last acquired frame.
GRID_TMAX_MIN = 20.0
GRID_DT_MIN = 0.2
N_CHANNELS = 31
MATCHED_G_MAIN = 0.8          # only common-support extent across all three conditions
MATCHED_G_TARGETS = [0.8, 0.6, 0.4, 0.2]
PLATEAU_BASIS = "plateau(t>=10)"   # recovery_corrected.csv basis that is plateau-eligible
CS_MEAS_UGML = {4.0: 17.12, 4.5: 6.43, 5.0: 2.0}   # canonical measured equilibrium solubility


def _paths(study_root: str | Path) -> dict:
    r = Path(study_root)
    mg = r.joinpath(*_MATCHED_G)
    s = r.joinpath(*_SUMMARY)
    return dict(
        root=r,
        run_metadata=s / "run_metadata.csv",
        uv_timecourse=s / "uv_timecourse_all.csv",
        recovery=s / "recovery_corrected.csv",
        mg_by_date=mg / "matched_g_by_date_condition.csv",
        mg_coverage=mg / "matched_g_coverage.csv",
        mg_coverage_by_date=mg / "matched_g_coverage_by_date.csv",
        mg_summary=mg / "matched_g_summary.csv",
    )


def _ph_from_condition(cond: str) -> float:
    """``'pH 4.0'`` / ``'pH4.0'`` → ``4.0``."""
    return round(float(str(cond).lower().replace("ph", "").strip()), 1)


def _find_rtf(study_root: str | Path, ph: float, date: int, rep: int) -> str | None:
    """Locate a run's measurement ``.rtf`` under the study tree (mirrors
    ``analysis/study_common.find_rtf``; kept here so the package stays importable without the
    ``analysis/`` scripts on the path)."""
    root = Path(study_root)
    hits = glob.glob(str(root / f"ph_{ph}" / f"{date}_pH*" / f"*measurement*Rep {rep}.rtf"))
    return hits[0] if hits else None


def _parse_run_id(rid: str) -> tuple[float, int, int]:
    """``'pH4.0_20260608_R1'`` → ``(4.0, 20260608, 1)``."""
    a, b, c = rid.split("_")
    return round(float(a[2:]), 1), int(b), int(c[1:])


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — pH-dependent angular-scattering kinetics
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Figure2Data:
    grid_t: np.ndarray                       # common time grid (min) for panel A
    panelA_runs: pd.DataFrame                # level=run: id/condition/date/rep/time_min/sigma_i_pct (native t)
    panelA_date: pd.DataFrame                # level=date_mean: condition/date/time_min/sigma_i_pct (grid)
    panelA_cond: pd.DataFrame                # level=condition: condition/time_min/mean_pct/sd_pct/n_dates
    panelBCD_date: pd.DataFrame              # condition/date + mean_relax_min/optical_decay_depth_pct/beta
    panelBCD_cond: pd.DataFrame              # condition + *_mean/*_sd across dates (n_dates)
    conditions: list = field(default_factory=list)


def _reconstruct_sigma_i(study_root, ph, date, rep):
    """Per-run total angular signal ΣI(t): (t_min, ΣI) reconstructed from the ``.rtf`` exactly as the
    canonical KWW driver does (``kinetics.total_signal`` then upward-glitch despiking). Returns
    ``(None, None)`` if the run's ``.rtf`` is absent."""
    from diffractomorph_pipeline import ingest, kinetics
    rtf = _find_rtf(study_root, ph, date, rep)
    if rtf is None:
        return None, None
    run = ingest.extract_run(rtf)
    t, sig = kinetics.despike_upward(run.t_min, kinetics.total_signal(run.I))
    return np.asarray(t, float), np.asarray(sig, float)


def _common_grid() -> np.ndarray:
    n = int(round(GRID_TMAX_MIN / GRID_DT_MIN)) + 1
    return np.round(np.linspace(0.0, GRID_TMAX_MIN, n), 6)


def figure2_data(study_root: str | Path, *, fits: pd.DataFrame,
                 by_date: pd.DataFrame) -> Figure2Data:
    """Assemble Figure-2 data from aggregate KWW tables + reconstructed ΣI(t) trajectories.

    The caller must pass tables recomputed from raw RTFs with the declared circulation-start
    policy. Historical derived KWW CSVs are deliberately not a fallback.

    Panel A trajectory pipeline (transform-first, then nest): each run's ΣI(t) → ``100·ΣI/i0_fit``
    (percent of the fitted back-extrapolated start, **not** percent of the first measured frame) →
    interpolated onto the common grid within its own time support → averaged across the three nested
    runs within a date → the three date-mean trajectories summarized to a condition mean ±
    between-date SD.
    """
    fits = fits.copy()
    by_date = by_date.copy()
    grid = _common_grid()

    run_rows, date_grid_rows = [], []
    # transform each run, interpolate onto the grid (NaN past its last frame)
    run_on_grid = {}   # (condition, date, rep) -> pct on grid
    for row in fits.itertuples():
        ph, date, rep = _parse_run_id(row.id)
        t, sig = _reconstruct_sigma_i(study_root, ph, date, rep)
        i0 = float(row.i0_fit)
        if t is None or not np.isfinite(i0) or i0 <= 0:
            continue
        pct = 100.0 * sig / i0
        cond = str(row.condition)
        for tt, vv in zip(t, pct):                         # native-resolution run trajectory (source data)
            run_rows.append(dict(level="run", condition=cond, date=int(date), rep=int(rep),
                                 id=row.id, time_min=round(float(tt), 4),
                                 sigma_i_pct=round(float(vv), 4)))
        on_grid = np.interp(grid, t, pct, left=pct[0], right=np.nan)
        on_grid[grid > t.max() + 1e-9] = np.nan
        run_on_grid[(cond, int(date), int(rep))] = on_grid

    # runs → date mean (nanmean across the reps present that date)
    date_mean = {}   # (condition, date) -> grid pct
    keys = sorted({(c, d) for (c, d, _r) in run_on_grid})
    for cond, date in keys:
        stack = np.vstack([run_on_grid[(c, d, r)] for (c, d, r) in run_on_grid if c == cond and d == date])
        with np.errstate(invalid="ignore"):
            dm = np.nanmean(stack, axis=0)
        date_mean[(cond, date)] = dm
        for tt, vv in zip(grid, dm):
            if np.isfinite(vv):
                date_grid_rows.append(dict(level="date_mean", condition=cond, date=int(date),
                                           time_min=round(float(tt), 4), sigma_i_pct=round(float(vv), 4)))

    # date means → condition mean ± between-date SD (only where all dates for that condition have support)
    cond_rows = []
    for cond in sorted({c for c, _ in date_mean}):
        dms = [date_mean[(c, d)] for (c, d) in date_mean if c == cond]
        M = np.vstack(dms)
        n_dates = M.shape[0]
        allfin = np.isfinite(M).all(axis=0)
        mean = np.where(allfin, np.nanmean(M, axis=0), np.nan)
        sd = np.where(allfin, np.nanstd(M, axis=0, ddof=1) if n_dates > 1 else 0.0, np.nan)
        for tt, m, s in zip(grid, mean, sd):
            if np.isfinite(m):
                cond_rows.append(dict(level="condition", condition=cond, time_min=round(float(tt), 4),
                                      mean_pct=round(float(m), 4), sd_pct=round(float(s), 4),
                                      n_dates=int(n_dates)))

    # Panels B/C/D — date-level values (n = dates) + condition summary across dates
    metrics = ["mean_relax_min", "optical_decay_depth_pct", "beta"]
    bcd_date = by_date[["condition", "date"] + metrics].copy()
    bcd_cond_rows = []
    for cond, sub in bcd_date.groupby("condition"):
        row = {"condition": cond, "n_dates": int(sub.shape[0])}
        for m in metrics:
            v = sub[m].to_numpy(float)
            row[f"{m}_mean"] = round(float(np.nanmean(v)), 4)
            row[f"{m}_sd"] = round(float(np.nanstd(v, ddof=1)), 4) if np.isfinite(v).sum() > 1 else np.nan
        bcd_cond_rows.append(row)

    conditions = sorted(bcd_date.condition.unique(), key=_ph_from_condition)
    return Figure2Data(
        grid_t=grid,
        panelA_runs=pd.DataFrame(run_rows),
        panelA_date=pd.DataFrame(date_grid_rows),
        panelA_cond=pd.DataFrame(cond_rows),
        panelBCD_date=bcd_date.reset_index(drop=True),
        panelBCD_cond=pd.DataFrame(bcd_cond_rows),
        conditions=list(conditions),
    )


def _dot_panel(ax, date_df, cond_df, metric, ylabel, conditions):
    """Shared B/C/D dot panel: three date-level points per condition (deterministic jitter) + the
    condition mean ± between-date SD."""
    for x, cond in enumerate(conditions):
        st = ps.ph_style(_ph_from_condition(cond))
        v = date_df.loc[date_df.condition == cond, metric].dropna().to_numpy(float)
        if v.size:
            jit = np.linspace(-0.11, 0.11, v.size) if v.size > 1 else np.array([0.0])
            ax.scatter(x + jit, v, s=26, facecolor=st["color"], edgecolor="none",
                       alpha=0.55, marker=st["marker"], zorder=2)
        cr = cond_df[cond_df.condition == cond]
        if not cr.empty:
            m = float(cr[f"{metric}_mean"].iloc[0])
            sd = cr[f"{metric}_sd"].iloc[0]
            sd = 0.0 if pd.isna(sd) else float(sd)
            ax.errorbar([x], [m], yerr=[sd], fmt=st["marker"], color=st["color"], ms=8,
                        capsize=4, markeredgecolor="k", elinewidth=1.4, zorder=3)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([ps.ph_style(_ph_from_condition(c))["label"] for c in conditions])
    ax.set_xlim(-0.5, len(conditions) - 0.5)
    ax.set_ylabel(ylabel)
    ps.setup_axes(ax)


def render_figure2(data: Figure2Data, out_dir, source_dir, formats=("png", "pdf"),
                   show_date_lines: bool = True) -> dict:
    """Render Figure 2 (2×2) and write its four per-panel source CSVs. Returns the paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps.apply_manuscript_style()
    out_dir, source_dir = Path(out_dir), Path(source_dir)
    conds = data.conditions

    fig, axes = plt.subplots(2, 2, figsize=(ps.DOUBLE_COL_WIDTH, ps.DOUBLE_COL_WIDTH * 0.82))
    (axA, axB), (axC, axD) = axes

    # ── Panel A — empirical particle-side trajectories (% of fitted start) ──
    for cond in conds:
        st = ps.ph_style(_ph_from_condition(cond))
        if show_date_lines:
            for date, dsub in data.panelA_date[data.panelA_date.condition == cond].groupby("date"):
                dsub = dsub.sort_values("time_min")
                axA.plot(dsub.time_min, dsub.sigma_i_pct, color=st["color"],
                         lw=ps.DATE_LINE_WIDTH, alpha=ps.DATE_LINE_ALPHA, zorder=1)
        cs = data.panelA_cond[data.panelA_cond.condition == cond].sort_values("time_min")
        if cs.empty:
            continue
        axA.fill_between(cs.time_min, cs.mean_pct - cs.sd_pct, cs.mean_pct + cs.sd_pct,
                         color=st["color"], alpha=ps.BAND_ALPHA, lw=0, zorder=1)
        axA.plot(cs.time_min, cs.mean_pct, color=st["color"], linestyle=st["linestyle"],
                 lw=ps.MEAN_LINEWIDTH, label=st["label"], zorder=3)
    axA.set_xlabel("Time (min)")
    axA.set_ylabel("Angular ΣI (% of fitted start)")
    axA.set_xlim(0, 12)
    axA.set_ylim(0, 100)
    axA.set_yticks([0, 25, 50, 75, 100])
    ps.setup_axes(axA)
    ps.panel_label(axA, "A", x=-0.32, y=1.13)

    # ── Panels B / C / D — date-level KWW descriptors ──
    # explicit upper limits land on a labeled tick so every y-axis shows its top boundary
    _dot_panel(axB, data.panelBCD_date, data.panelBCD_cond, "mean_relax_min",
               "Mean relaxation time (min)", conds)
    axB.set_ylim(0, 4)
    axB.set_yticks([0, 1, 2, 3, 4])
    ps.panel_label(axB, "B", x=-0.32, y=1.13)
    _dot_panel(axC, data.panelBCD_date, data.panelBCD_cond, "optical_decay_depth_pct",
               "Optical decay depth (%)", conds)
    axC.set_ylim(0, 100)
    axC.set_yticks([0, 20, 40, 60, 80, 100])
    ps.panel_label(axC, "C", x=-0.32, y=1.13)
    _dot_panel(axD, data.panelBCD_date, data.panelBCD_cond, "beta",
               "KWW stretch exponent, β", conds)
    axD.set_ylim(0.7, 1.1)
    axD.set_yticks([0.7, 0.8, 0.9, 1.0, 1.1])
    ps.panel_label(axD, "D", x=-0.32, y=1.13)

    # single shared legend along the bottom (keeps it off the panel-A curves)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=ps.ph_style(_ph_from_condition(c))["color"],
                      linestyle=ps.ph_style(_ph_from_condition(c))["linestyle"],
                      marker=ps.ph_style(_ph_from_condition(c))["marker"], lw=ps.MEAN_LINEWIDTH,
                      markeredgecolor="k", label=ps.ph_style(_ph_from_condition(c))["label"])
               for c in conds]
    fig.tight_layout(w_pad=2.5, h_pad=3.6, rect=(0, 0.05, 1, 0.97))
    fig.legend(handles=handles, loc="lower center", ncol=len(conds), handlelength=2.2,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    written = _savefig(fig, out_dir / "Figure_pH_angular_scattering_kinetics", formats)
    plt.close(fig)

    # ── per-panel source data ──
    srcs = {}
    src_A = pd.concat([
        data.panelA_runs.assign(),
        data.panelA_date.assign(rep=np.nan, id=np.nan),
        data.panelA_cond.rename(columns={"mean_pct": "sigma_i_pct"}).assign(
            date=np.nan, rep=np.nan, id=np.nan),
    ], ignore_index=True)
    stem2 = "Figure_pH_angular_scattering_kinetics"
    srcs[f"{stem2}_panelA_angular_trajectories.csv"] = src_A
    srcs[f"{stem2}_panelB_mean_relaxation_time.csv"] = _bcd_source(data, "mean_relax_min")
    srcs[f"{stem2}_panelC_optical_decay_depth.csv"] = _bcd_source(data, "optical_decay_depth_pct")
    srcs[f"{stem2}_panelD_stretch_exponent.csv"] = _bcd_source(data, "beta")
    written += _write_sources(source_dir, srcs)
    return {"figure": written[:len(formats)], "sources": written[len(formats):]}


def _bcd_source(data: Figure2Data, metric: str) -> pd.DataFrame:
    date_rows = data.panelBCD_date[["condition", "date", metric]].copy()
    date_rows.insert(0, "level", "date")
    date_rows = date_rows.rename(columns={metric: "value"})
    cond = data.panelBCD_cond[["condition", "n_dates", f"{metric}_mean", f"{metric}_sd"]].copy()
    cond.insert(0, "level", "condition")
    cond = cond.rename(columns={f"{metric}_mean": "value", f"{metric}_sd": "sd_between_dates"})
    cond["date"] = np.nan
    return pd.concat([date_rows, cond], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — UV dissolved-mass recovery and optical comparison
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Figure3Data:
    panelA_runs: pd.DataFrame        # ph/date/rep/time_min/dissolved_pct
    panelA_date: pd.DataFrame        # ph/date/time_min/dissolved_pct (run mean)
    panelA_cond: pd.DataFrame        # ph/time_min/mean_pct/sd_pct/n_dates (dates with support)
    panelB_date: pd.DataFrame        # ph/date/recovery_pct (eligible run mean) + n_eligible_runs
    panelB_cond: pd.DataFrame        # ph/recovery_mean/recovery_sd/n_dates
    panelC: pd.DataFrame             # ph/date/optical_depth_pct/uv_recovery_pct
    table2: pd.DataFrame             # one row per date × condition
    cond_summary: pd.DataFrame       # condition-level summary
    conditions: list = field(default_factory=list)


def figure3_data(study_root: str | Path, *, by_date: pd.DataFrame) -> Figure3Data:
    p = _paths(study_root)
    uv = pd.read_csv(p["uv_timecourse"])
    rec = pd.read_csv(p["recovery"])
    by_date = by_date.copy()

    # ── Panel A — corrected UV dissolved mass (% of injected) ──
    uv = uv.copy()
    uv["dissolved_pct"] = 100.0 * uv.true_dissolved_ugml / uv.loaded_ugml
    panelA_runs = uv[["ph", "date", "rep", "time_min", "dissolved_pct"]].copy()
    panelA_date = (uv.groupby(["ph", "date", "time_min"], as_index=False)
                   .dissolved_pct.mean())
    # condition mean across dates, only at times where all 3 dates for the condition are present
    cond_rows = []
    for ph, sub in panelA_date.groupby("ph"):
        n_dates_cond = sub.date.nunique()
        for t, tsub in sub.groupby("time_min"):
            if tsub.date.nunique() < n_dates_cond:
                continue                                # not comparable time-point support
            v = tsub.dissolved_pct.to_numpy(float)
            cond_rows.append(dict(ph=ph, time_min=t, mean_pct=round(float(v.mean()), 4),
                                  sd_pct=round(float(v.std(ddof=1)), 4) if v.size > 1 else 0.0,
                                  n_dates=int(tsub.date.nunique())))
    panelA_cond = pd.DataFrame(cond_rows)

    # ── Panel B — plateau recovered fraction (plateau-eligible runs only) ──
    elig = rec[rec.basis == PLATEAU_BASIS].copy()
    pb_date = (elig.groupby(["ph", "date"], as_index=False)
               .agg(recovery_pct=("recovery_corrected", lambda s: 100.0 * s.mean()),
                    recovery_sd_within_date=("recovery_corrected",
                                             lambda s: 100.0 * s.std(ddof=1) if s.size > 1 else np.nan),
                    n_eligible_runs=("recovery_corrected", "size")))
    pb_cond_rows = []
    for ph, sub in pb_date.groupby("ph"):
        v = sub.recovery_pct.to_numpy(float)
        pb_cond_rows.append(dict(ph=ph, recovery_mean=round(float(v.mean()), 3),
                                 recovery_sd=round(float(v.std(ddof=1)), 3) if v.size > 1 else np.nan,
                                 n_dates=int(sub.shape[0])))
    panelB_cond = pd.DataFrame(pb_cond_rows)

    # ── Panel C — optical decay depth vs eligible UV recovery, date-level merge ──
    opt = by_date.copy()
    opt["ph"] = opt.condition.map(_ph_from_condition)
    merge = opt.merge(pb_date[["ph", "date", "recovery_pct", "n_eligible_runs"]],
                      on=["ph", "date"], how="inner")
    panelC = merge[["ph", "date", "optical_decay_depth_pct", "recovery_pct"]].rename(
        columns={"recovery_pct": "uv_recovery_pct"}).reset_index(drop=True)

    # ── Table 2 — one row per date × condition ──
    t2 = by_date.copy()
    t2["ph"] = t2.condition.map(_ph_from_condition)
    # nested run count = runs in the (date × condition) cell, from the canonical per-run recovery
    # table (one row per run), NOT the UV timecourse (which under-counts early-only reps).
    n_runs_dc = rec.groupby(["ph", "date"]).rep.nunique().rename("nested_run_count").reset_index()
    t2 = t2.merge(n_runs_dc, on=["ph", "date"], how="left")
    t2 = t2.merge(pb_date.rename(columns={"recovery_pct": "uv_plateau_recovery_mean",
                                          "recovery_sd_within_date": "uv_plateau_recovery_sd_within_date",
                                          "n_eligible_runs": "n_plateau_eligible_uv_runs"}),
                  on=["ph", "date"], how="left")
    t2["n_plateau_eligible_uv_runs"] = t2.n_plateau_eligible_uv_runs.fillna(0).astype(int)

    def _notes(r):
        n = int(r.n_plateau_eligible_uv_runs)
        tag = "" if n == int(r.nested_run_count) else f"; {int(r.nested_run_count) - n} run(s) early-only excluded"
        return f"{n}/{int(r.nested_run_count)} runs plateau-eligible (t>=10 min){tag}"

    t2["eligibility_notes"] = t2.apply(_notes, axis=1)
    table2 = t2[["condition", "date", "nested_run_count", "mean_relax_min", "beta",
                 "optical_decay_depth_pct", "n_plateau_eligible_uv_runs",
                 "uv_plateau_recovery_mean", "uv_plateau_recovery_sd_within_date",
                 "eligibility_notes"]].copy()

    # condition-level summary from the date-condition rows (between-date SD)
    cs_rows = []
    for cond, sub in table2.groupby("condition"):
        ph = _ph_from_condition(cond)
        def _ms(col):
            v = sub[col].to_numpy(float)
            v = v[np.isfinite(v)]
            return (round(float(v.mean()), 3), round(float(v.std(ddof=1)), 3) if v.size > 1 else np.nan)
        mr_m, mr_s = _ms("mean_relax_min")
        b_m, b_s = _ms("beta")
        od_m, od_s = _ms("optical_decay_depth_pct")
        rec_m, rec_s = _ms("uv_plateau_recovery_mean")
        cs_rows.append(dict(condition=cond, n_dates=int(sub.shape[0]),
                            mean_relax_min_mean=mr_m, mean_relax_min_sd=mr_s,
                            beta_mean=b_m, beta_sd=b_s,
                            optical_decay_depth_pct_mean=od_m, optical_decay_depth_pct_sd=od_s,
                            uv_plateau_recovery_mean=rec_m, uv_plateau_recovery_sd_between_dates=rec_s,
                            cs_meas_ugml=CS_MEAS_UGML.get(ph, np.nan)))
    cond_summary = pd.DataFrame(cs_rows)

    conditions = sorted(by_date.condition.unique(), key=_ph_from_condition)
    return Figure3Data(panelA_runs=panelA_runs, panelA_date=panelA_date, panelA_cond=panelA_cond,
                       panelB_date=pb_date, panelB_cond=panelB_cond, panelC=panelC,
                       table2=table2, cond_summary=cond_summary, conditions=list(conditions))


def render_figure3(data: Figure3Data, out_dir, source_dir, tables_dir, formats=("png", "pdf")) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps.apply_manuscript_style()
    from matplotlib.lines import Line2D
    out_dir, source_dir, tables_dir = Path(out_dir), Path(source_dir), Path(tables_dir)
    conds = data.conditions
    handles = [Line2D([0], [0], color=ps.ph_style(_ph_from_condition(c))["color"],
                      linestyle=ps.ph_style(_ph_from_condition(c))["linestyle"],
                      marker=ps.ph_style(_ph_from_condition(c))["marker"], lw=ps.MEAN_LINEWIDTH,
                      markeredgecolor="k", label=ps.ph_style(_ph_from_condition(c))["label"])
               for c in conds]

    # Main figure: Panel A (UV time series) beside Panel B (plateau recovery).
    fig = plt.figure(figsize=(ps.DOUBLE_COL_WIDTH, ps.DOUBLE_COL_WIDTH * 0.44),
                     constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ── Panel A — UV dissolved-mass trajectories (% injected) ──
    # emphasized date/condition means are anchored at the t=0 dosing boundary (0 % dissolved);
    # that origin is a known dosing assumption, not an acquired UV sample (light run points are
    # measured values only, from t = 2 min).
    for cond in conds:
        ph = _ph_from_condition(cond)
        st = ps.ph_style(ph)
        runs = data.panelA_runs[data.panelA_runs.ph == ph]
        axA.plot(runs.time_min, runs.dissolved_pct, st["marker"], ms=2.4, alpha=0.28,
                 color=st["color"], ls="none", zorder=1)
        for date, dsub in data.panelA_date[data.panelA_date.ph == ph].groupby("date"):
            dsub = dsub.sort_values("time_min")
            axA.plot(np.r_[0.0, dsub.time_min], np.r_[0.0, dsub.dissolved_pct], color=st["color"],
                     lw=ps.DATE_LINE_WIDTH, alpha=ps.DATE_LINE_ALPHA, zorder=2)
        cs = data.panelA_cond[data.panelA_cond.ph == ph].sort_values("time_min")
        if not cs.empty:
            axA.plot(np.r_[0.0, cs.time_min], np.r_[0.0, cs.mean_pct], color=st["color"],
                     linestyle=st["linestyle"], lw=ps.MEAN_LINEWIDTH, marker=st["marker"], ms=3.5,
                     label=st["label"], zorder=3)
    axA.set_xlabel("Time (min)")
    axA.set_ylabel("Dissolved CFZ (% of injected)")
    axA.set_xlim(0, 20)
    axA.set_ylim(0, 100)
    axA.set_yticks([0, 20, 40, 60, 80, 100])
    ps.setup_axes(axA)
    ps.panel_label(axA, "A", x=-0.18, y=1.05)

    # ── Panel B — plateau recovered fraction ──
    for x, cond in enumerate(conds):
        ph = _ph_from_condition(cond)
        st = ps.ph_style(ph)
        v = data.panelB_date.loc[data.panelB_date.ph == ph, "recovery_pct"].to_numpy(float)
        if v.size:
            jit = np.linspace(-0.11, 0.11, v.size) if v.size > 1 else np.array([0.0])
            axB.scatter(x + jit, v, s=26, color=st["color"], alpha=0.55, marker=st["marker"], zorder=2)
        cr = data.panelB_cond[data.panelB_cond.ph == ph]
        if not cr.empty:
            sd = cr.recovery_sd.iloc[0]
            axB.errorbar([x], [float(cr.recovery_mean.iloc[0])],
                         yerr=[0.0 if pd.isna(sd) else float(sd)], fmt=st["marker"], color=st["color"],
                         ms=8, capsize=4, markeredgecolor="k", elinewidth=1.4, zorder=3)
    axB.set_xticks(range(len(conds)))
    axB.set_xticklabels([ps.ph_style(_ph_from_condition(c))["label"] for c in conds])
    axB.set_xlim(-0.5, len(conds) - 0.5)
    axB.set_ylim(0, 100)
    axB.set_yticks([0, 20, 40, 60, 80, 100])
    axB.set_ylabel("UV plateau recovery (%)")
    ps.setup_axes(axB)
    ps.panel_label(axB, "B", x=-0.20, y=1.05)

    fig.legend(handles=handles, loc="outside lower center", ncol=len(conds), handlelength=2.2,
               frameon=False)
    written = _savefig(fig, out_dir / "Figure_UV_dissolved_mass_and_recovery", formats)
    plt.close(fig)

    # ── Supplementary figure — optical decay depth vs UV recovery (date-level) ──
    # no equality/regression line is drawn: the two axes measure different things and no
    # calibration is claimed.
    # sized to roughly one panel of the multi-panel figures so text reads at the same scale
    figS, axS = plt.subplots(figsize=(3.2, 3.0), constrained_layout=True)
    for cond in conds:
        ph = _ph_from_condition(cond)
        st = ps.ph_style(ph)
        sub = data.panelC[data.panelC.ph == ph]
        axS.scatter(sub.uv_recovery_pct, sub.optical_decay_depth_pct, s=36, color=st["color"],
                    marker=st["marker"], edgecolor="k", linewidth=0.4, label=st["label"], zorder=3)
    axS.set_xlabel("UV plateau recovery (%)")
    axS.set_ylabel("Optical decay depth (%)")
    axS.set_xlim(0, 100)
    axS.set_ylim(0, 100)
    axS.set_xticks([0, 20, 40, 60, 80, 100])
    axS.set_yticks([0, 20, 40, 60, 80, 100])
    ps.setup_axes(axS)
    axS.legend(loc="upper left", handlelength=1.4)
    written_supp = _savefig(figS, out_dir / "Figure_optical_decay_vs_uv_recovery", formats)
    plt.close(figS)
    written = written + written_supp

    # sources + tables
    stem3 = "Figure_UV_dissolved_mass_and_recovery"
    srcs = {
        f"{stem3}_panelA_uv_trajectories.csv": pd.concat([
            data.panelA_runs.assign(level="run"),
            data.panelA_date.assign(level="date_mean", rep=np.nan),
            data.panelA_cond.rename(columns={"mean_pct": "dissolved_pct"}).assign(
                level="condition", rep=np.nan, date=np.nan),
        ], ignore_index=True),
        f"{stem3}_panelB_plateau_recovery.csv": pd.concat([
            data.panelB_date.assign(level="date"),
            data.panelB_cond.assign(level="condition"),
        ], ignore_index=True),
        f"{stem3}_panelC_optical_vs_uv.csv": data.panelC.copy(),
    }
    src_paths = _write_sources(source_dir, srcs)
    tbl_paths = _write_sources(tables_dir, {
        "Table_date_level_endpoints.csv": data.table2,
        "Table_condition_level_summary.csv": data.cond_summary,
    })
    return {"figure": written, "sources": src_paths, "tables": tbl_paths}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — matched-extent channel redistribution
# ─────────────────────────────────────────────────────────────────────────────
# One residual statistic is used for BOTH panels: r_c = (x_c - g·x_{0,c}) / Σ_c x_{0,c}, with
# g = Σ_c x_c / Σ_c x_{0,c} and x_c = max(I_c - I_REF,c, 0). The anchor (x_0 = median of the first
# three frames) and the copt≤40 / synchronized-glitch QC mirror
# analysis/psd_redistribution_matched_g.py exactly; only the EVALUATION POINT differs between panels
# (each run's own endpoint vs the common g = 0.8 crossing).
N_ENDPOINT_FRAMES = 5          # endpoint = median of the final N valid frames
N_SIGMA_FLOOR = 5.0            # matched-g target rejected if remaining signal < N_SIGMA × total noise
R_C_DISPLAY_SCALE = 1000       # r_c is plotted as 10³·r_c; source CSVs keep the unscaled value


def _run_channel_matrix(rtf):
    """Despiked, background-subtracted channel matrix ``X`` (frames × 31, clipped ≥ 0) + times (min).

    Mirrors ``analysis/psd_redistribution_matched_g._run_matrix`` — same copt≤40 admission,
    synchronized-glitch despiking, and ``X = max(I − I_REF, 0)`` reference subtraction."""
    from diffractomorph_pipeline import ingest
    from diffractomorph_pipeline.noise_filter import despike_frames
    raw = ingest.extract_run(rtf)
    keep = np.isfinite(raw.copt) & (raw.copt <= 40.0)
    I, t, _copt, _ = despike_frames(np.asarray(raw.I, float)[keep],
                                    np.asarray(raw.t_min, float)[keep],
                                    np.asarray(raw.copt, float)[keep])
    ref = np.asarray(raw.ref, float)
    ref = np.nanmedian(ref, axis=0) if ref.ndim == 2 else ref
    return np.clip(I - ref[None, :], 0.0, None), np.asarray(t, float)


def _r_c(x, anchor, s0):
    """The shared residual-from-proportional-collapse statistic (Σ_c r_c = 0):
    ``g = Σ x / s0``; ``r_c = (x − g·anchor) / s0``. Returns ``(g, r_c_vector)``."""
    g = float(np.sum(x) / s0)
    return g, (np.asarray(x, float) - g * np.asarray(anchor, float)) / s0


def _endpoint_vector(X):
    """``r_c`` at a run's endpoint (median of its final ``N_ENDPOINT_FRAMES`` valid frames)."""
    anchor = np.nanmedian(X[:3], axis=0)
    s0 = float(anchor.sum())
    if s0 <= 0:
        return None
    x_end = np.nanmedian(X[-N_ENDPOINT_FRAMES:], axis=0)
    g, r = _r_c(x_end, anchor, s0)
    return dict(g=g, r=r)


def _matched_vector(X, target=MATCHED_G_MAIN):
    """``r_c`` at the first crossing of ``g = target`` (same anchor + noise-floor QC as the canonical
    matched-g module). Returns ``None`` if the run never reaches the target above its noise floor."""
    anchor = np.nanmedian(X[:3], axis=0)
    s0 = float(anchor.sum())
    if s0 <= 0 or len(X) < 5:
        return None
    total = X.sum(axis=1)
    g_env = np.minimum.accumulate(total / s0)
    d = np.diff(total)
    sigma = 1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d))) / np.sqrt(2) if len(d) else 0.0
    if target * s0 <= N_SIGMA_FLOOR * sigma:
        return None
    below = np.where(g_env <= target)[0]
    if below.size == 0 or below[0] == 0:
        return None
    j = int(below[0])
    g0, g1 = g_env[j - 1], g_env[j]
    f = 0.0 if g0 == g1 else (g0 - target) / (g0 - g1)
    xg = X[j - 1] + f * (X[j] - X[j - 1])
    g, r = _r_c(xg, anchor, s0)
    if g <= 0:
        return None
    return dict(g=g, r=r)


def _figure5_run_records(study_root):
    """Per-run endpoint and matched-``g``=0.8 vectors over the declared pH-study cohort.

    The run universe comes from ``summary/run_metadata.csv`` rather than a historical fitted-result
    table, so inclusion is independent of any prior KWW analysis.
    """
    p = _paths(study_root)
    metadata = pd.read_csv(p["run_metadata"])
    required = {"ph", "date_i", "rep"}
    if missing := required - set(metadata.columns):
        raise ValueError(f"run metadata lacks required columns: {sorted(missing)}")
    endpoint, matched = [], []
    for row in metadata.sort_values(["ph", "date_i", "rep"]).itertuples():
        ph, date, rep = float(row.ph), int(row.date_i), int(row.rep)
        rtf = _find_rtf(study_root, ph, date, rep)
        if rtf is None:
            continue
        X, _t = _run_channel_matrix(rtf)
        if len(X) < 5:
            continue
        base = dict(condition=f"pH {ph:.1f}", date=date, rep=rep)
        ev = _endpoint_vector(X)
        if ev is not None:
            endpoint.append({**base, "g": ev["g"], "r": ev["r"]})
        mv = _matched_vector(X, MATCHED_G_MAIN)
        if mv is not None:
            matched.append({**base, "g": mv["g"], "r": mv["r"]})
    return endpoint, matched


def _agg_profiles(records):
    """Nested aggregation (runs → date × condition → condition). Returns date-level and condition-level
    profile DataFrames, a run-level long DataFrame, and per-date / per-condition endpoint-``g`` summaries."""
    from collections import defaultdict
    by_dc = defaultdict(list)
    for rec in records:
        by_dc[(rec["condition"], rec["date"])].append(rec)
    date_rows, gend_date, date_vecs = [], [], defaultdict(list)
    for (cond, date), recs in by_dc.items():
        R = np.vstack([r["r"] for r in recs])
        m = R.mean(axis=0)
        gmean = float(np.mean([r["g"] for r in recs]))
        date_vecs[cond].append(m)
        gend_date.append(dict(condition=cond, date=date, n_reps=len(recs), g=round(gmean, 4)))
        for i in range(N_CHANNELS):
            date_rows.append(dict(condition=cond, date=date, n_reps=len(recs), channel=i + 1,
                                  r_c=float(m[i])))
    cond_rows = []
    for cond, vecs in date_vecs.items():
        M = np.vstack(vecs)
        n = M.shape[0]
        mean = M.mean(axis=0)
        sd = M.std(axis=0, ddof=1) if n > 1 else np.full(N_CHANNELS, np.nan)
        for i in range(N_CHANNELS):
            cond_rows.append(dict(condition=cond, channel=i + 1, n_dates=n,
                                  r_c_mean=float(mean[i]),
                                  r_c_sd=float(sd[i]) if np.isfinite(sd[i]) else np.nan))
    gd = pd.DataFrame(gend_date)
    gend_cond = []
    for cond, sub in gd.groupby("condition"):
        v = sub.g.to_numpy(float)
        gend_cond.append(dict(condition=cond, g_mean=round(float(v.mean()), 4),
                              g_sd=round(float(v.std(ddof=1)), 4) if v.size > 1 else np.nan,
                              n_dates=int(sub.shape[0])))
    run_rows = []
    for rec in records:
        for i in range(N_CHANNELS):
            run_rows.append(dict(condition=rec["condition"], date=rec["date"], rep=rec["rep"],
                                 channel=i + 1, g=round(rec["g"], 4), r_c=float(rec["r"][i])))
    return (pd.DataFrame(date_rows), pd.DataFrame(cond_rows), pd.DataFrame(run_rows),
            gd, pd.DataFrame(gend_cond))


@dataclass
class Figure5Data:
    # Panel A — unmatched endpoint comparison (each run at its own g_end)
    endpoint_run: pd.DataFrame
    endpoint_date: pd.DataFrame
    endpoint_cond: pd.DataFrame
    endpoint_gend_date: pd.DataFrame
    endpoint_gend_cond: pd.DataFrame
    # Panel B — matched comparison at g = 0.8
    matched_run: pd.DataFrame
    matched_date: pd.DataFrame
    matched_cond: pd.DataFrame
    matched_gend_cond: pd.DataFrame
    # Supporting-information coverage matrix (from the canonical coverage CSV)
    coverage: pd.DataFrame
    conditions: list = field(default_factory=list)
    targets: list = field(default_factory=lambda: list(MATCHED_G_TARGETS))
    matched_g: float = MATCHED_G_MAIN


def figure5_data(study_root: str | Path) -> Figure5Data:
    """Assemble Figure-5 data. Panel A (endpoint) and Panel B (matched g=0.8) are computed from the
    SAME ``r_c`` statistic (:func:`_r_c`) — only the evaluation point differs — so the panels are a
    like-for-like comparison of endpoint vs matched extent, not two different metrics."""
    p = _paths(study_root)
    endpoint_recs, matched_recs = _figure5_run_records(study_root)
    e_date, e_cond, e_run, e_gd, e_gc = _agg_profiles(endpoint_recs)
    m_date, m_cond, m_run, _m_gd, m_gc = _agg_profiles(matched_recs)

    coverage = pd.read_csv(p["mg_coverage"])[["condition", "target_g", "n_runs", "n_days"]].copy()
    conditions = sorted({r["condition"] for r in endpoint_recs}, key=_ph_from_condition)
    return Figure5Data(
        endpoint_run=e_run, endpoint_date=e_date, endpoint_cond=e_cond,
        endpoint_gend_date=e_gd, endpoint_gend_cond=e_gc,
        matched_run=m_run, matched_date=m_date, matched_cond=m_cond, matched_gend_cond=m_gc,
        coverage=coverage, conditions=list(conditions),
    )


def build_deeper_g_comparison(data: Figure5Data):
    """Guard: constructing a three-condition comparison at any extent deeper than ``g = 0.8`` is
    invalid because pH 5.0 has zero supported dates there. Always raises — the manuscript figure
    must not draw a deeper-``g`` trend across all three conditions.
    """
    raise ValueError(
        "Refusing a three-condition comparison at g < 0.8: pH 5.0 has no supported dates at the "
        f"deeper matched-g targets ({[g for g in MATCHED_G_TARGETS if g < MATCHED_G_MAIN]}); "
        f"g = {MATCHED_G_MAIN} is the only common-support extent.")


def _profile_panel(ax, date_df, cond_df, conds, *, scale=R_C_DISPLAY_SCALE):
    """Shared A/B profile panel: thin transparent date-level lines + thick condition-mean line, all in
    10³·r_c. Returns the (min, max) of the plotted scaled values for common-limit selection."""
    lo = hi = 0.0
    for cond in conds:
        st = ps.ph_style(_ph_from_condition(cond))
        for _date, dsub in date_df[date_df.condition == cond].groupby("date"):
            dsub = dsub.sort_values("channel")
            y = dsub.r_c.to_numpy(float) * scale
            ax.plot(dsub.channel, y, color=st["color"], lw=ps.DATE_LINE_WIDTH,
                    alpha=ps.DATE_LINE_ALPHA, zorder=1)
            lo, hi = min(lo, float(np.nanmin(y))), max(hi, float(np.nanmax(y)))
        cc = cond_df[cond_df.condition == cond].sort_values("channel")
        if cc.empty:
            continue
        ym = cc.r_c_mean.to_numpy(float) * scale
        ax.plot(cc.channel, ym, color=st["color"], linestyle=st["linestyle"],
                lw=ps.MEAN_LINEWIDTH, zorder=3)
        lo, hi = min(lo, float(np.nanmin(ym))), max(hi, float(np.nanmax(ym)))
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)
    ax.set_xlim(0.5, N_CHANNELS + 0.5)
    ax.set_xticks([1, 10, 20, 31])
    ax.set_xlabel("Detector channel")
    ps.setup_axes(ax)
    return lo, hi


def render_figure5(data: Figure5Data, out_dir, source_dir, formats=("png", "pdf")) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    ps.apply_manuscript_style()
    out_dir, source_dir = Path(out_dir), Path(source_dir)
    conds = data.conditions

    # ── Main figure: (A) endpoint comparison  |  (B) matched g = 0.8 comparison ──
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(ps.DOUBLE_COL_WIDTH, ps.DOUBLE_COL_WIDTH * 0.46),
                                   sharey=True, constrained_layout=True)
    loA, hiA = _profile_panel(axA, data.endpoint_date, data.endpoint_cond, conds)
    loB, hiB = _profile_panel(axB, data.matched_date, data.matched_cond, conds)
    # common, symmetric y-limits so the collapse from (A) to (B) is visible on one scale
    amax = max(abs(loA), abs(hiA), abs(loB), abs(hiB))
    ylim = 1.08 * amax
    axA.set_ylim(-ylim, ylim)
    axA.set_ylabel("Redistribution beyond\nproportional loss, 10³ $r_c$")

    # colour-keyed condition labels + endpoint g on Panel A (doubles as the legend for both panels).
    # placed upper-middle, clear of the channel-1/31 edge excursions.
    gc = data.endpoint_gend_cond.set_index("condition")
    ytxt = 0.97
    for cond in conds:
        st = ps.ph_style(_ph_from_condition(cond))
        g_m = gc.loc[cond, "g_mean"] if cond in gc.index else np.nan
        g_s = gc.loc[cond, "g_sd"] if cond in gc.index else np.nan
        lab = f"{st['label']}   $g_{{end}}$ = {g_m:.2f} ± {g_s:.2f}"
        axA.text(0.30, ytxt, lab, transform=axA.transAxes, fontsize=6.2, color=st["color"],
                 fontweight="bold", va="top", ha="left")
        ytxt -= 0.08
    axA.set_title("Endpoint comparison:\ndissolution extent differs", fontsize=8)
    axB.set_title(f"Matched comparison:\n{100*(1-data.matched_g):.0f}% total signal lost", fontsize=8)
    # condition colour key on Panel B too (matched g is common, so no per-condition g here)
    ytxt = 0.97
    for cond in conds:
        st = ps.ph_style(_ph_from_condition(cond))
        axB.text(0.35, ytxt, st["label"], transform=axB.transAxes, fontsize=6.2, color=st["color"],
                 fontweight="bold", va="top", ha="left")
        ytxt -= 0.08
    ps.panel_label(axA, "A", x=-0.20, y=1.02)
    ps.panel_label(axB, "B", x=-0.07, y=1.02)
    written = _savefig(fig, out_dir / "Figure_5_matched_extent_redistribution", formats)
    plt.close(fig)

    # ── Supporting-information coverage matrix ──
    figS, axS = plt.subplots(figsize=(3.4, 1.9), constrained_layout=True)
    targets = data.targets
    for i, cond in enumerate(conds):
        for j, g in enumerate(targets):
            row = data.coverage[(data.coverage.condition == cond) & np.isclose(data.coverage.target_g, g)]
            n_days = int(row.n_days.iloc[0]) if not row.empty else 0
            n_runs = int(row.n_runs.iloc[0]) if not row.empty else 0
            y = len(conds) - 1 - i
            if n_days == 0:
                axS.add_patch(Rectangle((j, y), 1, 1, facecolor="0.9", edgecolor="white",
                                        hatch="////", lw=0))
                axS.text(j + 0.5, y + 0.5, "0/3", ha="center", va="center", color="0.55", fontsize=6)
            else:
                st = ps.ph_style(_ph_from_condition(cond))
                axS.add_patch(Rectangle((j, y), 1, 1, facecolor=st["color"], alpha=0.22,
                                        edgecolor="white", lw=0))
                lbl = f"{n_days}/3" + (f"\n({n_runs} runs)" if n_runs != n_days * 3 else "")
                axS.text(j + 0.5, y + 0.5, lbl, ha="center", va="center", fontsize=6, fontweight="bold")
    axS.add_patch(Rectangle((0, 0), 1, len(conds), fill=False, edgecolor="k", lw=1.6, zorder=5))
    axS.text(0.5, len(conds) + 0.10, "only 3-condition\ncommon support", ha="center", va="bottom",
             fontsize=5.5, fontweight="bold")
    axS.set_xlim(0, len(targets))
    axS.set_ylim(0, len(conds))
    axS.set_xticks(np.arange(len(targets)) + 0.5)
    axS.set_xticklabels([f"{g:g}" for g in targets])
    axS.set_yticks(np.arange(len(conds)) + 0.5)
    axS.set_yticklabels([ps.ph_style(_ph_from_condition(c))["label"] for c in reversed(conds)])
    axS.set_xlabel("Matched remaining fraction, g")
    axS.tick_params(length=0)
    for s in axS.spines.values():
        s.set_visible(False)
    written_supp = _savefig(figS, out_dir / "Figure_S2_matched_extent_coverage", formats)
    plt.close(figS)

    # ── per-panel / SI source data (unscaled r_c preserved; scaled column added for display) ──
    def _with_display(df):
        d = df.copy()
        if "r_c" in d:
            d["r_c_x1000"] = (d["r_c"] * R_C_DISPLAY_SCALE).round(5)
        for col in ("r_c_mean", "r_c_sd"):
            if col in d:
                d[f"{col}_x1000"] = (d[col] * R_C_DISPLAY_SCALE).round(5)
        return d

    srcs = {
        "Figure_5_panelA_endpoint_run.csv":
            _with_display(data.endpoint_run.assign(endpoint_type="endpoint")).merge(
                data.endpoint_gend_date.rename(columns={"g": "g_end_date", "n_reps": "nested_run_count"}),
                on=["condition", "date"], how="left"),
        "Figure_5_panelA_endpoint_date.csv":
            _with_display(data.endpoint_date.assign(endpoint_type="endpoint").rename(
                columns={"n_reps": "nested_run_count"})),
        "Figure_5_panelA_endpoint_condition.csv":
            _with_display(data.endpoint_cond.assign(endpoint_type="endpoint")).merge(
                data.endpoint_gend_cond.rename(columns={"g_mean": "g_end_mean", "g_sd": "g_end_sd"}),
                on="condition", how="left"),
        "Figure_5_panelB_matched_g0.8_date.csv":
            _with_display(data.matched_date.assign(endpoint_type="matched", g=data.matched_g).rename(
                columns={"n_reps": "nested_run_count"})),
        "Figure_5_panelB_matched_g0.8_condition.csv":
            _with_display(data.matched_cond.assign(endpoint_type="matched", g=data.matched_g)),
        "Figure_S2_matched_extent_coverage.csv": data.coverage.copy(),
    }
    src_paths = _write_sources(source_dir, srcs)
    return {"figure": written + written_supp, "sources": src_paths}


# ── shared IO ────────────────────────────────────────────────────────────────
def _savefig(fig, stem: Path, formats) -> list:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        pth = stem.with_suffix(f".{fmt}")
        fig.savefig(pth, dpi=ps.DPI, bbox_inches="tight")
        out.append(pth)
    return out


def _write_sources(source_dir: Path, named: dict) -> list:
    source_dir = Path(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name, df in named.items():
        pth = source_dir / name
        df.to_csv(pth, index=False)
        out.append(pth)
    return out
