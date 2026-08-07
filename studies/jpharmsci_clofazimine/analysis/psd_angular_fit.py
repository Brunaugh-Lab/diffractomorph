"""Empirical KWW fits to the angular ΣI decay — nested (run → date × condition → condition).

Fits each run's total angular signal ΣI(t) to the free-amplitude stretched exponential (KWW,
``I(t) = plateau + amp·exp(−(t/τ)^β)``; :func:`empirical_fit.fit_channel_rate`). This gives a
characteristic decay time ``τ``, a stretch ``β``, the plateau ``floor``, the **optical decay depth**
(the fraction of the START scattering that the fit removes — an OPTICAL quantity, *not* the dissolved
mass fraction, since the raw ΣI carries the non-drug ``Σref`` floor), and — key for the ~30 s
injection→first-frame delay — the **back-extrapolated start** ``i0`` (the fitted t=0 scattering; ◆ at
t=0 in panel B).

**Primary empirical input is the raw measured ΣI** (sum of the per-frame ``Measured Value`` over the 31
rings; not background-subtracted). **Primary endpoints are the mean relaxation time ⟨t⟩ and β** (both
robust to the additive ``Σref`` offset, which the free ``plateau`` absorbs). A background-subtracted
``ΣI − Σref`` KWW is fit **only as a sensitivity comparison** (``*_bgsub`` columns), not promoted.

**Nested design (locked).** A fresh suspension **date** is the independent block; the three runs within a
(date × condition) cell are nested replicate runs. Condition-level numbers are therefore built
**day-first**: reps are averaged within (date × condition), then summarized across those date-level means
(``n_days`` reported), never by pooling runs as independent observations.

The N-B forward comparison is kept **separate** from the empirical fit: the model-free ΣI ``t50`` vs the
N-B area ``t50`` (and their ratio) live in their own table, not mixed into the empirical KWW endpoints.

Writes into ``<experiment>/psd_evolution/mass_and_angular_signal/``:
  - ``angular_kww_fits.csv``               — one row per **run** (empirical KWW params + ``_bgsub``
                                             sensitivity; no forward columns),
  - ``angular_kww_by_date_condition.csv``  — reps averaged within (date × condition),
  - ``angular_kww_by_condition.csv``       — summarized across date-level means, with ``n_days``/``n_runs``,
  - ``forward_vs_empirical.csv``           — per run + day-first: model-free ΣI ``t50`` vs N-B area ``t50``,
  - ``_angular_kww_fits.png``              — 2-panel by condition: (A) mass (forward + UV), (B) raw ΣI +
                                             KWW fit with the ◆ back-extrapolated t=0 start,
  - ``_angular_kww_summary_byday.png``     — condition summary, one point per **day** (reps averaged).

Run with the pipeline venv:  ``python psd_angular_fit.py``
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from diffractomorph_pipeline import ingest, kinetics, solubility
from diffractomorph_pipeline.empirical_fit import kww_decay
from diffractomorph_pipeline.processing import AggregateKWWConfig, fit_aggregate_kww
from diffractomorph_pipeline.study import summarize_hierarchy

from forward_predict import V_ML, _arm_a_runs, _arm_b_runs, _ph_dependent_runs, _predict_run, T_END_S
from psd_evolution_common import screen_uv_dissolved
from study_common import BASE as BASE_PH, SUMMARY, find_rtf

SUB = ("psd_evolution", "mass_and_angular_signal")
CMAPS = ["Blues", "Reds", "Greens", "Purples", "Oranges", "Greys"]   # one sequential ramp per condition
EXPERIMENTS = [("pH study", "ph", _ph_dependent_runs),
               ("antisolvent Tween 80 concentration", "arm_a", _arm_a_runs),
               ("arm_b", "arm_b", _arm_b_runs)]


def _safe(s):
    return re.sub(r"[^0-9A-Za-z]+", "_", str(s)).strip("_")


def _short(rid):
    """Short run label for the legend, retaining date when a condition spans days."""
    p = rid.split("_")
    if rid.startswith("pH") and len(p) >= 3:
        return f"{p[1]} {p[2]}"
    if rid.startswith("Tween") and len(p) >= 3:
        return f"{p[-2]} {p[-1]}"
    return p[-1]


def _rtf_for(rd, exp, base):
    """Locate the measurement ``.rtf`` for a forward-model run dict (per experiment)."""
    if rd.get("measurement_rtf") is not None:
        path = str(rd["measurement_rtf"])
        return path if Path(path).exists() else None
    rep = int(rd["rep"])
    if exp == "ph":
        return find_rtf(rd["ph"], int(rd["date"]), rep)
    if exp == "arm_a":
        pct = str(rd["cond"]).split("%")[0].strip()
        pat = f"*Tween {pct}*Measurement*Rep {rep}.rtf"
    else:  # arm_b — LD filenames use "1x", not "1.0x"
        pat = f"*{str(rd['cond']).replace('1.0x', '1x')} Tween Spike*Measurement*Rep {rep}.rtf"
    hits = [x for x in glob.glob(str(base / pat))
            if "blank" not in x.lower() and " q0" not in x.lower()]
    return hits[0] if hits else None


def _uv_for(rd, exp, base):
    """(times_min, dissolved_ugml) measured, from the summary UV table (or None)."""
    rep = int(rd["rep"])
    if exp == "ph":
        uv = pd.read_csv(SUMMARY / "uv_timecourse_all.csv")
        r = uv[(uv.ph == rd["ph"]) & (uv.date == int(rd["date"])) & (uv.rep == rep)]
        diss = "true_dissolved_ugml"
    else:
        name = ("antisolvent_tween80_conc_uv_timecourse.csv"
                if exp == "arm_a" else "arm_b_uv_timecourse.csv")
        hits = glob.glob(str(base / "**" / name), recursive=True)
        if not hits:
            return None
        uv = pd.read_csv(hits[0])
        if exp == "arm_a":
            target_date = pd.to_datetime(rd["date"]).strftime("%Y-%m-%d")
            dates = pd.to_datetime(uv["date"]).dt.strftime("%Y-%m-%d")
            r = uv[(uv.tween_pct_wv.astype(float) == float(str(rd["cond"]).split("%")[0]))
                   & (dates == target_date) & (uv.rep == rep)]
        else:
            # Arm B is replicated: rep numbers repeat across the four independent preps, so the
            # run_id (condition + prep + rep) is the only unambiguous key.
            r = uv[uv.run_id == rd["id"]] if "run_id" in uv else uv.iloc[0:0]
        diss = "conc_ugml"
    r = r.sort_values("time_min")
    if r.empty:
        return None
    t = r.time_min.to_numpy(float)
    d = r[diss].to_numpy(float)
    loaded = (float(r.loaded_ugml.iloc[0]) if "loaded_ugml" in r
              else float(rd["dose_mg"]) * 1e3 / V_ML)   # Arm B: dose from the audited dose table
    keep = screen_uv_dissolved(d, loaded)    # drop impossible / spike UV points
    return t[keep], d[keep]


def _forward_area(run):
    """N-B predicted area decay ``Σ q/r`` (∝ Σ N r²). Returns (t_min, A)."""
    q = np.asarray(run.qundiss, float)
    rad = np.asarray(run.radius_um, float)
    ok = rad > 1e-9
    A = np.where(ok, q / np.where(ok, rad, 1.0), 0.0).sum(axis=1)
    return run.t / 60.0, A


def _t50(t, y):
    """Model-free time to fall halfway from the start to the plateau (last value)."""
    y = np.asarray(y, float)
    y0, yf = y[0], y[-1]
    if y0 <= yf:
        return float("nan")
    below = np.where(y <= yf + 0.5 * (y0 - yf))[0]
    return round(float(t[below[0]]), 3) if below.size else float("nan")


def _fit_figure(fits, out_png, title, cmap_name, *, normalize=False):
    """One figure for a single condition; reps distinguished by a wide light→dark ramp of ``cmap_name``.
    Both legends (runs, marker types) live OFF the panels. With ``normalize`` every curve is scaled to
    its own start: panel A → % of injected, panel B ΣI → % of the fitted start ``i0``, UV dissolved →
    % of injected (so runs overlay by shape, magnitude/dose removed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    n = len(fits)
    cmap = plt.get_cmap(cmap_name)
    shades = np.linspace(0.35, 0.95, n) if n > 1 else [0.7]                        # wide spread per rep
    runcol = {r["id"]: cmap(s) for r, s in zip(fits, shades)}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))
    axB2 = axB.twinx()                                                             # UV dissolved conc
    for r in fits:
        col = runcol[r["id"]]
        loaded, i0 = r["loaded"], r["i0"]
        aA = (100.0 / loaded) if normalize else 1.0                               # panel-A scale (→ % injected)
        aB = (100.0 / i0) if (normalize and i0 > 0) else 1.0                      # ΣI scale (→ % of start)
        if r["fwd_mass"] is not None:
            axA.plot(r["fwd_mass"][0], r["fwd_mass"][1] * aA, "-", lw=1.3, alpha=0.85, color=col)
        if r["uv"] is not None:
            axA.plot(r["uv"][0], r["uv"][1] * aA, "o", ms=4, alpha=0.85, color=col)
        t, sig, f = r["t"], r["sig"], r["f"]
        axB.plot(t, sig * aB, "o", ms=2.5, alpha=0.5, color=col)                   # measured ΣI
        tt = np.linspace(0.0, float(t.max()), 200)
        axB.plot(tt, kww_decay(tt, i0 * f["floor"], i0 * (1 - f["floor"]), f["tau_min"], f["beta"]) * aB,
                 "-", lw=1.5, color=col)                                           # KWW fit
        axB.plot([0.0], [i0 * aB], "D", ms=6, color=col, zorder=5)                 # back-extrapolated t=0
        if r["uv_diss"] is not None:                                               # UV dissolved (right axis)
            axB2.plot(r["uv_diss"][0], r["uv_diss"][1] * aA, "s--", ms=4, lw=0.9, alpha=0.75, color=col)
    if normalize:
        axA.set_ylabel("remaining undissolved (% of injected)"); axA.set_ylim(0, 105)
        axA.set_title("A — remaining undissolved (% of injected): forward (line) + UV (points)", fontsize=10)
        axB.set_ylabel("ΣI  (% of starting)")
        axB.set_title("B — ΣI % of start + KWW fit  (right: UV dissolved % of injected)", fontsize=10)
        axB2.set_ylabel("UV dissolved (% of injected)")
    else:
        axA.set_ylabel("undissolved suspended conc. (µg/mL)")
        axA.set_title("A — undissolved suspended conc: forward (line) + UV (points)", fontsize=10)
        axB.set_ylabel("total angular signal  ΣI (31 channels)")
        axB.set_title("B — raw ΣI + KWW fit  (right: UV dissolved conc)", fontsize=10)
        axB2.set_ylabel("UV dissolved conc. (µg/mL)")
    axB2.set_ylim(bottom=0)
    for ax in (axA, axB):
        ax.set_xlabel("time (min)"); ax.grid(alpha=0.3)

    run_h = [Line2D([0], [0], color=runcol[r["id"]], lw=3, label=_short(r["id"])) for r in fits]
    fig.legend(handles=run_h, loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=8,
               title="run", frameon=False)
    mk = [Line2D([0], [0], color="0.35", lw=1.5, label="KWW fit"),
          Line2D([0], [0], color="0.35", marker="o", ls="", ms=4, label="measured ΣI"),
          Line2D([0], [0], color="0.35", marker="D", ls="", ms=6, label="i0 (back-extrap. t=0)"),
          Line2D([0], [0], color="0.35", marker="s", ls="--", ms=4, label="UV dissolved (right axis)")]
    fig.legend(handles=mk, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close(fig)


def _summary_figure(df, out_png, title, per_day=False):
    """Two panels comparing central tendency + spread across conditions: (A) β (stretch exponent),
    (B) mean relaxation time ⟨t⟩. Each point is a run, or — with ``per_day`` — one day (reps of a date
    averaged first, so the replicate unit is the day); the error bar is mean ± SD over those points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = (df.groupby(["condition", "date"], as_index=False)[["beta", "mean_relax_min"]].mean()
           if per_day else df)
    unit = "day" if per_day else "run"
    conds = sorted(pts.condition.unique())
    color = {c: plt.get_cmap(CMAPS[i % len(CMAPS)])(0.62) for i, c in enumerate(conds)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    specs = [("beta", "β (stretch exponent)", "A — β: decay heterogeneity",
              "higher = more UNIFORM   ·   lower = more HETEROGENEOUS"),
             ("mean_relax_min", "mean relaxation time  ⟨t⟩ (min)", "B — ⟨t⟩: shape-robust rate",
              "higher = SLOWER   ·   lower = FASTER dissolution")]
    for ax, (col, ylab, ttl, interp) in zip(axes, specs):
        for x, c in enumerate(conds):
            v = pts.loc[pts.condition == c, col].dropna().to_numpy(float)
            if not v.size:
                continue
            jit = np.linspace(-0.12, 0.12, v.size) if v.size > 1 else [0.0]
            ax.scatter(x + np.asarray(jit), v, color=color[c], alpha=0.55, s=28, zorder=2)
            m = float(v.mean()); sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
            ax.errorbar([x], [m], yerr=[sd], fmt="o", color=color[c], ms=11, capsize=6,
                        markeredgecolor="k", elinewidth=1.6, zorder=3)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds)
        ax.set_xlim(-0.5, len(conds) - 0.5)
        ax.set_ylabel(ylab)
        ax.set_title(f"{ttl}  (n {unit}s)", fontsize=11, pad=18)
        ax.text(0.5, 1.012, interp, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8, style="italic", color="0.4")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=140); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment", nargs="?", default="all",
        choices=("all", "ph", "arm_a", "antisolvent_tween80_conc", "arm_b"),
        help="experiment to analyze; arm_a is retained as an internal compatibility alias",
    )
    selected = parser.parse_args(argv).experiment
    if selected == "antisolvent_tween80_conc":
        selected = "arm_a"
    cs_model = solubility.load_default()
    for name, exp, builder in EXPERIMENTS:
        if selected != "all" and selected != exp:
            continue
        runs, base, _ = builder()
        rows, fits = [], []
        for rd in runs:
            rtf = _rtf_for(rd, exp, base)
            if not rtf:
                continue
            mrun = ingest.extract_run(rtf)
            primary = fit_aggregate_kww(mrun, AggregateKWWConfig(reference_mode="raw_measured"))
            sensitivity = fit_aggregate_kww(
                mrun, AggregateKWWConfig(reference_mode="reference_adjusted"),
            )
            t, sig, f = primary.time_min, primary.aggregate_signal, primary.fit
            f_bg = sensitivity.fit                                      # sensitivity only; not promoted
            frun, cs = _predict_run(rd, cs_model, T_END_S, 80)
            tA, A = _forward_area(frun)
            i0 = f["i0"] if np.isfinite(f["i0"]) and f["i0"] > 0 else float(sig[0])
            tmax = float(t.max())
            loaded = float(rd["dose_mg"]) * 1e3 / V_ML                  # injected conc = mass / cuvette (µg/mL)
            fmass_t = frun.t / 60.0
            keep = fmass_t <= tmax + 1e-9
            fwd_conc = loaded * (1.0 - np.asarray(frun.pct_dissolved)[keep] / 100.0)   # undissolved µg/mL
            uv_raw = _uv_for(rd, exp, base)
            uv = uv_diss = None
            if uv_raw is not None:                                     # anchor at injected, then measured
                ut, diss = uv_raw
                uv = (np.r_[0.0, ut], np.r_[loaded, loaded - diss])    # undissolved (panel A), from injected
                uv_diss = (np.r_[0.0, ut], np.r_[0.0, diss])           # dissolved (panel B right axis), from 0
            mrt = f["mean_relax_min"]
            mrt_bg = f_bg["mean_relax_min"]
            rows.append(dict(id=rd["id"], condition=str(rd["cond"]), date=str(rd.get("date", "")),
                             loaded_ugml=round(loaded, 2),
                             # PRIMARY empirical endpoints (raw measured ΣI)
                             tau_min=round(f["tau_min"], 3), beta=round(f["beta"], 3),
                             mean_relax_min=round(mrt, 3) if np.isfinite(mrt) else np.nan,
                             t50_min=round(f["t50_min"], 3) if np.isfinite(f["t50_min"]) else np.nan,
                             floor=round(f["floor"], 4),
                             optical_decay_depth_pct=round(100 * f["depth"], 2),   # OPTICAL depth, not dissolved fraction
                             i0_fit=round(i0, 1), meas_frame0=round(float(sig[0]), 1),
                             start_gain_pct=round(100 * (i0 / sig[0] - 1), 1), r2=round(f["r2"], 4),
                             flag=f["flag"],
                             # SENSITIVITY: background-subtracted ΣI − Σref (not promoted)
                             mean_relax_min_bgsub=round(mrt_bg, 3) if np.isfinite(mrt_bg) else np.nan,
                             beta_bgsub=round(f_bg["beta"], 3) if np.isfinite(f_bg["beta"]) else np.nan,
                             optical_decay_depth_pct_bgsub=round(100 * f_bg["depth"], 2)
                             if np.isfinite(f_bg["depth"]) else np.nan,
                             r2_bgsub=round(f_bg["r2"], 4) if np.isfinite(f_bg["r2"]) else np.nan,
                             flag_bgsub=f_bg["flag"],
                             # FORWARD comparison (kept separate from empirical endpoints; split out on write)
                             t50_emp_min=_t50(t, sig), nb_t50_min=_t50(tA, A),
                             t50_ratio=(round(_t50(t, sig) / _t50(tA, A), 2) if _t50(tA, A) else np.nan)))
            fits.append(dict(id=rd["id"], cond=str(rd["cond"]), t=t, sig=sig, f=f, i0=i0, loaded=loaded,
                             uv=uv, uv_diss=uv_diss, fwd_mass=(fmass_t[keep], fwd_conc)))
        if not rows:
            print(f"  [{name}] no runs"); continue
        df = pd.DataFrame(rows)
        # Study-scoped arms keep their outputs under the study's own analysis/ folder; only the
        # pH study uses the psd_evolution/ tree.
        out = {"arm_a": base / "analysis" / "antisolvent_tween80_conc_empirical",
               "arm_b": base / "analysis" / "empirical_angular_kww"}.get(
                   exp, base / SUB[0] / SUB[1])
        out.mkdir(parents=True, exist_ok=True)

        # ── per-RUN empirical table (empirical KWW only; forward columns split out below) ──
        EMP_COLS = ["id", "condition", "date", "loaded_ugml", "tau_min", "beta", "mean_relax_min", "t50_min",
                    "floor", "optical_decay_depth_pct", "i0_fit", "meas_frame0", "start_gain_pct", "r2",
                    "flag", "mean_relax_min_bgsub", "beta_bgsub", "optical_decay_depth_pct_bgsub",
                    "r2_bgsub", "flag_bgsub"]
        df[EMP_COLS].to_csv(out / "angular_kww_fits.csv", index=False)

        # ── day-first aggregation: reps → (date × condition) → condition (across date means) ──
        PRIMARY = ["tau_min", "beta", "mean_relax_min", "t50_min", "optical_decay_depth_pct", "i0_fit"]
        SENS = ["mean_relax_min_bgsub", "beta_bgsub", "optical_decay_depth_pct_bgsub"]
        hierarchy_input = df.rename(columns={"id": "run_id", "date": "independent_unit_id"})
        hierarchy = summarize_hierarchy(
            hierarchy_input,
            value_columns=tuple(PRIMARY + SENS),
        )
        by_dc = hierarchy.independent_units.rename(
            columns={"independent_unit_id": "date"},
        ).round(4)
        by_dc.to_csv(out / "angular_kww_by_date_condition.csv", index=False)
        by_condition = hierarchy.conditions.rename(
            columns={"n_independent_units": "n_days"},
        ).round(4)
        by_condition.to_csv(out / "angular_kww_by_condition.csv", index=False)

        # ── forward-vs-empirical comparison, kept SEPARATE from the empirical endpoints ──
        FWD = ["t50_emp_min", "nb_t50_min", "t50_ratio"]
        fdc = df.groupby(["condition", "date"], as_index=False)[FWD].mean().round(3)
        fcond = fdc.groupby("condition")[FWD].agg(["mean", "std"]).round(3)
        fcond.columns = [f"{a}_{b}" for a, b in fcond.columns]
        fcond["n_days"] = fdc.groupby("condition").size()
        with open(out / "forward_vs_empirical.csv", "w") as fh:
            fh.write("# per-run model-free ΣI t50 vs N-B area t50 (empirical KWW kept separate)\n")
        df[["id", "condition", "date"] + FWD].to_csv(out / "forward_vs_empirical.csv", mode="a", index=False)
        fcond.to_csv(out / "forward_vs_empirical_by_condition.csv")
        (out / "_angular_kww_fits.png").unlink(missing_ok=True)        # retire the old combined figure
        conds = sorted({r["cond"] for r in fits})
        for i, c in enumerate(conds):
            cf = [r for r in fits if r["cond"] == c]
            _fit_figure(cf, out / f"_angular_kww_fits_{_safe(c)}.png", f"{name} — {c}",
                        CMAPS[i % len(CMAPS)])
            _fit_figure(cf, out / f"_angular_kww_fits_{_safe(c)}_norm.png",
                        f"{name} — {c}  (normalized to start)", CMAPS[i % len(CMAPS)], normalize=True)
        _summary_figure(df, out / "_angular_kww_summary.png",
                        f"{name} — KWW fit summary by condition (per run)")
        if df.groupby("condition").date.nunique().max() > 1:      # only when a condition spans days
            _summary_figure(df, out / "_angular_kww_summary_byday.png",
                            f"{name} — KWW summary, one point per day (reps averaged)", per_day=True)
        print(f"  [{name}] {len(df)} runs, {len(conds)} conditions → CSVs + per-condition figures + summary")


if __name__ == "__main__":
    main()
