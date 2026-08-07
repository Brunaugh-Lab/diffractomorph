"""Run the forward dissolution model per run → a ``forward_prediction/`` folder.

Mode 1 (from injection): the QC suspension PSD + injected dose + pH + measured Cs(pH) drive the
mechanistic surface-pH model (``forward.predict``) with NO fitting and no use of the dissolution
trace. For each experiment we write, into ``<experiment>/forward_prediction/``:

  - ``predictions.csv``   — per-run summary (dose, Cs, supersaturation, ceiling, predicted final %)
  - ``trajectories.csv``  — long (run, t_min, % dissolved of the injected dose)
  - ``forward_prediction.png`` — the predicted trajectories

Run with the pipeline venv.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from diffractomorph_pipeline.config import data_root
from diffractomorph_pipeline import solubility
from diffractomorph_pipeline.assay import uv_timecourse
from diffractomorph_pipeline.forward import PSD, predict, predict_from_snapshot, q3_evolution

from arm_a_common import default_study_root, find_measurement_rtf, find_q3_source, find_qc_psd

DISSO = data_root() / "disso_experiments"
T_END_S = 1800.0
V_ML = 40.0
MW_CFZ = 473.4


def _run_and_write(runs, out_dir, title):
    """``runs`` = list of dicts {id, date, ph, rep, dose_mg, psd, **meta[, cond]}.

    Runs Mode 1 for each and writes predictions.csv / trajectories.csv / forward_prediction.png
    into ``out_dir``.
    """
    cs_model = solubility.load_default()
    results = []                                        # (run, summary_row, trajectory_rows)
    for r in runs:
        ph, dose_mg, psd = float(r["ph"]), float(r["dose_mg"]), r["psd"]
        run, cs = _predict_run(r, cs_model, T_END_S, 80)
        loaded = dose_mg * 1e3 / V_ML                       # µg/mL
        row = dict(
            id=r["id"], date=r.get("date"), ph=ph, rep=r.get("rep"),
            **({"condition": r["cond"]} if r.get("cond") else {}),
            qc_Dv50=round(psd.dv50, 2), qc_D32=round(psd.d32, 2),
            dose_mg=round(dose_mg, 4), loaded_ugml=round(loaded, 2), cs_ugml=round(cs, 2),
            supersat=round(loaded / cs, 2), ceiling_frac=round(min(1.0, cs / loaded), 3),
            pred_final_frac=round(run.frac_dissolved_final, 3),
            surf_pH0=round(float(run.ph_surf[0]), 2))
        traj = [dict(id=r["id"], ph=ph, t_min=round(float(t), 2), pct_dissolved=round(float(pct), 2))
                for t, pct in zip(run.t / 60.0, run.pct_dissolved)]
        results.append((r, row, traj))

    pred = _write(results, out_dir, title)              # experiment-level summary (all runs)
    groups = {}                                         # + one folder per condition
    for item in results:
        cd = item[0].get("cond_out")
        if cd is not None:
            groups.setdefault(str(cd), []).append(item)
    for cd_str, grp in sorted(groups.items()):
        _write(grp, Path(cd_str), f"{title} — {grp[0][0].get('cond', Path(cd_str).name)}")
    if "ph" in pred:
        print("  predicted final % dissolved, median by pH:")
        print((pred.groupby("ph").pred_final_frac.median() * 100).round(0).to_string())
    return pred


def _frame_to_psd(lines):
    """One q3 frame's lines (PAQXOS ``xo,Q3,…,xm`` table) → PSD."""
    hi = next((i for i, r in enumerate(lines) if r.startswith("xo")), None)
    if hi is None:
        return None
    Q3, xm = [], []
    for r in lines[hi + 1:]:
        if r.startswith("xo"):
            break                                       # next frame block (stacked file)
        p = r.split(",")
        if len(p) >= 5:
            try:
                q, x = float(p[1]), float(p[4])
            except ValueError:
                continue
            Q3.append(q); xm.append(x)
    rel = np.clip(np.diff(np.asarray(Q3), prepend=0.0), 0.0, None) if len(Q3) >= 5 else np.array([])
    return PSD.from_q3(xm, rel / rel.sum()) if rel.sum() > 0 else None


def _q3_frame_at(source, t_min, cadence_s=12.0):
    """The LD q3 frame nearest ``t_min`` → PSD. ``source`` = a folder of per-frame CSVs
    (ph_dependent) or one stacked CSV with many frame blocks (arm_a/arm_b)."""
    if source is None:
        return None
    idx = int(round(t_min * 60.0 / cadence_s))
    source = Path(source)
    if source.is_dir():
        files = sorted(glob.glob(str(source / "*.csv")))
        if not files:
            return None
        return _frame_to_psd(Path(files[min(idx, len(files) - 1)]).read_text(encoding="utf-8-sig").splitlines())
    if not source.exists():
        return None
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    starts = [i for i, r in enumerate(lines) if r.startswith("xo")]
    if not starts:
        return None
    idx = min(idx, len(starts) - 1)
    hi = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
    return _frame_to_psd(lines[starts[idx]:hi])


def _nice_log_size_axis(ax):
    """Label a log size axis at readable µm values (1, 2, 3, 5, 10, 20…) instead of 10⁰/10¹."""
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _predict_run(r, cs_model, t_end, n_eval, morph=None):
    """Run Mode 1 for one run dict, honouring an explicit per-run ``cs_ugml`` (→ S0 override).
    ``morph`` (optional :class:`MorphologyParams`) selects the exploratory independent-exponent
    model instead of the base Nernst–Brunner. Returns (DissolutionRun, cs_ugml_used)."""
    ph, dose_mg, psd = float(r["ph"]), float(r["dose_mg"]), r["psd"]
    kw = {}
    if r.get("cs_ugml") is not None:                    # explicit Cs (e.g. Tween ladder) → S0 reproducing it
        cs = float(r["cs_ugml"])
        kw["s0_uM"] = cs * cs_model.s0_ugml / cs_model.branch(ph) / 1e3 / MW_CFZ * 1e6
    else:
        cs = cs_model.cs_for_ph(ph)
    return predict(psd, ph=ph, dose_mg=dose_mg, drug="CFZ", t_end=t_end, n_eval=n_eval,
                   morph=morph, **kw), cs


def _write(results, out_dir, title):
    """Write predictions.csv / trajectories.csv / forward_prediction.png for a set of results."""
    rows = [row for (_, row, _) in results]
    traj = [t for (_, _, ts) in results for t in ts]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame(rows).sort_values([c for c in ("ph", "date", "rep") if c in rows[0]])
    pred.to_csv(out_dir / "predictions.csv", index=False)
    tdf = pd.DataFrame(traj); tdf.to_csv(out_dir / "trajectories.csv", index=False)
    _plot(tdf, pred, out_dir / "forward_prediction.png", title)
    print(f"  wrote {out_dir}  ({len(pred)} runs)")
    return pred


def _plot(tdf, pred, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    phs = sorted(tdf.ph.unique())
    cmap = {p: c for p, c in zip(phs, ["#1b7837", "#762a83", "#b2182b", "#2166ac"])}
    for rid, g in tdf.groupby("id"):
        ph = g.ph.iloc[0]
        ax.plot(g.t_min, g.pct_dissolved, lw=1.2, alpha=0.8, color=cmap.get(ph, "k"))
    for ph in phs:
        ax.plot([], [], color=cmap.get(ph, "k"), label=f"pH {ph}")
    ax.set_xlabel("time (min)"); ax.set_ylabel("% of injected dose dissolved")
    ax.set_title(title, fontsize=11); ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"  wrote {out_png}")


# ── ph-dependent dissolution study ───────────────────────────────────────────
def _ph_dependent_runs():
    base = DISSO / "ph_dependent_dissolution_study"
    meta = pd.read_csv(base / "summary" / "run_metadata.csv")
    qc = base / "QC" / "drug_antisolvent_daily_suspension_qc"
    psd_cache, runs = {}, []
    for r in meta.itertuples():
        date, ph, rep = int(r.date_i), float(r.ph), int(r.rep)
        if date not in psd_cache:
            hits = glob.glob(str(qc / str(date) / "size" / f"CFZ QC Q0 {date}"))
            psd_cache[date] = PSD.from_sympatec(hits[0]) if hits else None
        if psd_cache[date] is None:
            print(f"  ! no QC size anchor for {date}"); continue
        q3 = glob.glob(str(base / "CFZ q3 csv" / f"pH={ph}" / f"*{date}*" / f"Rep {rep}"))
        uv = glob.glob(str(base / f"ph_{ph}" / f"{date}_pH*" / "UV-VIs" / f"pH={ph}*Rep{rep}.xlsx"))
        runs.append(dict(id=f"pH{ph}_{date}_R{rep}", date=date, ph=ph, rep=rep, cond=f"pH {ph}",
                         dose_mg=float(r.mass_mg), psd=psd_cache[date],       # dose = run_metadata (correct DF)
                         q3_source=q3[0] if q3 else None, uv_file=uv[0] if uv else None,
                         cond_out=base / f"ph_{ph}" / "forward_prediction"))  # into the pH condition folder
    return runs, base, "pH-dependent dissolution — forward prediction (Mode 1, from injection)"


def run_ph_dependent():
    runs, base, title = _ph_dependent_runs()
    return _run_and_write(runs, base / "forward_prediction", title)


# ── antisolvent Tween 80 concentration, pH 4.5 ──────────────────────────────
def _arm_a_runs(study_root=None):
    """Build replicated concentration-study runs from the local dose table and daily QC PSDs."""
    base = Path(study_root) if study_root is not None else default_study_root()
    dose = pd.read_csv(base / "analysis" / "antisolvent_tween80_conc_injected_mass.csv")
    psd_cache, runs = {}, []
    for r in dose.itertuples():
        pct, day, rep = f"{float(r.tween_pct_wv):.2f}", int(r.day), int(r.rep)
        key = (pct, day)
        if key not in psd_cache:
            psd_cache[key] = PSD.from_sympatec(find_qc_psd(base, pct, day))
        date = str(r.date)
        runs.append(dict(id=f"Tween{pct}_{date}_R{rep}", date=date, day=day, ph=4.5, rep=rep,
                         cond=f"{pct}% Tween", dose_mg=float(r.injected_mass_mg),
                         psd=psd_cache[key],
                         q3_source=find_q3_source(base, pct, day, rep),
                         measurement_rtf=find_measurement_rtf(base, pct, day, rep),
                         cond_out=base / "analysis" / "forward_prediction" / f"tween_{pct}"))
    return runs, base, "Antisolvent Tween 80 concentration — forward prediction (Mode 1, from injection)"


def run_arm_a():
    runs, base, title = _arm_a_runs()
    return _run_and_write(runs, base / "analysis" / "forward_prediction", title)


# ── in-medium Tween micelle effect (Arm B), pH 4.5 ───────────────────────────
def _arm_b_runs(study_root=None):
    """Build the replicated Arm B runs from the audited discovery, dose table and primary Cs.

    Rewritten for the replicated study: 27 runs over 12 independent suspension preps (the prep is
    the injection date, not the date folder — see :mod:`arm_b_common`). Dose comes per replicate
    from :mod:`arm_b_injected_mass`, the starting PSD from the q0 QC routed to that same
    replicate, and Cs from the primary filtered 48 h ladder (:mod:`arm_b_cs`), supplied per run so
    the micellar solubility is inside the model rather than credited to kinetics.
    """
    import arm_b_cs
    from arm_b_common import default_study_root as _arm_b_root, discover_runs
    from arm_b_injected_mass import build_dose_table

    base = Path(study_root) if study_root is not None else _arm_b_root()
    cs = arm_b_cs.cs_map()                       # primary ladder
    runs_tbl = discover_runs(base)
    dose = build_dose_table(base).set_index("run_id")["injected_mass_mg"]
    psd_cache, runs = {}, []
    for r in runs_tbl.itertuples():
        key = str(r.qc_psd_dir)
        if key not in psd_cache:
            psd_cache[key] = PSD.from_sympatec(r.qc_psd_dir)
        tag = r.condition.split()[0]             # "0.5x" / "1.0x" / "10x"
        runs.append(dict(id=r.run_id, date=r.prep, ph=4.5, rep=int(r.rep), cond=r.condition,
                         prep=r.prep, cs_ugml=cs[r.condition],
                         dose_mg=float(dose.loc[r.run_id]), psd=psd_cache[key],
                         q3_source=str(r.q3_dir), uv_file=str(r.uv_file),
                         measurement_rtf=str(r.rtf),
                         cond_out=base / "analysis" / "forward_prediction" / f"tween_{tag}"))
    return runs, base, ("Arm B in-medium Tween — forward prediction "
                        "(Mode 1, primary filtered Cs, n=4 preps per condition)")


def run_arm_b():
    runs, base, title = _arm_b_runs()
    return _run_and_write(runs, base / "analysis" / "forward_prediction", title)


BUILDERS = {"ph_dependent": _ph_dependent_runs, "arm_a": _arm_a_runs, "arm_b": _arm_b_runs}


# ── Mode 1 vs Mode 2 vs measured UV (ph-dependent) ───────────────────────────
def _q3_frame0_psd(base, ph, date, rep):
    """First LD q3 frame → PSD (the eroded first snapshot, ≈ injected suspension)."""
    hits = glob.glob(str(base / "CFZ q3 csv" / f"pH={ph}" / f"*{date}*" / f"Rep {rep}"))
    csvs = sorted(glob.glob(str(Path(hits[0]) / "*.csv"))) if hits else []
    if not csvs:
        return None
    lines = Path(csvs[0]).read_text(encoding="utf-8-sig").splitlines()
    hi = next((i for i, r in enumerate(lines) if r.startswith("xo")), None)
    if hi is None:
        return None
    Q3, xm = [], []
    for r in lines[hi + 1:]:
        p = r.split(",")
        if len(p) >= 5:
            try:
                q, x = float(p[1]), float(p[4])
            except ValueError:
                continue
            Q3.append(q); xm.append(x)
    relvol = np.clip(np.diff(np.asarray(Q3), prepend=0.0), 0.0, None) if Q3 else np.array([])
    if relvol.sum() <= 0:
        return None
    return PSD.from_q3(xm, relvol / relvol.sum())


def _uv_file(base, ph, date, rep):
    hits = glob.glob(str(base / f"ph_{ph}" / f"{date}_pH*" / "UV-VIs" / f"pH={ph}*Rep{rep}.xlsx"))
    return hits[0] if hits else None


def _plot_compare(group, ph, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, (m1, t0, m2, t_uv, pct_meas) in enumerate(group):
        first = i == 0
        ax.plot(m1.t / 60.0, m1.pct_dissolved, color="#2166ac", lw=1.0, ls="--", alpha=0.7,
                label="Mode 1 (from injection)" if first else None)
        if m2 is not None:
            ax.plot(t0 + m2.t / 60.0, m2.pct_dissolved, color="#1b7837", lw=1.4, alpha=0.85,
                    label="Mode 2 (from UV+LD snapshot)" if first else None)
        ax.plot(t_uv, pct_meas, "o", color="k", ms=4, alpha=0.7,
                label="measured UV" if first else None)
    ax.axhline(100, ls=":", color="gray", lw=0.8)
    ax.set_xlabel("time since dose (min)"); ax.set_ylabel("% of injected dose dissolved")
    ax.set_title(f"pH {ph} — forward model vs measured", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def run_compare_ph_dependent():
    """Overlay Mode 1 (injection), Mode 2 (grounded snapshot), and measured UV — all % of dose."""
    base = DISSO / "ph_dependent_dissolution_study"
    meta = pd.read_csv(base / "summary" / "run_metadata.csv")
    qc = base / "QC" / "drug_antisolvent_daily_suspension_qc"
    psd_cache, summary, curves = {}, [], {}
    for r in meta.itertuples():
        date, ph, rep, dose = int(r.date_i), float(r.ph), int(r.rep), float(r.mass_mg)
        if date not in psd_cache:
            h = glob.glob(str(qc / str(date) / "size" / f"CFZ QC Q0 {date}"))
            psd_cache[date] = PSD.from_sympatec(h[0]) if h else None
        qcpsd, uv, psd0 = psd_cache[date], _uv_file(base, ph, date, rep), _q3_frame0_psd(base, ph, date, rep)
        if qcpsd is None or uv is None or psd0 is None:
            continue
        try:
            df = uv_timecourse(uv, ph, injected_mg=dose, volume_mL=V_ML)
        except ValueError:
            continue                                    # endpoint-only plate (no kinetic UV table)
        if len(df) < 2:
            continue
        C0, t0 = float(df.conc_ugml.iloc[0]), float(df.time_min.iloc[0])
        m1 = predict(qcpsd, ph=ph, dose_mg=dose, drug="CFZ", t_end=T_END_S, n_eval=80)
        try:
            m2 = predict_from_snapshot(psd0, ph=ph, injected_mg=dose, conc_ugml=C0,
                                       volume_mL=V_ML, drug="CFZ",
                                       t_end=max(60.0, T_END_S - t0 * 60.0), n_eval=70)
        except ValueError:
            m2 = None                                   # anchor already exceeds the injected dose
        curves.setdefault(ph, []).append((m1, t0, m2, df.time_min.values, df.pct_injected.values))
        summary.append(dict(id=f"pH{ph}_{date}_R{rep}", ph=ph, date=date, rep=rep, dose_mg=round(dose, 4),
                            uv_t0_min=t0, uv_C0_ugml=round(C0, 2),
                            pct_inj_anchor=round(float(df.pct_injected.iloc[0]), 1),
                            mode1_final_pct=round(float(m1.pct_dissolved[-1]), 1),
                            mode2_final_pct=round(float(m2.pct_dissolved[-1]), 1) if m2 is not None else None,
                            measured_final_pct=round(float(df.pct_injected.iloc[-1]), 1)))

    out = base / "forward_prediction"; out.mkdir(parents=True, exist_ok=True)
    sdf = pd.DataFrame(summary).sort_values(["ph", "date", "rep"])
    sdf.to_csv(out / "compare_summary.csv", index=False)
    for ph, group in curves.items():
        cond = base / f"ph_{ph}" / "forward_prediction"; cond.mkdir(parents=True, exist_ok=True)
        _plot_compare(group, ph, cond / "compare.png")
        _plot_compare(group, ph, out / f"compare_pH{ph}.png")
        sdf[sdf.ph == ph].to_csv(cond / "compare.csv", index=False)
    print(f"  wrote {out}/compare_summary.csv  ({len(sdf)} runs)")
    print("  median final % dissolved (Mode 1 / Mode 2 / measured) by pH:")
    print(sdf.groupby("ph")[["mode1_final_pct", "mode2_final_pct", "measured_final_pct"]].median().round(0).to_string())
    return sdf


# ── predicted q3 size-distribution evolution (Mode 1, from injection) ─────────
def _plot_q3_evolution(q3, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    times = np.array(sorted(q3.time_min.unique()))
    norm = Normalize(0.0, times.max())
    fig, ax = plt.subplots(figsize=(8, 5))
    for tm in times:                                    # every frame (12 s cadence), coloured by time
        g = q3[q3.time_min == tm].sort_values("size_um")
        ax.plot(g.size_um, g.q3_pct, "-", lw=0.8, alpha=0.6, color=plt.cm.viridis(norm(tm)))
    ax.set_xscale("log"); ax.set_xlabel("size (µm)"); ax.set_ylabel("q3 (volume %)")
    ax.set_title(title, fontsize=11); ax.grid(alpha=0.3, which="both"); _nice_log_size_axis(ax)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
    cb.set_label("time since dose (min)")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def _plot_buckets(run, out_png, title):
    """Multipanel: per populated size bucket, % of its initial volume still solid vs time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    q = np.asarray(run.qundiss, float); q0 = q[0]
    diam0 = np.asarray(run.diam0_um, float); t = np.asarray(run.t, float) / 60.0
    live = np.where(q0 > q0.max() * 1e-4)[0]
    ncol = 4; nrow = int(np.ceil(len(live) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.1 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for k, i in enumerate(live):
        ax = axes[k]
        ax.plot(t, 100.0 * q[:, i] / q0[i], color="#2166ac", lw=1.5)
        ax.axhline(50, ls=":", color="gray", lw=0.6)
        ax.set_title(f"{diam0[i]:.2f} µm", fontsize=9); ax.set_ylim(0, 105); ax.grid(alpha=0.3)
    for k in range(len(live), len(axes)):
        axes[k].axis("off")
    for k in range(len(axes)):
        if k >= len(axes) - ncol:
            axes[k].set_xlabel("time (min)", fontsize=8)
        if k % ncol == 0:
            axes[k].set_ylabel("% remaining", fontsize=8)
    fig.suptitle(title, fontsize=12); fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=140); plt.close(fig)


def _psd_and_buckets(runs, base):
    """Per run: predict from injection over 20 min (12 s cadence) → q3 evolution + per-bucket kinetics."""
    from diffractomorph_pipeline.forward import bucket_kinetics
    cs_model = solubility.load_default()
    out = base / "forward_prediction" / "psd_evolution"; out.mkdir(parents=True, exist_ok=True)
    for r in runs:
        run, _ = _predict_run(r, cs_model, 1200.0, 101)   # 20 min, every 12 s
        rid = r["id"]
        q3_evolution(run).to_csv(out / f"{rid}_q3.csv", index=False)
        _plot_q3_evolution(q3_evolution(run), out / f"{rid}_q3.png", f"{rid} — predicted q3 evolution (from injection)")
        bucket_kinetics(run).to_csv(out / f"{rid}_bucket_params.csv", index=False)
        _plot_buckets(run, out / f"{rid}_buckets.png", f"{rid} — per-size-bucket dissolution")
    print(f"  wrote {out}  ({len(runs)} runs × q3 + bucket params/figures)")


def _predict_snapshot_run(r, cs_model, psd_snap, C0, t0_min):
    """Mode 2 for one run: anchor at the UV+LD snapshot; ladder Cs (if any) → S0 override."""
    ph, dose = float(r["ph"]), float(r["dose_mg"])
    kw = {}
    if r.get("cs_ugml") is not None:
        kw["s0_uM"] = float(r["cs_ugml"]) * cs_model.s0_ugml / cs_model.branch(ph) / 1e3 / MW_CFZ * 1e6
    t_end = max(120.0, 1200.0 - t0_min * 60.0)          # run to ~20 min absolute
    return predict_from_snapshot(psd_snap, ph=ph, injected_mg=dose, conc_ugml=C0,
                                 volume_mL=V_ML, drug="CFZ",
                                 t_end=t_end, n_eval=int(t_end / 12.0) + 1, **kw)


def _mode2_and_buckets(runs, base):
    """Mode 2 per run: anchor the model at the LD frame + UV concentration at the first UV timepoint
    (not the injected frame). Writes trajectory + per-bucket kinetics into forward_prediction/mode2/."""
    from diffractomorph_pipeline.forward import bucket_kinetics
    cs_model = solubility.load_default()
    out = base / "forward_prediction" / "mode2"; out.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in runs:
        if not r.get("q3_source") or not r.get("uv_file"):
            continue
        try:
            df = uv_timecourse(r["uv_file"], float(r["ph"]), injected_mg=float(r["dose_mg"]), volume_mL=V_ML)
        except ValueError:
            continue                                    # endpoint-only UV plate (no kinetic table)
        if len(df) < 2:
            continue
        t0, C0 = float(df.time_min.iloc[0]), float(df.conc_ugml.iloc[0])
        psd_snap = _q3_frame_at(r["q3_source"], t0)     # LD frame AT the UV timepoint, not frame 0
        if psd_snap is None:
            continue
        try:
            run = _predict_snapshot_run(r, cs_model, psd_snap, C0, t0)
        except ValueError:
            print(f"  ! {r['id']}: UV anchor exceeds injected dose; skipped"); continue
        rid = r["id"]
        pd.DataFrame({"t_min_since_snapshot": (run.t / 60.0).round(2),
                      "t_min_since_dose": (t0 + run.t / 60.0).round(2),
                      "pct_dissolved": run.pct_dissolved.round(2)}).to_csv(out / f"{rid}_mode2_traj.csv", index=False)
        bucket_kinetics(run).to_csv(out / f"{rid}_mode2_bucket_params.csv", index=False)
        _plot_buckets(run, out / f"{rid}_mode2_buckets.png",
                      f"{rid} — Mode 2 per-size-bucket dissolution (anchored at UV+LD snapshot, {t0:.0f} min)")
        n += 1
    print(f"  wrote {out}  ({n} runs × Mode 2 traj + bucket params/figures)")


def _plot_rate_compare(cmp, out_png, rid):
    """Two panels: per-bucket τ for both modes, and the Mode2/Mode1 slowdown ratio vs size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    a.loglog(cmp.size_um, cmp.tau_mode1_min, "o-", color="#2166ac", label="Mode 1 (from injection)")
    a.loglog(cmp.size_um, cmp.tau_mode2_min, "s-", color="#b2182b", label="Mode 2 (from UV+LD snapshot)")
    a.set_xlabel("size (µm)"); a.set_ylabel("bucket dissolution time τ (min)")
    a.grid(alpha=0.3, which="both"); a.legend(fontsize=9, frameon=False)
    a.set_title("Per-bucket rate", fontsize=11); _nice_log_size_axis(a)
    b.semilogx(cmp.size_um, cmp.tau_ratio, "o-", color="#333")
    b.axhline(1.0, ls=":", color="gray"); b.set_xlabel("size (µm)")
    b.set_ylabel("τ ratio  (Mode 2 / Mode 1)  =  slowdown"); b.grid(alpha=0.3, which="both")
    b.set_title("Slowdown from the already-filled sink", fontsize=11); _nice_log_size_axis(b)
    fig.suptitle(f"{rid} — per-bucket dissolution rate: Mode 1 vs Mode 2", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out_png, dpi=140); plt.close(fig)


def _bucket_rate_compare(runs, base):
    """Per run: compute Mode 1 + Mode 2 bucket kinetics and compare per-bucket τ.
    Writes bucket_rate_compare/<id>_bucket_rate_compare.csv + .png."""
    from diffractomorph_pipeline.forward import bucket_kinetics, compare_bucket_kinetics
    cs_model = solubility.load_default()
    out = base / "forward_prediction" / "bucket_rate_compare"; out.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in runs:
        if not r.get("q3_source") or not r.get("uv_file"):
            continue
        try:
            df = uv_timecourse(r["uv_file"], float(r["ph"]), injected_mg=float(r["dose_mg"]), volume_mL=V_ML)
        except ValueError:
            continue
        if len(df) < 2:
            continue
        t0, C0 = float(df.time_min.iloc[0]), float(df.conc_ugml.iloc[0])
        psd_snap = _q3_frame_at(r["q3_source"], t0)
        if psd_snap is None:
            continue
        run1, _ = _predict_run(r, cs_model, 1200.0, 101)            # Mode 1 (from injection)
        try:
            run2 = _predict_snapshot_run(r, cs_model, psd_snap, C0, t0)   # Mode 2 (from snapshot)
        except ValueError:
            continue
        cmp = compare_bucket_kinetics(bucket_kinetics(run1), bucket_kinetics(run2), labels=("mode1", "mode2"))
        if cmp.empty:
            continue
        cmp.to_csv(out / f"{r['id']}_bucket_rate_compare.csv", index=False)
        _plot_rate_compare(cmp, out / f"{r['id']}_bucket_rate_compare.png", r["id"])
        n += 1
    print(f"  wrote {out}  ({n} runs)")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "compare":
        run_compare_ph_dependent(); sys.exit()
    if which == "ratecompare":
        for name, build in BUILDERS.items():
            print(f"[{name}]")
            runs, base, _ = build()
            _bucket_rate_compare(runs, base)
        sys.exit()
    if which in ("psd", "psd_evolution"):
        for name, build in BUILDERS.items():
            print(f"[{name}]")
            runs, base, _ = build()
            _psd_and_buckets(runs, base)
        sys.exit()
    if which == "mode2":
        for name, build in BUILDERS.items():
            print(f"[{name}]")
            runs, base, _ = build()
            _mode2_and_buckets(runs, base)
        sys.exit()
    if which in ("ph", "all"):
        run_ph_dependent()
    if which in ("arm_a", "all"):
        run_arm_a()
    if which in ("arm_b", "all"):
        run_arm_b()
