"""Arm B — solubility-normalized dissolution rate vs in-medium Tween.

k per run (size-free Nernst–Brunner, integral form, same as ``nb_trajectory_fit``):

    M(t) = k · ∫₀ᵗ (Copt − Copt∞)·(Cs − C) dt        →  k = median_late( M_UV / integral )

with Copt(t) from LD, C(t) from UV, and Cs from an Arm C ladder (:mod:`arm_b_cs`).

**Reading k.** Cs is already inside k, in the ``(Cs − C)`` driving force in the denominator.
So k is *the rate constant that remains once solubility is accounted for*:

    k roughly CONSTANT across Tween   → a solubility-only effect; the extra dissolution rate is
                                       what the raised Cs buys (rugosity / "just solubility")
    k CHANGES with Tween              → an effect solubility does not contain, i.e. an
                                       interfacial/kinetic change (partition)

A previous version additionally divided the k ratio by the Cs ratio (``k_ratio_over_cs_ratio``).
That was a **second** Cs normalization with no derived physical meaning — Cs had already been
divided out once — and it is removed. Compare k directly.

**Cs is the primary filtered 48 h ladder** (:mod:`arm_b_cs`), the operational saturation
solubility selected prospectively from the 48 h centrifuge-vs-filter experiment. The
particulate-containing centrifuged ladders are available only as secondary *method*
sensitivities. The optical preprocessing mode (:mod:`arm_b_optical`) is a genuine robustness
axis and the headline result is reported across all three modes. Both choices, and the UV
calibration convention, are recorded in the provenance sidecar.

**Design.** n = 4 independent suspension preps per condition, keyed by injection date, with
1/2/3/3 technical repeats. Technical repeats collapse to a per-prep value first; prep means are
combined unweighted. Every condition-level output here is prep-balanced.

Run with the pipeline venv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit

from diffractomorph_pipeline.assay import uv_timecourse
from diffractomorph_pipeline.optics import mie

import arm_b_cs
from arm_b_common import (
    CONDITIONS,
    condition_pairs_sharing_preps,
    default_study_root,
    discover_runs,
    replication_summary,
)
import arm_b_optical
from arm_b_optical import optical_run
from arm_b_provenance import provenance_record, write_provenance

PH, V_ML = 4.5, 40.0
# Conditions where the *centrifuged* preparation disagrees most with the selected filtered Cs.
# This is a diagnostic of particulate carryover in the centrifuged read (the levels a one-minute
# spin cannot clear) — NOT a caveat on the filtered Cs, which is the operational measurement.
CENTRIFUGE_FILTER_DISAGREEMENT = {"0.5x CMC", "1.0x CMC"}
COLORS = {"0.5x CMC": "#718096", "1.0x CMC": "#2b6cb0", "10x CMC": "#c05621"}
TAU_BOUNDS = (0.05, 30.0)                  # minutes; fits landing here are not measurements
TAU_BOUND_TOL = 1e-3
MIN_DECAY_DEPTH = 0.10                     # a channel must actually fall to time a decay
MIN_FIT_R2 = 0.80

# Back-compat for callers that only need the default ladder's numbers.
CS_UGML = None                             # populated lazily by cs_for()


def cs_for(ladder: str = arm_b_cs.DEFAULT_LADDER) -> dict[str, float]:
    global CS_UGML
    CS_UGML = arm_b_cs.cs_map(ladder)
    return CS_UGML


def uv_curve(xlsx, cs):
    tc = uv_timecourse(xlsx, ph=PH, cs_ugml=cs)
    return tc["time_min"].to_numpy(), tc["conc_ugml"].to_numpy()


def fit_k(tC, copt, tU, CU, Cs):
    """k and its late-time stability; ``floor`` is the terminal Copt the amplitude is measured against."""
    floor = float(np.mean(copt[-5:]))
    A = np.clip(copt - floor, 0.0, None)
    C_on = np.interp(tC, np.concatenate([[0.0], tU]), np.concatenate([[0.0], CU]))
    drive = np.clip(Cs - C_on, 0.0, None)
    integ = cumulative_trapezoid(A * drive, tC, initial=0.0)
    I_at_U = np.interp(tU, tC, integ)
    M_U = CU * V_ML
    ok = I_at_U > 0
    kp = M_U[ok] / I_at_U[ok]
    kp_late = kp[1:] if kp.size > 2 else kp
    cv = float(np.std(kp_late) / np.mean(kp_late)) if kp_late.size and np.mean(kp_late) else np.nan
    return float(np.median(kp_late)), cv, floor


def _decay(t, c, tau):
    return c + (1.0 - c) * np.exp(-t / np.clip(tau, 1e-3, None))


def channel_taus(tC, I, eff_size, admitted):
    """Per-channel τ with fit QC. Returns (per-channel table, slope on QC-passing channels).

    A τ is only a measurement if the fit converged, landed off both bounds, the channel actually
    decayed, and the fit describes the data. Channels failing any of these are retained in the
    table with the reason recorded and excluded from the slope.
    """
    In = I / np.where(I[0] > 0, I[0], 1.0)
    rows = []
    for c in range(I.shape[1]):
        y = In[:, c]
        rec = {"channel_index": c, "eff_size_um": float(eff_size[c]),
               "admitted": bool(admitted[c]), "tau_min": np.nan, "plateau": np.nan,
               "converged": False, "at_bound": False, "decay_depth": float(y[0] - y.min()),
               "r2": np.nan, "qc_reason": ""}
        if y[0] <= 0:
            rec["qc_reason"] = "nonpositive_start"
            rows.append(rec); continue
        try:
            p, _ = curve_fit(_decay, tC, y, p0=[y[-1], 1.0],
                             bounds=([0, TAU_BOUNDS[0]], [1, TAU_BOUNDS[1]]), maxfev=4000)
        except Exception:
            rec["qc_reason"] = "fit_failed"
            rows.append(rec); continue
        tau = float(p[1])
        resid = y - _decay(tC, *p)
        sstot = float(np.sum((y - y.mean()) ** 2))
        rec.update(converged=True, tau_min=tau, plateau=float(p[0]),
                   at_bound=bool(min(abs(tau - TAU_BOUNDS[0]), abs(tau - TAU_BOUNDS[1]))
                                 <= TAU_BOUND_TOL),
                   r2=float(1.0 - np.sum(resid ** 2) / sstot) if sstot > 0 else np.nan)
        reasons = []
        if rec["at_bound"]:
            reasons.append("tau_at_bound")
        if rec["decay_depth"] < MIN_DECAY_DEPTH:
            reasons.append("shallow_decay")
        if not (rec["r2"] >= MIN_FIT_R2):
            reasons.append("poor_fit")
        if not rec["admitted"]:
            reasons.append("channel_not_admitted")
        rec["qc_reason"] = "|".join(reasons)
        rows.append(rec)

    table = pd.DataFrame(rows)
    table["qc_pass"] = table["qc_reason"].eq("") & table["converged"]
    good = table[table["qc_pass"]]
    slope = (float(np.polyfit(np.log10(good["eff_size_um"]), good["tau_min"], 1)[0])
             if len(good) > 4 else np.nan)
    slope_all = table[table["converged"]]
    slope_unfiltered = (float(np.polyfit(np.log10(slope_all["eff_size_um"]),
                                         slope_all["tau_min"], 1)[0])
                        if len(slope_all) > 4 else np.nan)
    return table, slope, slope_unfiltered


def fit_runs(runs: pd.DataFrame, cs_map: dict, *, clean: bool = True):
    """Fit every run; returns (per-run table, per-channel τ table, trajectories)."""
    kernel = mie.load_kernel()
    A = np.asarray(kernel.A); xm = np.asarray(kernel.xm)
    eff = (A * xm).sum(1) / A.sum(1)

    rows, taus, traj = [], [], {}
    for run in runs.itertuples():
        cs = cs_map[run.condition]
        opt = optical_run(run.rtf, clean=clean)
        tC, copt, I = opt.t_min, opt.copt, opt.I
        tU, CU = uv_curve(run.uv_file, cs)
        k, kcv, floor = fit_k(tC, copt, tU, CU, cs)
        tau_table, slope, slope_unfiltered = channel_taus(tC, I, eff, opt.admitted)
        tau_table.insert(0, "run_id", run.run_id)
        taus.append(tau_table)
        rel = copt / copt[0]
        thalf = float(np.interp(0.5, rel[::-1], tC[::-1])) if rel.min() < 0.5 else np.nan
        rows.append({
            "run_id": run.run_id, "condition": run.condition, "xcmc": run.xcmc,
            "prep": run.prep, "prep_index": run.prep_index, "folder": run.folder,
            "rep": run.rep, "Cs_ugml": cs, "centrifuge_filter_disagreement": run.condition in CENTRIFUGE_FILTER_DISAGREEMENT,
            "k": k, "k_cv": kcv, "copt0": float(copt[0]), "copt_floor": floor,
            "copt_frac_loss": float((copt[0] - floor) / copt[0]),
            "copt_thalf_min": thalf, "C_end_ugml": float(CU[-1]),
            "tau_size_slope_qc": slope, "tau_size_slope_unfiltered": slope_unfiltered,
            "n_channels_qc_pass": int(tau_table["qc_pass"].sum()),
            "n_channels_at_bound": int(tau_table["at_bound"].sum()),
            "run_has_bound_hit": bool(tau_table["at_bound"].any()),
            "n_frames_retained": opt.n_frames,
            "n_frames_dropped": opt.provenance["n_frames_input"] - opt.n_frames,
            "n_interior_interpolated": opt.provenance["n_interior_interpolated"],
            "n_channels_admitted": opt.provenance["n_channels_admitted"],
        })
        traj[run.run_id] = dict(tC=tC, copt=copt, eff=eff, tau_table=tau_table)
    return pd.DataFrame(rows), pd.concat(taus, ignore_index=True), traj


# ── prep-balanced aggregation ────────────────────────────────────────────────────────────────

def prep_means(per_run: pd.DataFrame) -> pd.DataFrame:
    """Collapse technical repeats — one row per independent prep."""
    agg = (per_run.groupby(["condition", "xcmc", "prep", "prep_index"], as_index=False)
           .agg(n_technical_reps=("k", "size"), k_mean=("k", "mean"), k_sd=("k", "std"),
                log_k_mean=("k", lambda s: float(np.mean(np.log(s)))),
                tau_slope_qc=("tau_size_slope_qc", "mean"),
                tau_slope_unfiltered=("tau_size_slope_unfiltered", "mean"),
                copt_floor=("copt_floor", "mean"),
                copt_frac_loss=("copt_frac_loss", "mean"),
                copt_thalf=("copt_thalf_min", "mean"),
                n_channels_qc_pass=("n_channels_qc_pass", "mean"),
                any_bound_hit=("run_has_bound_hit", "any")))
    return agg.sort_values(["xcmc", "prep"]).reset_index(drop=True)


def condition_means(preps: pd.DataFrame) -> pd.DataFrame:
    """Condition means over independent preps — unweighted, spread taken across preps only."""
    metrics = ["k_mean", "tau_slope_qc", "tau_slope_unfiltered", "copt_floor",
               "copt_frac_loss", "copt_thalf"]
    out = (preps.groupby(["condition", "xcmc"], as_index=False)
           .agg(n_preps=("prep", "nunique"),
                n_technical_reps=("n_technical_reps", "sum"),
                preps=("prep", lambda s: " | ".join(sorted(set(s)))),
                **{m: (m, "mean") for m in metrics},
                k_sd=("k_mean", "std")))
    out["k_sem"] = out["k_sd"] / np.sqrt(out["n_preps"])
    out["centrifuge_filter_disagreement"] = out["condition"].isin(
        CENTRIFUGE_FILTER_DISAGREEMENT)
    return out.sort_values("xcmc").reset_index(drop=True)


def paired_contrasts(preps: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Within-prep log-ratio of k for pairs spiked from one suspension.

    Cs is already divided out inside k, so the k ratio is read directly: ≈1 means the Tween
    effect was solubility only; a ratio away from 1 means an effect solubility does not contain.
    No further Cs normalization is applied.
    """
    by_prep = {(r.condition, r.prep): r.log_k_mean for r in preps.itertuples()}
    rows = []
    for pair in pairs.itertuples():
        shared = [p for p in pair.shared_preps.split(" | ") if p]
        diffs = [by_prep[(pair.condition_b, p)] - by_prep[(pair.condition_a, p)]
                 for p in shared
                 if (pair.condition_a, p) in by_prep and (pair.condition_b, p) in by_prep]
        if not diffs:
            continue
        mean_log = float(np.mean(diffs))
        rows.append({"condition_a": pair.condition_a, "condition_b": pair.condition_b,
                     "n_shared_preps": len(diffs), "shared_preps": " | ".join(shared),
                     "mean_log_k_ratio": mean_log,
                     "k_ratio_b_over_a": float(np.exp(mean_log)),
                     "k_ratio_sd_across_preps": float(np.std(np.exp(diffs), ddof=1))
                     if len(diffs) > 1 else np.nan,
                     "per_prep_k_ratio": " | ".join(f"{np.exp(d):.3f}" for d in diffs)})
    return pd.DataFrame(rows)


def _plot(preps, conditions, contrasts, taus, traj, per_run, out_png, subtitle):
    fig, (axC, axK, axF) = plt.subplots(1, 3, figsize=(18, 5.4))

    for run in per_run.itertuples():
        tr = traj[run.run_id]
        axC.plot(tr["tC"], tr["copt"] / tr["copt"][0], "-", color=COLORS[run.condition],
                 lw=1.0, alpha=0.40)
    for condition in CONDITIONS:
        axC.plot([], [], "-", color=COLORS[condition], lw=1.8, label=condition)
    axC.set_xlabel("time (min)"); axC.set_ylabel("Copt / Copt₀")
    axC.set_title("Optical dissolution trajectories", fontsize=10)
    axC.grid(alpha=0.3); axC.legend(fontsize=8, frameon=False)

    for prep in preps.itertuples():
        axK.plot(prep.xcmc, prep.k_mean, "o", ms=6, color=COLORS[prep.condition], alpha=0.55)
    axK.errorbar(conditions["xcmc"], conditions["k_mean"], yerr=conditions["k_sd"],
                 fmt="k--", lw=1.2, alpha=0.7, capsize=4, zorder=1,
                 label="condition mean ± SD across preps")
    for c in conditions.itertuples():
        axK.annotate(f"{c.condition}\nk={c.k_mean:.2g}\nCs={c.Cs_ugml:g}", (c.xcmc, c.k_mean),
                     textcoords="offset points", xytext=(7, -6), fontsize=7)
    axK.set_xscale("log"); axK.set_xlabel("in-cuvette Tween (× CMC)")
    axK.set_ylabel("solubility-normalized rate k  (mL %⁻¹ min⁻¹)")
    axK.grid(alpha=0.3, which="both"); axK.legend(fontsize=7, frameon=False)
    axK.set_title("k vs Tween — Cs already divided out\n"
                  "(flat → solubility-only; changing → interfacial)", fontsize=10)

    kept = taus[taus["qc_pass"]]
    for condition in CONDITIONS:
        ids = per_run.loc[per_run["condition"].eq(condition), "run_id"]
        sub = kept[kept["run_id"].isin(ids)]
        if sub.empty:
            continue
        med = sub.groupby("channel_index").agg(eff=("eff_size_um", "first"),
                                               tau=("tau_min", "median"))
        axF.plot(med["eff"], med["tau"], "o", ms=4, color=COLORS[condition],
                 label=f"{condition}  (n={len(sub)} ch)")
    axF.set_xscale("log"); axF.set_xlabel("channel characteristic size (µm)")
    axF.set_ylabel("per-channel τ (min), QC-passing only")
    axF.grid(alpha=0.3, which="both"); axF.legend(fontsize=8, frameon=False)
    n_bound = int(taus["at_bound"].sum())
    axF.set_title(f"τ vs size — EXPLORATORY, not a partition claim\n"
                  f"({n_bound} boundary-capped channel fits excluded)", fontsize=10)

    paired = "   ".join(
        f"{r.condition_b.split()[0]}/{r.condition_a.split()[0]}: k×{r.k_ratio_b_over_a:.2f}"
        for r in contrasts.itertuples()) if not contrasts.empty else ""
    fig.suptitle(f"Arm B in-medium Tween — solubility-normalized rate  ({subtitle})\n"
                 f"n=4 independent preps per condition · within-prep k ratios:  {paired}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def run_analysis(study_root=None, *, ladder=arm_b_cs.DEFAULT_LADDER, clean=True):
    """The whole Arm B optical analysis for one (Cs ladder, cleaning) choice."""
    root = Path(study_root) if study_root is not None else default_study_root()
    runs = discover_runs(root)
    cs = arm_b_cs.cs_map(ladder)
    per_run, taus, traj = fit_runs(runs, cs, clean=clean)
    preps = prep_means(per_run)
    conditions = condition_means(preps)
    conditions["Cs_ugml"] = conditions["condition"].map(cs)
    contrasts = paired_contrasts(preps, condition_pairs_sharing_preps(runs))
    return dict(runs=runs, per_run=per_run, taus=taus, traj=traj, preps=preps,
                conditions=conditions, contrasts=contrasts,
                replication=replication_summary(runs), cs_ladder=ladder, cleaned=clean)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Arm B solubility-normalized rate analysis.")
    p.add_argument("--study-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--cs-ladder", default=arm_b_cs.DEFAULT_LADDER, choices=arm_b_cs.LADDER_NAMES)
    p.add_argument("--optical", default="pipeline+copt", choices=arm_b_optical.MODES,
                   help="Optical QC mode; 'raw' reproduces the pre-audit path.")
    args = p.parse_args(argv)
    root = args.study_root or default_study_root()
    clean = args.optical
    tag = f"{args.cs_ladder}_{clean.replace('+', '_')}"
    out = args.output_dir or (root / "analysis" / "partition" / tag)
    out.mkdir(parents=True, exist_ok=True)

    res = run_analysis(root, ladder=args.cs_ladder, clean=clean)
    res["per_run"].to_csv(out / "arm_b_partition_runs.csv", index=False)
    res["preps"].to_csv(out / "arm_b_partition_preps.csv", index=False)
    res["conditions"].to_csv(out / "arm_b_partition_conditions.csv", index=False)
    res["contrasts"].to_csv(out / "arm_b_partition_paired_contrasts.csv", index=False)
    res["taus"].to_csv(out / "arm_b_partition_channel_tau_qc.csv", index=False)
    res["replication"].to_csv(out / "arm_b_partition_replication.csv", index=False)
    _plot(res["preps"], res["conditions"], res["contrasts"], res["taus"], res["traj"],
          res["per_run"], out / "arm_b_partition.png",
          f"Cs = {args.cs_ladder}; optical = {clean}")
    write_provenance(out / "provenance.json",
                     provenance_record("arm_b_partition", cs_ladder=args.cs_ladder,
                                       optical_cleaned=clean, study_root=root))

    print(f"Arm B — Cs ladder '{args.cs_ladder}', optical mode '{clean}'\n")
    print(res["conditions"][["condition", "xcmc", "Cs_ugml", "n_preps", "n_technical_reps",
                             "k_mean", "k_sd", "k_sem", "copt_floor", "copt_frac_loss",
                             "centrifuge_filter_disagreement"]]
          .to_string(index=False), "\n")
    print("within-prep paired k ratios (Cs already divided out — no second normalization):")
    print(res["contrasts"][["condition_a", "condition_b", "n_shared_preps", "k_ratio_b_over_a",
                            "per_prep_k_ratio"]].to_string(index=False))
    t = res["taus"]
    print(f"\nchannel τ QC: {int(t['qc_pass'].sum())}/{len(t)} fits usable; "
          f"{int(t['at_bound'].sum())} at the τ bound, "
          f"{int((t['qc_reason'].str.contains('shallow_decay')).sum())} too shallow, "
          f"{int((t['qc_reason'].str.contains('poor_fit')).sum())} poor fit. "
          f"τ-vs-size is EXPLORATORY.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
