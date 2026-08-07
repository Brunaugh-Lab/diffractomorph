"""Arm B UV-Vis — dissolved mass fraction over time, with observation-level QC.

Each of the 27 BioTek kinetic plates → vessel-basis C(t) through the packaged assay, then to a
cumulative dissolved fraction of the delivered dose, correcting for the unreplaced aliquot
withdrawn at each timepoint (:func:`assay.cumulative_dissolved`). Dose is per replicate from
:mod:`arm_b_injected_mass`.

**Measured values are never modified or deleted.** Suspect observations are *flagged*, and the
analysis is reported both inclusive and under stated exclusions:

``qc_recovery_above_1``   cumulative recovery exceeds 100 % of the delivered dose
``qc_large_decrease``     recovery falls > ``DECREASE_PP`` points from the previous sample
                          (dissolved mass should not decrease; the mass balance already accounts
                          for aliquot removal, so a real decrease implies precipitation,
                          deposition, or assay noise)
``qc_isolated_extreme``   above ``EXTREME_RECOVERY`` and not corroborated by its own neighbours

Two calibration uncertainties are distinct and must not be conflated:

1. **The additive filter offset.** Large (~30 % of the pre-offset signal) and it slides every
   condition's *level* together, so absolute recovery is not interpretable on its own — but
   because it is additive and the doses are near-identical, condition *differences* are stable
   under it (10x−0.5x moves only ~16.2→16.6 pp across 0→1.48 µg/mL).
2. **A Tween/UV matrix effect.** No CFZ standard curve exists at these in-medium Tween levels,
   so absolute concentrations carry an uncalibrated matrix term, and stability under (1) says
   nothing about it. It is a **residual assay limitation** on absolute levels, recorded rather
   than resolved; no Tween-matched curve is planned and interpretation is not gated on one. Arm
   B's conclusions rest on condition comparisons and on the calibration-free optical observables
   (the Copt floor and fractional loss), not on absolute UV levels.

Aggregation is prep-balanced: technical repeats collapse within a prep, then preps are weighted
equally (n = 4 per condition). Run with the pipeline venv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diffractomorph_pipeline.assay import cumulative_dissolved, uv_timecourse

import arm_b_cs
from arm_b_common import CONDITIONS, default_study_root, discover_runs
from arm_b_injected_mass import build_dose_table
from arm_b_provenance import provenance_record, write_provenance

PH, CELL_ML = 4.5, 40.0
METRICS = ["conc_ugml", "pct_injected", "cumulative_dissolved_ug", "recovery_mass_fraction"]
COLORS = {"0.5x CMC": "#718096", "1.0x CMC": "#2b6cb0", "10x CMC": "#c05621"}
DECREASE_PP = 5.0             # recovery drop between successive samples that gets flagged
EXTREME_RECOVERY = 1.30       # isolated points above this are the stated sensitivity exclusion


def _flag(long: pd.DataFrame) -> pd.DataFrame:
    """Attach observation-level QC columns. Nothing is dropped here."""
    long = long.sort_values(["run_id", "time_min"]).copy()
    rec = long.groupby("run_id")["recovery_mass_fraction"]
    long["recovery_change_pp"] = rec.diff() * 100.0
    long["qc_recovery_above_1"] = long["recovery_mass_fraction"] > 1.0
    long["qc_large_decrease"] = long["recovery_change_pp"] < -DECREASE_PP
    # "isolated" = extreme, while the neighbouring samples in the same run are not
    neigh = pd.concat([rec.shift(1), rec.shift(-1)], axis=1).max(axis=1)
    long["qc_isolated_extreme"] = (long["recovery_mass_fraction"] > EXTREME_RECOVERY) & (
        neigh.fillna(0.0) <= EXTREME_RECOVERY)
    long["qc_any"] = long[["qc_recovery_above_1", "qc_large_decrease",
                           "qc_isolated_extreme"]].any(axis=1)
    return long


def process_uv(study_root: Path | str | None = None,
               ladder: str = arm_b_cs.DEFAULT_LADDER) -> pd.DataFrame:
    """Every run → a long C(t) + mass-balance + QC table carrying prep identity and dose."""
    root = Path(study_root) if study_root is not None else default_study_root()
    cs = arm_b_cs.cs_map(ladder)
    runs = discover_runs(root)[["run_id", "uv_file"]]
    dose = build_dose_table(root).merge(runs, on="run_id", how="left", validate="1:1")
    if dose["uv_file"].isna().any():
        raise ValueError("dose table and run discovery disagree on run_id")

    frames = []
    for run in dose.itertuples():
        frame = uv_timecourse(run.uv_file, ph=PH, cs_ugml=cs[run.condition],
                              injected_mg=run.injected_mass_mg, volume_mL=CELL_ML)
        balance = cumulative_dissolved(frame["conc_ugml"], run.injected_mass_mg, v0_mL=CELL_ML)
        frame = pd.concat([frame, balance.drop(columns="sample_index")], axis=1)
        for i, (name, value) in enumerate([
                ("run_id", run.run_id), ("condition", run.condition), ("xcmc", run.xcmc),
                ("prep", run.prep), ("prep_index", run.prep_index), ("folder", run.folder),
                ("rep", run.rep), ("injected_mass_mg", run.injected_mass_mg),
                ("uv_file", str(Path(run.uv_file).relative_to(root)))]):
            frame.insert(i, name, value)
        frames.append(frame)
    return _flag(pd.concat(frames, ignore_index=True))


def prep_means(long: pd.DataFrame) -> pd.DataFrame:
    """Collapse technical repeats to one trajectory per independent prep."""
    return (long.groupby(["condition", "xcmc", "prep", "prep_index", "time_min"], as_index=False)
            .agg(n_technical_reps=("rep", "nunique"), n_obs=("rep", "size"),
                 n_flagged=("qc_any", "sum"), **{m: (m, "mean") for m in METRICS}))


def condition_means(preps: pd.DataFrame) -> pd.DataFrame:
    """Condition trajectories over independent preps — unweighted, SD/SEM across preps."""
    out = (preps.groupby(["condition", "xcmc", "time_min"], as_index=False)
           .agg(n_preps=("prep", "nunique"),
                n_technical_reps=("n_technical_reps", "sum"),
                **{f"{m}_mean": (m, "mean") for m in METRICS},
                **{f"{m}_sd": (m, "std") for m in METRICS}))
    for m in METRICS:
        out[f"{m}_sem"] = out[f"{m}_sd"] / np.sqrt(out["n_preps"])
    return out


def separation(conditions: pd.DataFrame, lo="0.5x CMC", hi="10x CMC",
               metric="pct_injected") -> pd.DataFrame:
    """10x − 0.5x at each time — the offset-invariant condition comparison."""
    w = conditions.pivot(index="time_min", columns="condition", values=f"{metric}_mean")
    return pd.DataFrame({"time_min": w.index, f"{metric}_separation_pp": w[hi] - w[lo]}) \
        .reset_index(drop=True)


def variants(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The stated inclusive / exclusion analyses. Keys name the exclusion applied."""
    return {
        "inclusive": long,
        "excl_isolated_extreme": long[~long["qc_isolated_extreme"]],
        "excl_all_flagged": long[~long["qc_any"]],
    }


def _plot(preps, conditions, long, path):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.7))
    panels = [("conc_ugml", "Dissolved CFZ (µg/mL)"),
              ("recovery_mass_fraction", "Cumulative dissolved fraction of dose")]
    for ax, (metric, ylabel) in zip(axes[:2], panels):
        for condition in CONDITIONS:
            color = COLORS[condition]
            for _, g in preps[preps["condition"].eq(condition)].groupby("prep"):
                ax.plot(g["time_min"], g[metric], color=color, alpha=0.30, lw=1.0)
            s = conditions[conditions["condition"].eq(condition)]
            x = s["time_min"].to_numpy(float)
            mean = s[f"{metric}_mean"].to_numpy(float)
            sd = s[f"{metric}_sd"].to_numpy(float)
            ax.plot(x, mean, "o-", color=color, lw=2.2, ms=4, label=condition)
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.14, linewidth=0)
        ax.set_xlabel("Time (min)"); ax.set_ylabel(ylabel); ax.grid(alpha=0.2)
    axes[1].axhline(1.0, color="#555555", ls=":", lw=0.9)
    axes[0].legend(frameon=False, fontsize=8)

    ax = axes[2]
    flagged = long[long["qc_any"]]
    ax.scatter(long["time_min"], long["recovery_mass_fraction"], s=8, color="0.75",
               label=f"all obs (n={len(long)})")
    ax.scatter(flagged["time_min"], flagged["recovery_mass_fraction"], s=22,
               facecolors="none", edgecolors="crimson",
               label=f"QC-flagged (n={len(flagged)})")
    ax.axhline(1.0, color="#555555", ls=":", lw=0.9)
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Cumulative recovery (fraction of dose)")
    ax.set_title("Observation QC — flagged, never deleted", fontsize=10)
    ax.grid(alpha=0.2); ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Arm B UV dissolved mass — n=4 preps per condition, band = SD across preps; "
                 "absolute level is calibration-dependent, condition differences are not",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Arm B UV dissolved-mass trajectories + QC.")
    p.add_argument("--study-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--cs-ladder", default=arm_b_cs.DEFAULT_LADDER, choices=arm_b_cs.LADDER_NAMES)
    args = p.parse_args(argv)
    root = args.study_root or default_study_root()
    out_dir = args.output_dir or root / "analysis" / "uv"
    out_dir.mkdir(parents=True, exist_ok=True)

    long = process_uv(root, args.cs_ladder)
    long.to_csv(out_dir / "arm_b_uv_timecourse.csv", index=False)

    summaries = []
    for name, subset in variants(long).items():
        preps = prep_means(subset)
        conditions = condition_means(preps)
        preps.to_csv(out_dir / f"arm_b_uv_prep_means_{name}.csv", index=False)
        conditions.to_csv(out_dir / f"arm_b_uv_condition_summary_{name}.csv", index=False)
        sep = separation(conditions)
        for t in (2.0, 4.0, conditions["time_min"].max()):
            row = sep[np.isclose(sep["time_min"], t)]
            if not row.empty:
                summaries.append({"variant": name, "n_obs": len(subset), "time_min": t,
                                  "separation_10x_minus_0.5x_pp":
                                      float(row["pct_injected_separation_pp"].iloc[0])})
        if name == "inclusive":
            _plot(preps, conditions, long, out_dir / "arm_b_uv_dissolved_mass.png")
    sens = pd.DataFrame(summaries)
    sens.to_csv(out_dir / "arm_b_uv_qc_sensitivity.csv", index=False)
    write_provenance(out_dir / "provenance.json",
                     provenance_record("arm_b_uv_timecourse", cs_ladder=args.cs_ladder,
                                       study_root=root,
                                       uv_qc={"decrease_pp": DECREASE_PP,
                                              "extreme_recovery": EXTREME_RECOVERY}))

    flags = long[["qc_recovery_above_1", "qc_large_decrease", "qc_isolated_extreme"]].sum()
    print(f"{len(long)} observations; flagged: "
          f"{int(flags['qc_recovery_above_1'])} recovery>100%, "
          f"{int(flags['qc_large_decrease'])} decrease>{DECREASE_PP:g}pp, "
          f"{int(flags['qc_isolated_extreme'])} isolated extreme (>{EXTREME_RECOVERY:.0%})\n")
    worst = long.nlargest(3, "recovery_mass_fraction")
    print("largest recoveries:")
    print(worst[["condition", "prep", "rep", "time_min", "recovery_mass_fraction",
                 "qc_isolated_extreme"]].to_string(index=False))
    print("\ncondition separation (10x − 0.5x, pp of dose) by QC variant:")
    print(sens.pivot(index="time_min", columns="variant",
                     values="separation_10x_minus_0.5x_pp").round(2).to_string())
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
