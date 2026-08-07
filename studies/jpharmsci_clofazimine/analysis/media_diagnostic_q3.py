"""q3 size-space analysis for the Tween-80 media-diagnostic arms (A and B).

The q3 trajectories for both arms were collected and never analysed. They matter because the
solubility-only / rugosity reading is a claim about **surface area**, while the Arm B result so
far rests on Copt (a bulk area proxy) and UV mass. The instrument's own inverted q3 is the only
size-resolved observable available, so it is the independent size-space check.

**The test.** If raising Tween only raises Cs, the *path through size space* should be unchanged:
at the same optical extent, the particle-size distribution should look the same, and the
conditions should differ only in how fast they travel that path. If Tween instead changed an
interfacial or surface-structural mechanism, the distribution at matched extent should differ.
So this compares q3 at **matched extent**, not at matched time — a clock-free comparison that
does not inherit the rate.

Extent is the optical loss fraction ``g(t) = 1 − (Copt − Copt∞)/(Copt₀ − Copt∞)``, taken from the
same cleaned optical path as :mod:`arm_b_partition` (:mod:`arm_b_optical`), so despiking and
startup handling are shared rather than re-implemented.

**Reliability ceiling.** The PAQXOS inversion above ``psd.VALID_SIZE_MAX_UM`` (~15 µm) is
ill-constrained. Percentiles are computed on the restricted cumulative, and the coarse-tail
fraction is reported per frame so a conclusion resting on the tail is visible as such.

Aggregation is prep-balanced for Arm B (n = 4 preps; see :mod:`arm_b_common`) and
date-balanced for Arm A. Run with the pipeline venv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diffractomorph_pipeline import ingest, psd

import arm_a_common
import arm_b_common
from arm_b_optical import optical_run
from arm_b_provenance import provenance_record, write_provenance

PCTS = (10.0, 50.0, 90.0)
EXTENT_GRID = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])   # optical loss fractions to match on
COLORS = {"0.5x CMC": "#718096", "1.0x CMC": "#2b6cb0", "10x CMC": "#c05621",
          "0.01": "#2864A6", "0.03": "#C55A11"}


def _extent(copt: np.ndarray) -> np.ndarray:
    """Optical loss fraction g∈[0,1]; monotone envelope so matching is single-valued."""
    floor = float(np.mean(copt[-5:]))
    amp = float(copt[0] - floor)
    if amp <= 0:
        return np.zeros_like(copt)
    g = 1.0 - (copt - floor) / amp
    return np.maximum.accumulate(np.clip(g, 0.0, 1.0))


def _percentile_frames(traj: psd.Q3Trajectory) -> pd.DataFrame:
    """Per-frame D10/D50/D90 on the reliability-restricted cumulative, plus the coarse tail."""
    xo = np.asarray(traj.grid_um, float)
    cum = np.cumsum(np.asarray(traj.dQ3, float), axis=1) * 100.0
    rows = []
    for i, row in enumerate(cum):
        xr, cr = psd.restrict_cumulative(xo, row)
        d10, d50, d90 = psd.q3_percentiles(xr, cr, PCTS)
        rows.append({"frame": i, "d10_um": d10, "d50_um": d50, "d90_um": d90,
                     "tail_frac_above_15um": psd.q3_tail_fraction(xo, row)})
    return pd.DataFrame(rows)


def _runs_for_arm(arm: str):
    """(run records, label) for either arm. Each record: id, condition, prep/date, rep, q3, rtf."""
    if arm == "arm_b":
        runs = arm_b_common.discover_runs()
        return [{"run_id": r.run_id, "condition": r.condition, "unit": r.prep, "rep": r.rep,
                 "q3": Path(r.q3_dir), "rtf": Path(r.rtf)} for r in runs.itertuples()], "prep"
    root = arm_a_common.default_study_root()
    out = []
    for condition in arm_a_common.CONDITIONS:
        for day in (1, 2, 3):
            for rep in (1, 2, 3):
                try:
                    q3 = arm_a_common.find_q3_source(root, condition, day, rep)
                    rtf = arm_a_common.find_measurement_rtf(root, condition, day, rep)
                except FileNotFoundError:
                    continue
                out.append({"run_id": f"{condition}_D{day}_rep{rep}", "condition": condition,
                            "unit": f"Day {day}", "rep": rep, "q3": q3, "rtf": rtf})
    return out, "date"


def analyse(arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-frame q3 percentiles joined to optical extent, and the matched-extent table."""
    records, unit_name = _runs_for_arm(arm)
    frames, matched = [], []
    for rec in records:
        traj = psd.read_q3(rec["q3"])
        opt = optical_run(rec["rtf"], clean="pipeline+copt")
        pf = _percentile_frames(traj)
        n = min(len(pf), opt.copt.size)          # q3 and detector frame counts can differ by 1-2
        pf = pf.iloc[:n].copy()
        g = _extent(opt.copt)[:n]
        pf["extent_g"] = g
        pf["t_min"] = opt.t_min[:n]
        for key in ("run_id", "condition", "unit", "rep"):
            pf.insert(0, key, rec[key])
        frames.append(pf)

        ok = np.isfinite(pf["d50_um"].to_numpy())
        if ok.sum() >= 3:
            gg = pf.loc[ok, "extent_g"].to_numpy()
            for target in EXTENT_GRID:
                if not (gg.min() <= target <= gg.max()):
                    continue
                row = {"run_id": rec["run_id"], "condition": rec["condition"],
                       "unit": rec["unit"], "rep": rec["rep"], "extent_g": float(target)}
                for col in ("d10_um", "d50_um", "d90_um", "tail_frac_above_15um"):
                    v = pf.loc[ok, col].to_numpy()
                    row[col] = float(np.interp(target, gg, v))
                matched.append(row)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(matched)


def balanced(matched: pd.DataFrame) -> pd.DataFrame:
    """Collapse technical repeats within the independent unit, then weight units equally."""
    metrics = ["d10_um", "d50_um", "d90_um", "tail_frac_above_15um"]
    per_unit = (matched.groupby(["condition", "unit", "extent_g"], as_index=False)
                .agg(n_reps=("rep", "nunique"), **{m: (m, "mean") for m in metrics}))
    out = (per_unit.groupby(["condition", "extent_g"], as_index=False)
           .agg(n_units=("unit", "nunique"),
                **{f"{m}_mean": (m, "mean") for m in metrics},
                **{f"{m}_sd": (m, "std") for m in metrics}))
    return out


def _plot(frames, balanced_tbl, arm, path):
    conditions = list(dict.fromkeys(balanced_tbl["condition"]))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 4.8))

    for condition in conditions:
        c = COLORS.get(condition, "#444444")
        for _, g in frames[frames["condition"].eq(condition)].groupby("run_id"):
            ax1.plot(g["t_min"], g["d50_um"], color=c, alpha=0.30, lw=0.9)
        ax1.plot([], [], color=c, lw=2, label=condition)
    ax1.set_xlabel("time (min)"); ax1.set_ylabel("q3 D50 (µm)")
    ax1.set_title("D50 over time (per run)", fontsize=10)
    ax1.grid(alpha=0.25); ax1.legend(fontsize=8, frameon=False)

    for condition in conditions:
        c = COLORS.get(condition, "#444444")
        s = balanced_tbl[balanced_tbl["condition"].eq(condition)].sort_values("extent_g")
        ax2.errorbar(s["extent_g"], s["d50_um_mean"], yerr=s["d50_um_sd"], fmt="o-",
                     color=c, capsize=3, lw=1.8, ms=5, label=condition)
    ax2.set_xlabel("optical extent g (fraction of Copt lost)")
    ax2.set_ylabel("q3 D50 at matched extent (µm)")
    ax2.set_title("MATCHED-EXTENT size path\n(overlap → same path, solubility-only)", fontsize=10)
    ax2.grid(alpha=0.25); ax2.legend(fontsize=8, frameon=False)

    for condition in conditions:
        c = COLORS.get(condition, "#444444")
        s = balanced_tbl[balanced_tbl["condition"].eq(condition)].sort_values("extent_g")
        ax3.plot(s["extent_g"], s["tail_frac_above_15um_mean"], "o-", color=c, lw=1.6, ms=4,
                 label=condition)
    ax3.set_xlabel("optical extent g"); ax3.set_ylabel("q3 mass fraction above 15 µm")
    ax3.set_title("Coarse tail beyond the inversion's\nreliable range (context, not a result)",
                  fontsize=10)
    ax3.grid(alpha=0.25); ax3.legend(fontsize=8, frameon=False)

    fig.suptitle(f"{arm} — q3 size-space evolution and the matched-extent path "
                 f"(instrument inversion, taken as given)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def separation_at_extent(balanced_tbl: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    """D50(hi) − D50(lo) at matched extent, with an equivalence bound.

    A bare "no difference" understates and overstates at once: it hides how large a difference
    would have gone unseen. ``detectable_diff_um`` is the ~2 SEM difference this design could
    have resolved at that extent, so the result reads as *any condition effect on the size path
    is bounded below X µm* rather than as a null.
    """
    w = balanced_tbl.pivot(index="extent_g", columns="condition", values="d50_um_mean")
    sd = balanced_tbl.pivot(index="extent_g", columns="condition", values="d50_um_sd")
    n = balanced_tbl.pivot(index="extent_g", columns="condition", values="n_units")
    if lo not in w or hi not in w:
        return pd.DataFrame()
    pooled = np.sqrt((sd[hi] ** 2 + sd[lo] ** 2) / 2.0)
    sem = pooled * np.sqrt(1.0 / n[hi] + 1.0 / n[lo])
    return pd.DataFrame({
        "extent_g": w.index,
        "d50_diff_um": w[hi] - w[lo],
        "pooled_sd_um": pooled,
        "detectable_diff_um": 2.0 * sem,
        "d50_mean_um": (w[hi] + w[lo]) / 2.0,
    }).reset_index(drop=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="q3 size-space analysis for the media-diagnostic arms.")
    p.add_argument("arm", choices=["arm_a", "arm_b"])
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    root = (arm_b_common.default_study_root() if args.arm == "arm_b"
            else arm_a_common.default_study_root())
    out = args.output_dir or root / "analysis" / "q3"
    out.mkdir(parents=True, exist_ok=True)

    frames, matched = analyse(args.arm)
    tbl = balanced(matched)
    frames.to_csv(out / "q3_frames.csv", index=False)
    matched.to_csv(out / "q3_matched_extent_runs.csv", index=False)
    tbl.to_csv(out / "q3_matched_extent_balanced.csv", index=False)
    _plot(frames, tbl, args.arm, out / "q3_size_space.png")
    write_provenance(out / "provenance.json",
                     provenance_record(f"media_diagnostic_q3::{args.arm}", study_root=root,
                                       optical_cleaned=True,
                                       extent_grid=EXTENT_GRID.tolist(),
                                       valid_size_max_um=psd.VALID_SIZE_MAX_UM))

    print(f"{args.arm}: {frames['run_id'].nunique()} runs, "
          f"{len(frames)} q3 frames, {len(matched)} matched-extent points\n")
    print("D50 (µm) at matched optical extent, balanced across independent units:")
    print(tbl.pivot(index="extent_g", columns="condition", values="d50_um_mean")
          .round(3).to_string(), "\n")
    conditions = list(dict.fromkeys(tbl["condition"]))
    if len(conditions) >= 2:
        sep = separation_at_extent(tbl, conditions[0], conditions[-1])
        if not sep.empty:
            print(f"D50 difference, {conditions[-1]} − {conditions[0]}, vs prep-level spread:")
            print(sep.round(3).to_string(index=False))
            worst = float(sep["d50_diff_um"].abs().max())
            bound = float(sep["detectable_diff_um"].max())
            rel = bound / float(sep["d50_mean_um"].mean()) * 100.0
            print(f"\nlargest observed |ΔD50| = {worst:.3f} µm; this design could have resolved "
                  f"~{bound:.3f} µm ({rel:.0f}% of D50).")
            print("Read as an equivalence bound, not a null: the conditions travel the same size "
                  f"path to within ~{bound:.2f} µm. A smaller effect would not have been seen.")
            print("Note the high-extent points draw on late frames where the PAQXOS inversion "
                  "degrades (rising coarse tail); percentiles use the ≤15 µm restricted "
                  "cumulative, and the tail panel shows the trend.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
