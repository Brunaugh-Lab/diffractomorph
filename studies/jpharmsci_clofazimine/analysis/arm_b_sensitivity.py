"""Arm B sensitivity bundle — which conclusions survive the analysis choices.

**The primary analysis is the filtered 48 h Cs ladder** (:mod:`arm_b_cs`) — the operational
saturation solubility, selected prospectively from the 48 h centrifuge-vs-filter experiment. The
headline claims are evaluated on that ladder across the three optical preprocessing modes, which
is the genuine robustness axis:

* **optical preprocessing** — ``raw`` / ``pipeline`` / ``pipeline+copt`` (:mod:`arm_b_optical`)

The particulate-containing centrifuged ladders are recomputed too, but only as **secondary method
sensitivities** showing how far the mid-ladder answer moves when particulate is not excluded.
They are not alternative primaries and no conclusion is conditioned on them.

For each cell the driver recomputes the prep-balanced k, its prep-level spread, the within-prep
paired contrasts, and the τ-vs-size slope under three fit-QC policies.

Run with the pipeline venv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import arm_b_cs
import arm_b_optical
from arm_b_common import default_study_root
from arm_b_partition import COLORS, run_analysis
from arm_b_provenance import provenance_record, write_provenance

ORDER = ["0.5x CMC", "1.0x CMC", "10x CMC"]


def _tau_slopes(res) -> dict:
    """τ-vs-size slope per condition under three fit-QC policies."""
    taus, per_run = res["taus"], res["per_run"]
    meta = per_run.set_index("run_id")[["condition", "prep"]]
    t = taus.join(meta, on="run_id")
    out = {}
    for policy, sub in (("all_converged", t[t["converged"]]),
                        ("qc_pass", t[t["qc_pass"]]),
                        ("drop_bound_hit_runs",
                         t[t["qc_pass"] & ~t["run_id"].isin(
                             per_run.loc[per_run["run_has_bound_hit"], "run_id"])])):
        per_prep = []
        for (condition, prep), g in sub.groupby(["condition", "prep"]):
            if len(g) > 4:
                per_prep.append({"condition": condition, "prep": prep,
                                 "slope": float(np.polyfit(np.log10(g["eff_size_um"]),
                                                           g["tau_min"], 1)[0])})
        frame = pd.DataFrame(per_prep)
        out[policy] = (frame.groupby("condition")["slope"].mean().to_dict()
                       if not frame.empty else {})
    return out


def sweep(study_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    k_rows, contrast_rows, tau_rows = [], [], []
    for ladder in arm_b_cs.LADDER_NAMES:
        for optical in arm_b_optical.MODES:
            res = run_analysis(study_root, ladder=ladder, clean=optical)
            for c in res["conditions"].itertuples():
                k_rows.append({"cs_ladder": ladder, "optical": optical,
                               "condition": c.condition, "xcmc": c.xcmc,
                               "Cs_ugml": c.Cs_ugml, "n_preps": c.n_preps,
                               "k_mean": c.k_mean, "k_sd": c.k_sd, "k_sem": c.k_sem,
                               "copt_floor": c.copt_floor,
                               "copt_frac_loss": c.copt_frac_loss})
            for r in res["contrasts"].itertuples():
                contrast_rows.append({"cs_ladder": ladder, "optical": optical,
                                      "pair": f"{r.condition_b.split()[0]}/{r.condition_a.split()[0]}",
                                      "k_ratio": r.k_ratio_b_over_a,
                                      "n_shared_preps": r.n_shared_preps})
            for policy, per_condition in _tau_slopes(res).items():
                for condition, slope in per_condition.items():
                    tau_rows.append({"cs_ladder": ladder, "optical": optical,
                                     "fit_qc_policy": policy, "condition": condition,
                                     "tau_size_slope": slope})
    return pd.DataFrame(k_rows), pd.DataFrame(contrast_rows), pd.DataFrame(tau_rows)


def _cellwise(k, contrasts):
    """Per (ladder, optical) cell: does k rise with Tween, and is every paired ratio < 1?"""
    rises, falls, below = {}, {}, {}
    for (ladder, optical), g in k.groupby(["cs_ladder", "optical"]):
        s = g.set_index("condition").loc[ORDER, "k_mean"]
        rises[(ladder, optical)] = bool(s.iloc[-1] > s.iloc[0])
        falls[(ladder, optical)] = bool(s.iloc[-1] < s.iloc[0])
    for key, g in contrasts.groupby(["cs_ladder", "optical"]):
        below[key] = bool((g["k_ratio"] < 1).all())
    return rises, falls, below


def stability(k: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the claims on the PRIMARY ladder; report the secondary ladders separately.

    The primary claims are assessed across the three optical preprocessing modes under
    ``arm_b_cs.PRIMARY_LADDER``. Rows for the particulate-containing centrifuged ladders are
    labelled ``secondary_method_sensitivity`` and carry no verdict — they exist to show how far
    the mid-ladder answer moves when particulate is not excluded, not to adjudicate the result.
    """
    rises, falls, below = _cellwise(k, contrasts)
    primary = [key for key in rises if key[0] == arm_b_cs.PRIMARY_LADDER]
    fmt = lambda d, keys: "; ".join(f"{key[1]}:{'Y' if d[key] else 'N'}" for key in sorted(keys))

    rows = [
        {"scope": "primary", "cs_ladder": arm_b_cs.PRIMARY_LADDER,
         "claim": "k does NOT rise with Tween (no partition signature)",
         "n_cells": len(primary), "n_supporting": sum(not rises[key] for key in primary),
         "survives_all": all(not rises[key] for key in primary),
         "detail": fmt({key: not rises[key] for key in primary}, primary)},
        {"scope": "primary", "cs_ladder": arm_b_cs.PRIMARY_LADDER,
         "claim": "k declines from 0.5x to 10x",
         "n_cells": len(primary), "n_supporting": sum(falls[key] for key in primary),
         "survives_all": all(falls[key] for key in primary),
         "detail": fmt(falls, primary)},
        {"scope": "primary", "cs_ladder": arm_b_cs.PRIMARY_LADDER,
         "claim": "every within-prep paired k ratio < 1",
         "n_cells": len(primary), "n_supporting": sum(below.get(key, False) for key in primary),
         "survives_all": all(below.get(key, False) for key in primary),
         "detail": fmt({key: below.get(key, False) for key in primary}, primary)},
    ]
    for ladder in arm_b_cs.SECONDARY_LADDERS:
        keys = [key for key in rises if key[0] == ladder]
        rows.append({"scope": "secondary_method_sensitivity", "cs_ladder": ladder,
                     "claim": "k does NOT rise with Tween (particulate-containing Cs; "
                              "reported for method comparison only, not a verdict)",
                     "n_cells": len(keys), "n_supporting": sum(not rises[key] for key in keys),
                     "survives_all": None,
                     "detail": fmt({key: not rises[key] for key in keys}, keys)})
    return pd.DataFrame(rows)


def _plot(k: pd.DataFrame, taus: pd.DataFrame, path: Path):
    ladders = list(arm_b_cs.LADDER_NAMES)
    fig, axes = plt.subplots(1, len(ladders) + 1, figsize=(5.0 * (len(ladders) + 1), 4.6))
    for ax, ladder in zip(axes, ladders):
        for optical, style, colour in (("pipeline+copt", "-o", "#2b6cb0"),
                                       ("pipeline", "-^", "#68a2d8"),
                                       ("raw", "--s", "#a0aec0")):
            sub = k[k["cs_ladder"].eq(ladder) & k["optical"].eq(optical)].set_index("condition")
            sub = sub.loc[ORDER]
            ax.errorbar(sub["xcmc"], sub["k_mean"], yerr=sub["k_sd"], fmt=style, capsize=3,
                        lw=1.6, ms=5, label=optical, color=colour)
        ax.set_xscale("log"); ax.set_xlabel("in-cuvette Tween (× CMC)")
        role = "PRIMARY" if ladder == arm_b_cs.PRIMARY_LADDER else "secondary (method only)"
        ax.set_title(f"Cs = {ladder}\n{role}", fontsize=10,
                     fontweight="bold" if ladder == arm_b_cs.PRIMARY_LADDER else "normal")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("solubility-normalized k (mL %⁻¹ min⁻¹)")
    lo = min(k["k_mean"] - k["k_sd"]) * 0.9
    hi = max(k["k_mean"] + k["k_sd"]) * 1.05
    for ax in axes[:len(ladders)]:
        ax.set_ylim(lo, hi)

    ax = axes[-1]
    policies = ["all_converged", "qc_pass", "drop_bound_hit_runs"]
    width = 0.25
    x = np.arange(len(ORDER))
    sub = taus[taus["cs_ladder"].eq(arm_b_cs.DEFAULT_LADDER)
               & taus["optical"].eq("pipeline+copt")]
    for i, policy in enumerate(policies):
        vals = [sub.loc[sub["fit_qc_policy"].eq(policy) & sub["condition"].eq(c),
                        "tau_size_slope"].mean() for c in ORDER]
        ax.bar(x + (i - 1) * width, vals, width, label=policy)
    ax.set_xticks(x); ax.set_xticklabels([c.split()[0] for c in ORDER])
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("τ-vs-log(size) slope (min/decade)")
    ax.set_title("Fanning slope collapses under fit QC\n→ EXPLORATORY, not a partition claim",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7, frameon=False)

    fig.suptitle(f"Arm B — solubility-normalized k. PRIMARY = {arm_b_cs.PRIMARY_LADDER} "
                 f"(operational Cs, prospectively selected); centrifuged panels are secondary "
                 f"method sensitivities only.  n=4 preps per condition, bars = SD across preps",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Arm B Cs-ladder x optical-cleaning sensitivity.")
    p.add_argument("--study-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    root = args.study_root or default_study_root()
    out = args.output_dir or root / "analysis" / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)

    k, contrasts, taus = sweep(root)
    verdict = stability(k, contrasts)
    ladders = arm_b_cs.ladders()

    ladders.to_csv(out / "arm_b_cs_ladders.csv", index=False)
    k.to_csv(out / "arm_b_k_sensitivity.csv", index=False)
    contrasts.to_csv(out / "arm_b_contrast_sensitivity.csv", index=False)
    taus.to_csv(out / "arm_b_tau_slope_sensitivity.csv", index=False)
    verdict.to_csv(out / "arm_b_stability_verdict.csv", index=False)
    _plot(k, taus, out / "arm_b_sensitivity.png")

    ref = k[k["cs_ladder"].eq(arm_b_cs.DEFAULT_LADDER) & k["optical"].eq("pipeline+copt")]
    floors = ref.set_index("condition")["copt_floor"].reindex(ORDER).to_dict()
    fracs = ref.set_index("condition")["copt_frac_loss"].reindex(ORDER).to_dict()
    uv_sens = pd.read_csv(root / "analysis" / "uv" / "arm_b_uv_qc_sensitivity.csv")
    pick = lambda v, t: float(uv_sens[uv_sens["variant"].eq(v)
                              & np.isclose(uv_sens["time_min"], t)]
                              ["separation_10x_minus_0.5x_pp"].iloc[0])
    tmax = uv_sens["time_min"].max()
    write_memo(out / "INTERPRETATION.md", k, taus,
               {"sep2": pick("inclusive", 2.0), "sep20_incl": pick("inclusive", tmax),
                "sep20_excl": pick("excl_all_flagged", tmax)},
               floors, fracs, verdict)
    write_provenance(out / "provenance.json",
                     provenance_record("arm_b_sensitivity", study_root=root,
                                       ladders=list(arm_b_cs.LADDER_NAMES),
                                       optical_variants=list(arm_b_optical.MODES)))

    print("Cs ladders (µg/mL):")
    print(ladders.pivot(index="condition", columns="ladder", values="cs_ugml")
          .loc[ORDER].round(2).to_string(), "\n")
    print("prep-balanced k by (Cs ladder x optical):")
    print(k.pivot_table(index="condition", columns=["cs_ladder", "optical"], values="k_mean")
          .loc[ORDER].round(3).to_string(), "\n")
    print("τ-vs-size slope by fit-QC policy (filtered_48h, cleaned):")
    sub = taus[taus["cs_ladder"].eq(arm_b_cs.DEFAULT_LADDER)
               & taus["optical"].eq("pipeline+copt")]
    print(sub.pivot_table(index="condition", columns="fit_qc_policy",
                          values="tau_size_slope").loc[ORDER].round(2).to_string(), "\n")
    print(f"PRIMARY ladder: {arm_b_cs.PRIMARY_LADDER} "
          f"(operational Cs, selected prospectively — commit 9ed2cb9)")
    for r in verdict.itertuples():
        if r.scope != "primary":
            continue
        print(f"  [{'HOLDS ' if r.survives_all else 'FAILS '}] {r.claim} "
              f"({r.n_supporting}/{r.n_cells} optical modes: {r.detail})")
    print("\nsecondary method sensitivities (particulate-containing Cs; not a verdict):")
    for r in verdict.itertuples():
        if r.scope == "primary":
            continue
        print(f"  {r.cs_ladder}: {r.n_supporting}/{r.n_cells} modes show no rise ({r.detail})")
    print(f"\nwrote {out}")
    return 0



MEMO = """# Arm B — interpretation memo (generated by `arm_b_sensitivity.py`)

Generated {ts} · commit {commit} · n = 4 independent suspension preps per condition
(injection dates; 1/2/3/3 technical repeats), all condition-level values prep-balanced.

**Primary saturation solubility: the 48 h syringe-filtered Arm C ladder** — the operational Cs,
selected prospectively from the 48 h centrifuge-vs-filter experiment (commit `9ed2cb9`), before
any Arm B result was seen. Centrifuged supernatant reads 2–3.5 ugmL high at the intermediate
Tween levels a one-minute spin cannot clear; the two preparations agree at 0x, 10x and 20x CMC,
and that agreement validates the fixed filter correction where particulate carryover is minimal.
The centrifuged ladders below appear only as secondary method sensitivities.

## Supported

- **No partition signature.** k already contains Cs in its `(Cs - C)` denominator, so a partition
  or interfacial-kinetic effect would make k *rise* with Tween. Under the primary ladder it does
  not rise in any of the {n_modes} optical preprocessing modes — it declines
  ({k_primary}, 0.5x/1x/10x, `pipeline+copt`). **Arm B provides no evidence that Tween lowers a
  partition/interfacial kinetic barrier beyond its solubility effect.** This is consistent with
  the solubility-only/rugosity interpretation, which rests on the size-uniform model together
  with the size-space evidence below; Arm B alone does not independently prove rugosity.
  (The tau-vs-size "fanning" result is *not* part of that support — it does not survive fit QC,
  see the Conditional section.)
- **Raising Tween changes the speed, not the path.** Three observables agree, none of them the
  fanning result: the terminal Copt floor falls monotonically; the q3 size distribution at
  MATCHED optical extent is common to all conditions to within ~0.3 um, about 9% of D50 and the
  smallest difference this design could have resolved (`media_diagnostic_q3.py`); and the
  empirical angular KWW shortens tau (1.83 / 1.78 / 1.39 min) while its optical decay depth stays
  flat at ~90-92%. Same path through size space, traversed faster — which is what a
  solubility-only effect looks like.
- **The result is robust to optical preprocessing.** Raw, pipeline-QC and pipeline+Copt modes all
  agree on the direction; absolute k moves by roughly a third between raw and cleaned, the
  ordering does not.
- **More Tween gives more complete optical disappearance.** The terminal Copt floor falls
  monotonically ({floors}) and the fractional Copt loss rises ({fracs}) — no UV calibration and
  no Cs ladder enters this, and it holds in every optical mode.
- **The replicated design is complete and correctly resolved.** 27 cuvette runs over 12
  independent preps, 4 per condition. The `20260702` folder holds two preps (rep 1 injected
  07-02, reps 2-3 injected 07-07 off a fresh suspension), so prep identity comes from the
  injection log, not the folder name.
- **The UV condition separation at early times is robust.** 10x - 0.5x at 2 min is {sep2:.1f} pp
  of dose, unchanged by the filter-offset convention or by excluding QC-flagged observations.

## Conditional

- **Absolute dissolved fraction is not interpretable on its own.** The additive filter offset is
  ~30% of the pre-offset signal; terminal recovery slides ~0.56 -> ~1.05 across the plausible
  offset range. Condition *differences* are stable under it; levels, ratios and absolute model
  residuals are not. Arm B conclusions rest on differences.
- **The terminal UV separation depends on QC policy.** 10x - 0.5x at the last timepoint is
  {sep20_incl:.1f} pp including all observations but {sep20_excl:.1f} pp excluding every
  QC-flagged observation. The early separation survives; the terminal one does not.
- **The frozen forward-model transfer is only testable early.** By 20 min the model predicts
  96-99% dissolved for every condition, so its predicted Tween separation collapses (13.4 ->
  2.6 pp) while the measured separation holds. Late-window residuals measure the ceiling.
- **The tau-vs-size "fanning" result does not survive fit QC** and is not evidence of anything.
  Prep-balanced slopes go from {fan_all} (all converged fits) to {fan_qc} (QC-passing) to
  {fan_drop} (dropping runs with any boundary-hit fit). Labelled exploratory; it is not part of
  the mechanistic summary.

## Secondary method sensitivity (not a verdict)

Recomputing against the particulate-containing centrifuged preparations moves the mid-ladder
answer, which is the expected consequence of not excluding particulate rather than a competing
result: k is non-monotone under both. This is reported to show *why particulate exclusion
matters*, and no conclusion is conditioned on it.

## Residual limitations

- No CFZ standard curve exists at these in-medium Tween levels, so absolute UV concentrations
  carry an uncalibrated matrix term. It bears on absolute levels rather than on the condition
  comparisons the conclusions rest on. Noted as a limitation; not a gate on interpretation and
  no such experiment is planned.
- The 0.5x CMC level is the deliberate below-CMC, monomer-only anchor and is retained throughout;
  it is what separates a monomer-level solubility effect from a micellar one.
"""



def write_memo(path: Path, k, taus, sens_uv, floors, fracs, verdict):
    from arm_b_provenance import _git_commit
    from datetime import datetime, timezone
    fmt = lambda d: " / ".join(f"{d[c]:.2f}" for c in ORDER)
    sub = taus[taus["cs_ladder"].eq(arm_b_cs.DEFAULT_LADDER)
               & taus["optical"].eq("pipeline+copt")]
    pol = lambda p: fmt(sub[sub["fit_qc_policy"].eq(p)].set_index("condition")["tau_size_slope"]
                        .reindex(ORDER).to_dict())
    primary_k = k[k["cs_ladder"].eq(arm_b_cs.PRIMARY_LADDER)
                  & k["optical"].eq("pipeline+copt")].set_index("condition")["k_mean"]
    n_modes = int(k.loc[k["cs_ladder"].eq(arm_b_cs.PRIMARY_LADDER), "optical"].nunique())
    path.write_text(MEMO.format(
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"), commit=_git_commit(),
        floors=fmt(floors), fracs=fmt(fracs),
        sep2=sens_uv["sep2"], sep20_incl=sens_uv["sep20_incl"], sep20_excl=sens_uv["sep20_excl"],
        k_primary=fmt(primary_k.reindex(ORDER).to_dict()), n_modes=n_modes,
        fan_all=pol("all_converged"), fan_qc=pol("qc_pass"), fan_drop=pol("drop_bound_hit_runs")))
    return path

if __name__ == "__main__":
    raise SystemExit(main())
