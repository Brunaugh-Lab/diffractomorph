"""Frozen pH 4.5 forward-model transfer to the Arm B in-medium Tween 80 study.

Parallels :mod:`antisolvent_tween80_model_transfer`, but perturbs the *medium* rather than the
starting suspension. Each run is predicted from its own delivered dose (:mod:`arm_b_injected_mass`)
and its own starting-suspension q0 PSD, with **the primary filtered Arm C Cs for that Tween level** supplied
as an explicit per-run Cs — so the model already contains whatever extra solubility the micelles
provide. Two models are reported:

  ``base_rate_scale_1``           the unscaled mechanistic Nernst-Brunner prediction
  ``frozen_selected_rate_scale``  the same, with the rate scale selected on the pH-dependent
                                  study frozen and transferred unchanged — nothing is refit here

**The model saturates, so only the early window is informative.** By 20 min the frozen model
predicts 96.5 / 97.2 / 99.2 % dissolved for 0.5x / 1.0x / 10x — pinned against the 100 % ceiling
with no dynamic range left to express a Cs effect. Its predicted Tween separation therefore decays
from 13.4 pp at 2 min to 2.6 pp at 20 min, while the measured separation holds at 13–20 pp
throughout. A residual computed on the late window is measuring that ceiling, not chemistry, which
is why ``SATURATION_PCT`` flags saturated rows and the summary reports the early window first.

Compare **differences** between conditions, not ratios or levels. The prediction is offset-free,
but the measured side carries the additive filter offset, which shifts every condition's level
together (see :mod:`arm_b_uv_timecourse`): differences are invariant to it — 10x−0.5x moves only
16.2→16.6 pp across the whole plausible offset range — whereas ratios and absolute residuals are
not (the 10x/0.5x ratio slides 1.22→1.42 over the same range).

On the early window the model's Cs-driven Tween effect is close to the measured one (13.4 vs
16.2 pp at 2 min; 11.2 vs 14.7 at 4 min), so the primary filtered Cs accounts for most of the
early Tween effect. What it does not reproduce is the *sustained* separation in extent. This is
consistent with the declining Cs-normalized rate constant in :mod:`arm_b_partition` rather than in
conflict with it: dissolution here is not linear in Cs (it saturates), so measured mass can sit
above the model while still growing more slowly than the (Cs−C) driving force.

Residuals aggregate on the replication structure — technical repeats within a prep, then preps
(n = 4 per condition), never runs. Run with the pipeline venv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diffractomorph_pipeline import solubility
from diffractomorph_pipeline.config import data_root
from diffractomorph_pipeline.forward import PSD

from arm_b_common import CONDITIONS, default_study_root, discover_runs
import arm_b_cs
from arm_b_partition import CENTRIFUGE_FILTER_DISAGREEMENT, COLORS
from arm_b_provenance import provenance_record, write_provenance
from arm_b_uv_timecourse import CELL_ML, PH, process_uv
from forward_predict import _predict_run

FROZEN_FIT = ("disso_experiments/ph_dependent_dissolution_study/forward_prediction/"
              "scalar_fit/selected_rate_only_fit_summary.csv")
# Above this predicted % dissolved the model is pinned against the 100% ceiling and can no longer
# express a Cs difference, so residuals there measure saturation rather than chemistry.
SATURATION_PCT = 95.0
EARLY_MAX_MIN = 5.0


def selected_rate_scale(path: Path) -> float:
    table = pd.read_csv(path)
    if len(table) != 1 or "rate_scale_datebalanced" not in table:
        raise ValueError(f"invalid selected-model summary: {path}")
    return float(table.loc[0, "rate_scale_datebalanced"])


def predict_at_uv_times(long: pd.DataFrame, runs: pd.DataFrame, rate_scale: float,
                        cs_map: dict) -> pd.DataFrame:
    """Predict each run at its own UV sampling times, base and frozen-rate-scale."""
    cs_model = solubility.load_default()
    psd_cache: dict[str, PSD] = {}
    meta = {r.run_id: r for r in runs.itertuples()}
    rows = []
    for rid, g in long.groupby("run_id", sort=False):
        g = g.sort_values("time_min")
        info = meta[rid]
        key = str(info.qc_psd_dir)
        if key not in psd_cache:
            psd_cache[key] = PSD.from_sympatec(info.qc_psd_dir)
        psd = psd_cache[key]

        time_min = g["time_min"].to_numpy(float)
        observed = g["pct_injected"].to_numpy(float)
        dose_mg = float(g["injected_mass_mg"].iloc[0])
        condition = g["condition"].iloc[0]
        cs = cs_map[condition]
        t_end = float(time_min.max() * 60.0 * max(1.0, rate_scale) * 1.02)
        spec = {"ph": PH, "dose_mg": dose_mg, "psd": psd, "cs_ugml": cs}
        run, cs_used = _predict_run(spec, cs_model, t_end, 500)
        base_pct = np.interp(time_min * 60.0, run.t, run.pct_dissolved)
        frozen_pct = np.interp(time_min * 60.0 * rate_scale, run.t, run.pct_dissolved)

        for t, obs, b, f in zip(time_min, observed, base_pct, frozen_pct):
            common = dict(run_id=rid, condition=condition, xcmc=float(g["xcmc"].iloc[0]),
                          prep=g["prep"].iloc[0], prep_index=int(g["prep_index"].iloc[0]),
                          rep=int(g["rep"].iloc[0]), time_min=float(t), dose_mg=dose_mg,
                          cs_ugml=cs_used,
                          centrifuge_filter_disagreement=(
                              condition in CENTRIFUGE_FILTER_DISAGREEMENT),
                          qc_dv50_um=psd.dv50, qc_d32_um=psd.d32,
                          measured_pct_injected=float(obs), selected_rate_scale=rate_scale)
            rows.append(dict(**common, model="base_rate_scale_1",
                             predicted_pct_injected=float(b),
                             residual_model_minus_measured_pct=float(b - obs),
                             model_saturated=bool(b >= SATURATION_PCT)))
            rows.append(dict(**common, model="frozen_selected_rate_scale",
                             predicted_pct_injected=float(f),
                             residual_model_minus_measured_pct=float(f - obs),
                             model_saturated=bool(f >= SATURATION_PCT)))
    return pd.DataFrame(rows)


def residual_summaries(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Residual RMSE per prep, then combined across preps — never pooled over raw runs."""
    def rmse(values):
        values = np.asarray(values, float)
        return float(np.sqrt(np.mean(values ** 2))) if values.size else np.nan

    by_prep = []
    for (model, condition, xcmc, prep), g in predictions.groupby(
            ["model", "condition", "xcmc", "prep"], sort=True):
        residual = g["residual_model_minus_measured_pct"].to_numpy(float)
        t = g["time_min"].to_numpy(float)
        early = t <= EARLY_MAX_MIN
        by_prep.append(dict(model=model, condition=condition, xcmc=xcmc, prep=prep,
                            n_technical_reps=g["rep"].nunique(), n_obs=len(g),
                            n_unsaturated_obs=int((~g["model_saturated"]).sum()),
                            early_rmse_pct=rmse(residual[early]),
                            early_signed_resid_pct=float(residual[early].mean()),
                            rmse_pct=rmse(residual),
                            signed_mean_resid_pct=float(residual.mean()),
                            late_rmse_pct=rmse(residual[~early])))
    by_prep = pd.DataFrame(by_prep)

    overall = []
    for (model, condition, xcmc), g in by_prep.groupby(["model", "condition", "xcmc"], sort=True):
        overall.append(dict(
            model=model, condition=condition, xcmc=xcmc, n_preps=g["prep"].nunique(),
            frac_obs_unsaturated=float(g["n_unsaturated_obs"].sum() / g["n_obs"].sum()),
            early_prep_balanced_rmse_pct=float(
                np.sqrt(np.mean(g["early_rmse_pct"].to_numpy(float) ** 2))),
            early_mean_signed_resid_pct=float(g["early_signed_resid_pct"].mean()),
            prep_balanced_rmse_pct=float(np.sqrt(np.mean(g["rmse_pct"].to_numpy(float) ** 2))),
            mean_signed_resid_pct=float(g["signed_mean_resid_pct"].mean()),
            signed_resid_sd_pct=float(g["signed_mean_resid_pct"].std(ddof=1)),
            prep_rmse_sd_pct=float(g["rmse_pct"].std(ddof=1))))
    return by_prep, pd.DataFrame(overall).sort_values(["model", "xcmc"]).reset_index(drop=True)


def _plot(predictions: pd.DataFrame, path: Path, cs_map: dict):
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(15, 4.7), sharey=True)
    for ax, condition in zip(axes, CONDITIONS):
        sub = predictions[predictions["condition"].eq(condition)]
        frozen = sub[sub["model"].eq("frozen_selected_rate_scale")]
        for _, g in frozen.groupby("prep"):
            m = g.groupby("time_min", as_index=False)[
                ["measured_pct_injected", "predicted_pct_injected"]].mean()
            ax.plot(m["time_min"], m["measured_pct_injected"], "o", ms=3,
                    color=COLORS[condition], alpha=0.30)
            ax.plot(m["time_min"], m["predicted_pct_injected"], "-", lw=1.0,
                    color=COLORS[condition], alpha=0.30)
        # prep-balanced: technical repeats collapse within a prep, then preps weigh equally
        cols = ["measured_pct_injected", "predicted_pct_injected"]
        per_prep = sub.groupby(["model", "prep", "time_min"], as_index=False)[cols].mean()
        means = per_prep.groupby(["model", "time_min"], as_index=False)[cols].mean()
        obs = means[means["model"].eq("frozen_selected_rate_scale")]
        base = means[means["model"].eq("base_rate_scale_1")]
        ax.plot(obs["time_min"], obs["measured_pct_injected"], "o", color="black", ms=5,
                label="UV mass")
        ax.plot(base["time_min"], base["predicted_pct_injected"], "--", color="0.45",
                label="base model")
        ax.plot(obs["time_min"], obs["predicted_pct_injected"], "-", lw=2.2,
                color=COLORS[condition], label="frozen selected model")
        flag = ("  (centrifuged read disagrees here)"
                if condition in CENTRIFUGE_FILTER_DISAGREEMENT else "")
        ax.set_title(f"{condition}   Cs={cs_map[condition]:g} µg/mL{flag}", fontsize=10)
        ax.set_xlabel("Time (min)"); ax.grid(alpha=0.2)
    axes[0].set_ylabel("Dissolved CFZ (% of delivered dose)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Arm B in-medium Tween 80 — frozen pH 4.5 model transfer "
                 "(primary filtered Cs per Tween level; nothing refit here)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Arm B frozen pH 4.5 model transfer.")
    p.add_argument("--study-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--fit-summary", type=Path, default=None)
    p.add_argument("--cs-ladder", default=arm_b_cs.DEFAULT_LADDER, choices=arm_b_cs.LADDER_NAMES)
    args = p.parse_args(argv)
    root = args.study_root or default_study_root()
    out_dir = args.output_dir or root / "analysis" / "model_transfer" / args.cs_ladder
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_summary = args.fit_summary or (data_root() / FROZEN_FIT)

    rate_scale = selected_rate_scale(fit_summary)
    cs_map = arm_b_cs.cs_map(args.cs_ladder)
    long = process_uv(root, args.cs_ladder)       # carries run_id, prep identity and dose
    predictions = predict_at_uv_times(long, discover_runs(root), rate_scale, cs_map)
    by_prep, overall = residual_summaries(predictions)

    paths = {"predictions": out_dir / "arm_b_model_transfer_predictions.csv",
             "residuals": out_dir / "arm_b_model_transfer_residuals_by_prep.csv",
             "summary": out_dir / "arm_b_model_transfer_summary.csv",
             "figure": out_dir / "arm_b_model_transfer.png"}
    predictions.to_csv(paths["predictions"], index=False)
    by_prep.to_csv(paths["residuals"], index=False)
    overall.to_csv(paths["summary"], index=False)
    _plot(predictions, paths["figure"], cs_map)
    write_provenance(out_dir / "provenance.json",
                     provenance_record("arm_b_model_transfer", cs_ladder=args.cs_ladder,
                                       study_root=root, frozen_rate_scale=rate_scale,
                                       saturation_pct=SATURATION_PCT))

    print(f"frozen rate scale (from the pH study, not refit): {rate_scale:.3f}\n")
    print(overall.to_string(index=False))

    frozen = predictions[predictions["model"].eq("frozen_selected_rate_scale")]
    cols = ["predicted_pct_injected", "measured_pct_injected"]
    per_prep = frozen.groupby(["condition", "prep", "time_min"], as_index=False)[cols].mean()
    sep = per_prep.groupby(["condition", "time_min"])[cols].mean().unstack(0)   # prep-balanced
    lo, hi = "0.5x CMC", "10x CMC"
    print("\nTween separation (10x minus 0.5x, pp of dose) — the offset-invariant comparison:")
    print(f"{'t_min':>6} {'model':>8} {'measured':>9}   model saturated?")
    for t in sep.index:
        m = sep[("predicted_pct_injected", hi)][t] - sep[("predicted_pct_injected", lo)][t]
        o = sep[("measured_pct_injected", hi)][t] - sep[("measured_pct_injected", lo)][t]
        sat = sep[("predicted_pct_injected", hi)][t] >= SATURATION_PCT
        print(f"{t:6.1f} {m:8.2f} {o:9.2f}   {'YES' if sat else '-'}")
    print(f"\nThe model reaches the {SATURATION_PCT:.0f}% ceiling and loses the ability to express a "
          f"Cs difference; read the early (<= {EARLY_MAX_MIN:.0f} min) columns, not the pooled ones.")
    for path in paths.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
