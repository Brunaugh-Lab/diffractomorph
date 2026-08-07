"""Regenerate Arm A candidate-model trajectories as **out-of-fold** predictions.

``rate_refinement_predictions.csv`` tabulates each candidate model at its **all-data** optimum:
one ``rate_scale`` per model, fitted on every preparation date including the one being drawn.
Plotting those next to a leave-one-preparation-date-out RMSE would show a trajectory that is
more informed than the error bar beside it.

This module re-applies :func:`rate_scale_refinement.predict_curve` — the same prediction function
the leave-one-date-out validation itself calls — using, for each preparation date, the parameters
that were fitted with that date **withheld**. Those fold parameters are read from
``rate_refinement_lodo_validation.csv`` (``fitted_rate_scale``, ``fitted_participating_pct``);
nothing is refitted here and no new fitting strategy is introduced.

Three of the four narrative models are parameter-free (``injection_base`` and ``first_uv_anchor``
at rate 1.0, ``injection_selected_rate`` at the historical 2.197), so their out-of-fold and
all-data trajectories are identical by construction. Only ``anchored_rate`` has a parameter that
changes between folds. Every row records which basis it was produced on, so the distinction
survives into the figure rather than being asserted in prose.

The Arm A curve construction below mirrors the Arm A branch of
:func:`rate_scale_refinement.build_curves`, importing that module's :class:`RunCurve` and
:func:`_interp_monotone_time` so the base trajectories are produced by the same code. The
equivalence is not assumed: :func:`verify_against_all_data_artifact` re-derives the all-data
predictions and checks them against ``rate_refinement_predictions.csv``.

Writes ``rate_refinement_out_of_fold_predictions.csv`` beside the artifacts it draws on. Solving
the 18 Arm A runs takes a few minutes. Run with the pipeline venv.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from diffractomorph_pipeline import solubility

from arm_a_common import default_study_root
from forward_predict import V_ML, _arm_a_runs, _predict_run
from psd_angular_fit import _uv_for
from rate_scale_refinement import RunCurve, _interp_monotone_time, predict_curve

COHORT = "antisolvent Tween 80 concentration transfer"
# The four models the revised Section 3.4.1 figure needs. The extent candidates stay in the
# rate_refinement artifacts and Table S8; they are not regenerated here.
MODELS = ("injection_base", "injection_selected_rate", "first_uv_anchor", "anchored_rate")
# Models whose parameters are fixed a priori, so no fold can change them.
PARAMETER_FREE = ("injection_base", "injection_selected_rate", "first_uv_anchor")
# Display grid. Injection-started models are drawn from injection; aligned models begin at the
# first UV observation, which is where they are initialised.
DENSE_STEP_MIN = 0.25


def build_arm_a_curves(study_root: Path) -> list[RunCurve]:
    """Arm A run curves, built exactly as ``rate_scale_refinement.build_curves`` builds them.

    The pH cohort that ``build_curves`` also assembles is deliberately not constructed: no
    pH-development observation is read by this module.
    """
    cs_model = solubility.load_default()
    runs, base, _ = _arm_a_runs(study_root)
    curves = []
    for rd in runs:
        measured = _uv_for(rd, "arm_a", base)
        if measured is None:
            continue
        time_min, dissolved_ugml = measured
        loaded = float(rd["dose_mg"]) * 1e3 / V_ML
        observed = 100.0 * np.asarray(dissolved_ugml, float) / loaded
        run, _ = _predict_run(rd, cs_model, t_end=10800.0, n_eval=1200)
        base_time = np.asarray(run.t, float) / 60.0
        base_pct = np.maximum.accumulate(np.asarray(run.pct_dissolved, float))
        first_model_time = _interp_monotone_time(base_time, base_pct, observed[0])
        curves.append(RunCurve(
            id=str(rd["id"]), cohort=COHORT, condition=str(rd["cond"]),
            date=str(rd.get("date", "")), rep=int(rd["rep"]),
            time_min=np.asarray(time_min, float), observed_pct=observed,
            base_time_min=base_time, base_pct=base_pct,
            first_model_time_min=first_model_time,
            first_anchor_clipped=bool(observed[0] < base_pct[0] - 1e-8
                                      or observed[0] > base_pct[-1] + 1e-8),
            dose_mg=float(rd["dose_mg"]),
            qc_dv50_um=float(rd["psd"].dv50), qc_d32_um=float(rd["psd"].d32),
        ))
    if not curves:
        raise ValueError("no Arm A run curves were built")
    return curves


def _fold_parameters(lodo: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    """(model, held_date) -> (rate_scale, participating_pct) fitted with that date withheld."""
    sub = lodo[lodo["cohort"].eq(COHORT)]
    if sub.empty:
        raise ValueError(f"cohort {COHORT!r} absent from the leave-one-date-out artifact")
    out = {}
    for r in sub.itertuples():
        out[(str(r.model), str(r.held_date))] = (float(r.fitted_rate_scale),
                                                 float(r.fitted_participating_pct))
    return out


def _all_data_parameters(candidates: pd.DataFrame) -> dict[str, tuple[float, float]]:
    sub = candidates[candidates["cohort"].eq(COHORT)]
    return {str(r.model): (float(r.rate_scale), float(r.participating_pct))
            for r in sub.itertuples()}


def _dense_grid(model: str, curve: RunCurve) -> np.ndarray:
    """Display times for one model: from injection, or from the first UV observation."""
    end = float(curve.time_min.max())
    start = 0.0 if model.startswith("injection") else float(curve.first_time_min)
    grid = np.arange(start, end + DENSE_STEP_MIN / 2, DENSE_STEP_MIN)
    return np.unique(np.r_[grid, curve.time_min[curve.time_min >= start]])


def predict_rows(curves, fold_params, all_data_params) -> pd.DataFrame:
    """Per-run predictions on the UV grid and a dense display grid, both bases recorded."""
    rows = []
    for model in MODELS:
        for curve in curves:
            fold = fold_params.get((model, curve.date))
            if fold is None:
                raise ValueError(f"no fold parameters for {model!r} with {curve.date!r} withheld")
            for basis, (rate, extent) in (("out_of_fold", fold),
                                          ("all_data", all_data_params[model])):
                for grid_name, times in (("uv", curve.time_min),
                                         ("dense", _dense_grid(model, curve))):
                    shifted = replace(curve, time_min=np.asarray(times, float))
                    pred = predict_curve(shifted, model, rate_scale=rate,
                                         participating_pct=extent)
                    observed = (curve.observed_pct if grid_name == "uv"
                                else np.full(len(times), np.nan))
                    for i, (t, p, o) in enumerate(zip(times, pred, observed)):
                        rows.append(dict(
                            cohort=COHORT, model=model, prediction_basis=basis,
                            grid=grid_name, id=curve.id, condition=curve.condition,
                            date=curve.date, rep=curve.rep, held_date=curve.date,
                            time_min=float(t), predicted_pct_injected=float(p),
                            observed_pct_injected=float(o),
                            is_first_observation=bool(grid_name == "uv" and i == 0),
                            rate_scale=rate, participating_pct=extent,
                            parameter_free=model in PARAMETER_FREE,
                            first_model_time_min=curve.first_model_time_min,
                            first_clock_time_min=curve.first_time_min))
    return pd.DataFrame(rows)


def verify_against_all_data_artifact(table: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """The regenerated all-data predictions must reproduce the published artifact exactly.

    This is what makes the out-of-fold regeneration auditable: if the curve construction here had
    drifted from ``rate_scale_refinement.build_curves``, the all-data arm would disagree.
    """
    mine = table[table["prediction_basis"].eq("all_data") & table["grid"].eq("uv")]
    theirs = predictions[predictions["cohort"].eq(COHORT) & predictions["model"].isin(MODELS)]
    keys = ["model", "id", "time_min"]
    merged = mine.merge(theirs[keys + ["predicted_pct_injected", "observed_pct_injected"]],
                        on=keys, suffixes=("_mine", "_artifact"), validate="one_to_one")
    if len(merged) != len(theirs):
        raise ValueError(f"all-data comparison covered {len(merged)} of {len(theirs)} rows")
    pred_diff = float(np.abs(merged["predicted_pct_injected_mine"]
                             - merged["predicted_pct_injected_artifact"]).max())
    obs_diff = float(np.abs(merged["observed_pct_injected_mine"]
                            - merged["observed_pct_injected_artifact"]).max())
    checks = {"n_rows_compared": int(len(merged)),
              "max_abs_predicted_diff_pct": pred_diff,
              "max_abs_observed_diff_pct": obs_diff,
              "tolerance_pct": 1e-6,
              "ok": pred_diff < 1e-6 and obs_diff < 1e-6}
    if not checks["ok"]:
        raise ValueError(f"regenerated all-data predictions diverge from the artifact: {checks}")
    return checks


def out_of_fold_differs_from_all_data(table: pd.DataFrame) -> dict:
    """Which models actually move between the two bases — recorded, not assumed."""
    out = {}
    for model in MODELS:
        sub = table[table["model"].eq(model) & table["grid"].eq("uv")]
        wide = sub.pivot_table(index=["id", "time_min"], columns="prediction_basis",
                               values="predicted_pct_injected")
        out[model] = {
            "max_abs_diff_pct": float(np.abs(wide["out_of_fold"] - wide["all_data"]).max()),
            "parameter_free": model in PARAMETER_FREE,
            "fold_rate_scales": sorted(
                {round(float(v), 6) for v in
                 table[table["model"].eq(model)
                       & table["prediction_basis"].eq("out_of_fold")]["rate_scale"]}),
        }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--study-root", type=Path, default=None)
    args = p.parse_args(argv)
    study_root = args.study_root or default_study_root()
    refinement = study_root / "analysis" / "rate_refinement"

    lodo = pd.read_csv(refinement / "rate_refinement_lodo_validation.csv")
    candidates = pd.read_csv(refinement / "rate_refinement_candidate_models.csv")
    predictions = pd.read_csv(refinement / "rate_refinement_predictions.csv")

    curves = build_arm_a_curves(study_root)
    table = predict_rows(curves, _fold_parameters(lodo), _all_data_parameters(candidates))
    checks = verify_against_all_data_artifact(table, predictions)
    moved = out_of_fold_differs_from_all_data(table)

    path = refinement / "rate_refinement_out_of_fold_predictions.csv"
    table.to_csv(path, index=False)
    print(f"runs={len(curves)} dates={len({c.date for c in curves})} rows={len(table)}")
    print(f"all-data equivalence vs published artifact: {checks}")
    for model, info in moved.items():
        print(f"  {model:26s} out-of-fold vs all-data max |diff| = "
              f"{info['max_abs_diff_pct']:.6f} pp   folds={info['fold_rate_scales']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
