"""Refine CFZ rate scaling by separating initialization, post-capture rate, and extent."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diffractomorph_pipeline import solubility
from diffractomorph_pipeline.config import data_root

from arm_a_common import default_study_root
from forward_predict import V_ML, _arm_a_runs, _ph_dependent_runs, _predict_run
from psd_angular_fit import _uv_for


RS_GRID = np.unique(np.r_[np.geomspace(0.25, 3.5, 81), 1.0])
PART_GRID = np.linspace(65.0, 105.0, 81)
N_BOOT = 1000
ANCHORED = {"first_uv_anchor", "anchored_rate", "anchored_participation", "anchored_rate_participation"}
MODELS = (
    ("injection_base", 0),
    ("injection_selected_rate", 0),
    ("first_uv_anchor", 0),
    ("anchored_rate", 1),
    ("anchored_participation", 1),
    ("anchored_rate_participation", 2),
)


@dataclass
class RunCurve:
    id: str
    cohort: str
    condition: str
    date: str
    rep: int
    time_min: np.ndarray
    observed_pct: np.ndarray
    base_time_min: np.ndarray
    base_pct: np.ndarray
    first_model_time_min: float
    first_anchor_clipped: bool = False
    dose_mg: float = np.nan
    qc_dv50_um: float = np.nan
    qc_d32_um: float = np.nan

    @property
    def first_time_min(self) -> float:
        return float(self.time_min[0])

    @property
    def first_observed_pct(self) -> float:
        return float(self.observed_pct[0])

    @property
    def model_ceiling_pct(self) -> float:
        return float(self.base_pct[-1])


def _interp_monotone_time(base_time, base_pct, target_pct):
    pct = np.maximum.accumulate(np.asarray(base_pct, float))
    keep = np.r_[True, np.diff(pct) > 1e-10]
    return float(np.interp(target_pct, pct[keep], np.asarray(base_time, float)[keep]))


def build_curves(study_root: Path) -> dict[str, list[RunCurve]]:
    """Solve each run once at base rate and attach run-matched UV mass observations."""
    cs_model = solubility.load_default()
    specs = []
    ph_runs, ph_base, _ = _ph_dependent_runs()
    specs.append(("historical pH 4.0/4.5 fit cohort", "ph", ph_base,
                  [r for r in ph_runs if float(r["ph"]) in (4.0, 4.5)]))
    tween_runs, tween_base, _ = _arm_a_runs(study_root)
    specs.append(("antisolvent Tween 80 concentration transfer", "arm_a", tween_base, tween_runs))
    cohorts: dict[str, list[RunCurve]] = {}
    for cohort, experiment, base, runs in specs:
        curves = []
        for rd in runs:
            measured = _uv_for(rd, experiment, base)
            if measured is None:
                continue
            time_min, dissolved_ugml = measured
            loaded = float(rd["dose_mg"]) * 1e3 / V_ML
            observed = 100.0 * np.asarray(dissolved_ugml, float) / loaded
            run, _ = _predict_run(rd, cs_model, t_end=10800.0, n_eval=1200)
            base_time = np.asarray(run.t, float) / 60.0
            base_pct = np.maximum.accumulate(np.asarray(run.pct_dissolved, float))
            first_model_time = _interp_monotone_time(base_time, base_pct, observed[0])
            anchor_clipped = bool(observed[0] < base_pct[0] - 1e-8 or observed[0] > base_pct[-1] + 1e-8)
            curves.append(RunCurve(
                id=str(rd["id"]), cohort=cohort, condition=str(rd["cond"]),
                date=str(rd.get("date", "")), rep=int(rd["rep"]),
                time_min=np.asarray(time_min, float), observed_pct=observed,
                base_time_min=base_time, base_pct=base_pct,
                first_model_time_min=first_model_time, first_anchor_clipped=anchor_clipped,
                dose_mg=float(rd["dose_mg"]),
                qc_dv50_um=float(rd["psd"].dv50), qc_d32_um=float(rd["psd"].d32),
            ))
        cohorts[cohort] = curves
    return cohorts


def predict_curve(curve: RunCurve, model: str, *, rate_scale=1.0,
                  participating_pct=100.0) -> np.ndarray:
    """Predict UV mass in percent of injected dose for one candidate model."""
    t = curve.time_min
    if model == "injection_base":
        lookup = t
    elif model == "injection_selected_rate":
        lookup = rate_scale * t
    else:
        lookup = curve.first_model_time_min + rate_scale * (t - curve.first_time_min)
    base = np.interp(lookup, curve.base_time_min, curve.base_pct)
    base_at_anchor = float(np.interp(curve.first_model_time_min, curve.base_time_min, curve.base_pct))
    if model in ("anchored_participation", "anchored_rate_participation"):
        if participating_pct < curve.first_observed_pct:
            return np.full(t.shape, np.nan)
        denom = curve.model_ceiling_pct - base_at_anchor
        progress = ((base - base_at_anchor) / denom
                    if abs(denom) > 1e-9 else np.zeros_like(base))
        return curve.first_observed_pct + (participating_pct - curve.first_observed_pct) * progress
    if model in ANCHORED:
        return curve.first_observed_pct + (base - base_at_anchor)
    return base


def _unit_mse(curves, model, rate_scale, participating_pct, *, include_first=False):
    blocks: dict[tuple[str, str], list[float]] = {}
    for curve in curves:
        pred = predict_curve(curve, model, rate_scale=rate_scale, participating_pct=participating_pct)
        keep = np.ones(curve.time_min.size, dtype=bool)
        if not include_first:
            keep[0] = False
        residual = pred[keep] - curve.observed_pct[keep]
        if not np.isfinite(residual).all():
            return np.inf
        blocks.setdefault((curve.date, curve.condition), []).extend(residual.tolist())
    return float(np.mean([np.mean(np.square(v)) for v in blocks.values()]))


def fit_candidate(curves, model):
    """Date-condition-balanced grid fit; parameters remain diagnostic until validated."""
    if model == "injection_selected_rate":
        return 2.197, 100.0, _unit_mse(curves, model, 2.197, 100.0)
    if model in ("injection_base", "first_uv_anchor"):
        return 1.0, 100.0, _unit_mse(curves, model, 1.0, 100.0)
    if model == "anchored_rate":
        values = np.array([_unit_mse(curves, model, r, 100.0) for r in RS_GRID])
        i = int(np.argmin(values))
        return float(RS_GRID[i]), 100.0, float(values[i])
    if model == "anchored_participation":
        values = np.array([_unit_mse(curves, model, 1.0, f) for f in PART_GRID])
        i = int(np.argmin(values))
        return 1.0, float(PART_GRID[i]), float(values[i])
    values = np.array([[_unit_mse(curves, model, r, f) for f in PART_GRID] for r in RS_GRID])
    i, j = np.unravel_index(np.argmin(values), values.shape)
    return float(RS_GRID[i]), float(PART_GRID[j]), float(values[i, j])


def score_candidate(curves, model, rate_scale, participating_pct):
    unit_rows = []
    all_residuals = []
    post_residuals = []
    for curve in curves:
        pred = predict_curve(curve, model, rate_scale=rate_scale, participating_pct=participating_pct)
        residual = pred - curve.observed_pct
        all_residuals.extend(residual.tolist())
        post_residuals.extend(residual[1:].tolist())
        unit_rows.append((curve.date, curve.condition, residual))
    blocks = {}
    for date, condition, residual in unit_rows:
        blocks.setdefault((date, condition), []).extend(residual.tolist())
    post_blocks = {}
    for curve in curves:
        pred = predict_curve(curve, model, rate_scale=rate_scale, participating_pct=participating_pct)
        post_blocks.setdefault((curve.date, curve.condition), []).extend((pred[1:] - curve.observed_pct[1:]).tolist())
    return dict(
        all_obs_rmse_pct=float(np.sqrt(np.mean(np.square(all_residuals)))),
        post_first_rmse_pct=float(np.sqrt(np.mean([np.mean(np.square(v)) for v in post_blocks.values()]))),
        post_first_signed_resid_pct=float(np.mean(post_residuals)),
        n_date_condition_units=len(blocks),
    )


def candidate_profiles(curves, cohort):
    rows = []
    for model in ("anchored_rate", "anchored_participation", "anchored_rate_participation"):
        values = []
        rates = RS_GRID if model != "anchored_participation" else np.array([1.0])
        extents = PART_GRID if model != "anchored_rate" else np.array([100.0])
        for rate in rates:
            for extent in extents:
                objective = _unit_mse(curves, model, float(rate), float(extent))
                values.append((float(rate), float(extent), objective))
        finite = [v[2] for v in values if np.isfinite(v[2])]
        minimum = min(finite)
        for rate, extent, objective in values:
            rows.append(dict(
                cohort=cohort, model=model, rate_scale=rate, participating_pct=extent,
                date_balanced_mse=objective,
                delta_mse=(objective - minimum if np.isfinite(objective) else np.nan),
            ))
    return rows


def leave_one_date_out(curves, cohort):
    rows = []
    for held_date in sorted({c.date for c in curves}):
        train = [c for c in curves if c.date != held_date]
        held = [c for c in curves if c.date == held_date]
        for model, n_params in MODELS:
            rate, extent, _ = fit_candidate(train, model)
            score = score_candidate(held, model, rate, extent)
            rows.append(dict(
                cohort=cohort, held_date=held_date, model=model, n_params=n_params,
                fitted_rate_scale=rate, fitted_participating_pct=extent,
                held_post_first_rmse_pct=score["post_first_rmse_pct"],
                held_all_obs_rmse_pct=score["all_obs_rmse_pct"],
                held_signed_resid_pct=score["post_first_signed_resid_pct"],
            ))
    return rows


def fitted_predictions(curves, cohort, fits):
    rows, residual_rows = [], []
    for fit in fits:
        model, rate, extent = fit["model"], fit["rate_scale"], fit["participating_pct"]
        by_unit = {}
        for curve in curves:
            pred = predict_curve(curve, model, rate_scale=rate, participating_pct=extent)
            residual = pred - curve.observed_pct
            for i, (time, observed, predicted, resid) in enumerate(zip(
                    curve.time_min, curve.observed_pct, pred, residual)):
                rows.append(dict(
                    cohort=cohort, model=model, id=curve.id, condition=curve.condition,
                    date=curve.date, rep=curve.rep, time_min=time, is_first_observation=i == 0,
                    observed_pct_injected=observed, predicted_pct_injected=predicted,
                    residual_model_minus_measured_pct=resid, rate_scale=rate,
                    participating_pct=extent, first_model_time_min=curve.first_model_time_min,
                    first_clock_time_min=curve.first_time_min,
                    effective_lead_vs_clock_min=curve.first_model_time_min - curve.first_time_min,
                    first_anchor_clipped=curve.first_anchor_clipped,
                    dose_mg=curve.dose_mg, qc_dv50_um=curve.qc_dv50_um, qc_d32_um=curve.qc_d32_um,
                ))
            by_unit.setdefault((curve.date, curve.condition), []).extend(residual.tolist())
        for (date, condition), residual in by_unit.items():
            r = np.asarray(residual, float)
            residual_rows.append(dict(
                cohort=cohort, model=model, date=date, condition=condition,
                n_obs=r.size, rmse_pct=float(np.sqrt(np.mean(r ** 2))),
                signed_mean_resid_pct=float(r.mean()),
            ))
    return rows, residual_rows


def condition_diagnostics(curves):
    rows = []
    for condition in sorted({c.condition for c in curves}):
        condition_curves = [c for c in curves if c.condition == condition]
        for model in ("anchored_rate", "anchored_rate_participation"):
            rate, extent, objective = fit_candidate(condition_curves, model)
            held_rmse = []
            for held_date in sorted({c.date for c in condition_curves}):
                train = [c for c in condition_curves if c.date != held_date]
                held = [c for c in condition_curves if c.date == held_date]
                fold_rate, fold_extent, _ = fit_candidate(train, model)
                held_rmse.append(score_candidate(held, model, fold_rate, fold_extent)["post_first_rmse_pct"])
            rows.append(dict(scope="condition", condition=condition, date="ALL", model=model,
                             n_dates=len({c.date for c in condition_curves}), n_runs=len(condition_curves),
                             rate_scale=rate, participating_pct=extent,
                             rate_grid_boundary=rate in (float(RS_GRID.min()), float(RS_GRID.max())),
                             extent_grid_boundary=extent in (float(PART_GRID.min()), float(PART_GRID.max())),
                             date_balanced_rmse_pct=float(np.sqrt(objective)),
                             lodo_rmse_pct=float(np.sqrt(np.mean(np.square(held_rmse))))))
        for date in sorted({c.date for c in condition_curves}):
            cell = [c for c in condition_curves if c.date == date]
            for model in ("anchored_rate", "anchored_rate_participation"):
                rate, extent, objective = fit_candidate(cell, model)
                rows.append(dict(scope="date_condition", condition=condition, date=date, model=model,
                                 n_dates=1, n_runs=len(cell), rate_scale=rate,
                                 participating_pct=extent,
                                 rate_grid_boundary=rate in (float(RS_GRID.min()), float(RS_GRID.max())),
                                 extent_grid_boundary=extent in (float(PART_GRID.min()), float(PART_GRID.max())),
                                 date_balanced_rmse_pct=float(np.sqrt(objective)), lodo_rmse_pct=np.nan))
    return rows


def mass_vs_optical(curves, empirical_csv: Path):
    empirical = pd.read_csv(empirical_csv)[["id", "condition", "date", "mean_relax_min", "beta"]]
    rows = []
    for curve in curves:
        rate, _, objective = fit_candidate([curve], "anchored_rate")
        rows.append(dict(id=curve.id, condition=curve.condition, date=curve.date,
                         rep=curve.rep, post_first_rate_scale=rate,
                         post_first_rmse_pct=float(np.sqrt(objective)),
                         rate_grid_boundary=rate in (float(RS_GRID.min()), float(RS_GRID.max()))))
    out = pd.DataFrame(rows).merge(empirical, on=["id", "condition", "date"], validate="one_to_one")
    out["rate_rank"] = out["post_first_rate_scale"].rank()
    out["optical_rank"] = out["mean_relax_min"].rank()
    correlation = float(out[["rate_rank", "optical_rank"]].corr().iloc[0, 1])
    out["spearman_all_runs"] = correlation
    return out


def date_cluster_bootstrap(curves, cohort, condition="ALL", *, n_boot=N_BOOT, seed=20260722):
    """Refit the anchored post-first rate after resampling independent preparation dates."""
    dates = sorted({c.date for c in curves})
    rng = np.random.default_rng(seed)
    rows = []
    for boot in range(n_boot):
        sampled = rng.choice(dates, size=len(dates), replace=True)
        draw = []
        for position, date in enumerate(sampled):
            draw.extend(replace(c, date=f"{date}#draw{position}") for c in curves if c.date == date)
        rate, _, objective = fit_candidate(draw, "anchored_rate")
        rows.append(dict(
            cohort=cohort, condition=condition, bootstrap=boot,
            rate_scale=rate, date_balanced_rmse_pct=float(np.sqrt(objective)),
            rate_grid_boundary=rate in (float(RS_GRID.min()), float(RS_GRID.max())),
        ))
    return rows


def _plot_model_comparison(summary, path):
    cohorts = list(summary["cohort"].unique())
    fig, axes = plt.subplots(1, len(cohorts), figsize=(13, 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    order = [m for m, _ in MODELS]
    labels = ["base", "old rate", "anchor", "anchor+rate", "anchor+extent", "anchor+rate+extent"]
    for ax, cohort in zip(axes, cohorts):
        sub = summary[summary["cohort"].eq(cohort)].set_index("model").loc[order]
        x = np.arange(len(order))
        ax.bar(x - 0.18, sub["post_first_rmse_pct"], width=0.36, color="#4C78A8", label="fit")
        ax.bar(x + 0.18, sub["lodo_rmse_pct"], width=0.36, color="#F58518", label="leave-date-out")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_title(cohort, fontsize=10)
        ax.grid(alpha=0.2, axis="y")
    axes[0].set_ylabel("Post-first-observation RMSE (percentage points)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Rate refinement: predictive comparison of timing, rate, and extent terms")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_profiles(profiles, path):
    cohorts = list(profiles["cohort"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#2166AC", "#B2182B"]
    for cohort, color in zip(cohorts, colors):
        rate = profiles[profiles["cohort"].eq(cohort) & profiles["model"].eq("anchored_rate")]
        axes[0].plot(rate["rate_scale"], rate["delta_mse"], color=color, label=cohort)
        full = profiles[profiles["cohort"].eq(cohort) & profiles["model"].eq("anchored_rate_participation")]
        prof = full.groupby("rate_scale", as_index=False)["date_balanced_mse"].min()
        prof["delta"] = prof["date_balanced_mse"] - prof["date_balanced_mse"].min()
        axes[1].plot(prof["rate_scale"], prof["delta"], color=color, label=cohort)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Post-first-observation rate scale")
        ax.set_ylabel("Δ date-balanced MSE")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_title("Rate only after first-UV anchor")
    axes[1].set_title("Rate profiled over participating extent")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_mass_optical(table, path):
    colors = {"0.01% Tween": "#2864A6", "0.03% Tween": "#C55A11"}
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for condition, g in table.groupby("condition"):
        ax.scatter(g["mean_relax_min"], g["post_first_rate_scale"], s=48,
                   color=colors.get(condition, "0.4"), label=condition, alpha=0.8)
    rho = table["spearman_all_runs"].iloc[0]
    ax.set_xlabel("Empirical optical mean relaxation time (min)")
    ax.set_ylabel("UV-mass post-first-observation rate scale")
    ax.set_title(f"Optical relaxation vs mass-space rate (Spearman ρ = {rho:.2f})")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_mass_trajectories(predictions, path):
    panels = [
        ("historical pH 4.0/4.5 fit cohort", "pH 4.0"),
        ("historical pH 4.0/4.5 fit cohort", "pH 4.5"),
        ("antisolvent Tween 80 concentration transfer", "0.01% Tween"),
        ("antisolvent Tween 80 concentration transfer", "0.03% Tween"),
    ]
    styles = {
        "injection_selected_rate": ("#B2182B", "--", "old rate 2.197"),
        "first_uv_anchor": ("0.45", ":", "first-UV anchor"),
        "anchored_rate": ("#2166AC", "-", "refined post-first rate"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, (cohort, condition) in zip(axes.flat, panels):
        sub = predictions[predictions["cohort"].eq(cohort) & predictions["condition"].eq(condition)]
        observed = (sub[sub["model"].eq("anchored_rate")]
                    .groupby("time_min", as_index=False)["observed_pct_injected"].mean())
        ax.plot(observed["time_min"], observed["observed_pct_injected"], "ko", ms=4, label="UV mass")
        for model, (color, linestyle, label) in styles.items():
            mean = (sub[sub["model"].eq(model)]
                    .groupby("time_min", as_index=False)["predicted_pct_injected"].mean())
            ax.plot(mean["time_min"], mean["predicted_pct_injected"], color=color,
                    ls=linestyle, lw=2, label=label)
        ax.set_title(condition)
        ax.set_xlabel("Time (min)")
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Dissolved CFZ (% injected mass)")
    axes[1, 0].set_ylabel("Dissolved CFZ (% injected mass)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Rate refinement: measured mass and candidate trajectories")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=None)
    args = parser.parse_args(argv)
    study_root = args.study_root or default_study_root()
    out = study_root / "analysis" / "rate_refinement"
    out.mkdir(parents=True, exist_ok=True)
    cohorts = build_curves(study_root)

    summary_rows, lodo_rows, profile_rows, prediction_rows, residual_rows = [], [], [], [], []
    for cohort, curves in cohorts.items():
        fits = []
        cohort_lodo = leave_one_date_out(curves, cohort)
        lodo_rows.extend(cohort_lodo)
        lodo = pd.DataFrame(cohort_lodo).groupby("model")["held_post_first_rmse_pct"].apply(
            lambda x: float(np.sqrt(np.mean(np.square(x)))))
        for model, n_params in MODELS:
            rate, extent, objective = fit_candidate(curves, model)
            score = score_candidate(curves, model, rate, extent)
            fits.append(dict(model=model, rate_scale=rate, participating_pct=extent))
            n_units = score["n_date_condition_units"]
            aicc = (n_units * np.log(max(objective, 1e-12)) + 2 * n_params
                    + (2 * n_params * (n_params + 1) / (n_units - n_params - 1)
                       if n_units > n_params + 1 else np.nan))
            summary_rows.append(dict(
                cohort=cohort, model=model, n_params=n_params, n_runs=len(curves),
                n_dates=len({c.date for c in curves}), n_date_condition_units=n_units,
                rate_scale=rate, participating_pct=extent,
                post_first_rmse_pct=score["post_first_rmse_pct"],
                all_obs_rmse_pct=score["all_obs_rmse_pct"],
                signed_resid_pct=score["post_first_signed_resid_pct"],
                lodo_rmse_pct=float(lodo[model]), cluster_aicc_diagnostic=aicc,
                rate_grid_boundary=rate in (float(RS_GRID.min()), float(RS_GRID.max())),
                extent_grid_boundary=extent in (float(PART_GRID.min()), float(PART_GRID.max())),
            ))
        pred, resid = fitted_predictions(curves, cohort, fits)
        prediction_rows.extend(pred); residual_rows.extend(resid)
        profile_rows.extend(candidate_profiles(curves, cohort))

    summary = pd.DataFrame(summary_rows)
    profiles = pd.DataFrame(profile_rows)
    antisolvent_key = "antisolvent Tween 80 concentration transfer"
    antisolvent = cohorts[antisolvent_key]
    diagnostic_rows = []
    for cohort, curves in cohorts.items():
        for row in condition_diagnostics(curves):
            diagnostic_rows.append({"cohort": cohort, **row})
    diagnostics = pd.DataFrame(diagnostic_rows)
    bootstrap_rows = []
    for index, (cohort, curves) in enumerate(cohorts.items()):
        bootstrap_rows.extend(date_cluster_bootstrap(curves, cohort, seed=20260722 + index))
        for offset, condition in enumerate(sorted({c.condition for c in curves}), start=1):
            subset = [c for c in curves if c.condition == condition]
            bootstrap_rows.extend(date_cluster_bootstrap(
                subset, cohort, condition=condition, seed=20260722 + 10 * index + offset))
    bootstrap = pd.DataFrame(bootstrap_rows)
    boot_summary = (bootstrap.groupby(["cohort", "condition"])["rate_scale"]
                    .quantile([0.025, 0.5, 0.975]).unstack().reset_index()
                    .rename(columns={0.025: "boot_ci2_5", 0.5: "boot_median", 0.975: "boot_ci97_5"}))
    cohort_boot = boot_summary[boot_summary["condition"].eq("ALL")].drop(columns="condition")
    summary = summary.merge(cohort_boot, on="cohort", how="left")
    summary.loc[~summary["model"].eq("anchored_rate"),
                ["boot_ci2_5", "boot_median", "boot_ci97_5"]] = np.nan
    condition_boot = boot_summary[~boot_summary["condition"].eq("ALL")]
    diagnostics = diagnostics.merge(condition_boot, on=["cohort", "condition"], how="left")
    not_rate = ~diagnostics["model"].eq("anchored_rate")
    diagnostics.loc[not_rate, ["boot_ci2_5", "boot_median", "boot_ci97_5"]] = np.nan
    optical = mass_vs_optical(
        antisolvent,
        study_root / "analysis/antisolvent_tween80_conc_empirical/angular_kww_fits.csv",
    )
    initialization = (pd.DataFrame(prediction_rows)
                      .query("model == 'first_uv_anchor'")
                      .sort_values(["cohort", "condition", "date", "rep", "time_min"])
                      .drop_duplicates(["cohort", "id"])
                      [["cohort", "id", "condition", "date", "rep", "first_clock_time_min",
                        "first_model_time_min", "effective_lead_vs_clock_min", "first_anchor_clipped",
                        "dose_mg", "qc_dv50_um", "qc_d32_um"]])
    files = {
        "candidate_models": out / "rate_refinement_candidate_models.csv",
        "predictions": out / "rate_refinement_predictions.csv",
        "residuals": out / "rate_refinement_residuals_by_date.csv",
        "profiles": out / "rate_refinement_parameter_profiles.csv",
        "lodo": out / "rate_refinement_lodo_validation.csv",
        "diagnostics": out / "rate_refinement_condition_diagnostics.csv",
        "mass_optical": out / "rate_refinement_mass_vs_optical_rate.csv",
        "initialization": out / "rate_refinement_initialization_diagnostics.csv",
        "bootstrap": out / "rate_refinement_date_cluster_bootstrap.csv",
        "comparison_figure": out / "rate_refinement_model_comparison.png",
        "profile_figure": out / "rate_refinement_parameter_profiles.png",
        "mass_optical_figure": out / "rate_refinement_mass_vs_optical_rate.png",
        "trajectory_figure": out / "rate_refinement_mass_trajectories.png",
    }
    summary.to_csv(files["candidate_models"], index=False)
    pd.DataFrame(prediction_rows).to_csv(files["predictions"], index=False)
    pd.DataFrame(residual_rows).to_csv(files["residuals"], index=False)
    profiles.to_csv(files["profiles"], index=False)
    pd.DataFrame(lodo_rows).to_csv(files["lodo"], index=False)
    diagnostics.to_csv(files["diagnostics"], index=False)
    optical.to_csv(files["mass_optical"], index=False)
    initialization.to_csv(files["initialization"], index=False)
    bootstrap.to_csv(files["bootstrap"], index=False)
    _plot_model_comparison(summary, files["comparison_figure"])
    _plot_profiles(profiles, files["profile_figure"])
    _plot_mass_optical(optical, files["mass_optical_figure"])
    _plot_mass_trajectories(pd.DataFrame(prediction_rows), files["trajectory_figure"])
    print(summary[["cohort", "model", "rate_scale", "participating_pct",
                   "post_first_rmse_pct", "lodo_rmse_pct"]].to_string(index=False))
    for path in files.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
