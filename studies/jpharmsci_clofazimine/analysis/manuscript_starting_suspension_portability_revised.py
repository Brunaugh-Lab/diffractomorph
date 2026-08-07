"""Manuscript figure (revised) — starting-suspension model portability (Section 3.4.1).

One scientific purpose: **the historical injection-started 2.197x rate correction did not transfer
to the changed starting-suspension cohort; after alignment at the first UV observation, the
subsequent trajectories required slower evolution.**

  A, B  Measured dissolved mass for each starting suspension (0.01% and 0.03% w/v polysorbate 80)
        with the model trajectories that establish the transfer result drawn on the same axes:
        the unmodified injection-started model, the historical 2.197x injection-started
        correction, and the first-UV-aligned slower-rate model selected in this cohort.
  C     Held-out prediction error for the four comparisons the narrative needs, grouped by where
        the prediction starts.

**Every plotted model trajectory is an out-of-fold prediction.** The published
``rate_refinement_predictions.csv`` holds all-data fits — one rate per model, fitted on every
preparation date including the one being drawn — which must not sit beside a leave-one-date-out
error bar. :mod:`arm_a_out_of_fold_predictions` re-applies the refinement module's own
``predict_curve`` using, for each preparation date, the parameters fitted with that date withheld,
and its output is what this figure reads. Three of the four models are parameter-free, so their
two bases coincide by construction; only the slower-rate model moves between folds. Each series
records its own ``prediction_basis``, and :func:`validate` re-checks it.

**The renderer never fits.** :func:`build_source_data` assembles every plotted number from the
artifacts in :data:`SOURCES`; :func:`render` consumes only that table.

Aggregation, everywhere: the three nested runs are averaged **within preparation date first**,
then the three preparation dates are weighted **equally**. Preparation date is the independent
unit; nested runs are technical replicates and never preparation-level replicates.

Claim boundaries enforced in wording and tests:

  * Leave-one-preparation-date-out is within-cohort cross-validation, not external validation.
  * The fitted post-capture rate is an empirical descriptor of this cohort, not a universal rate
    constant, and is not attributed to polysorbate concentration, wetting, surface area,
    deaggregation, or a participating fraction.
  * The two conditions did **not** differ in mean dissolved fraction; only between-date
    variability differed. Nothing in this figure may imply otherwise.
  * Mass domain only — no laser-diffraction, Copt, q3, or other optical quantity.

The extent candidates (extent-only, rate-plus-extent) are deliberately absent from the figure;
they remain in the rate_refinement artifacts and Table S8, and the caption states only that
allowing a separate terminal extent did not improve held-out prediction.

Run with the pipeline venv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from diffractomorph_pipeline.config import data_root

from arm_b_provenance import provenance_record, write_provenance

# ── authoritative sources ────────────────────────────────────────────────────────────────────
ARM_A = Path("disso_experiments/dissolution_media_diagnostic/tween80_suspension_wetting_arm_a")
SOURCES = {
    # Panels A/B observations: nested runs already averaged within preparation date upstream.
    "date_means": ARM_A / "analysis/antisolvent_tween80_conc_uv_date_means.csv",
    # Panels A/B trajectories: out-of-fold predictions, fold parameters from the LODO artifact.
    "out_of_fold": ARM_A / "analysis/rate_refinement/rate_refinement_out_of_fold_predictions.csv",
    # Panel C: the summary across held-out dates, and the per-fold values it summarises.
    "candidates": ARM_A / "analysis/rate_refinement/rate_refinement_candidate_models.csv",
    "lodo": ARM_A / "analysis/rate_refinement/rate_refinement_lodo_validation.csv",
}
COHORT = "antisolvent Tween 80 concentration transfer"   # the only cohort admitted

CONDITIONS = ["0.01", "0.03"]
# the out-of-fold artifact carries the refinement module's condition strings
ARTIFACT_CONDITION = {"0.01": "0.01% Tween", "0.03": "0.03% Tween"}
PANEL = {"0.01": "0.01% w/v polysorbate 80", "0.03": "0.03% w/v polysorbate 80"}

# model key -> (reader-facing series label, timing family). Order fixes Panel C rows.
INJECTION = "Prediction begins at injection"
ALIGNED = "Prediction begins at first UV measurement"
MODELS = [
    ("injection_base", "Base model", INJECTION),
    ("injection_selected_rate", "Historical 2.2× rate", INJECTION),
    ("first_uv_anchor", "Base rate", ALIGNED),
    ("anchored_rate", "Slower fitted rate", ALIGNED),
]
SELECTED_MODEL = "anchored_rate"
# Trajectories drawn in A/B. The first-UV-aligned base-rate model stays in Panel C only: it is
# the reference the slower rate is measured against, and drawing four curves obscured all of them.
SHOWN_IN_TRAJECTORY = ("injection_base", "injection_selected_rate", "anchored_rate")
TRAJECTORY_LABEL = {
    "injection_base": "Injection-started model",
    "injection_selected_rate": "Historical 2.2× correction",
    "anchored_rate": "First-UV aligned, slower rate",
}
EXPECTED_LODO = {"injection_base": 8.1, "injection_selected_rate": 13.7,
                 "first_uv_anchor": 7.8, "anchored_rate": 5.5}
EXCLUDED_FROM_FIGURE = ("anchored_participation", "anchored_rate_participation")
VALUE_COLUMNS = {
    "observations": "antisolvent_tween80_conc_uv_date_means.csv::pct_injected",
    "trajectories": "rate_refinement_out_of_fold_predictions.csv::predicted_pct_injected "
                    "(prediction_basis=out_of_fold, grid=dense)",
    "held_out_rmse": "rate_refinement_candidate_models.csv::lodo_rmse_pct",
    "held_out_per_fold": "rate_refinement_lodo_validation.csv::held_post_first_rmse_pct",
    "fold_parameters": "rate_refinement_lodo_validation.csv::fitted_rate_scale",
}
FORBIDDEN_TOKENS = ("career", "objective1", "nist", "ph_dependent", "ph 4.0", "ph 5.0",
                    "arm_b", "arm b", "copt", "q3", "angular", "historical ph 4.0/4.5 fit cohort")


def _read(key: str) -> pd.DataFrame:
    path = data_root() / SOURCES[key]
    if not path.exists():
        raise FileNotFoundError(
            f"authoritative source '{key}' missing: {path}. This figure does not substitute "
            f"defaults for absent artifacts — regenerate the analysis that produces it "
            f"(out_of_fold comes from arm_a_out_of_fold_predictions.py).")
    frame = pd.read_csv(path, dtype={"tween_pct_wv": str})
    if frame.empty:
        raise ValueError(f"authoritative source '{key}' is empty: {path}")
    return frame


def _equal_weight_over_dates(frame: pd.DataFrame, value: str, keys: list[str]) -> pd.DataFrame:
    """Average within preparation date, then weight the dates equally."""
    per_date = frame.groupby(keys + ["date"], as_index=False).agg(value=(value, "mean"))
    return per_date.groupby(keys, as_index=False).agg(
        value_pct=("value", "mean"), sd_pct=("value", "std"), n_dates=("date", "nunique"))


# ── panels A + B ─────────────────────────────────────────────────────────────────────────────

def _trajectory_panels() -> tuple[pd.DataFrame, dict]:
    date_means = _read("date_means")
    date_means["date"] = date_means["date"].astype(str)
    date_means = date_means[date_means["tween_pct_wv"].isin(CONDITIONS)].copy()

    oof = _read("out_of_fold")
    oof = oof[oof["cohort"].eq(COHORT)].copy()
    if oof.empty:
        raise ValueError(f"cohort {COHORT!r} absent from the out-of-fold artifact")
    oof["date"] = oof["date"].astype(str)
    first_uv = float(date_means["time_min"].min())
    # The injection-started models are EVALUATED from injection; they are DISPLAYED over the UV
    # observation window, where the measurements constrain them. Showing their rise from zero
    # would stretch the shared mass axis over a region no observation occupies and flatten the
    # part of the trajectory the comparison is about. Panel C's grouping states where each model
    # starts. The source table carries exactly what is drawn, so this filter applies here.
    curves = oof[oof["prediction_basis"].eq("out_of_fold") & oof["grid"].eq("dense")
                 & oof["time_min"].ge(first_uv)]
    if curves.empty:
        raise ValueError("the out-of-fold artifact carries no dense out-of-fold rows")

    rows = []
    for condition in CONDITIONS:
        panel = PANEL[condition]
        obs = date_means[date_means["tween_pct_wv"].eq(condition)]
        for r in obs.itertuples():
            rows.append({"panel": panel, "kind": "date_mean", "series": "preparation-date mean",
                         "condition": condition, "date": r.date, "time_min": float(r.time_min),
                         "value_pct": float(r.pct_injected), "n_runs": int(r.n_nested_runs),
                         "prediction_basis": "measured"})
        cohort_mean = _equal_weight_over_dates(
            obs.rename(columns={"tween_pct_wv": "condition"}), "pct_injected",
            ["condition", "time_min"])
        for r in cohort_mean.itertuples():
            rows.append({"panel": panel, "kind": "cohort_mean",
                         "series": "mean across preparation dates", "condition": condition,
                         "date": "all", "time_min": float(r.time_min),
                         "value_pct": float(r.value_pct), "sd_pct": float(r.sd_pct),
                         "n_dates": int(r.n_dates), "n_runs": 9,
                         "prediction_basis": "measured"})

        sub = curves[curves["condition"].eq(ARTIFACT_CONDITION[condition])]
        for model in SHOWN_IN_TRAJECTORY:
            model_rows = sub[sub["model"].eq(model)]
            if model_rows.empty:
                raise ValueError(f"no out-of-fold rows for {model!r} at {condition}%")
            curve = _equal_weight_over_dates(model_rows, "predicted_pct_injected",
                                             ["model", "time_min"])
            folds = sorted({round(float(v), 6) for v in model_rows["rate_scale"]})
            for r in curve.itertuples():
                rows.append({"panel": panel, "kind": "model_curve",
                             "series": TRAJECTORY_LABEL[model], "model_key": model,
                             "condition": condition, "date": "all", "time_min": float(r.time_min),
                             "value_pct": float(r.value_pct), "sd_pct": float(r.sd_pct),
                             "n_dates": int(r.n_dates),
                             "prediction_basis": "out_of_fold",
                             "fold_rate_scales": ";".join(f"{v:g}" for v in folds)})

    meta = {
        "conditions": CONDITIONS,
        "n_dates_per_condition": {c: int(date_means[date_means["tween_pct_wv"].eq(c)]
                                         ["date"].nunique()) for c in CONDITIONS},
        "nested_runs_per_date": sorted(int(v) for v in date_means["n_nested_runs"].unique()),
        "first_uv_time_min": first_uv,
        "time_min_range": [float(date_means["time_min"].min()),
                           float(date_means["time_min"].max())],
        "trajectories_shown": list(SHOWN_IN_TRAJECTORY),
        "trajectory_basis": "out_of_fold",
        "trajectory_display_window_min": [first_uv, float(date_means["time_min"].max())],
        "trajectory_evaluation_start": "injection for the injection-started models; the first UV "
                                       "observation for the aligned model",
        "fold_rate_scales": {m: sorted({round(float(v), 6) for v in
                                        curves[curves["model"].eq(m)]["rate_scale"]})
                             for m in SHOWN_IN_TRAJECTORY},
        "parameter_free_models": sorted(
            {str(m) for m in curves[curves["parameter_free"].astype(bool)]["model"].unique()}),
        "aggregation": "nested runs averaged within preparation date, then preparation dates "
                       "weighted equally",
    }
    return pd.DataFrame(rows), meta


# ── panel C ──────────────────────────────────────────────────────────────────────────────────

def _heldout_panel() -> tuple[pd.DataFrame, dict]:
    cand = _read("candidates")
    lodo = _read("lodo")
    if COHORT not in set(cand["cohort"]) or COHORT not in set(lodo["cohort"]):
        raise ValueError(f"cohort {COHORT!r} absent from a rate_refinement artifact")
    cand = cand[cand["cohort"].eq(COHORT)]
    lodo = lodo[lodo["cohort"].eq(COHORT)]

    rows = []
    for key, label, family in MODELS:
        summary = cand[cand["model"].eq(key)]
        if len(summary) != 1:
            raise ValueError(f"expected one candidate row for {key!r}, found {len(summary)}")
        folds = lodo[lodo["model"].eq(key)]["held_post_first_rmse_pct"].dropna()
        rows.append({"panel": "held-out error", "kind": "heldout_summary", "series": label,
                     "model_key": key, "timing_family": family,
                     "value_pct": float(summary["lodo_rmse_pct"].iloc[0]),
                     "n_dates": int(folds.size), "is_selected": key == SELECTED_MODEL,
                     "prediction_basis": "out_of_fold"})
    meta = {"cohort_admitted": COHORT,
            "n_held_out_dates": int(lodo["held_date"].nunique()),
            "held_out_dates": sorted(str(d) for d in lodo["held_date"].unique()),
            "residual_basis": "held_post_first_rmse_pct — the aligning first observation "
                              "initialises a withheld run's trajectory but is excluded from "
                              "its residuals",
            "summary_estimator": "root mean square across held-out preparation dates",
            "selected_model": SELECTED_MODEL,
            "models_excluded_from_figure": list(EXCLUDED_FROM_FIGURE),
            "excluded_documented_in": "Table S8"}
    return pd.DataFrame(rows), meta


def build_source_data() -> tuple[pd.DataFrame, dict]:
    """Every plotted observation and prediction, from the authoritative Arm A outputs only."""
    traj, traj_meta = _trajectory_panels()
    held, held_meta = _heldout_panel()
    table = pd.concat([traj, held], ignore_index=True)
    table["study"] = "tween80_suspension_wetting_arm_a"
    return table, {"trajectories": traj_meta, "held_out": held_meta}


# ── validation ───────────────────────────────────────────────────────────────────────────────

def _limits(table: pd.DataFrame) -> tuple[float, float]:
    """Common mass axis: covers every drawn observation and trajectory, padded, then rounded."""
    drawn = table[table["panel"].isin(PANEL.values())]["value_pct"]
    lo = float(np.floor((drawn.min() - 3.0) / 2.0) * 2.0)
    hi = float(np.ceil((drawn.max() + 3.0) / 2.0) * 2.0)
    return lo, hi


def validate(table: pd.DataFrame, meta: dict) -> dict:
    checks: dict = {}

    # Panel C: the four displayed values reproduce the artifact and are RMS over their folds.
    lodo = _read("lodo")
    lodo = lodo[lodo["cohort"].eq(COHORT)]
    summary = table[table["kind"].eq("heldout_summary")].set_index("model_key")
    for key, label, _ in MODELS:
        got = float(summary.loc[key, "value_pct"])
        folds = lodo[lodo["model"].eq(key)]["held_post_first_rmse_pct"].dropna()
        rms = float(np.sqrt((folds ** 2).mean()))
        checks[f"heldout_rmse_{key}"] = {
            "value": round(got, 3), "displayed": round(got, 1),
            "expected_displayed": EXPECTED_LODO[key],
            "match": round(got, 1) == EXPECTED_LODO[key],
            "rms_matches_summary": bool(np.isclose(got, rms)),
            "n_folds_used": int(folds.size)}
    bad = [k for k, v in checks.items() if not (v["match"] and v["rms_matches_summary"])]
    if bad:
        raise ValueError(f"held-out RMSE does not reproduce the authoritative outputs: {bad}")

    values = {k: float(summary.loc[k, "value_pct"]) for k, _, _ in MODELS}
    checks["narrative"] = {
        "historical_correction_worse_than_injection_base":
            values["injection_selected_rate"] > values["injection_base"],
        "selected_is_lowest_of_the_four":
            min(values, key=values.get) == SELECTED_MODEL,
        "alignment_alone_beats_injection_base":
            values["first_uv_anchor"] < values["injection_base"],
        "slower_rate_beats_aligned_base_rate":
            values["anchored_rate"] < values["first_uv_anchor"]}
    for name, ok in checks["narrative"].items():
        if not ok:
            raise ValueError(f"the figure's narrative claim failed against the artifacts: {name}")

    # Every plotted trajectory is out-of-fold, and its declared model matches its fold parameters.
    curves = table[table["kind"].eq("model_curve")]
    if not curves["prediction_basis"].eq("out_of_fold").all():
        raise ValueError("a plotted trajectory is not an out-of-fold prediction")
    oof = _read("out_of_fold")
    oof = oof[oof["cohort"].eq(COHORT) & oof["prediction_basis"].eq("out_of_fold")]
    declared = {}
    for model in SHOWN_IN_TRAJECTORY:
        artifact_rates = sorted({round(float(v), 6) for v in
                                 oof[oof["model"].eq(model)]["rate_scale"]})
        shown = sorted({round(float(v), 6) for v in
                        curves[curves["model_key"].eq(model)]["fold_rate_scales"]
                        .str.split(";").explode()})
        declared[model] = {"fold_rate_scales_artifact": artifact_rates,
                           "fold_rate_scales_series": shown,
                           "match": artifact_rates == shown,
                           "parameter_free": model in meta["trajectories"]["parameter_free_models"],
                           "n_distinct_fold_values": len(artifact_rates)}
        if not declared[model]["match"]:
            raise ValueError(f"plotted series for {model!r} does not carry its fold parameters")
    checks["trajectories_are_out_of_fold"] = {
        "basis": "out_of_fold", "per_model": declared,
        "note": "parameter-free models have one fold value, so their out-of-fold and all-data "
                "trajectories coincide by construction; only the slower-rate model moves"}
    checks["selected_model_varies_between_folds"] = {
        "model": SELECTED_MODEL,
        "fold_rate_scales": declared[SELECTED_MODEL]["fold_rate_scales_artifact"],
        "ok": len(declared[SELECTED_MODEL]["fold_rate_scales_artifact"]) > 1}

    # Panel A/B structure and date-first aggregation.
    dates = table[table["kind"].eq("date_mean")]
    per_condition = dates.groupby("condition")["date"].nunique().to_dict()
    checks["preparation_dates"] = {
        "n_dates_per_condition": {k: int(v) for k, v in per_condition.items()},
        "nested_runs_per_date": sorted(int(v) for v in dates["n_runs"].unique()),
        "ok": set(per_condition.values()) == {3} and set(per_condition) == set(CONDITIONS)}
    if not checks["preparation_dates"]["ok"]:
        raise ValueError(f"expected three preparation dates per condition, got {per_condition}")

    cohort = table[table["kind"].eq("cohort_mean")]
    diffs = []
    for r in cohort.itertuples():
        own = dates[dates["condition"].eq(r.condition) & dates["time_min"].eq(r.time_min)]
        if len(own) != 3:
            raise ValueError(f"cohort mean at {r.condition}/{r.time_min} does not summarise "
                             f"three preparation-date means (found {len(own)})")
        diffs.append(abs(float(own["value_pct"].mean()) - float(r.value_pct)))
    checks["date_first_aggregation"] = {
        "rule": "cohort mean is the unweighted mean of the three preparation-date means",
        "max_abs_deviation_pct": float(max(diffs)), "ok": bool(max(diffs) < 1e-9),
        "run_weighting_used": False}
    if not checks["date_first_aggregation"]["ok"]:
        raise ValueError("cohort means are not the equal-weight mean of preparation-date means")

    # The two conditions did not differ in mean dissolved fraction; only variability did.
    end_time = float(cohort["time_min"].max())
    end = cohort[cohort["time_min"].eq(end_time)].set_index("condition")["value_pct"]
    checks["condition_means_are_similar"] = {
        "terminal_time_min": end_time,
        "terminal_mean_pct": {c: round(float(end[c]), 2) for c in CONDITIONS},
        "abs_difference_pct": round(abs(float(end["0.01"]) - float(end["0.03"])), 2),
        "between_date_sd_pct": {c: round(float(cohort[cohort["condition"].eq(c)]
                                               ["sd_pct"].mean()), 2) for c in CONDITIONS},
        "claim": "condition means are similar; between-date variability is larger at 0.03%"}

    # Excluded candidates really are absent.
    checks["extent_models_absent"] = {
        "excluded": list(EXCLUDED_FROM_FIGURE),
        "present_model_keys": sorted(set(table["model_key"].dropna())),
        "ok": not set(EXCLUDED_FROM_FIGURE) & set(table["model_key"].dropna())}
    if not checks["extent_models_absent"]["ok"]:
        raise ValueError("an extent candidate reached the figure")

    # Scope.
    blob = " ".join(str(v) for v in table.to_dict("list").values()).lower()
    blob += " " + " ".join(str(p).lower() for p in SOURCES.values())
    hits = [t for t in FORBIDDEN_TOKENS if t in blob]
    checks["no_foreign_input"] = {
        "sources": [str(p) for p in SOURCES.values()],
        "all_sources_under_arm_a": all(str(p).startswith(str(ARM_A)) for p in SOURCES.values()),
        "forbidden_tokens_found": hits, "cohort_admitted": COHORT,
        "ok": not hits and all(str(p).startswith(str(ARM_A)) for p in SOURCES.values())}
    if not checks["no_foreign_input"]["ok"]:
        raise ValueError(f"a foreign (CAREER / pH-development / Arm B / optical) input "
                         f"reached the figure: {hits}")

    lo, hi = _limits(table)
    drawn = table[table["panel"].isin(PANEL.values())]["value_pct"]
    checks["axis_limits_do_not_clip"] = {
        "y_limits_pct": [lo, hi], "min_plotted_pct": round(float(drawn.min()), 2),
        "max_plotted_pct": round(float(drawn.max()), 2),
        "ok": bool(drawn.min() >= lo and drawn.max() <= hi)}
    if not checks["axis_limits_do_not_clip"]["ok"]:
        raise ValueError("the common mass axis clips a plotted value")
    return checks


# ── rendering ────────────────────────────────────────────────────────────────────────────────

import matplotlib                                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.lines import Line2D                                        # noqa: E402

STEM = "Figure_starting_suspension_model_portability_revised"
# Okabe-Ito. Model strategy carries colour and line style, held identical in both panels;
# measurement is always black (mean) or pale grey (individual preparation dates).
MODEL_STYLE = {
    "injection_base": {"color": "#9A9A9A", "ls": (0, (1, 1.6)), "lw": 1.3},
    "injection_selected_rate": {"color": "#D55E00", "ls": (0, (5, 2)), "lw": 1.6},
    "anchored_rate": {"color": "#0072B2", "ls": "-", "lw": 2.1},
}
ACCENT = "#0072B2"
DARK = "#1A1A1A"
PALE = "#BFBFBF"
GREY = "#8A8A8A"
FIG_W = 8.0


def _apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.2, "axes.labelsize": 8.4, "axes.titlesize": 8.6,
        "xtick.labelsize": 7.6, "ytick.labelsize": 7.6, "legend.fontsize": 6.8,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


def _trajectory_axis(ax, table, condition, ylim, first_uv, show_ylabel):
    sub = table[table["panel"].eq(PANEL[condition])]
    ax.axvline(first_uv, color=PALE, ls=":", lw=0.8, zorder=1)

    for _, g in sub[sub["kind"].eq("date_mean")].groupby("date"):
        g = g.sort_values("time_min")
        ax.plot(g["time_min"], g["value_pct"], color=PALE, lw=0.8, zorder=2)
    for model in SHOWN_IN_TRAJECTORY:
        curve = sub[sub["kind"].eq("model_curve")
                    & sub["model_key"].eq(model)].sort_values("time_min")
        style = MODEL_STYLE[model]
        ax.plot(curve["time_min"], curve["value_pct"], color=style["color"], ls=style["ls"],
                lw=style["lw"], zorder=3, solid_capstyle="round",
                label=TRAJECTORY_LABEL[model])
    mean = sub[sub["kind"].eq("cohort_mean")].sort_values("time_min")
    ax.plot(mean["time_min"], mean["value_pct"], "o", ms=3.8, color=DARK,
            markeredgecolor="white", markeredgewidth=0.4, zorder=5,
            label="UV measurement (mean of dates)")

    ax.set_title(PANEL[condition], fontsize=8.4, pad=3)
    ax.set_xlabel("Time (min)")
    if show_ylabel:
        ax.set_ylabel("Dissolved mass (% of injected dose)")
    ax.set_xlim(0.9, 20.7)
    ax.set_ylim(*ylim)
    ax.set_xticks([2, 5, 10, 15, 20])
    _clean(ax)


def _trajectory_legend():
    """Shared key for A and B. Placed below both panels: every in-panel position collides with
    either the measurements or a model curve at this figure size."""
    handles = [Line2D([], [], color=DARK, marker="o", ls="none", ms=3.8),
               Line2D([], [], color=PALE, lw=0.8)]
    labels = ["UV measurement (mean of dates)", "Individual preparation date"]
    for model in SHOWN_IN_TRAJECTORY:
        style = MODEL_STYLE[model]
        handles.append(Line2D([], [], color=style["color"], ls=style["ls"], lw=style["lw"]))
        labels.append(TRAJECTORY_LABEL[model])
    return handles, labels


# Panel C is a paired comparison, not a ranking. Each row is one starting point for the
# prediction and holds the two models that share it, so the group name is the row's axis label
# rather than free text floating in the plotting field. The arrow runs base -> modified, making
# the opposite signs of the two changes the first thing the panel shows.
PAIRS = [(INJECTION, "Prediction starts\nat injection", "injection_base",
          "injection_selected_rate"),
         (ALIGNED, "Prediction aligned\nto first UV", "first_uv_anchor", "anchored_rate")]
ROW_Y = {INJECTION: 1.15, ALIGNED: 0.0}
HISTORICAL = "#D55E00"
# Endpoint labels alternate above/below the row so two names never share a line: the base model
# sits below its marker, the modified model above its own.
ENDPOINT = {
    "injection_base": {"name": "Base model", "color": DARK, "side": "below", "size": 30},
    "injection_selected_rate": {"name": "Historical 2.2×", "color": HISTORICAL,
                                "side": "above", "size": 30},
    "first_uv_anchor": {"name": "Base rate", "color": DARK, "side": "below", "size": 30},
    "anchored_rate": {"name": "Slower fitted rate", "color": ACCENT, "side": "above", "size": 44},
}


def _heldout_axis(ax, table):
    summary = table[table["kind"].eq("heldout_summary")].set_index("model_key")["value_pct"]
    for family, _, base_key, changed_key in PAIRS:
        y = ROW_Y[family]
        base, changed = float(summary[base_key]), float(summary[changed_key])
        ax.annotate("", xy=(changed, y), xytext=(base, y),
                    # the aligned pair is only 2.3 points apart, so the shrink either side has to
                    # stay small or the arrow disappears between its own endpoints
                    arrowprops=dict(arrowstyle="-|>", color=ENDPOINT[changed_key]["color"],
                                    lw=1.0, shrinkA=3.5, shrinkB=4.5, mutation_scale=9),
                    zorder=3)
        for key, value in ((base_key, base), (changed_key, changed)):
            style = ENDPOINT[key]
            selected = key == SELECTED_MODEL
            ax.scatter([value], [y], s=style["size"], color=style["color"], zorder=4,
                       edgecolors="white", linewidths=0.5)
            above = style["side"] == "above"
            ax.text(value, y + (0.15 if above else -0.15),
                    f"{style['name']}\n{value:.1f}" if above
                    else f"{value:.1f}\n{style['name']}",
                    ha="center", va="bottom" if above else "top", fontsize=6.8,
                    color=style["color"], linespacing=1.25,
                    fontweight="bold" if selected else "normal", zorder=5)

    ax.set_yticks([ROW_Y[f] for f, _, _, _ in PAIRS])
    ax.set_yticklabels([label for _, label, _, _ in PAIRS], fontsize=7.2, linespacing=1.3)
    ax.set_title("Held-out prediction error", fontsize=8.4, pad=3)
    ax.set_xlabel("Leave-one-preparation-date-out RMSE\n(percentage points; lower is better)")
    ax.set_xlim(0, 16.8)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_ylim(-0.72, 1.95)
    ax.grid(axis="x", color="0.90", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    _clean(ax)


def _assert_legend_is_centred_under(fig, legend, boxA, boxB, boxC):
    """The A/B key must read as one legend belonging to both panels.

    Two ways that fails silently: the legend drifts off the pair's centre line, or it runs under
    panel C and looks like a figure-wide key. Both are geometry, so both are checked.
    """
    fig.canvas.draw()
    box = legend.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted())
    pair_centre = (boxA.x0 + boxB.x1) / 2
    offset = abs((box.x0 + box.x1) / 2 - pair_centre)
    if offset > 0.01:
        raise ValueError(f"the shared legend is off the A/B centre line by {offset:.3f} of the "
                         f"figure width; it must read as one key for both panels")
    if box.x1 > boxC.x0:
        raise ValueError("the shared legend runs under panel C and would read as a figure-wide "
                         "key rather than the A/B key")


def _assert_no_text_is_cropped(fig):
    """Fail loudly if any label falls outside the bounding box the figure will be saved with.

    ``bbox_inches="tight"`` does NOT simply grow to fit every artist. ``Axes.get_tightbbox``
    caps a centred axis label at roughly its own axes width, so a label wider than its panel is
    reported as narrower than it is and the saved page cuts it off — silently, with no warning
    and no exception. Panel C's two-line label is wider than a narrow panel, so the invariant
    that actually matters is checked here: every text artist must lie inside the tight bbox that
    savefig will use.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)          # inches
    lo, hi = tight.x0 * fig.dpi, tight.x1 * fig.dpi
    offenders = []
    for ax in fig.axes:
        artists = [ax.title, ax.xaxis.label, ax.yaxis.label,
                   *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts]
        for artist in artists:
            if not artist.get_text() or not artist.get_visible():
                continue
            box = artist.get_window_extent(renderer)
            if box.x0 < lo - 0.5 or box.x1 > hi + 0.5:
                offenders.append((artist.get_text().split("\n")[0],
                                  round(box.x0, 1), round(box.x1, 1)))
    if offenders:
        raise ValueError(
            f"text falls outside the tight bounding box ({lo:.0f}..{hi:.0f} px) and would be "
            f"cropped in the saved figure — widen its panel or shorten the label: {offenders}")


def render(table: pd.DataFrame, meta: dict, out_dir: Path, formats=("pdf", "png", "svg")):
    _apply_style()
    ylim = _limits(table)
    first_uv = meta["trajectories"]["first_uv_time_min"]
    fig = plt.figure(figsize=(FIG_W, 3.8))
    # One gridspec, so the gap between A and B equals the gap between B and C: matplotlib applies
    # `wspace` uniformly between columns even when their widths differ. The page is sized around
    # those gaps rather than the gaps being squeezed to fit a page. Panel C is the widest column
    # because its two-line axis label is wider than a trajectory panel, and a label wider than its
    # own axes is silently cropped by the tight bounding box (see _assert_no_text_is_cropped).
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.4],
                          left=0.0775, right=0.990, top=0.915, bottom=0.415,
                          wspace=0.566)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1], sharey=axA)
    axC = fig.add_subplot(gs[0, 2])

    _trajectory_axis(axA, table, "0.01", ylim, first_uv, True)
    _trajectory_axis(axB, table, "0.03", ylim, first_uv, False)
    plt.setp(axB.get_yticklabels(), visible=False)
    _heldout_axis(axC, table[table["panel"].eq("held-out error")])

    # One legend for A and B, centred on the pair — not stretched across them, which reads as two
    # separate per-panel legends, and not centred on the figure, which would drift under C.
    handles, labels = _trajectory_legend()
    boxA, boxB, boxC = axA.get_position(), axB.get_position(), axC.get_position()
    legend = fig.legend(handles, labels, loc="upper center",
                        bbox_to_anchor=((boxA.x0 + boxB.x1) / 2, 0.275), ncol=2, frameon=False,
                        fontsize=6.5, handlelength=2.3, borderpad=0.1, labelspacing=0.34,
                        handletextpad=0.5, columnspacing=1.4)
    _assert_legend_is_centred_under(fig, legend, boxA, boxB, boxC)

    # Offsets differ because the panels differ: A and C carry axis labels to their left, B does
    # not (it shares A's y axis), so a shared offset would push B's tag into A's plotting area.
    for ax, letter, pad_in in ((axA, "A", 0.50), (axB, "B", 0.13), (axC, "C", 0.50)):
        box = ax.get_position()
        fig.text(box.x0 - pad_in / FIG_W, box.y1 + 0.035, letter, fontsize=10.0,
                 fontweight="bold", va="bottom", ha="left")
    _assert_no_text_is_cropped(fig)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_dir / f"{STEM}.{fmt}"
        fig.savefig(path, format=fmt, dpi=600 if fmt == "png" else None, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


CAPTION = """# {stem} — caption draft

**Changing the starting-suspension preparation reversed the direction of the empirical correction
required by the forward model.** The historical 2.2× injection-started correction overpredicted
dissolution, whereas alignment at the first UV measurement followed by slower post-capture
evolution gave the lowest held-out error. Clofazimine was dispersed at pH 4.5 from antisolvent
suspensions containing 0.01% or 0.03% w/v polysorbate 80, on {n_dates} independent preparation
dates per condition with three nested runs per date. Nested runs were averaged within preparation
date before any cross-date summary and preparation dates were then weighted equally; preparation
date is the independent unit and nested runs are technical replicates, not preparation-level
replicates.

**(A, B)** Dissolved clofazimine measured by UV–visible spectroscopy as a percentage of the
assayed injected dose. Pale grey lines are the three individual preparation-date means and black
points their equal-weight mean; the dotted vertical guide marks the first UV observation at
{first_uv:.0f} min, where the aligned model is initialised. Overlaid are the size-resolved forward
model started at injection (grey dotted), the same model with the historical {historical}×
injection-started rate correction (orange dashed), and the model aligned at each run's first UV
observation with the slower fitted post-capture rate selected in this cohort (blue). **All three
trajectories are out-of-fold predictions**: each preparation date is drawn using the parameters
fitted with that date withheld, so the curves and the errors in (C) rest on the same evidence.
The two starting suspensions reached a similar cohort-average dissolved fraction ({m1:.0f}% and
{m2:.0f}% of injected dose at {t_end:.0f} min); what differed was between-date variability, which
was larger at 0.03% (mean SD across dates {sd2:.1f} versus {sd1:.1f} percentage points).

**(C)** Leave-one-preparation-date-out RMSE, pooled across both conditions, for the four
comparisons that establish the result; lower is better. Each withheld run's first UV observation
is excluded from its residuals in every model and initialises the trajectory in the aligned
models, and each value is the root mean square across the held-out preparation dates. Starting
from injection, the historical {historical}× correction raised held-out error from {base:.1f} to
{hist:.1f} percentage points — it did not transfer to this cohort. Starting instead from the
first UV measurement gave {aligned:.1f} percentage points with no rate correction, and
{sel:.1f} once the post-capture rate was allowed to be slower ({rate_lo:.2f}–{rate_hi:.2f}×
across folds, highlighted). Allowing a separate terminal extent did not improve held-out
prediction (Table S8).

This is within-cohort leave-one-preparation-date-out evaluation, not external validation. The
fitted post-capture rate is an empirical descriptor of this cohort rather than a universal rate
constant, and is not attributed here to polysorbate concentration, wetting, surface area,
deaggregation, or a participating particle fraction. No laser-diffraction or other optical
measurement entered this figure.
"""


def _emit(out: Path, formats) -> int:
    table, meta = build_source_data()
    checks = validate(table, meta)
    written = render(table, meta, out, formats)

    csv_path = out / f"{STEM}_source_data.csv"
    table.to_csv(csv_path, index=False)

    summary = table[table["kind"].eq("heldout_summary")].set_index("model_key")["value_pct"]
    cohort = table[table["kind"].eq("cohort_mean")]
    t_end = float(cohort["time_min"].max())
    end = cohort[cohort["time_min"].eq(t_end)].set_index("condition")["value_pct"]
    sd = {c: checks["condition_means_are_similar"]["between_date_sd_pct"][c] for c in CONDITIONS}
    folds = checks["selected_model_varies_between_folds"]["fold_rate_scales"]
    caption = CAPTION.format(
        stem=STEM, n_dates=meta["trajectories"]["n_dates_per_condition"]["0.01"],
        first_uv=meta["trajectories"]["first_uv_time_min"], historical="2.2",
        m1=float(end["0.01"]), m2=float(end["0.03"]), t_end=t_end,
        sd1=sd["0.01"], sd2=sd["0.03"],
        base=summary["injection_base"], hist=summary["injection_selected_rate"],
        aligned=summary["first_uv_anchor"], sel=summary[SELECTED_MODEL],
        rate_lo=min(folds), rate_hi=max(folds))
    caption_path = out / f"{STEM}_caption.md"
    caption_path.write_text(caption)

    prov = provenance_record(
        "manuscript_starting_suspension_portability_revised",
        study_root=str(data_root() / ARM_A), uv_ph_values=(4.5,), figure_stem=STEM,
        supersedes="Figure_starting_suspension_model_portability (kept; not overwritten)",
        sources={k: str(data_root() / v) for k, v in SOURCES.items()},
        source_paths_relative={k: str(v) for k, v in SOURCES.items()},
        cohort_admitted=COHORT,
        prediction_basis={
            "trajectories": "out_of_fold — each preparation date drawn with the parameters "
                            "fitted while that date was withheld",
            "generator": "arm_a_out_of_fold_predictions.py",
            "fold_parameter_source": "rate_refinement_lodo_validation.csv::fitted_rate_scale",
            "all_data_artifact_not_plotted":
                "rate_refinement_predictions.csv holds all-data fits and is NOT drawn",
            "parameter_free_models": meta["trajectories"]["parameter_free_models"],
            "fold_rate_scales": meta["trajectories"]["fold_rate_scales"]},
        aggregation_hierarchy={
            "level_1": "three nested technical runs averaged within preparation date",
            "level_2": "three preparation dates weighted equally",
            "independent_unit": "preparation date",
            "nested_runs_are_not_independent_replicates": True,
            "applies_to": "observations and model trajectories alike"},
        model_value_columns=VALUE_COLUMNS,
        models={key: {"label": label, "timing_family": family,
                      "heldout_rmse_pct": round(float(summary[key]), 6),
                      "displayed": EXPECTED_LODO[key],
                      "drawn_in_trajectory_panels": key in SHOWN_IN_TRAJECTORY,
                      "selected": key == SELECTED_MODEL}
                for key, label, family in MODELS},
        design={"panels": ["A — 0.01% w/v polysorbate 80", "B — 0.03% w/v polysorbate 80",
                           "C — held-out prediction error"],
                "n_panels": 3, "y_limits_pct": list(_limits(table)),
                "trajectories_shown": list(SHOWN_IN_TRAJECTORY),
                "models_excluded_from_figure": list(EXCLUDED_FROM_FIGURE),
                "excluded_documented_in": "Table S8"},
        counts={"conditions": len(CONDITIONS),
                "preparation_dates_per_condition":
                    meta["trajectories"]["n_dates_per_condition"],
                "nested_runs_per_date": meta["trajectories"]["nested_runs_per_date"],
                "held_out_dates": meta["held_out"]["n_held_out_dates"],
                "models_in_panel_c": len(MODELS)},
        numerical_checks=checks, career_artifacts_used=False,
        scope={"experiment": "replicated starting-suspension polysorbate 80 study (Arm A) only",
               "domain": "dissolved mass (UV-visible) only",
               "excluded": ["NSF CAREER feasibility artifacts", "pH-development study and cohort",
                            "Arm B", "laser-diffraction intensity", "Copt", "q3",
                            "optical relaxation"],
               "claim_boundaries": [
                   "within-cohort leave-one-preparation-date-out evaluation, not external "
                   "validation",
                   "the fitted post-capture rate is an empirical descriptor of this cohort, "
                   "not a universal rate constant",
                   "the slower rate is not attributed to polysorbate concentration, wetting, "
                   "surface area, deaggregation, or participating fraction",
                   "the two conditions did NOT differ in mean dissolved fraction; only "
                   "between-date variability differed"]})
    prov_path = write_provenance(out / f"{STEM}_provenance.json", prov)

    print(f"cohort: {COHORT}")
    print(f"trajectory basis: {meta['trajectories']['trajectory_basis']} "
          f"(fold rates for {SELECTED_MODEL}: {folds})")
    print("panel C — held-out RMSE (percentage points):")
    for key, label, family in MODELS:
        mark = "  <- selected" if key == SELECTED_MODEL else ""
        start = "injection" if family == INJECTION else "first UV"
        print(f"  {summary[key]:6.3f} -> {EXPECTED_LODO[key]:4.1f}   "
              f"[from {start:9s}] {label}{mark}")
    print(f"terminal means: 0.01% {float(end['0.01']):.1f}%, 0.03% {float(end['0.03']):.1f}% "
          f"(|diff| {checks['condition_means_are_similar']['abs_difference_pct']:.1f} pp)")
    print(f"between-date SD: 0.01% {sd['0.01']:.2f}, 0.03% {sd['0.03']:.2f} pp")
    print(f"y limits {_limits(table)} — clipping: "
          f"{not checks['axis_limits_do_not_clip']['ok']}")
    for path in [*written, csv_path, caption_path, prov_path]:
        print(f"wrote {path}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--formats", default="pdf,png,svg")
    args = p.parse_args(argv)
    return _emit(args.output_dir,
                 tuple(f.strip() for f in args.formats.split(",") if f.strip()))


if __name__ == "__main__":
    raise SystemExit(main())
