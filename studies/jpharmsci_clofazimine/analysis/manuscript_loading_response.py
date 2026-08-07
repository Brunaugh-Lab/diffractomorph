"""Manuscript figure — starting particle loading (Section 3.4.2).

One scientific purpose: **within one pH 4.5 preparation, raising the delivered dose raised the
starting optical concentration, the finite-sink model with every parameter frozen predicted that a
smaller fraction of the dose would dissolve, and the UV measurements followed that ordering under
the established additive filter-recovery correction — conditionally.**

  A  The loading ladder was realized: delivered dose against starting Copt, pH 4.5 only.
  B  Frozen finite-sink prediction against UV terminal recovery, on delivered dose.
  C  Why the UV agreement is conditional: the low-load minus high-load recovery contrast against
     the additive filter offset, with its zero crossing and the calibrated value.

**Conditional, not validation.** This is an independent-dataset, within-preparation
loading-response evaluation. All three loadings are aliquots of ONE pH 4.5 suspension and the
three replicates per level are TECHNICAL, so there is no preparation-level error term. The
additive filter correction contributes a different number of recovery points at each loading
(~49, 37 and 30 pp at 133.0, 177.4 and 221.7 ug), and the 12 % -> 24 % ordering reverses below
0.59 ug/mL — ~40 % of the calibrated 1.48 ug/mL, for which the calibration artifact carries no
uncertainty interval. Panel C is that dependence, drawn as a single contrast so the reader sees
the sign change rather than having to difference three curves by eye.

**Nothing is refitted and nothing optical enters.** The prediction uses each run's delivered dose,
the shared suspension QC PSD (not the per-run in-cuvette q0), pH 4.5 solubility, and the frozen
historical rate scale 2.197 from ``selected_rate_only_fit_summary.csv``
(``rate_scale_datebalanced``). Copt appears in panel A only as the realized loading coordinate;
fractional Copt loss, q3 and the per-run q0 sensitivity belong to the Supporting Information
figure or stay in the analysis outputs, and are absent here.

:func:`build_source_data` assembles every plotted number from the :mod:`copt_loading` artifacts
listed in :data:`manuscript_loading_common.SOURCES`; :func:`render` consumes only that table.

Run with the pipeline venv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diffractomorph_pipeline.config import data_root

import manuscript_loading_common as mlc
from arm_b_provenance import provenance_record, write_provenance

STEM = "Figure_loading_response_model_transfer"

# Displayed (one-decimal) values the figure is required to reproduce. These are assertions about
# the authoritative artifacts, not inputs to the drawing.
EXPECTED_DOSE_UG = {12: 133.0, 18: 177.4, 24: 221.7}
EXPECTED_COPT0_PCT = {12: 9.9, 18: 12.0, 24: 14.2}
EXPECTED_MODEL_PCT = {12: 96.7, 18: 94.8, 24: 91.5}
EXPECTED_UV_PCT = {12: (99.0, 7.7), 18: (92.9, 2.7), 24: (87.2, 6.1)}

MODEL_SERIES = "Frozen model"
UV_SERIES = "UV measurement"
CONTRAST_SERIES = "Low-load minus high-load UV recovery"
LOW_LEVEL, HIGH_LEVEL = 12, 24          # the two ends of the ladder the contrast is taken between


# ── panel A — the pH 4.5 loading ladder was realized ─────────────────────────────────────────

def _panel_a() -> tuple[list[dict], dict]:
    runs = mlc.read_runs()
    summary = mlc.level_summary(runs, "copt0")
    rows = []
    for r in runs.itertuples():
        rows.append({"panel": "A", "kind": "run_point", "series": "Individual technical run",
                     "substudy": mlc.PH45, "level_pct": int(r.level_pct), "rep": int(r.rep),
                     "x_quantity": "delivered_dose_ug", "x_value": mlc.dose_ug(r.dose_mg),
                     "y_quantity": "starting_copt_pct", "y_value": float(r.copt0),
                     "source_artifact": "copt_loading_runs.csv", "source_column": "copt0"})
    for r in summary.itertuples():
        rows.append({"panel": "A", "kind": "level_mean",
                     "series": "Mean ± technical SD", "substudy": mlc.PH45,
                     "level_pct": int(r.level_pct), "x_quantity": "delivered_dose_ug",
                     "x_value": float(r.dose_ug), "y_quantity": "starting_copt_pct",
                     "y_value": float(r.mean), "y_sd": float(r.sd),
                     "n_technical_reps": int(r.n_technical_reps),
                     "source_artifact": "copt_loading_runs.csv", "source_column": "copt0"})

    # The tabulated within-preparation regression is recorded but NOT drawn: over three loading
    # levels it lies under the level means, so a fitted line would add a redundant mark whose only
    # effect is to invite an R^2 reading the design does not support.
    fit = mlc.read_scoped("linearity")
    if len(fit) != 1:
        raise ValueError(f"expected one in-scope linearity row, found {len(fit)}")

    meta = {"substudy": mlc.PH45,
            "levels_pct": [int(v) for v in summary["level_pct"]],
            "dose_ug": {int(r.level_pct): round(float(r.dose_ug), 4)
                        for r in summary.itertuples()},
            "copt0_mean_pct": {int(r.level_pct): round(float(r.mean), 4)
                               for r in summary.itertuples()},
            "copt0_technical_sd_pct": {int(r.level_pct): round(float(r.sd), 4)
                                       for r in summary.itertuples()},
            "tabulated_linearity_not_drawn": {
                "slope_copt_per_mg": float(fit["slope_copt_per_mg"].iloc[0]),
                "intercept_copt": float(fit["intercept_copt"].iloc[0]),
                "r2": float(fit["r2"].iloc[0]),
                "source": "copt_loading_linearity.csv",
                "reason": "recorded for provenance; not drawn and not displayed, so no R^2 "
                          "reading is invited from three within-preparation loading levels"},
            "message": "increasing delivered dose produced increasing starting optical "
                       "concentration WITHIN this single preparation"}
    return rows, meta


# ── panel B — frozen prediction vs UV recovery ───────────────────────────────────────────────

def _panel_b() -> tuple[list[dict], dict]:
    runs = mlc.read_runs()
    uv = mlc.level_summary(runs, "uv_pct_injected_end")
    model = mlc.level_summary(runs, "model_pct_end")

    rows = []
    for r in runs.itertuples():
        rows.append({"panel": "B", "kind": "uv_run", "series": f"{UV_SERIES} (technical run)",
                     "substudy": mlc.PH45, "level_pct": int(r.level_pct), "rep": int(r.rep),
                     "x_quantity": "delivered_dose_ug", "x_value": mlc.dose_ug(r.dose_mg),
                     "y_quantity": "terminal_dissolved_pct_of_dose",
                     "y_value": float(r.uv_pct_injected_end),
                     "source_artifact": "copt_loading_runs.csv",
                     "source_column": "uv_pct_injected_end"})
    for r in uv.itertuples():
        rows.append({"panel": "B", "kind": "uv_level_mean",
                     "series": f"{UV_SERIES} (mean ± technical SD)", "substudy": mlc.PH45,
                     "level_pct": int(r.level_pct), "x_quantity": "delivered_dose_ug",
                     "x_value": float(r.dose_ug),
                     "y_quantity": "terminal_dissolved_pct_of_dose",
                     "y_value": float(r.mean), "y_sd": float(r.sd),
                     "n_technical_reps": int(r.n_technical_reps),
                     "source_artifact": "copt_loading_runs.csv",
                     "source_column": "uv_pct_injected_end"})
    for r in model.itertuples():
        # The three technical runs share a delivered dose and the shared QC PSD, so their frozen
        # predictions differ only in the last decimals of the ODE grid; no band is drawn because
        # no model uncertainty exists to draw.
        rows.append({"panel": "B", "kind": "model_level",
                     "series": f"{MODEL_SERIES} (no parameter refitted)", "substudy": mlc.PH45,
                     "level_pct": int(r.level_pct), "x_quantity": "delivered_dose_ug",
                     "x_value": float(r.dose_ug),
                     "y_quantity": "terminal_dissolved_pct_of_dose",
                     "y_value": float(r.mean), "n_technical_reps": int(r.n_technical_reps),
                     "source_artifact": "copt_loading_runs.csv",
                     "source_column": "model_pct_end"})

    meta = {"substudy": mlc.PH45,
            "model_pct": {int(r.level_pct): round(float(r.mean), 4) for r in model.itertuples()},
            "model_spread_within_level_pct": {int(r.level_pct): round(float(r.sd), 6)
                                              for r in model.itertuples()},
            "uv_pct": {int(r.level_pct): round(float(r.mean), 4) for r in uv.itertuples()},
            "uv_technical_sd_pct": {int(r.level_pct): round(float(r.sd), 4)
                                    for r in uv.itertuples()},
            "model_uncertainty_band_drawn": False,
            "model_uncertainty_band_reason": "no model uncertainty exists — every parameter is "
                                             "frozen and the only within-level variation is ODE "
                                             "grid resolution",
            "prediction_inputs": {
                "dose": "each run's own delivered dose",
                "psd": "shared suspension QC PSD (NOT the per-run in-cuvette q0)",
                "solubility": "pH 4.5 Cs",
                "rate_scale": mlc.FROZEN_RATE_SCALE},
            "agreement_wording": "conditional agreement under the established additive "
                                 "filter-recovery correction"}
    return rows, meta


# ── panel C — the loading contrast against the filter offset ─────────────────────────────────

def _panel_c() -> tuple[list[dict], dict]:
    """Contrast = tabulated recovery at the LOWEST loading minus that at the HIGHEST.

    Positive means the recovered fraction decreases with loading; zero means no loading trend
    between the endpoints; negative means it increases with loading. Both recoveries are cells of
    ``copt_loading_filter_offset_sensitivity.csv``; the only operation is their difference.
    """
    offsets = mlc.read("offsets")
    wide = offsets.pivot(index="offset_ugml", columns="level_pct", values="recovery_pct")
    for level in (LOW_LEVEL, HIGH_LEVEL):
        if level not in wide.columns:
            raise ValueError(f"loading level {level} absent from the offset-sensitivity artifact")
    contrast = (wide[LOW_LEVEL] - wide[HIGH_LEVEL]).sort_index()

    rows = []
    for offset, value in contrast.items():
        rows.append({"panel": "C", "kind": "offset_contrast", "series": CONTRAST_SERIES,
                     "substudy": mlc.PH45, "x_quantity": "additive_filter_offset_ugml",
                     "x_value": float(offset),
                     "y_quantity": "low_minus_high_load_recovery_pp",
                     "y_value": float(value),
                     "source_artifact": "copt_loading_filter_offset_sensitivity.csv",
                     "source_column": f"recovery_pct[{LOW_LEVEL}] - recovery_pct[{HIGH_LEVEL}]"})

    rec = mlc.offset_record()
    for name, value, artifact in (
            ("Crossover offset", rec["crossover_offset_ugml"], "provenance.json"),
            ("Calibrated offset", rec["calibrated_offset_ugml"], "provenance.json")):
        rows.append({"panel": "C", "kind": "reference_offset", "series": name,
                     "substudy": mlc.PH45, "x_quantity": "additive_filter_offset_ugml",
                     "x_value": float(value), "y_quantity": "reference_line",
                     "source_artifact": artifact,
                     "source_column": "filter_offset_sensitivity"})

    x = contrast.index.to_numpy(float)
    y = contrast.to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    residual = float(np.max(np.abs(y - (slope * x + intercept))))
    meta = dict(rec)
    meta.update({
        "contrast_definition": f"recovery at the {LOW_LEVEL} % level minus recovery at the "
                               f"{HIGH_LEVEL} % level, at the same additive offset",
        "sign_convention": {"positive": "recovered fraction DECREASES with loading",
                            "zero": "no loading trend between the endpoints",
                            "negative": "recovered fraction INCREASES with loading"},
        "offset_grid_ugml": [float(v) for v in x],
        "contrast_pp": {f"{v:g}": round(float(c), 4) for v, c in zip(x, y)},
        "contrast_at_zero_offset_pp": round(float(contrast.iloc[0]), 4),
        "contrast_at_calibrated_pp": round(
            float(contrast.loc[min(contrast.index,
                                   key=lambda v: abs(v - rec["calibrated_offset_ugml"]))]), 4),
        "linear_in_offset": {"slope_pp_per_ugml": float(slope), "max_abs_residual_pp": residual,
                             "reason": "the correction is additive, so each level's recovery is "
                                       "exactly linear in the offset"},
        "uncertainty_band_drawn": False,
        "uncertainty_band_reason": "the calibration artifact provides no offset uncertainty "
                                   "interval; a band would imply the calibrated value has been "
                                   "shown to lie safely above the crossover",
    })
    return rows, meta


def build_source_data() -> tuple[pd.DataFrame, dict]:
    """Every plotted value, assembled from the authoritative copt_loading artifacts only."""
    runs = mlc.read_runs()
    rows_a, meta_a = _panel_a()
    rows_b, meta_b = _panel_b()
    rows_c, meta_c = _panel_c()
    table = pd.DataFrame(rows_a + rows_b + rows_c)
    table["study"] = "conc_dependent_disso_study"
    meta = {"panel_a": meta_a, "panel_b": meta_b, "panel_c": meta_c,
            "design": mlc.design_counts(runs),
            "frozen_model": mlc.frozen_model_record(runs),
            "aggregations": mlc.AGGREGATIONS,
            "claim_boundaries": mlc.CLAIM_BOUNDARIES}
    return table, meta


# ── validation ───────────────────────────────────────────────────────────────────────────────

def validate(table: pd.DataFrame, meta: dict) -> dict:
    checks: dict = {}

    # 1 ── design: one preparation, three technical replicates, one sub-study.
    design = meta["design"]
    checks["design"] = {
        "n_preparations": design["n_preparations"],
        "preparation": design["preparation"],
        "technical_reps_per_level": design["technical_reps_per_level"],
        "replicates_are_technical": True,
        "loading_levels_are_aliquots_of_one_suspension": True,
        "substudies_in_figure": sorted(set(table["substudy"])),
        "ok": (design["n_preparations"] == 1 and design["technical_reps_per_level"] == [3]
               and design["preparation"] == [mlc.MANUSCRIPT_PREP]
               and set(table["substudy"]) == {mlc.MANUSCRIPT_SUBSTUDY})}
    if not checks["design"]["ok"]:
        raise ValueError(f"the main figure must show one {mlc.MANUSCRIPT_SUBSTUDY} preparation "
                         f"with three technical replicates per level: {checks['design']}")

    # 2 ── the frozen model really is frozen.
    frozen = meta["frozen_model"]
    checks["frozen_model"] = {
        "rate_scale": frozen["rate_scale"], "source_file": frozen["source_file"],
        "source_column": frozen["source_column"], "refitted_here": frozen["refitted_here"],
        "psd_primary_is_shared_qc": "shared suspension QC" in frozen["psd_primary"],
        "per_run_q0_plotted": bool(table["source_column"].astype(str)
                                   .str.contains("per_run_q0").any()),
        "ok": (frozen["rate_scale"] == mlc.FROZEN_RATE_SCALE
               and frozen["source_file"] == mlc.FROZEN_RATE_SOURCE
               and frozen["source_column"] == mlc.FROZEN_RATE_COLUMN
               and frozen["refitted_here"] is False
               and not table["source_column"].astype(str).str.contains("per_run_q0").any())}
    if not checks["frozen_model"]["ok"]:
        raise ValueError(f"frozen-model provenance failed: {checks['frozen_model']}")

    # 3 ── displayed values reproduce the artifacts at one decimal place.
    displayed = {}
    for level in mlc.LEVELS:
        a = table[table["panel"].eq("A") & table["kind"].eq("level_mean")
                  & table["level_pct"].eq(level)]
        b_uv = table[table["panel"].eq("B") & table["kind"].eq("uv_level_mean")
                     & table["level_pct"].eq(level)]
        b_mo = table[table["panel"].eq("B") & table["kind"].eq("model_level")
                     & table["level_pct"].eq(level)]
        uv_mean, uv_sd = EXPECTED_UV_PCT[level]
        displayed[level] = {
            "dose_ug": round(float(a["x_value"].iloc[0]), 1),
            "dose_ug_expected": EXPECTED_DOSE_UG[level],
            "copt0_pct": round(float(a["y_value"].iloc[0]), 1),
            "copt0_pct_expected": EXPECTED_COPT0_PCT[level],
            "model_pct": round(float(b_mo["y_value"].iloc[0]), 1),
            "model_pct_expected": EXPECTED_MODEL_PCT[level],
            "uv_pct": round(float(b_uv["y_value"].iloc[0]), 1), "uv_pct_expected": uv_mean,
            "uv_sd_pp": round(float(b_uv["y_sd"].iloc[0]), 1), "uv_sd_pp_expected": uv_sd}
        displayed[level]["ok"] = all(
            displayed[level][k] == displayed[level][f"{k}_expected"]
            for k in ("dose_ug", "copt0_pct", "model_pct", "uv_pct")) and \
            displayed[level]["uv_sd_pp"] == displayed[level]["uv_sd_pp_expected"]
    checks["displayed_values"] = displayed
    bad = [lv for lv, v in displayed.items() if not v["ok"]]
    if bad:
        raise ValueError(f"displayed values do not reproduce the artifacts at levels {bad}: "
                         f"{ {lv: displayed[lv] for lv in bad} }")

    # 4 ── level means and SDs agree with the tabulated level-mean artifact.
    level_means = mlc.read("level_means")
    lm = level_means[level_means["substudy"].eq(mlc.PH45)].set_index("level_pct")
    deviations = {}
    for level in mlc.LEVELS:
        a = table[table["panel"].eq("A") & table["kind"].eq("level_mean")
                  & table["level_pct"].eq(level)]
        b_uv = table[table["panel"].eq("B") & table["kind"].eq("uv_level_mean")
                     & table["level_pct"].eq(level)]
        b_mo = table[table["panel"].eq("B") & table["kind"].eq("model_level")
                     & table["level_pct"].eq(level)]
        deviations[level] = {
            "copt0": abs(float(a["y_value"].iloc[0]) - float(lm.loc[level, "copt0"])),
            "copt0_sd": abs(float(a["y_sd"].iloc[0]) - float(lm.loc[level, "copt0_sd"])),
            "uv": abs(float(b_uv["y_value"].iloc[0]) - float(lm.loc[level,
                                                                   "uv_pct_injected_end"])),
            "model": abs(float(b_mo["y_value"].iloc[0]) - float(lm.loc[level, "model_pct_end"]))}
    worst = max(v for d in deviations.values() for v in d.values())
    checks["agrees_with_level_mean_artifact"] = {
        "max_abs_deviation": worst, "per_level": deviations, "ok": bool(worst < 1e-9)}
    if not checks["agrees_with_level_mean_artifact"]["ok"]:
        raise ValueError("panel means disagree with copt_loading_level_means.csv")

    # 5 ── the narrative orderings the figure asserts.
    copt = [meta["panel_a"]["copt0_mean_pct"][lv] for lv in mlc.LEVELS]
    model = [meta["panel_b"]["model_pct"][lv] for lv in mlc.LEVELS]
    uv = [meta["panel_b"]["uv_pct"][lv] for lv in mlc.LEVELS]
    checks["orderings"] = {
        "copt_ladder_monotonic_increasing": bool(np.all(np.diff(copt) > 0)),
        "model_monotonic_decreasing": bool(np.all(np.diff(model) < 0)),
        "uv_monotonic_decreasing": bool(np.all(np.diff(uv) < 0)),
        "uv_follows_model_direction": True,
        "copt_pct": copt, "model_pct": model, "uv_pct": uv}
    for name in ("copt_ladder_monotonic_increasing", "model_monotonic_decreasing",
                 "uv_monotonic_decreasing"):
        if not checks["orderings"][name]:
            raise ValueError(f"the figure's stated ordering failed against the artifacts: {name}")

    # 6 ── the conditional-agreement boundary: the contrast crossing and its calibrated fraction.
    c = meta["panel_c"]
    contrast = table[table["panel"].eq("C") & table["kind"].eq("offset_contrast")]
    x = contrast["x_value"].to_numpy(float)
    y = contrast["y_value"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    zero_crossing = float(-intercept / slope)
    checks["offset_conditionality"] = {
        "crossover_offset_ugml": c["crossover_offset_ugml"],
        "crossover_displayed": mlc.CROSSOVER_OFFSET_UGML,
        "zero_crossing_of_plotted_contrast": zero_crossing,
        "agrees_with_provenance": bool(abs(zero_crossing - c["crossover_offset_ugml"]) < 1e-6),
        "calibrated_offset_ugml": c["calibrated_offset_ugml"],
        "crossover_as_fraction_of_calibrated": c["crossover_as_fraction_of_calibrated"],
        "fraction_displayed_pct": round(c["crossover_as_fraction_of_calibrated"] * 100),
        "contrast_negative_below_crossover": bool(y[x < c["crossover_offset_ugml"]].max() < 0),
        "contrast_positive_above_crossover": bool(y[x > c["crossover_offset_ugml"]].min() > 0),
        "uncertainty_interval_available": c["uncertainty_interval_available"],
        "uncertainty_band_drawn": c["uncertainty_band_drawn"]}
    oc = checks["offset_conditionality"]
    if not (oc["agrees_with_provenance"] and oc["contrast_negative_below_crossover"]
            and oc["contrast_positive_above_crossover"]
            and round(c["calibrated_offset_ugml"], 2) == mlc.CALIBRATED_OFFSET_UGML
            and oc["fraction_displayed_pct"] == 40 and not oc["uncertainty_band_drawn"]):
        raise ValueError(f"the offset-conditionality panel does not reproduce the established "
                         f"crossover: {oc}")

    # 7 ── scope: one sub-study, no out-of-scope study, no particle-side optical quantity.
    quantities = sorted(set(table["y_quantity"]) | set(table["x_quantity"]))
    excluded = ("copt_frac_loss", "d50", "q3", "tail_pct", "per_run_q0")
    blob = " ".join(str(v) for v in table.to_dict("list").values()).lower()
    found_q = [q for q in excluded if q in blob]
    found_t = mlc.find_foreign_tokens(blob)
    under_study = all(str(p).startswith(str(mlc.STUDY_REL)) for p in mlc.SOURCES.values())
    checks["scope"] = {
        "y_and_x_quantities": quantities,
        "excluded_quantities_found": found_q,
        "sources_all_under_study": under_study,
        "foreign_tokens_found": found_t,
        "substudies_present": sorted(set(table["substudy"])),
        "ok": (not found_q and under_study and not found_t
               and set(table["substudy"]) == {mlc.MANUSCRIPT_SUBSTUDY})}
    if not checks["scope"]["ok"]:
        raise ValueError(f"a quantity or sub-study outside the main figure's scope reached it: "
                         f"{checks['scope']}")

    # 8 ── axis limits do not clip anything drawn.
    limits = {"A": _limits_a(table), "B": _limits_b(table), "C": _limits_c(table)}
    clipping = {}
    for panel, (lo, hi) in limits.items():
        # the reference offsets are vertical lines: they have an x position and no y value
        drawn = table[table["panel"].eq(panel) & table["kind"].ne("reference_offset")]
        extremes = _extremes(drawn)
        low, high = min(extremes), max(extremes)
        clipping[panel] = {"y_limits": [lo, hi], "min_drawn": round(low, 3),
                           "max_drawn": round(high, 3), "ok": bool(low >= lo and high <= hi)}
    checks["axis_limits_do_not_clip"] = clipping
    bad = [p for p, v in clipping.items() if not v["ok"]]
    if bad:
        raise ValueError(f"a panel's y limits clip a drawn value or error bar: {bad}")
    return checks


# ── rendering ────────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.lines import Line2D                                        # noqa: E402

DOSE_LIM = (120.0, 235.0)
DOSE_TICKS = (133, 177, 222)


def _pad(values, frac_lo=0.08, frac_hi=0.08):
    """Limits padded as a fraction of the drawn range — no snapping, so no dead bands."""
    lo, hi = float(min(values)), float(max(values))
    span = hi - lo
    return lo - frac_lo * span, hi + frac_hi * span


def _extremes(panel: pd.DataFrame) -> list[float]:
    """Every value the panel draws, error bars included."""
    sd = panel["y_sd"].fillna(0.0) if "y_sd" in panel else 0.0
    return (panel["y_value"] - sd).tolist() + (panel["y_value"] + sd).tolist()


def _limits_a(table):
    return _pad(_extremes(table[table["panel"].eq("A")]))


def _limits_b(table):
    return _pad(_extremes(table[table["panel"].eq("B")]))


def _limits_c(table):
    # extra headroom: the crossover / calibrated marks live in a band above the contrast
    c = table[table["panel"].eq("C") & table["kind"].eq("offset_contrast")]
    return _pad(list(c["y_value"]), 0.08, 0.20)


def _axis_a(ax, table):
    a = table[table["panel"].eq("A")]
    runs = a[a["kind"].eq("run_point")]
    ax.plot(runs["x_value"], runs["y_value"], "o", ms=2.6, mfc="white",
            mec=mlc.BLUE_PALE, mew=0.8, ls="none", zorder=3)

    mean = a[a["kind"].eq("level_mean")].sort_values("x_value")
    ax.plot(mean["x_value"], mean["y_value"], "-", lw=1.0, color=mlc.BLUE_MID, zorder=4)
    ax.errorbar(mean["x_value"], mean["y_value"], yerr=mean["y_sd"], fmt="o", ms=4.4,
                color=mlc.BLUE, ecolor=mlc.BLUE, elinewidth=0.9, capsize=2.4, capthick=0.9,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5)

    ax.set_xlabel("Delivered CFZ dose (µg)")
    ax.set_ylabel("Starting Copt (%)")
    ax.set_title("Loading ladder was realized", pad=4)
    ax.set_xlim(*DOSE_LIM)
    ax.set_xticks(DOSE_TICKS)
    ax.set_ylim(*_limits_a(table))
    mlc.clean_axes(ax)


def _axis_b(ax, table):
    b = table[table["panel"].eq("B")]
    runs = b[b["kind"].eq("uv_run")]
    ax.plot(runs["x_value"], runs["y_value"], "o", ms=2.6, mfc="white",
            mec=mlc.BLUE_PALE, mew=0.8, ls="none", zorder=3)

    model = b[b["kind"].eq("model_level")].sort_values("x_value")
    ax.plot(model["x_value"], model["y_value"], ls=(0, (5, 2.2)), lw=1.4, marker="s", ms=4.0,
            color=mlc.VERMILLION, markeredgecolor="white", markeredgewidth=0.5, zorder=4)

    uv = b[b["kind"].eq("uv_level_mean")].sort_values("x_value")
    ax.plot(uv["x_value"], uv["y_value"], "-", lw=1.0, color=mlc.BLUE_MID, zorder=4)
    ax.errorbar(uv["x_value"], uv["y_value"], yerr=uv["y_sd"], fmt="o", ms=4.4,
                color=mlc.BLUE, ecolor=mlc.BLUE, elinewidth=0.9, capsize=2.4, capthick=0.9,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5)

    ax.set_xlabel("Delivered CFZ dose (µg)")
    ax.set_ylabel("Dissolved at end (% of dose)")
    ax.set_title("Dissolved fraction fell", pad=4)
    ax.set_xlim(*DOSE_LIM)
    ax.set_xticks(DOSE_TICKS)
    ax.set_ylim(*_limits_b(table))
    mlc.clean_axes(ax)


def _axis_c(ax, table, meta):
    c = table[table["panel"].eq("C")]
    contrast = c[c["kind"].eq("offset_contrast")].sort_values("x_value")
    refs = c[c["kind"].eq("reference_offset")].set_index("series")["x_value"]
    lo, hi = _limits_c(table)
    # the reference lines stop short of the label band so the two never overprint
    label_floor = lo + 0.82 * (hi - lo)

    ax.axhline(0.0, color=mlc.GREY, lw=0.8, zorder=1)
    ax.vlines(float(refs["Crossover offset"]), lo, label_floor, color=mlc.VERMILLION,
              ls=(0, (3, 2)), lw=1.0, zorder=2)
    ax.vlines(float(refs["Calibrated offset"]), lo, label_floor, color=mlc.DARK, lw=0.9, zorder=2)
    ax.plot(contrast["x_value"], contrast["y_value"], "-", lw=1.3, color=mlc.BLUE, zorder=3)
    ax.plot(contrast["x_value"], contrast["y_value"], "o", ms=3.0, color=mlc.BLUE,
            markeredgecolor="white", markeredgewidth=0.4, zorder=4)

    # Subtle marks in the empty band above the contrast; the caption carries the interpretation.
    ax.text(float(refs["Crossover offset"]), 0.985,
            f"trend reverses\n{mlc.CROSSOVER_OFFSET_UGML:.2f}",
            transform=ax.get_xaxis_transform(), fontsize=6.0, color=mlc.VERMILLION,
            ha="center", va="top", linespacing=1.2)
    ax.text(float(refs["Calibrated offset"]), 0.985,
            f"calibrated\n{meta['panel_c']['calibrated_offset_ugml']:.2f}",
            transform=ax.get_xaxis_transform(), fontsize=6.0, color=mlc.DARK,
            ha="center", va="top", linespacing=1.2)

    ax.set_xlabel("Additive filter-recovery offset (µg/mL)")
    ax.set_ylabel("Low-load minus high-load\nUV recovery (percentage points)")
    ax.set_title("UV ordering depends on offset", pad=4)
    ax.set_xlim(-0.08, 3.10)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(lo, hi)
    mlc.clean_axes(ax)


def _shared_legend(fig, axes):
    """One key for A and B. Per-panel legends collide with the data at this figure size, and the
    three series mean the same thing in both panels: a technical run, its level mean, the frozen
    prediction."""
    handles = [
        Line2D([], [], color=mlc.BLUE_PALE, marker="o", ms=2.6, ls="none", mfc="white", mew=0.8),
        Line2D([], [], color=mlc.BLUE, marker="o", ms=4.4, lw=1.0,
               markeredgecolor="white", markeredgewidth=0.5),
        Line2D([], [], color=mlc.VERMILLION, marker="s", ms=4.0, ls=(0, (5, 2.2)), lw=1.4,
               markeredgecolor="white", markeredgewidth=0.5)]
    labels = ["Technical run", "Mean ± technical SD",
              "Frozen model (no parameter refitted)"]
    left = axes[0].get_position().x0
    right = axes[1].get_position().x1
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(left, 0.085, right - left, 0.06),
               mode="expand", ncol=3, frameon=False, fontsize=6.4, handlelength=1.9,
               borderpad=0.0, handletextpad=0.45, columnspacing=1.0)


def render(table: pd.DataFrame, meta: dict, out_dir: Path, formats=("pdf", "png", "svg")):
    mlc.apply_style()
    fig = plt.figure(figsize=(mlc.FIG_W, 3.10))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.06],
                          left=0.072, right=0.995, top=0.870, bottom=0.255, wspace=0.52)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    _axis_a(axA, table)
    _axis_b(axB, table)
    _axis_c(axC, table, meta)
    _shared_legend(fig, (axA, axB))
    mlc.panel_tags(fig, ((axA, "A"), (axB, "B"), (axC, "C")))
    return mlc.save(fig, out_dir, STEM, formats)


CAPTION = """# {stem} — caption draft

**Increasing the delivered dose produced a monotonic pH 4.5 loading ladder, and the frozen
finite-sink model predicted the observed decrease in dissolved-dose fraction under the established
filter-recovery correction.** This is a single-preparation comparison and the UV ordering reverses
below the offset crossover, so the agreement is conditional. All three loadings were aliquots of
ONE pH 4.5 suspension (2026-07-27) delivered at three injected volumes, and the three runs per
loading are technical replicates measured on that one preparation; error bars are technical SDs
and there is no preparation-level error term. It is not a preparation-level validation and nothing
here generalises across preparations.

**(A)** Starting optical concentration against delivered clofazimine dose. Open symbols are the
{n_reps} technical runs at each loading and filled symbols the mean {pm} technical SD, joined by a
guide line. Delivered doses of {d12:.1f}, {d18:.1f} and {d24:.1f} {ug} gave starting Copt of
{c12:.1f}, {c18:.1f} and {c24:.1f} %, so the intended loading ladder was realized within this
preparation.

**(B)** Terminal dissolved clofazimine as a percentage of the delivered dose. Orange squares are
the frozen finite-sink prediction and blue circles the UV measurement (mean {pm} technical SD, open
symbols the individual runs). **No model parameter was refitted for this dataset**: each run was
predicted from its own delivered dose, the shared suspension QC particle size distribution rather
than the per-run in-cuvette q0, pH 4.5 solubility, and the rate scale {rate} frozen from the
historical fit ({source}, column {column}). The model predicted {m12:.1f}, {m18:.1f} and
{m24:.1f} % of dose dissolved across the ladder; UV recovered {u12:.1f} {pm} {s12:.1f},
{u18:.1f} {pm} {s18:.1f} and {u24:.1f} {pm} {s24:.1f} %, following the predicted ordering. No
uncertainty band is drawn on the model because with every parameter frozen there is no model
uncertainty to draw.

**(C)** The UV ordering is conditional on the additive filter-recovery correction. Plotted is the
contrast between the ends of the ladder — recovery at {low} % loading minus recovery at {high} %
loading — against the assumed additive offset; positive values mean the recovered fraction
decreases with loading, negative values mean it increases. Because the correction is additive and
the delivered dose differs across the ladder, the same offset adds a different number of recovery
points at each loading ({pp12:.0f}, {pp18:.0f} and {pp24:.0f} percentage points at the calibrated
value), so the offset sets the sign of the trend. The contrast crosses zero at
{crossover:.2f} {ugml}, {fraction:.0f} % of the calibrated {calibrated:.2f} {ugml}; below that
offset the recovered fraction rises with loading instead of falling. The calibration artifact
carries no uncertainty interval for the offset, so no band is drawn and the calibrated value is
not shown to lie safely above the crossover.

Copt in (A) is the realized loading coordinate, not a dissolved-dose fraction. The particle-side
optical measurements — fractional Copt loss and reliability-gated q3 — are reported descriptively
in {si_stem}; they are scattering-weighted particle coordinates and do not independently confirm
the mass-domain result.
"""


def _emit(out: Path, formats) -> int:
    table, meta = build_source_data()
    checks = validate(table, meta)
    written = render(table, meta, out, formats)

    csv_path = out / f"{STEM}_source_data.csv"
    table.to_csv(csv_path, index=False)

    a, b, c = meta["panel_a"], meta["panel_b"], meta["panel_c"]
    offsets = mlc.read("offsets")
    contrib = (offsets[np.isclose(offsets["offset_ugml"], c["calibrated_offset_ugml"])]
               .set_index("level_pct")["offset_contribution_pp"])
    caption = CAPTION.format(
        stem=STEM, si_stem="FigureS_loading_optical_and_reliability_diagnostics",
        pm="±", ug="µg", ugml="µg/mL", n_reps=3,
        d12=a["dose_ug"][12], d18=a["dose_ug"][18], d24=a["dose_ug"][24],
        c12=a["copt0_mean_pct"][12], c18=a["copt0_mean_pct"][18], c24=a["copt0_mean_pct"][24],
        m12=b["model_pct"][12], m18=b["model_pct"][18], m24=b["model_pct"][24],
        u12=b["uv_pct"][12], u18=b["uv_pct"][18], u24=b["uv_pct"][24],
        s12=b["uv_technical_sd_pct"][12], s18=b["uv_technical_sd_pct"][18],
        s24=b["uv_technical_sd_pct"][24],
        rate=mlc.FROZEN_RATE_SCALE, source=mlc.FROZEN_RATE_SOURCE, column=mlc.FROZEN_RATE_COLUMN,
        low=LOW_LEVEL, high=HIGH_LEVEL,
        pp12=float(contrib[12]), pp18=float(contrib[18]), pp24=float(contrib[24]),
        crossover=c["crossover_offset_ugml"], calibrated=c["calibrated_offset_ugml"],
        fraction=c["crossover_as_fraction_of_calibrated"] * 100)
    wording = mlc.check_wording(caption, allow_negated=("preparation-level validation",))
    if not wording["ok"]:
        raise ValueError(f"the caption overstates the claim or names a foreign study: {wording}")
    checks["caption_wording"] = wording
    caption_path = out / f"{STEM}_caption.md"
    caption_path.write_text(caption)

    prov = provenance_record(
        "manuscript_loading_response", study_root=str(data_root() / mlc.STUDY_REL),
        uv_ph_values=(4.5,), figure_stem=STEM,
        manuscript_section="3.4.2 starting particle loading",
        does_not_overwrite="conc_dependent_disso_study/analysis/copt_loading.* "
                           "(the analysis figure is left untouched)",
        sources={k: str(data_root() / v) for k, v in mlc.SOURCES.items()},
        source_paths_relative={k: str(v) for k, v in mlc.SOURCES.items()},
        panel_sources={
            "A": "copt_loading_runs.csv::copt0 (+ dose_mg). The tabulated regression in "
                 "copt_loading_linearity.csv is recorded under panel_details.A."
                 "tabulated_linearity_not_drawn but is NOT plotted",
            "B": "copt_loading_runs.csv::model_pct_end and ::uv_pct_injected_end (+ dose_mg)",
            "C": "copt_loading_filter_offset_sensitivity.csv::recovery_pct differenced between "
                 "the 12 % and 24 % levels; reference offsets from provenance.json"},
        panel_details={"A": a, "B": b, "C": c},
        design=meta["design"], frozen_model=meta["frozen_model"],
        aggregations=mlc.AGGREGATIONS,
        filter_offset=c, numerical_checks=checks,
        career_artifacts_used=False,
        scope={"substudy": mlc.MANUSCRIPT_SUBSTUDY,
               "row_admission": "copt_loading artifacts restricted to the "
                                f"{mlc.MANUSCRIPT_SUBSTUDY} sub-study by "
                                "manuscript_loading_common.read_scoped",
               "domain": "delivered dose, starting Copt (loading coordinate) and UV dissolved "
                         "mass; no particle-side optical result is used as confirmation",
               "excluded": ["fractional Copt loss", "q3", "per-run in-cuvette q0",
                            "optical reliability diagnostics", "NSF CAREER artifacts",
                            "Arm A / Arm B", "dissolution-medium polysorbate data"],
               "claim": "independent-dataset, within-preparation loading-response evaluation "
                        "with forward-model parameters frozen; conditional agreement under the "
                        "established additive filter-recovery correction",
               "claim_boundaries": mlc.CLAIM_BOUNDARIES})
    # the sidecar may NAME the other studies it excludes; it may not mention the
    # out-of-scope sub-study at all
    stray = mlc.find_foreign_tokens(json.dumps(prov), mlc.OUT_OF_SCOPE_SUBSTUDY_TOKENS)
    if stray:
        raise ValueError(f"the provenance sidecar names an out-of-scope study: {stray}")
    prov_path = write_provenance(out / f"{STEM}_provenance.json", prov)

    print(f"sub-study: {mlc.MANUSCRIPT_SUBSTUDY} — {meta['design']['n_preparations']} "
          f"preparation, {meta['design']['technical_reps_per_level']} technical reps per level")
    print(f"frozen rate scale {mlc.FROZEN_RATE_SCALE} from {mlc.FROZEN_RATE_SOURCE}"
          f"[{mlc.FROZEN_RATE_COLUMN}] — refitted_here = "
          f"{meta['frozen_model']['refitted_here']}")
    print(f"{'dose (ug)':>10} {'Copt0 (%)':>12} {'model (%)':>10} {'UV (%)':>16}")
    for level in mlc.LEVELS:
        print(f"{a['dose_ug'][level]:10.1f} "
              f"{a['copt0_mean_pct'][level]:7.1f} ± {a['copt0_technical_sd_pct'][level]:.1f} "
              f"{b['model_pct'][level]:10.1f} "
              f"{b['uv_pct'][level]:11.1f} ± {b['uv_technical_sd_pct'][level]:.1f}")
    print(f"contrast crosses zero at {c['crossover_offset_ugml']:.4f} ug/mL = "
          f"{c['crossover_as_fraction_of_calibrated']:.0%} of the calibrated "
          f"{c['calibrated_offset_ugml']:.2f}; no uncertainty interval exists for the offset, "
          f"so the UV agreement is CONDITIONAL.")
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
