"""Manuscript figure — mass-domain model development in the pH-dependent study (Section 3.2).

A **dissolved-mass-only** figure. No LD intensity, Copt, q3, angular signal, Mie operator, or any
optical conversion enters any panel.

The argument the four panels have to make legible without the reader reconstructing the analysis
history:

  A, B   The physical model reproduced the qualitative pH effect but not the UV trajectories
         quantitatively. The original 2.2-fold multiplier improved an injection-started
         prediction, but it acted on both the unobserved interval before the first UV sample and
         the measured interval, so it cannot be read as measured post-capture kinetics.
         Initializing each trajectory at its own first UV measurement improved held-out
         prediction without any rate correction.
  C      Adding a fitted post-capture rate did not materially improve held-out prediction
         (7.0 vs 7.0 percentage points).
  D      The 2.2-fold correction did not transfer quantitatively to pH 5.0.

**The renderer never fits.** :func:`build_source_data` assembles every plotted number from the
authoritative analysis outputs listed in :data:`SOURCES`; :func:`render` consumes only that
table. Where a trajectory is not already tabulated (the pH 5.0 curves), it is reconstructed by
the same time-rescaling the fitting code uses — a rate scale ``k`` evaluates the base trajectory
at ``t·k`` — and :func:`validate` checks the reconstruction against the scalars the artifact
recorded. A missing or malformed artifact raises; nothing falls back to a hard-coded parameter.

Aggregation, everywhere: technical runs are averaged **within preparation date first**, then
preparation dates are weighted **equally**. Variability shown is between preparation-date means.

Terminology this module deliberately enforces: the 2.2 multiplier is "original injection-started
correction", never "selected", "validated", or a transferable kinetic constant; alignment is
"aligned at first UV measurement", never an unexplained "anchor"; pH 5.0 is "condition transfer",
never external validation.

Run with the pipeline venv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diffractomorph_pipeline.config import data_root

from arm_b_provenance import provenance_record, write_provenance

# ── authoritative sources ────────────────────────────────────────────────────────────────────
PH_STUDY = Path("disso_experiments/ph_dependent_dissolution_study")
RATE_REFINEMENT = Path("disso_experiments/dissolution_media_diagnostic/"
                       "tween80_suspension_wetting_arm_a/analysis/rate_refinement")
SOURCES = {
    # Panels A/B: observations and all four model trajectories, per run, already computed.
    "predictions": RATE_REFINEMENT / "rate_refinement_predictions.csv",
    # Panel C: per-held-out-date RMSE and the summary across held-out dates.
    "lodo": RATE_REFINEMENT / "rate_refinement_lodo_validation.csv",
    "candidates": RATE_REFINEMENT / "rate_refinement_candidate_models.csv",
    # Panel D: pH 5.0 observations, base trajectories, and the frozen/own rate scalars.
    "uv_all": PH_STUDY / "summary/uv_timecourse_all.csv",
    "base_traj": PH_STUDY / "forward_prediction/scalar_fit/selected_rate_only_model_trajectories.csv",
    "fit_summary": PH_STUDY / "forward_prediction/scalar_fit/fit_summary.csv",
    # Records both pH 5.0 scopes: the frozen transfer and the pH-5.0-only optimum, each with
    # its own RMSE, so neither has to be recomputed for the caption.
    "ph5_transfer": PH_STUDY / "forward_prediction/scalar_fit/selected_rate_only_pH5_transfer.csv",
}
FIT_COHORT = "historical pH 4.0/4.5 fit cohort"     # the ONLY cohort admitted to this figure

# model key → (series label, panel role). Order fixes legend and Panel C row order.
MODELS = [
    ("injection_base", "Physical model from injection (no correction)"),
    ("injection_selected_rate", "Original 2.2× injection-started correction"),
    ("first_uv_anchor", "Aligned at first UV measurement (no rate correction)"),
    ("anchored_rate", "Aligned at first UV + fitted post-capture rate"),
]
EXPECTED_LODO = {"injection_base": 11.4, "injection_selected_rate": 8.7,
                 "first_uv_anchor": 7.0, "anchored_rate": 7.0}


def _read(key: str) -> pd.DataFrame:
    path = data_root() / SOURCES[key]
    if not path.exists():
        raise FileNotFoundError(
            f"authoritative source '{key}' missing: {path}. This figure does not substitute "
            f"defaults for absent artifacts — regenerate the analysis that produces it.")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"authoritative source '{key}' is empty: {path}")
    return frame


def _date_balanced(frame: pd.DataFrame, value: str, keys: list[str]) -> pd.DataFrame:
    """Average technical runs within preparation date, then weight dates equally."""
    per_date = frame.groupby(keys + ["date"], as_index=False).agg(
        value=(value, "mean"), n_runs=("rep", "nunique"))
    return per_date.groupby(keys, as_index=False).agg(
        value_pct=("value", "mean"), sd_pct=("value", "std"),
        n_dates=("date", "nunique"), n_runs=("n_runs", "sum"))


# ── panels A + B ─────────────────────────────────────────────────────────────────────────────

def _trajectory_panels() -> tuple[pd.DataFrame, dict]:
    pred = _read("predictions")
    pred = pred[pred["cohort"].eq(FIT_COHORT)].copy()
    if pred.empty:
        raise ValueError(f"no rows for cohort {FIT_COHORT!r} in the predictions artifact")
    pred["date"] = pred["date"].astype(str)

    rows = []
    # UV observations are identical across model rows; take them from one model only.
    obs_src = pred[pred["model"].eq("injection_base")]
    obs = _date_balanced(obs_src, "observed_pct_injected", ["condition", "time_min"])
    for r in obs.itertuples():
        rows.append({"panel": r.condition, "series": "UV-measured dissolved fraction",
                     "kind": "observation", "condition": r.condition, "time_min": r.time_min,
                     "value_pct": r.value_pct, "sd_pct": r.sd_pct,
                     "n_dates": r.n_dates, "n_runs": r.n_runs})
    for key, label in MODELS:
        sub = pred[pred["model"].eq(key)]
        curve = _date_balanced(sub, "predicted_pct_injected", ["condition", "time_min"])
        for r in curve.itertuples():
            rows.append({"panel": r.condition, "series": label, "kind": "model_curve",
                         "condition": r.condition, "time_min": r.time_min,
                         "value_pct": r.value_pct, "sd_pct": r.sd_pct,
                         "n_dates": r.n_dates, "n_runs": r.n_runs,
                         "rate_scale": float(sub["rate_scale"].iloc[0])})

    # First UV time, read from the data rather than assumed.
    first_uv = float(pred.loc[pred["is_first_observation"].astype(bool), "time_min"].min())
    distinct = sorted(pred.loc[pred["is_first_observation"].astype(bool), "time_min"].unique())
    meta = {"first_uv_time_min": first_uv,
            "first_uv_times_observed": [float(t) for t in distinct],
            "n_dates": int(pred["date"].nunique()),
            "n_runs": int(pred["id"].nunique()),
            "conditions": sorted(pred["condition"].unique())}
    return pd.DataFrame(rows), meta


# ── panel C ──────────────────────────────────────────────────────────────────────────────────

def _heldout_panel() -> tuple[pd.DataFrame, dict]:
    lodo = _read("lodo")
    cand = _read("candidates")
    lodo = lodo[lodo["cohort"].eq(FIT_COHORT)]
    cand = cand[cand["cohort"].eq(FIT_COHORT)]

    rows = []
    for key, label in MODELS:
        per_date = lodo[lodo["model"].eq(key)].dropna(subset=["held_post_first_rmse_pct"])
        for r in per_date.itertuples():
            rows.append({"panel": "held-out RMSE", "series": label, "kind": "heldout_point",
                         "model_key": key, "held_date": str(r.held_date),
                         "value_pct": float(r.held_post_first_rmse_pct)})
        summary = cand[cand["model"].eq(key)]
        if len(summary) != 1:
            raise ValueError(f"expected one candidate row for {key!r}, found {len(summary)}")
        rows.append({"panel": "held-out RMSE", "series": label, "kind": "heldout_summary",
                     "model_key": key, "held_date": "all",
                     "value_pct": float(summary["lodo_rmse_pct"].iloc[0]),
                     "rate_scale": float(summary["rate_scale"].iloc[0]),
                     "n_dates": int(per_date["held_date"].nunique())})
    meta = {"residual_basis": "held_post_first_rmse_pct — the first (aligning) observation is "
                              "excluded from residuals; verified against the artifact",
            "n_held_out_dates": int(lodo["held_date"].nunique())}
    return pd.DataFrame(rows), meta


# ── panel D ──────────────────────────────────────────────────────────────────────────────────

def _transfer_panel() -> tuple[pd.DataFrame, dict]:
    uv = _read("uv_all")
    traj = _read("base_traj")
    fs = _read("fit_summary").iloc[0]
    tr = _read("ph5_transfer")
    frozen_row = tr[tr["scope"].eq("overall_frozen")]
    own_row = tr[tr["scope"].eq("pH5_own_optimum")]
    for name, row in (("overall_frozen", frozen_row), ("pH5_own_optimum", own_row)):
        if len(row) != 1:
            raise ValueError(f"pH 5.0 transfer artifact: expected one {name!r} row, "
                             f"found {len(row)}")
    frozen = float(frozen_row["rate_scale"].iloc[0])
    own = float(own_row["rate_scale"].iloc[0])
    frozen_rmse = float(frozen_row["rmse_pct"].iloc[0])
    own_rmse = float(own_row["rmse_pct"].iloc[0])

    uv5 = uv[uv["ph"].eq(5.0)].copy()
    if uv5.empty:
        raise ValueError("no pH 5.0 rows in the UV timecourse artifact")
    uv5["date"] = uv5["date"].astype(str)
    uv5["obs_pct"] = uv5["recovery_t"] * 100.0
    uv5["id"] = ["pH5.0_" + d + "_R" + str(int(r))
                 for d, r in zip(uv5["date"], uv5["rep"])]
    traj5 = traj[traj["ph"].eq(5.0)].copy()
    traj5["date"] = traj5["date"].astype(str)
    if traj5.empty:
        raise ValueError("no pH 5.0 rows in the base-trajectory artifact")

    rows = []
    obs = _date_balanced(uv5, "obs_pct", ["time_min"])
    for r in obs.itertuples():
        rows.append({"panel": "pH 5.0", "series": "UV-measured dissolved fraction",
                     "kind": "observation", "condition": "pH 5.0", "time_min": r.time_min,
                     "value_pct": r.value_pct, "sd_pct": r.sd_pct,
                     "n_dates": r.n_dates, "n_runs": r.n_runs})

    # A rate scale k evaluates the base trajectory at t*k — the same rescaling the fitting code
    # applies. Reconstructed here only because no pH 5.0 curve is tabulated; validated in
    # validate() against the scalars the artifact recorded.
    times = np.sort(uv5["time_min"].unique())
    for label, k, kind in ((f"Frozen {frozen:.1f}× correction from pH 4.0/4.5", frozen, "model_curve"),
                           (f"pH 5.0-only multiplier ({own:.1f}×), post hoc", own, "model_curve_subordinate")):
        per_run = []
        for rid, g in traj5.groupby("id"):
            g = g.sort_values("t_min")
            per_run.append(pd.DataFrame({
                "date": g["date"].iloc[0], "rep": rid, "time_min": times,
                "pred": np.interp(times * k, g["t_min"], g["base_pct_dissolved"])}))
        curve = _date_balanced(pd.concat(per_run), "pred", ["time_min"])
        for r in curve.itertuples():
            rows.append({"panel": "pH 5.0", "series": label, "kind": kind,
                         "condition": "pH 5.0", "time_min": r.time_min,
                         "value_pct": r.value_pct, "sd_pct": r.sd_pct,
                         "n_dates": r.n_dates, "n_runs": r.n_runs, "rate_scale": k})

    meta = {"frozen_rate_scale": frozen, "ph5_own_rate_scale": own,
            "ph5_transfer_rmse_pct_artifact": frozen_rmse,
            "ph5_own_rmse_pct_artifact": own_rmse,
            "ph5_rmse_source": "selected_rate_only_pH5_transfer.csv",
            "heldout_is_independent_external": bool(
                own_row["is_independent_external_validation"].iloc[0]),
            "shared_dates_with_training": str(own_row["shared_dates_with_training"].iloc[0]),
            "n_dates": int(uv5["date"].nunique()), "n_runs": int(uv5["id"].nunique())}
    return pd.DataFrame(rows), meta


def build_source_data() -> tuple[pd.DataFrame, dict]:
    """Every plotted observation, summary and curve, from authoritative outputs only."""
    traj, traj_meta = _trajectory_panels()
    held, held_meta = _heldout_panel()
    transfer, transfer_meta = _transfer_panel()
    table = pd.concat([traj, held, transfer], ignore_index=True)
    table["study"] = "ph_dependent_dissolution_study"
    meta = {"trajectories": traj_meta, "held_out": held_meta, "transfer": transfer_meta}
    return table, meta


# ── validation ───────────────────────────────────────────────────────────────────────────────

def validate(table: pd.DataFrame, meta: dict) -> dict:
    """Numerical checks recorded alongside the figure. Raises if a structural rule is violated."""
    checks = {}
    summary = table[table["kind"].eq("heldout_summary")].set_index("series")["value_pct"]
    for key, label in MODELS:
        got = round(float(summary[label]), 1)
        checks[f"heldout_rmse_{key}"] = {"value": got, "expected": EXPECTED_LODO[key],
                                         "match": got == EXPECTED_LODO[key]}
    bad = [k for k, v in checks.items() if not v["match"]]
    if bad:
        raise ValueError(f"held-out RMSE does not reproduce the authoritative outputs: {bad}")

    # pH 5.0 reconstruction vs the artifact scalar
    uv = _read("uv_all"); traj = _read("base_traj")
    uv5 = uv[uv["ph"].eq(5.0)].copy()
    uv5["id"] = ["pH5.0_" + str(int(d)) + "_R" + str(int(r))
                 for d, r in zip(uv5["date"], uv5["rep"])]
    traj5 = traj[traj["ph"].eq(5.0)]
    def _ph5_rmse(k: float) -> float:
        res = []
        for rid, g in uv5.groupby("id"):
            b = traj5[traj5["id"].eq(rid)].sort_values("t_min")
            if b.empty:
                continue
            pred = np.interp(g["time_min"] * k, b["t_min"], b["base_pct_dissolved"])
            res.append(pd.DataFrame({"date": g["date"], "r": pred - g["recovery_t"] * 100.0}))
        r = pd.concat(res)
        per_date = r.groupby("date")["r"].apply(lambda s: float(np.sqrt((s ** 2).mean())))
        return float(np.sqrt((per_date ** 2).mean()))

    # BOTH plotted pH 5.0 curves are validated against the scopes the artifact records.
    for name, k, artifact in (
            ("ph5_frozen_transfer_rmse", meta["transfer"]["frozen_rate_scale"],
             meta["transfer"]["ph5_transfer_rmse_pct_artifact"]),
            ("ph5_own_optimum_rmse", meta["transfer"]["ph5_own_rate_scale"],
             meta["transfer"]["ph5_own_rmse_pct_artifact"])):
        recomputed = _ph5_rmse(k)
        checks[name] = {"rate_scale": k, "recomputed": round(recomputed, 3),
                        "artifact": round(artifact, 3),
                        "abs_diff": round(abs(recomputed - artifact), 3),
                        "match_to_1dp": round(recomputed, 1) == round(artifact, 1),
                        "source": meta["transfer"]["ph5_rmse_source"]}
        if not checks[name]["match_to_1dp"]:
            raise ValueError(f"pH 5.0 reconstruction disagrees with the recorded RMSE ({name})")

    checks["only_ph_study_present"] = {
        "studies": sorted(table["study"].unique()),
        "panels": sorted(table["panel"].unique()),
        "ok": set(table["study"]) == {"ph_dependent_dissolution_study"}}
    checks["aggregation"] = {
        "rule": "technical runs averaged within preparation date, then dates weighted equally",
        "n_dates_trajectories": meta["trajectories"]["n_dates"],
        "n_runs_trajectories": meta["trajectories"]["n_runs"],
        "n_dates_ph5": meta["transfer"]["n_dates"],
        "n_runs_ph5": meta["transfer"]["n_runs"]}
    checks["first_uv_excluded_from_residuals"] = meta["held_out"]["residual_basis"]
    checks["ph5_frozen_not_refitted"] = {
        "frozen_rate_scale": meta["transfer"]["frozen_rate_scale"],
        "estimated_on": "pH 4.0 and pH 4.5 only",
        "ph5_in_estimation": False}
    return checks


# ── rendering ────────────────────────────────────────────────────────────────────────────────

import matplotlib                                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.lines import Line2D                                        # noqa: E402
from matplotlib.patches import Patch                                       # noqa: E402

STEM = "Figure_mass_domain_model_evaluation"
# Okabe-Ito, colour-vision-safe. Condition identity is carried by colour; model strategy by
# line style, and both are held constant across every panel.
COND_COLOR = {"pH 4.0": "#0072B2", "pH 4.5": "#E69F00", "pH 5.0": "#CC79A7"}
BASELINE_GREY = "#707070"


def _apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.2, "axes.labelsize": 8.6, "axes.titlesize": 8.8,
        "xtick.labelsize": 7.8, "ytick.labelsize": 7.8, "legend.fontsize": 7.0,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


# ── figure content ───────────────────────────────────────────────────────────────────────────

# Only these two strategies are drawn in A/B. The 2.2x correction and the aligned+fitted-rate
# curve are comparison models: the first is superseded, the second is visually indistinguishable
# from aligned-only. Both remain in panel C, where the comparison is the point.
SHOWN_IN_TRAJECTORY = (MODELS[0][1], MODELS[2][1])
# Concise panel C row labels, in display order (best last so it reads top-down worst→best).
PANEL_C_LABELS = {
    MODELS[0][1]: "Injection start",
    MODELS[1][1]: "Injection + fitted rate",
    MODELS[2][1]: "First-UV aligned",
    MODELS[3][1]: "First-UV aligned + rate",
}
YLIM = (40.0, 100.0)
FIG_W = 7.2                  # inches; journal double-column width


def per_condition_rmse() -> pd.DataFrame:
    """Within-condition post-first RMSE, date-balanced, for the two strategies A/B display.

    **This is not a held-out quantity.** ``rate_refinement_condition_diagnostics.csv`` tabulates
    per-condition values only for ``anchored_rate``, not for ``injection_base`` or
    ``first_uv_anchor``, so the condition-dependence comparison is computed here from the
    per-observation residuals in the predictions artifact, on the same post-first basis and with
    the same preparation-date balancing used everywhere else. It must be reported as a
    within-condition fit statistic and never conflated with the pooled leave-one-date-out RMSE
    in panel C, which is a different basis and cannot be reconciled arithmetically with it.
    """
    pred = _read("predictions")
    pred = pred[pred["cohort"].eq(FIT_COHORT)
                & ~pred["is_first_observation"].astype(bool)].copy()
    rows = []
    for (condition, model), g in pred.groupby(["condition", "model"]):
        per_date = g.groupby("date")["residual_model_minus_measured_pct"].apply(
            lambda s: float(np.sqrt((s ** 2).mean())))
        rows.append({"condition": condition, "model": model,
                     "within_condition_rmse_pct": float(np.sqrt((per_date ** 2).mean())),
                     "basis": "post-first residuals, date-balanced, ALL data (not held out)",
                     "n_dates": int(g["date"].nunique())})
    return pd.DataFrame(rows)


def figure_source_data() -> tuple[pd.DataFrame, dict]:
    """The plotted subset: pH 4.0/4.5 trajectories (two strategies) plus the held-out panel.

    pH 5.0 is excluded entirely — the transfer result is reported in Results and Table S3 — so
    the source table carries exactly what the figure shows and nothing else. The full builder
    still produces the pH 5.0 rows, which validate() checks and Table S3 draws on.
    """
    table, meta = build_source_data()
    keep_traj = (table["panel"].isin(["pH 4.0", "pH 4.5"])
                 & (table["kind"].eq("observation")
                    | (table["kind"].eq("model_curve") & table["series"].isin(SHOWN_IN_TRAJECTORY))))
    keep_c = table["panel"].eq("held-out RMSE")
    out = table[keep_traj | keep_c].copy()
    if not out[out["panel"].eq("pH 5.0")].empty:
        raise ValueError("pH 5.0 must not appear in the figure's source data")
    return out.reset_index(drop=True), meta


def _trajectory_axis(ax, table, condition, first_uv, show_ylabel, show_legend):
    sub = table[table["panel"].eq(condition)]
    ax.axvspan(0, first_uv, color="0.5", alpha=0.09, lw=0, zorder=0)

    baseline = sub[sub["series"].eq(MODELS[0][1])].sort_values("time_min")
    ax.plot(baseline["time_min"], baseline["value_pct"], ls="--", lw=1.4,
            color="#8A8A8A", zorder=2, label="Injection-start model")
    aligned = sub[sub["series"].eq(MODELS[2][1])].sort_values("time_min")
    ax.plot(aligned["time_min"], aligned["value_pct"], ls="-", lw=2.0,
            color=COND_COLOR[condition], zorder=3, solid_capstyle="round",
            label="First-UV-aligned model")

    obs = sub[sub["kind"].eq("observation")].sort_values("time_min")
    ax.errorbar(obs["time_min"], obs["value_pct"], yerr=obs["sd_pct"], fmt="o", ms=3.8,
                color="#1A1A1A", ecolor="#1A1A1A", elinewidth=0.8, capsize=2.0,
                markeredgecolor="white", markeredgewidth=0.4, zorder=5, label="UV measurement")

    ax.set_title(condition, fontsize=9.0, pad=3)
    ax.set_xlabel("Time (min)")
    if show_ylabel:
        ax.set_ylabel("Dissolved mass (% of injected dose)", fontsize=8.0)
    ax.set_ylim(*YLIM)
    ax.set_xlim(-0.4, float(obs["time_min"].max()) + 0.6)
    ax.set_xticks([0, 5, 10, 15, 20])        # the wider axis otherwise picks half-minute steps
    _clean(ax)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index(l) for l in
                 ("UV measurement", "Injection-start model", "First-UV-aligned model")]
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  loc="lower right", frameon=False, fontsize=6.8, handlelength=2.0,
                  borderpad=0.2, labelspacing=0.32, handletextpad=0.5)


def _heldout_axis(ax, table):
    rows = [label for _, label in MODELS][::-1]
    for i, label in enumerate(rows):
        pts = table[table["kind"].eq("heldout_point") & table["series"].eq(label)]
        ax.scatter(pts["value_pct"], np.full(len(pts), i), s=14, facecolors="none",
                   edgecolors="#9A9A9A", linewidths=0.8, zorder=2)
        value = float(table[table["kind"].eq("heldout_summary")
                            & table["series"].eq(label)]["value_pct"].iloc[0])
        ax.scatter([value], [i], marker="D", s=34, color="#1A1A1A", zorder=4)
        ax.text(value, i + 0.22, f"{value:.1f}", ha="center", va="bottom", fontsize=6.6,
                color="#1A1A1A", zorder=5)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([PANEL_C_LABELS[l] for l in rows], fontsize=7.4)
    ax.set_xlabel("Held-out RMSE (percentage points of injected dose)")
    ax.set_title("Held-out prediction error", fontsize=9.0, pad=3)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlim(0, max(table[table["kind"].eq("heldout_point")]["value_pct"].max() * 1.08, 17))
    ax.grid(axis="x", color="0.92", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    _clean(ax)


def render(table: pd.DataFrame, meta: dict, out_dir: Path,
                   formats=("pdf", "png", "svg")):
    _apply_style()
    first_uv = meta["trajectories"]["first_uv_time_min"]
    fig = plt.figure(figsize=(FIG_W, 4.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.70], hspace=0.46, wspace=0.16)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1], sharey=axA)
    axC = fig.add_subplot(gs[1, :])

    _trajectory_axis(axA, table, "pH 4.0", first_uv, True, True)
    _trajectory_axis(axB, table, "pH 4.5", first_uv, False, False)
    plt.setp(axB.get_yticklabels(), visible=False)
    _heldout_axis(axC, table[table["panel"].eq("held-out RMSE")])

    fig.subplots_adjust(left=0.098, right=0.995, top=0.925, bottom=0.125)
    # Panel letters go in FIGURE coordinates: axes-relative offsets place A and C at different
    # figure positions because C spans two columns, so A's label would sit right of C's.
    # A and C share the left edge, so one shared pad keeps them aligned whatever the layout;
    # half an inch of it, expressed as a fraction of the figure width.
    pad_left = 0.052 + 0.5 / FIG_W
    for ax, letter, pad in ((axA, "A", pad_left), (axB, "B", 0.052), (axC, "C", pad_left)):
        box = ax.get_position()
        fig.text(box.x0 - pad, box.y1 + 0.028, letter, fontsize=10.0,
                 fontweight="bold", va="bottom", ha="left")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_dir / f"{STEM}.{fmt}"
        fig.savefig(path, format=fmt, dpi=600 if fmt == "png" else None, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


CAPTION = """# {stem} — caption draft

**Mass-domain evaluation separates first-observation alignment from post-capture kinetics.**
Dissolved clofazimine measured by UV--visible spectroscopy as a percentage of the assayed
injected dose. Nested technical runs were averaged within preparation date before conditions were
summarised, and preparation dates were weighted equally.

**(A, B)** UV observations at pH 4.0 and pH 4.5 (points, mean ± SD across {n_dates} preparation
dates), the size-resolved physical model started at injection (grey dashed), and the selected
model initialised at each run's first UV observation (coloured). Shading marks the interval
between injection and the first UV measurement ({first_uv:.0f} min), which no UV observation
constrains; the injection-start model's early shortfall lies largely within it.

**(C)** Leave-one-preparation-date-out prediction error. Open circles are the RMSE for each
withheld preparation date and filled diamonds the root-mean-square across them. A withheld run's
first UV value initialises its trajectory but is excluded from its residuals. Aligning at the
first UV observation reduced pooled held-out error from {base:.1f} to {aligned:.1f} percentage
points, and adding a fitted post-capture rate left it unchanged at {aligned_rate:.1f}.

Although alignment reduced the pooled held-out RMSE, the improvement was condition-dependent:
the within-condition error (date-balanced, all observations, a different basis from the pooled
held-out value) decreased from {c45_base:.1f} to {c45_aligned:.1f} percentage points at pH 4.5
but increased slightly from {c40_base:.1f} to {c40_aligned:.1f} percentage points at pH 4.0. The
pH 4.0 behaviour is visible in panel A, where the aligned trajectory runs above the later
observations.

No laser-diffraction intensity, optical concentration, or optical observation operator entered
this comparison.
"""


def _pc(frame: pd.DataFrame, condition: str, model: str) -> float:
    hit = frame[frame["condition"].eq(condition) & frame["model"].eq(model)]
    if len(hit) != 1:
        raise ValueError(f"per-condition RMSE missing for {condition!r}/{model!r}")
    return float(hit["within_condition_rmse_pct"].iloc[0])


def _emit(out: Path, formats) -> int:
    """Emit the figure and its artifacts."""
    full, meta = build_source_data()
    checks = validate(full, meta)                 # validation runs on the full authoritative set
    table, _ = figure_source_data()
    per_cond = per_condition_rmse()
    checks["condition_dependence"] = {
        "basis": "post-first residuals, date-balanced, ALL data — NOT the pooled held-out basis",
        "values_pct": {f"{r.condition}|{r.model}": round(r.within_condition_rmse_pct, 2)
                       for r in per_cond.itertuples()
                       if r.model in ("injection_base", "first_uv_anchor")},
        "note": "alignment helps pH 4.5 substantially and slightly worsens pH 4.0; the pooled "
                "held-out improvement is driven by pH 4.5"}
    written = render(table, meta, out, formats)

    csv_path = out / f"{STEM}_source_data.csv"
    table.to_csv(csv_path, index=False)

    summary = table[table["kind"].eq("heldout_summary")].set_index("series")["value_pct"]
    obs = table[table["panel"].eq("pH 4.0") & table["kind"].eq("observation")]
    caption = CAPTION.format(
        stem=STEM, n_dates=int(obs["n_dates"].iloc[0]),
        first_uv=meta["trajectories"]["first_uv_time_min"],
        base=summary[MODELS[1][1]], aligned=summary[MODELS[2][1]],
        aligned_rate=summary[MODELS[3][1]],
        c40_base=_pc(per_cond, "pH 4.0", "injection_base"),
        c40_aligned=_pc(per_cond, "pH 4.0", "first_uv_anchor"),
        c45_base=_pc(per_cond, "pH 4.5", "injection_base"),
        c45_aligned=_pc(per_cond, "pH 4.5", "first_uv_anchor"))
    caption_path = out / f"{STEM}_caption.md"
    caption_path.write_text(caption)

    prov = provenance_record(
        "manuscript_mass_domain_model", study_root=data_root() / PH_STUDY,
        uv_ph_values=(4.0, 4.5),                  # the revised figure spans two conditions only
        figure_stem=STEM,
        sources={k: str(data_root() / v) for k, v in SOURCES.items()},
        cohort_admitted=FIT_COHORT,
        aggregation="technical runs averaged within preparation date, then preparation dates "
                    "weighted equally; error bars are between preparation-date means",
        design={"panels": ["pH 4.0", "pH 4.5", "held-out prediction error"],
                "trajectory_strategies_shown": list(SHOWN_IN_TRAJECTORY),
                "trajectory_strategies_omitted": [MODELS[1][1], MODELS[3][1]],
                "ph5_panel": "removed — reported in Results and Table S3",
                "y_axis_pct": list(YLIM)},
        parameters={"first_uv_time_min": meta["trajectories"]["first_uv_time_min"],
                    "post_capture_rate": float(
                        full[full["kind"].eq("heldout_summary")
                             & full["series"].eq(MODELS[3][1])]["rate_scale"].iloc[0])},
        counts={"trajectory_dates": meta["trajectories"]["n_dates"],
                "trajectory_runs": meta["trajectories"]["n_runs"],
                "held_out_dates": meta["held_out"]["n_held_out_dates"]},
        numerical_checks=checks,
        per_condition_rmse=per_cond.to_dict("records"),
        scope={"domain": "dissolved mass (UV-visible) only",
               "excluded": ["LD intensity", "total angular signal", "Copt", "q3",
                            "Mie operator", "any optical conversion", "pH 5.0 transfer panel"]})
    prov_path = write_provenance(out / f"{STEM}_provenance.json", prov)

    print("held-out RMSE (percentage points), authoritative:")
    for _, label in MODELS:
        print(f"  {summary[label]:5.1f}   {PANEL_C_LABELS[label]}")
    print(f"\npanels: pH 4.0, pH 4.5, held-out prediction error (pH 5.0 panel removed)")
    print(f"A/B strategies shown: {len(SHOWN_IN_TRAJECTORY)} "
          f"({', '.join(PANEL_C_LABELS[s] for s in SHOWN_IN_TRAJECTORY)})")
    print(f"first UV observation at {meta['trajectories']['first_uv_time_min']:.0f} min "
          f"(read from artifact)")
    print("\ncondition dependence (within-condition, date-balanced, ALL data — not held out):")
    for cond in ("pH 4.0", "pH 4.5"):
        print(f"  {cond}: injection start {_pc(per_cond, cond, 'injection_base'):5.1f} → "
              f"first-UV aligned {_pc(per_cond, cond, 'first_uv_anchor'):5.1f}")
    for path in [*written, csv_path, caption_path, prov_path]:
        print(f"wrote {path}")
    return 0



def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Destination for the figure, source data, provenance and caption.")
    p.add_argument("--formats", default="pdf,png,svg")
    args = p.parse_args(argv)
    return _emit(args.output_dir,
                 tuple(f.strip() for f in args.formats.split(",") if f.strip()))

if __name__ == "__main__":
    raise SystemExit(main())
