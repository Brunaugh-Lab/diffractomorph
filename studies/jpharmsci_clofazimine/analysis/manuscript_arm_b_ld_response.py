"""Manuscript figure — laser-diffraction response to dissolution-medium polysorbate (Arm B).

One scientific purpose: **laser diffraction, on its own, resolves what changing polysorbate in the
dissolution medium does to the particle side.** UV, the forward model, saturation solubility and
the solubility-normalised coefficient are validation and interpretation tools; none of them
appears here, so the LD result stands as the primary observation rather than as corroboration.

  A  Aggregate angular-signal disappearance: ΣI(t) as a percentage of each run's fitted starting
     signal, preparation-level paths under the equal-weight condition mean.
  B  Preparation-level mean relaxation time ⟨t⟩ — the prespecified primary descriptor of that
     decay, not the KWW τ.
  C  Preparation-level fractional Copt loss.
  D  Reliability-gated q3 D50 of the remaining detected particles, at matched Copt loss.

**The claim is narrowed to what four preparations separate.** At 10× CMC the angular signal
disappeared earlier than at either lower level — its four preparation values sit entirely below
both — whereas the 0.5× and 1× clouds overlap and that contrast is not resolved. Fractional
Copt-loss MEANS were ordered across the ladder, but there too the 0.5×–1× preparation clouds
overlap. Overlap is reported as a limit of resolution, not as evidence of no effect, and no
inferential test is drawn.

**Everything the figure draws is optical.** ΣI is particle-side angular scattering carrying the
non-drug Σref floor, Copt is an optical concentration coordinate, and q3 is the instrument's
inverted relative composition of the particles still detected. None of the three is dissolved
drug mass, and the caption says so rather than leaving it to be inferred from the axis labels.

**Preparation is the independent unit.** Four independent suspension preparations per condition,
with nested technical runs inside each. Technical runs are averaged within preparation first and
the four preparations are then weighted equally; every error bar and ribbon is a
between-preparation SD, never a pooled run SD and never a SEM.

**The renderer never re-ingests.** Panel A reads the frozen
:mod:`arm_b_angular_trajectories` export, whose own provenance records the signal definition and
cross-checks it against the published KWW fits. Panels B–D read the published condition and
preparation tables. :func:`build_source_data` assembles the table; :func:`render` consumes only it.

Run with the pipeline venv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import manuscript_style as ms
from arm_b_angular_trajectories import arm_b_root
from arm_b_provenance import provenance_record, write_provenance

STEM = "Figure_medium_polysorbate_ld_response"

CONDITIONS = ("0.5x CMC", "1.0x CMC", "10x CMC")
LABEL = {"0.5x CMC": "0.5× CMC", "1.0x CMC": "1× CMC", "10x CMC": "10× CMC"}
COLOR = dict(zip(CONDITIONS, ms.ORDERED_RAMP))

SOURCES = {
    # panel A — frozen trajectory export (arm_b_angular_trajectories.py)
    "traj_preps": "analysis/angular_trajectories/arm_b_angular_trajectories_preps.csv",
    "traj_conditions": "analysis/angular_trajectories/arm_b_angular_trajectories_conditions.csv",
    "traj_provenance": "analysis/angular_trajectories/provenance.json",
    # panel B — empirical angular KWW
    "kww_by_prep": "analysis/empirical_angular_kww/angular_kww_by_date_condition.csv",
    "kww_by_condition": "analysis/empirical_angular_kww/angular_kww_by_condition.csv",
    # panel C — primary filtered-48 h, pipeline+Copt partition path
    "partition_preps": "analysis/partition/filtered_48h_pipeline_copt/arm_b_partition_preps.csv",
    "partition_conditions":
        "analysis/partition/filtered_48h_pipeline_copt/arm_b_partition_conditions.csv",
    # panel D — reliability-gated q3 at matched Copt loss
    "q3_balanced": "analysis/q3/q3_matched_extent_balanced.csv",
    "q3_runs": "analysis/q3/q3_matched_extent_runs.csv",
}

# Cross-checks against the authoritative tables. Assertions, not drawing inputs.
# Panel B shows the manuscript's PRESPECIFIED primary descriptor, the mean relaxation time <t>,
# not the KWW tau. Both are published in the same table; <t> is the endpoint the analysis nominated
# because it is robust to the additive Sigma-ref offset that the free KWW plateau absorbs.
EXPECTED_MEAN_RELAX = {"0.5x CMC": (2.1384, 0.2999), "1.0x CMC": (2.0222, 0.2634),
                       "10x CMC": (1.4909, 0.0981)}
EXPECTED_COPT_LOSS_PCT = {"0.5x CMC": 57.5, "1.0x CMC": 63.9, "10x CMC": 71.1}
EXPECTED_D50_ENDS_UM = {"0.5x CMC": (3.3317, 3.1342), "1.0x CMC": (3.3937, 3.1616),
                        "10x CMC": (3.3486, 3.1583)}
G_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
DISPLAY_WINDOW_MIN = 8.0     # panel A x limit; the record runs to 20.8 min (see OMITTED_REGION)

AGGREGATION = {
    "level_1": "technical runs averaged within preparation (suspension date)",
    "level_2": "the four preparations per condition weighted EQUALLY",
    "spread": "between-preparation SD (ddof=1); never a pooled run SD and never a SEM",
    "independent_unit": "suspension preparation",
    "technical_runs_are_not_independent_replicates": True,
}

# What was and was not done to the signal. Recorded explicitly because "cleaned / not cleaned" is
# too coarse to be true here: an upward-spike filter WAS applied, a background subtraction was
# NOT, and the two are different operations on different parts of the signal.
SIGNAL_CONDITIONING = {
    "upward_despiking": {
        "applied": True,
        "how": "kinetics.despike_upward on ΣI(t) — the same upward-spike handling psd_angular_fit "
               "used for the published KWW fits",
        "removes": "transient upward excursions of individual frames",
    },
    "background_subtraction": {
        "applied": False,
        "why_not": "ΣI deliberately retains the non-drug Σref floor; the free KWW plateau absorbs "
                   "that additive offset, which is why the mean relaxation time and β are the "
                   "prespecified endpoints. The *_bgsub sensitivity columns are not used.",
        "would_remove": "a constant additive offset across all frames",
    },
    "distinction": "despiking removes transient per-frame excursions; background subtraction "
                   "would remove a constant offset. Applying the first says nothing about the "
                   "second, and this artifact applies only the first.",
    "per_frame_channel_cleaning": {
        "applied": False,
        "note": "the angular sum is taken over all 31 rings of the raw export; no per-channel "
                "noise-surface admission is applied to it",
    },
}

# Interpretation tools that must not enter this figure — the LD result is the primary observation.
FORBIDDEN_FIELDS = ("uv", "pct_injected", "dissolved", "forward", "model", "cs_ugml",
                    "solubility", "k_mean", "k_sd", "k_sem", "log_k", "normalized",
                    "tau_slope", "predicted")
# The coarse tail is outside the established reliable inversion range: not plotted, not discussed.
COARSE_TAIL_FIELDS = ("tail_frac_above_15um", "tail_pct_above_15um", "d90_um", "above_15")

BANNED_PHRASES = (
    "dissolved fraction", "dissolved mass fraction", "particle-mass loss", "mass loss",
    "equivalent", "equivalence", "invariant", "indistinguishable", "identical paths",
    "no difference", "significantly different", "significant difference", "unchanged",
    "validates", "confirms", "proves",
)


def _read(key: str, base: Path) -> pd.DataFrame:
    path = base / SOURCES[key]
    if not path.exists():
        raise FileNotFoundError(
            f"authoritative source '{key}' missing: {path}. This figure does not reconstruct "
            f"analysis outputs — rerun the analysis that writes it "
            f"(panel A comes from arm_b_angular_trajectories.py).")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"authoritative source '{key}' is empty: {path}")
    return frame


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(CONDITIONS) - set(frame["condition"])
    if missing:
        raise ValueError(f"conditions absent from an authoritative table: {sorted(missing)}")
    return frame[frame["condition"].isin(CONDITIONS)]


def _prep_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Equal-weight condition mean and between-preparation SD over preparation-level values."""
    return (frame.groupby("condition", as_index=False)
            .agg(mean=(column, "mean"), sd=(column, "std"), n_preps=(column, "count")))


# ── panel A — aggregate angular-signal trajectories ──────────────────────────────────────────

def _panel_a(base: Path) -> tuple[list[dict], dict]:
    preps = _ordered(_read("traj_preps", base))
    conditions = _ordered(_read("traj_conditions", base))
    prov = json.loads((base / SOURCES["traj_provenance"]).read_text())

    rows = []
    for r in preps.itertuples():
        rows.append({"panel": "A", "kind": "prep_trajectory",
                     "series": f"{LABEL[r.condition]} preparation",
                     "condition": r.condition, "prep": str(r.prep),
                     "x_quantity": "time_min", "x_value": float(r.t_min),
                     "y_quantity": "angular_signal_pct_of_fitted_start",
                     "y_value": float(r.pct_of_i0_fit),
                     "n_technical_reps": int(r.n_technical_reps),
                     "displayed_in_figure": float(r.t_min) <= DISPLAY_WINDOW_MIN,
                     "source_artifact": "arm_b_angular_trajectories_preps.csv",
                     "source_column": "pct_of_i0_fit"})
    for r in conditions.itertuples():
        rows.append({"panel": "A", "kind": "condition_trajectory",
                     "series": LABEL[r.condition], "condition": r.condition,
                     "x_quantity": "time_min", "x_value": float(r.t_min),
                     "y_quantity": "angular_signal_pct_of_fitted_start",
                     "y_value": float(r.pct_mean),
                     "y_sd": float(r.pct_sd_between_preps), "n_preps": int(r.n_preps),
                     "displayed_in_figure": float(r.t_min) <= DISPLAY_WINDOW_MIN,
                     "source_artifact": "arm_b_angular_trajectories_conditions.csv",
                     "source_column": "pct_mean,pct_sd_between_preps"})

    # The acquisition runs to ~21 min, but drawing the full record spends three quarters of the
    # panel on the late region and squeezes the part the claim is about. The omitted region is
    # kept in the source data and its extent is quantified here — as a measured range, not as a
    # claim that it is flat — so the window is a stated display choice, not a silent truncation.
    omitted = conditions[conditions["t_min"] > DISPLAY_WINDOW_MIN]
    omitted_range = {c: round(float(g["pct_mean"].max() - g["pct_mean"].min()), 3)
                     for c, g in omitted.groupby("condition")}
    meta = {"signal_definition": prov["signal_definition"],
            "normalization": prov["normalization"],
            "time_grid_min": prov["time_grid"],
            "aggregation": prov["aggregation"],
            "is_dissolved_mass": False,
            "upstream_export": "arm_b_angular_trajectories.py",
            "upstream_cross_check": prov["numerical_checks"]["reproduces_published_kww_fits"],
            "display_window_min": [0.0, DISPLAY_WINDOW_MIN],
            "acquisition_window_min": [float(conditions["t_min"].min()),
                                       float(conditions["t_min"].max())],
            "omitted_region": {
                "window_min": [DISPLAY_WINDOW_MIN, float(conditions["t_min"].max())],
                "condition_mean_range_pp": omitted_range,
                "max_condition_mean_range_pp": max(omitted_range.values()),
                "described_as": "a maximum condition-mean range over the omitted region — NOT a "
                                "claim that the region is flat",
                "note": "the omitted record is retained in the source data with "
                        "displayed_in_figure = False"},
            "sd_band_drawn": False,
            "sd_band_reason": "the four preparation-level paths already show the spread directly; "
                              "a band over three overlapping conditions obscured them. The "
                              "between-preparation SD is retained in the source data.",
            "n_preps_per_condition": {c: int(preps[preps["condition"].eq(c)]["prep"].nunique())
                                      for c in CONDITIONS}}
    return rows, meta


# ── panel B — mean relaxation time <t> ───────────────────────────────────────────────────────

def _panel_b(base: Path) -> tuple[list[dict], dict]:
    """The manuscript's prespecified primary descriptor of the angular decay.

    ``mean_relax_min`` is the mean relaxation time <t> of the KWW fit, not the KWW time constant
    tau. The upstream analysis nominated <t> (with beta) as the primary endpoint because both are
    robust to the additive Sigma-ref offset the free plateau absorbs, whereas tau is not; the
    figure therefore shows <t>. tau remains in the same published table and is not drawn here.
    """
    by_prep = _ordered(_read("kww_by_prep", base))
    by_condition = _ordered(_read("kww_by_condition", base)).set_index("condition")

    rows = []
    for r in by_prep.itertuples():
        rows.append({"panel": "B", "kind": "prep_point", "series": LABEL[r.condition],
                     "condition": r.condition, "prep": str(r.date),
                     "x_quantity": "condition", "x_value": CONDITIONS.index(r.condition),
                     "y_quantity": "mean_relaxation_time_min",
                     "y_value": float(r.mean_relax_min),
                     "n_technical_reps": int(r.n_runs),
                     "source_artifact": "angular_kww_by_date_condition.csv",
                     "source_column": "mean_relax_min"})
    for condition in CONDITIONS:
        row = by_condition.loc[condition]
        rows.append({"panel": "B", "kind": "condition_mean", "series": LABEL[condition],
                     "condition": condition, "x_quantity": "condition",
                     "x_value": CONDITIONS.index(condition),
                     "y_quantity": "mean_relaxation_time_min",
                     "y_value": float(row["mean_relax_min_mean"]),
                     "y_sd": float(row["mean_relax_min_sd"]),
                     "n_preps": int(row["n_days"]), "n_technical_reps": int(row["n_runs"]),
                     "source_artifact": "angular_kww_by_condition.csv",
                     "source_column": "mean_relax_min_mean,mean_relax_min_sd"})

    derived = _prep_summary(by_prep.rename(columns={"mean_relax_min": "v"}),
                            "v").set_index("condition")
    spans = {c: (round(float(by_prep[by_prep["condition"].eq(c)]["mean_relax_min"].min()), 4),
                 round(float(by_prep[by_prep["condition"].eq(c)]["mean_relax_min"].max()), 4))
             for c in CONDITIONS}
    meta = {"descriptor": "mean relaxation time <t> of the angular KWW decay",
            "descriptor_is_prespecified_primary": True,
            "kww_tau_not_drawn": "tau is published in the same table and is not shown; <t> and "
                                 "beta are the endpoints the upstream analysis nominated, because "
                                 "they are robust to the additive Sigma-ref offset",
            "mean_relax_min": {c: round(float(by_condition.loc[c, "mean_relax_min_mean"]), 4)
                               for c in CONDITIONS},
            "mean_relax_sd_between_preps": {
                c: round(float(by_condition.loc[c, "mean_relax_min_sd"]), 4) for c in CONDITIONS},
            "prep_value_range_min": spans,
            "separation": _separation(spans),
            "n_preps": {c: int(by_condition.loc[c, "n_days"]) for c in CONDITIONS},
            "n_runs": {c: int(by_condition.loc[c, "n_runs"]) for c in CONDITIONS},
            "condition_summary_is_prep_first": {
                c: {"from_prep_values": round(float(derived.loc[c, "mean"]), 4),
                    "published": round(float(by_condition.loc[c, "mean_relax_min_mean"]), 4)}
                for c in CONDITIONS},
            "inferential_test_shown": False,
            "quantity": "mean relaxation time of the aggregate angular signal — an optical "
                        "relaxation, not a dissolution rate constant"}
    return rows, meta


def _separation(spans: dict) -> dict:
    """Which condition pairs have non-overlapping preparation clouds, and which do not.

    With four preparations and no inferential test, an overlap of the preparation ranges is the
    honest statement of what the design resolves. This is descriptive: a separation is not a
    significance claim, and an overlap is not evidence of no effect.
    """
    pairs = {}
    for i, a in enumerate(CONDITIONS):
        for b in CONDITIONS[i + 1:]:
            lo_a, hi_a = spans[a]
            lo_b, hi_b = spans[b]
            pairs[f"{a} vs {b}"] = {
                "ranges": {a: [lo_a, hi_a], b: [lo_b, hi_b]},
                "preparation_clouds_overlap": bool(lo_a <= hi_b and lo_b <= hi_a),
                "contrast_resolved_by_this_design": not bool(lo_a <= hi_b and lo_b <= hi_a)}
    return {"pairs": pairs,
            "basis": "overlap of the four preparation-level values, not an inferential test",
            "caveat": "a non-overlap is a descriptive separation, not a significance claim; an "
                      "overlap is not evidence that the levels do not differ"}


# ── panel C — fractional Copt loss ───────────────────────────────────────────────────────────

def _panel_c(base: Path) -> tuple[list[dict], dict]:
    preps = _ordered(_read("partition_preps", base))
    conditions = _ordered(_read("partition_conditions", base)).set_index("condition")
    preps = preps.assign(copt_loss_pct=preps["copt_frac_loss"] * 100.0)
    summary = _prep_summary(preps, "copt_loss_pct").set_index("condition")

    rows = []
    for r in preps.itertuples():
        rows.append({"panel": "C", "kind": "prep_point", "series": LABEL[r.condition],
                     "condition": r.condition, "prep": str(r.prep),
                     "x_quantity": "condition", "x_value": CONDITIONS.index(r.condition),
                     "y_quantity": "copt_fractional_loss_pct",
                     "y_value": float(r.copt_loss_pct),
                     "n_technical_reps": int(r.n_technical_reps),
                     "source_artifact": "arm_b_partition_preps.csv",
                     "source_column": "copt_frac_loss (x100)"})
    for condition in CONDITIONS:
        rows.append({"panel": "C", "kind": "condition_mean", "series": LABEL[condition],
                     "condition": condition, "x_quantity": "condition",
                     "x_value": CONDITIONS.index(condition),
                     "y_quantity": "copt_fractional_loss_pct",
                     "y_value": float(summary.loc[condition, "mean"]),
                     "y_sd": float(summary.loc[condition, "sd"]),
                     "n_preps": int(summary.loc[condition, "n_preps"]),
                     "source_artifact": "arm_b_partition_preps.csv",
                     "source_column": "copt_frac_loss (x100)"})

    spans = {c: (round(float(preps[preps["condition"].eq(c)]["copt_loss_pct"].min()), 2),
                 round(float(preps[preps["condition"].eq(c)]["copt_loss_pct"].max()), 2))
             for c in CONDITIONS}
    meta = {"copt_loss_pct": {c: round(float(summary.loc[c, "mean"]), 4) for c in CONDITIONS},
            "prep_value_range_pct": spans,
            "separation": _separation(spans),
            "means_ordered_across_ladder": True,
            "copt_loss_sd_between_preps_pp": {c: round(float(summary.loc[c, "sd"]), 4)
                                              for c in CONDITIONS},
            "n_preps": {c: int(summary.loc[c, "n_preps"]) for c in CONDITIONS},
            "published_condition_value_pct": {
                c: round(float(conditions.loc[c, "copt_frac_loss"]) * 100.0, 4)
                for c in CONDITIONS},
            "partition_path": "filtered_48h_pipeline_copt (the primary path)",
            "quantity": "fractional loss of optical concentration — NOT a dissolved fraction "
                        "and NOT a particle-mass loss"}
    return rows, meta


# ── panel D — reliability-gated q3 at matched Copt loss ──────────────────────────────────────

def _panel_d(base: Path) -> tuple[list[dict], dict]:
    runs = _ordered(_read("q3_runs", base))
    balanced = _ordered(_read("q3_balanced", base))
    # preparation-level paths: technical runs averaged within preparation, as upstream does
    preps = (runs.groupby(["condition", "unit", "extent_g"], as_index=False)
             .agg(d50_um=("d50_um", "mean"), n_technical_reps=("run_id", "nunique")))

    rows = []
    for r in preps.itertuples():
        rows.append({"panel": "D", "kind": "prep_path",
                     "series": f"{LABEL[r.condition]} preparation",
                     "condition": r.condition, "prep": str(r.unit),
                     "x_quantity": "fraction_of_copt_lost_g", "x_value": float(r.extent_g),
                     "y_quantity": "q3_d50_um", "y_value": float(r.d50_um),
                     "n_technical_reps": int(r.n_technical_reps),
                     "source_artifact": "q3_matched_extent_runs.csv", "source_column": "d50_um"})
    for r in balanced.itertuples():
        rows.append({"panel": "D", "kind": "condition_path", "series": LABEL[r.condition],
                     "condition": r.condition, "x_quantity": "fraction_of_copt_lost_g",
                     "x_value": float(r.extent_g), "y_quantity": "q3_d50_um",
                     "y_value": float(r.d50_um_mean), "y_sd": float(r.d50_um_sd),
                     "n_preps": int(r.n_units),
                     "source_artifact": "q3_matched_extent_balanced.csv",
                     "source_column": "d50_um_mean,d50_um_sd"})

    ends = {}
    for condition in CONDITIONS:
        sub = balanced[balanced["condition"].eq(condition)].set_index("extent_g")
        ends[condition] = (round(float(sub.loc[0.2, "d50_um_mean"]), 4),
                           round(float(sub.loc[0.8, "d50_um_mean"]), 4))
    meta = {"extent_grid_g": list(G_GRID),
            "g_definition": "fraction of Copt lost — g = 0.2 is 20 % optical loss, "
                            "g = 0.8 is 80 %",
            "d50_um_at_g_0_2_and_0_8": ends,
            "n_preps": {c: int(preps[preps["condition"].eq(c)]["unit"].nunique())
                        for c in CONDITIONS},
            "coarse_tail_displayed": False,
            "coarse_tail_reason": "the >15 µm tail is outside the established reliable inversion "
                                  "range, so it is neither plotted nor interpreted",
            "verdict_issued": False,
            "quantity": "q3 D50 — the instrument-inverted RELATIVE composition of the particles "
                        "still detected; not particle mass and not an independent modality"}
    return rows, meta


def build_source_data(base: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Every plotted value, from the frozen trajectory export and the published Arm B tables."""
    base = Path(base or arm_b_root())
    rows_a, meta_a = _panel_a(base)
    rows_b, meta_b = _panel_b(base)
    rows_c, meta_c = _panel_c(base)
    rows_d, meta_d = _panel_d(base)
    table = pd.DataFrame(rows_a + rows_b + rows_c + rows_d)
    table["study"] = "micelle_effects_tween80_arm_b"
    meta = {"panel_a": meta_a, "panel_b": meta_b, "panel_c": meta_c, "panel_d": meta_d,
            "aggregation": AGGREGATION, "conditions": list(CONDITIONS)}
    return table, meta


# ── validation ───────────────────────────────────────────────────────────────────────────────

def validate(table: pd.DataFrame, meta: dict, base: Path | None = None) -> dict:
    base = Path(base or arm_b_root())
    checks: dict = {}

    # 1 ── four independent preparations per condition, technical runs nested inside.
    preps = {}
    for panel, kind in (("A", "prep_trajectory"), ("B", "prep_point"),
                        ("C", "prep_point"), ("D", "prep_path")):
        sub = table[table["panel"].eq(panel) & table["kind"].eq(kind)]
        preps[panel] = {c: int(sub[sub["condition"].eq(c)]["prep"].nunique()) for c in CONDITIONS}
    checks["four_preparations_per_condition"] = {
        "per_panel": preps, "independent_unit": AGGREGATION["independent_unit"],
        "technical_runs_are_not_independent_replicates": True,
        "ok": all(set(v.values()) == {4} for v in preps.values())}
    if not checks["four_preparations_per_condition"]["ok"]:
        raise ValueError(f"every panel must rest on four preparations per condition: {preps}")

    # 2 ── every drawn spread is a BETWEEN-PREPARATION SD.
    spreads = table[table["y_sd"].notna()] if "y_sd" in table else table.iloc[0:0]
    summary_kinds = {"condition_trajectory", "condition_mean", "condition_path"}
    checks["spread_is_between_preparation_sd"] = {
        "kinds_carrying_sd": sorted(set(spreads["kind"])),
        "n_preps_behind_each_sd": sorted(set(int(v) for v in spreads["n_preps"].dropna())),
        "definition": AGGREGATION["spread"],
        "ok": set(spreads["kind"]) <= summary_kinds
              and set(int(v) for v in spreads["n_preps"].dropna()) == {4}}
    if not checks["spread_is_between_preparation_sd"]["ok"]:
        raise ValueError("a drawn spread is not a between-preparation SD over four preparations")

    # 3 ── panel A reproduces the frozen export, prep-first.
    prep_traj = table[table["kind"].eq("prep_trajectory")]
    cond_traj = table[table["kind"].eq("condition_trajectory")]
    derived = (prep_traj.groupby(["condition", "x_value"], as_index=False)
               .agg(mean=("y_value", "mean"), sd=("y_value", "std")))
    merged = cond_traj.merge(derived, on=["condition", "x_value"])
    checks["panel_a_equal_weight_over_preparations"] = {
        "n_grid_points": int(cond_traj["x_value"].nunique()),
        "max_abs_mean_deviation_pp": float((merged["y_value"] - merged["mean"]).abs().max()),
        "max_abs_sd_deviation_pp": float((merged["y_sd"] - merged["sd"]).abs().max()),
        "upstream_cross_check": meta["panel_a"]["upstream_cross_check"],
        "normalization": meta["panel_a"]["normalization"],
        "is_dissolved_mass": False}
    pa = checks["panel_a_equal_weight_over_preparations"]
    pa["ok"] = (pa["max_abs_mean_deviation_pp"] < 1e-9 and pa["max_abs_sd_deviation_pp"] < 1e-9
                and len(merged) == len(cond_traj)
                and meta["panel_a"]["upstream_cross_check"]["ok"])
    if not pa["ok"]:
        raise ValueError(f"the condition trajectory is not the equal-weight mean of the four "
                         f"preparation trajectories: {pa}")

    # 4 ── panels B, C, D reproduce the published endpoints exactly.
    endpoints = {}
    for condition in CONDITIONS:
        relax_mean, relax_sd = EXPECTED_MEAN_RELAX[condition]
        b = table[table["panel"].eq("B") & table["kind"].eq("condition_mean")
                  & table["condition"].eq(condition)]
        c = table[table["panel"].eq("C") & table["kind"].eq("condition_mean")
                  & table["condition"].eq(condition)]
        d_lo, d_hi = EXPECTED_D50_ENDS_UM[condition]
        got = meta["panel_d"]["d50_um_at_g_0_2_and_0_8"][condition]
        endpoints[condition] = {
            "mean_relax_min": round(float(b["y_value"].iloc[0]), 4),
            "mean_relax_min_expected": relax_mean,
            "mean_relax_sd_min": round(float(b["y_sd"].iloc[0]), 4),
            "mean_relax_sd_expected": relax_sd,
            "copt_loss_pct": round(float(c["y_value"].iloc[0]), 1),
            "copt_loss_pct_expected": EXPECTED_COPT_LOSS_PCT[condition],
            "d50_um_g0.2": got[0], "d50_um_g0.2_expected": d_lo,
            "d50_um_g0.8": got[1], "d50_um_g0.8_expected": d_hi}
        e = endpoints[condition]
        e["ok"] = (e["mean_relax_min"] == relax_mean and e["mean_relax_sd_min"] == relax_sd
                   and e["copt_loss_pct"] == EXPECTED_COPT_LOSS_PCT[condition]
                   and e["d50_um_g0.2"] == d_lo and e["d50_um_g0.8"] == d_hi)
    checks["published_endpoints"] = endpoints
    bad = [c for c, v in endpoints.items() if not v["ok"]]
    if bad:
        raise ValueError(f"displayed endpoints do not reproduce the authoritative tables for "
                         f"{bad}: { {c: endpoints[c] for c in bad} }")

    # Panel C's preparation-first mean must also match the published condition table.
    published = meta["panel_c"]["published_condition_value_pct"]
    drift = {c: abs(meta["panel_c"]["copt_loss_pct"][c] - published[c]) for c in CONDITIONS}
    checks["panel_c_matches_published_condition_table"] = {
        "prep_first_mean_pct": meta["panel_c"]["copt_loss_pct"],
        "published_pct": published, "max_abs_deviation_pp": max(drift.values()),
        "ok": max(drift.values()) < 1e-6}
    if not checks["panel_c_matches_published_condition_table"]["ok"]:
        raise ValueError("the preparation-first Copt-loss mean disagrees with the published "
                         "condition table")

    # Panel B's published summary must itself be preparation-first.
    b_meta = meta["panel_b"]["condition_summary_is_prep_first"]
    checks["panel_b_summary_is_prep_first"] = {
        "per_condition": b_meta,
        "max_abs_deviation_min": max(abs(v["from_prep_values"] - v["published"])
                                     for v in b_meta.values()),
        "ok": all(abs(v["from_prep_values"] - v["published"]) < 5e-4 for v in b_meta.values())}
    if not checks["panel_b_summary_is_prep_first"]["ok"]:
        raise ValueError("the published mean-relaxation summary is not the equal-weight mean of "
                         "the four preparation values")

    # Panel B must show the prespecified primary descriptor, not the KWW tau.
    checks["panel_b_shows_the_prespecified_descriptor"] = {
        "descriptor": meta["panel_b"]["descriptor"],
        "is_prespecified_primary": meta["panel_b"]["descriptor_is_prespecified_primary"],
        "source_columns": sorted(set(table[table["panel"].eq("B")]["source_column"])),
        "y_quantity": sorted(set(table[table["panel"].eq("B")]["y_quantity"])),
        "ok": (set(table[table["panel"].eq("B")]["y_quantity"]) == {"mean_relaxation_time_min"}
               and not any("tau" in c for c in table[table["panel"].eq("B")]["source_column"]))}
    if not checks["panel_b_shows_the_prespecified_descriptor"]["ok"]:
        raise ValueError("panel B must draw mean_relax_min, the prespecified primary descriptor, "
                         "not the KWW tau")

    # 5 ── the NARROWED claim: only what the preparation-level evidence separates.
    relax = [meta["panel_b"]["mean_relax_min"][c] for c in CONDITIONS]
    loss = [meta["panel_c"]["copt_loss_pct"][c] for c in CONDITIONS]
    relax_pairs = meta["panel_b"]["separation"]["pairs"]
    loss_pairs = meta["panel_c"]["separation"]["pairs"]
    ten_vs_lower = ["0.5x CMC vs 10x CMC", "1.0x CMC vs 10x CMC"]
    checks["supported_contrasts"] = {
        "mean_relax_min": relax, "copt_loss_pct": loss,
        "relaxation_10x_separated_from_both_lower_levels":
            all(relax_pairs[k]["contrast_resolved_by_this_design"] for k in ten_vs_lower),
        "relaxation_lower_two_overlap":
            relax_pairs["0.5x CMC vs 1.0x CMC"]["preparation_clouds_overlap"],
        "copt_loss_means_ordered": bool(np.all(np.diff(loss) > 0)),
        "copt_loss_lower_two_overlap":
            loss_pairs["0.5x CMC vs 1.0x CMC"]["preparation_clouds_overlap"],
        "copt_loss_pairs": {k: v["contrast_resolved_by_this_design"]
                            for k, v in loss_pairs.items()},
        "q3_verdict_issued": meta["panel_d"]["verdict_issued"],
        "claim_scope": "10x CMC produced earlier angular-signal disappearance than either lower "
                       "level; Copt-loss MEANS were ordered across the ladder, but the 0.5x-1x "
                       "preparation clouds overlapped and that contrast is not resolved"}
    sc = checks["supported_contrasts"]
    if not (sc["relaxation_10x_separated_from_both_lower_levels"]
            and sc["copt_loss_means_ordered"] and sc["copt_loss_lower_two_overlap"]):
        raise ValueError(f"the narrowed claim failed against the artifacts: {sc}")

    # 6 ── scope: no UV / model / solubility / normalized-rate field, no coarse tail.
    blob = " ".join(str(v) for v in table.to_dict("list").values()).lower()
    columns = " ".join(table.columns).lower()
    found_forbidden = [f for f in FORBIDDEN_FIELDS if f in blob or f in columns]
    found_tail = [f for f in COARSE_TAIL_FIELDS if f in blob or f in columns]
    q3 = table[table["y_quantity"].eq("q3_d50_um")]["y_value"]
    checks["scope"] = {
        "y_quantities": sorted(set(table["y_quantity"])),
        "forbidden_fields_found": found_forbidden,
        "coarse_tail_fields_found": found_tail,
        "max_plotted_q3_um": round(float(q3.max()), 4),
        "coarse_tail_boundary_um": 15.0,
        "all_plotted_sizes_below_boundary": bool(q3.max() < 15.0),
        "ok": (not found_forbidden and not found_tail and bool(q3.max() < 15.0))}
    if not checks["scope"]["ok"]:
        raise ValueError(f"a validation/interpretation field or the coarse tail reached the "
                         f"figure: {checks['scope']}")

    # 7 ── axis limits.
    clipping = {}
    for panel, (lo, hi) in {"A": _limits_a(table), "B": _limits_b(table),
                            "C": _limits_c(table), "D": _limits_d(table)}.items():
        drawn = table[table["panel"].eq(panel)]
        sd = drawn["y_sd"].fillna(0.0)
        low, high = float((drawn["y_value"] - sd).min()), float((drawn["y_value"] + sd).max())
        clipping[panel] = {"y_limits": [round(lo, 3), round(hi, 3)], "min_drawn": round(low, 3),
                           "max_drawn": round(high, 3), "ok": bool(low >= lo and high <= hi)}
    checks["axis_limits_do_not_clip"] = clipping
    bad = [p for p, v in clipping.items() if not v["ok"]]
    if bad:
        raise ValueError(f"a panel's y limits clip a drawn value or error bar: {bad}")
    return checks


# ── rendering ────────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.lines import Line2D                                        # noqa: E402

JITTER = 0.085         # preparation points fan out around their condition position
MEAN_DX = 0.21         # condition mean sits clear of its preparation points


def _pad(values, frac_lo=0.08, frac_hi=0.08):
    lo, hi = float(min(values)), float(max(values))
    span = hi - lo
    return lo - frac_lo * span, hi + frac_hi * span


def _extremes(panel: pd.DataFrame) -> list[float]:
    sd = panel["y_sd"].fillna(0.0)
    return (panel["y_value"] - sd).tolist() + (panel["y_value"] + sd).tolist()


def _limits_a(table):
    a = table[table["panel"].eq("A")]
    return _pad(_extremes(a[a["displayed_in_figure"].fillna(True).astype(bool)]), 0.05, 0.05)


def _limits_b(table):
    return _pad(_extremes(table[table["panel"].eq("B")]), 0.12, 0.12)


def _limits_c(table):
    return _pad(_extremes(table[table["panel"].eq("C")]), 0.12, 0.12)


def _limits_d(table):
    return _pad(_extremes(table[table["panel"].eq("D")]), 0.10, 0.10)


def _axis_a(ax, table):
    a = table[table["panel"].eq("A")]
    a = a[a["displayed_in_figure"].fillna(True).astype(bool)]
    for condition in CONDITIONS:
        color = COLOR[condition]
        for _, g in a[a["kind"].eq("prep_trajectory")
                      & a["condition"].eq(condition)].groupby("prep"):
            g = g.sort_values("x_value")
            ax.plot(g["x_value"], g["y_value"], "-", lw=0.55, color=color, alpha=0.55, zorder=2)
    for condition in CONDITIONS:
        g = a[a["kind"].eq("condition_trajectory")
              & a["condition"].eq(condition)].sort_values("x_value")
        ax.plot(g["x_value"], g["y_value"], "-", lw=1.7, color=COLOR[condition], zorder=4,
                solid_capstyle="round", label=LABEL[condition])

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Angular signal remaining\n(% of fitted start)")
    ax.set_title("Earliest disappearance at 10× CMC", pad=4)
    ax.set_xlim(-0.25, DISPLAY_WINDOW_MIN + 0.25)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_ylim(*_limits_a(table))
    ms.clean_axes(ax)
    ax.legend(loc="upper right", frameon=False, handlelength=1.5, handletextpad=0.5,
              borderpad=0.0, labelspacing=0.3)


def _category_axis(ax, table, panel, ylabel, title, fmt):
    p = table[table["panel"].eq(panel)]
    for condition in CONDITIONS:
        x0 = CONDITIONS.index(condition)
        color = COLOR[condition]
        points = p[p["kind"].eq("prep_point") & p["condition"].eq(condition)].sort_values("prep")
        offsets = np.linspace(-JITTER, JITTER, len(points))
        ax.plot(x0 - MEAN_DX + offsets, points["y_value"], "o", ms=ms.MS_UNIT, mfc="white",
                mec=color, mew=0.9, ls="none", zorder=3)
        mean = p[p["kind"].eq("condition_mean") & p["condition"].eq(condition)]
        ax.errorbar([x0 + MEAN_DX], mean["y_value"], yerr=mean["y_sd"], fmt="o",
                    ms=ms.MS_MEAN, color=color, ecolor=color, elinewidth=ms.ELINEWIDTH,
                    capsize=ms.CAPSIZE, capthick=ms.ELINEWIDTH, markeredgecolor="white",
                    markeredgewidth=0.5, zorder=5)
        ax.annotate(fmt.format(float(mean["y_value"].iloc[0])),
                    xy=(x0 + MEAN_DX, float(mean["y_value"].iloc[0])
                        + float(mean["y_sd"].iloc[0])),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.4, color=color)

    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([LABEL[c] for c in CONDITIONS])
    ax.set_xlim(-0.75, len(CONDITIONS) - 0.25)
    ax.set_xlabel("Polysorbate 80 in dissolution medium")
    ms.clean_axes(ax)


def _axis_d(ax, table):
    d = table[table["panel"].eq("D")]
    for condition in CONDITIONS:
        color = COLOR[condition]
        for _, g in d[d["kind"].eq("prep_path") & d["condition"].eq(condition)].groupby("prep"):
            g = g.sort_values("x_value")
            ax.plot(g["x_value"], g["y_value"], "-", lw=0.5, color=color, alpha=0.38, zorder=2)
    # The three conditions sit almost on top of one another, so the between-preparation SD is
    # drawn as bars nudged apart in x. Three translucent bands of one hue cannot be told apart,
    # and the reader has to see that the spread is wide relative to the separation.
    for i, condition in enumerate(CONDITIONS):
        g = d[d["kind"].eq("condition_path")
              & d["condition"].eq(condition)].sort_values("x_value")
        dx = (i - 1) * 0.016
        ax.errorbar(g["x_value"] + dx, g["y_value"], yerr=g["y_sd"], fmt="none",
                    ecolor=COLOR[condition], elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=4)
        ax.plot(g["x_value"] + dx, g["y_value"], "o-", ms=3.2, lw=1.6, color=COLOR[condition],
                markeredgecolor="white", markeredgewidth=0.4, zorder=5, label=LABEL[condition])

    ax.set_xlabel("Fraction of Copt lost, g")
    ax.set_ylabel("q3 D50 of remaining\ndetected particles (µm)")
    ax.set_title("Size paths at matched Copt loss", pad=4)
    ax.set_xlim(0.15, 0.85)
    ax.set_xticks(list(G_GRID))
    ax.set_ylim(*_limits_d(table))
    ms.clean_axes(ax)
    ax.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.4, handletextpad=0.4,
              borderpad=0.0, labelspacing=0.3)


def render(table: pd.DataFrame, meta: dict, out_dir: Path, formats=("pdf", "png", "svg")):
    ms.apply_style()
    fig = plt.figure(figsize=(ms.FIG_W, 5.05))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.995, top=0.930, bottom=0.083,
                          wspace=0.34, hspace=0.46)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    _axis_a(axA, table)
    _category_axis(axB, table, "B", r"Mean relaxation time, $\langle t \rangle$ (min)",
                   "Shortest relaxation at 10× CMC", "{:.2f}")
    _category_axis(axC, table, "C", "Fractional Copt loss (%)",
                   "Copt loss means ordered", "{:.1f}")
    _axis_d(axD, table)
    ms.panel_tags(fig, ((axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")))
    return ms.save(fig, out_dir, STEM, formats)


CAPTION = """# {stem} — caption draft

**Raising dissolution-medium polysorbate to {c3} the working critical micelle concentration
produced earlier aggregate angular-signal disappearance than either lower level; fractional
Copt-loss means were ordered across the ladder; and the reliability-restricted q3 D50 paths of the
remaining detected particles overlapped descriptively at matched Copt loss.** Clofazimine was
dispersed into dissolution media containing polysorbate 80 at {c1}, {c2} and {c3} the working
critical micelle concentration, with **four independent suspension preparations per condition** and
nested technical runs within each preparation. Technical runs were averaged within preparation
before the four preparations were weighted equally; **every error bar is a between-preparation
SD**, never a pooled run SD and never a standard error. Every quantity shown is a
laser-diffraction observation: **the angular signal and Copt are optical coordinates, not dissolved
drug mass.**

**(A)** Total measured angular signal ΣI(t) as a percentage of each run's fitted, back-extrapolated
starting signal. Thin lines are the {n_preps} preparation-level trajectories per condition after
technical-run averaging; heavy lines are the equal-weight condition means. ΣI is the raw per-frame
sum over the 31 detector rings, upward-despiked but **not** background-subtracted, so it retains
the non-drug Σref floor and is particle-side scattering rather than undissolved mass; the curve
starts below 100 % because acquisition begins after injection and the starting value is the fitted
back-extrapolation to t = 0. Acquisition continued to {t_end:.1f} min; the first {t_win:.0f} min
are shown, and over the omitted {t_win:.0f}–{t_end:.1f} min region each condition mean spans at
most {tail_span:.1f} percentage points.

**(B)** Mean relaxation time ⟨t⟩ of that angular decay, the prespecified primary descriptor of the
stretched-exponential fit. Open symbols are the four preparation-level values, filled symbols the
condition mean {pm} between-preparation SD: {t1:.2f} {pm} {s1:.2f}, {t2:.2f} {pm} {s2:.2f} and
{t3:.2f} {pm} {s3:.2f} min at {c1}, {c2} and {c3} CMC. The four {c3} preparations
({r3lo:.2f}–{r3hi:.2f} min) lie entirely below both lower levels ({r1lo:.2f}–{r1hi:.2f} and
{r2lo:.2f}–{r2hi:.2f} min), so that is the contrast this design separates; the {c1} and {c2}
preparation clouds overlap and that contrast is not resolved here. No inferential test is applied,
and an overlap is a limit of resolution rather than evidence that the levels do not differ.

**(C)** Fractional loss of optical concentration over the run, from the primary filtered-48 h
pipeline-plus-Copt path. Condition means were ordered across the ladder — {l1:.1f}, {l2:.1f} and
{l3:.1f} % — but the {c1} and {c2} preparation clouds ({p1lo:.1f}–{p1hi:.1f} and
{p2lo:.1f}–{p2hi:.1f} %) overlap, so that contrast is not resolved; the {c2} and {c3} clouds
overlap as well. This is fractional Copt loss — an optical extent — and is not a dissolved fraction
or a particle-mass loss.

**(D)** q3 D50 of the particles still detected, against the fraction of Copt lost, g, so that
conditions are compared at matched optical loss rather than at matched clock time (g = 0.2 is 20 %
loss, g = 0.8 is 80 %). Thin lines are preparation-level paths and heavy symbols the condition mean
{pm} between-preparation SD. Across g = 0.2 to 0.8 the condition means moved from {d1lo:.2f} to
{d1hi:.2f} {um} at {c1} CMC, {d2lo:.2f} to {d2hi:.2f} {um} at {c2} CMC, and {d3lo:.2f} to
{d3hi:.2f} {um} at {c3} CMC. **q3 is the instrument-inverted relative composition of the remaining
detected particles**, not particle mass and not an independent modality, since it is inverted from
the same laser-diffraction acquisition that yields Copt. The paths are described as overlapping;
no verdict of equivalence and no verdict of difference is made between them. The coarse tail above
15 {um} lies outside the established reliable inversion range and is neither displayed nor
interpreted here.
"""


def _emit(out: Path, formats, base: Path | None = None) -> int:
    base = Path(base or arm_b_root())
    table, meta = build_source_data(base)
    checks = validate(table, meta, base)
    written = render(table, meta, out, formats)

    csv_path = out / f"{STEM}_source_data.csv"
    table.to_csv(csv_path, index=False)

    b, c, d = meta["panel_b"], meta["panel_c"], meta["panel_d"]
    ends = d["d50_um_at_g_0_2_and_0_8"]
    caption = CAPTION.format(
        stem=STEM, pm="±", um="µm",
        c1="0.5×", c2="1×", c3="10×",
        n_preps=4, t_win=DISPLAY_WINDOW_MIN,
        t_end=meta["panel_a"]["acquisition_window_min"][1],
        tail_span=meta["panel_a"]["omitted_region"]["max_condition_mean_range_pp"],
        t1=b["mean_relax_min"]["0.5x CMC"], t2=b["mean_relax_min"]["1.0x CMC"],
        t3=b["mean_relax_min"]["10x CMC"],
        s1=b["mean_relax_sd_between_preps"]["0.5x CMC"],
        s2=b["mean_relax_sd_between_preps"]["1.0x CMC"],
        s3=b["mean_relax_sd_between_preps"]["10x CMC"],
        r1lo=b["prep_value_range_min"]["0.5x CMC"][0], r1hi=b["prep_value_range_min"]["0.5x CMC"][1],
        r2lo=b["prep_value_range_min"]["1.0x CMC"][0], r2hi=b["prep_value_range_min"]["1.0x CMC"][1],
        r3lo=b["prep_value_range_min"]["10x CMC"][0], r3hi=b["prep_value_range_min"]["10x CMC"][1],
        l1=c["copt_loss_pct"]["0.5x CMC"], l2=c["copt_loss_pct"]["1.0x CMC"],
        l3=c["copt_loss_pct"]["10x CMC"],
        p1lo=c["prep_value_range_pct"]["0.5x CMC"][0], p1hi=c["prep_value_range_pct"]["0.5x CMC"][1],
        p2lo=c["prep_value_range_pct"]["1.0x CMC"][0], p2hi=c["prep_value_range_pct"]["1.0x CMC"][1],
        d1lo=ends["0.5x CMC"][0], d1hi=ends["0.5x CMC"][1],
        d2lo=ends["1.0x CMC"][0], d2hi=ends["1.0x CMC"][1],
        d3lo=ends["10x CMC"][0], d3hi=ends["10x CMC"][1])
    wording = check_wording(caption)
    if not wording["ok"]:
        raise ValueError(f"the caption makes a claim this figure cannot support: {wording}")
    checks["caption_wording"] = wording
    caption_path = out / f"{STEM}_caption.md"
    caption_path.write_text(caption)

    prov = provenance_record(
        "manuscript_arm_b_ld_response", study_root=str(base),
        uv_ph_values=(), figure_stem=STEM,
        signal_conditioning=SIGNAL_CONDITIONING,
        manuscript_section="dissolution-medium polysorbate — main laser-diffraction figure",
        does_not_overwrite="every upstream analysis output under <arm_b>/analysis is read only",
        sources={k: str(base / v) for k, v in SOURCES.items()},
        source_paths_relative=dict(SOURCES),
        panel_sources={
            "A": "arm_b_angular_trajectories_preps.csv::pct_of_i0_fit and "
                 "arm_b_angular_trajectories_conditions.csv::pct_mean,pct_sd_between_preps "
                 "(frozen export from arm_b_angular_trajectories.py; its provenance carries the "
                 "ΣI definition, despiking, time alignment and the KWW cross-check)",
            "B": "angular_kww_by_date_condition.csv::mean_relax_min (preparation points) and "
                 "angular_kww_by_condition.csv::mean_relax_min_mean,mean_relax_min_sd "
                 "(condition summary) — the prespecified primary descriptor; the KWW tau in the "
                 "same table is NOT drawn",
            "C": "arm_b_partition_preps.csv::copt_frac_loss from the filtered_48h_pipeline_copt "
                 "path, cross-checked against arm_b_partition_conditions.csv::copt_frac_loss",
            "D": "q3_matched_extent_runs.csv::d50_um averaged within preparation, and "
                 "q3_matched_extent_balanced.csv::d50_um_mean,d50_um_sd"},
        panel_details=meta, aggregation=AGGREGATION,
        normalization=meta["panel_a"]["normalization"],
        signal_definition=meta["panel_a"]["signal_definition"],
        reliability_rules={
            "q3_coarse_tail": d["coarse_tail_reason"],
            "q3_coarse_tail_displayed": False,
            "partition_path": c["partition_path"],
            "kww_input": "raw measured ΣI — upward-despiked, NOT background-subtracted; the "
                         "*_bgsub sensitivity columns are not used",
            "panel_a_display_window": meta["panel_a"]["omitted_region"]},
        numerical_checks=checks, career_artifacts_used=False,
        scope={"study": "micelle_effects_tween80_arm_b",
               "domain": "laser diffraction only — angular signal, Copt and q3",
               "excluded": ["UV dissolved mass", "forward model", "saturation solubility",
                            "solubility-normalized coefficient", "coarse tail above 15 µm",
                            "NSF CAREER artifacts"],
               "is_dissolved_mass": False,
               "independent_of_ld_acquisition": False,
               "verdict_issued_on_q3_paths": False,
               "inferential_test_shown": False,
               "claim": "10x CMC produced earlier aggregate angular-signal disappearance than "
                        "either lower level (its four preparation values lie entirely below "
                        "both); fractional Copt-loss MEANS were ordered across the ladder, but "
                        "the 0.5x-1x preparation clouds overlapped and that contrast is not "
                        "resolved; the reliability-restricted q3 D50 paths overlapped "
                        "descriptively at matched Copt loss",
               "claim_not_made": [
                   "that polysorbate increased Copt loss monotonically in a way this design "
                   "resolves — only the means are ordered",
                   "that an overlap of preparation clouds shows the levels do not differ",
                   "any equivalence or difference verdict on the q3 size paths"],
               "separation_basis": "overlap of the four preparation-level values; no inferential "
                                   "test is applied"})
    stray = [f for f in COARSE_TAIL_FIELDS if f in json.dumps(prov).lower()
             and f not in ("above_15",)]
    if stray:
        raise ValueError(f"the provenance sidecar carries a coarse-tail field: {stray}")
    prov_path = write_provenance(out / f"{STEM}_provenance.json", prov)

    print(f"{len(CONDITIONS)} conditions x 4 preparations (technical runs nested); "
          f"all spreads are between-preparation SD")
    print(f"{'condition':>10}  {'<t> (min)':>16}  {'Copt loss (%)':>16}  "
          f"{'D50 g=0.2->0.8 (um)':>22}")
    for condition in CONDITIONS:
        lo, hi = ends[condition]
        print(f"{LABEL[condition]:>10}  {b['mean_relax_min'][condition]:8.4f} ± "
              f"{b['mean_relax_sd_between_preps'][condition]:.4f}  "
              f"{c['copt_loss_pct'][condition]:8.1f} ± "
              f"{c['copt_loss_sd_between_preps_pp'][condition]:5.1f}  "
              f"{lo:10.4f} -> {hi:.4f}")
    sc = checks["supported_contrasts"]
    print(f"claim scope: {sc['claim_scope']}")
    print("preparation-cloud separation (descriptive, no inferential test):")
    for panel, key in (("<t>", "panel_b"), ("Copt loss", "panel_c")):
        pairs = meta[key]["separation"]["pairs"]
        resolved = [k for k, v in pairs.items() if v["contrast_resolved_by_this_design"]]
        overlapping = [k for k, v in pairs.items() if not v["contrast_resolved_by_this_design"]]
        print(f"  {panel:10s} separated: {resolved or 'none'}")
        print(f"  {'':10s} overlapping: {overlapping or 'none'}")
    print("q3 paths reported descriptively — no equivalence and no difference verdict; "
          "the >15 µm coarse tail is not displayed or interpreted.")
    for path in [*written, csv_path, caption_path, prov_path]:
        print(f"wrote {path}")
    return 0


NEGATION_CUES = ("not ", "no ", "neither ", "nor ", "never ", "rather than ", "without ")
NEGATION_WINDOW = 70        # characters of lead-in searched for a denial


def check_wording(text: str) -> dict:
    """No mass-equivalence, q3-invariance or q3-difference claim may reach the caption.

    Each banned phrase is allowed only inside an explicit denial, so every occurrence must carry a
    negation cue in the clause leading up to it. A window is used rather than fixed templates
    because the denials are written as prose ("is not a dissolved fraction or a particle-mass
    loss"), and a template list silently misses the second item in a coordinated pair.
    """
    low = " ".join(str(text).split()).lower()
    hits = []
    for phrase in BANNED_PHRASES:
        start = low.find(phrase)
        while start != -1:
            lead = low[max(0, start - NEGATION_WINDOW):start]
            if not any(cue in lead for cue in NEGATION_CUES):
                hits.append({"phrase": phrase, "context": low[max(0, start - 60):start + 40]})
            start = low.find(phrase, start + 1)
    return {"banned_phrases_found": hits,
            "rule": "mass-equivalence, q3-invariance and q3-difference claims may appear only "
                    f"as explicit denials — a negation cue within {NEGATION_WINDOW} characters",
            "ok": not hits}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--study-root", type=Path, default=None)
    p.add_argument("--formats", default="pdf,png,svg")
    args = p.parse_args(argv)
    return _emit(args.output_dir,
                 tuple(f.strip() for f in args.formats.split(",") if f.strip()),
                 args.study_root)


if __name__ == "__main__":
    raise SystemExit(main())
