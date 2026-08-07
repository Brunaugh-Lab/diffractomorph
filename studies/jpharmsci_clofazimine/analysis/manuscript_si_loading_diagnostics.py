"""Supporting Information figure — particle-side optical response to loading (Section 3.4.2).

One scientific purpose: **to report what the two particle-side optical coordinates did across the
loading ladder, and to make plain that neither confirms the mass-domain result in the main-text
figure.**

  A  Fractional Copt loss against delivered dose. It RISES with loading (58.0, 72.0, 78.3 %)
     while the UV dissolved-mass fraction FALLS (99.0, 92.9, 87.2 %). The two coordinates weight
     the population differently, so this is a divergence to state, not a contradiction to resolve
     and not a confirmation to claim.
  B  q3 D50 against matched optical extent for the three loadings, using only frames that pass the
     established inversion-reliability gate (q3 mass above 15 um at or below ``TAIL_MAX_PCT``).

**Both panels are descriptive.** Copt is a scattering-weighted particle coordinate, not a
dissolved-dose fraction. q3 is the PAQXOS-inverted relative composition of the particles still
detected — not particle mass, and not an independent modality, since it is inverted from the same
laser-diffraction acquisition Copt is derived from. With one preparation there is no error term
that licenses an invariance or difference verdict on the size paths, so the figure reports the
across-loading D50 range beside the technical-replicate scatter and adjudicates neither.

Scope is the pH 4.5 sub-study only, enforced by :func:`manuscript_loading_common.read_scoped`.

:func:`build_source_data` assembles every plotted number from the :mod:`copt_loading` artifacts;
:func:`render` consumes only that table.

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

STEM = "FigureS_loading_optical_and_reliability_diagnostics"

EXPECTED_COPT_LOSS_PCT = {12: 58.0, 18: 72.0, 24: 78.3}
EXPECTED_UV_PCT = {12: 99.0, 18: 92.9, 24: 87.2}      # the opposing mass-domain direction
EXPECTED_D50_RANGE_UM = (0.02, 0.18)
EXPECTED_RATIO_AT_LARGEST_RANGE = 1.5

# Verdict wording the design cannot support for the size paths. One preparation gives no
# error term, so neither "these are the same" nor "these differ" may be asserted.
BANNED_VERDICTS = ("invariant", "size-invariant", "loading-invariant", "unchanged across",
                   "no difference", "significantly different", "statistically")


# ── panel A — particle-side Copt response ────────────────────────────────────────────────────

def _panel_a() -> tuple[list[dict], dict]:
    runs = mlc.read_runs()
    # Copt loss is stored as a fraction; percent reads better beside the UV percentages it is
    # deliberately NOT being equated with.
    runs = runs.assign(copt_loss_pct=runs["copt_frac_loss"] * 100.0)
    summary = mlc.level_summary(runs, "copt_loss_pct")
    uv = mlc.level_summary(runs, "uv_pct_injected_end")

    rows = []
    for r in runs.itertuples():
        rows.append({"panel": "A", "kind": "run_point", "series": "Individual technical run",
                     "substudy": mlc.PH45, "level_pct": int(r.level_pct), "rep": int(r.rep),
                     "x_quantity": "delivered_dose_ug", "x_value": mlc.dose_ug(r.dose_mg),
                     "y_quantity": "copt_fractional_loss_pct",
                     "y_value": float(r.copt_loss_pct),
                     "source_artifact": "copt_loading_runs.csv",
                     "source_column": "copt_frac_loss (x100)"})
    for r in summary.itertuples():
        rows.append({"panel": "A", "kind": "level_mean", "series": "Mean ± technical SD",
                     "substudy": mlc.PH45, "level_pct": int(r.level_pct),
                     "x_quantity": "delivered_dose_ug", "x_value": float(r.dose_ug),
                     "y_quantity": "copt_fractional_loss_pct", "y_value": float(r.mean),
                     "y_sd": float(r.sd), "n_technical_reps": int(r.n_technical_reps),
                     "source_artifact": "copt_loading_runs.csv",
                     "source_column": "copt_frac_loss (x100)"})

    loss = {int(r.level_pct): float(r.mean) for r in summary.itertuples()}
    mass = {int(r.level_pct): float(r.mean) for r in uv.itertuples()}
    meta = {"substudy": mlc.PH45,
            "copt_loss_pct": {k: round(v, 4) for k, v in loss.items()},
            "copt_loss_technical_sd_pp": {int(r.level_pct): round(float(r.sd), 4)
                                          for r in summary.itertuples()},
            "uv_dissolved_pct_for_contrast": {k: round(v, 4) for k, v in mass.items()},
            "copt_loss_rises_with_loading": bool(
                np.all(np.diff([loss[lv] for lv in mlc.LEVELS]) > 0)),
            "uv_mass_fraction_falls_with_loading": bool(
                np.all(np.diff([mass[lv] for lv in mlc.LEVELS]) < 0)),
            "directions_oppose": True,
            "used_as_confirmation_of_mass_result": False,
            "interpretation": "Copt is a scattering-weighted particle coordinate, not a "
                              "dissolved-dose fraction; the two coordinates weight the population "
                              "differently, so the opposing directions are reported, not "
                              "reconciled"}
    return rows, meta


# ── panel B — reliability-gated q3 size paths ────────────────────────────────────────────────

def _panel_b() -> tuple[list[dict], dict]:
    matched = mlc.read_scoped("q3_matched")
    frames = mlc.read_scoped("q3_frames")
    spread = mlc.read_scoped("q3_spread")

    rows = []
    for r in matched.itertuples():
        rows.append({"panel": "B", "kind": "q3_run_point",
                     "series": f"{int(r.level_pct)} % loading (technical run)",
                     "substudy": mlc.PH45, "level_pct": int(r.level_pct), "rep": int(r.rep),
                     "x_quantity": "matched_optical_extent_g", "x_value": float(r.extent_g),
                     "y_quantity": "q3_d50_um", "y_value": float(r.d50_um),
                     "n_reliable_frames": int(r.n_reliable_frames),
                     "source_artifact": "copt_loading_q3_matched_extent.csv",
                     "source_column": "d50_um"})
    level = (matched.groupby(["level_pct", "extent_g"], as_index=False)
             .agg(mean=("d50_um", "mean"), sd=("d50_um", "std"), n=("d50_um", "count")))
    for r in level.itertuples():
        rows.append({"panel": "B", "kind": "q3_level_mean",
                     "series": f"{int(r.level_pct)} % loading",
                     "substudy": mlc.PH45, "level_pct": int(r.level_pct),
                     "x_quantity": "matched_optical_extent_g", "x_value": float(r.extent_g),
                     "y_quantity": "q3_d50_um", "y_value": float(r.mean), "y_sd": float(r.sd),
                     "n_technical_reps": int(r.n),
                     "source_artifact": "copt_loading_q3_matched_extent.csv",
                     "source_column": "d50_um"})

    largest = spread.loc[spread["d50_range_across_loadings_um"].idxmax()]
    reliable = frames.groupby("run_id")["q3_frame_reliable"].sum()
    meta = {"substudy": mlc.PH45,
            "gate": {"rule": f"frames carrying more than {mlc.TAIL_MAX_PCT:g} % of q3 mass above "
                             f"15 um are not size measurements and are excluded",
                     "source": "copt_loading.TAIL_MAX_PCT, applied upstream in "
                               "copt_loading_q3_matched_extent.csv",
                     "reliable_frames_per_run": {"min": int(reliable.min()),
                                                 "max": int(reliable.max())}},
            "extents_evaluated_g": sorted(float(v) for v in matched["extent_g"].unique()),
            "d50_range_across_loadings_um": {f"{float(r.extent_g):g}":
                                             round(float(r.d50_range_across_loadings_um), 4)
                                             for r in spread.itertuples()},
            "mean_technical_rep_sd_um": {f"{float(r.extent_g):g}":
                                         round(float(r.mean_technical_rep_sd_um), 4)
                                         for r in spread.itertuples()},
            "range_over_rep_sd": {f"{float(r.extent_g):g}": round(float(r.range_over_rep_sd), 4)
                                  for r in spread.itertuples()},
            "largest_comparison": {"extent_g": float(largest["extent_g"]),
                                   "d50_range_um": float(largest["d50_range_across_loadings_um"]),
                                   "rep_sd_um": float(largest["mean_technical_rep_sd_um"]),
                                   "range_over_rep_sd": float(largest["range_over_rep_sd"])},
            "verdict_issued": False,
            "interpretation": "q3 is the PAQXOS-inverted relative composition of the particles "
                              "still detected — not particle mass, and not independent of the "
                              "laser-diffraction acquisition Copt comes from; with one "
                              "preparation there is no error term for an invariance or "
                              "difference verdict, so the spread is reported, not adjudicated"}
    return rows, meta


def build_source_data() -> tuple[pd.DataFrame, dict]:
    """Every plotted value, assembled from the authoritative copt_loading artifacts only."""
    runs = mlc.read_runs()
    rows_a, meta_a = _panel_a()
    rows_b, meta_b = _panel_b()
    table = pd.DataFrame(rows_a + rows_b)
    table["study"] = "conc_dependent_disso_study"
    meta = {"panel_a": meta_a, "panel_b": meta_b,
            "design": mlc.design_counts(runs),
            "aggregations": mlc.AGGREGATIONS,
            "claim_boundaries": mlc.CLAIM_BOUNDARIES}
    return table, meta


# ── validation ───────────────────────────────────────────────────────────────────────────────

def validate(table: pd.DataFrame, meta: dict) -> dict:
    checks: dict = {}

    design = meta["design"]
    checks["design"] = {
        "n_preparations": design["n_preparations"], "preparation": design["preparation"],
        "technical_reps_per_level": design["technical_reps_per_level"],
        "replicates_are_technical": True,
        "substudies_in_figure": sorted(set(table["substudy"])),
        "ok": (design["n_preparations"] == 1 and design["technical_reps_per_level"] == [3]
               and design["preparation"] == [mlc.MANUSCRIPT_PREP]
               and set(table["substudy"]) == {mlc.MANUSCRIPT_SUBSTUDY})}
    if not checks["design"]["ok"]:
        raise ValueError(f"the SI figure must show one {mlc.MANUSCRIPT_SUBSTUDY} preparation "
                         f"with three technical replicates per level: {checks['design']}")

    # Panel A: displayed values, and agreement with the tabulated level means.
    level_means = mlc.read_scoped("level_means").set_index("level_pct")
    a_checks = {}
    for level in mlc.LEVELS:
        row = table[table["panel"].eq("A") & table["kind"].eq("level_mean")
                    & table["level_pct"].eq(level)]
        value = float(row["y_value"].iloc[0])
        artifact = float(level_means.loc[level, "copt_frac_loss"]) * 100.0
        sd_artifact = float(level_means.loc[level, "frac_loss_sd"]) * 100.0
        a_checks[level] = {
            "copt_loss_pct": round(value, 1),
            "copt_loss_pct_expected": EXPECTED_COPT_LOSS_PCT[level],
            "abs_deviation_from_artifact": abs(value - artifact),
            "sd_pp": round(float(row["y_sd"].iloc[0]), 1),
            "sd_abs_deviation_from_artifact": abs(float(row["y_sd"].iloc[0]) - sd_artifact)}
        a_checks[level]["ok"] = (
            a_checks[level]["copt_loss_pct"] == EXPECTED_COPT_LOSS_PCT[level]
            and a_checks[level]["abs_deviation_from_artifact"] < 1e-9
            and a_checks[level]["sd_abs_deviation_from_artifact"] < 1e-9)
    checks["panel_a_values"] = a_checks
    bad = [lv for lv, v in a_checks.items() if not v["ok"]]
    if bad:
        raise ValueError(f"panel A does not reproduce the tabulated Copt loss at levels {bad}")

    # The point of panel A: the optical extent moves OPPOSITE to the mass fraction.
    pa = meta["panel_a"]
    checks["optical_opposes_mass"] = {
        "copt_loss_pct": pa["copt_loss_pct"],
        "uv_dissolved_pct": pa["uv_dissolved_pct_for_contrast"],
        "copt_loss_rises_with_loading": pa["copt_loss_rises_with_loading"],
        "uv_mass_fraction_falls_with_loading": pa["uv_mass_fraction_falls_with_loading"],
        "uv_matches_expected": all(round(pa["uv_dissolved_pct_for_contrast"][lv], 1)
                                   == EXPECTED_UV_PCT[lv] for lv in mlc.LEVELS),
        "used_as_confirmation": False,
        "ok": (pa["copt_loss_rises_with_loading"] and pa["uv_mass_fraction_falls_with_loading"]
               and all(round(pa["uv_dissolved_pct_for_contrast"][lv], 1) == EXPECTED_UV_PCT[lv]
                       for lv in mlc.LEVELS))}
    if not checks["optical_opposes_mass"]["ok"]:
        raise ValueError(f"the Copt/UV divergence the panel reports failed against the "
                         f"artifacts: {checks['optical_opposes_mass']}")

    # Panel B: every plotted size came through the reliability gate, and the descriptive spread
    # reproduces the tabulated summary.
    matched = mlc.read_scoped("q3_matched")
    plotted = (table[table["panel"].eq("B") & table["kind"].eq("q3_run_point")]
               .rename(columns={"x_value": "extent_g"}))
    merged = plotted.merge(matched, on=["level_pct", "rep", "extent_g"], suffixes=("", "_art"))
    pb = meta["panel_b"]
    largest = pb["largest_comparison"]
    ranges = list(pb["d50_range_across_loadings_um"].values())
    checks["panel_b_gate_and_spread"] = {
        "n_plotted_points": int(len(plotted)),
        "n_matched_to_artifact": int(len(merged)),
        "max_abs_deviation_um": float((merged["y_value"] - merged["d50_um"]).abs().max()),
        "source_is_the_gated_artifact": True,
        "gate": pb["gate"]["rule"],
        "d50_range_um_min": round(min(ranges), 2), "d50_range_um_max": round(max(ranges), 2),
        "expected_range_um": list(EXPECTED_D50_RANGE_UM),
        "ratio_at_largest_comparison": round(largest["range_over_rep_sd"], 1),
        "expected_ratio_at_largest_comparison": EXPECTED_RATIO_AT_LARGEST_RANGE,
        "verdict_issued": pb["verdict_issued"]}
    pbc = checks["panel_b_gate_and_spread"]
    pbc["ok"] = (pbc["n_plotted_points"] == pbc["n_matched_to_artifact"] == len(matched)
                 and pbc["max_abs_deviation_um"] < 1e-12
                 and (pbc["d50_range_um_min"], pbc["d50_range_um_max"]) == EXPECTED_D50_RANGE_UM
                 and pbc["ratio_at_largest_comparison"] == EXPECTED_RATIO_AT_LARGEST_RANGE
                 and pbc["verdict_issued"] is False)
    if not pbc["ok"]:
        raise ValueError(f"panel B does not reproduce the gated matched-extent artifact: {pbc}")

    # Scope.
    blob = " ".join(str(v) for v in table.to_dict("list").values()).lower()
    found_t = mlc.find_foreign_tokens(blob)
    under_study = all(str(p).startswith(str(mlc.STUDY_REL)) for p in mlc.SOURCES.values())
    checks["scope"] = {
        "foreign_tokens_found": found_t, "sources_all_under_study": under_study,
        "substudies_present": sorted(set(table["substudy"])),
        "ok": (not found_t and under_study
               and set(table["substudy"]) == {mlc.MANUSCRIPT_SUBSTUDY})}
    if not checks["scope"]["ok"]:
        raise ValueError(f"a sub-study outside the SI figure's scope reached it: "
                         f"{checks['scope']}")

    limits = {"A": _limits_a(table), "B": _limits_b(table)}
    clipping = {}
    for panel, (lo, hi) in limits.items():
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

FIG_W = 5.9
DOSE_LIM = (120.0, 235.0)
DOSE_TICKS = (133, 177, 222)


def _pad(values, frac_lo=0.09, frac_hi=0.09):
    lo, hi = float(min(values)), float(max(values))
    span = hi - lo
    return lo - frac_lo * span, hi + frac_hi * span


def _extremes(panel: pd.DataFrame) -> list[float]:
    sd = panel["y_sd"].fillna(0.0)
    return (panel["y_value"] - sd).tolist() + (panel["y_value"] + sd).tolist()


def _limits_a(table):
    return _pad(_extremes(table[table["panel"].eq("A")]))


def _limits_b(table):
    return _pad(_extremes(table[table["panel"].eq("B")]), 0.09, 0.16)


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
    ax.set_ylabel("Fractional Copt loss (%)")
    ax.set_title("Particle-side Copt loss increased", pad=4)
    ax.set_xlim(*DOSE_LIM)
    ax.set_xticks(DOSE_TICKS)
    ax.set_ylim(*_limits_a(table))
    mlc.clean_axes(ax)
    ax.legend(handles=[
        Line2D([], [], color=mlc.BLUE, marker="o", ms=4.4, lw=1.0,
               markeredgecolor="white", markeredgewidth=0.5),
        Line2D([], [], color=mlc.BLUE_PALE, marker="o", ms=2.6, ls="none", mfc="white", mew=0.8)],
        labels=["Mean ± technical SD", "Technical run"],
        loc="upper left", frameon=False, handlelength=1.7, handletextpad=0.5,
        borderpad=0.0, labelspacing=0.3)


def _axis_b(ax, table):
    b = table[table["panel"].eq("B")]
    runs = b[b["kind"].eq("q3_run_point")]
    # individual runs carry their own loading colour, so the replicate scatter the panel is about
    # can be read against the level means rather than forming one anonymous grey cloud
    for level in mlc.LEVELS:
        g = runs[runs["level_pct"].eq(level)]
        ax.plot(g["x_value"], g["y_value"], "o", ms=2.0, mfc="white",
                mec=mlc.LOADING_RAMP[level], mew=0.6, ls="none", alpha=0.85, zorder=2)
    for level in mlc.LEVELS:
        g = b[b["kind"].eq("q3_level_mean") & b["level_pct"].eq(level)].sort_values("x_value")
        ax.errorbar(g["x_value"], g["y_value"], yerr=g["y_sd"], fmt="o-", ms=3.4, lw=1.2,
                    color=mlc.LOADING_RAMP[level], ecolor=mlc.LOADING_RAMP[level],
                    elinewidth=0.8, capsize=1.8, capthick=0.8, markeredgecolor="white",
                    markeredgewidth=0.4, zorder=4, label=f"{level} %")

    ax.set_xlabel("Matched optical extent, g")
    ax.set_ylabel("q3 D50 at matched extent (µm)")
    ax.set_title("Gated q3 size paths (descriptive)", pad=4)
    ax.set_xlim(0.15, 0.75)
    ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_ylim(*_limits_b(table))
    mlc.clean_axes(ax)
    ax.legend(loc="lower left", frameon=False, ncol=3, handlelength=1.5, handletextpad=0.4,
              borderpad=0.0, columnspacing=0.9, title="Nominal loading",
              title_fontsize=6.4)


def render(table: pd.DataFrame, meta: dict, out_dir: Path, formats=("pdf", "png", "svg")):
    mlc.apply_style()
    fig = plt.figure(figsize=(FIG_W, 2.85))
    gs = fig.add_gridspec(1, 2, left=0.098, right=0.995, top=0.878, bottom=0.175, wspace=0.40)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    _axis_a(axA, table)
    _axis_b(axB, table)
    mlc.panel_tags(fig, ((axA, "A"), (axB, "B")))
    return mlc.save(fig, out_dir, STEM, formats)


CAPTION = """# {stem} — caption draft

**Particle-side optical response to loading, reported descriptively.** Both panels come from the
same single pH 4.5 suspension preparation ({prep}) used for the main-text loading figure; the
three runs at each loading are technical replicates and error bars are technical SDs. Neither
panel is an independent check of the mass-domain result: Copt and q3 are derived from the same
laser-diffraction acquisition and describe the particles, not the dissolved dose.

**(A)** Fractional loss of optical concentration over the run, against delivered dose. Open
symbols are the {n_reps} technical runs, filled symbols the mean {pm} technical SD. Fractional
Copt loss **increased** with loading ({l12:.1f}, {l18:.1f} and {l24:.1f} %) while the UV
dissolved-mass fraction **decreased** over the same ladder ({u12:.1f}, {u18:.1f} and {u24:.1f} %
of delivered dose). The two coordinates weight the particle population differently — Copt is a
scattering-weighted particle coordinate, not a dose fraction — so the opposing directions are
reported as they stand. Fractional Copt loss is not evidence for the frozen-model prediction and
is not used as confirmation of it.

**(B)** q3 D50 against matched optical extent g, the clock-free comparison, for the three
loadings. Only frames passing the established inversion-reliability gate contribute: frames
carrying more than {tail:g} % of q3 mass above 15 {um} are not size measurements and are excluded
upstream, leaving {frames_min}–{frames_max} reliable frames per run. Pale symbols are individual
technical runs and coloured symbols the level means {pm} technical SD. Across the evaluated
extents the D50 range between loadings was {rng_lo:.2f}–{rng_hi:.2f} {um}, about
{ratio:.1f} times the technical-replicate SD at the largest comparison (g = {g_large:g}). This is
a descriptive statement of spread. q3 is the PAQXOS-inverted relative composition of the particles
still detected — it is not particle mass, and it is not an independent modality — and with a
single preparation there is no error term that would license a verdict either way on whether the
size path depends on loading.
"""


def _emit(out: Path, formats) -> int:
    table, meta = build_source_data()
    checks = validate(table, meta)
    written = render(table, meta, out, formats)

    csv_path = out / f"{STEM}_source_data.csv"
    table.to_csv(csv_path, index=False)

    pa, pb = meta["panel_a"], meta["panel_b"]
    ranges = list(pb["d50_range_across_loadings_um"].values())
    caption = CAPTION.format(
        stem=STEM, prep="2026-07-27", pm="±", um="µm", n_reps=3,
        tail=mlc.TAIL_MAX_PCT,
        l12=pa["copt_loss_pct"][12], l18=pa["copt_loss_pct"][18], l24=pa["copt_loss_pct"][24],
        u12=pa["uv_dissolved_pct_for_contrast"][12], u18=pa["uv_dissolved_pct_for_contrast"][18],
        u24=pa["uv_dissolved_pct_for_contrast"][24],
        frames_min=pb["gate"]["reliable_frames_per_run"]["min"],
        frames_max=pb["gate"]["reliable_frames_per_run"]["max"],
        rng_lo=min(ranges), rng_hi=max(ranges),
        ratio=pb["largest_comparison"]["range_over_rep_sd"],
        g_large=pb["largest_comparison"]["extent_g"])
    wording = mlc.check_wording(caption)
    verdicts = [v for v in BANNED_VERDICTS if v in " ".join(caption.split()).lower()]
    wording["banned_verdicts_found"] = verdicts
    wording["ok"] = wording["ok"] and not verdicts
    if not wording["ok"]:
        raise ValueError(f"the caption overstates the claim, issues a verdict the design cannot "
                         f"support, or names an out-of-scope study: {wording}")
    checks["caption_wording"] = wording
    caption_path = out / f"{STEM}_caption.md"
    caption_path.write_text(caption)

    prov = provenance_record(
        "manuscript_si_loading_diagnostics", study_root=str(data_root() / mlc.STUDY_REL),
        uv_ph_values=(4.5,), figure_stem=STEM,
        manuscript_section="3.4.2 starting particle loading (Supporting Information)",
        does_not_overwrite="conc_dependent_disso_study/analysis/copt_loading.* "
                           "(the analysis figure is left untouched)",
        sources={k: str(data_root() / v) for k, v in mlc.SOURCES.items()},
        source_paths_relative={k: str(v) for k, v in mlc.SOURCES.items()},
        panel_sources={
            "A": "copt_loading_runs.csv::copt_frac_loss (x100) and ::dose_mg; the UV values "
                 "quoted for contrast come from ::uv_pct_injected_end",
            "B": "copt_loading_q3_matched_extent.csv::d50_um (already reliability-gated "
                 "upstream); frame counts from copt_loading_q3_frames.csv::q3_frame_reliable; "
                 "descriptive spread from copt_loading_q3_size_path_spread.csv"},
        panel_details={"A": pa, "B": pb},
        design=meta["design"], aggregations=mlc.AGGREGATIONS,
        reliability_gate=pb["gate"], numerical_checks=checks, career_artifacts_used=False,
        scope={"substudy": mlc.MANUSCRIPT_SUBSTUDY,
               "row_admission": "copt_loading artifacts restricted to the "
                                f"{mlc.MANUSCRIPT_SUBSTUDY} sub-study by "
                                "manuscript_loading_common.read_scoped",
               "domain": "particle-side optical coordinates only — fractional Copt loss and "
                         "reliability-gated q3 at matched extent",
               "is_mass_measurement": False,
               "independent_of_ld_acquisition": False,
               "used_as_confirmation_of_mass_result": False,
               "verdict_issued_on_size_paths": False,
               "excluded": ["NSF CAREER artifacts", "Arm A / Arm B",
                            "dissolution-medium polysorbate data"],
               "claim_boundaries": mlc.CLAIM_BOUNDARIES})
    # the sidecar may NAME the other studies it excludes; it may not mention the
    # out-of-scope sub-study at all
    stray = mlc.find_foreign_tokens(json.dumps(prov), mlc.OUT_OF_SCOPE_SUBSTUDY_TOKENS)
    if stray:
        raise ValueError(f"the provenance sidecar names an out-of-scope study: {stray}")
    prov_path = write_provenance(out / f"{STEM}_provenance.json", prov)

    print(f"sub-study: {mlc.MANUSCRIPT_SUBSTUDY} — {meta['design']['n_preparations']} "
          f"preparation, {meta['design']['technical_reps_per_level']} technical reps per level")
    print("panel A — Copt loss RISES while the UV mass fraction FALLS:")
    for level in mlc.LEVELS:
        print(f"  {level:2d} %  Copt loss {pa['copt_loss_pct'][level]:5.1f} % "
              f"± {pa['copt_loss_technical_sd_pp'][level]:.1f}   "
              f"UV {pa['uv_dissolved_pct_for_contrast'][level]:5.1f} %")
    print(f"panel B — gated q3: across-loading D50 range {min(ranges):.2f}-{max(ranges):.2f} µm; "
          f"{pb['largest_comparison']['range_over_rep_sd']:.1f}x the technical-replicate SD at "
          f"g = {pb['largest_comparison']['extent_g']:g}. Descriptive; no verdict issued.")
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
