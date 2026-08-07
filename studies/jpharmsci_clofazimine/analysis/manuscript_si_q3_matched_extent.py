"""Supporting Information figure — q3 matched-extent comparison and the pH 4.0 reliability boundary.

Supports two narrow Section 3.3 statements and nothing wider:

  A  At g = 0.8 — a 20 % loss of total angular signal, the only extent every pH condition reaches —
     the PAQXOS-inverted q3 D50 changed little from each run's own starting value.
  B  The large late pH 4.0 coarse-q3 excursion came from one preparation date, was retained, and did
     not determine the prospectively defined cross-date-median summary.

**Scope limits this figure must not exceed.** q3 is the instrument's inverted *relative composition of
the particles still detected*. It is not a mass measurement, it is not independent of the LD
acquisition, and nothing here supports aggregation, deaggregation, uniform dissolution, a
Mie-derived mechanism, or the absence of redistribution. This is a reliability and matched-extent
figure.

Sources are the canonical artifacts; nothing is refit and no figure is digitised:

  Panel A  ``psd_evolution/matched_extent/q3_matched_extent_by_run.csv``, cross-checked against
           ``..._by_condition.csv``. Settings are read from :mod:`q3_matched_extent` rather than
           assumed — ``DEFAULT_FLOOR`` (the floor used for the figures) and the working eligibility.
           The run-specific STARTING q3 D50 — the mean of that run's first three eligible frames,
           ``q3_matched_extent.ANCHOR_N`` — is recovered from the artifact's own ``dlog10_D50`` as
           ``D50 / 10**dlog10_D50``, so both plotted values come from one stored quantity. (The
           module calls this the "anchor" internally; the figure and caption say "starting".)
  Panel B  ``psd_evolution/model_vs_q3/observed_date_level_descriptors.csv`` (per-date means),
           ``observed_condition_descriptors_median.csv`` (the primary cross-date median), and
           ``q3_absolute_percentile_outlier_audit.csv`` for the audited timepoint.

Aggregation is date-first everywhere: nested runs are averaged within preparation date before any
cross-date summary, so the plotted independent units are preparation-date means. Panel B's heavy
curve is the cross-date **median** — the same prospectively applied estimator used for every pH,
timepoint and percentile — never the mean.

Run with the pipeline venv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diffractomorph_pipeline import plot_styles as ps
from diffractomorph_pipeline.config import data_root

import q3_matched_extent as qme
from arm_b_provenance import provenance_record, write_provenance

STEM = "FigureS_q3_matched_extent_and_reproducibility"
PH_STUDY = Path("disso_experiments/ph_dependent_dissolution_study")
SOURCES = {
    "matched_by_run": PH_STUDY / "psd_evolution/matched_extent/q3_matched_extent_by_run.csv",
    "matched_by_condition": PH_STUDY / "psd_evolution/matched_extent/q3_matched_extent_by_condition.csv",
    "date_level": PH_STUDY / "psd_evolution/model_vs_q3/observed_date_level_descriptors.csv",
    "condition_median": PH_STUDY / "psd_evolution/model_vs_q3/observed_condition_descriptors_median.csv",
    "outlier_audit": PH_STUDY / "psd_evolution/model_vs_q3/q3_absolute_percentile_outlier_audit.csv",
}
TARGET_G = 0.8                 # the only extent every pH condition reaches
FLOOR = qme.DEFAULT_FLOOR      # read from the module: the floor used for the figures
CONDITIONS = ["pH 4.0", "pH 4.5", "pH 5.0"]
COLOR = {f"pH {ph:.1f}": st["color"] for ph, st in ps.PH_STYLE.items()}
EXPECTED_A = {"pH 4.0": 2.3, "pH 4.5": 2.6, "pH 5.0": 2.6}
OUTLIER_DATE, OUTLIER_T = 20260608, 12.6


def _read(key: str) -> pd.DataFrame:
    path = data_root() / SOURCES[key]
    if not path.exists():
        raise FileNotFoundError(
            f"canonical source '{key}' missing: {path}. This figure reads generated artifacts and "
            f"does not reconstruct them — run the analysis that produces it.")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"canonical source '{key}' is empty: {path}")
    return frame


# ── panel A ──────────────────────────────────────────────────────────────────────────────────

def panel_a_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Per-date paired (starting, matched) D50 at g = 0.8, and the condition summary."""
    runs = _read("matched_by_run")
    sub = runs[np.isclose(runs["target_g"], TARGET_G) & np.isclose(runs["floor_frac"], FLOOR)
               & runs["D50"].notna() & runs["dlog10_D50"].notna()].copy()
    if sub.empty:
        raise ValueError(f"no eligible runs at g={TARGET_G}, floor={FLOOR}")
    # both plotted values derive from one stored quantity: start = D50 / 10**dlog10_D50
    sub["starting_D50"] = sub["D50"] / 10.0 ** sub["dlog10_D50"]

    per_date = (sub.groupby(["condition", "date"], as_index=False)
                .agg(n_runs=("run_id", "nunique"),
                     starting_D50=("starting_D50", "mean"), matched_D50=("D50", "mean")))
    per_cond = (per_date.groupby("condition", as_index=False)
                .agg(n_dates=("date", "nunique"),
                     starting_mean=("starting_D50", "mean"), starting_sd=("starting_D50", "std"),
                     matched_mean=("matched_D50", "mean"), matched_sd=("matched_D50", "std")))
    per_cond["n_runs"] = per_cond["condition"].map(
        sub.groupby("condition")["run_id"].nunique())

    # cross-check the condition means against the independently written condition artifact
    cond_art = _read("matched_by_condition")
    cond_art = cond_art[np.isclose(cond_art["target_g"], TARGET_G)
                        & np.isclose(cond_art["floor_frac"], FLOOR)]
    checks = {}
    for r in per_cond.itertuples():
        art = cond_art[cond_art["condition"].eq(r.condition)]
        checks[r.condition] = {
            "matched_D50_from_runs": round(float(r.matched_mean), 3),
            "matched_D50_artifact": round(float(art["D50_mean"].iloc[0]), 3) if len(art) else None,
            "expected_approx": EXPECTED_A[r.condition],
            "n_dates": int(r.n_dates), "n_runs": int(r.n_runs),
            "agrees_with_artifact": bool(len(art) and abs(float(r.matched_mean)
                                                          - float(art["D50_mean"].iloc[0])) < 0.05),
            "matches_expected_1dp": round(float(r.matched_mean), 1) == EXPECTED_A[r.condition]}
    bad = [c for c, v in checks.items() if not (v["agrees_with_artifact"] and v["matches_expected_1dp"])]
    if bad:
        raise ValueError(f"panel A matched D50 does not reproduce the canonical values: {bad}")
    return per_date, per_cond, {"target_g": TARGET_G, "floor_frac": FLOOR, "checks": checks}


# ── panel B ──────────────────────────────────────────────────────────────────────────────────

def panel_b_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """pH 4.0 date-level D50 trajectories and the primary cross-date median."""
    date_level = _read("date_level")
    med = _read("condition_median")
    audit = _read("outlier_audit")

    dates = date_level[date_level["ph"].eq(4.0) & date_level["D50"].notna()][
        ["ph", "target_min", "date", "D50"]].copy()
    primary = med[med["ph"].eq(4.0) & med["drawable"].astype(bool)][
        ["ph", "target_min", "D50", "summary_statistic", "n_dates", "n_runs"]].copy()
    if primary.empty:
        raise ValueError("no drawable pH 4.0 cross-date-median rows")
    stat = set(primary["summary_statistic"])
    if stat != {"cross_date_median"}:
        raise ValueError(f"primary trajectory must be the cross-date median, found {stat}")

    # the audited excursion: date-level means at the audited timepoint, from the audit artifact
    t_star = float(audit["target_min"].iloc[0])
    dlv = audit.groupby("date")[["D50", "D90"]].mean()
    audit_meta = {
        "target_min": t_star,
        "driving_date": int(dlv["D50"].idxmax()),
        "driving_date_D50": round(float(dlv["D50"].max()), 3),
        "cross_date_median_D50": round(float(dlv["D50"].median()), 3),
        "cross_date_mean_D50": round(float(dlv["D50"].mean()), 3),
        "n_contributing_dates": int(audit["date"].nunique()),
        "n_contributing_runs": int(len(audit)),
        "all_contributors_passed_working_eligibility": bool(
            audit["passed_working_eligibility"].all()),
        "underlying_observation_deleted": False,
        "point_specific_exclusion_applied": False,
    }
    kept = dates[dates["date"].eq(OUTLIER_DATE) & np.isclose(dates["target_min"], OUTLIER_T)]
    if kept.empty:
        raise ValueError(f"the {OUTLIER_DATE} pH 4.0 {OUTLIER_T} min observation is missing — it "
                         f"must be retained, not excluded")
    audit_meta["retained_point_D50"] = round(float(kept["D50"].iloc[0]), 3)
    return dates, primary, audit_meta


def build_source_data() -> tuple[pd.DataFrame, dict]:
    """Every plotted value, tagged by panel."""
    a_dates, a_cond, a_meta = panel_a_data()
    b_dates, b_primary, b_meta = panel_b_data()

    rows = []
    for r in a_dates.itertuples():
        rows.append({"panel": "A", "kind": "date_pair", "condition": r.condition, "date": r.date,
                     "n_runs": r.n_runs, "starting_D50_um": r.starting_D50,
                     "matched_D50_um": r.matched_D50, "target_g": TARGET_G})
    for r in a_cond.itertuples():
        rows.append({"panel": "A", "kind": "condition_summary", "condition": r.condition,
                     "n_dates": r.n_dates, "n_runs": r.n_runs,
                     "starting_D50_um": r.starting_mean, "starting_sd_um": r.starting_sd,
                     "matched_D50_um": r.matched_mean, "matched_sd_um": r.matched_sd,
                     "target_g": TARGET_G})
    for r in b_dates.itertuples():
        rows.append({"panel": "B", "kind": "date_trajectory", "condition": "pH 4.0",
                     "date": r.date, "time_min": r.target_min, "D50_um": r.D50})
    for r in b_primary.itertuples():
        rows.append({"panel": "B", "kind": "cross_date_median", "condition": "pH 4.0",
                     "time_min": r.target_min, "D50_um": r.D50,
                     "summary_statistic": r.summary_statistic,
                     "n_dates": r.n_dates, "n_runs": r.n_runs})
    return pd.DataFrame(rows), {"panel_a": a_meta, "panel_b": b_meta}


# ── rendering ────────────────────────────────────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [ps.FONT_FAMILY, "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.2, "axes.labelsize": 8.6, "axes.titlesize": 8.8,
        "xtick.labelsize": 7.8, "ytick.labelsize": 7.8, "legend.fontsize": 7.0,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


def _panel_a(ax, per_date, per_cond):
    offs = (-0.17, 0.17)
    for i, cond in enumerate(CONDITIONS):
        c = COLOR[cond]
        for r in per_date[per_date["condition"].eq(cond)].itertuples():
            ax.plot([i + offs[0], i + offs[1]], [r.starting_D50, r.matched_D50],
                    "-", color=c, lw=0.8, alpha=0.45, zorder=2)
            ax.plot([i + offs[0], i + offs[1]], [r.starting_D50, r.matched_D50],
                    "o", color=c, ms=2.8, alpha=0.55, zorder=2)
        s = per_cond[per_cond["condition"].eq(cond)].iloc[0]
        ax.errorbar([i + offs[0]], [s.starting_mean], yerr=[s.starting_sd], fmt="o", ms=5.2,
                    color=c, ecolor=c, elinewidth=1.0, capsize=2.6,
                    markerfacecolor="white", markeredgewidth=1.4, zorder=5)
        ax.errorbar([i + offs[1]], [s.matched_mean], yerr=[s.matched_sd], fmt="o", ms=5.2,
                    color=c, ecolor=c, elinewidth=1.0, capsize=2.6, zorder=5)
        ax.plot([i + offs[0], i + offs[1]], [s.starting_mean, s.matched_mean], "-",
                color=c, lw=1.8, zorder=4)
        ax.text(i, 0.035, f"{int(s.n_dates)} dates · {int(s.n_runs)} runs",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=6.2, color="0.35")
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS)
    ax.set_xlim(-0.5, len(CONDITIONS) - 0.5)
    ax.set_ylabel("q3 $D_{50}$ (µm)")
    ax.set_title(f"Starting vs matched extent (g = {TARGET_G:g})", fontsize=8.8, pad=4)
    ax.set_ylim(0, max(per_date["starting_D50"].max(), per_date["matched_D50"].max()) * 1.18)
    from matplotlib.lines import Line2D
    # large symbols are condition means; the thin paired lines are preparation-date means
    ax.legend(handles=[Line2D([0], [0], marker="o", color="0.35", ls="none", ms=5.2,
                              markerfacecolor="white", markeredgewidth=1.4,
                              label="Starting q3 $D_{50}$"),
                       Line2D([0], [0], marker="o", color="0.35", ls="none", ms=5.2,
                              label=f"At g = {TARGET_G:g}"),
                       Line2D([0], [0], color="0.55", lw=0.9, alpha=0.6,
                              label="preparation-date mean")],
              loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False,
              fontsize=6.8, handletextpad=0.4, borderpad=0.2, labelspacing=0.32,
              columnspacing=1.2)
    _clean(ax)


def _panel_b(ax, dates, primary, audit_meta):
    c = COLOR["pH 4.0"]
    for date, g in dates.groupby("date"):
        g = g.sort_values("target_min")
        ax.plot(g["target_min"], g["D50"], "-", lw=0.85, alpha=0.55, color=c, zorder=2)
    p = primary.sort_values("target_min")
    ax.plot(p["target_min"], p["D50"], "-", lw=2.2, color=c, zorder=4)

    # the single-date excursion: restrained open marker, explained in the caption
    hit = dates[dates["date"].eq(OUTLIER_DATE) & np.isclose(dates["target_min"], OUTLIER_T)]
    ax.plot(hit["target_min"], hit["D50"], "o", ms=7.5, mfc="none", mec="#B03A2E",
            mew=1.4, zorder=6)

    ax.set_yscale("log")
    # plain readable decades rather than 10^0 / 10^1 exponent labels
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("q3 $D_{50}$ (µm)")
    ax.set_title("pH 4.0 date-level reproducibility", fontsize=8.8, pad=4)
    ax.grid(axis="y", which="major", color="0.92", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=c, lw=0.85, alpha=0.6,
                              label="preparation-date mean"),
                       Line2D([0], [0], color=c, lw=2.2, label="cross-date median"),
                       Line2D([0], [0], marker="o", color="#B03A2E", ls="none", ms=6,
                              mfc="none", mew=1.4, label="single-date excursion")],
              loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False,
              fontsize=6.8, handletextpad=0.5, borderpad=0.2, labelspacing=0.32,
              columnspacing=1.2)
    _clean(ax)


def render(out_dir: Path, formats=("pdf", "png", "svg")):
    _style()
    a_dates, a_cond, a_meta = panel_a_data()
    b_dates, b_primary, b_meta = panel_b_data()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.6))
    _panel_a(axA, a_dates, a_cond)
    _panel_b(axB, b_dates, b_primary, b_meta)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.30, wspace=0.26)
    for ax, letter in ((axA, "A"), (axB, "B")):
        box = ax.get_position()
        fig.text(box.x0 - 0.062, box.y1 + 0.045, letter, fontsize=10.0,
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

**Supporting Information. Matched-extent q3 comparison and the pH 4.0 late-time reliability
boundary.** q3 is the PAQXOS-inverted, volume-weighted *relative composition of the particles still
detected*. It is not a mass measurement and it is not independent of the laser-diffraction
acquisition. Nested runs were averaged within preparation date before any cross-date summary, so
the independent units are preparation-date means.

**(A)** q3 \\(D_{{50}}\\) at each run's starting distribution and at a matched angular-loss extent
of g = {g:g} — a {loss:.0f} % loss of total angular signal, the only extent reached by all three pH
conditions. The starting q3 distribution is the average of that run's first three valid (eligible)
frames. Thin paired lines are preparation-date means; the large open and filled symbols with bars
are the condition mean ± SD between preparation-date means. Matched-extent \\(D_{{50}}\\) was
{d40:.1f}, {d45:.1f}, and {d50:.1f} µm at pH 4.0, 4.5, and 5.0, close to the corresponding
starting values.
pH 5.0 reached g = {g:g} on all three preparation dates but with {n5_runs} contributing runs.
Deeper extents are not shown because pH 5.0 does not reach them.

**(B)** Eligible pH 4.0 q3 \\(D_{{50}}\\) against time. Thin lines are preparation-date means and
the heavy line is the cross-date median. The open symbol marks a single-date excursion at
{t_star:g} min, where preparation date {driving_date} reached \\(D_{{50}}\\) = {driving_D50:.1f} µm
while the cross-date median was {median_D50:.1f} µm. That observation passed the same working
eligibility as every other plotted point and was retained, but it was not reproduced on the other
preparation dates. The cross-date median was adopted prospectively for every pH, timepoint, and
percentile, and for both model and observed trajectories; it was not a point-specific exclusion,
and no eligibility or signal threshold was changed. The axis is logarithmic so that the ordinary
1--6 µm behaviour remains legible alongside the excursion.

These panels bound the reliability of the q3 comparison. They do not establish aggregation,
deaggregation, uniform dissolution, or the absence of redistribution.
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--formats", default="pdf,png,svg")
    args = p.parse_args(argv)
    out = args.output_dir
    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())

    table, meta = build_source_data()
    written = render(out, formats)
    csv_path = out / f"{STEM}_source_data.csv"
    table.to_csv(csv_path, index=False)

    cond = table[table["kind"].eq("condition_summary")].set_index("condition")
    b = meta["panel_b"]
    caption = CAPTION.format(
        stem=STEM, g=TARGET_G, loss=(1 - TARGET_G) * 100,
        d40=cond.loc["pH 4.0", "matched_D50_um"], d45=cond.loc["pH 4.5", "matched_D50_um"],
        d50=cond.loc["pH 5.0", "matched_D50_um"],
        n5_runs=int(cond.loc["pH 5.0", "n_runs"]),
        t_star=b["target_min"], driving_date=b["driving_date"],
        driving_D50=b["driving_date_D50"], median_D50=b["cross_date_median_D50"])
    caption_path = out / f"{STEM}_caption.md"
    caption_path.write_text(caption)

    prov = provenance_record(
        "manuscript_si_q3_matched_extent", study_root=data_root() / PH_STUDY,
        uv_ph_values=(),                        # no UV assay enters this figure
        figure_stem=STEM,
        sources={k: str(data_root() / v) for k, v in SOURCES.items()},
        settings={"target_g": TARGET_G, "floor_frac": FLOOR,
                  "floor_source": "q3_matched_extent.DEFAULT_FLOOR",
                  "aggregation": "nested runs averaged within preparation date, then dates "
                                 "weighted equally; independent units are preparation-date means",
                  "panel_b_primary_statistic": "cross_date_median"},
        numerical_checks=meta,
        scope={"observable": "PAQXOS-inverted q3 — relative composition of detected particles",
               "is_mass_measurement": False,
               "independent_of_ld_acquisition": False,
               "claims_excluded": ["aggregation", "deaggregation", "uniform dissolution",
                                   "Mie-derived mechanism", "absence of redistribution",
                                   "independent validation"]})
    prov_path = write_provenance(out / f"{STEM}_provenance.json", prov)

    print(f"Panel A — q3 D50 at g={TARGET_G:g} (floor {FLOOR}):")
    for c in CONDITIONS:
        r = cond.loc[c]
        print(f"  {c}: start {r.starting_D50_um:.2f} → matched {r.matched_D50_um:.2f} µm "
              f"({int(r.n_dates)} dates, {int(r.n_runs)} runs)")
    print(f"\nPanel B — audited timepoint {b['target_min']:g} min:")
    print(f"  driving date {b['driving_date']} D50 {b['driving_date_D50']:.1f} µm | "
          f"cross-date median {b['cross_date_median_D50']:.1f} | mean {b['cross_date_mean_D50']:.1f}")
    print(f"  retained point present: D50 {b['retained_point_D50']:.1f} µm | "
          f"all contributors passed eligibility: {b['all_contributors_passed_working_eligibility']}")
    for path in [*written, csv_path, caption_path, prov_path]:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
