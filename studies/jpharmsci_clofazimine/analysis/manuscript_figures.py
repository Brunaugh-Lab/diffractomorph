"""Command-line driver for the manuscript-ready figures (Figures 2 and 3).

Renders the publication figures from the raw measurement ``.rtf`` files, declared run metadata, and
canonical UV CSVs into a supplied output directory, writing a per-panel source-data CSV for
every numerical panel plus a ``README.md`` recording the exact command, git commit, input files, panel
transformations, and draft captions. It never edits the existing diagnostic figures. The aggregate
KWW tables are recomputed with the explicitly declared circulation-start policy used by the current
manuscript; the UV and matched-``g`` results are re-expressed from their canonical outputs.

The manuscript set is the pH-dependent **angular-scattering kinetics** figure
(``Figure_pH_angular_scattering_kinetics``) and the **UV dissolved-mass and recovery** figure
(``Figure_UV_dissolved_mass_and_recovery``, with its optical-vs-UV supplement
``Figure_optical_decay_vs_uv_recovery``). Figure numbering is kept provisional — output names are
descriptive, not numbered. The matched-extent redistribution figure is retained in
:mod:`diffractomorph_pipeline.figures.manuscript` and can be rendered with ``--figure 5`` but is not part
of the default manuscript set.

Both paths are supplied per invocation; nothing here is hard-coded. The generated PNG/PDF
outputs belong in the manuscript folder, not in this repository::

    python analysis/manuscript_figures.py \
      --study-root "<data_root>/disso_experiments/ph_dependent_dissolution_study" \
      --output-dir "<manuscript folder>/manuscript_outputs" \
      --formats png,pdf --figure all

``<data_root>`` is the configured data corpus (``DFM_DATA_ROOT`` or ``.dfm.toml``).
Run with the pipeline venv.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from figures import manuscript as M
from ph_aggregate import build_tables as build_ph_aggregate_tables

DEFAULT_SET = (2, 3)          # the manuscript figure set; Figure 5 is opt-in via --figure 5


def _git_commit() -> str:
    try:
        here = Path(__file__).resolve().parent
        return subprocess.check_output(["git", "-C", str(here), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "(unavailable)"


def _file_stamp(path: Path) -> str:
    p = Path(path)
    if not p.exists():
        return f"{p}  — MISSING"
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    return f"{p.name}  ({p.stat().st_size} B, sha256:{h})"


def _input_files(study_root: Path, figures) -> list[Path]:
    p = M._paths(study_root)
    keys = set()
    if 2 in figures:
        keys |= {"run_metadata"}
    if 3 in figures:
        keys |= {"run_metadata", "uv_timecourse", "recovery"}
    if 5 in figures:
        keys |= {"run_metadata", "mg_coverage"}
    files = [p[k] for k in sorted(keys)]
    if set(figures) & {2, 3, 5}:
        files.extend(sorted(study_root.glob("ph_*/*/*measurement*.rtf")))
    return files


def _render(figures, study_root, figures_dir, tables_dir, source_dir, formats):
    out, discrepancies = {}, []
    current_fits = current_by_date = current_by_condition = None
    if set(figures) & {2, 3}:
        current_fits, current_by_date, current_by_condition = build_ph_aggregate_tables(study_root)
        current_fits.to_csv(tables_dir / "aggregate_kww_by_run.csv", index=False)
        current_by_date.to_csv(tables_dir / "aggregate_kww_by_independent_unit.csv", index=False)
        current_by_condition.to_csv(tables_dir / "aggregate_kww_by_condition.csv", index=False)
    if 2 in figures:
        d2 = M.figure2_data(study_root, fits=current_fits, by_date=current_by_date)
        out["Figure 2"] = M.render_figure2(d2, figures_dir, source_dir, formats)
        # Regression checks against the current manuscript condition summary.
        expect = {"pH 4.0": (1.13, 0.845, 82.5), "pH 4.5": (3.06, 0.765, 85.8),
                  "pH 5.0": (2.27, 0.952, 25.2)}
        cc = d2.panelBCD_cond.set_index("condition")
        for cond, (mr, beta, depth) in expect.items():
            got = (cc.loc[cond, "mean_relax_min_mean"], cc.loc[cond, "beta_mean"],
                   cc.loc[cond, "optical_decay_depth_pct_mean"])
            if abs(got[0] - mr) > 0.05 or abs(got[1] - beta) > 0.01 or abs(got[2] - depth) > 1.5:
                discrepancies.append(f"Figure 2 {cond}: got {got}, expected ≈ ({mr}, {beta}, {depth})")
        out["_fig2_data"] = d2
    if 3 in figures:
        d3 = M.figure3_data(study_root, by_date=current_by_date)
        out["Figure 3"] = M.render_figure3(d3, figures_dir, source_dir, tables_dir, formats)
        # date-balanced plateau recovery after the UV-parser fix (all 27 runs, 9/9/9)
        expect_rec = {4.0: 89.7, 4.5: 87.0, 5.0: 33.4}
        pb = d3.panelB_cond.set_index("ph")
        for ph, exp in expect_rec.items():
            got = float(pb.loc[ph, "recovery_mean"])
            if abs(got - exp) > 2.0:
                discrepancies.append(f"Figure 3 pH {ph}: plateau recovery {got:.1f}%, expected ≈ {exp}%")
        out["_fig3_data"] = d3
    if 5 in figures:
        d5 = M.figure5_data(study_root)
        out["Figure 5"] = M.render_figure5(d5, figures_dir, source_dir, formats)
        out["_fig5_data"] = d5
    return out, discrepancies


def _write_readme(readme_path: Path, *, command, commit, study_root, output_dir, figures,
                  input_files, out, discrepancies):
    lines = []
    lines.append("# Manuscript figure outputs\n")
    lines.append("Generated by `analysis/manuscript_figures.py`. This directory is regenerated from the "
                 "canonical study CSVs; do not hand-edit the figures, and do not commit these files to "
                 "the code repository.\n")
    lines.append("## Provenance\n")
    lines.append(f"- **Command:** `{command}`")
    lines.append(f"- **Git commit:** `{commit}`")
    lines.append(f"- **Study root:** `{study_root}`")
    lines.append(f"- **Output dir:** `{output_dir}`")
    lines.append(f"- **Figures rendered:** {', '.join('Figure ' + str(f) for f in figures)}\n")
    lines.append("### Input files (with size and content hash)\n")
    for f in input_files:
        lines.append(f"- {_file_stamp(f)}")
    lines.append("")
    lines.append("## Experimental unit and uncertainty\n")
    lines.append("The primary experimental unit is the **independent suspension-preparation date**. "
                 "The three runs within a (date × condition) cell are **nested replicate runs**. Every "
                 "condition-level number is built *day-first*: runs are averaged within (date × "
                 "condition), then summarized across the three date-level means. **All condition-level "
                 "error bars/bands are the between-date SD (n = 3 dates)**, not the run-to-run SD "
                 "(n = 9). Total angular signal ΣI, Copt, and q3 are particle-side optical quantities — "
                 "they are not dissolved drug mass, and no detector channel is assigned a single "
                 "particle diameter.\n")
    lines.append("## Panel-by-panel transformations\n")
    if 2 in figures:
        lines.append("### Figure 2 — pH-dependent angular-scattering kinetics\n")
        lines.append("- **A** — each run's total angular signal ΣI(t) is reconstructed from its `.rtf` "
                     "and expressed as `100·ΣI(t)/I0_fit` (percent of the fitted back-extrapolated "
                     "start `i0_fit`, **not** the first measured frame). Per-run curves are interpolated "
                     "onto a common time grid within their own support, averaged across the three nested "
                     "runs within a date, then summarized as the condition mean (thick line) ± "
                     "between-date SD (light band); the three date-mean trajectories are the thin lines.")
        lines.append("- **B/C/D** — date-level KWW descriptors recomputed from the raw RTF exports "
                     "using the declared circulation-start policy: "
                     "mean relaxation time ⟨t⟩ (min), optical decay depth (%), and stretch exponent β. "
                     "Three date-level points per condition (deterministic jitter) plus the condition "
                     "mean ± between-date SD. These are empirical KWW descriptors, not "
                     "complete-dissolution rate, dissolved fraction, or a uniform/heterogeneous claim.")
    if 3 in figures:
        lines.append("\n### Figure 3 — UV dissolved mass and recovery\n")
        lines.append("- **A** — corrected UV dissolved CFZ as percent of assayed injected mass, from "
                     "`uv_timecourse_all.csv`. Light points are measured run observations; thin lines "
                     "are date-level means; thick lines are condition means where time-point support is "
                     "comparable. The emphasized trajectories are anchored at the **t = 0 dosing "
                     "boundary (0 % dissolved)**, which is a known dosing assumption, not an acquired UV "
                     "sample (the earliest UV sample is at t = 2 min).")
        lines.append("- **B** — plateau recovered fraction from `recovery_corrected.csv`, restricted to "
                     "plateau-eligible runs (`basis == plateau(t>=10)`); `single(~2min)` early-only "
                     "records are excluded. Run-level eligible values → date-condition means → condition "
                     "mean ± between-date SD (n = 3 dates).")
        lines.append("- **Figure S1** — optical decay depth vs eligible UV plateau recovery, date-level "
                     "(one point per date × condition). No calibration/regression line is drawn; the two "
                     "axes measure different things.")
    if 5 in figures:
        lines.append("\n### Figure 5 — matched-extent redistribution (opt-in; not in the manuscript set)\n")
        lines.append("- **A/B** — a single residual statistic `r_c = (x_c − g·x0_c)/Σ x0_c` evaluated at "
                     "each run's endpoint (A) and at the common-support `g = 0.8` (B). Figure S2 is the "
                     "coverage matrix.")
    lines.append("\n## Known limitations\n")
    lines.append("- The pH 5.0 relaxation time (Fig 2B) describes only its **shallow observed optical "
                 "decay** (optical depth ≈ 25 %); it is not a complete-dissolution time.")
    lines.append("- Optical decay depth is an optical quantity (fraction of starting scattering "
                 "removed); it is not dissolved fraction and is not equivalent to UV recovery "
                 "(Fig S1 plots them together only for visual comparison).")
    lines.append("- Between-date SD with n = 3 dates is a small-sample descriptive spread; no "
                 "significance testing is claimed.")
    lines.append("\n## Draft captions\n")
    if 2 in figures:
        lines.append("**Figure 2. pH-dependent angular-scattering kinetics.** (A) Total angular "
                     "scattering signal ΣI as a percentage of the fitted back-extrapolated start, "
                     "``100·ΣI(t)/I0,fit``. Thin lines are preparation-date means (three nested runs "
                     "averaged); thick lines are condition means across three dates and the shaded band "
                     "is the between-date SD. (B) Mean KWW relaxation time ⟨t⟩, (C) optical decay depth, "
                     "and (D) KWW stretch exponent β, each shown as three date-level values per "
                     "condition (points) with the condition mean ± between-date SD (n = 3 dates). "
                     "Angular signal is a particle-side optical quantity, not dissolved mass; detector "
                     "channels are not one-to-one particle-size bins. The pH 5.0 relaxation time "
                     "characterizes only its shallow observed optical decay (depth ≈ 25 %).\n")
    if 3 in figures:
        lines.append("**Figure 3. UV dissolved-mass recovery.** (A) Corrected UV dissolved CFZ as a "
                     "percentage of the assayed injected mass. Light points are individual run "
                     "observations, thin lines are preparation-date means, and thick lines are condition "
                     "means; trajectories are anchored at the t = 0 dosing boundary (0 % dissolved, a "
                     "dosing assumption rather than an acquired sample). (B) Plateau recovered fraction "
                     "from plateau-eligible measurements (t ≥ 10 min); early-only samples are excluded. "
                     "Points are date-level means; error bars are the between-date SD (n = 3 dates). "
                     "pH 4.0 and 4.5 recover ~90 % of the injected dose while pH 5.0 plateaus near "
                     "~33 %.\n")
        lines.append("**Figure S1. Optical decay depth versus UV plateau recovery (date-level).** Each "
                     "point is one preparation date × condition. pH 4.0/4.5 sit high on both axes; "
                     "pH 5.0 sits low on both. The axes measure different quantities (optical scattering "
                     "loss vs dissolved-mass recovery) and no calibration is implied.\n")
    if 5 in figures:
        lines.append("**Figure 5. Matching observed particle-signal loss removes apparent pH-dependent "
                     "channel-profile differences.** (A) At the experimental endpoint the conditions "
                     "reached substantially different remaining-signal fractions (g_end annotated). "
                     "(B) At the only common-support extent, g = 0.8, the condition profiles "
                     "substantially overlap. Thin lines are preparation-date means; thick lines are "
                     "condition means. Detector channels are angularly ordered and are not one-to-one "
                     "particle-size bins; channels 1 and 31 carry the largest detector-edge variability. "
                     "(Not part of the current manuscript figure set.)\n")
    if discrepancies:
        lines.append("## ⚠ Discrepancies vs expected values\n")
        for d in discrepancies:
            lines.append(f"- {d}")
    else:
        lines.append("## Validation\n")
        lines.append("- All rendered figures reproduced the expected canonical values within rounding.")
    readme_path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study-root", required=True, help="pH-study root directory (canonical CSVs + rtf)")
    ap.add_argument("--output-dir", required=True, help="manuscript_outputs directory (created if absent)")
    ap.add_argument("--formats", default="png,pdf", help="comma-separated raster/vector formats")
    ap.add_argument("--figure", default="all", choices=["2", "3", "5", "all"],
                    help="which figure to render ('all' = the manuscript set: Figures 2 and 3)")
    args = ap.parse_args(argv)

    study_root = Path(args.study_root)
    output_dir = Path(args.output_dir)
    formats = tuple(s.strip() for s in args.formats.split(",") if s.strip())
    figures = list(DEFAULT_SET) if args.figure == "all" else [int(args.figure)]

    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    source_dir = output_dir / "source_data"
    for d in (figures_dir, tables_dir, source_dir):
        d.mkdir(parents=True, exist_ok=True)

    command = "python " + " ".join(sys.argv[:1] + (argv if argv is not None else sys.argv[1:]))
    commit = _git_commit()
    out, discrepancies = _render(figures, study_root, figures_dir, tables_dir, source_dir, formats)

    _write_readme(output_dir / "README.md", command=command, commit=commit, study_root=study_root,
                  output_dir=output_dir, figures=figures,
                  input_files=_input_files(study_root, figures), out=out, discrepancies=discrepancies)

    # ── report ──
    print(f"Rendered {', '.join('Figure ' + str(f) for f in figures)} → {figures_dir}")
    for key in ("Figure 2", "Figure 3", "Figure 5"):
        if key in out:
            for p in out[key].get("figure", []):
                print(f"  fig    {p}")
            for p in out[key].get("tables", []):
                print(f"  table  {p}")
            for p in out[key].get("sources", []):
                print(f"  source {p}")
    if "_fig2_data" in out:
        cc = out["_fig2_data"].panelBCD_cond
        print("\nFigure 2 condition summary (mean ± between-date SD, n=3):")
        print(cc.to_string(index=False))
    if "_fig3_data" in out:
        print("\nFigure 3 plateau recovery (mean ± between-date SD, n=3):")
        print(out["_fig3_data"].panelB_cond.to_string(index=False))
    if "_fig5_data" in out:
        print("\nFigure 5 endpoint g by condition (mean ± SD across dates):")
        print(out["_fig5_data"].endpoint_gend_cond.to_string(index=False))
    if discrepancies:
        print("\n⚠ DISCREPANCIES vs expected values:")
        for d in discrepancies:
            print(f"  - {d}")
    else:
        print("\nAll figures reproduced the expected canonical values within rounding.")
    print(f"\nREADME + captions → {output_dir / 'README.md'}")
    return 1 if discrepancies else 0


if __name__ == "__main__":
    raise SystemExit(main())
