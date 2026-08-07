"""CLI entry points for diffractomorph_pipeline.

Pipeline commands, run in order, each consuming the previous step's output:

    dfm-ingest        raw exports   → tidy CSV
    dfm-noise-filter  raw exports   → admitted channels + per-channel grid + overlay
    dfm-extract       admitted channels → parameters CSV
    dfm-aggregate-kww manifest → manuscript-authoritative aggregate KWW summaries

The listed ingestion, noise-filter, extraction, and aggregate-KWW commands are
implemented. Any retained experimental stage fails explicitly when invoked.
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd

from diffractomorph_pipeline import ingest, extract


def diagnostics_main(argv=None):
    """Write a support record that deliberately excludes research data and local paths."""
    parser = argparse.ArgumentParser(
        description="Create a privacy-safe DiffractoMorph diagnostic JSON for issue reports.",
    )
    parser.add_argument("manifest", type=Path, help="Project manifest to validate.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file.")
    parser.add_argument(
        "--inspect-runs", action="store_true",
        help="Read declared runs and include only frame/channel dimensions, never values or paths.",
    )
    args = parser.parse_args(argv)

    from diffractomorph_pipeline import __version__
    from diffractomorph_pipeline.study import ManifestError, load_manifest

    try:
        project = load_manifest(args.manifest)
    except (ManifestError, KeyError) as exc:
        parser.error(str(exc))
    role_summary = {
        role: {
            "profile_id": profile.profile_id,
            "declaration": (
                "not_applicable" if profile.not_applicable_reason
                else "file" if profile.path is not None else "inline"
            ),
        }
        for role, profile in project.profiles.items()
    }
    run_kinds = {}
    adapters = set()
    for spec in project.runs:
        run_kinds[spec.run_kind] = run_kinds.get(spec.run_kind, 0) + 1
        adapters.add(spec.adapter)
    payload = {
        "diagnostic_schema_version": 1,
        "privacy": {
            "contains_raw_data": False,
            "contains_signal_values": False,
            "contains_local_paths": False,
            "contains_sample_or_independent_unit_ids": False,
        },
        "environment": {
            "diffractomorph_version": __version__,
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "operating_system": platform.system(),
            "machine": platform.machine(),
        },
        "project": {
            "project_id_sha256": hashlib.sha256(project.project_id.encode()).hexdigest(),
            "schema_version": project.schema_version,
            "independent_unit_type": project.independent_unit,
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "run_count": len(project.runs),
            "run_kinds": run_kinds,
            "adapters": sorted(adapters),
            "profiles": role_summary,
        },
    }
    if args.inspect_runs:
        payload["run_dimensions"] = [
            {
                "ordinal": index,
                "kind": spec.run_kind,
                "adapter": spec.adapter,
                "frames": int(run.signal.shape[0]),
                "channels": int(run.signal.shape[1]),
                "acquisition_variables": sorted(run.acquisition),
            }
            for index, (spec, run) in enumerate(
                zip(project.runs, project.read_all_runs()), start=1
            )
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote privacy-safe diagnostics to {args.output}")


# ── Public project manifest ──────────────────────────────────────────────────

def manifest_main(argv=None):
    """Validate an explicit project manifest and optionally inspect every run."""
    parser = argparse.ArgumentParser(
        description="Validate a DiffractoMorph project manifest without selecting hidden profiles.",
    )
    parser.add_argument("manifest", type=Path, nargs="?", help="YAML project manifest.")
    parser.add_argument(
        "--example", action="store_true",
        help="Use the redistributable example included in the installed package.",
    )
    parser.add_argument(
        "--inspect-runs", action="store_true",
        help="Read each declared source through its explicit adapter and print its shape.",
    )
    args = parser.parse_args(argv)

    from diffractomorph_pipeline.study import (
        ManifestError, bundled_example_manifest, load_manifest,
    )

    if args.example == (args.manifest is not None):
        parser.error("provide exactly one of MANIFEST or --example")
    manifest_path = bundled_example_manifest() if args.example else args.manifest
    try:
        project = load_manifest(manifest_path)
    except (ManifestError, KeyError) as exc:
        parser.error(str(exc))
    print(f"project={project.project_id} schema={project.schema_version} runs={len(project.runs)}")
    print(f"independent_unit={project.independent_unit} data_root={project.data_root}")
    print("profiles=" + ",".join(
        f"{role}:{profile.profile_id}" for role, profile in project.profiles.items()
    ))
    if args.inspect_runs:
        for spec in project.runs:
            run = project.read_run(spec.run_id)
            print(
                f"run={spec.run_id} adapter={spec.adapter} sample={spec.sample_id} "
                f"independent_unit={spec.independent_unit_id} frames={run.signal.shape[0]} "
                f"channels={run.signal.shape[1]}"
            )


def aggregate_kww_main(argv=None):
    """Run the manuscript-authoritative aggregate KWW estimand from a manifest."""
    parser = argparse.ArgumentParser(
        description="Fit raw all-channel aggregate KWW trajectories with independent-unit summaries.",
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-correction", choices=("auto", "required", "off"), default="auto",
        help="Apply synchronized detector/acquisition correction when available.",
    )
    args = parser.parse_args(argv)

    from diffractomorph_pipeline.processing import (
        AggregateKWWConfig, ArtifactCorrectionConfig, correct_artifacts, fit_aggregate_kww,
    )
    from diffractomorph_pipeline.study import load_manifest, summarize_hierarchy

    project = load_manifest(args.manifest)
    analysis_profile = project.require_profile("analysis")
    required_analysis = {
        "channel_set", "stored_reference_subtraction", "start_boundary", "upward_hampel",
        "tau_bounds_min", "beta_bounds",
    }
    missing_analysis = sorted(required_analysis - set(analysis_profile.parameters))
    if missing_analysis:
        parser.error("analysis profile missing: " + ", ".join(missing_analysis))
    try:
        config = AggregateKWWConfig.from_profile(analysis_profile.parameters)
    except ValueError as exc:
        parser.error(str(exc))
    artifact_parameters = analysis_profile.parameters.get("artifact_correction")
    artifact_config = (
        ArtifactCorrectionConfig.from_profile(artifact_parameters)
        if artifact_parameters is not None else None
    )
    if (
        config.start_policy != "first_frame"
        and artifact_config is not None
        and args.artifact_correction != "off"
    ):
        parser.error(
            "concordant acquisition-start selection and artifact_correction are "
            "alternative startup treatments; rerun with --artifact-correction off"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in project.runs:
        if spec.run_kind != "measurement":
            continue
        run = project.read_run(spec.run_id)
        if args.artifact_correction == "required" and artifact_config is None:
            parser.error("analysis profile requires an explicit artifact_correction mapping")
        correction_status = "off" if args.artifact_correction == "off" else "skipped_no_profile"
        if args.artifact_correction == "off":
            pass
        elif artifact_config is not None and artifact_config.acquisition_variable not in run.acquisition:
            correction_status = "skipped_missing_acquisition"
            if args.artifact_correction == "required":
                parser.error(
                    f"run {spec.run_id!r} lacks acquisition variable "
                    f"{artifact_config.acquisition_variable!r}"
                )
        elif args.artifact_correction != "off" and artifact_config is not None:
            corrected = correct_artifacts(run, artifact_config)
            run = corrected.run
            corrected.ledger.to_csv(args.output_dir / f"{spec.run_id}_frame_ledger.csv", index=False)
            correction_status = "applied"
        try:
            result = fit_aggregate_kww(run, config)
        except KeyError as exc:
            parser.error(f"run {spec.run_id!r}: {exc}")
        condition = spec.metadata.get("condition")
        if condition is None:
            parser.error(f"measurement run {spec.run_id!r} metadata requires condition")
        rows.append({"condition": str(condition), "artifact_correction": correction_status,
                     **result.to_row()})
    if not rows:
        parser.error("manifest contains no measurement runs")

    by_run = pd.DataFrame(rows)
    values = ("tau_min", "beta", "mean_relax_min", "t50_min", "optical_decay_depth_pct", "i0_fit")
    summary = summarize_hierarchy(by_run, value_columns=values)
    unit_start_counts = (
        by_run.assign(_started_late=by_run["start_index"].gt(0).astype(int))
        .groupby(["condition", "independent_unit_id"], as_index=False)
        .agg(n_runs_started_late=("_started_late", "sum"))
    )
    independent_units = summary.independent_units.merge(
        unit_start_counts,
        on=["condition", "independent_unit_id"],
        how="left",
        validate="one_to_one",
    )
    condition_start_counts = (
        unit_start_counts.assign(
            _unit_started_late=unit_start_counts["n_runs_started_late"].gt(0).astype(int),
        )
        .groupby("condition", as_index=False)
        .agg(
            n_runs_started_late=("n_runs_started_late", "sum"),
            n_independent_units_started_late=("_unit_started_late", "sum"),
        )
    )
    conditions = summary.conditions.merge(
        condition_start_counts,
        on="condition",
        how="left",
        validate="one_to_one",
    )
    by_run.to_csv(args.output_dir / "aggregate_kww_by_run.csv", index=False)
    independent_units.to_csv(
        args.output_dir / "aggregate_kww_by_independent_unit.csv", index=False,
    )
    conditions.to_csv(args.output_dir / "aggregate_kww_by_condition.csv", index=False)
    print(
        f"runs={len(by_run)} independent_units={summary.independent_units['independent_unit_id'].nunique()} "
        f"conditions={len(summary.conditions)} reference_mode={config.reference_mode}"
    )
    print(f"Wrote Gate 2 aggregate outputs to {args.output_dir}")


# ── Module 1: Ingestion ──────────────────────────────────────────────────────

def _collect_rtfs(paths):
    """Expand file/dir inputs into a sorted list of .rtf paths."""
    out = []
    for p in paths:
        if p.is_dir():
            out += sorted(p.glob("*.rtf")) + sorted(p.glob("*.RTF"))
        else:
            out.append(p)
    return out


def ingest_main(argv=None):
    """Entry point for ``dfm-ingest`` — raw PAQXOS RTF → RawRun + CSV mirror (Step 0).

    Writes one ``<run>.csv`` + ``<run>_meta.json`` per input and prints a
    one-line summary (frames, run_kind, Copt range, flags) per file.
    """
    parser = argparse.ArgumentParser(
        description="Extract Sympatec PAQXOS RTF exports into tidy CSV + RawRun (Step 0).",
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="RTF file(s) or directory(ies).")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Where to write the CSV + meta. Default: CWD.")
    parser.add_argument("--run-kind", choices=["measurement", "blank"], default=None,
                        help="Force run kind (default: infer from filename).")
    parser.add_argument("--no-csv", action="store_true", help="Summary only; don't write CSVs.")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = _collect_rtfs(args.inputs)
    if not files:
        print("No .rtf files found.", file=sys.stderr); sys.exit(1)

    print(f"{'file':<48} {'kind':<11} {'frames':>6} {'copt range':>16}  flags")
    print("-" * 100)
    for path in files:
        run = ingest.extract_run(path, run_kind=args.run_kind)
        copt = run.copt[~pd.isna(run.copt)]
        copt_rng = f"{copt.max():.2f}→{copt.min():.2f}" if copt.size else "n/a"
        bad = [k for k in ("ref_static",) if not run.flags[k]]
        notes = []
        if run.flags["reverse_order_detected"]:
            notes.append("rev-order")
        if not run.flags["ref_static"]:
            notes.append("REF-NONSTATIC")
        if run.flags["max_gap_min"] > 2:
            notes.append(f"gap{run.flags['max_gap_min']:.1f}m")
        if run.flags["dropped_frames"]:
            notes.append(f"dropped{run.flags['dropped_frames']}")
        print(f"{path.name:<48} {run.run_kind:<11} {run.flags['n_frames']:>6} "
              f"{copt_rng:>16}  {','.join(notes) or '-'}")
        if not args.no_csv:
            run.write_csv(args.output_dir / f"{path.stem}.csv")
    if not args.no_csv:
        print(f"\nWrote CSV + meta to {args.output_dir}/")


# ── Step 1: Channel triage ───────────────────────────────────────────────────

def noise_filter_main(argv=None):
    """Entry point for ``dfm-noise-filter`` — admit channels by per-channel noise.

    Takes one or more raw PAQXOS exports, applies the per-channel noise filter
    (which channels carry real directional change above their own noise floor),
    prints the admitted/masked counts, and (unless ``--no-figure``) writes the
    per-channel noise grid + the admitted-channel overlay next to each input.
    """
    from diffractomorph_pipeline.noise_filter import noise_filter
    from diffractomorph_pipeline.figures.diagnostic import (
        plot_channel_noise_grid, plot_channel_overlay,
    )

    parser = argparse.ArgumentParser(
        description="Noise filter: admit channels with real directional change.",
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="Raw PAQXOS export file(s).")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Summary CSV path. Default: noise_filter_summary.csv in CWD.")
    parser.add_argument("--noise-floor", type=float, default=None,
                        help="External per-channel floor (counts); overrides the per-run heuristic.")
    parser.add_argument("--noise-surface", nargs="?", const="default", default=None,
                        help="Use the CFZ-pH-7 noise surface for super-floor channel "
                             "selection. Bare flag = packaged default; or pass a path.")
    parser.add_argument("--z-thresh", type=float, default=4.0,
                        help="Admission threshold on per-channel drift z (default 4).")
    parser.add_argument("--no-figure", action="store_true", help="Skip diagnostic figures.")
    args = parser.parse_args(argv)

    surface = None
    if args.noise_surface is not None:
        from diffractomorph_pipeline.noise_surface import load_surface
        surface = load_surface(None if args.noise_surface == "default" else Path(args.noise_surface))
        print(f"(super-floor selection via noise surface: k={surface.k:.3g} p={surface.p:+.2f})")

    rows = []
    print(f"{'file':<48} {'admitted':>9} {'masked':>7} {'dropped':>8} {'spikes':>7}  flags")
    print("-" * 95)
    for path in args.inputs:
        arrays = ingest.load_run(path)
        # Pass copt always so the despike (intensity+Copt corroboration) can run.
        tri = noise_filter(
            arrays.I, arrays.t_min, arrays.channels, noise_floor=args.noise_floor,
            noise_surface=surface, copt=arrays.copt, z_thresh=args.z_thresh,
        )
        flag_str = ",".join(tri.flags) if tri.flags else "-"
        n_act, n_msk = len(tri.active_channels), len(tri.masked_channels)
        fI = tri.clean_I if tri.clean_I is not None else arrays.I
        ft = tri.clean_t if tri.clean_t is not None else arrays.t_min
        print(f"{path.name:<48} {n_act:>9} {n_msk:>7} {tri.n_lead_dropped:>8} "
              f"{len(tri.spike_frames):>7}  {flag_str}")
        rows.append({
            "file": path.name, "n_active": n_act, "n_masked": n_msk,
            "lead_dropped": tri.n_lead_dropped, "n_spikes": len(tri.spike_frames),
            "active_channels": " ".join(str(c) for c in tri.active_channels),
            "gap_rezeroed": tri.gap_rezeroed, "flags": flag_str,
        })
        if not args.no_figure:
            ov = path.with_name(f"{path.stem}_overlay.png")
            plot_channel_overlay(tri, fI, ft, arrays.channels, ov)
            print(f"    → {ov.name}")
            if surface is not None:
                grid = path.with_name(f"{path.stem}_channels.png")
                plot_channel_noise_grid(tri, fI, ft, arrays.channels, surface, grid)
                print(f"    → {grid.name}")

    out = args.output or Path("noise_filter_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


# ── Noise surface (per-channel σ on CFZ pH 7) ────────────────────────────────

def noise_surface_main(argv=None):
    """Entry point for ``dfm-noise-surface`` — build the per-channel noise surface.

    Takes the CFZ-pH-7 titration RTF(s) (the drug in its antisolvent, non-dissolving),
    fits the per-channel σ(signal, Copt) surface, and saves the artifact. Fouled
    runs (elevated reference background) are auto-excluded.
    """
    from diffractomorph_pipeline.noise_surface import build_noise_surface

    parser = argparse.ArgumentParser(
        description="Build the per-channel CFZ-pH-7 noise surface (super-floor calibration).",
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="CFZ-pH-7 titration RTF(s) or folder(s).")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Explicit writable output path for the surface JSON.")
    parser.add_argument("--include-fouled", action="store_true")
    args = parser.parse_args(argv)

    files = _collect_rtfs(args.inputs)
    runs = [(Path(f).stem, ingest.extract_run(f, run_kind="measurement")) for f in files]
    surface = build_noise_surface(runs, exclude_fouled=not args.include_fouled)
    out = args.output
    surface.save(out)
    m = surface.meta
    print(f"surface: σ = {surface.k:.4g}·S^{surface.p:+.2f}   ρ={surface.rho:+.2f} "
          f"(SE inflation ×{surface.infl:.2f})")
    print(f"  reps={m['n_reps']}  ref_sums={m['ref_sums']}  excluded_fouled={m['excluded_fouled']}")
    print(f"  pooled points={m['n_points']}  relative-noise slope={m['rel_slope']:+.2f}")
    print(f"Wrote {out}")


# ── Shared optics: build the Mie kernel ──────────────────────────────────────

def build_kernel_main(argv=None):
    """Entry point for ``dfm-build-kernel`` — fit + assemble a Mie kernel from QC files.

    Run for the **weekly NIST** geometry check (re-fit geometry; reuse stored RI)
    or when introducing a **new drug/lot** (``--fit-ri``). Reads the
    ``data_<sample>_<type>`` QC files in ``--qc-dir``: each standard's channel
    intensities (``.rtf``) + its number (q0) distribution (``.csv`` preferred,
    ``.pdf`` fallback). See the README "Required inputs" section.
    """
    from diffractomorph_pipeline.optics import mie_build as mb, standards

    parser = argparse.ArgumentParser(
        description="Build a Mie forward-kernel from QC files (NIST + drug standards).",
    )
    parser.add_argument("--qc-dir", type=Path, default=None,
                        help="Folder of explicit data_<sample>_<type> QC files.")
    parser.add_argument("--drug", required=True,
                        help="Explicit material identifier; no material is selected by default.")
    parser.add_argument("--lens", required=True, choices=sorted(mb.LENS_THETA_MIN_DEG),
                        help="Instrument lens identifier; no detector geometry is selected by default.")
    parser.add_argument("--cal-date", required=True,
                        help="NIST QC date (YYYY-MM-DD) this geometry comes from.")
    parser.add_argument("--fit-ri", action="store_true",
                        help="Fit refractive index from the drug QC (default: stored RI for known drugs).")
    parser.add_argument("--registry", type=Path, required=True,
                        help="Explicit writable registry.yaml path outside the installed package.")
    # Explicit file paths (for QC folders that don't follow the data_ convention,
    # e.g. the real optics_QC layout). A PSD path may be a CSV/PDF or a folder.
    parser.add_argument("--nist-intensity", type=Path, help="NIST .rtf (channel intensities).")
    parser.add_argument("--nist-psd", type=Path, help="NIST q0 distribution: CSV/PDF or folder.")
    parser.add_argument("--drug-intensity", type=Path, help="Drug .rtf (for --fit-ri).")
    parser.add_argument("--drug-psd", type=Path, help="Drug q0 distribution: CSV/PDF or folder.")
    args = parser.parse_args(argv)

    explicit_nist = args.nist_intensity is not None or args.nist_psd is not None
    if explicit_nist and not (args.nist_intensity and args.nist_psd):
        parser.error("--nist-intensity and --nist-psd must be supplied together")
    if not explicit_nist and args.qc_dir is None:
        parser.error("supply --qc-dir, or both --nist-intensity and --nist-psd")

    if explicit_nist:
        print(f"NIST  : intensity={args.nist_intensity.name}  psd={args.nist_psd.name}")
        if args.drug_intensity:
            print(f"{args.drug:<6}: intensity={args.drug_intensity.name}  psd={getattr(args.drug_psd,'name',None)}")
        kernel = mb.build_kernel_from_files(
            args.nist_intensity, args.nist_psd, args.drug, lens=args.lens,
            cal_date=args.cal_date, drug_intensity=args.drug_intensity,
            drug_psd=args.drug_psd, fit_ri=args.fit_ri, registry=args.registry)
    else:
        need_drug = args.fit_ri or args.drug not in mb.RI_LIBRARY
        qc = standards.discover_qc_files(args.qc_dir, args.drug, need_drug=need_drug)
        print(f"NIST  : intensity={qc.nist.intensity.name}  psd={qc.nist.psd.name} ({qc.nist.psd_type})")
        if qc.drug:
            print(f"{args.drug:<6}: intensity={qc.drug.intensity.name}  psd={qc.drug.psd.name} ({qc.drug.psd_type})")
        else:
            print(f"{args.drug:<6}: refractive index from library (n={mb.RI_LIBRARY[args.drug]})")
        kernel = mb.build_kernel_from_qc(args.qc_dir, args.drug, lens=args.lens,
                                         cal_date=args.cal_date, fit_ri=args.fit_ri,
                                         registry=args.registry)
    m = kernel.meta
    print(f"geometry: theta=[{m['theta_min']:.2f}, {m['theta_max']:.2f}]deg  r_bead={m['r_bead']:.3f}")
    ri_note = f"r_cfz={m['r_cfz']:.3f}" if m.get("r_cfz") is not None else "library"
    print(f"n_{args.drug}={m['n_drug']:.3f} ({ri_note})")
    print(f"built + registered kernel: {m['kernel_id']}")


# ── Shared optics: daily size-consistency QC ─────────────────────────────────

def qc_main(argv=None):
    """Entry point for ``dfm-qc`` — daily size-consistency (shape-drift) check.

    Compares today's drug-QC channel pattern against a reference suspension via
    ``shape_consistency`` (Pearson r), PASS/FAILs at the threshold, and appends
    ``(date, file, r, verdict)`` to a trend CSV so drift is visible over time.
    The refractive index is a material constant and is NOT re-fit here — this is
    the kernel's *use*-side QC, not a rebuild.
    """
    from diffractomorph_pipeline.optics import mie, standards

    parser = argparse.ArgumentParser(
        description="Daily size-consistency QC: shape drift vs a reference suspension.",
    )
    parser.add_argument("qc", type=Path, help="Today's drug-QC .rtf (channel intensities).")
    parser.add_argument("--reference", type=Path, required=True,
                        help="Explicit reference-suspension .rtf used for shape comparison.")
    parser.add_argument("--log", type=Path, default=Path("qc_trend.csv"),
                        help="Trend CSV to append to (default: ./qc_trend.csv).")
    parser.add_argument("--pass-r", type=float, default=mie.CONSISTENCY_PASS_R,
                        help=f"PASS threshold on r (default {mie.CONSISTENCY_PASS_R}).")
    parser.add_argument("--date", default=None, help="Override the QC date (YYYY-MM-DD).")
    args = parser.parse_args(argv)

    run = ingest.extract_run(args.qc)
    r = mie.shape_consistency(run.I.mean(axis=0), standards.read_qc_intensity(args.reference))
    verdict = "PASS" if r >= args.pass_r else "FLAG"
    date = args.date or run.t0.strftime("%Y-%m-%d")

    row = {"date": date, "file": args.qc.name, "r": round(r, 4),
           "verdict": verdict, "pass_r": args.pass_r, "reference": args.reference.name}
    if args.log.exists():
        df = pd.read_csv(args.log)
        df = pd.concat([df[df["file"] != row["file"]], pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(args.log, index=False)

    print(f"{date}  {args.qc.name}")
    print(f"  shape-consistency r = {r:.4f}  (threshold {args.pass_r})  ->  {verdict}")
    if verdict == "FLAG":
        print("  ⚠  size/prep drift vs reference — investigate. If it's a new lot/"
              "polymorph, rebuild the kernel: dfm-build-kernel --fit-ri")
    print(f"\n  trend ({args.log}):")
    for _, t in df.tail(8).iterrows():
        ticks = int(round((min(max(t["r"], 0.90), 1.0) - 0.90) / 0.10 * 20))
        print(f"    {t['date']}  r={t['r']:.4f}  {t['verdict']:<4} {'▇' * ticks}")


# ── Noise-floor characterization (glass-bead W1 vs Copt) ─────────────────────

def noise_floor_main(argv=None):
    """Entry point for ``dfm-noise-floor`` — build the W₁ noise floor vs Copt.

    Takes glass-bead Copt-titration RTF(s), auto-detects Copt plateaus, computes
    the frame-to-frame W₁ 95th percentile + bootstrap CI per plateau, and writes a
    noise-floor-vs-Copt curve (CSV + meta). Runs with fouled optics (elevated
    reference background) are auto-excluded.
    """
    from diffractomorph_pipeline import noise

    parser = argparse.ArgumentParser(
        description="Characterize the W₁ noise floor vs Copt from glass-bead titrations.",
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="Glass-bead titration RTF(s) or folder(s).")
    parser.add_argument("-o", "--output", type=Path, default=Path("noise_floor_curve.csv"))
    parser.add_argument("--channel-window", default="15-25",
                        help="PAQXOS channel window for W₁ (default 15-25).")
    parser.add_argument("--include-fouled", action="store_true",
                        help="Don't auto-exclude runs with elevated reference background.")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    lo, hi = (int(x) for x in args.channel_window.split("-"))
    ch_start, ch_end = lo - 1, hi          # PAQXOS 1-indexed → 0-indexed slice

    files = _collect_rtfs(args.inputs)
    runs = [(Path(f).stem, ingest.extract_run(f)) for f in files]
    print(f"{'run':<46} {'ref_bg':>8} {'frames':>6}")
    for label, run in runs:
        print(f"{label:<46} {noise.ref_level(run):>8.1f} {run.flags['n_frames']:>6}")

    fouled = [] if args.include_fouled else noise.flag_fouled(runs)
    if fouled:
        print(f"\n⚠  excluding fouled (elevated reference background): {fouled}")
    clean = [(l, r) for l, r in runs if l not in fouled]
    if not clean:
        print("No clean runs to characterize.", file=sys.stderr); sys.exit(1)

    curve = noise.characterize_noise_floor(clean, ch_start=ch_start, ch_end=ch_end, seed=args.seed)
    curve.meta.update({"channel_window_paqxos": args.channel_window, "excluded_fouled": fouled})
    curve.save(args.output)
    print(f"\nNoise floor (W₁ p95, channels {args.channel_window}):")
    print(f"  {'Copt%':>6} {'W1_p95':>9} {'95% CI':>22}")
    for p in sorted(curve.points, key=lambda p: p.copt):
        print(f"  {p.copt:>6.1f} {p.w1_p95:>9.4f}   [{p.ci_lo:.4f}, {p.ci_hi:.4f}]")
    print(f"\nWrote {args.output} (+ meta)")

    if not args.no_figure:
        _plot_noise_floor(curve, args.output.with_suffix(".png"))
        print(f"Wrote {args.output.with_suffix('.png')}")


def _plot_noise_floor(curve, out_path):
    import matplotlib.pyplot as plt
    from diffractomorph_pipeline.plot_styles import apply_lab_style, get_color, setup_axes
    apply_lab_style()
    import pandas as pd
    df = curve.to_frame()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (src, g) in enumerate(df.groupby("source")):
        g = g.sort_values("copt")
        ax.plot(g["copt"], g["w1_p95"], "o-", color=get_color(i), label=src, ms=6)
        ax.fill_between(g["copt"], g["ci_lo"], g["ci_hi"], alpha=0.25, color=get_color(i))
    ax.set_yscale("log")
    ax.set_xlabel("optical concentration Copt (%)")
    ax.set_ylabel(f"W1 95th pct (noise floor)\nchannels {curve.meta.get('channel_window_paqxos','15-25')}")
    ax.set_title("Noise floor vs Copt (95% bootstrap CI)")
    setup_axes(ax)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


# ── Study batch runner ───────────────────────────────────────────────────────

def run_main(argv=None):
    """Entry point for ``dfm-run`` — run the built pipeline over a study tree.

    Runs ingest + triage on each measurement subfolder, routing outputs into
    ``<subfolder>/ingest/`` and ``<subfolder>/triage/``, plus a study-level
    ``<root>/summary/triage_verdicts_all.csv``. QC folders are skipped.
    """
    from diffractomorph_pipeline import batch

    parser = argparse.ArgumentParser(
        description="Run the pipeline-thus-far (ingest + noise_filter) over a study tree.",
    )
    parser.add_argument("path", type=Path,
                        help="Study root, or a single measurement subfolder.")
    parser.add_argument("--steps", default="ingest,noise_filter",
                        help="Comma-separated steps (default: ingest,noise_filter; add "
                             "'extract' to fit KWW dissolution kinetics — that also "
                             "runs the parked single/multi-band classifier).")
    parser.add_argument("--list", action="store_true",
                        help="List the measurement folders that would run, then exit.")
    parser.add_argument("--noise-surface", nargs="?", const="default", default=None,
                        help="Use the CFZ-pH-7 noise surface for super-floor triage "
                             "(bare flag = packaged default; or pass a path).")
    parser.add_argument("--kernel", type=Path, default=None,
                        help="Explicit optical-kernel artifact for optional size-space plots.")
    args = parser.parse_args(argv)
    steps = tuple(s.strip() for s in args.steps.split(",") if s.strip())

    if args.list:
        for d in batch.find_measurement_dirs(args.path):
            print(d)
        return

    surface = None
    if args.noise_surface is not None:
        from diffractomorph_pipeline.noise_surface import load_surface
        surface = load_surface(None if args.noise_surface == "default" else Path(args.noise_surface))
        print(f"(super-floor filter via noise surface: k={surface.k:.3g} p={surface.p:+.2f})")

    # A folder holding .rtf runs directly = a single subfolder; otherwise walk.
    direct = sorted(args.path.glob("*.rtf"))
    if direct:
        rows, krows = batch.run_subfolder(
            args.path, steps, noise_surface=surface, kernel=args.kernel,
        )
        done = len(rows) if rows else len(krows)
        print(f"{args.path.name}: {len(direct)} runs → {done} processed")
    else:
        summary = batch.run_study(
            args.path, steps, noise_surface=surface, kernel=args.kernel,
        )
        if not summary.empty and "n_active" in summary.columns:
            mean_adm = summary["n_active"].mean()
            print(f"\nTotal: {len(summary)} measurements  "
                  f"(mean {mean_adm:.0f} channels admitted/run)")


# ── Module 3: Extraction ─────────────────────────────────────────────────────

def extract_main(argv=None):
    """Entry point for ``dfm-extract`` — fit KWW dissolution kinetics per run.

    Ingests each RTF, triages it (Step 1), and fits the stretched-exponential
    decay of the triage-routed dissolution signal (aggregate for single-mode, the
    dissolution band for multi-band). Prints a table and writes a parameters CSV.
    """
    parser = argparse.ArgumentParser(
        description="Step 3: fit KWW dissolution kinetics on triage-routed runs.",
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="Raw PAQXOS export file(s).")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Parameters CSV path. Default: kinetics_params.csv in CWD.")
    parser.add_argument("--noise-surface", nargs="?", const="default", default=None,
                        help="Use the CFZ-pH-7 noise surface for super-floor triage "
                             "(bare flag = packaged default, or pass a path).")
    parser.add_argument("--window", type=float, nargs=2, metavar=("TMIN", "TMAX"),
                        default=None, help="Restrict the fit to [TMIN, TMAX] minutes.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the triage noise-null surrogates (default 0).")
    parser.add_argument("--no-figure", action="store_true", help="Skip diagnostic figures.")
    args = parser.parse_args(argv)

    from diffractomorph_pipeline.band_routing import triage_channels
    surface = None
    if args.noise_surface is not None:
        from diffractomorph_pipeline.noise_surface import load_surface
        surface = load_surface(None if args.noise_surface == "default" else Path(args.noise_surface))
        print(f"(super-floor triage via noise surface: k={surface.k:.3g} p={surface.p:+.2f})")

    rows = []
    print(f"{'file':<42} {'target':<16} {'model':<11} {'k':>7} {'beta':>6} {'R2':>7}  flags")
    print("-" * 100)
    for path in args.inputs:
        run = ingest.load_run(path)
        tri = triage_channels(run.I, run.t_min, run.channels, random_state=args.seed,
                              noise_surface=surface,
                              copt=(run.copt if surface is not None else None))
        win = tuple(args.window) if args.window else None
        fit = extract.fit_dissolution_kinetics(run, tri, window=win)
        flag_str = ",".join(fit.flags) if fit.flags else "-"
        print(f"{path.name:<42} {fit.target:<16} {fit.model:<11} "
              f"{fit.k:>7.3f} {fit.beta:>6.3f} {fit.r2:>7.4f}  {flag_str}")
        rows.append({"file": path.name, **fit.to_row()})
        if not args.no_figure:
            from diffractomorph_pipeline.figures.diagnostic import plot_kinetics_diagnostic
            plot_kinetics_diagnostic(fit, run, tri, path.with_name(f"{path.stem}_kinetics.png"),
                                     window=win)

    out = args.output or Path("kinetics_params.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nParameters → {out}")


# ── Module 4: Figures ────────────────────────────────────────────────────────

def figure_main():
    """Entry point for ``dfm-figure`` — parameters CSV → publication figures."""
    parser = argparse.ArgumentParser(
        description="Generate publication figures from extracted parameters.",
    )
    parser.add_argument("params_csv", type=Path, help="Parameters CSV from dfm-extract.")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    parser.error(
        "dfm-figure is experimental and is not installed as a public command. "
        "Use the explicit analysis APIs; a future reproduction bundle will provide "
        "named manuscript figure recipes."
    )
