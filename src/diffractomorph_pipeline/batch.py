"""Study batch runner — run the built pipeline steps over a study tree.

Walks a study folder, finds each measurement subfolder (the leaf folders that
hold the ``.rtf`` runs), and runs the pipeline-thus-far on each, **routing the
outputs into per-type subfolders inside that measurement folder**:

    <date>_pH<x>/
        ...the raw .rtf runs...
        ingest/        <run>.csv + <run>_meta.json   (every run, measurement + blank)
        noise_filter/  <run>_channels.png + noise_filter_summary.csv   (measurements only)
        explore/       <run>_overlay.png   (admitted channels superimposed)

A study-level roll-up of channel admission (date / pH / day / rep) is written under
``<root>/summary/`` — handy for the 3×3 (3 reps × 3 prep-days) overview.

QC folders (``optics_QC/``) are skipped — they're calibration, not runs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from diffractomorph_pipeline import extract, ingest
from diffractomorph_pipeline.figures.diagnostic import (
    plot_channel_noise_grid,
    plot_channel_overlay,
    plot_day_overlay,
    plot_dissolution_vs_size,
    plot_kinetics_diagnostic,
)
from diffractomorph_pipeline.band_routing import triage_channels
from diffractomorph_pipeline.utils import guess_from_filename

DEFAULT_STEPS = ("ingest", "noise_filter")
# Calibration / QC folders (not dissolution runs) and our own output folders.
_SKIP_DIRS = {"QC", "optics_QC", "noise_floor", "ingest", "noise_filter",
              "explore", "triage", "extract", "summary"}


def _is_measurement(name: str) -> bool:
    return "blank" not in name.lower()


def find_measurement_dirs(root: Path | str) -> list[Path]:
    """Leaf folders containing ``.rtf`` runs (excluding QC / output folders)."""
    root = Path(root)
    dirs = set()
    for rtf in root.rglob("*.rtf"):
        if _SKIP_DIRS & set(rtf.parts):
            continue
        dirs.add(rtf.parent)
    return sorted(dirs)


def run_subfolder(subdir: Path | str, steps=DEFAULT_STEPS, noise_surface=None, kernel=None):
    """Run the pipeline steps on one measurement folder.

    Returns ``(filter_rows, kinetic_rows)``. The ``noise_filter`` step admits the
    channels carrying real directional change (super-floor against ``noise_surface``,
    or the static mask) and writes the per-channel grid + the admitted-channel
    overlay. The single/multi-band classifier is parked — it's only run when
    ``extract`` is requested (the kinetics fit needs the verdict).
    """
    subdir = Path(subdir)
    rtfs = sorted(subdir.glob("*.rtf"))
    ingest_dir = subdir / "ingest"
    nf_dir, explore_dir = subdir / "noise_filter", subdir / "explore"
    rows: list[dict] = []
    krows: list[dict] = []
    day_reps: list[tuple] = []          # (clean_I, clean_t) per measurement, for the day mean
    channels = None
    want_filter = "noise_filter" in steps
    want_extract = "extract" in steps

    for rtf in rtfs:
        run = ingest.extract_run(rtf)
        info = guess_from_filename(rtf.name)
        ident = {"date": run.t0.strftime("%Y-%m-%d"), "ph": info["ph"],
                 "day": info["day"], "rep": info["rep"], "file": rtf.name}

        if "ingest" in steps:
            ingest_dir.mkdir(exist_ok=True)
            run.write_csv(ingest_dir / f"{rtf.stem}.csv")

        if (want_filter or want_extract) and _is_measurement(rtf.name):
            # Only route single/multi-band modes when the extractor needs it. Pass
            # copt always so the despike (intensity+Copt corroboration) can run.
            tri = triage_channels(run.I, run.t_min, run.channels, random_state=0,
                                  noise_surface=noise_surface, copt=run.copt,
                                  route_modes=want_extract)
            # Figures show the despiked data the admission ran on.
            fI = tri.clean_I if tri.clean_I is not None else run.I
            ft = tri.clean_t if tri.clean_t is not None else run.t_min

            if want_filter:
                nf_dir.mkdir(exist_ok=True)
                explore_dir.mkdir(exist_ok=True)
                if noise_surface is not None:
                    plot_channel_noise_grid(
                        tri, fI, ft, run.channels, noise_surface,
                        nf_dir / f"{rtf.stem}_channels.png")
                plot_channel_overlay(tri, fI, ft, run.channels,
                                     explore_dir / f"{rtf.stem}_overlay.png")
                day_reps.append((fI, ft))
                channels = run.channels
                rows.append({
                    **ident,
                    "n_active": len(tri.active_channels),
                    "n_masked": len(tri.masked_channels),
                    "lead_dropped": tri.n_lead_dropped,
                    "n_spikes": len(tri.spike_frames),
                    "active_channels": " ".join(str(c) for c in tri.active_channels),
                    "flags": ",".join(tri.flags) or "-",
                })

            if want_extract:
                extract_dir = subdir / "extract"
                extract_dir.mkdir(exist_ok=True)
                fit = extract.fit_dissolution_kinetics(run, tri)
                plot_kinetics_diagnostic(fit, run, tri,
                                         extract_dir / f"{rtf.stem}_kinetics.png")
                krows.append({**ident, **fit.to_row()})

    if want_filter and rows:
        pd.DataFrame(rows).to_csv(nf_dir / "noise_filter_summary.csv", index=False)
    # Day-level replicate mean (needs ≥2 measurement reps in the folder).
    if want_filter and len(day_reps) >= 2:
        plot_day_overlay(day_reps, channels, explore_dir / "day_mean_overlay.png",
                         label=subdir.name)
        # A size-space plot is only scientifically defined when the caller supplies
        # the instrument/material optical operator explicitly.
        if kernel is not None:
            from diffractomorph_pipeline.optics import mie
            selected_kernel = kernel if isinstance(kernel, mie.MieKernel) else mie.load_kernel(kernel)
            char_size = mie.channel_size_map(selected_kernel)
            plot_dissolution_vs_size(day_reps, channels, char_size,
                                     explore_dir / "dissolution_vs_size.png",
                                     label=subdir.name)
    if want_extract and krows:
        pd.DataFrame(krows).to_csv(subdir / "extract" / "kinetics_params.csv", index=False)
    return rows, krows


def run_study(root: Path | str, steps=DEFAULT_STEPS, log=print, noise_surface=None,
              kernel=None) -> pd.DataFrame:
    """Run the pipeline over every measurement subfolder under ``root``.

    Writes per-folder outputs and a study-level ``summary/noise_filter_all.csv``.
    Returns the combined channel-admission table. ``noise_surface`` enables the
    super-floor (per-channel directional) filter.
    """
    root = Path(root)
    dirs = find_measurement_dirs(root)
    log(f"Found {len(dirs)} measurement folder(s) under {root.name}/")
    all_rows: list[dict] = []
    all_krows: list[dict] = []
    for d in dirs:
        rows, krows = run_subfolder(d, steps, noise_surface=noise_surface, kernel=kernel)
        all_rows.extend(rows)
        all_krows.extend(krows)
        rel = d.relative_to(root) if d.is_relative_to(root) else d
        if rows:
            tail = (f"{r['file'].split('Rep')[-1].strip()}:{r['n_active']}ch" for r in rows)
            label = "admitted"
        else:
            tail = (f"{r['file'].split('Rep')[-1].strip()}:{r.get('model')}" for r in krows)
            label = "extract"
        log(f"  {rel}  ({len(list(d.glob('*.rtf')))} runs) → {label}: {', '.join(tail) or '—'}")

    out = root / "summary"
    summary = pd.DataFrame(all_rows)
    if not summary.empty and "noise_filter" in steps:
        out.mkdir(exist_ok=True)
        summary = summary.sort_values(["ph", "date", "rep"], na_position="last")
        summary.to_csv(out / "noise_filter_all.csv", index=False)
        log(f"\nStudy summary → {out / 'noise_filter_all.csv'}")
    if all_krows and "extract" in steps:
        out.mkdir(exist_ok=True)
        kin = pd.DataFrame(all_krows).sort_values(["ph", "date", "rep"], na_position="last")
        kin.to_csv(out / "kinetics_params_all.csv", index=False)
        log(f"Kinetics summary → {out / 'kinetics_params_all.csv'}")
    return summary
