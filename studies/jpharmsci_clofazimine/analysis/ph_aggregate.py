"""Current manuscript-authoritative aggregate KWW tables for the pH study.

The historical corpus contains pre-start-boundary CSVs. This module recomputes the particle-side
endpoint from the raw PAQXOS exports with the explicitly declared circulation-start policy used by
the current manuscript. It writes only to a caller-selected output directory.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from diffractomorph_pipeline import ingest
from diffractomorph_pipeline.processing import AggregateKWWConfig, fit_aggregate_kww


CONFIG = AggregateKWWConfig(
    reference_mode="raw_measured",
    start_policy="concordant_early_maximum",
    start_acquisition_variable="copt",
    start_search_frames=3,
    start_maximum_time_min=1.0,
    start_minimum_relative_increase=0.20,
    start_minimum_spectral_cosine=0.995,
)


def _measurement_rtf(study_root: Path, ph: float, date: int, rep: int) -> Path:
    hits = sorted(glob.glob(str(
        study_root / f"ph_{ph}" / f"{date}_pH*" / f"*measurement*Rep {rep}.rtf"
    )))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected one pH={ph}, date={date}, rep={rep} measurement RTF; found {len(hits)}"
        )
    return Path(hits[0])


def build_tables(study_root: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(study_root)
    metadata = pd.read_csv(root / "summary" / "run_metadata.csv")
    rows = []
    for record in metadata.sort_values(["ph", "date_i", "rep"]).itertuples():
        ph, date, rep = float(record.ph), int(record.date_i), int(record.rep)
        run = ingest.extract_run(_measurement_rtf(root, ph, date, rep))
        result = fit_aggregate_kww(run, CONFIG)
        row = result.to_row()
        rows.append({
            "id": f"pH{ph}_{date}_R{rep}",
            "condition": f"pH {ph:.1f}",
            "date": date,
            "rep": rep,
            **row,
        })
    by_run = pd.DataFrame(rows)
    metrics = ["tau_min", "beta", "mean_relax_min", "t50_min", "floor",
               "optical_decay_depth_pct", "i0_fit", "r2"]
    run_counts = (by_run.groupby(["condition", "date"], as_index=False)
                  .size().rename(columns={"size": "n_runs"}))
    by_date = (run_counts.merge(
        by_run.groupby(["condition", "date"], as_index=False)[metrics].mean(),
        on=["condition", "date"], validate="one_to_one"))
    date_counts = (by_date.groupby("condition", as_index=False)
                   .size().rename(columns={"size": "n_dates"}))
    by_condition = date_counts.merge(
        by_date.groupby("condition", as_index=False)[metrics].mean(),
        on="condition", validate="one_to_one")
    return by_run, by_date, by_condition


def write_tables(study_root: Path | str, output_dir: Path | str) -> tuple[Path, Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = build_tables(study_root)
    names = ("aggregate_kww_by_run.csv", "aggregate_kww_by_independent_unit.csv",
             "aggregate_kww_by_condition.csv")
    paths = tuple(output / name for name in names)
    for frame, path in zip(frames, paths):
        frame.to_csv(path, index=False)
    return paths
