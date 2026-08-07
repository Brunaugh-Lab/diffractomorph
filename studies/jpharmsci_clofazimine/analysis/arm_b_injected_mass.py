"""Build the run-specific dose record for the Arm B in-medium Tween 80 (micelle) study.

Delivered dose = injected volume x the suspension concentration measured by the QC that covers
that run. Both vary per replicate, so dose is resolved per run, never per condition:

  * volume comes from the shared ``CFZ volume added.xlsx`` log, whose condition labels read
    ``4.5 0.02% w/v Tween (<level>[ - Day N])``. Most runs are 40 uL; two are 45 uL.
  * concentration comes from the QC routed to that replicate by :mod:`arm_b_common` — which
    matters most in the day-1 folder, where replicate 1 and replicates 2-3 carry different QC.

**The prep is the injection date, not the folder name.** The ``20260702`` folder holds replicate 1
injected on 2026-07-02 and replicates 2-3 injected on 2026-07-07 off a fresh suspension. Both
``prep`` (the replicate identity) and ``folder`` (where the files live) are carried, with
``prep_differs_from_folder`` marking the six runs where they diverge.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from diffractomorph_pipeline.assay import calibration as cal, read_qc, suspension

from arm_b_common import VOLUME_WORKBOOK, default_study_root, discover_runs
from arm_b_provenance import provenance_record, write_provenance


def build_dose_table(study_root: Path | str | None = None) -> pd.DataFrame:
    """Join each discovered run with its injected volume and its own QC → delivered dose."""
    root = Path(study_root) if study_root is not None else default_study_root()
    merged = discover_runs(root)          # already carries injected_uL + injection_date
    conc_cache: dict[Path, tuple[float, float, float]] = {}
    rows = []
    for run in merged.itertuples():
        qc = Path(run.qc_xlsx)
        if qc not in conc_cache:
            reads = read_qc(qc)
            if len(reads) != 1:
                raise ValueError(f"{qc.name}: expected 1 suspension read, found {len(reads)}")
            abs_bs = float(reads[0].abs_bs)
            conc_cache[qc] = (abs_bs,
                              float(suspension.suspension_conc_mgml(abs_bs, filter_corrected=True)),
                              float(suspension.suspension_conc_mgml(abs_bs)))
        abs_bs, corr, uncorr = conc_cache[qc]
        rows.append({
            "run_id": run.run_id, "condition": run.condition, "xcmc": run.xcmc,
            "prep": run.prep, "prep_index": run.prep_index, "rep": run.rep,
            "folder": run.folder, "injection_date": run.injection_date,
            "prep_differs_from_folder": run.prep != run.folder,
            "qc_abs_blank_subtracted": round(abs_bs, 6),
            "susp_mgml_filtercorr": round(corr, 6),
            "susp_mgml_uncorr": round(uncorr, 6),
            "injected_uL": run.injected_uL,
            "injected_mass_mg": round(float(suspension.injected_mass_mg(uncorr, run.injected_uL)), 6),
            "injected_mass_mg_uncorr":
                round(float(suspension.injected_mass_mg(uncorr, run.injected_uL)), 6),
            "qc_source": qc.name,
            "volume_source": VOLUME_WORKBOOK,
        })
    return pd.DataFrame(rows).sort_values(["xcmc", "prep", "rep"]).reset_index(drop=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Arm B delivered-dose table.")
    p.add_argument("--study-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    root = args.study_root or default_study_root()
    out_dir = args.output_dir or root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = build_dose_table(root)
    out = out_dir / "arm_b_injected_mass.csv"
    table.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
    write_provenance(out_dir / "provenance_injected_mass.json",
                     provenance_record("arm_b_injected_mass", study_root=root,
                                       volume_workbook=VOLUME_WORKBOOK))

    print(table[["condition", "prep", "folder", "rep", "susp_mgml_uncorr",
                 "injected_uL", "injected_mass_mg"]].to_string(index=False))
    split = table[table["prep_differs_from_folder"]]
    if not split.empty:
        print(f"\n{len(split)} run(s) injected on a date other than their folder name:")
        print(split[["condition", "folder", "rep", "injection_date", "qc_source"]].to_string(index=False))
    print(f"\ndose spread: {table['injected_mass_mg'].min():.4f}–{table['injected_mass_mg'].max():.4f} mg "
          f"(mean {table['injected_mass_mg'].mean():.4f})")
    off, df = cal.SUSPENSION["filter_offset_ugml"], cal.SUSPENSION["dilution_factor"]
    print(f"legacy filter-correction sensitivity (not applied to primary dose): +{off} µg/mL additive (×{df:.0f} DF = "
          f"+{off * df / 1e3:.3f} mg/mL on the suspension)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
