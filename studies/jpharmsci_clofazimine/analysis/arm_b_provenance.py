"""Provenance sidecars for Arm B outputs.

Every Arm B result is conditional on choices that are not visible in the numbers themselves —
which Cs ladder, whether the optical input was cleaned, the UV calibration convention, and the
code version. Each output folder gets a ``provenance.json`` recording them, so a CSV picked up
later can be traced back to the analysis that produced it.

Timestamps come from the caller's clock at write time; nothing here reads the data.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from diffractomorph_pipeline import __version__
from diffractomorph_pipeline.assay import calibration as cal

import arm_b_cs


def _git_commit() -> str:
    try:
        repo = Path(__file__).resolve().parents[1]
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
        commit = out.stdout.strip() or "unknown"
        return f"{commit}-dirty" if dirty.stdout.strip() else commit
    except Exception:
        return "unknown"


def provenance_record(analysis: str, *, cs_ladder: str | None = None,
                      optical_cleaned: bool | None = None, study_root: Path | str | None = None,
                      uv_ph_values: tuple[float, ...] = (4.5,), **extra) -> dict:
    """Everything needed to reproduce one output folder.

    ``uv_ph_values`` lists every pH whose UV calibration the output actually depends on. The
    standard curves and the additive filter offset are pH-specific (the offset is 1.48 ug/mL at
    pH 4.0 and 4.5 but 1.02 at pH 5.0), so an output spanning several pH values must record all
    of them rather than a single condition's constants.
    """
    record = {
        "analysis": analysis,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline_version": __version__,
        "git_commit": _git_commit(),
        "study_root": str(study_root) if study_root is not None else None,
        "uv_calibration": {
            "ph_values": [float(ph) for ph in uv_ph_values],
            "dilution": cal.DILUTION,
            "filter_offset_ugml": {str(ph): cal.FILTER_OFFSET.get(float(ph))
                                   for ph in uv_ph_values},
            "filter_offset_convention": "additive, applied to the well concentration before dilution",
            "blank_convention": "per-plate blank when the export reports one, packaged global "
                                "otherwise",
            "standard_curves": {
                f"{ph}_{wl}nm": {"slope": cal.CURVES[(float(ph), wl)][0],
                                 "intercept": cal.CURVES[(float(ph), wl)][1]}
                for ph in uv_ph_values for wl in (280, 490)
                if (float(ph), wl) in cal.CURVES},
            "note": "curves and filter offsets are pH-specific; packaged aqueous calibration, "
                    "NOT calibrated at in-medium Tween levels",
        },
    }
    if optical_cleaned is not None:
        record["optical"] = {
            "cleaned": bool(optical_cleaned),
            "steps": (["synchronized-frame despike with Copt corroboration",
                       "leading startup-frame drop + time re-zero",
                       "interior glitch interpolation",
                       "acquisition-gap re-zero",
                       "per-channel noise-surface admission (z>4)"]
                      if optical_cleaned else ["none — raw Copt and raw channels"]),
        }
    if cs_ladder is not None:
        record["solubility"] = arm_b_cs.provenance(cs_ladder)
    record.update(extra)
    return record


def write_provenance(path: Path | str, record: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n")
    return path
