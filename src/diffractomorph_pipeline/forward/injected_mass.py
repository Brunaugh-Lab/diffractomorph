"""Injected CFZ mass per run — the dose record the forward model consumes.

Each run's delivered dose is the day's suspension concentration (mg/mL, from the QC via
``assay.suspension``) times the injected volume (µL, from a per-experiment ``run_volumes.json``).
Build the records here, then feed ``mass_mg`` to :func:`predict` / :func:`simulate_dissolution`.

    from diffractomorph_pipeline.forward import injected_mass as im
    runs, _src = im.load_run_volumes(".../QC/run_volumes.json")
    doses = im.build(runs, susp_mgml=lambda r: conc_by_pct[r["tween_pct_wv"]])
    df = im.to_frame(doses)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from diffractomorph_pipeline.assay import suspension


@dataclass
class InjectedMass:
    date: str
    ph: float | None
    rep: int
    volume_uL: float
    susp_mgml: float
    mass_mg: float
    condition: str = ""


def load_run_volumes(path):
    """Read a per-experiment ``run_volumes.json`` → (runs list, source-provenance dict)."""
    d = json.loads(Path(path).read_text())
    return d["runs"], d.get("source", {})


def build(run_volumes, susp_mgml):
    """Per-run :class:`InjectedMass` from a runs list and the suspension concentration.

    ``susp_mgml`` is one mg/mL shared by every run, or a callable ``run -> mg/mL`` (for days
    with more than one suspension, keyed off the run's condition).
    """
    conc_of = susp_mgml if callable(susp_mgml) else (lambda r: susp_mgml)
    out = []
    for r in run_volumes:
        c = float(conc_of(r)); vol = float(r["volume_uL"])
        out.append(InjectedMass(
            date=r.get("date", ""), ph=r.get("ph"), rep=int(r["rep"]), volume_uL=vol,
            susp_mgml=c, mass_mg=float(suspension.injected_mass_mg(c, vol)),
            condition=r.get("tween_pct_wv") or r.get("condition", "")))
    return out


def to_frame(masses) -> pd.DataFrame:
    """InjectedMass records → a tidy DataFrame (one row per run)."""
    return pd.DataFrame([m.__dict__ for m in masses])
