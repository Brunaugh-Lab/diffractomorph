"""Shared, port-free helpers for the pH-dependent dissolution study scripts.

Paths to the study data + small input readers (locate a run's RTF, pull a run's injected
q3 PSD, back out S0 from a free-base Cs). These carry no dependency on the dissolution ODE —
they were previously exported from ``shape_compare``; consolidated here so the analysis layer
does not import the (removed) legacy ODE port.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from diffractomorph_pipeline.config import data_root

MW = 473.4          # CFZ molar mass (g/mol)
PKA = 6.08          # CFZ BH+ apparent solubility-pH pKa'

BASE = data_root() / "disso_experiments" / "ph_dependent_dissolution_study"
SUMMARY = BASE / "summary"
OUT = Path(__file__).parent / "figures"


def s0_from_cs_ugml(cs_ugml: float, ph: float) -> float:
    """Back out intrinsic base solubility S0 (mol/L) from a free-base Cs(pH) (µg/mL)."""
    cs_M = cs_ugml / 1e3 / MW
    return cs_M / (1.0 + 10.0 ** (PKA - ph))


def q3_psd(starting_full: pd.DataFrame, ph, date, rep):
    """The injected q3 PSD for one run: (diam_um grid, volume_frac) on the 31-bin grid.

    The injected PSD is the day's QC suspension — the SAME suspension is pipetted into
    every pH condition that day, so it's really a per-DATE quantity. If the exact
    (ph,date,rep) row is missing (an export gap), fall back to any same-date injected
    row rather than dropping the run.
    """
    inj = starting_full[(starting_full.stage == "injected") & (starting_full.date == date)]
    sub = inj[(inj.ph == ph) & (inj.rep == rep)].sort_values("bin")
    if sub.empty and not inj.empty:                     # same suspension → reuse same-date PSD
        donor = inj.sort_values(["ph", "rep", "bin"])
        first = donor[(donor.ph == donor.ph.iloc[0]) & (donor.rep == donor.rep.iloc[0])]
        print(f"    (PSD for pH{ph} {date} R{rep} missing; reusing same-date QC suspension "
              f"pH{first.ph.iloc[0]} R{int(first.rep.iloc[0])})")
        sub = first.sort_values("bin")
    return sub.d0_um.to_numpy(float), sub.volume_frac.to_numpy(float)


def find_rtf(ph, date, rep) -> str | None:
    """Locate the measurement RTF for (ph,date,rep) in the study tree."""
    hits = glob.glob(str(BASE / f"ph_{ph}" / f"{date}_pH*" / f"*measurement*Rep {rep}.rtf"))
    return hits[0] if hits else None


def arith_diameters(d0_um: np.ndarray) -> np.ndarray:
    """Representative diameter = ARITHMETIC mean of the bin edges, (d_lo + d_hi)/2, vs the
    Sympatec geometric class mean stored in d0_um. Edges are the geometric midpoints of the
    d0 grid."""
    d0 = np.asarray(d0_um, float)
    mids = np.sqrt(d0[:-1] * d0[1:])
    edges = np.concatenate([[d0[0] ** 2 / mids[0]], mids, [d0[-1] ** 2 / mids[-1]]])
    return 0.5 * (edges[:-1] + edges[1:])


def _scatter_soft(centers_log, d_log, q, nbin):
    """Distribute each cohort's volume `q` across the two bracketing fixed-bin centers,
    linearly in log-diameter (a tent/CIC assignment). Smooth analog of hard digitize —
    avoids staircase cliffs when a shrinking cohort crosses a bin boundary."""
    out = np.zeros(nbin)
    for dc, qc in zip(d_log, q):
        if qc <= 0 or dc < centers_log[0]:          # dissolved below the grid -> gone
            continue
        j = int(np.searchsorted(centers_log, dc)) - 1
        j = min(max(j, 0), nbin - 2)
        w = (dc - centers_log[j]) / (centers_log[j + 1] - centers_log[j])
        w = min(max(w, 0.0), 1.0)
        out[j] += qc * (1.0 - w)
        out[j + 1] += qc * w
    return out
