"""Shared plumbing for the pH-study analysis drivers — run discovery + q3 frame-masking + QC helpers.

Holds the pieces the ΣI / q3 drivers reuse: pH-study run discovery (``iter_runs`` / ``q3_folder``), the
two-part q3 frame mask (``masked_inputs``), the low-signal QC helpers (``copt_start``,
``LOW_COPT_START``), and ``despiked_copt``. The generic primitives ``despike_upward``
(:mod:`diffractomorph_pipeline.kinetics`) and ``screen_uv_dissolved``
(:mod:`diffractomorph_pipeline.uv_qc`) live in the installable package and are re-exported here.

Run with the pipeline venv.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from diffractomorph_pipeline import psd
from diffractomorph_pipeline.kinetics import despike_upward   # re-exported; canonical home is the package
from diffractomorph_pipeline.noise_filter import despike_frames
from diffractomorph_pipeline.uv_qc import screen_uv_dissolved   # re-exported; canonical home is the package
from diffractomorph_pipeline.optics.mie import VALID_SIZE_MAX_UM

from study_common import BASE, SUMMARY, find_rtf

Q3ROOT = BASE / "CFZ q3 csv"
COPT_FLOOR_FRAC = 0.3          # legacy shape scripts only; not the manuscript matched-q3 rule
LOW_COPT_START = 5.0          # a run starting below this optical concentration has too little dynamic
                              # range to resolve dissolution — route it to a low_copt/ subfolder


def copt_start(copt):
    """The run's starting optical concentration = first finite non-zero value of the Copt trajectory."""
    c = np.asarray(copt, dtype=float)
    fin = c[np.isfinite(c) & (c != 0)]
    return float(fin[0]) if fin.size else float("nan")


def q3_folder(ph, date, rep):
    """Locate a run's per-frame q3 CSV folder (``…/CFZ q3 csv/pH=4.0/Day N - <date>/Rep R``)."""
    hits = glob.glob(str(Q3ROOT / f"pH={ph}" / f"*{date}*" / f"Rep {rep}"))
    return hits[0] if hits else None


def iter_runs():
    """Yield ``(ph, date, rep, rtf, q3_folder)`` for every pH-study run with both a q3 export and a
    measurement ``.rtf`` (ordered by ``run_metadata.csv``)."""
    meta = pd.read_csv(SUMMARY / "run_metadata.csv")
    for r in meta.itertuples():
        ph, date, rep = float(r.ph), int(r.date_i), int(r.rep)
        fo, rtf = q3_folder(ph, date, rep), find_rtf(ph, date, rep)
        if fo and rtf:
            yield ph, date, rep, rtf, fo


def despiked_copt(run, **kw):
    """``Copt(t)`` with upward optical glitches removed — see :func:`despike_upward`. (This is the
    envelope's own cleaner; the q3 frame mask separately reuses :func:`despike_frames` for the
    synchronized-glitch frames that corrupt the *inversion*, a different job.)"""
    return despike_upward(run.t_min, run.copt, **kw)


def masked_inputs(traj, run, total, *, copt_floor_frac=COPT_FLOOR_FRAC, despike=True):
    """Two-part frame mask, aligned across the q3 / ``.rtf`` frame-count mismatch.

    The q3 export and its ``.rtf`` are the same acquisition but differ by a frame or two, so first trim
    everything to the common length ``n`` (they start aligned at frame 0). Then:
      (A) reuse :func:`despike_frames` on the paired channel data to flag startup / synchronized-glitch
          frames, and (B) apply :func:`psd.frame_mask`'s Copt low-signal floor. The floor always runs on
          ``run.copt`` (the scattering signal), even for the UV basis.

    Returns ``(masked_traj, masked_total, masked_t_min, info)``.
    """
    n = min(traj.n_frames, len(run.copt), len(total))
    copt = np.asarray(run.copt[:n], dtype=float)
    drop = []
    if despike and n >= 5:
        *_, dinfo = despike_frames(np.asarray(run.I[:n], float), np.asarray(run.t_min[:n], float), copt)
        drop = dinfo.get("spike_frames", [])
    keep = psd.frame_mask(copt, copt_floor_frac=copt_floor_frac, drop_frames=drop)
    mtraj = psd.apply_frame_mask(psd.Q3Trajectory(traj.grid_um, traj.dQ3[:n], traj.layout, traj.source), keep)
    mtotal = np.asarray(total, dtype=float)[:n][keep]
    mt = np.asarray(run.t_min, dtype=float)[:n][keep]
    info = dict(n=n, n_keep=int(keep.sum()), n_dropped=int(n - keep.sum()), n_despike=len(drop))
    return mtraj, mtotal, mt, info
