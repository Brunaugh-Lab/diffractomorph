"""Arm B optical trajectories, raw or routed through the established pipeline QC.

``arm_b_partition`` originally fit raw Copt and raw detector channels, which carry visible
synchronized frame artifacts. This module supplies the same trajectories either raw or cleaned
by the pipeline's own preprocessing, so every optical result can be reported both ways rather
than silently depending on the choice:

* **synchronized-frame despiking with Copt corroboration** —
  :func:`noise_filter.despike_frames`. A whole-spectrum one-frame jump is dwarfed by each
  dissolving channel's own dynamic range, so it is detected at the frame level and required to
  be corroborated by a same-direction Copt move. Leading flagged frames (the obscuration/laser
  startup transient) are dropped and time re-zeroed; interior ones are interpolated per channel.
* **acquisition-gap re-zeroing** — :func:`noise_filter._rezero_gap`, applied through
  :func:`noise_filter.noise_filter`.
* **per-channel admission** — the CFZ-pH-7 noise surface directional-drift test, so per-channel
  work (the τ-vs-size fanning analysis) sees only channels carrying real directional change.

**A residual artifact class the established test does not target.** ``despike_frames`` is
intensity-primary and requires Copt to move the *same way* — it is built for glitches that dim or
brighten the whole detector. Arm B also carries Copt-primary excursions where obscuration spikes
while the spectral shape moves the other way (in the worst run, z_I = −1.2e5 against z_C = +555 at
one frame), which that sign test rejects by construction. Since ``k`` integrates
``(Copt − Copt∞)`` directly, such a frame inflates the denominator and depresses k. Mode
``"pipeline+copt"`` adds a Copt-only robust-z repair on top; it is kept separate and reported as
its own sensitivity axis rather than folded silently into "cleaned", and the shared pipeline
function is left untouched because the pH-study manuscript analyses depend on it.

Modes: ``"raw"`` (no QC), ``"pipeline"`` (established QC only), ``"pipeline+copt"`` (adds the
Copt repair). Every dropped, interpolated and retained frame is recorded in
:class:`OpticalRun.provenance`; nothing is written back over raw measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from diffractomorph_pipeline import ingest, noise_filter as nf, noise_surface as ns

COPT_MAX = 40.0          # ingest-level obstruction guard, unchanged from the original analysis
Z_ADMIT = 4.0            # per-channel directional-drift admission threshold
Z_COPT = 5.0             # robust-z on Copt's local ratio for the supplementary Copt repair
MODES = ("raw", "pipeline", "pipeline+copt")


@dataclass
class OpticalRun:
    t_min: np.ndarray            # minutes, re-zeroed to the first retained frame
    copt: np.ndarray             # optical concentration, %
    I: np.ndarray                # (frames × channels) intensities
    channels: np.ndarray         # channel labels aligned to I's columns
    admitted: np.ndarray         # bool per channel; all True when cleaning is off
    provenance: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.t_min.size)


def _trim(t, copt, I):
    """The original analysis' frame guard: finite Copt below the obstruction ceiling, and drop
    everything before a large acquisition gap in the first half of the run."""
    keep = np.isfinite(copt) & (copt <= COPT_MAX)
    t, copt, I = t[keep], copt[keep], I[keep]
    dropped_ceiling = int((~keep).sum())
    gap_dropped = 0
    if t.size >= 2:
        dt = np.diff(t)
        j = int(np.argmax(dt))
        if dt[j] > 0.4 and j < t.size // 2:
            gap_dropped = j + 1
            t, copt, I = t[j + 1:], copt[j + 1:], I[j + 1:]
    return t - t[0], copt, I, dropped_ceiling, gap_dropped


def _copt_repair(t, copt, z_thresh=Z_COPT, w=2):
    """Interpolate isolated Copt excursions the intensity-primary despike leaves behind.

    Flags frames whose Copt departs from its local neighbour median by more than ``z_thresh``
    robust z, then linearly interpolates them from retained neighbours. Only Copt is touched;
    the intensities are left as the pipeline despike produced them.
    """
    n = copt.size
    if n < 5:
        return copt, []
    nbm = np.array([np.median(copt[[j for j in range(max(0, k - w), min(n, k + w + 1)) if j != k]])
                    for k in range(n)])
    ratio = copt / np.maximum(nbm, 1e-9)
    z = nf._robust_z(ratio)
    flagged = list(np.flatnonzero(np.abs(z) > z_thresh))
    if not flagged:
        return copt, []
    keep = np.array([j for j in range(n) if j not in flagged])
    if keep.size < 2:
        return copt, []
    out = copt.copy()
    out[flagged] = np.interp(t[flagged], t[keep], copt[keep])
    return out, [int(k) for k in flagged]


def optical_run(rtf: Path | str, *, clean: bool | str = True, surface=None) -> OpticalRun:
    """Load one Arm B measurement under one QC mode (see :data:`MODES`).

    ``clean=True`` maps to ``"pipeline+copt"`` and ``clean=False`` to ``"raw"``.
    """
    mode = {True: "pipeline+copt", False: "raw"}.get(clean, clean)
    if mode not in MODES:
        raise ValueError(f"unknown optical mode {mode!r}; expected one of {MODES}")
    raw = ingest.extract_run(rtf)
    t = np.asarray(raw.t_min, float)
    copt = np.asarray(raw.copt, float)
    I = np.asarray(raw.I, float)
    channels = np.asarray(raw.channels)
    n_input = int(t.size)

    t, copt, I, dropped_ceiling, gap_dropped = _trim(t, copt, I)
    prov = {"mode": mode, "cleaned": mode != "raw", "n_frames_input": n_input,
            "n_dropped_copt_ceiling": dropped_ceiling, "n_dropped_pre_gap": gap_dropped,
            "copt_max": COPT_MAX}

    if mode == "raw":
        prov.update(n_lead_dropped=0, n_interior_interpolated=0, spike_frames=[],
                    n_copt_repaired=0, copt_repaired_frames=[],
                    n_frames_retained=int(t.size), n_channels_admitted=int(channels.size),
                    admission="none (raw)")
        return OpticalRun(t, copt, I, channels, np.ones(channels.size, bool), prov)

    I, t, copt, info = nf.despike_frames(I, t, copt)
    interior = list(info.get("interior_fixed", []))
    prov.update(n_lead_dropped=int(info.get("n_lead_dropped", 0)),
                n_interior_interpolated=len(interior),
                interior_interpolated_frames=[int(k) for k in interior],
                spike_frames=[int(k) for k in info.get("spike_frames", [])],
                n_frames_retained=int(t.size))

    if mode == "pipeline+copt":
        copt, repaired = _copt_repair(t, copt)
    else:
        repaired = []
    prov.update(n_copt_repaired=len(repaired), copt_repaired_frames=repaired,
                copt_repair_z=Z_COPT if mode == "pipeline+copt" else None)

    surface = ns.load_surface() if surface is None else surface
    triage = nf.noise_filter(I, t, list(channels), noise_surface=surface, copt=copt,
                             z_thresh=Z_ADMIT, despike=False)
    admitted = np.isin(channels, np.asarray(triage.active_channels))
    prov.update(n_channels_admitted=int(admitted.sum()),
                admitted_channels=[int(c) for c in channels[admitted]],
                admission=f"noise-surface directional drift, z>{Z_ADMIT}")
    return OpticalRun(t, copt, I, channels, admitted, prov)
