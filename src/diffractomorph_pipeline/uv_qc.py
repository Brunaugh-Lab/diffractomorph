"""UV dissolution-timecourse QC — keep-mask for a run's dissolved-concentration series.

The UV assay reports dissolved concentration C(t) over a dissolution run. Two kinds of point are not
real dissolution and must be dropped before fitting: a **physically impossible** point (dissolved above
the injected dose, beyond assay tolerance) and a **large local spike** (a sampled well bubble or a
pipetting error). :func:`screen_uv_dissolved` returns a boolean keep-mask over the timecourse.

    from diffractomorph_pipeline import uv_qc
    keep = uv_qc.screen_uv_dissolved(dissolved, loaded)
    t, c = t[keep], dissolved[keep]
"""
from __future__ import annotations

import numpy as np


def screen_uv_dissolved(dissolved, loaded, *, rec_cap=1.10, z=4.0, dev_min=0.25):
    """QC keep-mask for a run's UV dissolved-concentration timecourse. Drops a timepoint that is
    (a) **physically impossible** — dissolved > ``rec_cap`` × injected (can't dissolve more than was
    injected, beyond ~10% assay tolerance), or (b) a **large local spike** — deviates from the median of
    its ±1 neighbours by more than *both* ``z``·1.4826·MAD **and** ``dev_min`` × injected. The absolute
    ``dev_min`` gate is essential: the dissolved curve *rises then plateaus*, so a bare MAD test would
    flag the legitimate early-rise/curvature points once the flat plateau makes the MAD tiny — the gate
    means only a genuinely large excursion (a well bubble / pipetting error) is dropped. Returns a bool
    mask over the timecourse."""
    d = np.asarray(dissolved, dtype=float)
    loaded = float(loaded)
    keep = np.isfinite(d) & (d <= rec_cap * loaded)
    if d.size >= 3:
        # nanmedian for the ±1 baseline so a missing point doesn't poison its neighbours' residual; the
        # spike stats are then taken over the finite residuals only (a lone NaN must not blank the run).
        base = np.array([np.nanmedian(d[max(0, i - 1):i + 2]) for i in range(d.size)])
        resid = d - base
        fin = np.isfinite(resid)
        r = resid[fin]
        mad = float(np.median(np.abs(r - np.median(r)))) if r.size else 0.0
        thresh = max(z * 1.4826 * mad, dev_min * loaded)          # don't flag on a tiny plateau MAD
        keep &= np.abs(resid) <= thresh                           # NaN residual → already dropped above
    return keep
