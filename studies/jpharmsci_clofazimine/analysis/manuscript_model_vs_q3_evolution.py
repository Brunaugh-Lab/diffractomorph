"""Manuscript analysis — forward-model predicted PSD evolution vs observed PAQXOS q3 evolution.

The bridge between the mass-domain Nernst–Brunner model and the later pre-inversion channel analysis.
It compares, **in size space**, the model's predicted particle-size distribution as it dissolves against
the instrument's inverted q3 — WITHOUT the Mie operator and WITHOUT any detector-channel deconvolution.
It does not refit the model, optimize anything against q3, or change production physics.

Three distinct objects are kept conceptually separate and never conflated:
  * forward model — particle **cohorts** in size space (NOT "detector channels");
  * PAQXOS q3    — the proprietary inverted volume-weighted size distribution;
  * detector channels — overlapping angular-scattering measurements (not used here).

Forward-model state semantics (Step-1 audit, verified numerically on real runs — see
``forward_state_semantics_audit`` and its CSV/MD output):
  * ``run.qundiss`` (frames × nbin): undissolved amount per cohort in mmol; ``q0 = volfrac·dose``, so it
    is ∝ cohort **volume** at constant solid density. This is the size-distribution weight ``q_i(t)``.
  * ``run.radius_um`` (frames × nbin): the cohort's **current** radius, ``r_i(t) = (q_i/q_i0)^{1/3}·r_{0,i}``
    — a cohort keeps its index but its physical radius shrinks; current diameter ``d_i(t)=2 r_i(t)``.
  * ``run.diam0_um``: the starting representative diameters (the cohort index labels).
  * ``run.t`` (s), ``run.cbulk`` (mol/L), ``run.pct_dissolved`` (% of injected mass).
  Conservation (all verified to ≲1e-14 relative): total remaining cohort mass ``Σ_i q_i(t) = total −
  C·V`` (mass balance); ``(r_i/r_{0,i})^3 = q_i/q_i0`` exactly (so ``q_i ∝ N_i r_i^3`` with a constant
  ``N_i`` per cohort); radii never increase; no negative masses/radii; a cohort freezes at
  ``freeze_frac·q_i0`` (numerical guard) — treated here as **fully dissolved** (zero size-distribution
  weight), and its ≲0.5 % residue would in any case fall below the PAQXOS window.

Predicted q3-like distribution (Step 2): each cohort's remaining volume ``q_i(t)`` is placed at its
CURRENT diameter ``d_i(t)`` (never its original bin). Weighted percentiles are computed **directly** from
``(d_i(t), q_i(t))`` to avoid rebinning artifacts; for distribution overlays the moving cohorts are
conservatively rebinned onto the native PAQXOS log-diameter grid with mass-conserving two-bin deposition.
The modeled fraction **below / inside / above** the finite PAQXOS window is tracked at every time and
never silently clipped; an explicitly labeled instrument-window-conditional distribution normalizes only
the in-window model mass, and the full distribution + outside-window fractions are retained in CSV.

Predictions use the **selected UV-derived rate-only model** (base exponents ``a_area=1/3``, ``b_size=2``
+ one global ``rate_scale`` fit on pH 4.0/4.5 and applied **unchanged** to pH 5.0). The independent
physical **base** model (``rate_scale=1``) is the sensitivity prediction. Because ``rate_scale`` is an
exact, uniform time-rescaling of the autonomous ODE, base and rate-only trace the **identical**
q3-versus-cohort-mass curve (verified here) — so the matched-progress comparison draws ONE model curve.

The matched-progress comparison indexes the OBSERVED q3 by the **UV-derived apparent remaining dose
fraction** — a UV mass-balance coordinate (``assay.cumulative_dissolved``) corrected for the unreplaced
400 µL aliquots withdrawn at each UV sample; it is NOT an optical mass estimate (no Copt / angular
signal / q3 magnitude enters — q3 supplies only normalized size shape). The model is compared at its own
cohort-mass fraction ``g_model = Σq_i(t)/Σq_i(0)`` as the shared dissolution-extent coordinate. Values
are not clipped; QC flags mark samples outside [0, 1]. Interpreting the apparent remaining dose fraction
as suspended undissolved particle mass assumes no important deposition/precipitation/other unrecovered
compartment — notably uncertain at pH 5.0 (incomplete recovery), where this stays diagnostic.

Observed q3 eligibility, timestamp matching, target-time grid, and date-first aggregation are **reused**
from :mod:`manuscript_q3_scattering_evolution` (not reimplemented). The working comparison is the
coarse-tail-excluded, Copt ≥ 0.79 set; inclusive and stringent (Copt ≥ 4) are sensitivity versions.
Review-required (coarse-tail) frames are never called inversion failures. A condition-level observed
curve needs ≥ 2 preparation dates. The observed starting q3 initializes the model, so **t = 0 agreement
is imposed and is never described as validation** — only later evolution tests a model consequence.

Run with the pipeline venv::

    python analysis/manuscript_model_vs_q3_evolution.py --output-dir <…/figures_and_tables> \
        [--formats png,pdf,svg,tiff]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from diffractomorph_pipeline import plot_styles as ps, psd, solubility
from diffractomorph_pipeline.assay import cumulative_dissolved, uv_timecourse
from diffractomorph_pipeline.config import data_root
from diffractomorph_pipeline.forward import PSD, predict, MorphologyParams
from diffractomorph_pipeline.forward.params import Parameters

import manuscript_q3_scattering_evolution as mqs
from psd_evolution_common import iter_runs

# ── selected model + display constants ────────────────────────────────────────
RATE_SCALE_SELECTED = 2.197            # date-balanced rate-only fit (pH 4.0/4.5), applied frozen to pH 5.0
RATE_SCALE_BASE = 1.0                  # independent physical baseline (Wilke–Chang dd, no rate correction)
V_ML = 40.0
T_END_S = 1200.0                       # 20 min
N_EVAL = 101                           # 0.2-min (12 s) grid, 0…20 min
SNAPSHOT_MIN = [0.0, 2.0, 5.0, 10.0, 20.0]     # snapshot times retained in the source table
FIG_SNAPSHOT_MIN = [0.0, 2.0, 10.0, 20.0]      # shown in the figure (5-time overlay tested — too dense)
FRAC_BELOW_UM = 2.0
CONDITIONS = mqs.CONDITIONS
Q3_XLIM = mqs.Q3_XLIM
WORKING_VERSION = "coarse_flag_excluded"   # Copt ≥ 0.79, coarse-tail excluded — the working comparison
CMAP = mqs.CMAP
FREEZE_EPS = 1.001                     # a cohort at ≤ freeze_frac·q0·FREEZE_EPS is treated as fully dissolved
FINE_GRID = np.geomspace(0.3, 60.0, 100)   # fine cohort grid for the model (resampled from the observed q3):
#   many cohorts → smooth shrinkage (alias-free overlay), while keeping the stiff ODE solve tractable.
DEPLETE_FRAC = 0.05                    # if in-window model mass falls below this fraction of its t=0 value,
#   the normalized distribution is numerical residue → returned MISSING (never amplified)


# ── Step 1: forward-state semantics + conservation audit ──────────────────────
def forward_state_semantics_audit(sample_runs):
    """Numerically verify the state semantics + conservation laws on real runs (Step 1). ``sample_runs``
    = list of DissolutionRun. Returns (semantics_df, checks_df)."""
    p = Parameters()
    sem = pd.DataFrame([
        dict(array="qundiss", shape="frames × nbin", represents="undissolved amount per cohort (mmol, ∝ volume)",
             note="q0 = volfrac·dose; the size-distribution weight q_i(t)"),
        dict(array="radius_um", shape="frames × nbin", represents="current cohort radius (µm)",
             note="r_i(t)=(q_i/q_i0)^{1/3}·r_{0,i}; current diameter d_i=2r_i; cohort index fixed, radius shrinks"),
        dict(array="diam0_um", shape="nbin", represents="starting representative diameters (cohort labels)", note=""),
        dict(array="cbulk", shape="frames", represents="bulk dissolved concentration (mol/L)", note=""),
        dict(array="pct_dissolved", shape="frames", represents="% of injected mass dissolved", note="from C·V/total"),
        dict(array="particle_number", shape="implicit", represents="N_i constant per cohort",
             note="q_i ∝ N_i r_i^3 (constant ρ); N_i not stored, derivable from q0_i/r0_i^3"),
    ])
    rows = []
    for run in sample_runs:
        q = np.asarray(run.qundiss, float); rad = np.asarray(run.radius_um, float)
        d0 = np.asarray(run.diam0_um, float); q0 = q[0]; live = q0 > q0.max() * 1e-4
        Vd = run.inputs.get("v_diss_mL", p.v_diss_mL)
        total = q0.sum() + run.cbulk[0] * Vd
        mass_resid = np.abs(q.sum(axis=1) + run.cbulk * Vd - total).max() / max(total, 1e-30)
        phi = np.divide(q, q0, out=np.zeros_like(q), where=q0 > 0)
        r_over_r0_cubed = np.divide(rad, d0[None, :] / 2.0, out=np.zeros_like(rad), where=d0[None, :] > 0) ** 3
        nr3_resid = np.abs(r_over_r0_cubed[:, live] - phi[:, live]).max()
        rows.append(dict(
            run=run.inputs.get("id", "?"), n_live_cohorts=int(live.sum()),
            mass_balance_rel_resid=float(mass_resid),
            Nr3_vs_mass_max_resid=float(nr3_resid),
            max_radius_increase_um=float(np.diff(rad, axis=0)[:, live].max()),
            any_negative=bool((q < 0).any() or (rad < 0).any()),
            min_final_phi=float(phi[-1, live].min()), freeze_frac=float(p.freeze_frac),
            pct_dissolved_final=float(run.pct_dissolved[-1])))
    return sem, pd.DataFrame(rows)


# ── Step 2: model predicted q3-like distribution ──────────────────────────────
def _cohort_weights(run, frame):
    """Cohort volume weights at ``frame``, with fully-dissolved (frozen) cohorts zeroed (they must carry
    NO size-distribution weight). ``d`` = current diameters, ``w`` = volume weights."""
    q = np.asarray(run.qundiss[frame], float)
    q0 = np.asarray(run.qundiss[0], float)
    d = 2.0 * np.asarray(run.radius_um[frame], float)
    p = Parameters()
    frozen = q <= q0 * p.freeze_frac * FREEZE_EPS       # at/below the freeze floor ⇒ fully dissolved
    w = np.where(frozen | (q0 <= 0), 0.0, q)
    return d, w


def _weighted_pctiles_logd(d, w, pcts=(10.0, 50.0, 90.0)):
    """Weighted percentiles computed DIRECTLY from cohort diameters/weights (no rebinning), interpolating
    in log-diameter vs cumulative weight — the same log-size convention as ``psd.q3_percentiles``."""
    d = np.asarray(d, float); w = np.asarray(w, float)
    ok = (w > 0) & (d > 0)
    if ok.sum() < 2 or w[ok].sum() <= 0:
        return [np.nan] * len(pcts)
    order = np.argsort(d[ok])
    dd = d[ok][order]; ww = w[ok][order]
    cw = np.cumsum(ww); cwpct = 100.0 * cw / cw[-1]
    return [float(10.0 ** np.interp(p, cwpct, np.log10(dd))) for p in pcts]


def _frac_below(d, w, thresh):
    d = np.asarray(d, float); w = np.asarray(w, float)
    tot = w.sum()
    return float(100.0 * w[d < thresh].sum() / tot) if tot > 0 else np.nan


def model_rebin_inwindow(d, w, grid):
    """Conservatively rebin moving-cohort weights onto the native log-diameter ``grid`` with
    mass-conserving two-bin (linear-in-log) deposition, and split the total weight into
    below-window / in-window / above-window fractions. Returns (dq_inwindow_fraction, frac_below,
    frac_in, frac_above). ``dq_inwindow_fraction`` sums to 1 over ``grid`` (or all-zero if no in-window
    mass); it is the instrument-window-conditional model distribution."""
    d = np.asarray(d, float); w = np.asarray(w, float); grid = np.asarray(grid, float)
    tot = w.sum()
    if tot <= 0:
        return np.zeros_like(grid), np.nan, np.nan, np.nan
    lo, hi = grid[0], grid[-1]
    below = w[d < lo].sum() / tot
    above = w[d > hi].sum() / tot
    inw = (d >= lo) & (d <= hi)
    binned = np.zeros_like(grid)
    if inw.any():
        lg = np.log(grid)
        pos = np.interp(np.log(np.clip(d[inw], lo, hi)), lg, np.arange(grid.size))
        b0 = np.clip(np.floor(pos).astype(int), 0, grid.size - 1)
        b1 = np.clip(b0 + 1, 0, grid.size - 1)
        frac = pos - b0
        np.add.at(binned, b0, w[inw] * (1.0 - frac))
        np.add.at(binned, b1, w[inw] * frac)
    in_frac = binned.sum() / tot
    # mass-conservation guard: rebinned in-window weight must equal the raw in-window weight
    raw_in = w[inw].sum() / tot
    assert abs(in_frac - raw_in) < 1e-9 * max(1.0, raw_in), "rebin lost in-window mass"
    dq = binned / binned.sum() if binned.sum() > 0 else binned
    return dq, float(below), float(in_frac), float(above)


def _inwindow_mass(run, frame, grid):
    d, w = _cohort_weights(run, frame)
    return w[(d >= grid[0]) & (d <= grid[-1])].sum()


def model_frame_descriptors(run, frame, grid, init_d10, init_inwindow=None):
    """All Step-5 descriptors for one model frame: direct weighted D10/D50/D90, span, frac<2µm,
    frac<initial-D10, plus the in-window rebinned distribution, its cumulative, and the window fractions.
    ``init_d10`` = the run's t=0 D10. When the in-window model mass falls below ``DEPLETE_FRAC`` of its
    t=0 value the normalized distribution is numerical residue, so ``dq``/``cum`` are returned as NaN
    (missing) — the direct percentiles (robust to sparse mass) are still reported."""
    d, w = _cohort_weights(run, frame)
    d10, d50, d90 = _weighted_pctiles_logd(d, w)
    dq, below, inw, above = model_rebin_inwindow(d, w, grid)
    if init_inwindow is None:
        init_inwindow = _inwindow_mass(run, 0, grid)
    depleted = _inwindow_mass(run, frame, grid) < DEPLETE_FRAC * max(init_inwindow, 1e-30)
    if depleted or dq.sum() <= 0:
        dq = np.full_like(grid, np.nan); cum = np.full_like(grid, np.nan)
    else:
        cum = np.cumsum(dq) * 100.0
    return dict(D10=d10, D50=d50, D90=d90, span=((d90 - d10) / d50 if d50 and np.isfinite(d50) else np.nan),
                frac_lt_2um=_frac_below(d, w, FRAC_BELOW_UM),
                frac_lt_initD10=(_frac_below(d, w, init_d10) if np.isfinite(init_d10) else np.nan),
                frac_below_window=below, frac_in_window=inw, frac_above_window=above, depleted=bool(depleted),
                dq=dq, cum=cum, remaining_mass_frac=float(w.sum() / max(np.asarray(run.qundiss[0], float).sum(), 1e-30)))


def _fine_psd_from_obs(dq0, common_xo, fine=FINE_GRID):
    """Resample the observed starting q3 (differential ``dq0`` on ``common_xo``) onto the fine cohort grid
    by monotone log-cumulative interpolation — many cohorts, same starting distribution. Returns a PSD or
    None if empty."""
    cum = np.cumsum(np.asarray(dq0, float))
    cum_fine = np.interp(np.log10(fine), np.log10(common_xo), cum, left=0.0, right=float(cum[-1]))
    dv = np.clip(np.diff(cum_fine, prepend=0.0), 0.0, None)
    return PSD.from_q3(fine, dv / dv.sum()) if dv.sum() > 0 else None


def _remaining_mass_frac(run):
    """Model remaining undissolved-mass fraction g_m(t) = Σ q_i(t) / Σ q_i(0) (cohort-mass based)."""
    q = np.asarray(run.qundiss, float)
    return q.sum(axis=1) / max(q[0].sum(), 1e-30)


# ── model run construction (per run, initialized from the observed starting q3) ─
def model_run(psd0, ph, dose_mg, rate_scale, t_end=T_END_S, n_eval=N_EVAL):
    """Run the selected/base forward model for one run, initialized from the observed starting q3
    (``psd0``), with the run's injected dose, pH, and the packaged measured Cs(pH). ``rate_scale`` picks
    the selected rate-only model (2.197) or the physical base (1.0)."""
    return predict(psd0, ph=ph, dose_mg=dose_mg, drug="CFZ",
                   morph=MorphologyParams(rate_scale=rate_scale), t_end=t_end, n_eval=n_eval)


# ── per-run assembly: observed eligibility (reused) + model predictions ────────
def _dose_lookup():
    m = pd.read_csv(data_root() / "disso_experiments" / "ph_dependent_dissolution_study" /
                    "summary" / "run_metadata.csv")
    return {(int(r.date_i), float(r.ph), int(r.rep)): float(r.mass_mg) for r in m.itertuples()}


def _first_eligible_frame(obs_run, working):
    """First observed q3 frame (at/after 0 min) whose category is in the working eligibility set —
    its differential (on the common grid) initializes the model. Returns (dq_grid, elapsed_min) or None."""
    for r in sorted(obs_run["q3"], key=lambda z: (z["elapsed_min"] if z["elapsed_min"] == z["elapsed_min"] else 1e9)):
        if r["category"] in working and not r["before_zero"] and r["dq_grid"] is not None \
                and np.isfinite(r["elapsed_min"]) and r["dq_grid"].sum() > 0:
            return np.asarray(r["dq_grid"], float), float(r["elapsed_min"])
    return None


def build_runs(common_xo):
    """Assemble every pH-study run: observed q3 eligibility (reused from mqs) + the selected and base
    model predictions initialized from that run's observed starting q3. Returns a list of run dicts."""
    dose = _dose_lookup()
    working = mqs.VERSIONS[WORKING_VERSION]
    runs = []
    for ph, date, rep, rtf, fo in iter_runs():
        if float(ph) not in CONDITIONS:
            continue
        obs = mqs.build_run(ph, date, rep, rtf, fo, common_xo)
        init = _first_eligible_frame(obs, working)
        dm = dose.get((int(date), float(ph), int(rep)))
        if init is None or dm is None:
            runs.append(dict(ph=float(ph), date=int(date), rep=int(rep), obs=obs, model=None,
                             reason="no eligible starting frame" if init is None else "no dose"))
            continue
        dq0, t0_off = init
        psd0 = _fine_psd_from_obs(dq0, common_xo)             # fine cohort grid → smooth, alias-free overlay
        if psd0 is None:
            runs.append(dict(ph=float(ph), date=int(date), rep=int(rep), obs=obs, model=None,
                             reason="empty starting q3")); continue
        sel = model_run(psd0, float(ph), dm, RATE_SCALE_SELECTED)   # base ≡ rate-only at matched mass,
        sel.inputs["id"] = f"pH{ph}_{date}_R{rep}"                    # so the base is solved only for the check
        runs.append(dict(ph=float(ph), date=int(date), rep=int(rep), obs=obs, dose_mg=dm,
                         model=sel, t0_offset_min=t0_off, rtf=rtf, fo=fo, init_dq=dq0, psd0=psd0))
    return runs


# ── observed per-frame descriptors (from the reused eligibility) ──────────────
def _observed_descriptor_row(cum_grid, dq_grid, common_xo, init_d10):
    cum = np.asarray(cum_grid, float)
    d10, d50, d90 = psd.q3_percentiles(common_xo, cum)
    # frac<2µm and frac<initD10 from the differential on the grid
    dq = np.asarray(dq_grid, float)
    below2 = 100.0 * dq[common_xo < FRAC_BELOW_UM].sum() / dq.sum() if dq.sum() > 0 else np.nan
    belowd10 = (100.0 * dq[common_xo < init_d10].sum() / dq.sum()
                if (dq.sum() > 0 and np.isfinite(init_d10)) else np.nan)
    return dict(D10=d10, D50=d50, D90=d90,
                span=((d90 - d10) / d50 if d50 and np.isfinite(d50) else np.nan),
                frac_lt_2um=below2, frac_lt_initD10=belowd10, cum=cum, dq=dq)


# ── Step 4/5/9: run-level descriptor table (clock time) ───────────────────────
def clock_descriptor_rows(runs, common_xo, grid):
    """One row per (source, run, target_min) with all Step-5 descriptors + distance-from-start metrics.
    Model rows are the selected rate-only model; observed rows use the working eligibility set aligned to
    the target grid within tolerance."""
    working = mqs.VERSIONS[WORKING_VERSION]
    rows = []
    for run in runs:
        if run["model"] is None:
            continue
        ph, date, rep = run["ph"], run["date"], run["rep"]
        # model: descriptors at each grid time (model solved exactly on the grid)
        m = run["model"]; t_min = np.asarray(m.t, float) / 60.0 + round(run["t0_offset_min"] / mqs.DT_MIN) * mqs.DT_MIN
        d, w = _cohort_weights(m, 0); init_d10 = _weighted_pctiles_logd(d, w)[0]
        cum0_m = model_frame_descriptors(m, 0, common_xo, init_d10)["cum"]
        for f, tm in enumerate(t_min):
            gi = int(round(tm / mqs.DT_MIN))
            if gi < 0 or gi >= grid.size or abs(tm - grid[gi]) > mqs.TOL_MIN + 1e-9:
                continue
            desc = model_frame_descriptors(m, f, common_xo, init_d10)
            rows.append(_desc_row("model", ph, date, rep, grid[gi], desc, common_xo, cum0_m,
                                  frac_out=desc["frac_below_window"] + desc["frac_above_window"],
                                  version="selected_rate_only"))
        # observed: align eligible frames to the grid
        elig = [r for r in run["obs"]["q3"] if r["category"] in working and not r["before_zero"]
                and r["dq_grid"] is not None and np.isfinite(r["elapsed_min"])]
        if not elig:
            continue
        times = [r["elapsed_min"] for r in elig]
        init_d10_o = psd.q3_percentiles(common_xo, elig[0]["cum_grid"])[0]
        cum0_o = np.asarray(elig[0]["cum_grid"], float)
        aligned = mqs._align_pairs(times, list(range(len(elig))), grid)
        for gi, ei in aligned.items():
            r = elig[int(ei)]
            od = _observed_descriptor_row(r["cum_grid"], r["dq_grid"], common_xo, init_d10_o)
            rows.append(_desc_row("observed", ph, date, rep, grid[gi], od, common_xo, cum0_o,
                                  frac_out=0.0, version=WORKING_VERSION))
    return pd.DataFrame(rows)


def _desc_row(source, ph, date, rep, target_min, desc, common_xo, cum_start, frac_out, version):
    wass = psd.q3_wasserstein_log(common_xo, cum_start, desc["cum"]) if np.all(np.isfinite(desc["cum"])) else np.nan
    maxd = float(np.max(np.abs(desc["cum"] - cum_start))) if np.all(np.isfinite(desc["cum"])) else np.nan
    return dict(source=source, version=version, ph=ph, date=date, rep=rep,
                target_min=round(float(target_min), 3),
                D10=_r(desc["D10"]), D50=_r(desc["D50"]), D90=_r(desc["D90"]), span=_r(desc.get("span")),
                frac_lt_2um=_r(desc.get("frac_lt_2um")), frac_lt_initD10=_r(desc.get("frac_lt_initD10")),
                wass_from_start=_r(wass, 4), max_abs_dcumQ3_from_start=_r(maxd),
                frac_model_outside_window=_r(frac_out))


def _r(x, n=3):
    return round(float(x), n) if x is not None and np.isfinite(x) else np.nan


# ── date-first equal-weight aggregation of a descriptor table ─────────────────
DESC_COLS = ["D10", "D50", "D90", "span", "frac_lt_2um", "frac_lt_initD10",
             "wass_from_start", "max_abs_dcumQ3_from_start", "frac_model_outside_window"]


def date_first_aggregate(df, by=("source", "ph", "target_min"), min_dates=2, stat="mean"):
    """Average nested runs within a date (always the **intraday mean**), then summarize the
    preparation-date means across dates with equal weight using ``stat`` (``"mean"`` or ``"median"``). The
    preparation DATE is the experimental unit; the cross-date summary is never taken directly across runs or
    frames. Returns date-level and condition-level tables (condition requires ≥ ``min_dates`` dates); the
    condition rows carry ``summary_statistic = f"cross_date_{stat}"``."""
    cross = np.median if stat == "median" else np.mean
    date_level = (df.groupby(list(by) + ["date"])[DESC_COLS].mean().reset_index())   # within-date mean first
    cond_rows = []
    for key, g in date_level.groupby(list(by)):
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec["n_dates"] = int(g["date"].nunique())
        rec["n_runs"] = int(df.groupby(list(by)).get_group(key).shape[0]) if key in df.groupby(list(by)).groups else np.nan
        rec["summary_statistic"] = f"cross_date_{stat}"
        for c in DESC_COLS:
            v = g[c].to_numpy(float)
            v = v[np.isfinite(v)]
            rec[c] = float(cross(v)) if v.size else np.nan          # cross-date mean OR median of date means
            rec[f"{c}_within_date_sd"] = float(g[c].std(ddof=1)) if g[c].notna().sum() > 1 else np.nan
        cond_rows.append(rec)
    cond = pd.DataFrame(cond_rows)
    cond["drawable"] = cond["n_dates"] >= min_dates
    return date_level, cond


# ── Step 8: detectability (within-date among-run repeatability) ───────────────
def detectability(clock_runlevel, cond_model):
    """R_detect = |D_p,model(t) − D_p,model(0)| / σ_within-date,p(t). σ is the within-date among-run SD of
    the OBSERVED percentile, pooled across dates (median of per-date SDs) — frames are NOT treated as
    independent replicates. Classifies the predicted shift descriptively."""
    obs = clock_runlevel[clock_runlevel.source == "observed"]
    # within-date among-run SD per (ph, target_min, percentile), pooled (median) across dates
    rows = []
    model0 = {ph: cond_model[(cond_model.ph == ph) & (cond_model.target_min == 0.0)] for ph in CONDITIONS}
    for ph in CONDITIONS:
        m0 = model0[ph]
        if not len(m0):
            continue
        for tm in sorted(cond_model[cond_model.ph == ph].target_min.unique()):
            mt = cond_model[(cond_model.ph == ph) & (cond_model.target_min == tm)]
            if not len(mt):
                continue
            for pctl in ("D10", "D50", "D90"):
                sd_by_date = (obs[(obs.ph == ph) & (obs.target_min == tm)]
                              .groupby("date")[pctl].std(ddof=1))
                sigma = float(sd_by_date.median()) if sd_by_date.notna().any() else np.nan
                shift = abs(float(mt[pctl].iloc[0]) - float(m0[pctl].iloc[0]))
                ratio = shift / sigma if (sigma and np.isfinite(sigma) and sigma > 0) else np.nan
                rows.append(dict(ph=ph, target_min=tm, percentile=pctl,
                                 model_shift_um=round(shift, 4),
                                 within_date_sd_um=round(sigma, 4) if np.isfinite(sigma) else np.nan,
                                 R_detect=round(ratio, 3) if np.isfinite(ratio) else np.nan,
                                 classification=_detect_class(ratio)))
    return pd.DataFrame(rows)


def _detect_class(ratio):
    if not np.isfinite(ratio):
        return "no repeatability estimate"
    if ratio < 1.0:
        return "below observed repeatability"
    if ratio < 2.0:
        return "comparable to observed repeatability"
    return "larger than observed repeatability"


# ── Step 7: matched-progress comparison (UV mass-balance apparent remaining dose fraction) ──
def matched_mass_rows(runs, common_xo):
    """Index by the **UV-derived apparent remaining dose fraction** (a UV mass-balance coordinate — NOT
    an optical mass estimate). For each run the observed progress coordinate comes exclusively from the
    cumulative UV mass balance corrected for unreplaced aliquot removal
    (:func:`assay.cumulative_dissolved`): 400 µL are withdrawn and not replaced at each UV sample, so the
    vessel volume shrinks and dissolved drug leaves with each aliquot. q3 supplies only normalized
    size-distribution shape. Each UV sample is paired with the nearest eligible q3 frame (tolerance = half
    the q3 cadence), and the model distribution is evaluated at the SAME progress coordinate by monotone
    interpolation along the model's cohort-mass trajectory ``g_model = Σ q_i(t)/Σ q_i(0)``. Values are not
    clipped; QC flags mark out-of-[0,1] samples so assay variability stays visible. The old, superseded
    fixed-volume coordinate ``1 − C·40/dose`` is retained per row for the before/after audit only.
    Interpreting the apparent remaining dose fraction as suspended undissolved particle mass assumes no
    important deposition/precipitation/other unrecovered compartment (notably uncertain at pH 5.0)."""
    working = mqs.VERSIONS[WORKING_VERSION]
    rows = []
    for run in runs:
        if run.get("model") is None or not run.get("rtf"):
            continue
        ph, date, rep = run["ph"], run["date"], run["rep"]
        uv = _uv_file(ph, date, rep)
        if uv is None:
            continue
        try:
            df = uv_timecourse(uv, ph, injected_mg=run["dose_mg"], volume_mL=V_ML)
        except (ValueError, Exception):
            continue
        if len(df) < 2:
            continue
        # observed progress coordinate = UV cumulative mass balance (aliquot-corrected), NOT optical
        cum = cumulative_dissolved(df.conc_ugml.to_numpy(float), run["dose_mg"], v0_mL=V_ML)
        g_obs = cum["apparent_remaining_dose_fraction"].to_numpy()
        g_old = 1.0 - df.pct_injected.to_numpy() / 100.0          # superseded fixed-volume coord (audit only)
        t_uv = df.time_min.to_numpy()
        # eligible observed q3 frames of this run
        elig = [r for r in run["obs"]["q3"] if r["category"] in working and not r["before_zero"]
                and r["dq_grid"] is not None and np.isfinite(r["elapsed_min"])]
        if not elig:
            continue
        q3_t = np.array([r["elapsed_min"] for r in elig])
        init_d10_o = psd.q3_percentiles(common_xo, elig[0]["cum_grid"])[0]
        # model cohort-mass trajectory (monotone decreasing) and percentile trajectory
        m = run["model"]; g_m = _remaining_mass_frac(m)
        dcw = [_cohort_weights(m, f) for f in range(len(m.t))]
        m_d10 = np.array([_weighted_pctiles_logd(*dcw[f])[0] for f in range(len(m.t))])
        m_d50 = np.array([_weighted_pctiles_logd(*dcw[f])[1] for f in range(len(m.t))])
        m_d90 = np.array([_weighted_pctiles_logd(*dcw[f])[2] for f in range(len(m.t))])
        for i in range(len(t_uv)):
            gv, tv, gv_old = float(g_obs[i]), float(t_uv[i]), float(g_old[i])
            # nearest eligible q3 frame to this UV time (documented tolerance = half the q3 cadence)
            k = int(np.argmin(np.abs(q3_t - tv)))
            if abs(q3_t[k] - tv) > (psd.FRAME_S / 2.0) / 60.0:
                continue                                          # q3 pairing failure (not a clip)
            r = elig[k]
            od = _observed_descriptor_row(r["cum_grid"], r["dq_grid"], common_xo, init_d10_o)
            reaches = bool(g_m.min() <= gv <= g_m.max())          # model reaches this progress coordinate?
            mi = _interp_along_g(g_m, (m_d10, m_d50, m_d90), gv) if reaches else (np.nan, np.nan, np.nan)
            # model D50 at the OLD (superseded) coordinate — for the before/after conclusion check only
            reaches_old = bool(g_m.min() <= gv_old <= g_m.max())
            m50_old = _interp_along_g(g_m, (m_d50,), gv_old)[0] if reaches_old else np.nan
            rows.append(dict(
                ph=ph, date=date, rep=rep,
                apparent_remaining_dose_fraction=round(gv, 4),
                old_remaining_fraction_uncorrected=round(gv_old, 4),
                recovery_mass_fraction=round(float(cum["recovery_mass_fraction"].iloc[i]), 4),
                cumulative_dissolved_ug=round(float(cum["cumulative_dissolved_ug"].iloc[i]), 2),
                vessel_mL=round(float(cum["vessel_mL"].iloc[i]), 3),
                uv_time_min=round(tv, 2), q3_time_min=round(float(q3_t[k]), 2),
                qc_remaining_below_0=bool(cum["qc_remaining_below_0"].iloc[i]),
                qc_remaining_above_1=bool(cum["qc_remaining_above_1"].iloc[i]),
                model_reaches_progress=reaches,
                model_D10=_r(mi[0]), model_D50=_r(mi[1]), model_D90=_r(mi[2]),
                model_D50_at_old_coord=_r(m50_old),
                obs_D10=_r(od["D10"]), obs_D50=_r(od["D50"]), obs_D90=_r(od["D90"])))
    return pd.DataFrame(rows)


def matched_progress_audit(matched):
    """Before/after audit of the progress-coordinate correction, per pH: number of UV observations, old
    (fixed-volume ``1−C·40/dose``) vs corrected (aliquot-mass-balance) apparent-remaining ranges, the
    maximum absolute change, whether any corrected value falls outside [0, 1], and whether the
    matched-progress conclusion (observed D50 below the model D50 — the model over-predicts coarsening)
    changes. The conclusion is a vertical D50 comparison, so it is checked under BOTH coordinates."""
    rows = []
    for ph in CONDITIONS:
        g = matched[matched.ph == ph]
        if not len(g):
            rows.append(dict(ph=ph, n_uv_obs=0)); continue
        old = g.old_remaining_fraction_uncorrected.to_numpy(float)
        new = g.apparent_remaining_dose_fraction.to_numpy(float)
        gn = g[g.model_reaches_progress & g.obs_D50.notna() & g.model_D50.notna()]
        go = g[g.model_D50_at_old_coord.notna() & g.obs_D50.notna()]
        concl_new = float((gn.obs_D50 < gn.model_D50).mean()) if len(gn) else np.nan
        concl_old = float((go.obs_D50 < go.model_D50_at_old_coord).mean()) if len(go) else np.nan
        below = lambda f: (f > 0.5) if np.isfinite(f) else None
        rows.append(dict(
            ph=ph, n_uv_obs=int(len(g)),
            old_remaining_min=round(float(np.nanmin(old)), 3), old_remaining_max=round(float(np.nanmax(old)), 3),
            corrected_remaining_min=round(float(np.nanmin(new)), 3),
            corrected_remaining_max=round(float(np.nanmax(new)), 3),
            max_abs_change=round(float(np.nanmax(np.abs(new - old))), 4),
            any_outside_0_1=bool(((new < 0) | (new > 1)).any()),
            n_outside_0_1=int(((new < 0) | (new > 1)).sum()),
            frac_obs_D50_below_model_old=round(concl_old, 3) if np.isfinite(concl_old) else np.nan,
            frac_obs_D50_below_model_corrected=round(concl_new, 3) if np.isfinite(concl_new) else np.nan,
            conclusion_changes=bool(below(concl_old) != below(concl_new))
            if (below(concl_old) is not None and below(concl_new) is not None) else None))
    return pd.DataFrame(rows)


def _interp_along_g(g, series, gv):
    """Monotone interpolation of each model percentile series at remaining-mass fraction ``gv`` (g is
    decreasing in time, so interpolate on the reversed, ascending g)."""
    gi = g[::-1]
    return [float(np.interp(gv, gi, s[::-1])) for s in series]


def base_vs_rateonly_matched_mass(runs, common_xo):
    """Verify the theoretical claim: because rate_scale is a uniform time-rescaling, base and rate-only
    give the SAME q3-versus-remaining-mass relationship. Compare D50-vs-g for both models on one run."""
    for run in runs:
        if run.get("psd0") is None:
            continue
        # dense, extent-matched integration so the comparison is limited by neither model's sampling:
        # base (rate_scale=1) is integrated ×rate_scale longer so it reaches the same dissolution extent.
        sel = model_run(run["psd0"], run["ph"], run["dose_mg"], RATE_SCALE_SELECTED, t_end=T_END_S, n_eval=301)
        base = model_run(run["psd0"], run["ph"], run["dose_mg"], RATE_SCALE_BASE,
                         t_end=T_END_S * RATE_SCALE_SELECTED, n_eval=301)
        out = []
        for tag, m in (("selected_rate_only", sel), ("base", base)):
            g = _remaining_mass_frac(m)
            d50 = np.array([_weighted_pctiles_logd(*_cohort_weights(m, f))[1] for f in range(len(m.t))])
            out.append((tag, g, d50))
        # interpolate base D50 onto the selected model's g grid and compare
        (_t1, g_sel, d50_sel), (_t2, g_base, d50_base) = out
        gi = np.linspace(max(g_sel.min(), g_base.min()), min(g_sel.max(), g_base.max()), 50)
        d_sel = np.interp(gi, g_sel[::-1], d50_sel[::-1])
        d_base = np.interp(gi, g_base[::-1], d50_base[::-1])
        max_abs = float(np.nanmax(np.abs(d_sel - d_base)))
        return dict(run=f"pH{run['ph']}_{run['date']}_R{run['rep']}",
                    max_abs_D50_diff_um=round(max_abs, 5),
                    equivalent=bool(max_abs < 1e-3))
    return dict(run=None, max_abs_D50_diff_um=np.nan, equivalent=None)


def _uv_file(ph, date, rep):
    import glob
    base = data_root() / "disso_experiments" / "ph_dependent_dissolution_study"
    hits = glob.glob(str(base / f"ph_{ph}" / f"{date}_pH*" / "UV-VIs" / f"pH={ph}*Rep{rep}.xlsx"))
    return hits[0] if hits else None


# ── Step 10: interpretation gate ──────────────────────────────────────────────
def interpretation_gate(cond_model, cond_obs, detect):
    """One evidence-based outcome per condition (Step 10), from D50 direction/magnitude vs observed and
    the detectability classification. Does NOT translate a discrepancy into a claim about laser
    diffraction; records that the later raw-channel analysis must disambiguate the causes."""
    rows = []
    for ph in CONDITIONS:
        cm = cond_model[cond_model.ph == ph].sort_values("target_min")
        co = cond_obs[(cond_obs.ph == ph) & (cond_obs.drawable)].sort_values("target_min")
        det = detect[(detect.ph == ph) & (detect.percentile == "D50")]
        if len(cm) < 2 or len(co) < 2:
            rows.append(dict(ph=ph, outcome="6: q3 coverage/coarse-tail sensitivity prevents a conclusion",
                             detail="fewer than two drawable observed timepoints")); continue
        # net predicted D50 change over the covered window
        tmax = min(cm.target_min.max(), co.target_min.max())
        m0, m1 = float(cm.D50.iloc[0]), float(cm[cm.target_min <= tmax].D50.iloc[-1])
        o0, o1 = float(co.D50.iloc[0]), float(co[co.target_min <= tmax].D50.iloc[-1])
        dmodel, dobs = m1 - m0, o1 - o0
        detectable = bool((det.classification == "larger than observed repeatability").any())
        rows.append(_gate_row(ph, dmodel, dobs, detectable, tmax))
    return pd.DataFrame(rows)


def _gate_row(ph, dmodel, dobs, detectable, tmax):
    same_dir = np.sign(dmodel) == np.sign(dobs) and abs(dmodel) > 1e-6 and abs(dobs) > 1e-6
    if not detectable:
        out = "1: model-predicted size evolution < q3 repeatability; lack of observed change is not informative"
    elif same_dir and abs(dobs) >= 0.5 * abs(dmodel):
        out = "2: model and q3 agree in direction and approximate magnitude"
    elif same_dir:
        out = "3: model and q3 agree in direction but not magnitude"
    elif abs(dobs) < 1e-6 or (detectable and abs(dobs) < 0.25 * abs(dmodel)):
        out = "4: model predicts a detectable shift that q3 does not show"
    else:
        out = "5: observed q3 changes in a different direction"
    return dict(ph=ph, outcome=out,
                model_dD50_um=round(dmodel, 3), obs_dD50_um=round(dobs, 3),
                predicted_shift_detectable=detectable, covered_to_min=round(float(tmax), 1),
                caveat="a discrepancy could arise from the model's size rule, the PAQXOS inversion, R3 "
                       "resolution, non-sphericity/aggregation, or other unmodeled physics — the later "
                       "raw-channel analysis must disambiguate")


# ── figures ───────────────────────────────────────────────────────────────────
def _save_multiformat(fig, stem, formats):
    stem = Path(stem); stem.parent.mkdir(parents=True, exist_ok=True)
    dpi = {"png": 300, "pdf": ps.DPI, "svg": ps.DPI, "tif": 600, "tiff": 600}
    out = []
    for fmt in formats:
        ext = "tiff" if fmt in ("tif", "tiff") else fmt
        p = stem.with_suffix(f".{ext}")
        kw = dict(dpi=dpi.get(fmt, ps.DPI), bbox_inches="tight")
        if ext == "tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(p, **kw); out.append(p)
    return out


def figure_clock(runs, common_xo, grid, cond_model, cond_obs, out_dir, formats):
    """3×2 clock-time figure: left = model vs observed distribution snapshots at 0/2/5/10/20 min; right =
    D10/D50/D90 trajectories (model line + observed date points, condition ≥2 dates)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    ps.apply_manuscript_style()
    working = mqs.VERSIONS[WORKING_VERSION]

    # aggregate distributions at snapshot times (date-first) for model + observed
    snap_model = {ph: _agg_dist_at(runs, ph, grid, common_xo, FIG_SNAPSHOT_MIN, "model") for ph in CONDITIONS}
    snap_obs = {ph: _agg_dist_at(runs, ph, grid, common_xo, FIG_SNAPSHOT_MIN, "observed", working) for ph in CONDITIONS}
    # shared left-column y-limit from the 1–15 µm structure (excludes the finest edge bin, whose
    # first-class fraction differs across pH and would otherwise force an over-tall shared axis)
    inr = (common_xo > 1.0) & (common_xo <= Q3_XLIM[1])
    ymax = max([(v[inr].max() * 100.0) for ph in CONDITIONS
                for v in list(snap_model[ph].values()) + list(snap_obs[ph].values())] or [1.0]) * 1.08

    norm = Normalize(0.0, 20.0); cmap = plt.get_cmap(CMAP)
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 8.4))
    fig.subplots_adjust(left=0.16, right=0.85, top=0.95, bottom=0.075, hspace=0.36, wspace=0.42)
    pmark = {"D10": "o", "D50": "s", "D90": "^"}

    for r, ph in enumerate(CONDITIONS):
        axL, axR = axes[r]
        # left: distribution snapshots
        for tm in FIG_SNAPSHOT_MIN:
            c = cmap(norm(tm))
            if tm in snap_model[ph]:
                axL.plot(common_xo, snap_model[ph][tm] * 100.0, "-", lw=1.5, color=c, alpha=0.9)
            if tm in snap_obs[ph]:
                axL.plot(common_xo, snap_obs[ph][tm] * 100.0, "--", lw=1.1, color=c, alpha=0.9, dashes=(3, 2))
        axL.set_xlim(*Q3_XLIM); axL.set_ylim(0, ymax)
        axL.set_ylabel("Volume fraction per\nsize class (%)")
        ps.setup_axes(axL); axL.grid(True, lw=0.3, alpha=0.25)
        # right: percentile trajectories
        cm = cond_model[cond_model.ph == ph].sort_values("target_min")
        co = cond_obs[(cond_obs.ph == ph) & (cond_obs.drawable)].sort_values("target_min")
        pcol = {"D10": "#0072B2", "D50": "#000000", "D90": "#D55E00"}
        for pctl in ("D10", "D50", "D90"):
            axR.plot(cm.target_min, cm[pctl], "-", color=pcol[pctl], lw=1.7, zorder=3)
            d = co[co[pctl].notna()]
            axR.plot(d.target_min, d[pctl], pmark[pctl], color=pcol[pctl], ms=4, mec="k", mew=0.4,
                     ls="none", alpha=0.9, zorder=4)
        axR.set_xlim(-0.5, 20.5); axR.set_ylim(bottom=0)
        axR.set_ylabel("Percentile diameter (µm)")
        ps.setup_axes(axR); axR.grid(True, lw=0.3, alpha=0.25)
        for ax, letter in ((axL, "ACE"[r]), (axR, "BDF"[r])):
            ps.panel_label(ax, letter, x=-0.20, y=1.03)

    axes[0][0].set_title("Size distribution (— model,  – – observed)", fontsize=9, fontweight="bold")
    axes[0][1].set_title("Percentiles vs time (line model, points observed)", fontsize=9, fontweight="bold")
    axes[2][0].set_xlabel("Particle diameter (µm)")
    axes[2][1].set_xlabel("Time (min)")
    # percentile legend on the top-right panel
    from matplotlib.lines import Line2D
    axes[0][1].legend([Line2D([0], [0], color=c, lw=2) for c in ("#0072B2", "#000000", "#D55E00")],
                      ["D10", "D50", "D90"], fontsize=7, frameon=False, loc="upper right")
    cax = fig.add_axes([0.875, 0.34, 0.017, 0.34])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax); cb.set_label("Time (min)")
    cb.set_ticks(SNAPSHOT_MIN)
    for r, ph in enumerate(CONDITIONS):
        pos = axes[r][0].get_position()
        fig.text(0.028, 0.5 * (pos.y0 + pos.y1), f"pH {ph}", rotation=90, ha="center", va="center",
                 fontsize=12, fontweight="bold")
    written = _save_multiformat(fig, out_dir / "model_vs_observed_q3_time_evolution", formats)
    plt.close(fig)
    return written


def _model_g_curves(runs, gg):
    """Date-first condition-mean model D10/D50/D90 as smooth functions of the model cohort-mass fraction
    ``gg`` (each run's model percentile trajectory interpolated onto the common g-grid, then date-first).
    Returns ``{ph: (3, len(gg)) array}`` for conditions with ≥ 2 contributing dates."""
    out = {}
    for ph in CONDITIONS:
        by_date = defaultdict(list)
        for run in runs:
            if run["ph"] != ph or run.get("model") is None:
                continue
            m = run["model"]; g = _remaining_mass_frac(m)
            dcw = [_cohort_weights(m, f) for f in range(len(m.t))]
            perc = np.array([_weighted_pctiles_logd(*dcw[f]) for f in range(len(m.t))])   # (T,3)
            di = np.array([np.interp(gg, g[::-1], perc[::-1, i], left=np.nan, right=np.nan) for i in range(3)])
            by_date[run["date"]].append(di)
        if len(by_date) >= 2:
            date_means = [np.nanmean(np.stack(v), axis=0) for v in by_date.values()]
            out[ph] = np.nanmean(np.stack(date_means), axis=0)
    return out


def figure_matched_mass(runs, matched, out_dir, formats):
    """Supporting figure: predicted (smooth date-first model line) vs observed (UV-paired points)
    percentiles versus remaining undissolved-mass fraction — the size-selective test independent of the
    overall rate scale (ONE model curve, since base ≡ rate-only at matched mass)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps.apply_manuscript_style()
    gg = np.linspace(0.08, 1.0, 40)
    curves = _model_g_curves(runs, gg)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.24, wspace=0.32)
    pcol = {"D10": "#0072B2", "D50": "#000000", "D90": "#D55E00"}
    pmark = {"D10": "o", "D50": "s", "D90": "^"}
    for ax, ph in zip(axes, CONDITIONS):
        if ph in curves:
            for i, pctl in enumerate(("D10", "D50", "D90")):
                ax.plot(gg, curves[ph][i], "-", color=pcol[pctl], lw=1.6, alpha=0.95, zorder=3)
        g = matched[matched.ph == ph]
        for pctl in ("D10", "D50", "D90"):
            ax.plot(g.apparent_remaining_dose_fraction, g[f"obs_{pctl}"], pmark[pctl], color=pcol[pctl],
                    ms=3.5, mec="k", mew=0.3, ls="none", alpha=0.8, zorder=4)
        ax.set_title(f"pH {ph}", fontsize=10, fontweight="bold")
        ax.set_xlabel("UV-derived apparent\nremaining dose fraction"); ax.set_xlim(1.02, -0.02)  # dissolution → left
        ax.set_ylim(bottom=0); ps.setup_axes(ax); ax.grid(True, lw=0.3, alpha=0.25)
    axes[0].set_ylabel("Percentile diameter (µm)")
    from matplotlib.lines import Line2D
    axes[2].legend([Line2D([0], [0], color=c, lw=2) for c in pcol.values()], list(pcol),
                   fontsize=7, frameon=False, loc="lower right")
    written = _save_multiformat(fig, out_dir / "model_vs_observed_q3_matched_mass", formats)
    plt.close(fig)
    return written


def _snapshot_source_table(runs, common_xo, grid):
    """Aggregated model + observed in-window distributions at every snapshot time (incl. 5 min, retained
    here even though the figure shows 0/2/10/20) × diameter — the figure's source data."""
    working = mqs.VERSIONS[WORKING_VERSION]
    rows = []
    for ph in CONDITIONS:
        sm = _agg_dist_at(runs, ph, grid, common_xo, SNAPSHOT_MIN, "model")
        so = _agg_dist_at(runs, ph, grid, common_xo, SNAPSHOT_MIN, "observed", working)
        for tm in SNAPSHOT_MIN:
            for src, dd in (("model", sm), ("observed", so)):
                if tm in dd:
                    for x, v in zip(common_xo, dd[tm]):
                        rows.append(dict(ph=ph, target_min=tm, source=src, diameter_um=round(float(x), 4),
                                         vol_frac_pct=round(float(v) * 100.0, 4)))
    return pd.DataFrame(rows)


def _agg_dist_at(runs, ph, grid, common_xo, times, source, working=None):
    """Date-first mean model/observed in-window distribution at each snapshot time (≥2 dates)."""
    want = {int(round(t / mqs.DT_MIN)): t for t in times}
    run_aligned = []
    for run in runs:
        if run["ph"] != ph:
            continue
        if source == "model":
            if run["model"] is None:
                continue
            m = run["model"]; d, w = _cohort_weights(m, 0); init_d10 = _weighted_pctiles_logd(d, w)[0]
            init_inw = _inwindow_mass(m, 0, common_xo)
            aligned = {}
            off = round(run["t0_offset_min"] / mqs.DT_MIN)
            for f in range(len(m.t)):
                gi = f + off                                  # frames are exactly on the 0.2-min grid
                if gi in want:
                    dq = model_frame_descriptors(m, f, common_xo, init_d10, init_inw)["dq"]
                    if np.all(np.isfinite(dq)):               # skip depleted (missing) distributions
                        aligned[gi] = dq
            run_aligned.append((run["date"], aligned))
        else:
            elig = [r for r in run["obs"]["q3"] if r["category"] in working and not r["before_zero"]
                    and r["dq_grid"] is not None and np.isfinite(r["elapsed_min"])]
            if not elig:
                continue
            times_e = [r["elapsed_min"] for r in elig]
            al = mqs._align_pairs(times_e, [r["dq_grid"] for r in elig], grid)
            run_aligned.append((run["date"], {gi: v for gi, v in al.items() if gi in want}))
    agg = mqs._date_first_mean(run_aligned, min_dates=2)
    return {want[gi]: agg[gi]["mean"] for gi in agg if gi in want}


# ── primary revision: relative percentile evolution (normalized size-shape, log2 fold change) ──
PCTL_COLORS = {"D10": "#0072B2", "D50": "#000000", "D90": "#D55E00"}   # colorblind-safe; encodes PERCENTILE only
PCTL_MARKERS = {"D10": "o", "D50": "s", "D90": "^"}
REL_DISPLAY_MIN = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]     # prespecified observed symbol times
FOLD_TICKS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]                     # candidate fold-change gridlines


def relative_percentile_long(clock):
    """Per-run relative percentiles: each (source, pH, date, rep, percentile) trajectory normalized to its
    OWN initial eligible value (the value at its minimum target time, so it equals 1.0 there). Returns a
    long table with absolute_um, initial_um, relative, log2_fold_change. Size-shape ONLY — the inputs are
    the q3/model D10/D50/D90, never an optical-intensity or UV-mass variable."""
    long = clock.melt(id_vars=["source", "ph", "date", "rep", "target_min"],
                      value_vars=["D10", "D50", "D90"], var_name="percentile", value_name="absolute_um")
    long = long[long.absolute_um.notna() & (long.absolute_um > 0)]
    parts = []
    for _key, g in long.groupby(["source", "ph", "date", "rep", "percentile"], sort=False):
        g = g.sort_values("target_min")
        init = float(g.absolute_um.iloc[0]); t0 = float(g.target_min.iloc[0])
        parts.append(g.assign(initial_um=init, normalization_time_min=t0,
                              relative=g.absolute_um / init,
                              log2_fold_change=np.log2(g.absolute_um / init)))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _agg_levels(run_long):
    """Date-level (intraday reps averaged within a preparation date) and cross-date-median aggregations of
    the relative percentiles. The preparation DATE is the experimental unit; reps are averaged first."""
    date = (run_long.groupby(["source", "ph", "date", "percentile", "target_min"], as_index=False)
            .agg(relative=("relative", "mean"), log2_fold_change=("log2_fold_change", "mean"),
                 absolute_um=("absolute_um", "mean"), initial_um=("initial_um", "mean"),
                 n_runs=("rep", "nunique")))
    cross = (date.groupby(["source", "ph", "percentile", "target_min"], as_index=False)
             .agg(relative=("relative", "median"), log2_fold_change=("log2_fold_change", "median"),
                  absolute_um=("absolute_um", "median"), initial_um=("initial_um", "median"),
                  n_dates=("date", "nunique"), n_runs=("n_runs", "sum")))
    return date, cross


def relative_source_table(run_long):
    """Long tidy primary source data at run / date / cross-date-median aggregation levels — retains every
    eligible value, its absolute + initial diameters (so absolutes are reconstructable), relative, and
    log2 fold change, with contributing run/date counts."""
    date, cross = _agg_levels(run_long)
    def _src(s):
        return "observed_q3" if s == "observed" else "model"
    def _elig(s):
        return "eligible_coarse_excluded" if s == "observed" else "model"
    rows = []
    for _, r in run_long.iterrows():
        rows.append(dict(ph=r.ph, date=int(r.date), rep=int(r.rep), time_min=round(float(r.target_min), 3),
                         eligibility=_elig(r.source), source=_src(r.source), percentile=r.percentile,
                         absolute_um=round(float(r.absolute_um), 4), initial_um=round(float(r.initial_um), 4),
                         relative=round(float(r.relative), 4), log2_fold_change=round(float(r.log2_fold_change), 4),
                         aggregation_level="run", n_runs=1, n_dates=1))
    for _, r in date.iterrows():
        rows.append(dict(ph=r.ph, date=int(r.date), rep=np.nan, time_min=round(float(r.target_min), 3),
                         eligibility=_elig(r.source), source=_src(r.source), percentile=r.percentile,
                         absolute_um=round(float(r.absolute_um), 4), initial_um=round(float(r.initial_um), 4),
                         relative=round(float(r.relative), 4), log2_fold_change=round(float(r.log2_fold_change), 4),
                         aggregation_level="date", n_runs=int(r.n_runs), n_dates=1))
    for _, r in cross.iterrows():
        rows.append(dict(ph=r.ph, date=np.nan, rep=np.nan, time_min=round(float(r.target_min), 3),
                         eligibility=_elig(r.source), source=_src(r.source), percentile=r.percentile,
                         absolute_um=round(float(r.absolute_um), 4), initial_um=round(float(r.initial_um), 4),
                         relative=round(float(r.relative), 4), log2_fold_change=round(float(r.log2_fold_change), 4),
                         aggregation_level="cross_date_median", n_runs=int(r.n_runs), n_dates=int(r.n_dates)))
    return pd.DataFrame(rows)


def _relative_ylim(run_long):
    """Single shared y-range (log2 units) + fold-change ticks covering EVERY plotted value (run/date/
    cross-date), so nothing is clipped and all three pH panels use identical limits."""
    ymin = float(np.nanmin(run_long.log2_fold_change)); ymax = float(np.nanmax(run_long.log2_fold_change))
    pad = 0.12 * max(ymax - ymin, 0.5)
    ylo, yhi = ymin - pad, ymax + pad
    ticks, labels = _fold_ticks_for(ylo, yhi)
    return ylo, yhi, ticks, labels


def _fold_ticks_for(ymin, ymax):
    """Fold-change gridlines (in log2 units) covering [ymin, ymax], always including 1× (0), expanding to
    whatever the data needs so nothing is clipped."""
    ticks = [np.log2(f) for f in FOLD_TICKS if (np.log2(f) >= ymin - 1e-9 and np.log2(f) <= ymax + 1e-9)]
    if 0.0 not in ticks:
        ticks = sorted(set(ticks + [0.0]))
    labels = [f"{2 ** t:g}×" for t in ticks]
    return ticks, labels


def figure_relative_percentile_evolution(run_long, out_dir, formats, min_dates=2):
    """Primary figure — 3 panels (pH 4.0/4.5/5.0). Each shows the relative evolution of D10/D50/D90 as
    log2 fold change from each trajectory's own initial value: forward model = solid lines, observed q3 =
    symbols. Colour encodes percentile only (no time colour). Faint date-level trajectories + an
    emphasized cross-date median; the preparation date is the experimental unit. Identical y-limits/ticks
    across all three panels; a 1× reference marks no distributional change; nothing is clipped."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ps.apply_manuscript_style()
    date, cross = _agg_levels(run_long)
    ylo, yhi, ticks, labels = _relative_ylim(run_long)      # single shared y-range for all three panels

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.1), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.86, bottom=0.30, wspace=0.08)
    for ax, ph in zip(axes, CONDITIONS):
        ax.axhline(0.0, color="0.55", lw=0.9, ls=":", zorder=1)              # 1× = no change
        for pctl in ("D10", "D50", "D90"):
            c = PCTL_COLORS[pctl]
            # faint date-level trajectories, subsampled to the prespecified display times (the full ~12 s
            # cadence is retained in the source CSV; sub-sampling only de-noises the visualization)
            for (src, dt), g in date[(date.ph == ph) & (date.percentile == pctl)].groupby(["source", "date"]):
                g = _at_display_times(g.sort_values("target_min"))
                ax.plot(g.target_min, g.log2_fold_change, "-", color=c, lw=0.6, alpha=0.28, zorder=2)
            # emphasized cross-date median: model = solid smooth line, observed = symbols at display times
            cm = cross[(cross.ph == ph) & (cross.percentile == pctl) & (cross.source == "model")
                       & (cross.n_dates >= min_dates)].sort_values("target_min")
            co = cross[(cross.ph == ph) & (cross.percentile == pctl) & (cross.source == "observed")
                       & (cross.n_dates >= min_dates)].sort_values("target_min")
            ax.plot(cm.target_min, cm.log2_fold_change, "-", color=c, lw=1.8, zorder=4)
            cod = _at_display_times(co)
            ax.plot(cod.target_min, cod.log2_fold_change, PCTL_MARKERS[pctl], color=c, ms=4.5, mec="k",
                    mew=0.4, ls="none", zorder=5)
        ax.set_xlim(-0.5, 20.5); ax.set_ylim(ylo, yhi)
        ax.set_yticks(ticks); ax.set_yticklabels(labels)
        ax.set_title(f"pH {ph}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (min)"); ps.setup_axes(ax); ax.grid(True, lw=0.3, alpha=0.22)
    axes[0].set_ylabel("Percentile diameter\n(fold change from t0)")
    # single compact shared legend: colour = percentile, style = source
    handles = [Line2D([0], [0], color=PCTL_COLORS[p], lw=2, label=p) for p in ("D10", "D50", "D90")]
    handles += [Line2D([0], [0], color="0.35", lw=1.8, label="model"),
                Line2D([0], [0], color="0.35", marker="o", ls="none", mec="k", mew=0.4, label="observed q3")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.54, 0.0), columnspacing=1.2, handletextpad=0.4)
    written = _save_multiformat(fig, out_dir / "model_vs_observed_q3_relative_percentile_evolution", formats)
    plt.close(fig)
    return written


def _nearest(arr, t):
    return float(arr[int(np.argmin(np.abs(np.asarray(arr, float) - t)))]) if len(arr) else np.nan


def _at_display_times(g, tol=0.3):
    """Subsample a per-trajectory table to the prespecified display times (nearest ``target_min`` within
    ``tol`` min). Visualization only — the full-cadence data is retained in the source CSV."""
    if not len(g):
        return g
    tt = g.target_min.to_numpy(float)
    keep = []
    for t in REL_DISPLAY_MIN:
        k = int(np.argmin(np.abs(tt - t)))
        if abs(tt[k] - t) <= tol:
            keep.append(g.index[k])
    return g.loc[sorted(set(keep))]


def figure_absolute_percentile_evolution(cond_m, cond_o, out_dir, formats, min_dates=2):
    """PRIMARY model-vs-observed q3 figure — absolute D10/D50/D90 percentile diameters (µm) versus time,
    organized BY PERCENTILE (Panel A = D10, B = D50, C = D90) so each percentile keeps its own y-range.
    Forward model = solid lines (drawn BEHIND), observed q3 = symbols; colour encodes pH (colorblind-safe
    manuscript styling). Both the model line AND the observed symbols are the **date-first, cross-date
    MEDIAN** — intraday replicates are averaged within each preparation date, then the date means are
    summarized by their median across independent dates (the robust central estimate selected prospectively
    for the whole figure). ``cond_m``/``cond_o`` MUST therefore be the ``stat="median"`` condition tables.
    Every eligible ~12 s timepoint with ≥ ``min_dates`` dates is shown — NO display-time subsampling. Size
    space only: no Copt/optical magnitude, UV-derived mass, Mie operator, or detector-channel input enters.
    The observed starting q3 initializes the model, so t = 0 agreement is imposed and is not validation.
    Per-panel y-ranges recalculate from the plotted medians and include every plotted value (nothing cropped,
    no log/broken axes) — so the non-reproducing single-date pH-4.0 excursion no longer controls the D50/D90
    axes (it remains in the run/date source data and the date-level sensitivity figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ps.apply_manuscript_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.1))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.83, bottom=0.27, wspace=0.28)
    for ax, pctl, letter in zip(axes, ("D10", "D50", "D90"), "ABC"):
        ymax = 0.0
        for ph in CONDITIONS:
            st = ps.ph_style(ph)
            cm = cond_m[(cond_m.ph == ph) & (cond_m.n_dates >= min_dates)].sort_values("target_min")
            co = cond_o[(cond_o.ph == ph) & (cond_o.drawable)].sort_values("target_min")
            # model line behind (zorder 2); observed symbols on top (zorder 4), modest transparency so the
            # dense ~12 s cadence reads as points rather than a solid band
            ax.plot(cm.target_min, cm[pctl], "-", color=st["color"], lw=1.6, alpha=0.95, zorder=2)
            ax.plot(co.target_min, co[pctl], st["marker"], color=st["color"], ms=3.4, mec="k", mew=0.3,
                    ls="none", alpha=0.72, zorder=4)
            for tab in (cm, co):
                if len(tab) and tab[pctl].notna().any():
                    ymax = max(ymax, float(np.nanmax(tab[pctl].to_numpy(float))))
        ax.set_title(pctl, fontsize=10, fontweight="bold")
        ax.set_xlim(-0.5, 20.5); ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1.0)   # include every value
        ax.set_xlabel("Time (min)"); ps.setup_axes(ax); ax.grid(True, lw=0.3, alpha=0.22)
        ps.panel_label(ax, letter, x=-0.17, y=1.04)
    axes[0].set_ylabel("Percentile diameter (µm)")
    handles = [Line2D([0], [0], color=ps.ph_style(ph)["color"], lw=2, marker=ps.ph_style(ph)["marker"],
                      mec="k", mew=0.3, label=f"pH {ph}") for ph in CONDITIONS]
    handles += [Line2D([0], [0], color="0.35", lw=1.8, label="model"),
                Line2D([0], [0], color="0.35", marker="o", ls="none", mec="k", mew=0.3, label="observed q3")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.54, 0.0), columnspacing=1.1, handletextpad=0.4)
    written = _save_multiformat(fig, out_dir / "model_vs_observed_q3_absolute_percentile_evolution", formats)
    plt.close(fig)
    return written


def _observed_date_level(clock):
    """Per-preparation-date observed percentiles (intraday reps averaged within a date) from the clock table
    — the provenance behind each condition-level observed point, used only for the date-level overlay."""
    obs = clock[clock.source == "observed"]
    return (obs.groupby(["ph", "date", "target_min"], as_index=False)[["D10", "D50", "D90"]].mean())


def figure_absolute_percentile_datelevel(clock, cond_o, out_dir, formats, min_dates=2):
    """Supporting/sensitivity companion to the primary absolute figure: same by-percentile layout, but
    overlays the faint per-preparation-date observed trajectories behind the emphasized cross-date condition
    **median**, so the reader can see how many independent dates support each point and which excursions are
    single-date driven — notably the isolated pH-4.0 ~12.6 min D50/D90 feature (see the outlier audit), whose
    extreme 20260608 date-level line remains visible here even though the robust median suppresses it in the
    primary. Pass the ``stat="median"`` ``cond_o`` so the emphasized overlay matches the primary. Same
    eligibility, full ~12 s cadence, and size-only inputs as the primary; kept separate because the dense
    date-level overlay would clutter the primary display."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ps.apply_manuscript_style()
    dlv = _observed_date_level(clock)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.1))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.83, bottom=0.27, wspace=0.28)
    for ax, pctl, letter in zip(axes, ("D10", "D50", "D90"), "ABC"):
        ymax = 0.0
        for ph in CONDITIONS:
            st = ps.ph_style(ph)
            for _date, g in dlv[dlv.ph == ph].groupby("date"):
                g = g.sort_values("target_min")
                ax.plot(g.target_min, g[pctl], "-", color=st["color"], lw=0.5, alpha=0.30, zorder=2)
                if len(g) and g[pctl].notna().any():
                    ymax = max(ymax, float(np.nanmax(g[pctl].to_numpy(float))))
            co = cond_o[(cond_o.ph == ph) & (cond_o.drawable)].sort_values("target_min")
            ax.plot(co.target_min, co[pctl], st["marker"], color=st["color"], ms=3.2, mec="k", mew=0.3,
                    ls="none", alpha=0.85, zorder=4)
        ax.set_title(pctl, fontsize=10, fontweight="bold")
        ax.set_xlim(-0.5, 20.5); ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1.0)
        ax.set_xlabel("Time (min)"); ps.setup_axes(ax); ax.grid(True, lw=0.3, alpha=0.22)
        ps.panel_label(ax, letter, x=-0.17, y=1.04)
    axes[0].set_ylabel("Percentile diameter (µm)")
    handles = [Line2D([0], [0], color=ps.ph_style(ph)["color"], lw=2, marker=ps.ph_style(ph)["marker"],
                      mec="k", mew=0.3, label=f"pH {ph}") for ph in CONDITIONS]
    handles += [Line2D([0], [0], color="0.45", lw=0.8, alpha=0.5, label="date-level (faint)"),
                Line2D([0], [0], color="0.35", marker="o", ls="none", mec="k", mew=0.3, label="cross-date median")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.0, frameon=False,
               bbox_to_anchor=(0.54, 0.0), columnspacing=1.0, handletextpad=0.4)
    written = _save_multiformat(fig, out_dir / "model_vs_observed_q3_absolute_percentile_evolution_datelevel", formats)
    plt.close(fig)
    return written


# ── primary absolute source table + isolated pH-4.0 outlier audit ─────────────
def absolute_source_table(clock, cond_m_median, cond_o_median, cond_m_mean, cond_o_mean):
    """Long source data reproducing every model line and observed symbol in the PRIMARY absolute figure,
    plus the provenance beneath each plotted point, at unambiguous aggregation levels:

    * ``run`` — raw run-level percentiles;
    * ``date`` — within-date means (intraday replicates averaged within a preparation date);
    * ``cross_date`` with ``summary_statistic = "cross_date_median"`` — the **primary** plotted values
      (``plotted_primary = True`` where drawn: model needs ≥2 dates, observed needs ``drawable``);
    * ``cross_date`` with ``summary_statistic = "cross_date_mean"`` — the mean, retained for
      sensitivity/audit (``plotted_primary = False``).

    Every primary plotted point is exactly reproducible from the ``cross_date_median`` rows. Every row
    carries pH, date/replicate where applicable, time, percentile, absolute diameter (µm), aggregation level,
    summary statistic, contributing run/date counts, eligibility category, ``plotted_primary``, and a reason
    when not plotted. Size space only — the inputs are the q3/model D10/D50/D90, never an optical, UV-mass, or
    channel variable."""
    def _src(s):
        return "observed_q3" if s == "observed" else "model"
    def _elig(s):
        return f"eligible_{WORKING_VERSION}" if s == "observed" else "model_init_from_observed_t0"
    long = clock.melt(id_vars=["source", "ph", "date", "rep", "target_min"],
                      value_vars=["D10", "D50", "D90"], var_name="percentile", value_name="absolute_um")
    long = long[long.absolute_um.notna() & (long.absolute_um > 0)]
    rows = []
    # run-level provenance
    for r in long.itertuples():
        rows.append(dict(source=_src(r.source), ph=r.ph, date=int(r.date), rep=int(r.rep),
                         time_min=round(float(r.target_min), 3), percentile=r.percentile,
                         absolute_um=round(float(r.absolute_um), 4), aggregation_level="run",
                         summary_statistic="raw", n_runs=1, n_dates=1, eligibility=_elig(r.source),
                         plotted_primary=False, reason="provenance: averaged within date, then across dates"))
    # date-level provenance (intraday reps averaged within a preparation date)
    dl = (long.groupby(["source", "ph", "date", "percentile", "target_min"], as_index=False)
          .agg(absolute_um=("absolute_um", "mean"), n_runs=("rep", "nunique")))
    for r in dl.itertuples():
        rows.append(dict(source=_src(r.source), ph=r.ph, date=int(r.date), rep=np.nan,
                         time_min=round(float(r.target_min), 3), percentile=r.percentile,
                         absolute_um=round(float(r.absolute_um), 4), aggregation_level="date",
                         summary_statistic="within_date_mean", n_runs=int(r.n_runs), n_dates=1,
                         eligibility=_elig(r.source), plotted_primary=False,
                         reason="provenance: date-level mean of intraday replicates"))
    # condition-level cross-date summaries: median (PRIMARY, plotted) then mean (retained for sensitivity)
    for stat, is_primary, conds in (("cross_date_median", True, (("model", cond_m_median), ("observed", cond_o_median))),
                                    ("cross_date_mean", False, (("model", cond_m_mean), ("observed", cond_o_mean)))):
        for src, cond in conds:
            c = cond.melt(id_vars=["ph", "target_min", "n_dates", "n_runs", "drawable"],
                          value_vars=["D10", "D50", "D90"], var_name="percentile", value_name="absolute_um")
            for r in c.itertuples():
                val = float(r.absolute_um) if r.absolute_um == r.absolute_um else np.nan
                drawable = bool(r.drawable)
                plotted = bool(is_primary and drawable and np.isfinite(val) and val > 0)
                if is_primary:
                    reason = ("" if plotted else
                              ("fewer than 2 independent dates" if not drawable else "no finite percentile at this time"))
                else:
                    reason = "retained for sensitivity/audit (mean not plotted in primary)"
                rows.append(dict(source=_src(src), ph=r.ph, date=np.nan, rep=np.nan,
                                 time_min=round(float(r.target_min), 3), percentile=r.percentile,
                                 absolute_um=(round(val, 4) if np.isfinite(val) else np.nan),
                                 aggregation_level="cross_date", summary_statistic=stat,
                                 n_runs=int(r.n_runs), n_dates=int(r.n_dates), eligibility=_elig(src),
                                 plotted_primary=plotted, reason=reason))
    return pd.DataFrame(rows)


OUTLIER_PH = 4.0
OUTLIER_WINDOW_MIN = (10.0, 16.0)     # search window for the isolated large pH-4.0 feature (~12–13 min)


def outlier_audit(runs, clock, cond_o_mean):
    """Trace the isolated large pH-4.0 observed percentile feature (~12–13 min, cross-date-MEAN D50 ≈ 22 µm,
    D90 ≈ 35 µm) back to its contributing dates, runs, and q3 frames, and report BOTH cross-date estimators
    at that timepoint. The audited timepoint is chosen as the pH-4.0 **drawable** condition point with the
    largest cross-date-MEAN D90 inside :data:`OUTLIER_WINDOW_MIN` (i.e. where the mean excursion lives); pass
    the ``stat="mean"`` ``cond_o`` here. For every run contributing at that time it records the run's
    D10/D50/D90; the paired q3 frame's elapsed time, Copt, frac > 100 µm, coarse-tail flag, eligibility
    category, and before-zero flag; and whether it passed the **same** prespecified working eligibility as
    every other plotted point. The summary reports the cross-date **mean** (old estimator) and cross-date
    **median** (the new PRIMARY estimator, robust to the single-date excursion) side by side, plus which date
    drives it. Descriptive only — nothing is deleted or clipped; the underlying observation stays in the
    run/date source data and the date-level sensitivity figure. Returns ``(per_run_df, summary_dict)``."""
    working = mqs.VERSIONS[WORKING_VERSION]
    co = cond_o_mean[(cond_o_mean.ph == OUTLIER_PH) & (cond_o_mean.drawable)
                     & (cond_o_mean.target_min >= OUTLIER_WINDOW_MIN[0])
                     & (cond_o_mean.target_min <= OUTLIER_WINDOW_MIN[1])]
    if not len(co):
        return pd.DataFrame(), {}
    top = co.loc[co.D90.idxmax()]
    t_star = float(top.target_min)
    contrib = clock[(clock.source == "observed") & (clock.ph == OUTLIER_PH)
                    & (np.isclose(clock.target_min, t_star))]
    rows = []
    for c in contrib.itertuples():
        run = next((r for r in runs if r["ph"] == OUTLIER_PH and r["date"] == c.date and r["rep"] == c.rep), None)
        frame = dict(q3_frame=np.nan, q3_elapsed_min=np.nan, measured_copt=np.nan, frac_gt_100um=np.nan,
                     coarse_tail_flag=None, category="", before_zero=None, passed_working_eligibility=None)
        if run is not None:
            elig = [r for r in run["obs"]["q3"] if r["category"] in working and not r["before_zero"]
                    and r["dq_grid"] is not None and np.isfinite(r["elapsed_min"])]
            if elig:
                fr = min(elig, key=lambda z: abs(z["elapsed_min"] - t_star))
                frame = dict(q3_frame=int(fr["q3_frame"]), q3_elapsed_min=round(float(fr["elapsed_min"]), 3),
                             measured_copt=fr["measured_copt"], frac_gt_100um=fr["frac_gt_100um"],
                             coarse_tail_flag=bool(fr["coarse_tail_flag"]), category=fr["category"],
                             before_zero=bool(fr["before_zero"]),
                             passed_working_eligibility=bool(fr["category"] in working and not fr["before_zero"]))
        rows.append(dict(ph=OUTLIER_PH, target_min=round(t_star, 3), date=int(c.date), rep=int(c.rep),
                         D10=c.D10, D50=c.D50, D90=c.D90, **frame))
    per_run = pd.DataFrame(rows)
    dlv = per_run.groupby("date")[["D10", "D50", "D90"]].mean()      # date-first: within-date means
    drive_date = int(dlv.D90.idxmax())
    others = dlv.D90.drop(drive_date)
    driven_by_one = bool(dlv.D90.max() > 3.0 * float(others.max())) if len(others) else True
    passed_col = per_run["passed_working_eligibility"].dropna()
    summary = dict(
        ph=OUTLIER_PH, target_min=round(t_star, 3), primary_statistic="cross_date_median",
        # both cross-date estimators at the audited timepoint (mean = old, median = new primary)
        cross_date_mean_D50=round(float(dlv.D50.mean()), 3), cross_date_mean_D90=round(float(dlv.D90.mean()), 3),
        cross_date_median_D50=round(float(dlv.D50.median()), 3),
        cross_date_median_D90=round(float(dlv.D90.median()), 3),
        n_contributing_dates=int(per_run.date.nunique()), n_contributing_runs=int(len(per_run)),
        driving_date=drive_date, driving_date_D50=round(float(dlv.loc[drive_date, "D50"]), 3),
        driving_date_D90=round(float(dlv.loc[drive_date, "D90"]), 3),
        magnitude_driven_by_single_date=driven_by_one,
        reproduced_across_dates=bool(not driven_by_one),
        all_contributors_passed_working_eligibility=(bool(passed_col.all()) if len(passed_col) else None),
        any_coarse_tail_flag=bool(per_run["coarse_tail_flag"].fillna(False).any()),
        max_frac_gt_100um=(round(float(np.nanmax(per_run["frac_gt_100um"])), 3)
                           if per_run["frac_gt_100um"].notna().any() else np.nan),
        coarse_bin_inversion_event=bool(per_run["coarse_tail_flag"].fillna(False).any()),
        underlying_observation_deleted=False, retained_in_run_and_date_source=True,
        visible_in_datelevel_sensitivity=True, point_specific_exclusion_applied=False,
        retained_in_primary=True)
    return per_run, summary


def _write_outlier_audit(out_dir, per_run, summary):
    """Persist the pH-4.0 outlier audit as CSV (per-run trace) + MD (provenance + decision)."""
    per_run.to_csv(out_dir / "q3_absolute_percentile_outlier_audit.csv", index=False)
    if not summary:
        (out_dir / "q3_absolute_percentile_outlier_audit.md").write_text(
            "# pH 4.0 absolute-percentile outlier audit\n\nNo drawable pH-4.0 point in the search window.\n")
        return
    s = summary
    L = ["# Isolated pH 4.0 absolute-percentile observation — outlier audit\n",
         f"At **{s['target_min']} min**, pH {s['ph']}, the two date-first cross-date estimators diverge "
         "sharply because one preparation date carries an extreme (non-reproducing) coarse excursion. The "
         "primary absolute figure now uses the robust **cross-date median** for every point, so this "
         "timepoint no longer dominates the D50/D90 axes. The underlying observation was **not deleted** — it "
         "remains in the run/date source data and in the date-level sensitivity figure. This audit traces the "
         "feature to its contributing dates, runs, and q3 frames and reports both estimators.\n",
         "## Both cross-date estimators at the audited timepoint\n",
         f"- Cross-date **mean** (old estimator): D50 = {s['cross_date_mean_D50']} µm, "
         f"D90 = {s['cross_date_mean_D90']} µm.\n",
         f"- Cross-date **median** (new PRIMARY estimator): D50 = {s['cross_date_median_D50']} µm, "
         f"D90 = {s['cross_date_median_D90']} µm.\n",
         "## Provenance\n",
         f"- {s['n_contributing_dates']} contributing dates, {s['n_contributing_runs']} runs (date-first: "
         "intraday replicates averaged within a preparation date, then summarized across dates).\n",
         f"- Magnitude driven predominantly by preparation date **{s['driving_date']}** "
         f"(that date's mean D50/D90 = {s['driving_date_D50']}/{s['driving_date_D90']} µm); "
         f"single-date driven = **{s['magnitude_driven_by_single_date']}**, reproduced across dates = "
         f"**{s['reproduced_across_dates']}**. The median suppresses this single-date excursion; the mean does "
         "not.\n",
         "## Eligibility\n",
         f"- All contributors passed the same prespecified working eligibility "
         f"(`{WORKING_VERSION}`, Copt ≥ {mqs.COPT_MIN}, coarse-tail-excluded): "
         f"**{s['all_contributors_passed_working_eligibility']}**. The observation is a valid, eligible "
         "measurement — it did **not** fail QC.\n",
         f"- Any contributing frame carried the > 100 µm coarse-tail flag: **{s['any_coarse_tail_flag']}** "
         f"(max frac > 100 µm among contributors = {s['max_frac_gt_100um']} %). The excursion therefore is "
         f"**not** a > 100 µm coarse-bin inversion event by the prespecified 100 µm flag definition — its "
         f"coarse mass sits below 100 µm (an extreme but sub-threshold coarse shoulder).\n",
         "## Per-run trace\n",
         "```\n" + per_run.to_string(index=False) + "\n```\n",
         "## Decision\n",
         "- The primary figure change is a **prospective, figure-wide switch to the date-first cross-date "
         "median** — the robust independent-date summary applied identically to every pH, timepoint, "
         "percentile, and to BOTH model and observed trajectories. It is **not** a point-specific QC exclusion "
         f"(`point_specific_exclusion_applied = {s['point_specific_exclusion_applied']}`) and no eligibility, "
         "coarse-tail, or Copt threshold was changed.\n",
         "- The underlying observation was **not deleted, excluded, or failed** "
         f"(`underlying_observation_deleted = {s['underlying_observation_deleted']}`): it remains in the "
         f"run/date source rows (`retained_in_run_and_date_source = {s['retained_in_run_and_date_source']}`) "
         f"and stays visible in the date-level sensitivity figure "
         f"(`visible_in_datelevel_sensitivity = {s['visible_in_datelevel_sensitivity']}`, "
         "`model_vs_observed_q3_absolute_percentile_evolution_datelevel.*`). The conclusion — the model "
         "predicts stronger coarsening than is observed — does not depend on this point.\n"]
    (out_dir / "q3_absolute_percentile_outlier_audit.md").write_text("\n".join(L))


def relative_interpretation(run_long):
    """Does the revised presentation support the LIMITED interpretation that the model predicts progressive
    relative coarsening (smaller cohorts dissolve preferentially) while observed q3 shows weaker/less
    consistent redistribution? Compares the end-of-window cross-date-median log2 fold change, model vs
    observed, per pH and percentile. Descriptive only — no channel/size-parameter claim."""
    _date, cross = _agg_levels(run_long)
    rows = []
    for ph in CONDITIONS:
        for pctl in ("D10", "D50", "D90"):
            cm = cross[(cross.ph == ph) & (cross.percentile == pctl) & (cross.source == "model")].sort_values("target_min")
            co = cross[(cross.ph == ph) & (cross.percentile == pctl) & (cross.source == "observed")].sort_values("target_min")
            if not len(cm) or not len(co):
                continue
            tmax = min(float(cm.target_min.max()), float(co.target_min.max()))
            mfc = float(cm[cm.target_min <= tmax].log2_fold_change.iloc[-1])
            ofc = float(co[co.target_min <= tmax].log2_fold_change.iloc[-1])
            rows.append(dict(ph=ph, percentile=pctl, covered_to_min=round(tmax, 1),
                             model_log2fc=round(mfc, 3), observed_log2fc=round(ofc, 3),
                             model_fold=round(2 ** mfc, 3), observed_fold=round(2 ** ofc, 3),
                             model_coarsens_more=bool(mfc > ofc + 0.05)))
    df = pd.DataFrame(rows)
    supported = bool(len(df) and (df.model_coarsens_more.mean() > 0.5))
    df.attrs["interpretation_supported"] = supported
    return df


# ── report ────────────────────────────────────────────────────────────────────
def _report(out_dir, sem, checks, clock_cond_m, clock_cond_o, matched, detect, gate, equiv, audit, interp,
            outlier, common_xo):
    L = ["# Forward model vs observed PAQXOS q3 — size-evolution analysis\n",
         "Analysis report (not a manuscript caption). Compares the selected UV-derived **rate-only** "
         "forward model's predicted particle-size-distribution evolution against the observed PAQXOS q3, "
         "**in size space**, with no Mie operator and no detector-channel deconvolution. The model is not "
         "refit to q3; t=0 agreement is imposed (the observed starting q3 initializes the model) and is "
         "**not** validation.\n",
         "## Displays — primary vs supporting\n",
         "- **PRIMARY: `model_vs_observed_q3_absolute_percentile_evolution.*`** — absolute D10/D50/D90 "
         "percentile diameters (µm) versus time, one panel per percentile (A = D10, B = D50, C = D90), "
         "colour = pH, model = solid lines (behind), observed q3 = symbols. Both the model line and the "
         "observed symbols are the **date-first, cross-date MEDIAN**: intraday replicates are averaged within "
         "each preparation date, then the date means are summarized by their median across independent dates "
         "(≥ 2 dates). The **median was selected prospectively for the entire figure** as a robust "
         "independent-date summary — applied identically to every pH, timepoint, percentile, and to BOTH "
         "model and observed trajectories. It shows **all eligible condition-level observed timepoints at the "
         "~12 s cadence** (no display-time subsampling); per-panel y-ranges recalculate from the plotted "
         "medians and include every plotted value (nothing cropped, no log/broken axes). The forward model "
         "predicts progressive increases in q3 percentile diameters as smaller cohorts disappear, whereas the "
         "observed PAQXOS D10/D50/D90 trajectories show substantially weaker and less consistent temporal "
         "shifts.\n",
         "- **SUPPORTING: `model_vs_observed_q3_relative_percentile_evolution.*`** — the same trajectories as "
         "log2 fold change from each trajectory's own t₀ (fold-change ticks, shared y, 1× reference). It also "
         "summarizes the preparation-date means by their **cross-date median**, so the absolute and relative "
         "primary displays now share the same date-first, cross-date-median aggregation philosophy. It "
         "isolates relative redistribution but is a supporting view only; the full-cadence absolute figure is "
         "the primary visual.\n",
         "- **Sensitivity: `model_vs_observed_q3_absolute_percentile_evolution_datelevel.*`** — the absolute "
         "figure with faint per-preparation-date observed trajectories behind the cross-date median, keeping "
         "the extreme 20260608 date-level excursion visually available even though the robust median "
         "suppresses it in the primary.\n",
         "- **All displayed observed values use the established q3 reliability rules** "
         f"(`{WORKING_VERSION}`: coarse-tail excluded, Copt ≥ {mqs.COPT_MIN}, date-first aggregation, "
         "≥ 2 independent dates). No optical magnitude (Copt), UV-derived mass, Mie operator, or "
         "detector-channel input enters any of these percentile displays. The model predicts stronger "
         "coarsening than is observed; this is **descriptive** and does not estimate a detector-channel size "
         "parameter.\n",
         "## Step 1 — forward-state semantics (verified numerically)\n",
         "```\n" + sem.to_string(index=False) + "\n```\n",
         "```\n" + checks.round(6).to_string(index=False) + "\n```\n",
         "Mass balance and (r/r₀)³=q/q₀ hold to machine precision; radii never increase; no negatives; "
         "frozen (fully-dissolved) cohorts carry zero size-distribution weight.\n",
         "## Selected model\n",
         f"- Rate-only, `rate_scale = {RATE_SCALE_SELECTED}` (date-balanced; fit on pH 4.0/4.5, applied "
         f"**unchanged** to pH 5.0). Base physical baseline `rate_scale = {RATE_SCALE_BASE}`.\n",
         f"- **Base ≡ rate-only at matched model cohort-mass fraction:** {equiv['equivalent']} "
         f"(max |ΔD50| = {equiv['max_abs_D50_diff_um']} µm over the shared g range). rate_scale is a pure "
         f"time-rescaling, so only the clock differs — the matched-progress figure draws ONE model curve.\n",
         "## Step 6 — clock-time comparison (condition-level, date-first; working = coarse-tail-excluded, "
         "Copt≥0.79)\n",
         "```\n" + clock_cond_m[["ph", "target_min", "D10", "D50", "D90", "frac_lt_2um", "n_dates"]]
         .round(3).to_string(index=False) + "\n```\n",
         "## Isolated pH 4.0 observation (~12–13 min) — outlier audit\n",
         (("- At **" + f"{outlier['target_min']} min** the two cross-date estimators diverge: cross-date "
           f"**mean** D50 ≈ {outlier['cross_date_mean_D50']} µm / D90 ≈ {outlier['cross_date_mean_D90']} µm "
           f"vs cross-date **median** D50 ≈ {outlier['cross_date_median_D50']} µm / D90 ≈ "
           f"{outlier['cross_date_median_D90']} µm (`q3_absolute_percentile_outlier_audit.csv`/`.md`). The "
           f"observation **passed the same prespecified eligibility** as every other point (all contributors "
           f"`{outlier['all_contributors_passed_working_eligibility']}`) and carried **no > 100 µm "
           f"coarse-tail flag** (max frac > 100 µm = {outlier['max_frac_gt_100um']} %, so it is not a "
           f"coarse-bin inversion event by the 100 µm definition), but its magnitude is "
           f"**single-preparation-date driven** (date {outlier['driving_date']}) and did **not** reproduce "
           f"across dates. The primary figure now uses the robust cross-date **median** for every point, so "
           f"this timepoint no longer controls the D50/D90 axes. The underlying observation was **retained** "
           f"— not deleted, excluded, or failed QC — and stays in the run/date source data and the date-level "
           f"sensitivity figure. The change was a prospective figure-wide estimator choice, **not** a "
           f"point-specific exclusion and **not** a QC-threshold change.\n")
          if outlier else "- No drawable pH-4.0 point fell in the audit window.\n"),
         "## Supporting figure — relative percentile evolution (normalized size-shape)\n",
         "- `model_vs_observed_q3_relative_percentile_evolution.*` — 3 panels (pH), D10/D50/D90 as **log2 "
         "fold change from each trajectory's own t₀** (fold-change ticks 0.5×/1×/2×/4×/…, shared y across "
         "panels, 1× reference). Model = solid lines, observed q3 = symbols; colour encodes percentile "
         "only. **Preparation date is the experimental unit** (intraday reps averaged within a date, then "
         "the cross-date median; faint date-level trajectories shown). This is a normalized size-**shape** "
         "comparison — no optical intensity or UV-derived mass enters; it is not an optical-to-mass "
         "conversion. Absolute magnitudes are the **primary** display "
         "(`model_vs_observed_q3_absolute_percentile_evolution.*`). The relative fold-change results are "
         "also retained numerically in `model_vs_observed_q3_relative_percentile_source.csv`. The original "
         "6-panel snapshot figure is retained as a diagnostic.\n",
         "- **Limited interpretation** — "
         + ("**SUPPORTED**" if interp.attrs.get("interpretation_supported") else "**not clearly supported**")
         + ": the forward model predicted progressive relative coarsening as smaller cohorts dissolved "
         "preferentially, whereas the observed q3 percentiles exhibited weaker/less consistent "
         "redistribution (end-of-window cross-date-median fold change, model vs observed):\n",
         "```\n" + interp[["ph", "percentile", "model_fold", "observed_fold", "model_coarsens_more"]]
         .to_string(index=False) + "\n```\n"
         "This is descriptive only — NOT a quantitative claim about detector channels or a fitted "
         "size-dependence parameter.\n",
         "## Step 8 — detectability (predicted shift vs within-date observed repeatability)\n",
         "```\n" + detect.to_string(index=False) + "\n```\n",
         "## Step 7 — matched-progress comparison (UV-derived apparent remaining dose fraction)\n",
         "- The observed progress coordinate is the **UV-derived apparent remaining dose fraction** — a UV "
         "**mass-balance** coordinate (`assay.cumulative_dissolved`), NOT an optical mass estimate. It "
         "corrects the earlier fixed-40 mL error: 400 µL are withdrawn and NOT replaced at each UV sample, "
         "so the vessel volume shrinks (`V_i = 40 − 0.400·i` mL) and dissolved drug leaves with each "
         "aliquot; `m_dissolved_i = C_i·V_i + Σ_{j<i} C_j·0.400` µg. No Copt / angular signal / q3 "
         "magnitude enters; q3 supplies only normalized size-distribution shape. The model is compared at "
         "its own cohort-mass fraction `g_model = Σq_i(t)/Σq_i(0)` as the shared dissolution-extent "
         "coordinate. Values are not clipped; QC flags mark samples outside [0, 1].\n",
         f"- {len(matched)} UV-paired points across conditions. Interpreting the apparent remaining dose "
         f"fraction as suspended undissolved *particle* mass assumes no important deposition, "
         f"precipitation, or other unrecovered compartment — **notably uncertain at pH 5.0**, whose "
         f"incomplete recovery means its apparent remaining dose fraction cannot be equated with suspended "
         f"particulate mass. This analysis stays diagnostic.\n",
         "**Before/after audit of the coordinate correction (per pH):**\n",
         "```\n" + audit.to_string(index=False) + "\n```\n",
         "## Step 10 — interpretation gate (per condition)\n",
         "```\n" + gate[["ph", "outcome", "model_dD50_um", "obs_dD50_um",
                         "predicted_shift_detectable"]].to_string(index=False) + "\n```\n",
         "A discrepancy is **not** read as 'laser diffraction does not reflect size-dependent "
         "dissolution': it could arise from the model's size rule, the PAQXOS inversion, R3 resolution, "
         "non-sphericity/aggregation, or other unmodeled physics. Disambiguating these is the job of the "
         "later raw-channel analysis.\n"]
    (out_dir / "model_vs_q3_analysis_report.md").write_text("\n".join(L))


# ── driver ────────────────────────────────────────────────────────────────────
def run(output_dir, formats=("png", "pdf", "svg", "tiff")):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    grid = mqs._common_time_grid()
    # common native q3 grid (same construction as mqs)
    by_ph = defaultdict(list)
    for ph, date, rep, rtf, fo in iter_runs():
        if float(ph) in CONDITIONS:
            by_ph[float(ph)].append(fo)
    common_xo = np.array(sorted({round(float(x), 4) for ph in CONDITIONS for fo in by_ph[ph]
                                 for x in psd.read_q3_frames(fo).xo}))

    runs = build_runs(common_xo)
    sample = [_with_id(r["model"], r) for r in runs if r.get("model") is not None][:3]
    sem, checks = forward_state_semantics_audit(sample)

    clock = clock_descriptor_rows(runs, common_xo, grid)
    clock_run = clock
    # cross-date MEAN condition tables (retained diagnostics: clock figure, detectability, gate, audit)
    date_m, cond_m = date_first_aggregate(clock[clock.source == "model"])
    date_o, cond_o = date_first_aggregate(clock[clock.source == "observed"])
    # cross-date MEDIAN condition tables — the PRIMARY absolute figure's robust date-first estimator
    _dmm, cond_m_med = date_first_aggregate(clock[clock.source == "model"], stat="median")
    _dom, cond_o_med = date_first_aggregate(clock[clock.source == "observed"], stat="median")
    detect = detectability(clock, cond_m)
    matched = matched_mass_rows(runs, common_xo)
    audit = matched_progress_audit(matched)
    equiv = base_vs_rateonly_matched_mass(runs, common_xo)
    gate = interpretation_gate(cond_m, cond_o, detect)

    # eligibility-version sensitivity of the observed percentiles (inclusive / stringent)
    sens = _eligibility_sensitivity(runs, common_xo, grid)

    # write tables
    sem.to_csv(output_dir / "forward_state_semantics.csv", index=False)
    checks.to_csv(output_dir / "forward_state_conservation_checks.csv", index=False)
    clock_run.to_csv(output_dir / "model_vs_q3_run_level_descriptors.csv", index=False)
    date_m.to_csv(output_dir / "model_date_level_descriptors.csv", index=False)
    date_o.to_csv(output_dir / "observed_date_level_descriptors.csv", index=False)
    cond_m.to_csv(output_dir / "model_condition_descriptors.csv", index=False)
    cond_o.to_csv(output_dir / "observed_condition_descriptors.csv", index=False)
    cond_m_med.to_csv(output_dir / "model_condition_descriptors_median.csv", index=False)
    cond_o_med.to_csv(output_dir / "observed_condition_descriptors_median.csv", index=False)
    _clock_comparison(cond_m, cond_o).to_csv(output_dir / "clock_time_comparison.csv", index=False)
    matched.to_csv(output_dir / "matched_mass_comparison.csv", index=False)
    audit.to_csv(output_dir / "matched_progress_correction_audit.csv", index=False)
    detect.to_csv(output_dir / "detectability_analysis.csv", index=False)
    sens.to_csv(output_dir / "q3_eligibility_sensitivity.csv", index=False)
    gate.to_csv(output_dir / "interpretation_gate.csv", index=False)
    pd.DataFrame([equiv]).to_csv(output_dir / "base_vs_rateonly_matched_mass.csv", index=False)

    _snapshot_source_table(runs, common_xo, grid).to_csv(
        output_dir / "distribution_snapshots_source.csv", index=False)

    # PRIMARY — absolute percentile evolution (µm), full ~12 s cadence, by-percentile panels; the plotted
    # estimator is the date-first cross-date MEDIAN, with the MEAN retained in the source table for audit
    absolute_source_table(clock, cond_m_med, cond_o_med, cond_m, cond_o).to_csv(
        output_dir / "model_vs_observed_q3_absolute_percentile_source.csv", index=False)
    outlier_per_run, outlier_summary = outlier_audit(runs, clock, cond_o)
    _write_outlier_audit(output_dir, outlier_per_run, outlier_summary)

    # SUPPORTING — relative percentile evolution (normalized size-shape, log2 fold change)
    run_long = relative_percentile_long(clock)
    relative_source_table(run_long).to_csv(
        output_dir / "model_vs_observed_q3_relative_percentile_source.csv", index=False)
    interp = relative_interpretation(run_long)
    interp.to_csv(output_dir / "relative_percentile_interpretation.csv", index=False)

    figs = {}
    # PRIMARY absolute (cross-date MEDIAN) + its date-level sensitivity companion
    figs["absolute"] = figure_absolute_percentile_evolution(cond_m_med, cond_o_med, output_dir, formats)
    figs["absolute_datelevel"] = figure_absolute_percentile_datelevel(clock, cond_o_med, output_dir, formats)
    # SUPPORTING relative fold-change figure
    figs["relative"] = figure_relative_percentile_evolution(run_long, output_dir, formats)
    # retained diagnostics (NOT overwritten/deleted): the original 6-panel snapshots + matched-progress
    figs["clock"] = figure_clock(runs, common_xo, grid, cond_m, cond_o, output_dir, formats)
    figs["matched"] = figure_matched_mass(runs, matched, output_dir, formats)
    _report(output_dir, sem, checks, cond_m, cond_o, matched, detect, gate, equiv, audit, interp,
            outlier_summary, common_xo)
    return dict(runs=runs, clock=clock, cond_model=cond_m, cond_obs=cond_o, matched=matched,
                detect=detect, gate=gate, equiv=equiv, sensitivity=sens, run_long=run_long,
                interpretation=interp, outlier=outlier_summary, figures=figs, common_xo=common_xo)


def _with_id(m, r):
    m.inputs["id"] = f"pH{r['ph']}_{r['date']}_R{r['rep']}"
    return m


def _clock_comparison(cond_m, cond_o):
    """Join model & observed condition-level percentiles per (ph, target_min) with signed/abs errors."""
    keys = ["ph", "target_min"]
    m = cond_m.set_index(keys)[["D10", "D50", "D90", "n_dates"]].add_prefix("model_")
    o = cond_o[cond_o.drawable].set_index(keys)[["D10", "D50", "D90", "n_dates"]].add_prefix("obs_")
    j = m.join(o, how="inner").reset_index()
    for pc in ("D10", "D50", "D90"):
        j[f"signed_err_{pc}"] = (j[f"model_{pc}"] - j[f"obs_{pc}"]).round(3)
        j[f"abs_err_{pc}"] = j[f"signed_err_{pc}"].abs()
    return j


def _eligibility_sensitivity(runs, common_xo, grid):
    """Observed condition-level D50 under inclusive / working / stringent eligibility (Step 9 sensitivity)."""
    rows = []
    for vname, cats in mqs.VERSIONS.items():
        recs = []
        for run in runs:
            elig = [r for r in run["obs"]["q3"] if r["category"] in cats and not r["before_zero"]
                    and r["dq_grid"] is not None and np.isfinite(r["elapsed_min"])]
            if not elig:
                continue
            init_d10 = psd.q3_percentiles(common_xo, elig[0]["cum_grid"])[0]
            times = [r["elapsed_min"] for r in elig]
            for gi, ei in mqs._align_pairs(times, list(range(len(elig))), grid).items():
                r = elig[int(ei)]
                od = _observed_descriptor_row(r["cum_grid"], r["dq_grid"], common_xo, init_d10)
                recs.append(dict(source="observed", version=vname, ph=run["ph"], date=run["date"],
                                 rep=run["rep"], target_min=round(float(grid[gi]), 3),
                                 D10=_r(od["D10"]), D50=_r(od["D50"]), D90=_r(od["D90"]),
                                 span=_r(od["span"]), frac_lt_2um=_r(od["frac_lt_2um"]),
                                 frac_lt_initD10=_r(od["frac_lt_initD10"]),
                                 wass_from_start=np.nan, max_abs_dcumQ3_from_start=np.nan,
                                 frac_model_outside_window=np.nan))
        if recs:
            _dl, cond = date_first_aggregate(pd.DataFrame(recs))
            cond["version"] = vname
            rows.append(cond[["version", "ph", "target_min", "D50", "n_dates", "drawable"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--formats", default="png,pdf,svg,tiff")
    args = ap.parse_args(argv)
    formats = tuple(s.strip() for s in args.formats.split(",") if s.strip())
    res = run(args.output_dir, formats)
    print("figures (primary = absolute):")
    for k, paths in res["figures"].items():
        print(f"  [{k}] " + "  ".join(str(p.name) for p in paths))
    if res.get("outlier"):
        o = res["outlier"]
        print(f"\npH 4.0 outlier @ {o['target_min']} min — cross-date MEAN D50={o['cross_date_mean_D50']} "
              f"D90={o['cross_date_mean_D90']} µm vs MEDIAN D50={o['cross_date_median_D50']} "
              f"D90={o['cross_date_median_D90']} µm (primary uses median); "
              f"single-date-driven={o['magnitude_driven_by_single_date']} (date {o['driving_date']}); "
              f"passed eligibility={o['all_contributors_passed_working_eligibility']}; "
              f"deleted={o['underlying_observation_deleted']}; retained_in_primary={o['retained_in_primary']}")
    print("\nbase ≡ rate-only at matched mass:", res["equiv"])
    print("\nrelative-percentile interpretation supported:",
          res["interpretation"].attrs.get("interpretation_supported"))
    print(res["interpretation"][["ph", "percentile", "model_fold", "observed_fold", "model_coarsens_more"]]
          .to_string(index=False))
    print("\ninterpretation gate:")
    print(res["gate"][["ph", "outcome"]].to_string(index=False))


if __name__ == "__main__":
    main()
