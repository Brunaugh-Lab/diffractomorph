"""Normalized-q3 shape at matched observed extent — Track 1 of the q3 audit/recovery plan.

These are **diagnostic** outputs, not a committed result: the q3 export is a per-frame *renormalized*
distribution (each frame sums to 100 %), so it carries composition, not amount. This module does **no**
de-normalization (no ×Copt, no ×UV — the source of the old aggregation mess) and asks only what q3 can
answer on its own:

    **At the same observed total scattering loss, has the size distribution *shape* moved, and does
    that movement differ by pH?**

The progress coordinate is built **identically** to the pre-inversion channel analysis
(:mod:`psd_redistribution_matched_g`): background-subtracted detector intensity, obstruction-filtered
``Copt ≤ 40`` and synchronized-glitch despiked, remaining fraction ``g(t) = ΣI(t)/ΣI(anchor)`` from 1
downward. q3 frames are matched to their detector frame **by timestamp** (the ``Time`` cell vs the
``.rtf`` frame clock, ``MATCH_TOL_S`` tolerance), so a one-frame slip in a fast pH-4 run cannot
mis-pair the shape with the loss coordinate; unmatched frames are reported. Because q3 and the detector
are the same acquisition, this tests what the PAQXOS inversion makes of the small detector-level
redistribution — it is not independent validation of it.

Per run, at each ``g`` target we read percentiles **directly off the cumulative ``Q₃(xo)``** (not
reconstructed bin fractions), and report ``D10/D50/D90``, ``span``, signed coarsening ``Δlog10 D50``,
and total-variation shape distance ``TV = ½Σ|Δq3|`` from the anchor. The ill-constrained coarse
inversion tail above ``VALID_SIZE_MAX_UM`` is handled explicitly: we report the q3 fraction above that
size, flag tail instability, **invalidate D90/span when D90 falls in the unsupported tail**, and also
report a prespecified ``≤ size_max`` restricted-range view.

Every (run, target) that yields no metric records **why**, one of:
``not_reached`` (g never descends that far — the shallow-loss case, e.g. arrested pH 5),
``below_noise`` (remaining signal < ``N_SIGMA`` × total-signal noise), ``q3_unreliable`` (bracketing
frames below the absolute Copt reliability floor), ``q3_malformed`` (nonfinite or improperly normalized
q3 frame), ``coarse_review`` (excluded only in a coarse-tail-omitting profile), ``unmatched``
(bracketing detector frame has no q3 within
tolerance), ``few_frames``, ``invalid_anchor``. The whole pipeline is re-run across a sweep of the Copt
reliability floor (audit step 7).

**q3 is secondary, normalized shape evidence** (PAQXOS-inverted), never de-normalized into mass. The
underlying q3 descriptors (D10/50/90, span, Δlog10 D50, TV) are the reportable quantities. Aggregation
is **date-first**: nested runs → (date × condition) unit → condition summary. The cross-condition
contrast is **descriptive only**: it blocks within date, keeps only date blocks carrying ≥2 conditions,
and references η² against the **exact enumeration** of the distinct within-date condition-label
arrangements (:mod:`study_design`) — not a Monte-Carlo p. With four date blocks the restricted group is
tiny, so ``exact_perm_p_provisional`` is flagged ``diagnostic_only`` and is NOT manuscript inference.

Writes into ``<study>/psd_evolution/matched_extent/`` (diagnostic CSVs + figures):
  - ``q3_matched_extent_by_run.csv``       — per run × target × floor, WITH ``rep``/``run_id``: reason,
                                             g, t, D10/50/90, span, restricted D10/50/90, Δlog10 D50,
                                             TV, tail % > size_max, tail_unstable, match diagnostics,
  - ``q3_matched_extent_reasons.csv``      — per condition × target × floor: count of runs by reason,
  - ``q3_matched_extent_by_condition.csv`` — per condition × target × floor: metric mean/sd/n, plus the
                                             per-condition contributing ``date_blocks`` (date unit),
  - ``q3_matched_extent_summary.csv``      — per common-support target × floor: condition η² +
                                             ``exact_perm_p_provisional`` (``diagnostic_only``),
                                             ``n_date_condition_units``, ``n_unique_date_blocks``,
                                             ``n_distinct_perms``, ``min_attainable_p`` for Δlog10 D50 & TV,
  - ``q3_matched_extent_primary_*.csv``    — the main-text Copt ≥ 0.79 profile with coarse-review
                                             frames omitted, kept separate from the inclusive SI table,
  - ``matched_extent_coverage.png`` / ``matched_extent_percentiles.png`` /
    ``matched_extent_distance.png``        — default-floor diagnostic figures.

Run with the pipeline venv:  ``python q3_matched_extent.py``
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffractomorph_pipeline import ingest, psd
from diffractomorph_pipeline.noise_filter import despike_frames
from diffractomorph_pipeline.optics.mie import VALID_SIZE_MAX_UM
from diffractomorph_pipeline.processing import MatchedExtentConfig, matched_q3_extent

from psd_evolution_common import BASE, iter_runs
from study_design import distinct_restricted_permutations, restricted_label_arrangements

SUB = Path("psd_evolution") / "matched_extent"
TARGETS = (0.8, 0.6, 0.4, 0.2)        # observed remaining-signal fractions to match runs at
N_SIGMA = 5.0                          # reject a target whose absolute remaining < N_SIGMA × total noise
MIN_DAYS = 2                           # a condition needs this many valid date blocks at a target to be compared
COPT_MAX = 40.0                        # obstruction filter — IDENTICAL to psd_redistribution_matched_g
MATCH_TOL_S = 6.0                      # one-half of the nominal 12-s acquisition interval
FLOOR_SWEEP = (0.79, 4.0)               # absolute Copt: inclusive/provisional and supported summaries
DEFAULT_FLOOR = 0.79                    # inclusive floor used for figures (legacy public constant name)
TAIL_FRAC_MAX = 5.0                    # q3 %-volume above size_max above which the coarse tail is unstable
SIZE_MAX = VALID_SIZE_MAX_UM           # 15 µm — inversion trustworthy ceiling
ANCHOR_N = 3                           # frames averaged for the start (anchor) shape and signal

METRICS = ("dlog10_D50", "tv_distance")   # scalars carried into the cross-condition permutation DIAGNOSTIC
REASONS = ("ok", "not_reached", "below_noise", "q3_unreliable", "unmatched",
           "q3_malformed", "coarse_review", "few_frames", "invalid_anchor")


# ── detector g coordinate (identical preprocessing to the channel matched-g module) ─
def detector_series(rtf):
    """Background-subtracted detector loss coordinate, built exactly as ``psd_redistribution_matched_g.
    _run_matrix`` (``Copt ≤ 40`` + :func:`despike_frames`), plus the absolute frame times the timestamp
    match needs. Returns ``dict(epoch, total, g_env, s0, sigma, copt, n)`` over the retained frames, or
    ``None`` if too few frames."""
    raw = ingest.extract_run(rtf)
    copt = np.asarray(raw.copt, float)
    keep = np.isfinite(copt) & (copt <= COPT_MAX)
    if keep.sum() < 5:
        return None
    I, _t, cp, info = despike_frames(np.asarray(raw.I, float)[keep],
                                     np.asarray(raw.t_min, float)[keep],
                                     copt[keep])
    lead = int(info.get("n_lead_dropped", 0))
    tmin_kept = np.asarray(raw.t_min, float)[keep][lead:]     # despike drops only leading frames
    epoch = np.array([(raw.t0 + timedelta(minutes=float(m))).timestamp() for m in tmin_kept])
    ref = np.asarray(raw.ref, float)
    ref = np.nanmedian(ref, axis=0) if ref.ndim == 2 else ref
    X = np.clip(I - ref[None, :], 0.0, None)
    anchor = np.nanmedian(X[:ANCHOR_N], axis=0)
    s0 = float(anchor.sum())
    total = X.sum(axis=1)
    g_env = np.minimum.accumulate(total / s0) if s0 > 0 else np.full(len(total), np.nan)
    d = np.diff(total)
    sigma = 1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d))) / np.sqrt(2) if len(d) else 0.0
    return dict(epoch=epoch, total=total, g_env=g_env, s0=s0, sigma=float(sigma),
                copt=np.asarray(cp, float), n=len(total))


# ── per-run matched-extent evaluation, with an explicit reason per target ───────────
def matched_extent_run(frames: psd.Q3Frames, det, *, copt_floor_absolute,
                       exclude_coarse_review=None):
    """For one run at one reliability floor: ``{target: record}`` with a ``reason`` for every target.

    ``det`` is :func:`detector_series`. q3 frames are timestamp-matched to detector frames; the Copt
    floor marks q3-unreliable frames. The g coordinate is the detector's; the interpolated **cumulative
    ``Q₃``** at each target supplies the shape metrics."""
    config = MatchedExtentConfig(
        targets=TARGETS,
        match_tolerance_s=MATCH_TOL_S,
        anchor_frames=ANCHOR_N,
        noise_multiplier=N_SIGMA,
        acquisition_floor_absolute=copt_floor_absolute,
        size_max_um=SIZE_MAX,
        tail_fraction_max_pct=TAIL_FRAC_MAX,
        coarse_review_size_um=100.0,
        coarse_review_fraction_pct=1.0,
        exclude_coarse_review=(copt_floor_absolute >= 4.0 if exclude_coarse_review is None
                               else bool(exclude_coarse_review)),
    )
    return matched_q3_extent(
        frames,
        detector_epoch_s=det["epoch"],
        detector_remaining_fraction=det["g_env"],
        detector_start_signal=det["s0"],
        detector_noise_sigma=det["sigma"],
        acquisition_variable=det["copt"],
        config=config,
    )


# ── gather (read each run once; evaluate every floor) ───────────────────────────────
def _gather():
    """``[(cond, date, rep, run_id, frames, det)]`` for every pH-study run with a detector series."""
    runs = []
    for ph, date, rep, rtf, fo in iter_runs():
        det = detector_series(rtf)
        if det is None:
            continue
        try:
            frames = psd.read_q3_frames(fo)
        except (FileNotFoundError, KeyError) as e:
            print(f"  pH{ph}_{date}_R{rep}: q3 read failed ({e}) — skip")
            continue
        runs.append((f"pH {ph}", int(date), int(rep), f"pH{ph}_{date}_R{rep}", frames, det))
    return runs


def _evaluate(runs, floor, *, exclude_coarse_review=None, eligibility_profile=None):
    """``[dict(condition,date,rep,run_id,target_g, <record>)]`` — one row per run × target at ``floor``."""
    rows = []
    for cond, date, rep, run_id, frames, det in runs:
        res = matched_extent_run(
            frames, det, copt_floor_absolute=floor,
            exclude_coarse_review=exclude_coarse_review,
        )
        for target, rec in res.items():
            rows.append(dict(copt_floor_absolute=floor, floor_frac=floor,
                             eligibility_profile=eligibility_profile or
                             ("stringent" if floor >= 4.0 else "inclusive"),
                             condition=cond, date=date, rep=rep, run_id=run_id,
                             target_g=target, **rec))
    return rows


# ── aggregation + cross-condition test (shared-date within-day permutation) ──────────
def _day_values(ok_rows, metric):
    """Collapse same-day reps to one per-day mean of ``metric`` → array over days."""
    by_day = defaultdict(list)
    for r in ok_rows:
        v = r.get(metric, np.nan)
        if v is not None and np.isfinite(v):
            by_day[r["date"]].append(v)
    return np.array([np.mean(v) for v in by_day.values() if v], float)


def _eta2(y, cond):
    y = np.asarray(y, float)
    cond = np.asarray(cond)
    mu = y.mean()
    ss = float(np.sum((y - mu) ** 2))
    if ss <= 0:
        return np.nan
    between = sum(np.sum(cond == c) * (y[cond == c].mean() - mu) ** 2 for c in np.unique(cond))
    return float(between / ss)


def _condition_test(ok_rows, metric):
    """DESCRIPTIVE-ONLY day-level condition contrast on ``metric`` (no promoted p-value).

    Collapses nested runs to one value per (condition × date) unit, restricts to date blocks carrying
    ≥2 conditions, and computes the effect size η². The permutation reference is the **exact enumeration
    of the distinct within-date condition-label arrangements** (via :mod:`study_design`), not a
    Monte-Carlo draw — with only four date blocks the restricted group is tiny, so ``exact_perm_p`` is a
    diagnostic, not manuscript inference. Returns a dict or ``None``.

    Fields: ``eta2``; ``exact_perm_p_provisional`` (+ ``diagnostic_only=True``); ``n_date_condition_units``
    (NOT independent days); ``n_unique_date_blocks``; ``n_conds``; ``n_distinct_perms`` and
    ``min_attainable_p = 1/N``."""
    day = defaultdict(list)                                   # (cond, date) -> [metric over nested runs]
    for r in ok_rows:
        v = r.get(metric, np.nan)
        if v is not None and np.isfinite(v):
            day[(r["condition"], r["date"])].append(v)
    conds_by_date = defaultdict(set)
    for (c, d) in day:
        conds_by_date[d].add(c)
    shared = {d for d, cs in conds_by_date.items() if len(cs) >= 2}
    ys, cs, ds = [], [], []
    for (c, d), vals in day.items():
        if d in shared:
            ys.append(float(np.mean(vals))); cs.append(c); ds.append(d)   # date × condition unit
    if len(ys) < 3 or len(set(cs)) < 2:
        return None
    y, cond, date = np.asarray(ys, float), np.asarray(cs), np.asarray(ds)
    obs = _eta2(y, cond)
    if not np.isfinite(obs):
        return None
    arrangements = restricted_label_arrangements(list(date), list(cond))   # EXACT enumeration (shared helper)
    n_distinct = len(arrangements)
    assert n_distinct == distinct_restricted_permutations(list(date), list(cond))
    null = np.array([_eta2(y, np.asarray(a, dtype=object)) for a in arrangements], float)
    exact_p = float(np.mean(null[np.isfinite(null)] >= obs))
    return dict(eta2=float(obs), exact_perm_p_provisional=exact_p, diagnostic_only=True,
                n_date_condition_units=len(y), n_unique_date_blocks=len(shared), n_conds=len(set(cs)),
                n_distinct_perms=int(n_distinct),
                min_attainable_p=round(1.0 / n_distinct, 4) if n_distinct else np.nan)


def _common_targets(rows, conds):
    """Targets where every condition has ≥ MIN_DAYS valid (reason=='ok') days."""
    ok = [r for r in rows if r["reason"] == "ok"]
    common = []
    for t in TARGETS:
        per = {c: {r["date"] for r in ok if r["target_g"] == t and r["condition"] == c} for c in conds}
        if all(len(per[c]) >= MIN_DAYS for c in conds):
            common.append(t)
    return common


# ── figures (default floor) ─────────────────────────────────────────────────────────
def _colors(conds):
    return {c: col for c, col in zip(conds, plt.cm.viridis(np.linspace(0.12, 0.85, len(conds))))}


def _ok_by(rows, target, cond):
    return [r for r in rows if r["reason"] == "ok" and r["target_g"] == target and r["condition"] == cond]


def _plot_coverage(rows, conds, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = _colors(conds)
    w = 0.8 / max(len(conds), 1)
    x = np.arange(len(TARGETS))
    for k, c in enumerate(conds):
        counts = [len({r["date"] for r in _ok_by(rows, t, c)}) for t in TARGETS]
        ax.bar(x + (k - (len(conds) - 1) / 2) * w, counts, width=w, color=colors[c], label=str(c))
    ax.set_xticks(x)
    ax.set_xticklabels([f"g={t:g}" for t in TARGETS])
    ax.set_xlabel("matched remaining-signal fraction  g  (deeper loss →)")
    ax.set_ylabel("valid days (reason = ok)")
    ax.set_title(title)
    ax.legend(title="condition", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def _plot_percentiles(rows, conds, out_path, title):
    colors = _colors(conds)
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in conds:
        gs, d50m, d10m, d90m = [], [], [], []
        for t in TARGETS:
            ok = _ok_by(rows, t, c)
            d50 = _day_values(ok, "D50")
            if d50.size < 1:
                continue
            gs.append(t); d50m.append(np.median(d50))
            d10m.append(np.median(_day_values(ok, "D10")))
            d90v = _day_values(ok, "D90")                    # NaN-invalidated in the tail → may be empty
            d90m.append(np.median(d90v) if d90v.size else np.nan)
        if not gs:
            continue
        gs = np.asarray(gs, float)
        ax.plot(gs, d50m, "-o", color=colors[c], label=str(c))
        band = np.isfinite(d90m)
        ax.fill_between(gs[band], np.array(d10m)[band], np.array(d90m)[band], color=colors[c], alpha=0.14)
    ax.invert_xaxis()
    ax.set_xlabel("matched remaining-signal fraction  g")
    ax.set_ylabel("diameter (µm) — D50 line, D10–D90 band (band dropped where tail invalidates D90)")
    ax.set_title(title)
    ax.legend(title="condition", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def _plot_distance(rows, conds, out_path, title):
    colors = _colors(conds)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.6))
    for c in conds:
        gs, tvm, dl = [], [], []
        for t in TARGETS:
            ok = _ok_by(rows, t, c)
            tv = _day_values(ok, "tv_distance")
            if tv.size < 1:
                continue
            gs.append(t); tvm.append(np.median(tv)); dl.append(np.median(_day_values(ok, "dlog10_D50")))
        if not gs:
            continue
        a0.plot(gs, tvm, "-o", color=colors[c], label=str(c))
        a1.plot(gs, dl, "-o", color=colors[c], label=str(c))
    for ax in (a0, a1):
        ax.invert_xaxis(); ax.set_xlabel("matched remaining-signal fraction  g"); ax.grid(alpha=0.3)
    a0.set_ylabel("shape distance from start  TV = ½Σ|Δq3|")
    a0.set_title("Distribution distance from anchor")
    a1.axhline(0, color="0.4", lw=0.7)
    a1.set_ylabel("Δlog10 D50  (+ coarsens, − fines)")
    a1.set_title("Median-diameter shift from anchor"); a1.legend(title="condition", fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out_path, dpi=140); plt.close(fig)


# ── csv writers ─────────────────────────────────────────────────────────────────────
def _r(x, nd=4):
    return round(float(x), nd) if (x is not None and np.isfinite(x)) else np.nan


_RUN_COLS = ("D10", "D50", "D90", "span", "D10_restricted", "D50_restricted", "D90_restricted",
             "span_restricted", "dlog10_D50", "tv_distance", "tail_pct_gt_size_max", "coarse_tail_pct")


def _rows_by_run(rows):
    out = []
    for r in rows:
        row = dict(copt_floor_absolute=r["copt_floor_absolute"], floor_frac=r["floor_frac"],
                   eligibility_profile=r.get("eligibility_profile"),
                   condition=r["condition"], date=r["date"], rep=r["rep"],
                   run_id=r["run_id"], target_g=r["target_g"], reason=r["reason"],
                   g=_r(r.get("g")), t_min=_r(r.get("t_min"), 3),
                   n_matched=r.get("n_matched"), n_unmatched_q3=r.get("n_unmatched_q3"),
                   n_unmatched_det=r.get("n_unmatched_detector"),
                   tail_unstable=r.get("tail_unstable"), coarse_review=r.get("coarse_review"))
        for k in _RUN_COLS:
            row[k] = _r(r.get(k), 5)
        out.append(row)
    return out


def _rows_reasons(rows, conds):
    out = []
    for t in TARGETS:
        for c in conds:
            sub = [r for r in rows if r["target_g"] == t and r["condition"] == c]
            counts = {rn: sum(1 for r in sub if r["reason"] == rn) for rn in REASONS}
            out.append(dict(floor_frac=rows[0]["floor_frac"] if rows else np.nan,
                            eligibility_profile=(rows[0].get("eligibility_profile") if rows else None),
                            target_g=t, condition=c, n_runs=len(sub), **counts))
    return out


def _rows_by_condition(rows, conds):
    out = []
    for t in TARGETS:
        for c in conds:
            ok = _ok_by(rows, t, c)
            if not ok:
                continue
            dates = sorted({r["date"] for r in ok})
            row = dict(floor_frac=ok[0]["floor_frac"],
                       eligibility_profile=ok[0].get("eligibility_profile"),
                       target_g=t, condition=c,
                       n_date_blocks=len(dates), date_blocks="|".join(str(d) for d in dates),
                       n_runs=len(ok))
            for m in ("D50", "span", "D50_restricted", "dlog10_D50", "tv_distance"):
                vals = _day_values(ok, m)
                row[f"{m}_mean"] = _r(np.mean(vals), 5) if vals.size else np.nan
                row[f"{m}_sd"] = _r(np.std(vals, ddof=0), 5) if vals.size > 1 else np.nan
            out.append(row)
    return out


def _rows_summary(rows, conds, common):
    out = []
    for t in common:
        ok = [r for r in rows if r["reason"] == "ok" and r["target_g"] == t]
        row = dict(floor_frac=rows[0]["floor_frac"] if rows else np.nan,
                   eligibility_profile=(rows[0].get("eligibility_profile") if rows else None), target_g=t,
                   inference_status="diagnostic_only")           # q3 is secondary shape evidence, not inference
        for m in METRICS:
            res = _condition_test(ok, m)
            row[f"{m}_eta2"] = _r(res["eta2"]) if res else np.nan
            row[f"{m}_exact_perm_p_provisional"] = _r(res["exact_perm_p_provisional"]) if res else np.nan
            row[f"{m}_diagnostic_only"] = True
            row[f"{m}_n_date_condition_units"] = res["n_date_condition_units"] if res else 0
            row[f"{m}_n_unique_date_blocks"] = res["n_unique_date_blocks"] if res else 0
            row[f"{m}_n_distinct_perms"] = res["n_distinct_perms"] if res else 0
            row[f"{m}_min_attainable_p"] = _r(res["min_attainable_p"]) if res else np.nan
            row[f"{m}_n_conds"] = res["n_conds"] if res else 0
        out.append(row)
    return out


def main():
    out_dir = BASE / SUB
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _gather()
    if not runs:
        print("no runs with a detector series + q3 export"); return
    conds = sorted({r[0] for r in runs})

    by_run, by_cond, reasons_all, summary_all = [], [], [], []
    for floor in FLOOR_SWEEP:
        rows = _evaluate(runs, floor)
        common = _common_targets(rows, conds)
        by_run += _rows_by_run(rows)
        by_cond += _rows_by_condition(rows, conds)
        reasons_all += _rows_reasons(rows, conds)
        summary_all += _rows_summary(rows, conds, common)

        print(f"[floor={floor}] conds={conds}  common-support targets={common}")
        for t in TARGETS:                                    # reason breakdown makes the pH5 dropout explicit
            brk = {c: {rn: sum(1 for r in rows if r['target_g'] == t and r['condition'] == c
                               and r['reason'] == rn) for rn in ('ok', 'not_reached', 'q3_unreliable',
                                                                  'below_noise', 'unmatched')}
                   for c in conds}
            print(f"    g={t}: " + " | ".join(f"{c}:{brk[c]['ok']}ok/"
                  f"{brk[c]['not_reached']}nr/{brk[c]['q3_unreliable']}unrel" for c in conds))
        for r in _rows_summary(rows, conds, common):
            print(f"      summary g={r['target_g']} [DIAGNOSTIC ONLY]: "
                  f"Δlog D50 η²={r['dlog10_D50_eta2']} exact_p*={r['dlog10_D50_exact_perm_p_provisional']} | "
                  f"TV η²={r['tv_distance_eta2']} exact_p*={r['tv_distance_exact_perm_p_provisional']} "
                  f"(date×cond units={r['dlog10_D50_n_date_condition_units']}, "
                  f"date blocks={r['dlog10_D50_n_unique_date_blocks']}, "
                  f"N_perm={r['dlog10_D50_n_distinct_perms']}, min_p={r['dlog10_D50_min_attainable_p']})")

        if floor == DEFAULT_FLOOR:
            _plot_coverage(rows, conds, out_dir / "matched_extent_coverage.png",
                           f"matched-extent coverage by condition (floor={floor})")
            _plot_percentiles(rows, conds, out_dir / "matched_extent_percentiles.png",
                              f"q3 percentiles at matched extent — cumulative-Q3 basis (floor={floor})")
            _plot_distance(rows, conds, out_dir / "matched_extent_distance.png",
                           f"q3 shape distance at matched extent (floor={floor})")

    pd.DataFrame(by_run).to_csv(out_dir / "q3_matched_extent_by_run.csv", index=False)
    pd.DataFrame(reasons_all).to_csv(out_dir / "q3_matched_extent_reasons.csv", index=False)
    pd.DataFrame(by_cond).to_csv(out_dir / "q3_matched_extent_by_condition.csv", index=False)
    pd.DataFrame(summary_all).to_csv(out_dir / "q3_matched_extent_summary.csv", index=False)

    # Main-text primary sensitivity: retain supported + provisional Copt frames but omit
    # coarse-tail review frames. Kept in separate artifacts so the inclusive SI table remains
    # backward-compatible and cannot be accidentally double-counted by floor-only consumers.
    primary = _evaluate(
        runs, DEFAULT_FLOOR, exclude_coarse_review=True,
        eligibility_profile="primary_coarse_excluded",
    )
    primary_common = _common_targets(primary, conds)
    pd.DataFrame(_rows_by_run(primary)).to_csv(
        out_dir / "q3_matched_extent_primary_by_run.csv", index=False,
    )
    pd.DataFrame(_rows_reasons(primary, conds)).to_csv(
        out_dir / "q3_matched_extent_primary_reasons.csv", index=False,
    )
    pd.DataFrame(_rows_by_condition(primary, conds)).to_csv(
        out_dir / "q3_matched_extent_primary_by_condition.csv", index=False,
    )
    pd.DataFrame(_rows_summary(primary, conds, primary_common)).to_csv(
        out_dir / "q3_matched_extent_primary_summary.csv", index=False,
    )
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
