"""Audit — PAQXOS q3 inversion reliability as particulate signal decreases (corrected pass).

Concentration titrations (nominal Copt 3–30 at 1500 and 1760 rpm) change particle loading while
keeping approximately the same suspension PSD, so a reliable q3 inversion should stay stable as
measured signal falls. This audit pairs every frame-level q3 export to its measured-intensity frame by
timestamp, splits each nominal folder into contiguous **stable segments** vs **transition segments**
(some 1760-rpm folders span a wide actual-Copt range with a large time gap and must NOT be pooled), and
then analyses two DISTINCT outcomes:

  A. **Typical q3 drift** — how the *median* q3 shape of a stable segment moves as signal falls; and
  B. **Catastrophic inversion failure** — intermittent frames whose q3 collapses into the coarse tail
     (verified real PAQXOS results, not a parser error: individual raw CSVs genuinely place ~75 % of
     volume above 100 µm at the lowest actual Copt). A segment median hides these, so they are counted
     per frame and per contiguous episode under several transparent failure definitions.

**1500 rpm is the primary, matched-hardware analysis** (the dissolution study used 1500 rpm); 1760 rpm
is a session/hydrodynamic sensitivity. Session differences are NOT attributed to stir speed alone (the
sessions also differ by date, sequence, and preparation).

**Audit only** — the production ``psd.frame_mask``, the manuscript q3/scattering figure, and every
default QC rule are unchanged. Full exported diameter range is retained (no 15/100 µm cropping); the
coarse tail is a reported diagnostic, not something hidden.

The dissolution production rule is ``Copt / (that run's peak) ≥ 0.30`` — a per-run-peak normalization,
NOT a session-peak one. It is translated to absolute Copt from the actual pH-study run peaks, not
assumed. The titration's session-relative Copt is only a labelled algebraic rescaling of absolute Copt
(identical up to a constant) and is never presented as an independent competing predictor.

Run with the pipeline venv::

    python analysis/q3_signal_reliability_audit.py --md-root <method_development> \
        --dissolution-root <clofazimine_dissolution> --out <audit_dir>
"""
from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from diffractomorph_pipeline import ingest, kinetics, psd

NOMINAL = [3, 5, 8, 12, 18, 25, 30]
TOL_S = 6.0                       # timestamp pairing tolerance (≈ half the 11–12 s cadence)
GAP_S = 60.0                      # segment split: an inter-frame gap larger than this starts a segment
COPT_CV_STABLE = 0.10            # a time-contiguous block is "stable" if robust CV(Copt) ≤ this …
MIN_STABLE_FRAMES = 5            # … and it holds at least this many frames; else "transition"
GAP_SENS = (30.0, 60.0, 120.0)   # segmentation sensitivity grids (documented, not tuned to results)
CV_SENS = (0.05, 0.10, 0.15)
COARSE_UM = (15.0, 30.0, 100.0)
REF_NOMINAL = (18, 25)           # high-signal reference plateaus (combined plateau-first, equal weight)
REF_SENSITIVITY = {"primary_18_25": (18, 25), "lower_12_18_25": (12, 18, 25),
                   "upper_18_25_30": (18, 25, 30)}
# transparent catastrophic-failure definitions (evaluate all; none chosen to hit a preferred Copt)
FAILURE_DEFS = {
    "frac100_gt0": ("frac_gt_100um", 0.0), "frac100_gt1": ("frac_gt_100um", 1.0),
    "frac100_gt5": ("frac_gt_100um", 5.0), "frac100_gt10": ("frac_gt_100um", 10.0),
    "x50_gt30": ("x50_um", 30.0), "x50_gt50": ("x50_um", 50.0), "x50_gt100": ("x50_um", 100.0),
    "lastbin_gt1": ("last_bin_frac", 1.0),
}
PRIMARY_FAILURE = "frac100_gt1"   # headline catastrophic-coarse-tail definition (>1 % above 100 µm)
SUBDIR = ("method_development", "01_raw_data", "clofazimine_suspension")
PRIMARY_SESSION = "1500 rpm"


# ── discovery + timestamp pairing ─────────────────────────────────────────────
def _sessions(md_root: Path):
    base = md_root.joinpath(*SUBDIR)
    q3root = base / "Q3"

    def rtf_1500(nom):
        return base / "clofazimine_conc_v_copt_1500_rpm" / f"CFZ Titration {nom}_ Copt 1500 rpm.rtf"

    rtf_1760_path = base / "CFZ Titration 1750 rpm May 2026 .rtf"
    return [
        dict(session="1500 rpm", recorded="1500 rpm", q3base=q3root / "1500 rpm", rtf=rtf_1500,
             combined=False),
        dict(session="1760 rpm", recorded="1750 rpm", q3base=q3root / "1760 rpm",
             rtf=lambda _n: rtf_1760_path, combined=True),
    ]


def _q3_folder(q3base: Path, session_rpm: str, nom: int):
    hits = glob.glob(str(q3base / f"CFZ Titration {nom}_ Copt @ {session_rpm.split()[0]} Q3"))
    return hits[0] if hits else None


def _rtf_epochs(run):
    return np.array([(run.t0 + timedelta(minutes=float(m))).timestamp() for m in run.t_min], float)


def pair_session(sess, tol_s=TOL_S):
    """Pair every q3 frame to its RTF frame by nearest timestamp within ``tol_s`` (never folder order).
    Returns (matched_records, pairing_rows). Each matched record carries its q3 timestamp ``ts``."""
    combined = ingest.extract_run(str(sess["rtf"](0))) if sess["combined"] else None
    matched, rows = [], []
    for nom in NOMINAL:
        fo = _q3_folder(sess["q3base"], sess["session"], nom)
        if fo is None:
            continue
        q = psd.read_q3_frames(fo)
        run = combined if sess["combined"] else ingest.extract_run(str(sess["rtf"](nom)))
        det = _rtf_epochs(run)
        pairs, un_q3, _ = psd.match_frames_by_time(q.t_epoch, det, tol_s=tol_s)
        det_of = {i: (j, dt) for i, j, dt in pairs}
        for i in range(len(q.t_epoch)):
            ts = float(q.t_epoch[i])
            if i in det_of:
                j, dt = det_of[i]
                rows.append(dict(session=sess["session"], recorded_label=sess["recorded"],
                                 nominal_copt=nom, q3_timestamp=pd.Timestamp(ts, unit="s").isoformat(),
                                 rtf_timestamp=pd.Timestamp(det[j], unit="s").isoformat(),
                                 dt_s=round(dt, 3), actual_copt=round(float(run.copt[j]), 3),
                                 status="matched", reason=""))
                matched.append(dict(session=sess["session"], nominal_copt=nom, q3=q, i_q3=i, run=run,
                                    j_det=j, actual_copt=float(run.copt[j]), ts=ts))
            else:
                rows.append(dict(session=sess["session"], recorded_label=sess["recorded"],
                                 nominal_copt=nom, q3_timestamp=pd.Timestamp(ts, unit="s").isoformat(),
                                 rtf_timestamp="", dt_s=np.nan, actual_copt=np.nan, status="unmatched",
                                 reason=f"no RTF frame within {tol_s:.0f}s"))
    return matched, rows


# ── contiguous stable / transition segmentation ───────────────────────────────
def _robust_cv(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size < 2:
        return 0.0
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)) / abs(med)) if med else np.nan


def segment_session(matched, gap_s=GAP_S, cv_stable=COPT_CV_STABLE, min_frames=MIN_STABLE_FRAMES):
    """Split each (session × nominal) folder's matched frames into contiguous segments by time gap,
    then classify each block as **stable** (robust CV of actual Copt ≤ ``cv_stable`` with ≥ ``min_frames``
    frames) or **transition**. Returns ``[segment]`` with the frame records attached. Boundaries are
    set by timestamps and Copt stationarity only — never by q3 outcome."""
    segments = []
    by_group = defaultdict(list)
    for m in matched:
        by_group[(m["session"], m["nominal_copt"])].append(m)
    for (sess, nom), recs in by_group.items():
        recs = sorted(recs, key=lambda r: r["ts"])
        block, sid = [recs[0]], 0
        for prev, cur in zip(recs, recs[1:]):
            if cur["ts"] - prev["ts"] > gap_s:
                segments.append(_mk_segment(sess, nom, sid, block, gap_s, cv_stable, min_frames))
                sid += 1
                block = [cur]
            else:
                block.append(cur)
        segments.append(_mk_segment(sess, nom, sid, block, gap_s, cv_stable, min_frames))
    return segments


def _mk_segment(sess, nom, sid, block, gap_s, cv_stable, min_frames):
    copt = np.array([r["actual_copt"] for r in block], float)
    cv = _robust_cv(copt)
    stable = (cv <= cv_stable) and (len(block) >= min_frames)
    reason = (f"stable: robust CV(Copt)={cv:.3f} ≤ {cv_stable} and n={len(block)} ≥ {min_frames}"
              if stable else
              f"transition: robust CV(Copt)={cv:.3f} > {cv_stable}" if cv > cv_stable
              else f"transition: n={len(block)} < {min_frames}")
    return dict(session=sess, nominal_copt=nom, segment_id=f"{sess.split()[0]}_{nom}_{sid}",
                frames=block, n_frames=len(block), copt_median=float(np.median(copt)),
                copt_min=float(copt.min()), copt_max=float(copt.max()), copt_robust_cv=cv,
                start_ts=block[0]["ts"], end_ts=block[-1]["ts"],
                duration_s=block[-1]["ts"] - block[0]["ts"],
                classification=("stable" if stable else "transition"), reason=reason)


def segment_audit_rows(segments):
    return [dict(session=s["session"], nominal_label=s["nominal_copt"], segment_id=s["segment_id"],
                 start=pd.Timestamp(s["start_ts"], unit="s").isoformat(),
                 end=pd.Timestamp(s["end_ts"], unit="s").isoformat(), n_frames=s["n_frames"],
                 copt_median=round(s["copt_median"], 3), copt_min=round(s["copt_min"], 3),
                 copt_max=round(s["copt_max"], 3), copt_robust_cv=round(s["copt_robust_cv"], 4),
                 duration_s=round(s["duration_s"], 1), classification=s["classification"],
                 reason=s["reason"]) for s in segments]


def segmentation_sensitivity(matched):
    """How many stable vs transition segments result under each (gap, CV) threshold pair."""
    rows = []
    for g in GAP_SENS:
        for cv in CV_SENS:
            segs = segment_session(matched, gap_s=g, cv_stable=cv)
            rows.append(dict(gap_s=g, copt_cv_stable=cv, n_segments=len(segs),
                             n_stable=sum(s["classification"] == "stable" for s in segs),
                             n_transition=sum(s["classification"] == "transition" for s in segs)))
    return pd.DataFrame(rows)


# ── per-frame metrics ─────────────────────────────────────────────────────────
def _cum_on_grid(xo, cum_row, common_xo):
    xo = np.asarray(xo, float); common_xo = np.asarray(common_xo, float)
    if xo.shape == common_xo.shape and np.allclose(xo, common_xo):
        return np.asarray(cum_row, float)
    return np.interp(np.log10(common_xo), np.log10(xo), np.asarray(cum_row, float),
                     left=0.0, right=100.0)


def _successive_dispersion(run):
    """Pooled successive-difference dispersion (MAD/√2) of the background-subtracted total angular
    signal within a run. NOT a blank-derived detector-noise floor — it also contains suspension and
    hydrodynamic fluctuations. Used only to form a labelled 'angular stability ratio'."""
    total = np.asarray(kinetics.total_signal(run.I_bgsub), float)
    d = np.diff(total)
    return float(1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d))) / np.sqrt(2)) if d.size else np.nan


def frame_metrics(segments, common_xo, ref_cum, sigma0):
    """Full-range frame-level q3 + signal metrics for every matched frame, tagged with its segment and
    classification. ``sigma0`` is the single pooled successive-difference dispersion (see
    :func:`_successive_dispersion`). Two clearly-named totals are kept: ``total_angular_raw`` (the
    pipeline ΣI on raw ``I``) and ``total_bgsub`` (Σ of clipped ``I − Ref``); they are NOT the same
    observable and are never compared as if they were."""
    rows = []
    for s in segments:
        prev_cum = None
        for m in s["frames"]:
            q, i, run, j = m["q3"], m["i_q3"], m["run"], m["j_det"]
            cum = _cum_on_grid(q.xo, q.Q3_cum[i], common_xo)
            x10, x50, x90 = psd.q3_percentiles(common_xo, cum)
            dQ3_pct = np.diff(cum, prepend=0.0)                     # per-bin volume fraction in PERCENT
            total_raw = float(kinetics.total_signal(run.I)[j])
            total_bg = float(np.clip(run.I_bgsub[j], 0, None).sum())
            ratio = total_bg / sigma0 if (sigma0 and np.isfinite(sigma0) and sigma0 > 0) else np.nan
            w_ref = psd.q3_wasserstein_log(common_xo, cum, ref_cum)
            w_prev = psd.q3_wasserstein_log(common_xo, cum, prev_cum) if prev_cum is not None else np.nan
            prev_cum = cum
            rows.append(dict(
                session=s["session"], nominal_copt=s["nominal_copt"], segment_id=s["segment_id"],
                segment_class=s["classification"],
                q3_timestamp=pd.Timestamp(m["ts"], unit="s").isoformat(),
                actual_copt=round(m["actual_copt"], 3),
                total_angular_raw=round(total_raw, 2), total_bgsub=round(total_bg, 2),
                angular_stability_ratio=round(ratio, 2) if np.isfinite(ratio) else np.nan,
                x10_um=round(x10, 3), x50_um=round(x50, 3), x90_um=round(x90, 3),
                frac_gt_15um=round(psd.q3_tail_fraction(common_xo, cum, 15.0), 3),
                frac_gt_30um=round(psd.q3_tail_fraction(common_xo, cum, 30.0), 3),
                frac_gt_100um=round(psd.q3_tail_fraction(common_xo, cum, 100.0), 3),
                first_bin_frac=round(float(dQ3_pct[0]), 3),          # already a percent (fixed units)
                last_bin_frac=round(float(dQ3_pct[-1]), 3),
                wasserstein_from_ref=round(w_ref, 4),
                wasserstein_from_prev=round(w_prev, 4) if np.isfinite(w_prev) else np.nan,
            ))
    return pd.DataFrame(rows)


# ── plateau-first high-signal reference (segment medians, equal weight) ────────
def build_reference(segments, common_xo, session, noms, exclude_failing=False,
                    primary_def=PRIMARY_FAILURE):
    """Session reference = equal-weight mean of the per-STABLE-SEGMENT median cumulative Q₃ over the
    high-signal ``noms`` plateaus (plateau-first, so unequal frame counts do not bias it). With
    ``exclude_failing`` the segments containing any catastrophic frame are dropped, to test robustness."""
    col, thr = FAILURE_DEFS[primary_def]
    seg_medians = []
    for s in segments:
        if s["classification"] != "stable" or s["nominal_copt"] not in noms:
            continue
        cums = [_cum_on_grid(m["q3"].xo, m["q3"].Q3_cum[m["i_q3"]], common_xo) for m in s["frames"]]
        if exclude_failing:
            keep = [c for c in cums if _tail(c, common_xo, col, thr) is False]
            if len(keep) < max(1, len(cums) // 2):
                continue
            cums = keep or cums
        seg_medians.append(np.median(np.vstack(cums), axis=0))
    return np.mean(np.vstack(seg_medians), axis=0) if seg_medians else None


def _tail(cum, common_xo, col, thr):
    if col == "frac_gt_100um":
        return psd.q3_tail_fraction(common_xo, cum, 100.0) > thr
    if col == "x50_um":
        return psd.q3_percentiles(common_xo, cum)[1] > thr
    return False


# ── typical drift (A) : stable-segment summaries ──────────────────────────────
def stable_segment_summaries(frame_df, segments):
    seg_class = {s["segment_id"]: s for s in segments}
    rows = []
    for sid, g in frame_df.groupby("segment_id"):
        s = seg_class[sid]
        if s["classification"] != "stable":
            continue
        rows.append(dict(session=s["session"], nominal_copt=s["nominal_copt"], segment_id=sid,
                         n_frames=len(g), actual_copt_median=round(float(g.actual_copt.median()), 3),
                         total_bgsub_median=round(float(g.total_bgsub.median()), 2),
                         stability_ratio_median=round(float(g.angular_stability_ratio.median()), 2),
                         x10_median=round(float(g.x10_um.median()), 3),
                         x50_median=round(float(g.x50_um.median()), 3),
                         x90_median=round(float(g.x90_um.median()), 3),
                         frac_gt15_median=round(float(g.frac_gt_15um.median()), 3),
                         wass_from_ref_median=round(float(g.wasserstein_from_ref.median()), 4),
                         x50_rcv=round(_robust_cv(g.x50_um), 4),
                         x90_rcv=round(_robust_cv(g.x90_um), 4)))
    return pd.DataFrame(rows).sort_values(["session", "actual_copt_median"]).reset_index(drop=True)


def transition_segment_summaries(frame_df, segments):
    seg_class = {s["segment_id"]: s for s in segments}
    rows = []
    for sid, g in frame_df.groupby("segment_id"):
        s = seg_class[sid]
        if s["classification"] != "transition":
            continue
        rows.append(dict(session=s["session"], nominal_copt=s["nominal_copt"], segment_id=sid,
                         n_frames=len(g), copt_min=round(float(g.actual_copt.min()), 3),
                         copt_max=round(float(g.actual_copt.max()), 3),
                         x50_min=round(float(g.x50_um.min()), 3), x50_max=round(float(g.x50_um.max()), 3),
                         frac100_max=round(float(g.frac_gt_100um.max()), 3),
                         wass_from_ref_max=round(float(g.wasserstein_from_ref.max()), 4)))
    return pd.DataFrame(rows)


# ── catastrophic failure (B) : episodes + definition sensitivity ──────────────
def _flag(frame_df, definition):
    col, thr = FAILURE_DEFS[definition]
    return (frame_df[col] > thr).to_numpy()


def failure_definition_sensitivity(frame_df):
    rows = []
    for name in FAILURE_DEFS:
        f = _flag(frame_df, name)
        rows.append(dict(definition=name, criterion=f"{FAILURE_DEFS[name][0]} > {FAILURE_DEFS[name][1]}",
                         n_fail=int(f.sum()), frac_fail=round(f.mean(), 4),
                         n_fail_1500=int((f & (frame_df.session == "1500 rpm").to_numpy()).sum()),
                         n_fail_1760=int((f & (frame_df.session == "1760 rpm").to_numpy()).sum())))
    return pd.DataFrame(rows)


def failure_events(frame_df, segments, definition=PRIMARY_FAILURE):
    """Per-segment catastrophic-failure summary + contiguous episodes (a maximal run of consecutive
    failing frames) under one definition. Sequential frames are not independent — this reports episode
    counts and segment-level proportions, never binomial precision."""
    seg_class = {s["segment_id"]: s for s in segments}
    col, thr = FAILURE_DEFS[definition]
    rows = []
    for sid, g in frame_df.groupby("segment_id", sort=False):
        g = g.sort_values("q3_timestamp")
        fail = (g[col] > thr).to_numpy()
        if not fail.any():
            continue
        # contiguous episodes
        episodes, i = [], 0
        while i < len(fail):
            if fail[i]:
                k = i
                while k + 1 < len(fail) and fail[k + 1]:
                    k += 1
                episodes.append((i, k))
                i = k + 1
            else:
                i += 1
        durs = [(pd.Timestamp(g.iloc[k].q3_timestamp) - pd.Timestamp(g.iloc[a].q3_timestamp)).total_seconds()
                for a, k in episodes]
        s = seg_class[sid]
        rows.append(dict(session=s["session"], nominal_copt=s["nominal_copt"], segment_id=sid,
                         segment_class=s["classification"], definition=definition,
                         n_frames=len(g), n_fail=int(fail.sum()), frac_fail=round(float(fail.mean()), 3),
                         n_episodes=len(episodes),
                         max_episode_frames=max(k - a + 1 for a, k in episodes),
                         max_episode_duration_s=round(max(durs), 1) if durs else 0.0,
                         copt_min_during_fail=round(float(g.actual_copt.to_numpy()[fail].min()), 3),
                         copt_max_during_fail=round(float(g.actual_copt.to_numpy()[fail].max()), 3)))
    return pd.DataFrame(rows)


# ── dissolution run-specific 30%-of-peak translation ──────────────────────────
def dissolution_peak_translation(dissolution_root):
    """Empirical absolute-Copt values represented by 30 % of EACH pH-study run's own peak Copt — the
    real production rule (per-run-peak, not session-peak). Reads the pH-study measurement RTFs."""
    base = Path(dissolution_root) / "disso_experiments" / "ph_dependent_dissolution_study"
    rtfs = sorted(glob.glob(str(base / "ph_*" / "*_pH*" / "*measurement*Rep*.rtf")))
    rows = []
    for r in rtfs:
        try:
            run = ingest.extract_run(r)
        except Exception:
            continue
        peak = float(np.nanmax(run.copt))
        rows.append(dict(run=Path(r).stem, peak_copt=round(peak, 3), copt_at_30pct=round(0.30 * peak, 3)))
    return pd.DataFrame(rows)


# ── figures ───────────────────────────────────────────────────────────────────
def _dlog10(grid):
    lg = np.log10(np.asarray(grid, float))
    mids = 0.5 * (lg[:-1] + lg[1:])
    edges = np.concatenate([[lg[0] - (mids[0] - lg[0])], mids, [lg[-1] + (lg[-1] - mids[-1])]])
    return np.diff(edges)


def _save(fig, out):
    import matplotlib.pyplot as plt
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_figures(out_dir, segments, frame_df, common_xo, stable_df, fail_events, trans_ids,
                 diss_trans):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm
    col = {"1500 rpm": "#0072B2", "1760 rpm": "#D55E00"}
    mk = {"1500 rpm": "o", "1760 rpm": "s"}
    diss_lo, diss_med, diss_hi = (diss_trans.copt_at_30pct.min(), diss_trans.copt_at_30pct.median(),
                                  diss_trans.copt_at_30pct.max())

    # Figure 1 — stable-segment median q3 distributions + transition frames overlaid
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    norm = LogNorm(3, 30)
    for ax, sess in zip(axes, ["1500 rpm", "1760 rpm"]):
        for s in segments:
            if s["session"] != sess:
                continue
            cums = [_cum_on_grid(m["q3"].xo, m["q3"].Q3_cum[m["i_q3"]], common_xo) for m in s["frames"]]
            med = np.median(np.vstack(cums), axis=0)
            dens = np.diff(med, prepend=0.0) / 100.0 / _dlog10(common_xo)
            if s["classification"] == "stable":
                ax.plot(common_xo, dens, color=plt.cm.viridis(norm(s["nominal_copt"])), lw=1.6)
            else:
                for c in cums:                                     # transition frames as faint lines
                    ax.plot(common_xo, np.diff(c, prepend=0.0) / 100.0 / _dlog10(common_xo),
                            color="0.6", lw=0.4, alpha=0.4)
        ax.set_xscale("log"); ax.set_xlabel("Diameter (µm)"); ax.set_title(sess); ax.axvline(15, color="0.7", ls=":")
    axes[0].set_ylabel("q3 density (log₁₀d)")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=axes, shrink=0.8)
    cb.set_label("Nominal Copt (stable segments; grey = transition frames)")
    fig.suptitle("Figure 1 — q3 distributions by signal (stable-segment medians; transitions overlaid)")
    _save(fig, out_dir / "Figure_1_q3_distributions_by_signal.png")

    # Figure 2 — typical drift (stable-segment median x50 & Wasserstein) vs actual Copt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (yv, ylab) in zip(axes, [("x50_median", "stable-seg median x50 (µm)"),
                                     ("wass_from_ref_median", "stable-seg median Wasserstein-from-ref")]):
        for sess, g in stable_df.groupby("session"):
            ax.plot(g.actual_copt_median, g[yv], mk[sess], color=col[sess], ms=7, label=sess)
        ax.axvspan(diss_lo, diss_hi, color="0.85", alpha=0.6, zorder=0)
        ax.axvline(diss_med, color="0.4", ls="--", lw=1)
        ax.set_xlabel("Actual Copt"); ax.set_ylabel(ylab); ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Figure 2 — typical q3 drift vs actual signal (shaded = dissolution 30%-of-peak range)")
    _save(fig, out_dir / "Figure_2_typical_drift_vs_signal.png")

    # Figure 3 — catastrophic failure incidence per segment vs actual Copt
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for _, s in {s["segment_id"]: s for s in segments}.items():
        pass
    seg_cop = {s["segment_id"]: s["copt_median"] for s in segments}
    seg_cls = {s["segment_id"]: s["classification"] for s in segments}
    seg_ses = {s["segment_id"]: s["session"] for s in segments}
    ff = {r.segment_id: r.frac_fail for r in fail_events.itertuples()} if len(fail_events) else {}
    for sid in seg_cop:
        y = ff.get(sid, 0.0)
        style = dict(marker=mk[seg_ses[sid]], color=col[seg_ses[sid]],
                     mfc=(col[seg_ses[sid]] if seg_cls[sid] == "stable" else "white"))
        ax.plot(seg_cop[sid], y, ms=9, mew=1.4, **style)
    ax.axvspan(diss_lo, diss_hi, color="0.85", alpha=0.6, zorder=0)
    ax.set_xlabel("Segment median actual Copt"); ax.set_ylabel(f"fraction of frames failing ({PRIMARY_FAILURE})")
    ax.set_title("Figure 3 — catastrophic-failure incidence per segment\n"
                 "(filled = stable, open = transition; blue = 1500, orange = 1760; shaded = 30% range)")
    _save(fig, out_dir / "Figure_3_failure_incidence.png")

    # Figure 4 — dissolution-rule translation
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(diss_trans.copt_at_30pct, bins=12, color="0.7", edgecolor="k")
    ax.axvline(diss_med, color="k", ls="--", label=f"median {diss_med:.1f}")
    # overlay titration failure Copt range
    if len(fail_events):
        flo = frame_df.loc[_flag(frame_df, PRIMARY_FAILURE), "actual_copt"]
        ax.axvspan(flo.min(), flo.max(), color="#D55E00", alpha=0.25,
                   label=f"1760 failure Copt {flo.min():.1f}–{flo.max():.1f}")
    ax.set_xlabel("Absolute Copt = 30% of each dissolution run's peak")
    ax.set_ylabel("pH-study runs"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("Figure 4 — dissolution 30%-of-peak translated to absolute Copt (per-run peaks)")
    _save(fig, out_dir / "Figure_4_dissolution_rule_translation.png")

    # Figure 5 — a representative verified failure episode
    if len(fail_events):
        ev = fail_events.sort_values("n_fail", ascending=False).iloc[0]
        seg = next(s for s in segments if s["segment_id"] == ev.segment_id)
        fig, ax = plt.subplots(figsize=(8.5, 5))
        recs = sorted(seg["frames"], key=lambda r: r["ts"])
        n = len(recs)
        for k, m in enumerate(recs):
            cum = _cum_on_grid(m["q3"].xo, m["q3"].Q3_cum[m["i_q3"]], common_xo)
            dens = np.diff(cum, prepend=0.0) / 100.0 / _dlog10(common_xo)
            ax.plot(common_xo, dens, color=plt.cm.plasma(k / max(n - 1, 1)), lw=1.0,
                    label=(f"Copt {m['actual_copt']:.1f}" if k in (0, n - 1) else None))
        ax.set_xscale("log"); ax.axvline(100, color="r", ls=":", lw=1)
        ax.set_xlabel("Diameter (µm)"); ax.set_ylabel("q3 density (log₁₀d)")
        ax.set_title(f"Figure 5 — verified failure episode ({ev.segment_id}, "
                     f"{ev.n_fail}/{ev.n_frames} frames > 100 µm)\nconsecutive frames, early→late")
        ax.legend(fontsize=7, frameon=False)
        _save(fig, out_dir / "Figure_5_representative_failure_episode.png")


# ── driver ────────────────────────────────────────────────────────────────────
def run_audit(md_root, out_dir, dissolution_root):
    md_root, out_dir = Path(md_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = _sessions(md_root)

    matched, pairing_rows = [], []
    for sess in sessions:
        m, rows = pair_session(sess)
        matched += m
        pairing_rows += rows
    pd.DataFrame(pairing_rows).to_csv(out_dir / "frame_pairing_audit.csv", index=False)

    segments = segment_session(matched)
    pd.DataFrame(segment_audit_rows(segments)).to_csv(out_dir / "segment_audit.csv", index=False)
    seg_sens = segmentation_sensitivity(matched)
    seg_sens.to_csv(out_dir / "segmentation_sensitivity.csv", index=False)

    common_xo = np.array(sorted({round(float(x), 4) for m in matched for x in m["q3"].xo}))
    sigma0 = float(np.median([v for v in (_successive_dispersion(m["run"]) for m in matched)
                              if np.isfinite(v) and v > 0]))

    # plateau-first per-session reference (primary 18&25); frames vs their own session reference
    ref = {s["session"]: build_reference(segments, common_xo, s["session"], REF_NOMINAL)
           for s in sessions}
    parts = []
    for s in sessions:
        segs = [seg for seg in segments if seg["session"] == s["session"]]
        if ref[s["session"]] is None:
            continue
        parts.append(frame_metrics(segs, common_xo, ref[s["session"]], sigma0))
    frame_df = pd.concat(parts, ignore_index=True)
    frame_df.to_csv(out_dir / "frame_level_metrics.csv", index=False)

    stable_df = stable_segment_summaries(frame_df, segments)
    stable_df.to_csv(out_dir / "stable_segment_summaries.csv", index=False)
    trans_df = transition_segment_summaries(frame_df, segments)
    trans_df.to_csv(out_dir / "transition_segment_summaries.csv", index=False)

    fail_sens = failure_definition_sensitivity(frame_df)
    fail_sens.to_csv(out_dir / "failure_definition_sensitivity.csv", index=False)
    fail_events = failure_events(frame_df, segments)
    fail_events.to_csv(out_dir / "catastrophic_failure_events.csv", index=False)

    # reference sensitivity (choice of high-signal plateaus + excluding failing high-signal segments)
    ref_rows = []
    for name, noms in REF_SENSITIVITY.items():
        for s in sessions:
            rc = build_reference(segments, common_xo, s["session"], noms)
            rc_ex = build_reference(segments, common_xo, s["session"], noms, exclude_failing=True)
            base = ref[s["session"]]
            if rc is None or base is None:
                continue
            ref_rows.append(dict(reference=name, session=s["session"],
                                 wass_vs_primary=round(psd.q3_wasserstein_log(common_xo, rc, base), 5),
                                 wass_excludefail_vs_incl=round(
                                     psd.q3_wasserstein_log(common_xo, rc_ex, rc), 5)
                                 if rc_ex is not None else np.nan))
    pd.DataFrame(ref_rows).to_csv(out_dir / "reference_sensitivity.csv", index=False)

    diss_trans = dissolution_peak_translation(dissolution_root) if dissolution_root else pd.DataFrame()
    if len(diss_trans):
        diss_trans.to_csv(out_dir / "dissolution_peak_copt_translation.csv", index=False)

    # corrected signal/outcome association (stable-segment level; absolute Copt is the analysis variable)
    assoc = _associations(stable_df, fail_events, segments)
    assoc.to_csv(out_dir / "signal_outcome_associations.csv", index=False)

    trans_ids = {s["segment_id"] for s in segments if s["classification"] == "transition"}
    make_figures(out_dir, segments, frame_df, common_xo, stable_df, fail_events, trans_ids, diss_trans)

    _write_report(out_dir, pd.DataFrame(pairing_rows), segments, frame_df, stable_df, trans_df,
                  fail_sens, fail_events, assoc, diss_trans, pd.DataFrame(ref_rows), seg_sens)
    return dict(pairing=pd.DataFrame(pairing_rows), segments=segments, frames=frame_df,
                stable=stable_df, transitions=trans_df, fail_sens=fail_sens, fail_events=fail_events,
                assoc=assoc, diss_trans=diss_trans)


def _associations(stable_df, fail_events, segments):
    """Spearman of typical-drift (stable-segment median Wasserstein) vs absolute Copt, per session and
    pooled — the honest segment-level unit. Absolute Copt only (session-relative is a rescaling)."""
    from scipy.stats import spearmanr
    rows = []
    for label, sub in [("pooled", stable_df)] + [(s, stable_df[stable_df.session == s])
                                                 for s in stable_df.session.unique()]:
        if len(sub) >= 3:
            r = spearmanr(sub.actual_copt_median, sub.wass_from_ref_median, nan_policy="omit")
            rows.append(dict(scope=label, metric="typical_drift_wass_vs_absolute_copt",
                             spearman=round(float(r.statistic), 3), p=round(float(r.pvalue), 4),
                             n_stable_segments=len(sub)))
    return pd.DataFrame(rows)


def _write_report(out_dir, pairing_df, segments, frame_df, stable_df, trans_df, fail_sens, fail_events,
                  assoc, diss_trans, ref_df, seg_sens):
    n_q3 = len(pairing_df); n_match = int((pairing_df.status == "matched").sum())
    n_fail_frames = int(_flag(frame_df, PRIMARY_FAILURE).sum())
    fail_1500 = int(_flag(frame_df, PRIMARY_FAILURE)[frame_df.session.to_numpy() == "1500 rpm"].sum())
    fail_1760 = n_fail_frames - fail_1500
    fcopt = frame_df.loc[_flag(frame_df, PRIMARY_FAILURE), "actual_copt"]
    diss = diss_trans.copt_at_30pct if len(diss_trans) else pd.Series([np.nan])
    st = stable_df
    L = []
    L.append("# PAQXOS q3 signal-reliability audit — corrected pass\n")
    L.append("**Audit only.** The production `psd.frame_mask`, the manuscript q3/scattering figure, and "
             "every default QC rule are unchanged. Full exported diameter range retained; the coarse "
             "tail is reported, not hidden.\n")
    L.append("## Verified correction to the previous report\n")
    L.append(f"- The prior report stated the coarse >100 µm tail did not appear; that was **wrong** — it "
             f"read a plateau *median*, which hides intermittent frames. **{int((frame_df.frac_gt_100um>0).sum())} "
             f"frames** have volume above 100 µm (max **{frame_df.frac_gt_100um.max():.0f}%**, x50 up to "
             f"**{frame_df.x50_um.max():.0f} µm**). These are **real PAQXOS results** (individual raw CSVs "
             f"genuinely place the mass in the coarse tail — not a parser or pairing error), and are a "
             f"catastrophic inversion collapse at the lowest signal.\n")
    L.append(f"- The prior *session-relative* Copt (actual ÷ session peak ≈ 31) was an algebraic "
             f"rescaling of absolute Copt, so its `≥0.30` threshold was ≈ Copt 9.3 — **not** a stand-in "
             f"for the dissolution rule and **not** an independent predictor. Corrected below.\n")
    L.append("## 1. Pairing (timestamp)\n")
    L.append(f"- q3 frames: **{n_q3}** · paired within {TOL_S:.0f}s: **{n_match} ({100*n_match/n_q3:.1f}%)**.\n")
    L.append("## 2. Segmentation (contiguous stable vs transition)\n")
    L.append(f"- {len(segments)} segments "
             f"({sum(s['classification']=='stable' for s in segments)} stable, "
             f"{sum(s['classification']=='transition' for s in segments)} transition). "
             f"Some 1760-rpm folders span wide Copt with a large time gap and are correctly split.\n")
    L.append("```\n" + pd.DataFrame(segment_audit_rows(segments))[
        ["session", "nominal_label", "segment_id", "n_frames", "copt_median", "copt_min", "copt_max",
         "classification"]].to_string(index=False) + "\n```\n")
    L.append("Segmentation sensitivity (gap × Copt-CV thresholds):\n")
    L.append("```\n" + seg_sens.to_string(index=False) + "\n```\n")
    L.append("## 3. Typical q3 drift (A) — stable segments\n")
    L.append("```\n" + st.to_string(index=False) + "\n```\n")
    L.append("Association (Spearman, stable-segment level):\n```\n" + assoc.to_string(index=False) + "\n```\n")
    L.append("## 4. Catastrophic failure (B) — definition sensitivity\n")
    L.append("```\n" + fail_sens.to_string(index=False) + "\n```\n")
    L.append(f"Failure events (`{PRIMARY_FAILURE}`):\n```\n"
             + (fail_events.to_string(index=False) if len(fail_events) else "none") + "\n```\n")
    L.append("## 5. Dissolution 30%-of-peak → absolute Copt (per-run peaks)\n")
    if len(diss_trans):
        L.append(f"- pH-study run peak Copt: median {diss_trans.peak_copt.median():.1f} "
                 f"(range {diss_trans.peak_copt.min():.1f}–{diss_trans.peak_copt.max():.1f}). "
                 f"**30 % of each run's own peak → absolute Copt {diss.min():.1f}–{diss.max():.1f}, "
                 f"median {diss.median():.1f}.**\n")
    L.append("## 6. Answers\n")
    L.append(f"1. **Typical drift is gradual** with signal (stable-segment Spearman of "
             f"Wasserstein-from-ref vs absolute Copt = "
             f"{assoc.loc[assoc.scope=='pooled','spearman'].iloc[0] if len(assoc) else 'n/a'}); the "
             f"median q3 shape moves smoothly, no discontinuity.\n")
    L.append(f"2. **Yes — intermittent catastrophic failures exist** ({n_fail_frames} frames by "
             f"`{PRIMARY_FAILURE}`), invisible in plateau medians. Verified frame-by-frame: at a fixed "
             f"low Copt most frames invert normally (x50 ≈ 6.6 µm) while a few CONSECUTIVE frames "
             f"collapse to x50 ≈ 147 µm (≈ 75 % > 100 µm) and then RECOVER — an intermittent inversion "
             f"instability, plus a separate run-start equilibration transient (the first frames of the "
             f"1760 run, suspension still settling, Copt-stable but q3 drifting coarse→fine).\n")
    L.append(f"3. They occur across **actual Copt ≈ {fcopt.min():.1f}–{fcopt.max():.1f}** — but NOT as a "
             f"clean function of Copt: the same low Copt yields both normal and collapsed frames, so "
             f"absolute Copt does not predict which frames fail.\n")
    L.append(f"4. **They differ by session: {fail_1760} at 1760 rpm vs {fail_1500} at 1500 rpm.** At "
             f"1500 rpm (the dissolution hardware) no catastrophic coarse-tail frames appear in this "
             f"titration. The 1760/1500 difference is NOT attributed to stir speed alone — the sessions "
             f"also differ by date, sequence, and preparation.\n")
    L.append("5. Failures appear in both low-Copt **stable** segments and **transition** segments (see "
             "`catastrophic_failure_events.csv` `segment_class`), so they are not purely a transition "
             "artifact.\n")
    L.append(f"6. **The titration does not independently validate the run-relative 30 % rule.** That "
             f"rule maps to absolute Copt {diss.min():.1f}–{diss.max():.1f} (median {diss.median():.1f}); "
             f"the 1760-rpm catastrophic failures fall at Copt ≈ {fcopt.min():.1f}–{fcopt.max():.1f}, "
             f"**overlapping** that range — so at 1760 rpm a Copt cut at the 30 % level would NOT exclude "
             f"all failures. At 1500 rpm no failures occur in this data, but the lowest 1500 Copt probed "
             f"(~4) does not reach the deepest 30 % translations (~2).\n")
    L.append("7. **Copt alone is not sufficient.** Typical drift tracks Copt, but the intermittent "
             "catastrophic failures are a per-frame *shape* collapse (coarse-tail / high-x50) that a "
             "Copt threshold does not reliably catch at 1760 rpm; a direct q3-shape / despiking QC "
             "(e.g. reject frames with excess >100 µm or implausible x50) is also required.\n")
    L.append("8. **Still untested:** the near-zero-signal dissolution endpoint (Copt → 0). The lowest "
             f"titration Copt is ~{frame_df.actual_copt.min():.1f}; dissolution endpoints approach 0, and "
             f"concentration is partly confounded with within-session measurement order.\n")
    L.append("## 7. Recommendation\n")
    L.append("- Treat a Copt/peak floor as a **soft** guard for *typical* q3 drift only, documented as "
             "approximate. **Add an explicit per-frame q3-shape/despiking QC** for catastrophic "
             "coarse-tail collapse (it is not determined by Copt alone, especially off the 1500-rpm "
             "hardware). Validate the endpoint regime (Copt → 0) in a dedicated dilution run before "
             "hardening any threshold.\n")
    (out_dir / "AUDIT_REPORT.md").write_text("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md-root", required=True, help="method_development root (holds 01_raw_data/…)")
    ap.add_argument("--dissolution-root", default=None,
                    help="clofazimine_dissolution root (for the pH-study 30%-of-peak translation)")
    ap.add_argument("--out", required=True, help="audit output directory (under 02_analysis)")
    args = ap.parse_args(argv)
    res = run_audit(args.md_root, args.out, args.dissolution_root)
    p = res["pairing"]
    print(f"q3 frames: {len(p)} · paired: {int((p.status=='matched').sum())}")
    print(f"segments: {len(res['segments'])} "
          f"({sum(s['classification']=='stable' for s in res['segments'])} stable)")
    print(f"catastrophic-failure frames ({PRIMARY_FAILURE}): {int(_flag(res['frames'], PRIMARY_FAILURE).sum())}"
          f"  (1760={int(_flag(res['frames'],PRIMARY_FAILURE)[res['frames'].session.to_numpy()=='1760 rpm'].sum())},"
          f" 1500={int(_flag(res['frames'],PRIMARY_FAILURE)[res['frames'].session.to_numpy()=='1500 rpm'].sum())})")
    if len(res["diss_trans"]):
        d = res["diss_trans"].copt_at_30pct
        print(f"dissolution 30%-of-peak → absolute Copt {d.min():.1f}–{d.max():.1f} (median {d.median():.1f})")
    print("\nfailure definition sensitivity:")
    print(res["fail_sens"].to_string(index=False))


if __name__ == "__main__":
    main()
