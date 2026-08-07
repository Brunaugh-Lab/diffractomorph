"""Matched nondissolving control: pH-7 empirical noise/drift null vs pH 4/4.5/5 dissolution.

Companion diagnostic to the Objective-1 operator-feasibility figure (kept SEPARATE from it). The
clofazimine pH-7 titration experiments are the same drug particles in antisolvent under
**nondissolving** conditions, so on their stable (constant-Copt) plateau segments every
frame-to-frame channel change is pure instrument/sampling noise and slow particle-bumping drift — a
*physically matched null distribution*. This driver contrasts that empirical null with the pH 4.0 /
4.5 / 5.0 dissolution channel-trajectories.

Metric — **sustained matched-window drift.** For each run and detector channel, on the
reference-adjusted signal ``X_c(t) = I_c(t) − ref_c``, the largest sustained change over any
``W``-minute window is ``drift_c = max |median(late half) − median(early half)|`` (with the sign of
that change: falling = dissolution-like). The pH-7 null is measured only on the run's stable Copt
plateaus (``detect_plateaus`` — the same segmentation the noise surface is built from), so the
titration ramps do NOT contaminate the null; the dissolution drift is measured over the run. Drift is
reported in absolute detector units and normalized by the CFZ-pH-7 surface's per-channel noise σ
(``drift / σ`` = how many empirical-noise-floor units the channel moved).

Why not the windowed-trend *z*-test (``NoiseSurface.is_significant``)? Its ``z = |slope|/SE``
saturates for any sustained change (SE shrinks ∝ 1/√N against a ~0.01-count per-frame σ) and, applied
to the whole pH-7 titration, registers the intentional Copt ramps as enormous "drift" — making the
null look as active as the signal. The matched-window drift magnitude on stable segments is the
correct null-vs-signal contrast.

Scientific boundary (also written into the JSON):
  * pH-7 is the empirical NULL (matched nondissolving particles) — it is NOT asserted that every pH-7
    channel is stationary (the null distribution has a nonzero tail).
  * The only intended conclusion is that dissolution-associated channel trajectories GREATLY exceed
    the noise and drift measured under matched nondissolving conditions.
  * The late low-angle CFZ signal is NOT taken as evidence of aggregation, precipitation, or physical
    coarsening.

Outputs (into ``--output-dir``): ``objective1_noise_control_summary.json``,
``objective1_noise_control_per_run.csv`` (per-run, per-channel drift / drift-to-noise / sign), and an
optional diagnostic PNG. Kept out of the canonical two-panel grant figure.

Run with the pipeline venv::

    python objective1_noise_control.py [--output-dir DIR] [--window-min W] [--no-figure]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diffractomorph_pipeline import ingest
from diffractomorph_pipeline.config import data_root
from diffractomorph_pipeline.noise import detect_plateaus
from diffractomorph_pipeline.noise_surface import load_surface

STUDY = data_root() / "disso_experiments" / "ph_dependent_dissolution_study"
NOISE_FLOOR = STUDY / "noise_floor"
DRUG_WINDOW_CH = (15, 31)          # large-angle channels carrying the CFZ dissolution signal
WINDOW_MIN = 2.0                   # matched sustained-drift window (min)
PLATEAU_TOLERANCE = 0.6            # Copt-plateau segmentation (matches the noise-surface build)
MIN_STABLE_FRAMES = 20
# A caller should supply the destination. Keeping the fallback inside the study root avoids
# import-time assumptions about how many parents an arbitrary clean-room data root has.
DEFAULT_OUTPUT_DIR = STUDY / "analysis" / "noise_control"


# ── discovery ────────────────────────────────────────────────────────────────────────────────────

def find_ph7_null_runs() -> list[tuple[str, Path]]:
    """The 3 pH-7 CFZ *measurement* reps (the matched nondissolving null), by explicit path.

    Under ``noise_floor/Day N - <date>/``, named ``CFZ Titration pH = 7.0 measurement Rep N.rtf`` —
    outside the ``ph_<x>/<date>_pH*`` layout the pH-study locators expect, so globbed directly. The
    paired particle-free ``blank`` RTFs are excluded.
    """
    hits = sorted(glob.glob(str(NOISE_FLOOR / "**" / "*measurement*Rep*.rtf"), recursive=True))
    return [(f"pH7_null::{Path(h).stem}", Path(h)) for h in hits]


def find_dissolution_runs() -> list[tuple[float, int, int, Path]]:
    """Every pH 4.0/4.5/5.0 dissolution measurement run: ``(ph, date_i, rep, rtf)`` via ``iter_runs``."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from psd_evolution_common import iter_runs

    return [(float(ph), int(date), int(rep), Path(rtf)) for ph, date, rep, rtf, _q3 in iter_runs()]


# ── sustained matched-window drift (pure) ──────────────────────────────────────────────────────

def _segment_windows(t: np.ndarray, idx: np.ndarray, w_min: float):
    """Yield frame-index sub-arrays spanning ≥ ``w_min`` minutes within a contiguous segment ``idx``."""
    idx = np.asarray(idx)
    if idx.size < 6:
        return
    ts = t[idx]
    for a in range(idx.size):
        for b in range(a + 5, idx.size):
            if ts[b] - ts[a] >= w_min:
                yield idx[a:b + 1]
                break


def max_window_drift(X: np.ndarray, t: np.ndarray, segments: list[np.ndarray],
                     w_min: float = WINDOW_MIN) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel largest sustained ``w_min``-window change and its sign.

    ``drift_c = max over windows |median(late half) − median(early half)|`` on the reference-adjusted
    signal ``X`` (T×C); ``sign_c`` is the direction of that maximal change (−1 falling, +1 rising).
    Windows are confined to the given ``segments`` (whole run for dissolution; stable Copt plateaus
    for the pH-7 null).
    """
    C = X.shape[1]
    drift = np.zeros(C)
    sign = np.zeros(C)
    for seg in segments:
        for win in _segment_windows(t, seg, w_min):
            h = len(win) // 2
            d = np.median(X[win[h:]], 0) - np.median(X[win[:h]], 0)
            m = np.abs(d) > np.abs(drift)
            drift[m] = np.abs(d)[m]
            sign[m] = np.sign(d)[m]
    return drift, sign


def run_channel_drift(surface, run, ph7: bool, w_min: float = WINDOW_MIN) -> dict:
    """Per-channel sustained drift (abs + drift/σ + sign) for one run.

    For the pH-7 null the drift is measured only on stable Copt plateaus (nondissolving); for a
    dissolution run it is measured over the whole run. ``drift/σ`` uses the CFZ-pH-7 surface σ at each
    channel's own signal level — the number of empirical-noise-floor units the channel moved.
    """
    I = np.asarray(run.I, float)
    ref = np.asarray(run.ref, float)
    X = I - ref[None, :]
    t = np.asarray(run.t_min, float)
    if ph7:
        copt = np.nan_to_num(np.asarray(run.copt, float), nan=1e3)
        pls = detect_plateaus(copt, t, tolerance=PLATEAU_TOLERANCE, min_stable_frames=MIN_STABLE_FRAMES)
        segments = [p.indices for p in pls]
    else:
        segments = [np.arange(I.shape[0])]

    drift, sign = max_window_drift(X, t, segments, w_min)
    sigma = surface.sigma(np.clip(np.abs(X).mean(0), 1e-6, None))
    d_over_sigma = drift / np.clip(sigma, 1e-9, None)
    ch = np.asarray(run.channels, int)
    drug = (ch >= DRUG_WINDOW_CH[0]) & (ch <= DRUG_WINDOW_CH[1])
    return dict(channels=ch, drift=drift, drift_over_sigma=d_over_sigma, sign=sign,
                n_segments=len(segments),
                drift_drug_median=float(np.median(drift[drug])),
                d_over_sigma_drug_median=float(np.median(d_over_sigma[drug])),
                frac_drug_falling=float(np.mean(sign[drug] < 0)))


def summarize_group(per_run: list[dict], label: str) -> dict:
    """Pool per-run per-channel drift across a condition into a null-or-signal summary.

    Runs are nested within preparation dates; channels×runs are pooled for this descriptive magnitude
    contrast (not treated as independent for any inferential claim)."""
    if not per_run:
        return dict(label=label, n_runs=0)
    drift = np.concatenate([r["drift"] for r in per_run])
    dos = np.concatenate([r["drift_over_sigma"] for r in per_run])
    return dict(
        label=label, n_runs=len(per_run), n_channel_obs=int(drift.size),
        drift_abs_median=round(float(np.median(drift)), 4),
        drift_abs_p95=round(float(np.percentile(drift, 95)), 4),
        drift_abs_max=round(float(drift.max()), 4),
        drift_over_sigma_median=round(float(np.median(dos)), 2),
        drift_over_sigma_p95=round(float(np.percentile(dos, 95)), 2),
        drift_over_sigma_max=round(float(dos.max()), 2),
        drug_window_drift_abs_median=round(float(np.median([r["drift_drug_median"] for r in per_run])), 4),
        drug_window_frac_falling_mean=round(float(np.mean([r["frac_drug_falling"] for r in per_run])), 3),
    )


# ── driver ─────────────────────────────────────────────────────────────────────────────────────

def compute_control(w_min: float = WINDOW_MIN) -> dict:
    surface = load_surface(None)                      # packaged cfz_ph7_surface.json
    null_runs = find_ph7_null_runs()
    disso_runs = find_dissolution_runs()
    if not null_runs:
        raise FileNotFoundError(f"No pH-7 null measurement RTFs under {NOISE_FLOOR}")

    rows = []
    per_group: dict[str, list[dict]] = {"pH7_null": []}

    def _record(cond, ph, date, rep, path, r):
        for c, d, dos, s in zip(r["channels"], r["drift"], r["drift_over_sigma"], r["sign"]):
            rows.append(dict(condition=cond, ph=ph, date=date, rep=rep, source=path.name,
                             channel=int(c), drift_abs=round(float(d), 5),
                             drift_over_sigma=round(float(dos), 3), sign=int(s)))

    for label, path in null_runs:
        r = run_channel_drift(surface, ingest.extract_run(path), ph7=True, w_min=w_min)
        per_group["pH7_null"].append(r)
        _record("pH7_null", 7.0, "", "", path, r)
    for ph, date, rep, path in disso_runs:
        key = f"pH{ph:g}"
        per_group.setdefault(key, [])
        r = run_channel_drift(surface, ingest.extract_run(path), ph7=False, w_min=w_min)
        per_group[key].append(r)
        _record(key, ph, date, rep, path, r)

    order = ["pH7_null", "pH4", "pH4.5", "pH5"]
    summaries = {k: summarize_group(per_group.get(k, []), k) for k in order if k in per_group}

    null = summaries["pH7_null"]
    contrast = {}
    for k, s in summaries.items():
        if k == "pH7_null" or s.get("n_runs", 0) == 0:
            continue
        contrast[k] = dict(
            drug_drift_over_null_p95=round(s["drug_window_drift_abs_median"] / null["drift_abs_p95"], 1)
            if null["drift_abs_p95"] else None,
            drug_drift_over_null_max=round(s["drug_window_drift_abs_median"] / null["drift_abs_max"], 2)
            if null["drift_abs_max"] else None,
            drug_d_over_sigma_median=s["drug_window_frac_falling_mean"] and round(
                float(np.median([r["d_over_sigma_drug_median"] for r in per_group[k]])), 1),
        )

    return dict(
        window_min=w_min,
        summaries=summaries,
        dissolution_vs_null=contrast,
        per_run_table=pd.DataFrame(rows),
        provenance=dict(
            noise_surface="src/diffractomorph_pipeline/data/noise/cfz_ph7_surface.json (packaged; "
                          "built from the 3 CFZ pH-7 measurement reps' stable Copt plateaus)",
            ph7_null_runs=[str(p) for _, p in null_runs],
            n_dissolution_runs=len(disso_runs),
            dissolution_run_root=str(STUDY),
            metric="max sustained W-min drift = max_window |median(late half) − median(early half)| of "
                   "the reference-adjusted channel signal; pH-7 restricted to stable Copt plateaus; "
                   "drift/sigma normalized by the CFZ-pH-7 noise surface",
            drug_window_channels=list(DRUG_WINDOW_CH),
        ),
        boundary=[
            "pH-7 is the empirical NULL (matched nondissolving particles) measured on stable Copt "
            "plateaus; it is NOT asserted that every pH-7 channel is stationary — the null has a "
            "nonzero tail.",
            "Intended conclusion: dissolution-associated channel trajectories greatly exceed the "
            "noise and drift measured under matched nondissolving conditions.",
            "The late low-angle CFZ signal is NOT taken as evidence of aggregation, precipitation, "
            "or physical coarsening.",
            "This is a descriptive magnitude contrast (null band vs signal band), not a per-channel "
            "stationarity test.",
        ],
    )


def render_diagnostic(result: dict, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from diffractomorph_pipeline import plot_styles as ps

    ps.apply_manuscript_style()
    plt.rcParams.update({"font.size": 8, "font.weight": "normal", "axes.labelweight": "normal",
                         "axes.titleweight": "normal", "axes.grid": False})
    df = result["per_run_table"]
    order = [k for k in ["pH7_null", "pH4", "pH4.5", "pH5"] if k in df.condition.unique()]
    colors = {"pH7_null": "#2C2C2A", "pH4": "#0072B2", "pH4.5": "#E69F00", "pH5": "#CC79A7"}
    labels = {"pH7_null": "pH 7 (null)", "pH4": "pH 4.0", "pH4.5": "pH 4.5", "pH5": "pH 5.0"}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.15))
    fig.subplots_adjust(left=0.1, right=0.985, bottom=0.155, top=0.94, wspace=0.36)

    # left: per-channel mean drift (over runs) vs channel
    for k in order:
        piv = df[df.condition == k].groupby("channel").drift_abs.mean()
        a1.plot(piv.index, piv.values, "-o", ms=2.6, lw=1.1, color=colors[k], label=labels[k])
    a1.set_yscale("log")
    a1.axvspan(DRUG_WINDOW_CH[0], DRUG_WINDOW_CH[1], color="0.85", alpha=0.5, zorder=0)
    a1.set_xlabel("detector channel")
    a1.set_ylabel(f"sustained {result['window_min']:g}-min drift (a.u.)")
    a1.legend(loc="upper left"); a1.spines["top"].set_visible(False); a1.spines["right"].set_visible(False)

    # right: pooled drift/sigma distribution per condition (strip + median bar)
    rng = np.random.default_rng(0)
    for i, k in enumerate(order):
        y = np.clip(df[df.condition == k].drift_over_sigma.values, 1e-1, None)
        x = i + (rng.random(y.size) - 0.5) * 0.5
        a2.scatter(x, y, s=5, color=colors[k], alpha=0.35, edgecolors="none")
        a2.plot([i - 0.32, i + 0.32], [np.median(y)] * 2, color=colors[k], lw=2.4)
    a2.set_yscale("log")
    a2.set_xticks(range(len(order))); a2.set_xticklabels([labels[k] for k in order])
    a2.set_ylabel("drift / noise-σ  (per channel, pooled)")
    a2.spines["top"].set_visible(False); a2.spines["right"].set_visible(False)

    for ax, letter in ((a1, "A"), (a2, "B")):
        ax.text(-0.215, 1.03, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="bottom", ha="right")
    png = out_dir / "objective1_noise_control_diagnostic.png"
    fig.savefig(png, dpi=300)
    fig.savefig(out_dir / "objective1_noise_control_diagnostic.pdf")
    plt.close(fig)
    return png


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="pH-7 empirical noise/drift null vs pH 4/4.5/5 dissolution channel-trajectories.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--window-min", type=float, default=WINDOW_MIN)
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[noise-control] measuring sustained channel drift: pH-7 null (stable plateaus) vs pH 4/4.5/5 ...")
    result = compute_control(w_min=args.window_min)

    df = result.pop("per_run_table")
    csv_path = args.output_dir / "objective1_noise_control_per_run.csv"
    df.to_csv(csv_path, index=False)
    json_path = args.output_dir / "objective1_noise_control_summary.json"
    json_path.write_text(json.dumps(result, indent=2))

    for k, s in result["summaries"].items():
        if s.get("n_runs", 0) == 0:
            continue
        print(f"  {s['label']:>10}: n_runs={s['n_runs']:>2}  drift(a.u.) med={s['drift_abs_median']} "
              f"p95={s['drift_abs_p95']}  drug_med={s['drug_window_drift_abs_median']}  "
              f"drift/σ med={s['drift_over_sigma_median']}  drug_falling={s['drug_window_frac_falling_mean']}")
    print("  dissolution vs null:", result["dissolution_vs_null"])
    print(f"[noise-control] wrote {json_path}\n[noise-control] wrote {csv_path}")
    result["per_run_table"] = df
    if not args.no_figure:
        print(f"[noise-control] wrote {render_diagnostic(result, args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
