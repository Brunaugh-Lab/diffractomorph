"""Concentration-dependent (varying starting Copt) study — loading response.

Two sub-studies varied the **injected volume** of one suspension to set the starting optical
concentration: pH 4.0 (2026-06-23, 25/35/50 uL) and pH 4.5 (2026-07-27, 30/40/50 uL), each with
three nominal Copt levels (12/18/24 %) and three replicates.

**What this design can and cannot support.** Within a sub-study every loading level is an aliquot
of the *same* suspension, verified by a shared QC export. That is a strength for the comparison —
loading is varied with no prep confound — but it means there is **one preparation per sub-study**,
so the three replicates are technical and there is no prep-level error term. The two sub-studies
are two independent preparations, but they differ in pH *and* in prep, so a difference between
them cannot be attributed to either. Reproduction of the same loading response at both pH values
is therefore corroboration; a discrepancy between them is uninterpretable. Nothing here is
prep-balanced in the sense used for Arm B, and none of it should be quoted as an n = 3 result.

pH 4.0 has no UV plates, so it is optical-only; pH 4.5 carries UV and gets the mass check.

**What is tested.**

1. *Is Copt a linear loading proxy?* Starting Copt against delivered dose. A nonlinearity would
   undermine every use of Copt as an area/loading surrogate elsewhere in the pipeline.
2. *Does the dissolved-mass fraction depend on loading?* Transport-limited Nernst-Brunner into a
   finite sink predicts that it must: more solid raises bulk C faster, shrinking the
   ``(Cs - C)`` driving force, so a higher load should dissolve a *smaller fraction* of its dose
   in the same time. This is a falsifiable prediction, not a fitted quantity. **It is a claim
   about MASS, and only the UV mass fraction tests it.**

   *Fractional Copt loss is not that quantity and is not used as confirmation.* Copt is a
   particle-side, surface-weighted optical coordinate, not a dose fraction. It moves the OPPOSITE
   way here — fractional Copt loss RISES with loading (0.58 / 0.72 / 0.78) while the UV mass
   fraction falls. That divergence is not a contradiction, because the two coordinates weight the
   population differently, but it does mean the optical extent cannot stand in for the mass
   prediction and is reported separately.

   **What the offset does to this.** The filter correction is ADDITIVE, so it contributes
   ``offset x dilution x V / dose`` recovery points — about 49, 37 and 30 pp at the three
   loadings, because dose differs. The loading trend therefore depends on it: at zero offset the
   direction REVERSES. :func:`offset_sensitivity` locates the crossover at ~0.59 ug/mL, i.e.
   ~40 % of the calibrated 1.48 ug/mL. The calibration artifact carries **no documented
   uncertainty interval** for that value, so no independently justified range can be tested and
   the result is not called robust.
3. *Does the frozen forward model reproduce the loading response?* Each run is predicted from its
   own delivered dose and the shared starting PSD with the pH-study rate scale frozen — nothing
   refitted here.

   **Claim boundary.** The strongest supportable wording is *an independent-dataset,
   within-preparation loading-response evaluation with the forward-model parameters frozen*.
   It is **not** a preparation-level validation: all three loadings are aliquots of one
   suspension and the replicates are technical, so there is no preparation-level error term and
   nothing here generalises across preparations. Because the UV direction reverses below ~40 % of
   the calibrated filter offset (above), the agreement is stated as *conditional agreement with
   the frozen finite-sink prediction under the established additive filter-recovery correction*
   — not as validation, and not as robust.

   **The starting PSD comes from the shared suspension QC, not the per-run in-cuvette q0.** The
   design injects one suspension at three volumes, and the QC q0 agrees across both sub-studies
   (Dv50 2.61 um at pH 4.0, 2.62 um at pH 4.5), as it should for one material. The per-run q0
   records do not: they scatter 2.5-6.7 um within a single pH 4.5 level and reach 13.7-14.2 um
   for the pH 4.0 24 % runs, at the top of the inversion's reliable range. Taking those at face
   value would attribute a five-fold starting-size difference to a change of injected volume,
   which the shared QC contradicts, and it inverts the model's predicted loading direction. The
   per-run q0 is therefore carried only as a labelled sensitivity
   (``model_pct_end_per_run_q0``) and treated as a q0 reliability artifact.
4. *Is the size path loading-invariant?* q3 at matched optical extent, the same clock-free
   comparison used in :mod:`media_diagnostic_q3` — but **gated on inversion reliability**. Frames
   carrying more than ``TAIL_MAX_PCT`` of q3 mass above 15 um are not size measurements and are
   excluded. This matters here more than anywhere else in the pipeline: at pH 4.0 the coarse tail
   is already 10-97 % at frame 0, because those runs start at only 2.9-9.5 % Copt and the
   inversion has too little signal to work with. The pH 4.0 size path is therefore not reported.
   At pH 4.5 frame 0 is clean (D50 3.24-3.45 um, tail < 0.6 % in every run, confirming the three
   loadings really do share one suspension) but the tail grows to 3-40 % late in the run, so the
   high-extent points are gated away rather than plotted as if they were sizes.

The UV plates are stored without a file extension, which openpyxl rejects by name. They are read
through a scratch copy; **raw files are never renamed or modified**.

Run with the pipeline venv.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diffractomorph_pipeline import psd, solubility
from diffractomorph_pipeline.assay import read_qc, suspension, uv_timecourse
from diffractomorph_pipeline.config import data_root
from diffractomorph_pipeline.forward import PSD

from arm_b_optical import optical_run
from arm_b_provenance import provenance_record, write_provenance
from forward_predict import _predict_run

STUDY_REL = Path("disso_experiments/conc_dependent_disso_study")
PH45_DIR = "Copt Disso Study (pH = 4.5)"
PH40_DIR = "pH = 4.0 changing Copt disso study"
LEVELS = (12, 18, 24)
REPS = (1, 2, 3)
V_ML = 40.0
EXTENT_GRID = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
TAIL_MAX_PCT = 5.0   # frames with more q3 mass than this above 15 um are not size measurements
# Filter-offset sensitivity grid. The calibrated fixed-capacity value is cal.FILTER_OFFSET[4.5]
# = 1.48 ug/mL; because it is ADDITIVE and the dose changes across the loading ladder, it
# contributes a different number of recovery points at each level, so the loading TREND depends
# on it. The grid spans zero to well above the calibrated value.
OFFSET_GRID = np.array([0.0, 0.25, 0.5, 0.59, 0.75, 1.0, 1.11, 1.48, 1.85, 2.22, 2.96])
COLORS = {12: "#2b6cb0", 18: "#dd6b20", 24: "#718096"}
FROZEN_FIT = ("disso_experiments/ph_dependent_dissolution_study/forward_prediction/"
              "scalar_fit/selected_rate_only_fit_summary.csv")


def study_root() -> Path:
    return data_root() / STUDY_REL


def frozen_rate_scale() -> tuple[float, Path, str]:
    """The pH-study rate scale transferred here, with the artifact it came from.

    ``model_pct_end`` is determined by this number, so it is recorded in the provenance sidecar
    and on every per-run row rather than left implicit in the code.
    """
    path = data_root() / FROZEN_FIT
    if not path.exists():
        raise FileNotFoundError(f"frozen rate-scale artifact missing: {path}")
    table = pd.read_csv(path)
    column = "rate_scale_datebalanced"
    if column not in table or len(table) != 1:
        raise ValueError(f"unexpected frozen-fit summary shape: {path}")
    return float(table.loc[0, column]), path, column


def _read_plate_any(path: Path):
    """Read a BioTek export whose filename carries no .xlsx extension, via a scratch copy."""
    from diffractomorph_pipeline.assay import read_plate
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        return read_plate(path)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / (path.name + ".xlsx")
        shutil.copy2(path, copy)
        return read_plate(copy)


def _uv_timecourse_any(path: Path, ph: float, filter_offset_ugml: float | None = None):
    """C(t) for an extensionless plate, mirroring :func:`assay.uv_timecourse` exactly.

    In particular the **per-plate** blanks are used when the export reports them, falling back to
    the packaged globals only when it does not — these plates read 0.052-0.054 at 280 nm against
    a global 0.050, so using the global would bias every concentration upward.
    ``filter_offset_ugml`` overrides the calibrated additive correction for sensitivity work.
    """
    from diffractomorph_pipeline.assay import calibration as cal
    from diffractomorph_pipeline.assay.curve import StandardCurve
    pl = _read_plate_any(path)
    offset = cal.FILTER_OFFSET.get(ph, 0.0) if filter_offset_ugml is None else filter_offset_ugml
    b280 = cal.BLANK[280] if np.isnan(pl.blank280) else pl.blank280
    b490 = cal.BLANK[490] if np.isnan(pl.blank490) else pl.blank490
    c280 = StandardCurve(*cal.CURVES[(ph, 280)]).concentration(pl.a280, blank=b280)
    c490 = StandardCurve(*cal.CURVES[(ph, 490)]).concentration(pl.a490, blank=b490)
    conc = ((c280 + c490) / 2.0 + offset) * cal.DILUTION
    return pd.DataFrame({"time_min": pl.times_min, "conc_ugml": conc,
                         "blank280_used": b280, "blank490_used": b490,
                         "filter_offset_ugml": offset})


def discover(root: Path | None = None) -> pd.DataFrame:
    """All 18 runs across both sub-studies, with volume, QC, PSD source and optical/q3 paths."""
    root = root or study_root()
    rows = []

    p45 = root / PH45_DIR
    qc_psd_45 = p45 / "12_ Copt" / "QC" / "q0 Data"          # one suspension, shared across levels
    qc_psd_40 = root / PH40_DIR / "QC" / "CFZ QC 20260623 q0.rtf"
    vol45 = {12: 30.0, 18: 40.0, 24: 50.0}          # shared injection log, 2026-07-27
    for level in LEVELS:
        cell = p45 / f"{level}_ Copt"
        qc = cell / "QC" / "CFZ_QC_20260727.xlsx"
        for rep in REPS:
            rtf = cell / "Raw Data" / f"CFZ Disso pH = 4.5 Measurement Copt {level}_ Rep {rep}.rtf"
            uv = next((p for p in (cell / "UV Data").iterdir()
                       if re.search(rf"Rep{rep}$", p.name, re.IGNORECASE)), None)
            rows.append({"substudy": "pH 4.5", "ph": 4.5, "prep": "20260727", "level_pct": level,
                         "rep": rep, "run_id": f"pH4.5_Copt{level}_rep{rep}",
                         "injected_uL": vol45[level], "rtf": rtf, "uv_file": uv,
                         "q3": cell / "q3 Data" / f"Rep {rep}",
                         "qc_xlsx": qc, "psd_src": qc_psd_45,
                         "psd_src_per_run": cell / "q0 Data" / f"Rep {rep}"})

    p40 = root / PH40_DIR
    vols = json.loads((p40 / "QC" / "run_volumes.json").read_text())["runs"]
    vol40 = {int(re.search(r"(\d+)", v["condition"]).group(1)): float(v["volume_uL"]) for v in vols}
    for level in LEVELS:
        for rep in REPS:
            rows.append({"substudy": "pH 4.0", "ph": 4.0, "prep": "20260623", "level_pct": level,
                         "rep": rep, "run_id": f"pH4.0_Copt{level}_rep{rep}",
                         "injected_uL": vol40[level],
                         "rtf": p40 / "Raw Data" / f"CFZ Disso pH 4.0 Measurement Copt {level}_ Rep {rep}.rtf",
                         "uv_file": None,
                         "q3": p40 / "q3" / f"CFZ Disso pH = 4.0 Measurement Copt {level}_ Rep {rep} q3.csv",
                         "qc_xlsx": p40 / "QC" / "cfz qc 20260623.xlsx",
                         "psd_src": qc_psd_40,
                         "psd_src_per_run": p40 / "q0" / f"CFZ Disso pH 4.0 Measurement Copt {level}_ Rep {rep} q0.rtf"})

    frame = pd.DataFrame(rows)
    missing = [c for c in ("rtf", "q3", "qc_xlsx") for _, r in frame.iterrows()
               if r[c] is not None and not Path(r[c]).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} expected inputs absent, e.g. {missing[0]}")
    return frame


def _dose_mg(qc_xlsx: Path, volume_uL: float) -> tuple[float, float]:
    reads = read_qc(qc_xlsx)
    abs_bs = float(reads[0].abs_bs)
    conc = float(suspension.suspension_conc_mgml(abs_bs))
    return float(suspension.injected_mass_mg(conc, volume_uL)), conc


def analyse(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-run optical/UV/model summary, per-frame q3, and the matched-extent size table."""
    runs = discover(root)
    cs_model = solubility.load_default()
    rate_scale, rate_scale_path, rate_scale_col = frozen_rate_scale()
    qc_cache, psd_cache = {}, {}

    per_run, frames, matched = [], [], []
    for r in runs.itertuples():
        if r.qc_xlsx not in qc_cache:
            qc_cache[r.qc_xlsx] = _dose_mg(Path(r.qc_xlsx), r.injected_uL)
        dose_mg, susp = _dose_mg(Path(r.qc_xlsx), r.injected_uL)

        opt = optical_run(r.rtf, clean="pipeline+copt")
        copt, tC = opt.copt, opt.t_min
        floor = float(np.mean(copt[-5:]))
        frac_loss = float((copt[0] - floor) / copt[0])
        rel = copt / copt[0]
        thalf = float(np.interp(0.5, rel[::-1], tC[::-1])) if rel.min() < 0.5 else np.nan

        preds = {}
        for tag, src in (("qc_shared", r.psd_src), ("per_run_q0", r.psd_src_per_run)):
            key = str(src)
            if key not in psd_cache:
                psd_cache[key] = PSD.from_sympatec(src)
            spec = {"ph": r.ph, "dose_mg": dose_mg, "psd": psd_cache[key], "cs_ugml": None}
            run, cs_used = _predict_run(spec, cs_model,
                                        float(tC.max() * 60 * rate_scale * 1.05), 400)
            preds[tag] = float(np.interp(tC[-1] * 60.0 * rate_scale, run.t, run.pct_dissolved))
        psd_shared = psd_cache[str(r.psd_src)]
        psd_run = psd_cache[str(r.psd_src_per_run)]

        uv_end = np.nan
        if r.uv_file is not None:
            tc = _uv_timecourse_any(Path(r.uv_file), r.ph)
            uv_end = float(tc["conc_ugml"].iloc[-1] * V_ML / (dose_mg * 1e3) * 100.0)

        per_run.append({"run_id": r.run_id, "substudy": r.substudy, "ph": r.ph, "prep": r.prep,
                        "level_pct": r.level_pct, "rep": r.rep, "injected_uL": r.injected_uL,
                        "susp_mgml": susp, "dose_mg": dose_mg, "cs_ugml": cs_used,
                        "copt0": float(copt[0]), "copt_floor": floor,
                        "copt_frac_loss": frac_loss, "copt_thalf_min": thalf,
                        "frozen_rate_scale": rate_scale,
                        "frozen_rate_scale_source": str(rate_scale_path.name),
                        "frozen_rate_scale_column": rate_scale_col,
                        "model_pct_end": preds["qc_shared"],
                        "model_pct_end_per_run_q0": preds["per_run_q0"],
                        "qc_psd_dv50_um": psd_shared.dv50, "per_run_q0_dv50_um": psd_run.dv50,
                        "q0_dv50_ratio_run_over_qc": psd_run.dv50 / psd_shared.dv50,
                        "uv_pct_injected_end": uv_end})

        traj = psd.read_q3(r.q3)
        xo = np.asarray(traj.grid_um, float)
        cum = np.cumsum(np.asarray(traj.dQ3, float), axis=1) * 100.0
        n = min(len(cum), copt.size)
        g = np.maximum.accumulate(np.clip(1.0 - (copt - floor) / max(copt[0] - floor, 1e-9),
                                          0.0, 1.0))[:n]
        d50, tail = [], []
        for row in cum[:n]:
            xr, cr = psd.restrict_cumulative(xo, row)
            d50.append(psd.q3_percentiles(xr, cr, (50.0,))[0])
            tail.append(psd.q3_tail_fraction(xo, row))
        d50 = np.asarray(d50, float)
        tail = np.asarray(tail, float)
        d50_gated = np.where(tail <= TAIL_MAX_PCT, d50, np.nan)
        frames.append(pd.DataFrame({"run_id": r.run_id, "substudy": r.substudy,
                                    "level_pct": r.level_pct, "rep": r.rep,
                                    "t_min": tC[:n], "extent_g": g, "d50_um": d50,
                                    "tail_pct_above_15um": tail,
                                    "q3_frame_reliable": tail <= TAIL_MAX_PCT}))
        ok = np.isfinite(d50_gated)
        # pH 4.0 is quarantined from size output: its inversion is already degraded at frame 0
        # (coarse tail 10-97 %), so no matched-extent size is emitted. Per-frame reliability
        # diagnostics above are retained precisely so that failure stays visible.
        if r.substudy == "pH 4.5" and ok.sum() >= 3:
            for target in EXTENT_GRID:
                if g[ok].min() <= target <= g[ok].max():
                    matched.append({"run_id": r.run_id, "substudy": r.substudy,
                                    "level_pct": r.level_pct, "rep": r.rep,
                                    "extent_g": float(target),
                                    "d50_um": float(np.interp(target, g[ok], d50_gated[ok])),
                                    "n_reliable_frames": int(ok.sum())})
    return (pd.DataFrame(per_run), pd.concat(frames, ignore_index=True), pd.DataFrame(matched))


def offset_sensitivity(root: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """pH 4.5 terminal recovery at each level across the filter-offset grid.

    The additive offset enters recovery as ``offset x dilution x V / dose``. Dose rises across the
    loading ladder, so the SAME offset adds fewer recovery points at higher load — which is by
    itself enough to manufacture (or erase) a downward loading trend. This locates the offset at
    which the 12 % -> 24 % direction reverses, expressed as a fraction of the calibrated value.
    """
    from diffractomorph_pipeline.assay import calibration as cal
    runs = discover(root)
    r45 = runs[runs["substudy"].eq("pH 4.5")]
    rows = []
    for r in r45.itertuples():
        dose_mg, _ = _dose_mg(Path(r.qc_xlsx), r.injected_uL)
        for offset in OFFSET_GRID:
            tc = _uv_timecourse_any(Path(r.uv_file), r.ph, filter_offset_ugml=float(offset))
            rec = float(tc["conc_ugml"].iloc[-1]) * V_ML / (dose_mg * 1e3) * 100.0
            rows.append({"level_pct": r.level_pct, "rep": r.rep, "offset_ugml": float(offset),
                         "dose_mg": dose_mg, "recovery_pct": rec,
                         "offset_contribution_pp":
                             float(offset) * cal.DILUTION * V_ML / (dose_mg * 1e3) * 100.0})
    table = pd.DataFrame(rows)
    level = (table.groupby(["offset_ugml", "level_pct"], as_index=False)
             .agg(recovery_pct=("recovery_pct", "mean"),
                  offset_contribution_pp=("offset_contribution_pp", "mean"),
                  dose_mg=("dose_mg", "mean")))

    # crossover: solve rec(12) - rec(24) = 0 in the offset, analytically (both are linear in it)
    base = level[np.isclose(level["offset_ugml"], 0.0)].set_index("level_pct")
    d12, d24 = base.loc[12, "dose_mg"] * 1e3, base.loc[24, "dose_mg"] * 1e3
    slope = cal.DILUTION * V_ML * 100.0 * (1.0 / d12 - 1.0 / d24)      # pp per ug/mL
    gap0 = base.loc[12, "recovery_pct"] - base.loc[24, "recovery_pct"]
    crossover = float(-gap0 / slope) if slope else float("nan")
    calibrated = cal.FILTER_OFFSET[4.5]
    meta = {"crossover_offset_ugml": crossover,
            "calibrated_offset_ugml": calibrated,
            "crossover_as_fraction_of_calibrated": crossover / calibrated,
            "direction_above_crossover": "recovery DECLINES with loading",
            "direction_below_crossover": "recovery RISES with loading"}
    return level, meta


def loading_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    """Level means within each sub-study. Replicates are TECHNICAL — sd is not a prep-level term."""
    metrics = ["injected_uL", "dose_mg", "copt0", "copt_floor", "copt_frac_loss",
               "copt_thalf_min", "model_pct_end", "model_pct_end_per_run_q0",
               "qc_psd_dv50_um", "per_run_q0_dv50_um", "uv_pct_injected_end"]
    return (per_run.groupby(["substudy", "ph", "prep", "level_pct"], as_index=False)
            .agg(n_technical_reps=("rep", "nunique"),
                 **{m: (m, "mean") for m in metrics},
                 copt0_sd=("copt0", "std"), frac_loss_sd=("copt_frac_loss", "std")))


def copt_linearity(per_run: pd.DataFrame) -> pd.DataFrame:
    """Regress starting Copt on delivered dose per sub-study — is Copt a linear loading proxy?"""
    rows = []
    for (sub, ph), g in per_run.groupby(["substudy", "ph"]):
        x = g["dose_mg"].to_numpy(float); y = g["copt0"].to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss = float(np.sum((y - y.mean()) ** 2))
        rows.append({"substudy": sub, "ph": ph, "n_runs": len(g),
                     "slope_copt_per_mg": float(slope), "intercept_copt": float(intercept),
                     "r2": float(1 - np.sum((y - pred) ** 2) / ss) if ss > 0 else np.nan,
                     "dose_range_mg": f"{x.min():.4f}-{x.max():.4f}",
                     "copt0_range": f"{y.min():.2f}-{y.max():.2f}"})
    return pd.DataFrame(rows)


def size_path_spread(matched: pd.DataFrame) -> pd.DataFrame:
    """Descriptive only: D50 spread ACROSS loadings vs scatter BETWEEN technical replicates.

    With one preparation there is no error term that licenses an invariance verdict, so this
    simply states how the between-loading spread compares with the replicate scatter at the same
    extent. It is reported, not adjudicated.
    """
    rows = []
    for (sub, g_ext), g in matched.groupby(["substudy", "extent_g"]):
        level_means = g.groupby("level_pct")["d50_um"].mean()
        rep_sd = g.groupby("level_pct")["d50_um"].std().mean()
        rows.append({"substudy": sub, "extent_g": g_ext,
                     "d50_range_across_loadings_um": float(level_means.max() - level_means.min()),
                     "mean_technical_rep_sd_um": float(rep_sd),
                     "range_over_rep_sd": float((level_means.max() - level_means.min()) / rep_sd)
                     if rep_sd else np.nan})
    return pd.DataFrame(rows).sort_values(["substudy", "extent_g"]).reset_index(drop=True)


def _plot(per_run, summary, matched, offsets, meta, path):
    fig, axes = plt.subplots(1, 5, figsize=(26, 4.8))
    subs = list(dict.fromkeys(per_run["substudy"]))
    marks = {"pH 4.5": "o", "pH 4.0": "s"}

    ax = axes[0]
    for sub in subs:
        g = per_run[per_run["substudy"].eq(sub)]
        ax.scatter(g["dose_mg"], g["copt0"], marker=marks[sub], s=36,
                   c=[COLORS[l] for l in g["level_pct"]], label=sub,
                   edgecolors="k", linewidths=0.4)
        x = np.linspace(g["dose_mg"].min(), g["dose_mg"].max(), 10)
        s, i = np.polyfit(g["dose_mg"], g["copt0"], 1)
        ax.plot(x, s * x + i, "--", lw=1, color="0.4")
    ax.set_xlabel("delivered dose (mg)"); ax.set_ylabel("starting Copt (%)")
    ax.set_title("Is Copt a linear loading proxy?", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    for sub in subs:
        s = summary[summary["substudy"].eq(sub)]
        ax.errorbar(s["copt0"], s["copt_frac_loss"], yerr=s["frac_loss_sd"], fmt=marks[sub] + "-",
                    capsize=3, lw=1.6, ms=6, label=sub)
    ax.set_xlabel("starting Copt (%)"); ax.set_ylabel("fractional Copt loss")
    ax.set_title("Optical extent vs loading — RISES with load,\n"
                 "opposite to the UV mass fraction (not the mass test)", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    for sub in subs:
        s = summary[summary["substudy"].eq(sub)]
        ax.plot(s["copt0"], s["model_pct_end"], marks[sub] + "--", lw=1.4, ms=6,
                label=f"{sub} model")
        if s["uv_pct_injected_end"].notna().any():
            ax.plot(s["copt0"], s["uv_pct_injected_end"], marks[sub] + "-", lw=2.0, ms=6,
                    label=f"{sub} UV")
    ax.set_xlabel("starting Copt (%)"); ax.set_ylabel("% of dose dissolved at end")
    ax.set_title("MASS: frozen model vs UV across loading\n"
                 "(nothing refitted; UV conditional on the filter offset)", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=7, frameon=False)

    ax = axes[3]
    for sub in sorted(matched["substudy"].unique()):        # pH 4.0 is quarantined, so absent
        m = matched[matched["substudy"].eq(sub)]
        agg = m.groupby(["level_pct", "extent_g"], as_index=False)["d50_um"].mean()
        for level in LEVELS:
            g = agg[agg["level_pct"].eq(level)].sort_values("extent_g")
            ax.plot(g["extent_g"], g["d50_um"], marks[sub] + "-", color=COLORS[level],
                    lw=1.4, ms=4, alpha=0.85, label=f"{sub} {level}%")
    ax.axhline(psd.VALID_SIZE_MAX_UM, color="crimson", ls="--", lw=1)
    ax.text(0.21, psd.VALID_SIZE_MAX_UM * 0.94, "inversion reliability ceiling",
            fontsize=7, color="crimson")
    ax.set_xlabel("optical extent g"); ax.set_ylabel("q3 D50 at matched extent (µm)")
    spread = size_path_spread(matched)
    rng = spread["d50_range_across_loadings_um"].max()
    ratio = spread["range_over_rep_sd"].median()
    ax.set_title(f"pH 4.5 size path vs loading (descriptive)\n"
                 f"max across-loading range {rng:.2f} µm ≈ {ratio:.1f}× replicate SD; "
                 f"single prep, model-inverted", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=6, frameon=False, ncol=2)

    ax = axes[4]
    for level in LEVELS:
        g = offsets[offsets["level_pct"].eq(level)].sort_values("offset_ugml")
        ax.plot(g["offset_ugml"], g["recovery_pct"], "o-", color=COLORS[level], lw=1.6, ms=4,
                label=f"{level}% Copt")
    ax.axvline(meta["calibrated_offset_ugml"], color="k", ls="-", lw=1.2)
    ax.axvline(meta["crossover_offset_ugml"], color="crimson", ls="--", lw=1.2)
    ax.axhline(100.0, color="0.6", ls=":", lw=0.9)
    ax.text(meta["calibrated_offset_ugml"], ax.get_ylim()[1] * 0.97, " calibrated", fontsize=7,
            rotation=90, va="top")
    ax.text(meta["crossover_offset_ugml"], ax.get_ylim()[1] * 0.97, " direction flips",
            fontsize=7, rotation=90, va="top", color="crimson")
    ax.set_xlabel("additive filter offset (µg/mL)")
    ax.set_ylabel("terminal recovery (% of dose)")
    ax.set_title(f"UV trend is CONDITIONAL on the offset\n"
                 f"direction reverses below {meta['crossover_offset_ugml']:.2f} µg/mL "
                 f"({meta['crossover_as_fraction_of_calibrated']:.0%} of calibrated)", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Concentration-dependent (varying starting Copt) study — one suspension prep per "
                 "sub-study; replicates are technical, pH is confounded with prep", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Varying-starting-Copt loading response.")
    p.add_argument("--study-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    root = args.study_root or study_root()
    out = args.output_dir or root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    per_run, frames, matched = analyse(root)
    summary = loading_summary(per_run)
    lin = copt_linearity(per_run)
    offsets, meta = offset_sensitivity(root)
    rate_scale, rate_scale_path, rate_scale_col = frozen_rate_scale()
    per_run.to_csv(out / "copt_loading_runs.csv", index=False)
    summary.to_csv(out / "copt_loading_level_means.csv", index=False)
    lin.to_csv(out / "copt_loading_linearity.csv", index=False)
    frames.to_csv(out / "copt_loading_q3_frames.csv", index=False)
    matched.to_csv(out / "copt_loading_q3_matched_extent.csv", index=False)
    offsets.to_csv(out / "copt_loading_filter_offset_sensitivity.csv", index=False)
    size_path_spread(matched).to_csv(out / "copt_loading_q3_size_path_spread.csv", index=False)
    _plot(per_run, summary, matched, offsets, meta, out / "copt_loading.png")
    write_provenance(out / "provenance.json",
                     provenance_record("copt_loading", study_root=root, optical_cleaned=True,
                                       replication="ONE suspension prep per sub-study; "
                                                   "replicates are technical; pH confounded "
                                                   "with prep between sub-studies",
                                       frozen_model={
                                           "rate_scale": rate_scale,
                                           "column": rate_scale_col,
                                           "source_file": str(rate_scale_path),
                                           "source_relative_to_data_root": FROZEN_FIT,
                                           "refitted_here": False,
                                           "determines": ["model_pct_end",
                                                          "model_pct_end_per_run_q0"]},
                                       filter_offset_sensitivity=meta,
                                       claim="independent-dataset, within-preparation "
                                             "loading-response evaluation with forward-model "
                                             "parameters frozen; NOT a preparation-level "
                                             "validation; UV direction conditional on the "
                                             "additive filter-recovery correction",
                                       ph40_size_output="excluded (q3 unreliable at frame 0); "
                                                        "per-frame diagnostics retained"))

    print(f"{len(per_run)} runs, {per_run['substudy'].nunique()} sub-studies "
          f"(one prep each — replicates are technical)")
    print(f"frozen rate scale {rate_scale:.4f} from {rate_scale_path.name} "
          f"[{rate_scale_col}] — not refitted here\n")
    print(summary[["substudy", "level_pct", "injected_uL", "dose_mg", "copt0", "copt0_sd",
                   "copt_frac_loss", "copt_thalf_min", "model_pct_end",
                   "uv_pct_injected_end"]].round(3).to_string(index=False), "\n")
    rel = frames.groupby("substudy")["q3_frame_reliable"].mean()
    print("q3 frames passing the inversion-reliability gate (tail <= "
          f"{TAIL_MAX_PCT:g}% above 15 um): "
          + ", ".join(f"{k} {v:.0%}" for k, v in rel.items()))
    print("Copt linearity in delivered dose:")
    print(lin.round(4).to_string(index=False))
    bad = per_run[per_run["q0_dv50_ratio_run_over_qc"] > 2.0]
    print(f"\nper-run q0 vs shared suspension QC: {len(bad)}/{len(per_run)} runs record a "
          f"starting Dv50 more than 2x the QC value (max {per_run['q0_dv50_ratio_run_over_qc'].max():.1f}x)."
          f"\nThe QC q0 agrees across sub-studies ({per_run.groupby('substudy')['qc_psd_dv50_um'].first().round(2).to_dict()}),"
          f" so those are treated as q0 reliability artifacts, not material differences.")
    print("\nfilter-offset sensitivity (pH 4.5 terminal recovery, % of dose):")
    print(offsets.pivot(index="offset_ugml", columns="level_pct",
                        values="recovery_pct").round(1).to_string())
    print(f"\n  direction reverses at {meta['crossover_offset_ugml']:.2f} µg/mL = "
          f"{meta['crossover_as_fraction_of_calibrated']:.0%} of the calibrated "
          f"{meta['calibrated_offset_ugml']:.2f}; above it recovery declines with loading, "
          f"below it recovery rises.")
    print("  The calibration artifact carries no uncertainty interval for the offset, so no "
          "independently justified\n  range can be tested — the UV loading trend is reported as "
          "CONDITIONAL, not robust.")
    sp = size_path_spread(matched)
    print(f"\npH 4.5 size path, descriptive: across-loading D50 range "
          f"{sp['d50_range_across_loadings_um'].min():.2f}-{sp['d50_range_across_loadings_um'].max():.2f} µm "
          f"vs mean technical-replicate SD {sp['mean_technical_rep_sd_um'].mean():.2f} µm "
          f"(median ratio {sp['range_over_rep_sd'].median():.1f}×). Reported, not adjudicated: "
          f"one preparation gives no error term for an invariance verdict.")
    print(f"\nmatched-extent size output covers {sorted(matched['substudy'].unique())} only "
          f"(pH 4.0 quarantined; its per-frame q3 diagnostics are retained).")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
