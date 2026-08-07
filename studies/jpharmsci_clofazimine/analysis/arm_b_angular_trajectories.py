"""Frozen export of the Arm B aggregate angular-signal trajectories, ΣI(t).

The empirical KWW analysis (:mod:`psd_angular_fit`) publishes the FITTED endpoints — τ, β, the
optical decay depth, the back-extrapolated start ``i0`` — but not the measured trajectories those
fits were made to. The main dissolution-medium figure draws the trajectories, so this module
exports them once, as an authoritative artifact with its own provenance, and the figure renderer
then reads only that export. The renderer never re-ingests a measurement file.

**The signal definition is psd_angular_fit's, not a new one.** Per run:

  1. :func:`ingest.extract_run` on the PAQXOS export (this is also what sets ``t_min``: the
     acquisition clock re-zeroed to the first retained frame);
  2. ``kinetics.total_signal`` — the raw per-frame sum of ``Measured Value`` over the 31 rings,
     **not** background-subtracted, so the non-drug ``Σref`` floor is still in it;
  3. ``kinetics.despike_upward`` — the same upward-spike handling;
  4. ``kinetics.fit_signal`` — the free-amplitude KWW, used ONLY for its back-extrapolated start
     ``i0``, which normalises the trajectory. ``i0`` falls back to the first measured frame when
     the fit does not return a finite positive value, exactly as upstream.

:func:`validate` re-derives τ and ``i0`` for all 27 runs and checks them against
``angular_kww_fits.csv``; if this module's signal path ever drifts from the published fits, the
export fails rather than emitting trajectories that disagree with the endpoints beside them.

**Aggregation is preparation-first.** A fresh suspension preparation (date) is the independent
block and the runs inside one are nested technical replicates. Technical runs are averaged within
preparation, then the four preparations per condition are weighted EQUALLY; the reported spread is
the between-preparation SD. Runs are interpolated onto one shared 0.2-min grid first — coarser
than the ~0.19-min acquisition cadence, so no resolution is invented, and bounded by the shortest
run, so nothing is extrapolated.

What this quantity is NOT: dissolved drug mass. ΣI is particle-side angular scattering and carries
the ``Σref`` floor; the percentage is of each run's own fitted starting signal.

Writes ``<arm_b>/analysis/angular_trajectories/``:
  - ``arm_b_angular_trajectories_runs.csv``       — one row per run per grid point
  - ``arm_b_angular_trajectories_preps.csv``      — technical runs averaged within preparation
  - ``arm_b_angular_trajectories_conditions.csv`` — preparations weighted equally, with SD
  - ``provenance.json``

Run with the pipeline venv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from diffractomorph_pipeline import ingest, kinetics

from arm_b_provenance import provenance_record, write_provenance
from forward_predict import _arm_b_runs
from psd_angular_fit import _rtf_for

OUT_SUB = Path("analysis/angular_trajectories")
FITS_REL = Path("analysis/empirical_angular_kww/angular_kww_fits.csv")
CONDITIONS = ("0.5x CMC", "1.0x CMC", "10x CMC")

# One shared grid, coarser than the ~0.183-0.200 min acquisition cadence so no resolution is
# invented, and stopping short of the shortest run (20.98 min) so nothing is extrapolated.
GRID_STEP_MIN = 0.2
GRID_END_MIN = 20.8
TIME_GRID = np.round(np.arange(0.0, GRID_END_MIN + 1e-9, GRID_STEP_MIN), 4)

SIGNAL_DEFINITION = (
    "raw measured ΣI — per-frame sum of Measured Value over the 31 rings, NOT background "
    "subtracted (the non-drug Σref floor remains), upward-despiked, on ingest's re-zeroed t_min"
)
NORMALIZATION = (
    "percent of the run's own KWW back-extrapolated starting signal i0 (kinetics.fit_signal); "
    "falls back to the first measured frame when the fit returns no finite positive i0"
)
# What was and was not done to the signal. Recorded explicitly because "cleaned / not cleaned" is
# too coarse to be true here: an upward-spike filter WAS applied, a background subtraction was
# NOT, and the two are different operations on different parts of the signal.
SIGNAL_CONDITIONING = {
    "upward_despiking": {
        "applied": True,
        "how": "kinetics.despike_upward on ΣI(t) — the same upward-spike handling psd_angular_fit "
               "used for the published KWW fits",
        "removes": "transient upward excursions of individual frames",
    },
    "background_subtraction": {
        "applied": False,
        "why_not": "ΣI deliberately retains the non-drug Σref floor; the free KWW plateau absorbs "
                   "that additive offset, which is why the mean relaxation time and β are the "
                   "prespecified endpoints. The *_bgsub sensitivity columns are not used.",
        "would_remove": "a constant additive offset across all frames",
    },
    "distinction": "despiking removes transient per-frame excursions; background subtraction "
                   "would remove a constant offset. Applying the first says nothing about the "
                   "second, and this artifact applies only the first.",
    "per_frame_channel_cleaning": {
        "applied": False,
        "note": "the angular sum is taken over all 31 rings of the raw export; no per-channel "
                "noise-surface admission is applied to it",
    },
}

AGGREGATION = {
    "level_1": "technical runs averaged within preparation (suspension date)",
    "level_2": "the four preparations per condition weighted EQUALLY",
    "spread": "between-preparation SD (ddof=1) across the four preparations",
    "independent_unit": "suspension preparation",
    "technical_runs_are_not_independent_replicates": True,
}


def arm_b_root() -> Path:
    return Path(_arm_b_runs()[1])


def _run_trajectory(rd: dict, base) -> dict:
    """One run's ΣI(t), its normalisation, and the fit values used to cross-check the path."""
    rtf = _rtf_for(rd, "arm_b", base)
    if not rtf:
        raise FileNotFoundError(f"no measurement export found for run {rd['id']!r}")
    mrun = ingest.extract_run(rtf)
    t, sig = kinetics.despike_upward(mrun.t_min, kinetics.total_signal(mrun.I))
    fit = kinetics.fit_signal(t, sig)
    i0 = fit["i0"] if np.isfinite(fit["i0"]) and fit["i0"] > 0 else float(sig[0])
    t = np.asarray(t, float)
    sig = np.asarray(sig, float)
    if TIME_GRID.max() > t.max():
        raise ValueError(f"run {rd['id']!r} ends at {t.max():.3f} min, before the shared grid "
                         f"ends at {TIME_GRID.max():.3f} min — the grid would extrapolate")
    return {"run_id": rd["id"], "condition": str(rd["cond"]), "prep": str(rd["date"]),
            "rep": int(rd["rep"]), "i0_fit": float(i0), "tau_min": float(fit["tau_min"]),
            "t_min": t, "sigma_i": sig,
            "pct_of_i0": np.interp(TIME_GRID, t, sig) / i0 * 100.0,
            "i0_source": "kww_back_extrapolated" if np.isfinite(fit["i0"]) and fit["i0"] > 0
                         else "first_measured_frame"}


def build(base: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Per-run, per-preparation and per-condition trajectories on the shared grid."""
    runs, discovered, _ = _arm_b_runs()
    base = Path(base or discovered)
    trajectories = [_run_trajectory(rd, base) for rd in runs]

    run_rows = []
    for tr in trajectories:
        for t, pct in zip(TIME_GRID, tr["pct_of_i0"]):
            run_rows.append({"run_id": tr["run_id"], "condition": tr["condition"],
                             "prep": tr["prep"], "rep": tr["rep"], "t_min": float(t),
                             "pct_of_i0_fit": float(pct), "i0_fit": tr["i0_fit"]})
    per_run = pd.DataFrame(run_rows)

    per_prep = (per_run.groupby(["condition", "prep", "t_min"], as_index=False)
                .agg(pct_of_i0_fit=("pct_of_i0_fit", "mean"),
                     n_technical_reps=("run_id", "nunique")))
    per_condition = (per_prep.groupby(["condition", "t_min"], as_index=False)
                     .agg(pct_mean=("pct_of_i0_fit", "mean"),
                          pct_sd_between_preps=("pct_of_i0_fit", "std"),
                          n_preps=("prep", "nunique")))

    meta = {
        "n_runs": len(trajectories),
        "n_preps_per_condition": {c: int(per_prep[per_prep["condition"].eq(c)]["prep"].nunique())
                                  for c in CONDITIONS},
        "technical_reps_per_prep": sorted(
            int(v) for v in per_prep["n_technical_reps"].unique()),
        "time_grid_min": {"start": float(TIME_GRID.min()), "end": float(TIME_GRID.max()),
                          "step": GRID_STEP_MIN, "n_points": int(TIME_GRID.size)},
        "acquisition_cadence_min": "0.183-0.200 (grid is coarser; no resolution invented)",
        "shortest_run_min": float(min(float(tr["t_min"].max()) for tr in trajectories)),
        "signal_definition": SIGNAL_DEFINITION,
        "normalization": NORMALIZATION,
        "aggregation": AGGREGATION,
        "i0_sources": sorted({tr["i0_source"] for tr in trajectories}),
        "is_dissolved_mass": False,
        "quantity": "particle-side angular scattering, not undissolved drug mass",
    }
    return per_run, per_prep, per_condition, meta


def validate(per_run: pd.DataFrame, meta: dict, base: Path | None = None) -> dict:
    """The reconstructed signal path must reproduce the published KWW fits, run for run."""
    runs, discovered, _ = _arm_b_runs()
    base = Path(base or discovered)
    fits_path = base / FITS_REL
    if not fits_path.exists():
        raise FileNotFoundError(f"authoritative KWW fits missing: {fits_path}")
    fits = pd.read_csv(fits_path).set_index("id")

    tau_dev, i0_dev = {}, {}
    for rd in runs:
        tr = _run_trajectory(rd, base)
        published = fits.loc[tr["run_id"]]
        tau_dev[tr["run_id"]] = abs(round(tr["tau_min"], 3) - float(published["tau_min"]))
        i0_dev[tr["run_id"]] = abs(round(tr["i0_fit"], 1) - float(published["i0_fit"]))
    checks = {
        "reproduces_published_kww_fits": {
            "n_runs_checked": len(tau_dev),
            "max_abs_tau_deviation_min": float(max(tau_dev.values())),
            "max_abs_i0_deviation": float(max(i0_dev.values())),
            "tolerance": "tau to the artifact's 3 dp, i0 to its 1 dp",
            "ok": max(tau_dev.values()) < 1e-9 and max(i0_dev.values()) < 0.05,
        }}
    if not checks["reproduces_published_kww_fits"]["ok"]:
        raise ValueError(f"the reconstructed ΣI path no longer reproduces angular_kww_fits.csv: "
                         f"{checks['reproduces_published_kww_fits']}")

    # Preparation-first structure, read off the export rather than asserted.
    structure = per_run.groupby(["condition", "prep"])["run_id"].nunique()
    checks["design"] = {
        "conditions": sorted(per_run["condition"].unique()),
        "preps_per_condition": meta["n_preps_per_condition"],
        "technical_reps_per_prep": sorted(int(v) for v in structure.unique()),
        "n_runs": meta["n_runs"],
        "ok": (sorted(per_run["condition"].unique()) == sorted(CONDITIONS)
               and set(meta["n_preps_per_condition"].values()) == {4}
               and meta["n_runs"] == 27)}
    if not checks["design"]["ok"]:
        raise ValueError(f"unexpected Arm B design in the export: {checks['design']}")

    checks["normalization"] = {
        "starts_at_or_below_100_pct": float(
            per_run[np.isclose(per_run["t_min"], 0.0)]["pct_of_i0_fit"].max()),
        "definition": NORMALIZATION,
        "note": "the first measured frame sits BELOW the back-extrapolated start because "
                "acquisition begins ~30 s after injection, so t=0 is under 100 %",
    }
    return checks


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--study-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    base = args.study_root or arm_b_root()
    out = args.output_dir or base / OUT_SUB
    out.mkdir(parents=True, exist_ok=True)

    per_run, per_prep, per_condition, meta = build(base)
    checks = validate(per_run, meta, base)

    per_run.to_csv(out / "arm_b_angular_trajectories_runs.csv", index=False)
    per_prep.to_csv(out / "arm_b_angular_trajectories_preps.csv", index=False)
    per_condition.to_csv(out / "arm_b_angular_trajectories_conditions.csv", index=False)
    prov = provenance_record(
        "arm_b_angular_trajectories", study_root=str(base), uv_ph_values=(),
        signal_conditioning=SIGNAL_CONDITIONING,
        purpose="frozen export of the measured ΣI(t) trajectories behind the published Arm B "
                "empirical KWW endpoints, so a manuscript renderer can draw them without "
                "re-ingesting measurement files",
        signal_definition=SIGNAL_DEFINITION, normalization=NORMALIZATION,
        upstream_signal_path="psd_angular_fit.py — ingest.extract_run, kinetics.total_signal, "
                             "kinetics.despike_upward, kinetics.fit_signal",
        cross_checked_against=str(base / FITS_REL),
        time_grid=meta["time_grid_min"], aggregation=AGGREGATION,
        counts={"runs": meta["n_runs"], "preps_per_condition": meta["n_preps_per_condition"],
                "technical_reps_per_prep": meta["technical_reps_per_prep"]},
        numerical_checks=checks,
        scope={"quantity": meta["quantity"], "is_dissolved_mass": False,
               "background_subtracted": False, "upward_despiked": True,
               "excluded": ["UV", "forward model", "solubility", "solubility-normalized rate"]})
    prov_path = write_provenance(out / "provenance.json", prov)

    rep = checks["reproduces_published_kww_fits"]
    print(f"{meta['n_runs']} runs across {sorted(meta['n_preps_per_condition'].items())}")
    print(f"reproduces angular_kww_fits.csv for all {rep['n_runs_checked']} runs "
          f"(max |Δτ| {rep['max_abs_tau_deviation_min']:.2e} min, "
          f"max |Δi0| {rep['max_abs_i0_deviation']:.3f})")
    print(f"shared grid {TIME_GRID.min():g}-{TIME_GRID.max():g} min step {GRID_STEP_MIN:g} "
          f"({TIME_GRID.size} points); shortest run {meta['shortest_run_min']:.3f} min")
    print("aggregation: technical runs within preparation, then four preparations weighted "
          "equally; SD is between-preparation")
    for name in ("arm_b_angular_trajectories_runs.csv", "arm_b_angular_trajectories_preps.csv",
                 "arm_b_angular_trajectories_conditions.csv"):
        print(f"wrote {out / name}")
    print(f"wrote {prov_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
