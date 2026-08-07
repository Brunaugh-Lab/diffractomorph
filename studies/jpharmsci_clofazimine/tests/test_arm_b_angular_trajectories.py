"""Guardrails for the frozen Arm B ΣI(t) trajectory export.

The manuscript figure draws these trajectories beside the published KWW endpoints, so the export
has one job beyond producing numbers: prove it is the SAME signal the fits were made to. Most
tests here read the written artifact (cheap); one re-derives a single run end to end, because a
cross-check that only ever reads its own recorded verdict is not a cross-check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))

import arm_b_angular_trajectories as tr               # noqa: E402


def _base():
    try:
        return tr.arm_b_root()
    except Exception:
        return None


BASE = _base()
OUT = (BASE / tr.OUT_SUB) if BASE else None
FILES = ("arm_b_angular_trajectories_runs.csv", "arm_b_angular_trajectories_preps.csv",
         "arm_b_angular_trajectories_conditions.csv", "provenance.json")


def _exported() -> bool:
    return OUT is not None and all((OUT / f).exists() for f in FILES)


exported = pytest.mark.skipif(not _exported(), reason="trajectory export not present")


@pytest.fixture(scope="module")
def tables():
    return {name.split("_")[-1].removesuffix(".csv"): pd.read_csv(OUT / name)
            for name in FILES[:3]}


@pytest.fixture(scope="module")
def provenance():
    return json.loads((OUT / "provenance.json").read_text())


# 1 ── the export reproduces the published KWW fits ──────────────────────────────────────────

@exported
def test_recorded_cross_check_against_the_published_fits(provenance):
    cross = provenance["numerical_checks"]["reproduces_published_kww_fits"]
    assert cross["ok"] and cross["n_runs_checked"] == 27
    assert cross["max_abs_tau_deviation_min"] < 1e-9
    assert cross["max_abs_i0_deviation"] < 0.05


@exported
def test_one_run_is_re_derived_end_to_end_and_matches_the_published_fit():
    """Re-ingest a single run and check τ and i0 against angular_kww_fits.csv."""
    runs, base, _ = tr._arm_b_runs()
    fits = pd.read_csv(BASE / tr.FITS_REL).set_index("id")
    rd = runs[0]
    got = tr._run_trajectory(rd, base)
    published = fits.loc[got["run_id"]]
    assert round(got["tau_min"], 3) == float(published["tau_min"])
    assert abs(round(got["i0_fit"], 1) - float(published["i0_fit"])) < 0.05
    # and the normalised path is the raw signal over that same i0
    assert np.isclose(got["pct_of_i0"][0], np.interp(0.0, got["t_min"], got["sigma_i"])
                      / got["i0_fit"] * 100.0)


# 2 ── preparation-first aggregation ─────────────────────────────────────────────────────────

@exported
def test_preparations_are_the_independent_unit(tables, provenance):
    runs, preps = tables["runs"], tables["preps"]
    assert runs.groupby("condition")["prep"].nunique().eq(4).all()
    assert runs["run_id"].nunique() == 27
    assert provenance["aggregation"]["independent_unit"] == "suspension preparation"
    assert provenance["aggregation"]["technical_runs_are_not_independent_replicates"] is True
    assert sorted(preps["n_technical_reps"].unique()) == [1, 2, 3]


@exported
def test_prep_curve_is_the_mean_of_its_technical_runs(tables):
    runs, preps = tables["runs"], tables["preps"]
    derived = (runs.groupby(["condition", "prep", "t_min"], as_index=False)["pct_of_i0_fit"]
               .mean())
    merged = derived.merge(preps, on=["condition", "prep", "t_min"], suffixes=("_d", "_a"))
    assert len(merged) == len(preps)
    assert (merged["pct_of_i0_fit_d"] - merged["pct_of_i0_fit_a"]).abs().max() < 1e-9


@exported
def test_condition_curve_weights_the_four_preparations_equally(tables):
    preps, conditions = tables["preps"], tables["conditions"]
    derived = (preps.groupby(["condition", "t_min"], as_index=False)
               .agg(m=("pct_of_i0_fit", "mean"), s=("pct_of_i0_fit", "std")))
    merged = derived.merge(conditions, on=["condition", "t_min"])
    assert len(merged) == len(conditions)
    assert (merged["m"] - merged["pct_mean"]).abs().max() < 1e-9
    assert (merged["s"] - merged["pct_sd_between_preps"]).abs().max() < 1e-9
    assert conditions["n_preps"].eq(4).all()


@exported
def test_pooling_runs_would_give_a_different_answer(tables):
    """The nesting is uneven (1/2/3 runs per preparation), so run-pooling is not equivalent."""
    runs, conditions = tables["runs"], tables["conditions"]
    pooled = runs.groupby(["condition", "t_min"], as_index=False)["pct_of_i0_fit"].mean()
    merged = pooled.merge(conditions, on=["condition", "t_min"])
    assert (merged["pct_of_i0_fit"] - merged["pct_mean"]).abs().max() > 0.1


# 3 ── the grid neither invents resolution nor extrapolates ──────────────────────────────────

@exported
def test_the_shared_grid_is_coarser_than_acquisition_and_inside_every_run(provenance):
    grid = provenance["time_grid"]
    assert grid["step"] == 0.2                       # acquisition cadence is 0.183-0.200 min
    assert grid["start"] == 0.0
    assert grid["end"] < 21.0
    assert grid["n_points"] == int(round(grid["end"] / grid["step"])) + 1
    assert "back-extrapolated" in provenance["numerical_checks"]["normalization"]["definition"]


@exported
def test_no_grid_point_lies_beyond_the_shortest_run():
    assert tr.TIME_GRID.max() <= 20.9833 + 1e-9


# 4 ── what the quantity is, and is not ──────────────────────────────────────────────────────

@exported
def test_the_signal_is_raw_and_not_dissolved_mass(provenance):
    assert provenance["scope"]["is_dissolved_mass"] is False
    assert provenance["scope"]["background_subtracted"] is False
    assert provenance["scope"]["upward_despiked"] is True
    definition = provenance["signal_definition"].lower()
    assert "measured value" in definition and "not background" in definition
    for excluded in ("UV", "forward model", "solubility"):
        assert excluded in provenance["scope"]["excluded"], excluded


@exported
def test_despiking_is_recorded_and_distinguished_from_background_subtraction(provenance):
    conditioning = provenance["signal_conditioning"]
    assert conditioning["upward_despiking"]["applied"] is True
    assert "despike_upward" in conditioning["upward_despiking"]["how"]
    assert conditioning["background_subtraction"]["applied"] is False
    assert "transient" in conditioning["distinction"]
    # the coarse, self-contradicting cleaned/not-cleaned block must not be emitted
    assert "optical" not in provenance
    assert "raw Copt and raw channels" not in json.dumps(provenance)


@exported
def test_normalization_is_the_fitted_back_extrapolated_start(tables, provenance):
    assert "back-extrapolated" in provenance["normalization"]
    runs = tables["runs"]
    # the first measured frame sits below the fitted start: acquisition begins after injection
    first = runs[np.isclose(runs["t_min"], 0.0)]["pct_of_i0_fit"]
    assert first.max() <= 100.0
