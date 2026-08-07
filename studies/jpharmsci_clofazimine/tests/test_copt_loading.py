"""Varying-starting-Copt (loading) study tests.

The things worth pinning are the two judgement calls the analysis rests on: that the starting
PSD comes from the shared suspension QC rather than the unreliable per-run in-cuvette q0, and
that q3 frames are gated on inversion reliability before any size is read off them. Both were
chosen because the raw records contradict the design, so a regression would silently reinstate
a five-fold phantom size difference or a 97%-coarse-tail "measurement".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))

import copt_loading as cl  # noqa: E402


def _study_root():
    try:
        return cl.study_root()
    except RuntimeError:
        return Path("/__dfm_data_root_unset__")


corpus = pytest.mark.skipif(not _study_root().is_dir(), reason="Copt study corpus not present")


# ── unit-testable logic ──────────────────────────────────────────────────────────────────────

def test_tail_gate_threshold_is_below_the_inversion_ceiling_usage():
    # A frame with most of its mass above 15 um is not a size measurement.
    assert 0 < cl.TAIL_MAX_PCT < 10


def test_copt_linearity_recovers_a_known_slope():
    dose = np.array([0.10, 0.15, 0.20, 0.25])
    per_run = pd.DataFrame({"substudy": "x", "ph": 4.5, "dose_mg": dose,
                            "copt0": 50.0 * dose + 2.0})
    out = cl.copt_linearity(per_run)
    assert np.isclose(out.loc[0, "slope_copt_per_mg"], 50.0)
    assert np.isclose(out.loc[0, "intercept_copt"], 2.0)
    assert out.loc[0, "r2"] > 0.999


def test_level_means_label_replicates_as_technical_not_independent():
    rows = [{"substudy": "pH 4.5", "ph": 4.5, "prep": "20260727", "level_pct": 12, "rep": r,
             "injected_uL": 30.0, "dose_mg": 0.13, "copt0": 10.0, "copt_floor": 4.0,
             "copt_frac_loss": 0.6, "copt_thalf_min": 4.0, "model_pct_end": 96.0,
             "model_pct_end_per_run_q0": 90.0, "qc_psd_dv50_um": 2.62,
             "per_run_q0_dv50_um": 2.7, "uv_pct_injected_end": 99.0} for r in (1, 2, 3)]
    out = cl.loading_summary(pd.DataFrame(rows))
    assert out.loc[0, "n_technical_reps"] == 3
    assert "n_preps" not in out.columns          # there is only one prep; never imply otherwise


# ── real-corpus structure (skipped when the tree is absent) ──────────────────────────────────

@corpus
def test_discovery_finds_both_substudies_at_three_loadings_by_three_reps():
    runs = cl.discover()
    assert len(runs) == 18
    assert set(runs["substudy"]) == {"pH 4.0", "pH 4.5"}
    assert (runs.groupby("substudy")["level_pct"].nunique() == 3).all()
    assert (runs.groupby(["substudy", "level_pct"]).size() == 3).all()
    # one preparation per sub-study — the whole caveat rests on this
    assert (runs.groupby("substudy")["prep"].nunique() == 1).all()


@corpus
def test_injected_volume_increases_with_nominal_copt_level():
    runs = cl.discover()
    for _, g in runs.groupby("substudy"):
        vols = g.groupby("level_pct")["injected_uL"].first()
        assert vols.is_monotonic_increasing, dict(vols)


@corpus
def test_starting_psd_comes_from_the_shared_qc_not_the_per_run_q0():
    runs = cl.discover()
    # every run in a sub-study must point at the SAME starting PSD: one suspension, three volumes
    assert (runs.groupby("substudy")["psd_src"].nunique() == 1).all()
    assert (runs["psd_src"] != runs["psd_src_per_run"]).all()


@corpus
def test_shared_qc_psd_agrees_across_substudies_but_per_run_q0_does_not():
    from diffractomorph_pipeline.forward import PSD
    runs = cl.discover()
    qc = {s: PSD.from_sympatec(g["psd_src"].iloc[0]).dv50 for s, g in runs.groupby("substudy")}
    assert abs(qc["pH 4.0"] - qc["pH 4.5"]) < 0.2      # same material, ~2.6 um both
    per_run = np.array([PSD.from_sympatec(p).dv50 for p in runs["psd_src_per_run"]])
    # the per-run q0 is what the analysis deliberately does not trust
    assert per_run.max() / np.mean(list(qc.values())) > 2.0


@corpus
def test_extensionless_uv_plates_are_readable_without_touching_the_originals():
    runs = cl.discover()
    uv = [p for p in runs["uv_file"] if p is not None]
    assert len(uv) == 9                                  # pH 4.5 only; pH 4.0 has no UV
    before = {p: Path(p).stat().st_mtime for p in uv}
    tc = cl._uv_timecourse_any(Path(uv[0]), 4.5)
    assert len(tc) > 1 and tc["conc_ugml"].notna().any()
    assert {p: Path(p).stat().st_mtime for p in uv} == before   # raw files untouched
    assert Path(uv[0]).suffix.lower() not in (".xlsx", ".xlsm")


@corpus
def test_ph40_q3_fails_the_reliability_gate_far_more_than_ph45():
    _, frames, _ = cl.analyse()
    rel = frames.groupby("substudy")["q3_frame_reliable"].mean()
    assert rel["pH 4.5"] > rel["pH 4.0"]
    # pH 4.0 starts at low Copt, so its inversion is degraded from the first frame
    first = frames.sort_values("t_min").groupby(["substudy", "run_id"]).first()
    assert first.loc["pH 4.0", "tail_pct_above_15um"].max() > 10.0
    assert first.loc["pH 4.5", "tail_pct_above_15um"].max() < 1.0


@corpus
def test_frozen_model_and_uv_both_decline_with_loading_at_ph45():
    # The sink-filling prediction: more solid into a fixed volume dissolves a smaller fraction.
    per_run, _, _ = cl.analyse()
    s = cl.loading_summary(per_run)
    s45 = s[s["substudy"].eq("pH 4.5")].sort_values("level_pct")
    assert s45["model_pct_end"].is_monotonic_decreasing
    assert s45["uv_pct_injected_end"].is_monotonic_decreasing


# ── assay-sensitivity corrections ────────────────────────────────────────────────────────────

@corpus
def test_uv_helper_uses_per_plate_blanks_not_the_packaged_global():
    # These plates report 280 nm blanks of 0.052-0.054 against a packaged global of 0.050.
    # Using the global would bias every concentration upward, so the helper must mirror
    # assay.uv_timecourse and prefer the plate's own blank.
    from diffractomorph_pipeline.assay import calibration as cal
    runs = cl.discover()
    uv = [p for p in runs["uv_file"] if p is not None]
    tc = cl._uv_timecourse_any(Path(uv[0]), 4.5)
    plate = cl._read_plate_any(Path(uv[0]))
    assert not np.isnan(plate.blank280)
    assert tc["blank280_used"].iloc[0] == plate.blank280
    assert tc["blank280_used"].iloc[0] != cal.BLANK[280]     # the bug this replaces


def test_uv_helper_falls_back_to_the_global_blank_when_the_plate_has_none(monkeypatch):
    from diffractomorph_pipeline.assay import calibration as cal

    class _Plate:
        times_min = np.array([2.0, 4.0])
        a280 = np.array([0.20, 0.30]); a490 = np.array([0.10, 0.15])
        blank280 = float("nan"); blank490 = float("nan")

    monkeypatch.setattr(cl, "_read_plate_any", lambda p: _Plate())
    tc = cl._uv_timecourse_any(Path("ignored"), 4.5)
    assert tc["blank280_used"].iloc[0] == cal.BLANK[280]
    assert tc["blank490_used"].iloc[0] == cal.BLANK[490]


def test_filter_offset_is_an_override_not_an_addition(monkeypatch):
    from diffractomorph_pipeline.assay import calibration as cal

    class _Plate:
        times_min = np.array([2.0]); a280 = np.array([0.20]); a490 = np.array([0.10])
        blank280 = 0.052; blank490 = 0.030

    monkeypatch.setattr(cl, "_read_plate_any", lambda p: _Plate())
    default = cl._uv_timecourse_any(Path("x"), 4.5)
    zero = cl._uv_timecourse_any(Path("x"), 4.5, filter_offset_ugml=0.0)
    assert default["filter_offset_ugml"].iloc[0] == cal.FILTER_OFFSET[4.5]
    assert zero["filter_offset_ugml"].iloc[0] == 0.0
    # the offset enters additively, scaled by the dilution
    assert np.isclose(default["conc_ugml"].iloc[0] - zero["conc_ugml"].iloc[0],
                      cal.FILTER_OFFSET[4.5] * cal.DILUTION)


@corpus
def test_offset_sensitivity_locates_the_direction_reversal():
    level, meta = cl.offset_sensitivity()
    assert set(level["level_pct"]) == set(cl.LEVELS)
    # additive offset contributes MORE recovery points at the lowest dose
    at_cal = level[np.isclose(level["offset_ugml"], meta["calibrated_offset_ugml"])] \
        .set_index("level_pct")["offset_contribution_pp"]
    assert at_cal[12] > at_cal[18] > at_cal[24]
    # direction genuinely reverses across the crossover
    lo = level[np.isclose(level["offset_ugml"], 0.0)].set_index("level_pct")["recovery_pct"]
    hi = level[np.isclose(level["offset_ugml"], meta["calibrated_offset_ugml"])] \
        .set_index("level_pct")["recovery_pct"]
    assert lo[12] < lo[24]        # rises with loading at zero offset
    assert hi[12] > hi[24]        # declines with loading at the calibrated offset
    assert 0.0 < meta["crossover_offset_ugml"] < meta["calibrated_offset_ugml"]
    assert 0.3 < meta["crossover_as_fraction_of_calibrated"] < 0.5


@corpus
def test_matched_extent_size_output_contains_only_ph45():
    _, frames, matched = cl.analyse()
    assert set(matched["substudy"]) == {"pH 4.5"}
    assert "pH 4.0" not in set(matched["substudy"])
    # ...while the per-frame reliability diagnostics for pH 4.0 are retained, since the point is
    # to keep that failure visible rather than silently drop the sub-study.
    assert "pH 4.0" in set(frames["substudy"])
    assert frames.loc[frames["substudy"].eq("pH 4.0"), "tail_pct_above_15um"].notna().any()


def test_claim_boundary_language_is_present_and_overclaims_are_absent():
    # Compare on whitespace-normalised text so docstring rewrapping cannot break the check.
    src = " ".join(Path(cl.__file__).read_text().split())
    assert "within-preparation loading-response evaluation" in src
    assert "not** a preparation-level validation" in src
    for overclaim in ("cleanest forward-model validation", "robust loading trend",
                      "validates the model", "confirms the finite-sink"):
        assert overclaim not in src, overclaim
    # fractional Copt loss must not be offered as confirmation of the mass prediction
    assert "is not used as confirmation" in src
    # the conditional wording must accompany the UV agreement
    assert "conditional agreement with the frozen finite-sink prediction" in src.lower()


@corpus
def test_frozen_rate_scale_is_recorded_not_implicit():
    # model_pct_end is determined by this number, so it must be traceable from the outputs
    # rather than buried in the code.
    value, path, column = cl.frozen_rate_scale()
    assert value > 0 and column == "rate_scale_datebalanced"
    assert path.exists() and path.name.endswith(".csv")
    per_run, _, _ = cl.analyse()
    assert (per_run["frozen_rate_scale"] == value).all()
    assert (per_run["frozen_rate_scale_source"] == path.name).all()
    assert (per_run["frozen_rate_scale_column"] == column).all()


def test_frozen_rate_scale_rejects_a_malformed_artifact(tmp_path, monkeypatch):
    import copt_loading as mod
    bad = tmp_path / "disso_experiments/ph_dependent_dissolution_study/forward_prediction/scalar_fit"
    bad.mkdir(parents=True)
    (bad / "selected_rate_only_fit_summary.csv").write_text("wrong_column\n1.0\n")
    monkeypatch.setattr(mod, "data_root", lambda: tmp_path)
    with pytest.raises(ValueError):
        mod.frozen_rate_scale()


def test_frozen_rate_scale_reports_a_missing_artifact_clearly(tmp_path, monkeypatch):
    import copt_loading as mod
    monkeypatch.setattr(mod, "data_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="frozen rate-scale artifact missing"):
        mod.frozen_rate_scale()
