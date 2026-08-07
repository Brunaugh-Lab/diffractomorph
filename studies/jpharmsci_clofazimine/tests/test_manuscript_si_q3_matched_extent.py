"""Guardrails for the Supporting Information q3 matched-extent figure.

This figure is easy to over-read: q3 is an inverted relative composition, not mass, and the pH 4.0
excursion is a reliability boundary rather than a finding. The tests therefore pin the scope
claims alongside the arithmetic — two panels only, g = 0.8 only, date-first aggregation, the
retained outlier, the median-not-mean primary curve, and the absence of mass/aggregation language.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))

import manuscript_si_q3_matched_extent as si  # noqa: E402


def _have_sources() -> bool:
    try:
        from diffractomorph_pipeline.config import data_root
        return all((data_root() / v).exists() for v in si.SOURCES.values())
    except Exception:
        return False


corpus = pytest.mark.skipif(not _have_sources(), reason="canonical q3 artifacts not present")


# 1 ── exactly two panels ─────────────────────────────────────────────────────────────────────

@corpus
def test_exactly_two_panels_are_produced():
    table, _ = si.build_source_data()
    assert set(table["panel"]) == {"A", "B"}


def test_renderer_creates_a_single_row_of_two_axes():
    import inspect
    body = inspect.getsource(si.render)
    assert "plt.subplots(1, 2" in body
    assert '(axA, "A"), (axB, "B")' in body


# 2 ── panel A contains only g = 0.8 ──────────────────────────────────────────────────────────

@corpus
def test_panel_a_uses_only_g_equals_0_8():
    assert si.TARGET_G == 0.8
    table, meta = si.build_source_data()
    a = table[table["panel"].eq("A")]
    assert set(a["target_g"].dropna().unique()) == {0.8}
    assert meta["panel_a"]["target_g"] == 0.8


@corpus
def test_deeper_extents_are_omitted_because_ph5_does_not_reach_them():
    # the omission is data-driven, not stylistic: pH 5.0 exists only at g = 0.8
    from diffractomorph_pipeline.config import data_root
    runs = pd.read_csv(data_root() / si.SOURCES["matched_by_run"])
    ok = runs[runs["D50"].notna() & np.isclose(runs["floor_frac"], si.FLOOR)]
    reach = ok[ok["condition"].eq("pH 5.0")]["target_g"].unique()
    assert set(np.round(reach, 3)) == {0.8}


@corpus
def test_panel_a_matched_values_reproduce_the_canonical_numbers():
    _, meta = si.build_source_data()
    checks = meta["panel_a"]["checks"]
    assert checks["pH 4.0"]["matches_expected_1dp"] and checks["pH 4.0"]["agrees_with_artifact"]
    assert checks["pH 4.5"]["matches_expected_1dp"] and checks["pH 4.5"]["agrees_with_artifact"]
    assert checks["pH 5.0"]["matches_expected_1dp"] and checks["pH 5.0"]["agrees_with_artifact"]
    assert round(checks["pH 4.0"]["matched_D50_from_runs"], 1) == 2.3
    assert round(checks["pH 4.5"]["matched_D50_from_runs"], 1) == 2.6
    assert round(checks["pH 5.0"]["matched_D50_from_runs"], 1) == 2.6


# 3 ── date-first aggregation ─────────────────────────────────────────────────────────────────

@corpus
def test_panel_a_plots_preparation_date_means_not_individual_runs():
    table, _ = si.build_source_data()
    pairs = table[table["panel"].eq("A") & table["kind"].eq("date_pair")]
    # 3 conditions x 3 dates = 9 plotted units, not the 23 contributing runs
    assert len(pairs) == 9
    assert (pairs.groupby("condition")["date"].nunique() == 3).all()
    assert pairs["n_runs"].sum() == 23


@corpus
def test_panel_b_primary_and_thin_lines_are_both_date_first():
    table, _ = si.build_source_data()
    b = table[table["panel"].eq("B")]
    traj = b[b["kind"].eq("date_trajectory")]
    assert set(traj["date"].dropna().unique()) == {20260608, 20260609, 20260610}
    primary = b[b["kind"].eq("cross_date_median")]
    assert set(primary["summary_statistic"]) == {"cross_date_median"}


def test_module_documents_date_first_aggregation():
    src = " ".join(Path(si.__file__).read_text().split())
    assert "nested runs are averaged within preparation date before any cross-date summary" in src


# 4 ── pH 5.0 contributes 3 dates and 5 runs ──────────────────────────────────────────────────

@corpus
def test_ph5_has_three_dates_and_five_runs():
    table, meta = si.build_source_data()
    cond = table[table["kind"].eq("condition_summary")].set_index("condition")
    assert int(cond.loc["pH 5.0", "n_dates"]) == 3
    assert int(cond.loc["pH 5.0", "n_runs"]) == 5
    assert meta["panel_a"]["checks"]["pH 5.0"]["n_dates"] == 3
    assert meta["panel_a"]["checks"]["pH 5.0"]["n_runs"] == 5
    # the other conditions contribute all nine
    assert int(cond.loc["pH 4.0", "n_runs"]) == 9 and int(cond.loc["pH 4.5", "n_runs"]) == 9


# 5 ── the 20260608 pH 4.0 12.6-min point is retained ─────────────────────────────────────────

@corpus
def test_the_20260608_excursion_point_is_retained_not_excluded():
    table, meta = si.build_source_data()
    b = table[table["panel"].eq("B") & table["kind"].eq("date_trajectory")]
    hit = b[b["date"].eq(si.OUTLIER_DATE) & np.isclose(b["time_min"], si.OUTLIER_T)]
    assert len(hit) == 1
    assert round(float(hit["D50_um"].iloc[0]), 1) == 60.0
    audit = meta["panel_b"]
    assert audit["underlying_observation_deleted"] is False
    assert audit["point_specific_exclusion_applied"] is False
    assert audit["all_contributors_passed_working_eligibility"] is True


@corpus
def test_the_excursion_is_a_single_date_and_the_median_suppresses_it():
    _, meta = si.build_source_data()
    b = meta["panel_b"]
    assert b["driving_date"] == si.OUTLIER_DATE
    assert round(b["driving_date_D50"], 1) == 60.0
    assert round(b["cross_date_median_D50"], 1) == 5.5
    assert round(b["cross_date_mean_D50"], 1) == 22.4
    # the estimator choice is what matters: median ≪ mean here
    assert b["cross_date_median_D50"] < b["cross_date_mean_D50"] / 3


# 6 ── the median, not the mean, is the heavy trajectory ──────────────────────────────────────

@corpus
def test_the_primary_heavy_curve_is_the_cross_date_median():
    table, _ = si.build_source_data()
    primary = table[table["panel"].eq("B") & table["kind"].eq("cross_date_median")]
    assert len(primary) > 0
    assert set(primary["summary_statistic"]) == {"cross_date_median"}
    # no mean-based condition series is carried into the figure at all
    assert "cross_date_mean" not in set(table.get("summary_statistic", pd.Series(dtype=str)).dropna())


def test_a_mean_primary_curve_is_rejected(monkeypatch):
    import manuscript_si_q3_matched_extent as mod
    real = mod._read

    def fake(key):
        frame = real(key)
        if key == "condition_median":
            frame = frame.copy()
            frame["summary_statistic"] = "cross_date_mean"
        return frame

    if not _have_sources():
        pytest.skip("canonical q3 artifacts not present")
    monkeypatch.setattr(mod, "_read", fake)
    with pytest.raises(ValueError, match="must be the cross-date median"):
        mod.panel_b_data()


# 7 ── no mass / aggregation / one-size claims ────────────────────────────────────────────────

def test_module_and_caption_make_no_mass_or_aggregation_claim():
    src = " ".join(Path(si.__file__).read_text().split()).lower()
    # these appear only inside explicit disclaimers
    for phrase in ("mass measurement", "aggregation", "deaggregation", "uniform dissolution"):
        assert phrase in src, f"{phrase} should be explicitly disclaimed"
    assert "it is not a mass measurement" in src
    assert "nothing here supports aggregation" in src
    for banned in ("proves aggregation", "demonstrates aggregation", "independent validation of",
                   "one channel corresponds to one size", "mass of the particles"):
        assert banned not in src, banned


@corpus
def test_caption_states_the_scope_limits(tmp_path):
    si.main(["--output-dir", str(tmp_path), "--formats", "pdf"])
    caption = " ".join((tmp_path / f"{si.STEM}_caption.md").read_text().split())
    assert "relative composition of the particles still detected" in caption
    assert "not a mass measurement" in caption
    assert "not independent of the laser-diffraction" in caption
    assert "was retained" in caption and "not reproduced" in caption
    assert "adopted prospectively for every pH, timepoint, and percentile" in caption
    assert "not a point-specific exclusion" in caption
    assert "do not establish aggregation, deaggregation, uniform dissolution" in caption


# 8 ── all outputs written ────────────────────────────────────────────────────────────────────

@corpus
def test_all_six_outputs_are_written(tmp_path):
    si.main(["--output-dir", str(tmp_path), "--formats", "pdf,png,svg"])
    for suffix in (".pdf", ".png", ".svg", "_source_data.csv",
                   "_provenance.json", "_caption.md"):
        path = tmp_path / f"{si.STEM}{suffix}"
        assert path.exists() and path.stat().st_size > 0, suffix


@corpus
def test_provenance_records_sources_settings_and_scope_limits(tmp_path):
    si.main(["--output-dir", str(tmp_path), "--formats", "pdf"])
    prov = pd.read_json(tmp_path / f"{si.STEM}_provenance.json", typ="series")
    assert set(prov["sources"]) == set(si.SOURCES)
    assert prov["settings"]["target_g"] == 0.8
    assert prov["settings"]["floor_source"] == "q3_matched_extent.DEFAULT_FLOOR"
    assert prov["settings"]["panel_b_primary_statistic"] == "cross_date_median"
    assert prov["scope"]["is_mass_measurement"] is False
    assert prov["scope"]["independent_of_ld_acquisition"] is False
    for claim in ("aggregation", "deaggregation", "uniform dissolution",
                  "absence of redistribution", "independent validation"):
        assert claim in prov["scope"]["claims_excluded"], claim
    assert prov["git_commit"]


def test_missing_source_raises_rather_than_reconstructing(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "data_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="does not reconstruct"):
        si._read("outlier_audit")
