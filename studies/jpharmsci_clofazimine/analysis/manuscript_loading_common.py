"""Shared authoritative inputs for the Section 3.4.2 starting-particle-loading figures.

Both manuscript builders — :mod:`manuscript_loading_response` (main text) and
:mod:`manuscript_si_loading_diagnostics` (Supporting Information) — read the same
:mod:`copt_loading` outputs through this module and never touch raw plates, RTF exports or q3
folders. Nothing here fits, models, gates or re-derives a scientific quantity: every value a figure
draws is either a cell of a tabulated artifact or an unweighted mean / SD / difference of such
cells, and each of those aggregations is declared in :data:`AGGREGATIONS` and re-checked against
the artifact that also tabulates it.

**Scope: the pH 4.5 sub-study only.** ``copt_loading`` analyses more than the manuscript reports;
:func:`read_runs` and :func:`read_scoped` admit only :data:`MANUSCRIPT_SUBSTUDY` rows, and
:func:`check_wording` fails any caption that names an out-of-scope sub-study. The manuscript
figures present the loading evaluation the manuscript analysis actually rests on and nothing else.

The design boundaries the figures must not outrun, all carried by the artifacts themselves:

  * **One suspension preparation.** Every loading level is an aliquot of one pH 4.5 suspension
    (2026-07-27), so the three replicates per level are TECHNICAL. There is no preparation-level
    error term and no result here is an n = 3 preparation result.
  * **The forward model is frozen**: rate scale 2.197 taken from
    ``selected_rate_only_fit_summary.csv`` column ``rate_scale_datebalanced``, with
    ``refitted_here = false``. The shared suspension QC PSD is primary; the per-run in-cuvette q0
    is a labelled q0-reliability sensitivity and never the primary prediction.
  * **The UV loading direction is conditional** on the additive filter-recovery correction: the
    12 % -> 24 % ordering reverses below 0.59 ug/mL, ~40 % of the calibrated 1.48 ug/mL, and the
    calibration artifact carries no uncertainty interval for that offset.
  * **Copt and q3 are particle-side optical coordinates.** Neither is a dissolved-dose fraction
    and neither is used as confirmation of the mass-domain prediction.

Run with the pipeline venv.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from diffractomorph_pipeline.config import data_root

# the gate, the levels and the tail threshold come from the analysis module that applied them,
# so the figures cannot drift from the established definitions
from copt_loading import LEVELS, STUDY_REL, TAIL_MAX_PCT

ANALYSIS = STUDY_REL / "analysis"
SOURCES = {
    "runs": ANALYSIS / "copt_loading_runs.csv",
    "level_means": ANALYSIS / "copt_loading_level_means.csv",
    "linearity": ANALYSIS / "copt_loading_linearity.csv",
    "offsets": ANALYSIS / "copt_loading_filter_offset_sensitivity.csv",
    "q3_frames": ANALYSIS / "copt_loading_q3_frames.csv",
    "q3_matched": ANALYSIS / "copt_loading_q3_matched_extent.csv",
    "q3_spread": ANALYSIS / "copt_loading_q3_size_path_spread.csv",
    "provenance": ANALYSIS / "provenance.json",
}

PH45 = "pH 4.5"
MANUSCRIPT_SUBSTUDY = PH45     # the only sub-study these manuscript figures admit
MANUSCRIPT_PREP = "20260727"

FROZEN_RATE_SCALE = 2.197
FROZEN_RATE_SOURCE = "selected_rate_only_fit_summary.csv"
FROZEN_RATE_COLUMN = "rate_scale_datebalanced"
CALIBRATED_OFFSET_UGML = 1.48
CROSSOVER_OFFSET_UGML = 0.59   # displayed value; the artifact carries 0.5937829...

# Only these aggregations are performed on the artifacts, and each is re-checked in validate().
AGGREGATIONS = {
    "level_mean": "unweighted mean of the three technical runs at one loading level",
    "level_sd": "sample SD of the three technical runs at one loading level — a TECHNICAL "
                "replicate SD, not a preparation-level error term",
    "loading_contrast": "difference of two tabulated level-mean recoveries "
                        "(12 % level minus 24 % level) at the same filter offset",
}

CLAIM_BOUNDARIES = [
    "one suspension preparation; the three replicates per loading level are technical and there "
    "is no preparation-level error term",
    "this is a within-preparation loading-response evaluation on an independent dataset, not a "
    "preparation-level validation, and nothing here generalises across preparations",
    "no forward-model parameter was refitted here; the rate scale is frozen at 2.197",
    "the shared suspension QC PSD is primary; the per-run in-cuvette q0 is a labelled "
    "sensitivity and is not used for the primary prediction",
    "the UV loading direction is conditional on the additive filter-recovery correction and "
    "reverses below 0.59 ug/mL, ~40 % of the calibrated 1.48 ug/mL",
    "the calibration artifact provides no uncertainty interval for the filter offset",
    "Copt loss and q3 are particle-side optical coordinates and are not used as confirmation "
    "of the mass-domain prediction",
    "q3 is the PAQXOS-inverted relative composition of the particles still detected; it is not "
    "particle mass and not an independent modality",
    "q3 results are descriptive; one preparation gives no error term for an invariance verdict",
]

# Wording that would overstate what this design supports. Checked against the captions.
BANNED_PHRASES = (
    "external validation", "externally validated", "preparation-level validation",
    "robust loading trend", "robust agreement", "cleanest validation", "validates the model",
    "validated the model", "confirms the model", "independent confirmation",
    "independently validates", "proves",
)
# The sub-study of this study that the manuscript analysis does not include. It is not reported,
# so it may not appear anywhere a reader looks: plotted rows, source data, captions, or the
# provenance sidecars that travel with the figures.
OUT_OF_SCOPE_SUBSTUDY_TOKENS = ("ph 4.0", "ph4.0", "ph=4.0", "20260623")
# Other studies whose data must not enter these figures. Unlike the tokens above these may be
# NAMED in a provenance sidecar's exclusion list — declaring what was kept out is the point of it.
OTHER_STUDY_TOKENS = ("career", "objective1", "nist", "arm_a", "arm a", "arm_b", "arm b",
                      "polysorbate", "tween", "antisolvent_tween80",
                      "dissolution_media_diagnostic")
FOREIGN_TOKENS = OUT_OF_SCOPE_SUBSTUDY_TOKENS + OTHER_STUDY_TOKENS


def _path(key: str) -> Path:
    return data_root() / SOURCES[key]


def read(key: str) -> pd.DataFrame:
    """One authoritative ``copt_loading`` table.

    Absent artifacts raise; nothing is reconstructed.
    """
    path = _path(key)
    if not path.exists():
        raise FileNotFoundError(
            f"authoritative source '{key}' missing: {path}. These figures do not reconstruct "
            f"analysis outputs — rerun analysis/copt_loading.py to regenerate it.")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"authoritative source '{key}' is empty: {path}")
    return frame


def read_scoped(key: str) -> pd.DataFrame:
    """One authoritative table restricted to the manuscript sub-study.

    ``copt_loading`` tabulates more than the manuscript reports. Every figure reads through this
    function so the out-of-scope rows are dropped once, in one place, instead of each panel being
    trusted to filter for itself.
    """
    frame = read(key)
    if "substudy" not in frame.columns:
        return frame                    # already single-sub-study by construction (e.g. offsets)
    scoped = frame[frame["substudy"].eq(MANUSCRIPT_SUBSTUDY)].copy()
    if scoped.empty:
        raise ValueError(f"'{key}' carries no {MANUSCRIPT_SUBSTUDY} rows")
    if set(scoped["substudy"]) != {MANUSCRIPT_SUBSTUDY}:
        raise ValueError(f"'{key}' still carries out-of-scope sub-study rows after filtering")
    return scoped.reset_index(drop=True)


def read_runs() -> pd.DataFrame:
    return read_scoped("runs")


def read_provenance() -> dict:
    path = _path("provenance")
    if not path.exists():
        raise FileNotFoundError(
            f"authoritative source 'provenance' missing: {path}. These figures do not "
            f"reconstruct analysis outputs — rerun analysis/copt_loading.py to regenerate it.")
    return json.loads(path.read_text())


def sources_present() -> bool:
    return all(_path(k).exists() for k in SOURCES)


def dose_ug(dose_mg) -> float:
    """Delivered dose in micrograms — the reader-facing x quantity in both figures."""
    return float(dose_mg) * 1e3


def level_summary(runs: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean +- TECHNICAL-replicate SD of one per-run column, by loading level.

    The three runs at a loading level are aliquots of one suspension measured three times, so the
    SD describes measurement scatter within a single preparation and is never a preparation-level
    error term. Levels are returned in ascending delivered dose.
    """
    if set(runs["substudy"]) != {MANUSCRIPT_SUBSTUDY}:
        raise ValueError("level_summary expects runs already scoped by read_runs()")
    out = (runs.groupby("level_pct", as_index=False)
           .agg(dose_mg=("dose_mg", "mean"), mean=(column, "mean"), sd=(column, "std"),
                n_technical_reps=(column, "count")))
    out["dose_ug"] = out["dose_mg"].map(dose_ug)
    return out.sort_values("dose_ug").reset_index(drop=True)


def design_counts(runs: pd.DataFrame) -> dict:
    """Preparation / replicate structure, read off the per-run artifact rather than asserted."""
    if set(runs["substudy"]) != {MANUSCRIPT_SUBSTUDY}:
        raise ValueError("design_counts expects runs already scoped by read_runs()")
    per_level = runs.groupby("level_pct")["rep"].nunique()
    return {
        "substudy": MANUSCRIPT_SUBSTUDY,
        "n_preparations": int(runs["prep"].nunique()),
        "preparation": sorted(str(p) for p in runs["prep"].unique()),
        "n_levels": int(runs["level_pct"].nunique()),
        "technical_reps_per_level": sorted(int(v) for v in per_level.unique()),
        "n_runs": int(len(runs)),
        "has_uv_timecourse": bool(runs["uv_pct_injected_end"].notna().any()),
        "replicates_are_technical": True,
        "loading_levels_are_aliquots_of_one_suspension": True,
    }


def frozen_model_record(runs: pd.DataFrame) -> dict:
    """The frozen-model provenance carried on every per-run row, checked for internal agreement."""
    scales = sorted({round(float(v), 6) for v in runs["frozen_rate_scale"]})
    sources = sorted({str(v) for v in runs["frozen_rate_scale_source"]})
    columns = sorted({str(v) for v in runs["frozen_rate_scale_column"]})
    if scales != [FROZEN_RATE_SCALE] or sources != [FROZEN_RATE_SOURCE] \
            or columns != [FROZEN_RATE_COLUMN]:
        raise ValueError(f"the frozen-model provenance on the run rows is not the expected "
                         f"{FROZEN_RATE_SCALE} from {FROZEN_RATE_SOURCE}[{FROZEN_RATE_COLUMN}]: "
                         f"{scales}, {sources}, {columns}")
    prov = read_provenance()["frozen_model"]
    if bool(prov.get("refitted_here", True)):
        raise ValueError("the study provenance reports refitted_here = true; these figures may "
                         "only present a frozen-parameter prediction")
    return {
        "rate_scale": FROZEN_RATE_SCALE,
        "source_file": FROZEN_RATE_SOURCE,
        "source_column": FROZEN_RATE_COLUMN,
        "source_relative_to_data_root": prov["source_relative_to_data_root"],
        "refitted_here": False,
        "psd_primary": "shared suspension QC q0 export (one suspension per sub-study)",
        "psd_sensitivity_not_plotted": "model_pct_end_per_run_q0 — per-run in-cuvette q0, "
                                       "retained in copt_loading_runs.csv as a q0 reliability "
                                       "sensitivity and NOT used for the primary prediction",
        "solubility": "pH 4.5 Cs from the packaged Cs(pH) calibration",
        "dose": "each run's own delivered dose",
    }


def offset_record() -> dict:
    """Filter-offset facts, taken verbatim from the study provenance."""
    prov = read_provenance()
    sens = prov["filter_offset_sensitivity"]
    return {
        "calibrated_offset_ugml": float(sens["calibrated_offset_ugml"]),
        "crossover_offset_ugml": float(sens["crossover_offset_ugml"]),
        "crossover_as_fraction_of_calibrated": float(sens["crossover_as_fraction_of_calibrated"]),
        "direction_above_crossover": sens["direction_above_crossover"],
        "direction_below_crossover": sens["direction_below_crossover"],
        "offset_convention": prov["uv_calibration"]["filter_offset_convention"],
        "uncertainty_interval_available": False,
    }


def find_foreign_tokens(text: str, tokens: tuple[str, ...] = FOREIGN_TOKENS) -> list[str]:
    """Out-of-scope study tokens in ``text``, matched only at a word start.

    The boundary matters: "between" must not read as the polysorbate token "tween", and
    "administer" must not read as "nist".
    """
    low = " ".join(str(text).split()).lower()
    return [t for t in tokens if re.search(rf"(?<![a-z0-9]){re.escape(t)}", low)]


def check_wording(text: str, *, allow_negated: tuple[str, ...] = ()) -> dict:
    """Banned overstatements and foreign-study tokens must be absent from a caption.

    ``allow_negated`` names phrases the caption is REQUIRED to disclaim: each occurrence must be
    immediately preceded by "not a " / "not " so the phrase can only appear as a denial.
    """
    low = " ".join(text.split()).lower()
    hits = []
    for phrase in BANNED_PHRASES:
        if phrase not in low:
            continue
        if phrase in allow_negated:
            total = low.count(phrase)
            negated = low.count(f"not a {phrase}") + low.count(f"not {phrase}")
            if negated == total:
                continue
        hits.append(phrase)
    foreign = find_foreign_tokens(low)
    return {"banned_phrases_found": hits, "foreign_tokens_found": foreign,
            "phrases_required_to_be_negated": list(allow_negated),
            "ok": not hits and not foreign}


# ── shared figure style ──────────────────────────────────────────────────────────────────────
# Type, marks, panel letters and the ordered ramp are shared across the whole manuscript figure
# set; only the level -> colour mapping is local, because loading is this study's ordered factor.

from manuscript_style import (BLUE, BLUE_MID, BLUE_PALE, DARK, FIG_W, GREY,  # noqa: E402,F401
                              PALE, VERMILLION, apply_style, clean_axes, panel_tags, ramp_for,
                              save)

LOADING_RAMP = ramp_for(LEVELS)     # 12 / 18 / 24 % -> light to dark


__all__ = ["LEVELS", "STUDY_REL", "TAIL_MAX_PCT", "SOURCES", "PH45", "MANUSCRIPT_SUBSTUDY",
           "MANUSCRIPT_PREP",
           "FROZEN_RATE_SCALE", "FROZEN_RATE_SOURCE", "FROZEN_RATE_COLUMN",
           "CALIBRATED_OFFSET_UGML", "CROSSOVER_OFFSET_UGML", "AGGREGATIONS", "CLAIM_BOUNDARIES",
           "BANNED_PHRASES", "FOREIGN_TOKENS", "OUT_OF_SCOPE_SUBSTUDY_TOKENS",
           "OTHER_STUDY_TOKENS", "find_foreign_tokens",
           "read", "read_scoped", "read_runs",
           "read_provenance", "sources_present",
           "dose_ug", "level_summary", "design_counts", "frozen_model_record", "offset_record",
           "check_wording", "apply_style", "clean_axes", "panel_tags", "save",
           "BLUE", "BLUE_MID", "BLUE_PALE", "LOADING_RAMP", "VERMILLION", "DARK", "GREY", "PALE",
           "FIG_W"]
