"""CFZ UV-Vis calibration (per-pH curves, blank, filter offset, DMSO dilution) — the single
source of truth, loaded from ``data/assay/calibration.json``."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from diffractomorph_pipeline.assay.curve import StandardCurve


@dataclass(frozen=True)
class AssayCalibration:
    """Explicit, material-specific UV assay profile."""

    calibration_id: str
    curves: Mapping[tuple[float, int], tuple[float, float]]
    blank: Mapping[int, float]
    filter_offset: Mapping[float, float]
    dilution: float
    suspension: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], calibration_id: str = "assay-profile") -> "AssayCalibration":
        dilution_record = raw["dilution"]
        dilution = (
            float(dilution_record["sample_uL"] + dilution_record["dmso_uL"])
            / float(dilution_record["sample_uL"])
        )
        curves = {
            (float(condition), int(wavelength)): (float(coefficients[0]), float(coefficients[1]))
            for condition, wavelengths in raw["curves"].items()
            for wavelength, coefficients in wavelengths.items()
        }
        if not curves or any(slope <= 0 for slope, _ in curves.values()):
            raise ValueError("assay calibration curves require positive slopes")
        profile_id = str(raw.get("meta", {}).get("calibration_id") or calibration_id)
        return cls(
            calibration_id=profile_id,
            curves=curves,
            blank={int(wavelength): float(value) for wavelength, value in raw["blank"].items()},
            filter_offset={
                float(condition): float(value)
                for condition, value in raw["filter_offset_ugml"].items()
            },
            dilution=dilution,
            suspension=raw.get("suspension", {}),
            metadata=raw.get("meta", {}),
        )

    def curve(self, condition: float, wavelength_nm: int) -> StandardCurve:
        key = (float(condition), int(wavelength_nm))
        if key not in self.curves:
            raise KeyError(f"assay profile {self.calibration_id!r} has no curve for {key}")
        slope, intercept = self.curves[key]
        return StandardCurve(
            slope, intercept, wavelength_nm=wavelength_nm,
            label=f"{self.calibration_id} condition={condition} {wavelength_nm}nm",
        )


def default_path() -> Path:
    """Caller-selected assay profile, falling back to the optional legacy artifact."""
    selected = os.environ.get("DFM_ASSAY_PROFILE")
    return (Path(selected).expanduser() if selected else
            Path(__file__).parent.parent / "data" / "assay" / "calibration.json")


def load_assay_profile(path: Path | str | Mapping[str, Any]) -> AssayCalibration:
    """Load a caller-selected assay artifact; no material is selected implicitly."""
    if isinstance(path, Mapping):
        return AssayCalibration.from_mapping(path)
    raw = json.loads(Path(path).read_text())
    return AssayCalibration.from_mapping(raw, calibration_id=Path(path).stem)


def load_calibration(path: Path | str | None = None) -> dict:
    """Read the calibration artifact → typed dicts (curves, blank, filter_offset, dilution)."""
    profile = load_assay_profile(path or default_path())
    return {"curves": dict(profile.curves), "blank": dict(profile.blank),
            "filter_offset": dict(profile.filter_offset), "dilution": profile.dilution,
            "suspension": dict(profile.suspension), "meta": dict(profile.metadata)}


try:
    _DEFAULT_PROFILE = load_assay_profile(default_path())
except FileNotFoundError:
    _DEFAULT_PROFILE = None

_CAL = ({"curves": dict(_DEFAULT_PROFILE.curves), "blank": dict(_DEFAULT_PROFILE.blank),
         "filter_offset": dict(_DEFAULT_PROFILE.filter_offset),
         "dilution": _DEFAULT_PROFILE.dilution,
         "suspension": dict(_DEFAULT_PROFILE.suspension),
         "meta": dict(_DEFAULT_PROFILE.metadata)}
        if _DEFAULT_PROFILE is not None else
        {"curves": {}, "blank": {}, "filter_offset": {}, "dilution": None,
         "suspension": {}, "meta": {}})
CURVES = _CAL["curves"]                 # {(pH, wavelength_nm): (slope, intercept)}
BLANK = _CAL["blank"]                   # {wavelength_nm: blank absorbance}
FILTER_OFFSET = _CAL["filter_offset"]   # {pH: additive µg/mL offset (filter adsorption)}
DILUTION = _CAL["dilution"]             # 300/270 (270 µL sample + 30 µL DMSO)
SUSPENSION = _CAL["suspension"]         # pH-7 dosing-suspension assay: 453 nm DMSO curve, DF, filter offset


def curve(ph: float, wavelength_nm: int) -> StandardCurve:
    """Legacy CFZ curve lookup; generic workflows use :func:`load_assay_profile`."""
    if _DEFAULT_PROFILE is None:
        raise FileNotFoundError(
            "the optional CFZ assay artifact is not installed; pass an explicit assay profile"
        )
    return _DEFAULT_PROFILE.curve(ph, wavelength_nm)
