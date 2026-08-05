"""UV-Vis standard-curve engine — Abs ↔ concentration (µg/mL).

Linear inverse Beer–Lambert: conc = ((raw_abs − blank) − intercept) / slope × dilution.
Standards are gravimetric/un-filtered; filter adsorption is a separate sample-side correction
(:func:`filter_recovery`), never applied to the curve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardCurve:
    """Linear UV-Vis calibration: ``abs = slope·conc + intercept`` (conc in µg/mL).

    Build with :meth:`fit` from gravimetric standards, or construct directly from known
    coefficients. ``lod``/``loq`` follow the ICH SE-of-intercept convention
    (3.3·σ_b/slope and 10·σ_b/slope, σ_b = standard error of the intercept).
    """
    slope: float
    intercept: float
    r2: float | None = None
    se_intercept: float | None = None
    lod: float | None = None          # µg/mL
    loq: float | None = None          # µg/mL
    wavelength_nm: float | None = None
    label: str = ""

    @classmethod
    def fit(cls, conc, absorbance, wavelength_nm=None, label="") -> "StandardCurve":
        c = np.asarray(conc, float); a = np.asarray(absorbance, float)
        if c.size < 3:
            raise ValueError("need ≥3 standards to fit a curve")
        n = c.size
        slope, intercept = np.polyfit(c, a, 1)
        resid = a - (slope * c + intercept)
        ss_res = float(np.sum(resid ** 2)); ss_tot = float(np.sum((a - a.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # SE of the intercept (ICH LOD/LOQ basis)
        sxx = float(np.sum((c - c.mean()) ** 2))
        s_yx = float(np.sqrt(ss_res / (n - 2))) if n > 2 else float("nan")
        se_int = s_yx * np.sqrt(1.0 / n + c.mean() ** 2 / sxx) if sxx > 0 else float("nan")
        lod = 3.3 * se_int / slope if slope else float("nan")
        loq = 10.0 * se_int / slope if slope else float("nan")
        return cls(float(slope), float(intercept), r2, float(se_int),
                   float(lod), float(loq), wavelength_nm, label)

    def concentration(self, raw_abs, blank=0.0, dilution_factor=1.0):
        """Invert the curve: ((raw_abs − blank) − intercept) / slope × dilution → µg/mL."""
        a = np.asarray(raw_abs, float) - blank
        return (a - self.intercept) / self.slope * dilution_factor

    def absorbance_of(self, conc):
        """Forward: expected absorbance at a given concentration (µg/mL)."""
        return self.slope * np.asarray(conc, float) + self.intercept

    def flag_below_loq(self, conc):
        """Boolean mask where a concentration is below the LOQ (result is uncertain)."""
        if self.loq is None or not np.isfinite(self.loq):
            return np.zeros(np.shape(conc), bool)
        return np.asarray(conc, float) < self.loq


@dataclass
class AssayResult:
    """A concentration readout from replicate wells (µg/mL)."""
    mean: float
    sd: float
    n: int
    below_loq: bool
    per_rep: np.ndarray
    curve_label: str = ""


def read_concentration(raw_abs, blank, curve: StandardCurve, dilution_factor=1.0) -> AssayResult:
    """Blank-correct replicate absorbances, apply the curve, summarize (mean ± SD)."""
    reps = curve.concentration(np.asarray(raw_abs, float), blank, dilution_factor)
    reps = np.atleast_1d(reps)
    mean = float(np.mean(reps))
    return AssayResult(mean=mean, sd=float(np.std(reps, ddof=1)) if reps.size > 1 else 0.0,
                       n=int(reps.size), below_loq=bool(mean < (curve.loq or 0.0)),
                       per_rep=reps, curve_label=curve.label)


def filter_recovery(syringe_conc, reference_conc):
    """Filter-adsorption recovery = syringe / reference (glass or centrifuged).

    CFZ adsorbs strongly and time-dependently to syringe filters, so a syringe-filtered
    reading under-reports. This returns the recovery fraction; divide a syringe reading by
    it to correct, or — preferably — use the filter-free (glass/centrifuged) reading
    directly. Recovery ≪ 1 means the filter is unusable for this analyte at this level.
    """
    ref = np.asarray(reference_conc, float)
    return np.asarray(syringe_conc, float) / np.where(ref == 0, np.nan, ref)
