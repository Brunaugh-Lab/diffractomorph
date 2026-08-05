"""Equilibrium solubility Cs(pH) — the third pipeline calibration artifact.

Like the Mie kernel and the noise surface, Cs(pH) is a periodically-measured,
material/lot-specific constant built once and consumed forward at analysis time. It is
the solubility the size-resolved Noyes–Whitney forward model needs (driving force
``1 − C/Cs``) and the ceiling the supersaturation check uses.

Inputs are the **filter-free (glass / centrifuged)** equilibrium concentrations from the
UV-Vis assay (:mod:`assay`) — CFZ adsorbs to syringe filters, so syringe readings are not
solubility. CFZ is a weak base; below its ``pKa′`` solubility rises ~decade/pH-unit, so we
fit the **free-base branch** ``Cs(pH) = S0·(1 + 10^(pKa′ − pH))`` with ``pKa′`` fixed
(the data lie below ``pKa′`` and cannot separate S0 from pKa′) and ``S0`` (intrinsic base
solubility) fitted in log space. The branch fit is also a diagnostic: how well does the
drug follow Henderson–Hasselbalch in this buffer?

``cs_for_ph`` returns the **measured** value at an experimental pH (exact), and the fitted
branch elsewhere (interpolation / extrapolation). Units: µg/mL.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PKA_PRIME_CFZ = 6.08          # apparent (solubility-curve) pKa, Woldemichael 2018 — an empirical
                             # descriptor of the Cs(pH) profile, not the thermodynamic pKa
                             # (9.43, Verbić 2023, Mol Pharm 20:3160).
MW_CFZ = 473.4                # g/mol


def _branch_factor(ph, pka_prime):
    """Free-base ionization factor f(pH) = 1 + 10^(pKa′ − pH); Cs = S0·f."""
    return 1.0 + 10.0 ** (pka_prime - np.asarray(ph, float))


@dataclass
class SolubilityModel:
    drug: str
    pka_prime: float
    mw: float
    s0_ugml: float                       # fitted intrinsic base solubility
    points: list = field(default_factory=list)   # [{"ph","cs_ugml","sd_ugml"}]
    fit: dict = field(default_factory=dict)       # {"resid_pct": [...], "rmse_pct": ...}
    meta: dict = field(default_factory=dict)

    # ── evaluation ──────────────────────────────────────────────────────────
    @property
    def s0_uM(self) -> float:
        return self.s0_ugml / 1e3 / self.mw * 1e6

    def branch(self, ph):
        """Fitted free-base Cs(pH) in µg/mL (the smooth model, no measured-point override)."""
        return self.s0_ugml * _branch_factor(ph, self.pka_prime)

    def cs_for_ph(self, ph, prefer_measured=True, tol=0.03):
        """Cs at ``ph`` (µg/mL): the measured value if ``ph`` matches a measured point
        (within ``tol``), else the fitted free-base branch."""
        if prefer_measured:
            for p in self.points:
                if abs(p["ph"] - float(ph)) <= tol:
                    return float(p["cs_ugml"])
        return float(self.branch(ph))

    def s0_for_ph_ugml(self, ph, **kw):
        """Intrinsic S0 (µg/mL) consistent with cs_for_ph(ph) — feed the forward model.

        Back-solves S0 = Cs(pH)/f(pH) so the model reproduces the measured *bulk* Cs at
        that bulk pH (the surface-pH solve then raises the local Cs further). Returns S0 in
        µg/mL; convert to mol/L with ``/1e3/mw`` for ``dissolution_ode``'s ``is_isb``.
        """
        return self.cs_for_ph(ph, **kw) / float(_branch_factor(ph, self.pka_prime))

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path: Path | str) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "drug": self.drug, "pka_prime": self.pka_prime, "mw": self.mw,
            "s0_ugml": self.s0_ugml, "s0_uM": self.s0_uM,
            "points": self.points, "fit": self.fit, "meta": self.meta}, indent=2))
        return path

    @classmethod
    def load(cls, path: Path | str) -> "SolubilityModel":
        d = json.loads(Path(path).read_text())
        return cls(d["drug"], d["pka_prime"], d["mw"], d["s0_ugml"],
                   d.get("points", []), d.get("fit", {}), d.get("meta", {}))


def fit_free_base(points, drug="CFZ", pka_prime=PKA_PRIME_CFZ, mw=MW_CFZ,
                  meta=None) -> SolubilityModel:
    """Fit ``Cs(pH) = S0·(1 + 10^(pKa′−pH))`` (pKa′ fixed) to measured Cs(pH) points.

    ``points``: iterable of ``{"ph", "cs_ugml"[, "sd_ugml"]}`` (filter-free Cs). S0 is fit
    in **log space** (equal relative weight across the solubility decade). Stores per-point
    residuals (%) so the H–H adherence is visible.
    """
    pts = [dict(p) for p in points]
    ph = np.array([p["ph"] for p in pts], float)
    cs = np.array([p["cs_ugml"] for p in pts], float)
    f = _branch_factor(ph, pka_prime)
    s0 = float(10.0 ** np.mean(np.log10(cs / f)))        # geometric LS for Cs = S0·f
    pred = s0 * f
    resid_pct = (100.0 * (cs - pred) / cs).round(1).tolist()
    rmse_pct = float(np.sqrt(np.mean((100.0 * (cs - pred) / cs) ** 2)))
    return SolubilityModel(
        drug=drug, pka_prime=pka_prime, mw=mw, s0_ugml=s0, points=pts,
        fit={"cs_pred_ugml": pred.round(3).tolist(), "resid_pct": resid_pct,
             "rmse_pct": round(rmse_pct, 1)},
        meta=meta or {})


def default_path() -> Path:
    """Packaged artifact location (mirrors data/kernels, data/noise)."""
    return Path(__file__).parent / "data" / "solubility" / "cfz_cs_ph.json"


def load_default() -> SolubilityModel:
    path = default_path()
    if not path.exists():
        raise FileNotFoundError(
            "the optional CFZ solubility artifact is not installed; pass an explicit "
            "SolubilityModel or set s0_uM explicitly"
        )
    return SolubilityModel.load(path)
