"""Britton–Robinson buffer chemistry and the bulk electroneutrality balance.

Equilibrium layer everything downstream rests on: speciation of the three BR weak acids
(phosphate / acetate / borate) and the protonated weak base (CFZ, BH+) as functions of [H+],
and the bulk charge-balance solve for [H+]_bulk. The spectator cation M+ (Na+ from the NaOH
titration) is fixed by the *initial* titration to the target pH, so the NaOH amount never has
to be supplied. All constants come from :class:`Parameters`. Part of the surface-pH dissolution
model — see :mod:`diffractomorph_pipeline.forward.surface_ode` for the model and references.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from diffractomorph_pipeline.forward.params import Parameters


@dataclass(frozen=True)
class Acid:
    """A weak acid, neutral when fully protonated (H_mA), with stepwise Ka's.

    Species j (0..m) has shed j protons and carries charge −j; abundance
    P_j = (Ka_1…Ka_j)/[H+]^j with P_0 = 1.
    """
    name: str
    total: float
    Kas: tuple

    def _P(self, H: float) -> np.ndarray:
        P = [1.0]; cum = 1.0
        for K in self.Kas:
            cum *= K
            P.append(cum / H ** len(P))
        return np.asarray(P)

    def speciation(self, H: float) -> np.ndarray:
        P = self._P(H)
        return self.total * P / P.sum()

    def anion_charge(self, H: float) -> float:
        """Negative charge concentration contributed (M, ≥0) = total·⟨protons lost⟩."""
        P = self._P(H); j = np.arange(P.size)
        return self.total * float((j * P).sum() / P.sum())


def acids_for(p: Parameters) -> list[Acid]:
    """The three Britton–Robinson acids at the parameter set's totals + pKa's."""
    return [
        Acid("phosphate", p.phosphate_M, p.kap),
        Acid("acetate", p.acetate_M, (p.kaa,)),
        Acid("borate", p.borate_M, (p.kab,)),
    ]


def drug_cation(H: float, c_drug: float, kad: float) -> float:
    """[DrugH+] = C_drug·[H+]/([H+]+Ka_BH) — protonated weak base, charge +1 (M)."""
    return c_drug * H / (H + kad)


def total_anion_charge(H: float, acids: list[Acid], kw: float) -> float:
    return kw / H + sum(a.anion_charge(H) for a in acids)


def charge_balance_residual(H, acids, mplus, kw, c_drug, kad) -> float:
    """Electroneutrality residual (cations − anions), M.  cations: H+ + M+ + DrugH+."""
    return (H + mplus + drug_cation(H, c_drug, kad)) - total_anion_charge(H, acids, kw)


def spectator_for_pH(p: Parameters, ph_target: float, c_drug: float = 0.0) -> float:
    """M+ (Na+, M) that puts the medium at ``ph_target`` (eliminates the NaOH unknown)."""
    H = 10.0 ** (-ph_target)
    return total_anion_charge(H, acids_for(p), p.kw) - H - drug_cation(H, c_drug, p.kad)


def solve_bulk_H(p: Parameters, mplus: float, c_drug: float = 0.0) -> float:
    """Solve the bulk charge balance for [H+]_bulk (M) given the spectator M+ and dissolved drug."""
    acids = acids_for(p)
    f = lambda H: charge_balance_residual(H, acids, mplus, p.kw, c_drug, p.kad)
    return brentq(f, 1e-14, 1.0, xtol=1e-20, rtol=1e-12)


def solve_bulk_pH(p: Parameters, mplus: float, c_drug: float = 0.0) -> float:
    return -float(np.log10(solve_bulk_H(p, mplus, c_drug)))
