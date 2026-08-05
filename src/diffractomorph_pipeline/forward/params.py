"""Model parameters and the starting size distribution.

Defines the model's inputs as dataclasses of DEFAULTS you can override — no config file is read.
Three containers: :class:`Parameters` (physical/chemical constants), :class:`Diffusivities`
(per-species diffusion coefficients), and :class:`PSD` (the starting size distribution, supplied
from data). Each default's source is noted inline where the field is declared.

Override any field with :func:`dataclasses.replace`, or by passing it to ``predict``::

    p = replace(Parameters(), s0_uM=0.35, v_diss_mL=20.0)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# 25 °C → 37 °C Stokes–Einstein scaling for water (D ∝ T/η):
#   (310.15 / 298.15 K) × (η_25 / η_37) = 1.040 × (0.890 / 0.691 mPa·s) ≈ 1.339.
_T37 = 1.339


# ── per-species diffusivities (cm²/s) ────────────────────────────────────────

@dataclass(frozen=True)
class Diffusivities:
    dh: float       # H+
    doh: float      # OH-
    dd: float       # drug (CFZ free base / BH+)
    da: float       # acetate OAc-
    dha: float      # acetic acid HOAc
    dbor: float     # borate B(OH)4-
    dborh: float    # boric acid B(OH)3
    dh3po4: float
    dh2po4: float
    dhpo4: float
    dpo4: float


def default_diffusivities(t37: float = _T37) -> Diffusivities:
    """Aqueous diffusion coefficients (cm²/s), all brought to 37 °C via two conventions:

    - **Buffer anions** (acetate, phosphate species): limiting-dilution values from the CRC
      Handbook (Vanýsek, "Ionic Conductivity and Diffusion at Infinite Dilution", 1992/93),
      tabulated at **25 °C**, scaled to 37 °C by the Stokes–Einstein factor ``t37``.
    - **H⁺ and OH⁻**: values already at **37 °C** from Al-Gousous et al. (2019) Mol. Pharm.
      16(6), 2626 (Table 2, "all at 37 °C"; originally Sheng, McNamara & Amidon, Mol. Pharm.
      2009; ultimately Lange's Handbook of Chemistry, Dean). H⁺/OH⁻ move by Grotthuss proton-
      hopping, not Stokes–Einstein, so they are **not** ``t37``-scaled.
    - **Neutral acids / borate** (H₃PO₄, boric acid, borate): infinite-dilution (c→0) values at
      **25 °C** from primary sources (Leaist 1984; Park & Lee 1994; Arcis 2016), ``t37``-scaled.
    - **CFZ (drug)**: no measured aqueous D exists; a Wilke–Chang estimate at 37 °C (see the
      ``dd`` comment).

    Every value is sourced inline to a primary paper or authoritative compilation.
    """
    return Diffusivities(
        # H⁺ / OH⁻ — 37 °C values (Grotthuss transport, not t37-scaled); Al-Gousous 2019
        # Mol Pharm 16(6):2626 Table 2, from Sheng et al. 2009 / Lange's Handbook (Dean):
        dh=104.9e-6,           # H⁺
        doh=63e-6,             # OH⁻
        # CFZ (drug) — no measured aqueous D; 37 °C estimate, ~4–6e-6. Wilke–Chang (AIChE J
        # 1955, 1:264) = 5.4e-6; Stokes–Einstein cross-check = 6.3e-6. Free base dimerizes in
        # water (Verbić 2023, Mol Pharm 20:3160), so the diffusing species may be slower.
        dd=5.4e-6,
        # buffer anions — CRC/Vanýsek limiting ionic-conductivity values at 25 °C, ×t37:
        da=1.089e-5 * t37,     # acetate CH₃COO⁻
        dh2po4=8.46e-6 * t37,  # H₂PO₄⁻
        dhpo4=6.9e-6 * t37,    # HPO₄²⁻
        dpo4=6.1e-6 * t37,     # PO₄³⁻
        # neutral acids / borate — infinite-dilution (c→0) values at 25 °C, ×t37:
        dha=1.201e-5 * t37,    # acetic acid CH₃COOH (neutral monomer) — Vitagliano & Lyons 1956, JACS 78:4538
        dh3po4=0.87e-5 * t37,  # phosphoric acid H₃PO₄ — Leaist 1984, J Chem Soc Faraday Trans I 80:3041
        dborh=1.64e-5 * t37,   # boric acid B(OH)₃ — Park & Lee 1994, J Chem Eng Data 39:891
        dbor=9.2e-6 * t37,     # borate B(OH)₄⁻ — Arcis 2016, Phys Chem Chem Phys 18:24081 (Nernst–Einstein)
    )


# ── the adjustable parameter set ─────────────────────────────────────────────

@dataclass
class Parameters:
    """Every tunable knob of the mechanistic CFZ dissolution model."""
    # drug
    mw: float = 473.4                 # CFZ molar mass, g/mol (literature)
    rho_g_cm3: float = 1.30           # CFZ free-base Form II true density, g/cm³ (flotation;
                                      # X-ray D_calc 1.331) — Bannigan et al. 2016, Cryst Growth Des 16:7240
    pka_bh: float = 6.08              # CFZ BH+ apparent solubility-pH pKa′ (Woldemichael et al.)
    # intrinsic free-base solubility, µM — Woldemichael 2018, Sci Rep 8:2934 (H–H fit).
    # predict() replaces this with the measured pH-specific Cs from solubility.py.
    s0_uM: float = 0.48
    # medium — Britton–Robinson buffer (0.04 M in each acid; the experimental recipe)
    kw: float = 2.39e-14              # water ion product Kw at 37 °C (pKw 13.62; CRC Handbook)
    phosphate_M: float = 0.04         # buffer acid molarities, M (experimental)
    acetate_M: float = 0.04
    borate_M: float = 0.04
    pka_phosphate: tuple = (2.16, 7.18, 12.1)  # thermodynamic (I=0) at 37 °C — Bates 1951 (pK1),
                                               #   Bates & Acree 1943/45 (pK2), IUPAC/NIST (pK3)
    pka_acetate: float = 4.76                  # thermodynamic (I=0) at 37 °C — Harned & Ehlers 1933
    pka_borate: float = 9.13                   # thermodynamic (I=0) at 37 °C — Manov et al. 1944
    v_diss_mL: float = 40.0           # dissolution cell volume, mL (experimental setup)
    # kinetics
    regime: str = "transport"         # dissolution rate-limiting step. Only "transport"
                                      # (diffusion-limited, τ∝r², h≈r) is implemented; "surface"
                                      # (reaction/detachment-limited, τ∝r) raises NotImplementedError.
    freeze_frac: float = 0.005        # numerical guard: a bin stops dissolving below this remnant
    diffusivities: Diffusivities = field(default_factory=default_diffusivities)  # per-species D (see default_diffusivities)
    # (Exploratory morphology knobs — the w_por rate blend and the f_acc depth cap — live in
    #  forward/morphology.py, not here; the base model is pure transport-limited Nernst–Brunner.)

    # ── derived equilibrium constants ────────────────────────────────────────
    @property
    def kad(self) -> float:          # CFZ BH+
        return 10.0 ** (-self.pka_bh)

    @property
    def s0_mol(self) -> float:       # intrinsic base solubility (mol/L)
        return self.s0_uM * 1e-6

    @property
    def kaa(self) -> float:          # acetate
        return 10.0 ** (-self.pka_acetate)

    @property
    def kab(self) -> float:          # borate
        return 10.0 ** (-self.pka_borate)

    @property
    def kap(self) -> tuple:          # phosphate (K1, K2, K3)
        return tuple(10.0 ** (-p) for p in self.pka_phosphate)

    @property
    def rho_mol(self) -> float:      # molar density of solid (mmol/cm³)
        return self.rho_g_cm3 * 1000.0 / self.mw

    @property
    def regime_n(self) -> int:
        return {"transport": 2, "surface": 1}[self.regime]

    def with_s0_ugml(self, s0_ugml: float) -> "Parameters":
        """Copy with S0 set from a µg/mL value (e.g. ``solubility.s0_for_ph_ugml(pH)``)."""
        return replace(self, s0_uM=s0_ugml / 1e3 / self.mw * 1e6)


# ── starting particle-size distribution (q3) ─────────────────────────────────

@dataclass
class PSD:
    """Starting size distribution: representative diameters + q3 volume fractions.

    ``volfrac`` is the per-bin volume (mass) fraction — the Sympatec q3 — and is what
    splits the dose across sizes. Swap this object to run the model on a different PSD.
    """
    diam_um: np.ndarray
    volfrac: np.ndarray
    number_frac: np.ndarray | None = None

    def __post_init__(self):
        self.diam_um = np.asarray(self.diam_um, float)
        self.volfrac = np.asarray(self.volfrac, float)
        if self.diam_um.shape != self.volfrac.shape:
            raise ValueError("diam_um and volfrac must be the same length")

    @classmethod
    def from_q3(cls, diam_um, volfrac, number_frac=None) -> "PSD":
        return cls(np.asarray(diam_um, float), np.asarray(volfrac, float), number_frac)

    @classmethod
    def from_sympatec(cls, path, dist_type: str | None = None) -> "PSD":
        """Read a Sympatec PSD export (CSV / folder of per-frame CSVs / PDF) into a PSD.

        Reuses ``optics.standards.read_number_psd`` (number fraction on the R3 31-class
        grid), then forms the q3 volume fractions ``volfrac ∝ number·d³``. The exported
        distribution kind (q0/q1/q2/q3) is auto-detected from the PAQXOS ``Qn / %`` header
        (or the filename); pass ``dist_type`` only to override a headerless file — it
        defaults to q3, the usual volume export. The starting *injected* suspension.
        """
        from diffractomorph_pipeline.optics import standards
        n = np.asarray(standards.read_number_psd(path, dist_type=dist_type), float)
        diam = np.asarray(standards.GRID, float)
        v = n * diam ** 3
        return cls(diam, v / v.sum() if v.sum() > 0 else v, number_frac=n)

    @property
    def volfrac_norm(self) -> np.ndarray:
        s = self.volfrac.sum()
        return self.volfrac / s if s > 0 else self.volfrac

    @property
    def dv50(self) -> float:
        """Volume-median diameter Dv50 (µm)."""
        d = self.diam_um; v = self.volfrac_norm
        o = np.argsort(d)
        return float(np.interp(0.5, np.cumsum(v[o]), d[o]))

    @property
    def d32(self) -> float:
        """Sauter mean D32 (µm) = Σ n d³ / Σ n d² ; from volume via n ∝ v/d³."""
        d = self.diam_um; n = np.where(d > 0, self.volfrac_norm / d ** 3, 0.0)
        num = float((n * d ** 3).sum()); den = float((n * d ** 2).sum())
        return num / den if den > 0 else float("nan")
