"""Mie kernel — use side (loader + forward operator + size QC primitive).

The kernel maps a number-PSD to a predicted 31-channel scattering pattern,
``I = A·n``. It is **shape-calibrated, not absolute**: ``A`` is fixed only up to
an overall scale (both ``I`` and ``n`` were normalized in the fit), so the kernel
gives pattern *shape* and *relative* PSD — never absolute number concentration.
Absolute mass comes from UV-Vis, not the kernel.

There is deliberately **no ``invert()``** (spec §3): naive channel→PSD inversion
is ill-posed for this system (~67 % of particles sit in the sub-0.9 µm floor bin
below R3 number resolution), and empirically (20260609) NNLS gave non-physical
PSDs. Any PSD recovery must be a *constrained forward fit* with an external shape
prior, implemented elsewhere and documented as such.

Spec module path was ``src/dissolution_optics/optics/mie.py``; reconciled to this
package as ``diffractomorph_pipeline/optics/mie.py``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

# Spec §6 defaults.
VALID_SIZE_MAX_UM = 15.0      # trustworthy span for any size reporting
CONSISTENCY_PASS_R = 0.97     # size-QC PASS threshold
DEFAULT_KERNEL = "R3_CFZ_20260601_1.71.npz"


@dataclass
class MieKernel:
    """Versioned optical forward operator (spec §3).

    ``n_cfz`` is the fitted drug refractive index — named for the reference drug
    (CFZ) per the spec; for other drugs it holds that drug's RI.
    """
    A: np.ndarray            # (C × S) forward operator: I = A·n
    xm: np.ndarray           # (S,) size grid (µm)
    theta: np.ndarray        # (C,) per-channel scattering angles (deg)
    char_size: np.ndarray    # (C,) channel → characteristic size map (µm)
    n_cfz: float             # fitted drug refractive index
    meta: dict = field(default_factory=dict)

    def save(self, path: Path | str) -> Path:
        """Write the kernel artifact (.npz with a JSON-encoded ``meta``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, A=self.A, xm=self.xm, theta=self.theta,
            char_size=self.char_size, n_cfz=np.array(self.n_cfz),
            meta_json=json.dumps(self.meta),
        )
        return path

    @classmethod
    def from_npz(cls, path: Path | str) -> "MieKernel":
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta_json"])) if "meta_json" in d.files else {}
        return cls(
            A=d["A"], xm=d["xm"], theta=d["theta"], char_size=d["char_size"],
            n_cfz=float(d["n_cfz"]), meta=meta,
        )


def _kernels_dir() -> Path:
    return Path(resources.files("diffractomorph_pipeline")) / "data" / "kernels"


def load_kernel(path: Path | str | None = None) -> MieKernel:
    """Load an explicit kernel or the optional legacy default."""
    selected = Path(path) if path is not None else _kernels_dir() / DEFAULT_KERNEL
    if not selected.exists():
        raise FileNotFoundError(
            "the optional optical kernel is not installed; pass an explicit kernel path"
        )
    return MieKernel.from_npz(selected)


def forward(kernel: MieKernel, n: np.ndarray) -> np.ndarray:
    """Number-PSD ``n`` → predicted 31-channel intensity ``I = A·n`` (normalized)."""
    n = np.asarray(n, dtype=float)
    I = kernel.A @ n
    s = I.sum()
    return I / s if s else I


def channel_size_map(kernel: MieKernel) -> np.ndarray:
    """Per-channel characteristic size (µm).

    Trustworthy over ~ch5–29; the map plateaus at both ends (ch1–5 ≈ largest,
    ch29–31 ≈ smallest), which are weakly constrained by the floor bin.
    """
    return kernel.char_size.copy()


def shape_consistency(I_obs: np.ndarray, I_ref: np.ndarray) -> float:
    """Pearson r between two normalized channel patterns — the size-QC primitive.

    r near 1 → same PSD shape (size unchanged); a drop flags a prep/size change.
    Compare against :data:`CONSISTENCY_PASS_R`.
    """
    a = np.asarray(I_obs, dtype=float)
    b = np.asarray(I_ref, dtype=float)
    a = a / a.sum() if a.sum() else a
    b = b / b.sum() if b.sum() else b
    return float(np.corrcoef(a, b)[0, 1])
