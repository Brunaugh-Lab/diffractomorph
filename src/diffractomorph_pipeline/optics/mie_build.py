"""Mie kernel — build / calibration side (spec §1, §1b, §3a, §3c).

The build factorizes into two independently-cadenced fits:

- **Geometry** (θ_min, θ_max) — from the **weekly NIST glass-bead** standard
  (RI 1.51, non-absorbing → no RI confound). Re-fit only if the weekly NIST
  drifts past tolerance; reused otherwise.
- **Refractive index** n_drug — from the drug dispersed in its **antisolvent**
  (a non-dissolving, time-invariant dispersion), at fixed geometry. Per drug /
  polymorph / lot. CFZ's antisolvent is pH 7 buffer (CFZ ~ insoluble there).

``build_kernel`` assembles ``A[c,i] = per-particle Mie intensity × ring solid
angle (≈ θ²)``; the ring-solid-angle weighting is essential (forward-peaked Mie
would otherwise fall toward ch31). ``resolve_kernel_for_day`` selects/rebuilds the
kernel valid for a given date & drug, before any optical stage runs.

Engine: ``miepython``. Spec path ``src/dissolution_optics/optics/mie_build.py``.
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import miepython as mp
import numpy as np
import yaml
from scipy.optimize import least_squares

from diffractomorph_pipeline.optics.mie import MieKernel
from diffractomorph_pipeline.optics import standards
from diffractomorph_pipeline.optics.standards import GRID as DEFAULT_GRID
from diffractomorph_pipeline.utils import is_interactive

# ── Physical constants (Sympatec HELOS R3, He-Ne) ────────────────────────────
LAMBDA_UM = 0.6328            # He-Ne wavelength (µm)
N_MED = 1.331                 # water medium refractive index
GLASS_RI = 1.51               # NIST glass-bead refractive index (non-absorbing)
N_CHANNELS = 31

# Detector inner angle is HARDWARE-FIXED by the lens, not fit from QC. Fitting it
# from floor-bin-dominated standards left it ~13× too large (θ_min=2.55° vs ~0.19°),
# collapsing the channel→size map to ≤5.5 µm when R3 resolves to 175 µm. θ_min comes
# from each lens's range upper limit via the diffraction relation θ≈1.22·λ/(n_med·x):
# R3 (f=100mm, x_max=175µm) → 0.19°; R4 (f=200mm) → half. See the lab note
# `R3_Detector_Geometry_and_Channel_Size_Map_Correction`. θ_max stays fitted (the
# abundant fines constrain the large-angle end well).
LENS_THETA_MIN_DEG = {"R3": 0.19, "R4": 0.095, "R5": 0.038}

# Stored per-drug refractive index library (fit from the antisolvent QC when new).
RI_LIBRARY = {"CFZ": 1.71}
# Drug-specific antisolvent (the medium where the drug is effectively insoluble).
ANTISOLVENT = {"CFZ": "pH 7 Britton-Robinson buffer"}


@dataclass
class Geometry:
    """Detector geometry fitted from the NIST glass-bead standard."""
    theta_min: float
    theta_max: float
    theta: np.ndarray         # (C,) log-spaced per-channel angles (deg)
    lens: str
    cal_date: str             # NIST QC date this geometry came from (YYYY-MM-DD)
    r_bead: float             # fit correlation


# ── Mie forward core ─────────────────────────────────────────────────────────

def _build_A(theta: np.ndarray, m: complex, xm: np.ndarray) -> np.ndarray:
    """``A[c,i]`` = per-particle unpolarized Mie intensity × ring solid angle (≈θ²)."""
    mu = np.cos(np.deg2rad(theta))
    x = np.pi * xm * N_MED / LAMBDA_UM
    A = np.empty((len(theta), len(xm)))
    for i in range(len(xm)):
        A[:, i] = mp.i_unpolarized(m, x[i], mu)
    return A * (theta ** 2)[:, None]


def _predict(thmin, thmax, m, n, xm, nch=N_CHANNELS):
    theta = np.logspace(np.log10(thmin), np.log10(thmax), nch)
    A = _build_A(theta, m, xm)
    Ip = A @ n
    return Ip / Ip.sum(), theta, A


def _norm(v):
    v = np.asarray(v, dtype=float)
    s = v.sum()
    return v / s if s else v


# ── The two fits (spec §3a) ──────────────────────────────────────────────────

def fit_geometry(nist_glass_qc, nist_number_psd, lens=None, *,
                 cal_date, xm=DEFAULT_GRID) -> Geometry:
    """Fit θ_min, θ_max from the NIST glass-bead standard (RI 1.51, non-absorbing).

    ``nist_glass_qc`` is the measured 31-channel intensity; ``nist_number_psd`` is
    the bead **number** fraction over the size grid. Geometry is isolated here
    (non-absorbing standard) so the later RI fit cannot confound it.
    """
    lens = _lens_id(lens)
    cal_date = _calibration_date_string(cal_date)
    I = _norm(nist_glass_qc)
    n = _norm(nist_number_psd)
    m = complex(GLASS_RI, 0) / N_MED
    thmin_hw = LENS_THETA_MIN_DEG.get(lens)
    if thmin_hw is not None:
        # θ_min is hardware (see LENS_THETA_MIN_DEG); fit only θ_max, which the
        # abundant sub-µm fines constrain well. The QC then validates geometry (r)
        # and supplies the RI — it no longer fits the ill-posed inner angle.
        res = least_squares(
            lambda p: _predict(thmin_hw, 10 ** p[0], m, n, xm)[0] - I,
            [np.log10(30)], bounds=([np.log10(12)], [np.log10(55)]),
        )
        thmin, thmax = thmin_hw, 10 ** res.x[0]
    else:
        res = least_squares(
            lambda p: _predict(10 ** p[0], 10 ** p[1], m, n, xm)[0] - I,
            [np.log10(0.3), np.log10(30)],
            bounds=([np.log10(0.05), np.log10(8)], [np.log10(5), np.log10(60)]),
        )
        thmin, thmax = 10 ** res.x[0], 10 ** res.x[1]
    Ip, theta, _ = _predict(thmin, thmax, m, n, xm)
    r = float(np.corrcoef(Ip, I)[0, 1])
    return Geometry(thmin, thmax, theta, lens, cal_date, r)


def fit_refractive_index(drug_antisolvent_qc, drug_number_psd, geometry: Geometry,
                         drug: str, xm=DEFAULT_GRID) -> tuple[float, float]:
    """Fit n_drug at fixed geometry from the drug-in-antisolvent (non-dissolving) QC.

    Returns ``(n_drug, r)``. The antisolvent dispersion is non-dissolving so its
    PSD is stable, making ``forward(PSD) ≈ measured pattern`` well-posed.
    """
    I = _norm(drug_antisolvent_qc)
    n = _norm(drug_number_psd)
    res = least_squares(
        lambda p: _predict(geometry.theta_min, geometry.theta_max,
                           complex(p[0], 0) / N_MED, n, xm)[0] - I,
        [1.7], bounds=([1.4], [2.2]),
    )
    n_drug = float(res.x[0])
    Ip, _, _ = _predict(geometry.theta_min, geometry.theta_max,
                        complex(n_drug, 0) / N_MED, n, xm)
    r = float(np.corrcoef(Ip, I)[0, 1])
    return n_drug, r


def build_kernel(geometry: Geometry, n_drug: float, drug: str,
                 calibration_psd=None, r_drug=None, size_grid=DEFAULT_GRID) -> MieKernel:
    """Assemble the drug's forward operator + channel→size map into a MieKernel."""
    _calibration_date_string(geometry.cal_date)
    xm = np.asarray(size_grid, dtype=float)
    m = complex(n_drug, 0) / N_MED
    A = _build_A(geometry.theta, m, xm)
    # Channel→size map: per-particle intensity (un-θ²-weighted) argmax per channel,
    # evaluated at the non-absorbing glass RI — the map is a geometry property. With
    # the hardware-anchored θ_min it spans the full R3 range (ch1≈70 → ch31≈0.67 µm),
    # not the old collapsed ≤5.5 µm.
    A_glass = _build_A(geometry.theta, complex(GLASS_RI, 0) / N_MED, xm)
    A_pp = A_glass / (geometry.theta ** 2)[:, None]
    char = np.array([xm[np.argmax(A_pp[c])] for c in range(A.shape[0])])
    th_hw = LENS_THETA_MIN_DEG.get(geometry.lens)
    meta = {
        "drug": drug, "lens": geometry.lens, "n_drug": n_drug,
        "geometry_cal_date": geometry.cal_date,
        "theta_min": geometry.theta_min, "theta_max": geometry.theta_max,
        "theta_min_source": ("hardware (lens range)" if th_hw is not None
                             else "fit"),
        "r_bead": geometry.r_bead, "r_cfz": r_drug,
        "kernel_id": kernel_id(geometry.lens, drug, geometry.cal_date, n_drug),
    }
    if calibration_psd is not None:
        meta["n_drug_psd"] = np.asarray(calibration_psd, dtype=float).tolist()
    return MieKernel(A=A, xm=xm, theta=geometry.theta, char_size=char,
                     n_cfz=n_drug, meta=meta)


def kernel_id(lens: str, drug: str, geometry_cal_date: str, n_drug: float) -> str:
    return f"{lens}_{drug}_{geometry_cal_date.replace('-', '')}_{n_drug:.2f}"


# ── Registry + per-day resolution (spec §3c) ─────────────────────────────────

def _external_registry_path(path: Path | str | None) -> Path:
    if path is None:
        raise ValueError(
            "registry is required; choose an explicit writable registry.yaml outside "
            "the installed package"
        )
    selected = Path(path).expanduser().resolve()
    package_root = Path(resources.files("diffractomorph_pipeline")).resolve()
    if selected == package_root or package_root in selected.parents:
        raise ValueError("registry must be outside the installed diffractomorph_pipeline package")
    return selected


def load_registry(path: Path | str | None = None) -> dict:
    path = _external_registry_path(path)
    if not path.exists():
        return {"kernels": []}
    return yaml.safe_load(path.read_text()) or {"kernels": []}


def _as_date(d) -> _dt.date:
    if isinstance(d, _dt.date):
        return d
    return _dt.date.fromisoformat(str(d))


def _calibration_date_string(value) -> str:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError("cal_date is required and must use YYYY-MM-DD")
    try:
        parsed = _dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("cal_date is required and must use YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise ValueError("cal_date is required and must use YYYY-MM-DD")
    return value


def _lens_id(value) -> str:
    if value not in LENS_THETA_MIN_DEG:
        allowed = ", ".join(sorted(LENS_THETA_MIN_DEG))
        raise ValueError(f"lens is required and must be one of: {allowed}")
    return str(value)


def resolve_kernel_for_day(date, drug, registry=None, lens=None, qc_dir=None,
                           rebuild=False, confirm=None) -> MieKernel:
    """Pick (or build) the kernel valid for ``date`` & ``drug`` (spec §3c).

    Search → reuse → build:

    1. Search the registry for a kernel for ``drug``+``lens`` whose geometry is
       from the most recent NIST on/before ``date``. If found and not
       ``rebuild``: reuse it (when interactive, confirm first).
    2. Otherwise build from the QC files in ``qc_dir`` — the NIST + drug
       ``data_<sample>_<type>`` files (see :mod:`optics.standards`) — then
       register and return it.

    ``confirm`` overrides interactivity (True/False); default = auto-detect a TTY.
    """
    lens = _lens_id(lens)
    rpath = _external_registry_path(registry)
    reg = load_registry(rpath)
    reg_dir = rpath.parent
    target = _as_date(date)

    candidates = [
        e for e in reg.get("kernels", [])
        if e["keys"].get("drug") == drug and e["keys"].get("lens") == lens
        and _as_date(e["keys"]["geometry_cal_date"]) <= target
    ]
    if candidates and not rebuild:
        best = max(candidates, key=lambda e: _as_date(e["keys"]["geometry_cal_date"]))
        if _ask_reuse(best, confirm):
            artifact_name = Path(best["file"])
            if artifact_name.name != best["file"]:
                raise ValueError("kernel registry file entries must be basenames")
            return MieKernel.from_npz(reg_dir / artifact_name)

    # Build from the QC files the user points to.
    if qc_dir is None:
        raise FileNotFoundError(
            f"No usable kernel for drug={drug} lens={lens} on/before {target}, and no "
            "qc_dir supplied to build one. Point qc_dir at the NIST + drug QC files "
            "(data_<sample>_<type>.{rtf,csv}) or run dfm-build-kernel."
        )
    return build_kernel_from_qc(qc_dir, drug, lens=lens, cal_date=str(target),
                                registry=registry)


def build_kernel_from_files(nist_intensity, nist_psd, drug, lens=None, cal_date=None,
                            drug_intensity=None, drug_psd=None,
                            nist_psd_type="q0", drug_psd_type="q0",
                            fit_ri=False, registry=None) -> MieKernel:
    """Build + register a kernel from explicit QC file paths (real-data path).

    ``*_psd`` may each be a single CSV/PDF or a directory of per-frame CSVs.
    NIST → geometry (always). RI: stored library value for a known drug, or fit
    from the drug-in-antisolvent standard when ``fit_ri`` or the drug is new.
    """
    lens = _lens_id(lens)
    registry = _external_registry_path(registry)
    cal_date = _calibration_date_string(cal_date)
    geom = fit_geometry(
        standards.read_qc_intensity(nist_intensity),
        standards.read_number_psd(nist_psd, dist_type=nist_psd_type),
        lens=lens, cal_date=cal_date,
    )
    if fit_ri or drug not in RI_LIBRARY:
        if drug_intensity is None or drug_psd is None:
            raise FileNotFoundError(
                f"Fitting refractive index for {drug} needs both drug intensity and "
                "drug PSD files.")
        dpsd = standards.read_number_psd(drug_psd, dist_type=drug_psd_type)
        n_drug, r_drug = fit_refractive_index(
            standards.read_qc_intensity(drug_intensity), dpsd, geom, drug)
    else:
        n_drug, r_drug, dpsd = RI_LIBRARY[drug], None, None
    kernel = build_kernel(geom, n_drug, drug, calibration_psd=dpsd, r_drug=r_drug)
    _register(kernel, registry)
    return kernel


def build_kernel_from_qc(qc_dir, drug, lens=None, cal_date=None, fit_ri=False,
                         registry=None) -> MieKernel:
    """Discover ``data_<sample>_<type>`` files in ``qc_dir`` and build a kernel.

    Convenience over :func:`build_kernel_from_files` for folders that follow the
    naming convention.
    """
    lens = _lens_id(lens)
    registry = _external_registry_path(registry)
    cal_date = _calibration_date_string(cal_date)
    qc = standards.discover_qc_files(qc_dir, drug, need_drug=fit_ri or drug not in RI_LIBRARY)
    return build_kernel_from_files(
        qc.nist.intensity, qc.nist.psd, drug, lens=lens, cal_date=cal_date,
        drug_intensity=qc.drug.intensity if qc.drug else None,
        drug_psd=qc.drug.psd if qc.drug else None,
        nist_psd_type=qc.nist.psd_type,
        drug_psd_type=qc.drug.psd_type if qc.drug else "q0",
        fit_ri=fit_ri, registry=registry)


def _ask_reuse(entry, confirm) -> bool:
    """Reuse-or-rebuild decision for a found kernel (interactive confirm)."""
    interactive = is_interactive() if confirm is None else confirm
    if not interactive:
        return True
    fq = entry.get("fit_quality", {})
    print(f"Found kernel '{entry['kernel_id']}' "
          f"(geometry {entry['keys']['geometry_cal_date']}, "
          f"r_bead={fq.get('r_bead')}, r_cfz={fq.get('r_cfz')}).")
    return input("Use it? [Y/n] ").strip().lower() not in ("n", "no")


def _register(kernel: MieKernel, registry_path) -> Path:
    """Write the kernel artifact and append/refresh its registry entry."""
    rpath = _external_registry_path(registry_path)
    reg_dir = rpath.parent
    kid = kernel.meta["kernel_id"]
    reg_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{kid}.npz"
    kernel.save(reg_dir / fname)

    reg = load_registry(rpath)
    reg.setdefault("kernels", [])
    reg["kernels"] = [e for e in reg["kernels"] if e["kernel_id"] != kid]
    reg["kernels"].append({
        "kernel_id": kid,
        "file": fname,
        "keys": {
            "lens": kernel.meta["lens"], "drug": kernel.meta["drug"],
            "geometry_cal_date": kernel.meta["geometry_cal_date"],
            "n_drug": round(float(kernel.meta["n_drug"]), 2),
        },
        "fit_quality": {
            "r_bead": _round(kernel.meta.get("r_bead")),
            "r_cfz": _round(kernel.meta.get("r_cfz")),
        },
    })
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text(yaml.safe_dump(reg, sort_keys=False))
    return rpath


def _round(v, nd=3):
    return round(float(v), nd) if v is not None else None
