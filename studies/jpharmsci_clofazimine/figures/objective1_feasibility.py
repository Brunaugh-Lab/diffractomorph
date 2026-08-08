"""JPharmSci figure: fixed-geometry optical-operator evaluation and resolution boundary.

Grant-facing (NSF CAREER) analysis + two-panel figure answering two questions about the
DiffractoMorph HELOS R3 observation operator, using the manual-derived physical-operator
implementation in :mod:`diffractomorph_pipeline.optics.mie_candidate`:

1. **Fixed-geometry evaluation (Panel A).** Can known scattering physics propagate an
   *independently certified* particle population (NIST SRM 1021) into an aggregate 31-channel
   detector response consistent with measurement? The HELOS R3 annular boundaries are reconstructed
   from the manufacturer's measuring-range table and Fraunhofer relation, then frozen before the
   NIST measurements are examined. The certified SRM-1021 Q3 distribution is propagated through the
   C_sca-inclusive Mie + annular-detector operator and compared with the normalized measured profile.

2. **Resolution boundary (Panel B).** Why does agreement for one broad standard NOT establish
   unique diameter→channel localization? Show that monodisperse diameters produce *distributed,
   overlapping* channel signatures, and quantify the operator's limited independent size information
   (effective rank, adjacent-column overlap).

This is an **evaluation and identifiability** analysis. It does NOT claim the production operator is
calibrated, that a single standard validates every diameter-response column, that detector channels
uniquely identify diameter, or that quantitative size-resolved flux is established. No detector
geometry parameter is fitted to NIST. Residual disagreement is retained because it is evidence of
the remaining observation-model boundary, not removed by an effective angular-scale fit.

The module is split into a pure compute core (operator build via ``mie_candidate``, the certified
number PSD, shape metrics, resolution diagnostics — all array-in/array-out and unit-testable with no
vault files), a thin data-loading layer (:func:`observed_net_shape` reads a PAQXOS ``.rtf`` via the
ingest layer), a renderer, and a ``main`` CLI that takes explicit input/output paths and never
depends on the working directory.

Reproduce / export::

    python studies/jpharmsci_clofazimine/figures/objective1_feasibility.py \
        --nist-rtf <first_session.rtf> --second-session-rtf <second_session.rtf> --output-dir <dir>

``--output-dir`` is required; the input defaults resolve from the configured data corpus
(``DFM_DATA_ROOT`` / ``.dfm.toml``). Outputs (into ``--output-dir``):
``optical_operator_feasibility_and_calibration_boundary.pdf`` (canonical vector) + ``.png`` (preview),
``objective1_metrics.json`` (metrics + geometry/RI assumptions + rank/overlap diagnostics +
provenance), and three source-data CSVs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from diffractomorph_pipeline import ingest
from diffractomorph_pipeline.config import corpus
from diffractomorph_pipeline.optics import mie_candidate as mc
from diffractomorph_pipeline.optics.standards import EDGES, GRID

# ── Certified SRM-1021 size truth — 1021.pdf Table 1 (cumulative VOLUME fraction finer) ──────────
# Soda-lime glass beads; certified x50,3 = 5.8 µm; supported span ≈ 2.1–12.9 µm. This is the
# INDEPENDENT particle-population truth (NOT a PAQXOS inverse). Identical to the values traced in the
# forward-operator audit (analysis/nist_forward_operator_audit.py).
CERT_PCT = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90], float)
CERT_D = np.array([2.1, 2.6, 3.0, 3.3, 3.7, 4.1, 4.5, 4.9, 5.3, 5.8, 6.3, 6.8, 7.4, 8.1, 8.9, 9.9,
                   11.1, 12.9], float)
CERT_X50_UM = 5.8
TAIL_CONVENTION = "log-linear quantile extrapolation"
MANUAL_SHA256 = "3de98054905e5dbc2df4659b53f802f188ea1b75c61c68876fa2102b9fe0b173"
CERTIFICATE_SHA256 = "a520543f1dad97d344efe9eabdafe70815dbc4d85d8a4d025937346d3fb6667d"

# The primary geometry is fixed from the manufacturer's R3 measuring-range table. The 31 class
# upper limits map to 31 nonzero detector-radius boundaries through r = 1.22 lambda_0 f / x; a
# central r=0 edge gives 31 annuli. The radii are converted to liquid-phase Mie angles with the
# sine-condition/Snell relation theta_med = asin[r/(n_med f)]. No geometry parameter is fitted to
# NIST. See ``mie_candidate.r3_manual_rings`` for the traceable construction.
RING_MAPPING = "medium"

# Panel-B representative monodisperse diameters (µm). 1 µm is below the certified span (deliberately —
# it shows the sub-2 µm signature the standard cannot constrain).
MONODISPERSE_UM = np.array([1.0, 2.0, 5.0, 10.0, 15.0])

# Resolution diagnostics are computed on the certified decision range only.
RESOLUTION_LO_UM = 2.0
RESOLUTION_HI_UM = 15.0

# Default measurement provenance (overridable on the CLI), resolved from the configured data corpus
# (DFM_DATA_ROOT / .dfm.toml) — never an absolute path. The first session evaluates the fixed
# operator; the second same-standard session provides a cross-session check without refitting. When the corpus is
# unconfigured these are non-existent sentinels, so `.exists()` guards skip rather than error.
_WEEKLY = corpus("disso_experiments", "ph_dependent_dissolution_study", "QC", "NIST_weekly_QCs")
DEFAULT_NIST_RTF = _WEEKLY / "NIST QC Week of 20260608.rtf"
DEFAULT_HELDOUT_RTF = _WEEKLY / "NIST QC Week of 20260601.rtf"
FIGURE_BASENAME = "optical_operator_feasibility_and_calibration_boundary"


# ── Certified distribution → number PSD (certificate only, no PAQXOS) ────────────────────────────

def certified_tail_anchors() -> tuple[float, float]:
    """Return prespecified 0 % and 100 % anchors outside the certified 5--90 % knots.

    The certificate does not specify the outer 0--5 % or 90--100 % tails. We therefore extend the
    first and last local slopes linearly in log-diameter to 0 % and 100 %. This preserves every
    certified knot, introduces no NIST detector data, and makes the tail convention explicit.
    """
    logd = np.log(CERT_D)
    low = np.exp(logd[0] + (0.0 - CERT_PCT[0]) * (logd[1] - logd[0]) /
                 (CERT_PCT[1] - CERT_PCT[0]))
    high = np.exp(logd[-1] + (100.0 - CERT_PCT[-1]) * (logd[-1] - logd[-2]) /
                  (CERT_PCT[-1] - CERT_PCT[-2]))
    return float(low), float(high)


def certified_cumulative(d_um: np.ndarray | float,
                         tail_anchors_um: tuple[float, float] | None = None) -> np.ndarray:
    """Certified cumulative volume fraction with explicit log-linear tails.

    PCHIP is confined to the certificate's reported 5--90 % knots. The unreported tails are linear
    in cumulative percentage versus log diameter between the prespecified 0/100 % anchors and the
    terminal certificate knots. Values outside the anchors are fixed at 0 or 100 %.
    """
    from scipy.interpolate import PchipInterpolator

    low, high = certified_tail_anchors() if tail_anchors_um is None else tail_anchors_um
    if not (0 < low < CERT_D[0] and high > CERT_D[-1]):
        raise ValueError("tail anchors must lie outside the certified 5--90 % diameter knots")
    d = np.asarray(d_um, float)
    q = np.zeros_like(d)
    positive = d > 0
    logd = np.zeros_like(d)
    logd[positive] = np.log(d[positive])

    low_tail = positive & (d >= low) & (d < CERT_D[0])
    q[low_tail] = CERT_PCT[0] * (
        (logd[low_tail] - np.log(low)) / (np.log(CERT_D[0]) - np.log(low))
    )

    interior = (d >= CERT_D[0]) & (d <= CERT_D[-1])
    pchip = PchipInterpolator(np.log(CERT_D), CERT_PCT, extrapolate=False)
    q[interior] = pchip(logd[interior])

    high_tail = (d > CERT_D[-1]) & (d <= high)
    q[high_tail] = CERT_PCT[-1] + (100.0 - CERT_PCT[-1]) * (
        (logd[high_tail] - np.log(CERT_D[-1])) /
        (np.log(high) - np.log(CERT_D[-1]))
    )
    q[d > high] = 100.0
    return np.clip(q, 0.0, 100.0)


def certified_number_psd(grid: np.ndarray = GRID, edges: np.ndarray = EDGES,
                         tail_anchors_um: tuple[float, float] | None = None) -> np.ndarray:
    """Nominal certified SRM-1021 **number** PSD on the R3 grid (n ∝ q3 / d³).

    PCHIP interpolation of the certified cumulative *volume* in log-diameter preserves every 5--90 %
    certificate knot. The unreported tails use explicit log-linear interpolation to the anchors from
    :func:`certified_tail_anchors`; the cumulative is then differenced on the R3 edges, converted to
    number, and normalized. No detector measurement or PAQXOS-derived size distribution enters this
    reconstruction.
    """
    Qe = certified_cumulative(edges, tail_anchors_um)
    Qe = np.maximum.accumulate(np.nan_to_num(Qe, nan=0.0))
    vol = np.empty(len(edges))
    vol[0] = Qe[0]
    vol[1:] = np.diff(Qe)
    vol = np.clip(vol, 0.0, None)
    num = vol / grid ** 3
    s = num.sum()
    return num / s if s else num


# ── Observed detector signal (from the .rtf, via ingest) ─────────────────────────────────────────

def observed_net_shape(rtf_path: Path | str) -> np.ndarray:
    """Normalized particle-signal channel shape from a PAQXOS NIST ``.rtf``.

    The particle signal is ``net = I.NORM − I.REF`` (reference-subtracted), averaged over the run's
    frames; negatives are clipped and the profile is normalized to sum 1. I.NORM alone is
    baseline-dominated, so the reference subtraction is what isolates the particle scattering.
    """
    run = ingest.extract_run(rtf_path)
    net = (np.asarray(run.I, float) - np.asarray(run.ref, float)[None, :]).mean(0)
    v = np.clip(net, 0.0, None)
    s = v.sum()
    return v / s if s else v


def observed_signal_summary(rtf_path: Path | str) -> dict:
    """Primary net profile, transform sensitivities, and clipping diagnostics for one RTF."""
    run = ingest.extract_run(rtf_path)
    frames = np.asarray(run.I, float)
    ref = np.asarray(run.ref, float)
    net_frames = frames - ref[None, :]
    net_mean = net_frames.mean(0)

    def _shape(v):
        x = np.clip(np.asarray(v, float), 0.0, None)
        return x / x.sum()

    return dict(
        primary=_shape(net_mean),
        measured_sensitivity=_shape(frames.mean(0)),
        net_over_ref_sensitivity=_shape(net_mean / ref),
        negative_channel_frame_fraction=float(np.mean(net_frames < 0.0)),
        negative_mean_channel_count=int(np.sum(net_mean < 0.0)),
        clipping_rule="average I minus stored reference over frames; clip negative channel means once; normalize",
    )


def _sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ── Physical operator (manual-derived annuli + C_sca-inclusive Mie response) ────────────────────

def detector_rings(n_med: float = mc.N_MED) -> mc.DetectorRings:
    """Return the frozen HELOS R3 annuli reconstructed from the manual's class limits.

    ``EDGES`` contains the 31 R3 class upper limits (0.9--175 µm). The helper converts those limits
    to detector-plane radii using the manual's Fraunhofer relation, adds the central radius, and maps
    the resulting 32 boundaries to liquid-phase angles. The geometry is independent of NIST data.
    """
    return mc.r3_manual_rings(EDGES, n_med=n_med, mapping=RING_MAPPING)


def build_physical_operator(diams: np.ndarray = GRID, n_quad: int = 64,
                            n_particle: float = mc.GLASS_RI,
                            n_med: float = mc.N_MED) -> np.ndarray:
    """Diameter→channel response matrix ``A`` (31 × D) on the frozen R3 annuli.

    Each entry is the particle scattering cross-section multiplied by the unpolarized Mie phase
    function integrated over one annulus. Optics are fixed at He–Ne λ = 0.6328 µm, soda-lime glass
    ``n=1.52``, and nominal aqueous-medium ``n=1.331`` unless explicitly changed for sensitivity.
    """
    return mc.response_matrix(
        diams, detector_rings(n_med), n_particle, n_med=n_med, n_quad=n_quad
    )


def predict_channel_shape(A: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Predicted normalized channel profile ``A·n`` (non-negative, summing to 1)."""
    I = np.clip(A @ n, 0.0, None)
    s = I.sum()
    return I / s if s else I


def shape_metrics(pred: np.ndarray, obs: np.ndarray) -> dict:
    """Compositional agreement between two normalized channel profiles (cosine, corr, total-variation)."""
    p = pred / pred.sum()
    q = obs / obs.sum()
    cos = float(p @ q / (np.linalg.norm(p) * np.linalg.norm(q)))
    corr = float(np.corrcoef(p, q)[0, 1])
    tv = float(0.5 * np.abs(p - q).sum())
    return dict(cosine=round(cos, 4), corr=round(corr, 4), tv=round(tv, 4))


def resolution_diagnostics(A: np.ndarray, grid: np.ndarray = GRID,
                           lo_um: float = RESOLUTION_LO_UM, hi_um: float = RESOLUTION_HI_UM) -> dict:
    """Independent-size-information diagnostics of a response matrix over ``[lo_um, hi_um]``.

    Effective rank (entropy of the singular spectrum), rank needed for 99 % of operator energy,
    condition number, and adjacent-column cosine — reported for BOTH the absolute columns (include the
    d² C_sca weighting) and the column-normalized matrix (channel-shape only). High adjacent-column
    cosine and low effective rank mean neighbouring diameters map to nearly the same channel pattern:
    the operator carries only a handful of independent size directions.
    """
    m = (grid >= lo_um) & (grid <= hi_um)
    Asub = A[:, m]

    def _svd(M):
        sv = np.linalg.svd(M, compute_uv=False)
        p = sv / sv.sum()
        eff = float(np.exp(-np.sum(np.where(p > 0, p * np.log(p), 0.0))))
        r99 = int(np.searchsorted(np.cumsum(sv ** 2) / np.sum(sv ** 2), 0.99) + 1)
        return round(eff, 2), r99, float(sv[0] / sv[-1])

    An = Asub / np.clip(np.linalg.norm(Asub, axis=0, keepdims=True), 1e-30, None)
    e_a, r_a, c_a = _svd(Asub)
    e_n, r_n, c_n = _svd(An)
    adj = np.array([float(An[:, i] @ An[:, i + 1]) for i in range(An.shape[1] - 1)])
    return dict(size_range_um=[lo_um, hi_um], n_size_columns=int(m.sum()),
                eff_rank_abs=e_a, rank99_abs=r_a, condition_abs=round(c_a, 1),
                eff_rank_colnorm=e_n, rank99_colnorm=r_n, condition_colnorm=round(c_n, 1),
                adjacent_col_cosine_median=round(float(np.median(adj)), 4),
                adjacent_col_cosine_max=round(float(adj.max()), 4),
                adjacent_col_cosine_min=round(float(adj.min()), 4))


def monodisperse_responses(diams: np.ndarray = MONODISPERSE_UM, n_quad: int = 128,
                           n_particle: float = mc.GLASS_RI,
                           n_med: float = mc.N_MED) -> dict:
    """Per-diameter normalized channel responses + centroid, FWHM (channels), and adjacent overlap.

    Uses the same fixed manual-derived geometry as Panel A. Each monodisperse diameter yields a
    distributed response (never one channel); neighbouring diameters overlap strongly.
    """
    A = mc.response_matrix(
        diams, detector_rings(n_med), n_particle, n_med=n_med, n_quad=n_quad
    )
    ch = np.arange(1, mc.N_CHANNELS + 1)
    P = A / np.clip(A.sum(0, keepdims=True), 1e-30, None)      # per-diameter normalized
    cent = (P * ch[:, None]).sum(0)
    fwhm = (P >= P.max(0) * 0.5).sum(0)
    adj = [float(P[:, i] @ P[:, i + 1] / (np.linalg.norm(P[:, i]) * np.linalg.norm(P[:, i + 1])))
           for i in range(len(diams) - 1)]
    return dict(diams_um=diams, profiles=P, centroid_ch=cent, fwhm_ch=fwhm.astype(int),
                adjacent_pair_cosine=adj)


# ── Assembled result ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Objective1Result:
    """Everything the figure and the machine-readable exports need."""
    measured: np.ndarray                 # observed normalized channel shape (first session)
    predicted: np.ndarray                # predicted normalized channel shape (certified PSD)
    n_cert: np.ndarray                   # certified number PSD on the R3 grid
    feasibility: dict                    # fixed-geometry metrics + second-session repeatability
    boundary: dict                       # resolution diagnostics
    monodisperse: dict                   # per-diameter responses
    assumptions: dict                    # geometry + refractive-index + optics
    provenance: dict                     # measurement files, certificate, grid
    meta: dict = field(default_factory=dict)


def compute_objective1(nist_rtf: Path | str = DEFAULT_NIST_RTF,
                       second_session_rtf: Path | str | None = DEFAULT_HELDOUT_RTF) -> Objective1Result:
    """Full Objective-1 computation from measurement files + the certified certificate.

    Builds the manual-derived geometry without consulting either measurement, propagates the
    certified PSD, scores agreement against ``nist_rtf``, applies the same fixed operator to
    ``second_session_rtf`` (same standard, different session), and computes the resolution diagnostics.
    """
    n_cert = certified_number_psd()
    sig = observed_signal_summary(nist_rtf)
    obs = sig["primary"]
    A = build_physical_operator(n_quad=96)
    predicted = predict_channel_shape(A, n_cert)
    primary = shape_metrics(predicted, obs)
    feasibility = dict(
        geometry="manual-derived R3 annuli; no parameter fitted to NIST",
        cosine=primary["cosine"], corr=primary["corr"], tv=primary["tv"],
        negative_channel_frame_fraction=round(sig["negative_channel_frame_fraction"], 6),
        negative_mean_channel_count=sig["negative_mean_channel_count"],
        signal_transform_sensitivity=dict(
            measured_value=shape_metrics(predicted, sig["measured_sensitivity"]),
            net_over_reference=shape_metrics(predicted, sig["net_over_ref_sensitivity"]),
        ),
    )

    tail_sensitivity = {}
    tail_predictions = {}
    for label, anchors in {
        "compact_2.05_13.0_um": (2.05, 13.0),
        "R3_bracket_1.8_15.0_um": (1.8, 15.0),
    }.items():
        pred_alt = predict_channel_shape(A, certified_number_psd(tail_anchors_um=anchors))
        tail_predictions[label] = pred_alt
        tail_sensitivity[label] = shape_metrics(pred_alt, obs)
    feasibility["certified_tail_sensitivity"] = tail_sensitivity

    arctan_rings = mc.r3_manual_rings(EDGES, n_med=mc.N_MED, mapping="arctan")
    arctan_A = mc.response_matrix(GRID, arctan_rings, mc.GLASS_RI, n_med=mc.N_MED, n_quad=96)
    arctan_pred = predict_channel_shape(arctan_A, n_cert)
    feasibility["angle_mapping_sensitivity_arctan"] = shape_metrics(arctan_pred, obs)
    feasibility["reversed_channel_order_cosine"] = shape_metrics(predicted, obs[::-1])["cosine"]

    if second_session_rtf is not None and Path(second_session_rtf).exists():
        sig_ho = observed_signal_summary(second_session_rtf)
        obs_ho = sig_ho["primary"]
        feasibility["second_session_cosine"] = shape_metrics(predicted, obs_ho)["cosine"]
        feasibility["second_session_rtf"] = str(second_session_rtf)
        feasibility["second_session_negative_channel_frame_fraction"] = round(
            sig_ho["negative_channel_frame_fraction"], 6)
        feasibility["second_session_negative_mean_channel_count"] = sig_ho["negative_mean_channel_count"]
        feasibility["second_session_signal_transform_sensitivity"] = dict(
            measured_value=shape_metrics(predicted, sig_ho["measured_sensitivity"]),
            net_over_reference=shape_metrics(predicted, sig_ho["net_over_ref_sensitivity"]),
        )
        feasibility["second_session_certified_tail_sensitivity"] = {
            label: shape_metrics(pred_alt, obs_ho) for label, pred_alt in tail_predictions.items()
        }
        feasibility["second_session_angle_mapping_sensitivity_arctan"] = shape_metrics(arctan_pred, obs_ho)
        feasibility["second_session_reversed_channel_order_cosine"] = shape_metrics(
            predicted, obs_ho[::-1])["cosine"]

    boundary = resolution_diagnostics(A)
    mono = monodisperse_responses()
    boundary["monodisperse_adjacent_pair_cosine"] = [round(c, 4) for c in mono["adjacent_pair_cosine"]]
    boundary["monodisperse_diams_um"] = mono["diams_um"].tolist()

    assumptions = dict(
        wavelength_um=mc.LAMBDA0_UM,
        wavelength_note="He-Ne vacuum wavelength",
        particle_refractive_index=mc.GLASS_RI,
        particle_ri_note="soda-lime glass, SRM 1021 certificate (n = 1.52 + 0i, non-absorbing)",
        medium_refractive_index=mc.N_MED,
        medium_ri_note="pH-7 Britton-Robinson buffer, unmeasured; nominal water value carried as a "
                       "bounded-sensitivity input, not a fact",
        detector_focal_mm=mc.R3_FOCAL_MM,
        detector_channels=mc.N_CHANNELS,
        class_upper_edges_um=EDGES.tolist(),
        detector_geometry=(
            "31 nonuniform R3 annuli reconstructed from the manufacturer's 0.9--175 um class "
            "upper limits and r = 1.22 lambda_0 f / x, with a central r=0 boundary"
        ),
        radius_to_angle_mapping="theta_medium = asin[r / (n_medium f)]",
        theta_min_deg=round(float(detector_rings().theta_lo[0]), 6),
        first_nonzero_boundary_deg=round(float(detector_rings().theta_hi[0]), 6),
        theta_max_deg=round(float(detector_rings().theta_hi[-1]), 6),
        geometry_fit_to_nist=False,
        geometry_status="nominal optical-equivalent annuli inferred from the manual class limits; "
                        "not mechanical detector metrology",
        omitted_detector_effects=["semiring gaps and center elements", "aberration", "channel gain",
                                  "blur", "polarization-specific response"],
        azimuthal_coverage="full-annulus equivalent; a uniform semiring collection factor cancels "
                           "when each channel profile is normalized to unit sum",
        channel_orientation="current RTF channel order is used; direct versus reversed comparison is "
                            "reported as an empirical orientation check, not manufacturer proof",
        certified_distribution_tail_convention=TAIL_CONVENTION,
        certified_distribution_tail_anchors_um=[round(v, 6) for v in certified_tail_anchors()],
        scattering_model="Mie C_sca(d) = pi (d/2)^2 Q_sca(d), annular integral of the unpolarized "
                         "phase function over each ring's [theta_lo, theta_hi]",
        normalization="both predicted and measured profiles normalized to unit sum; agreement scored "
                      "by cosine / correlation / total variation (shape only, amplitude-invariant)",
    )

    provenance = dict(
        primary_evaluation_rtf=str(nist_rtf),
        primary_evaluation_rtf_sha256=_sha256(nist_rtf),
        second_session_rtf_sha256=(
            _sha256(second_session_rtf)
            if second_session_rtf is not None and Path(second_session_rtf).exists() else None
        ),
        certificate="NIST SRM 1021, 1021.pdf Table 1 (cumulative volume fraction finer); certified "
                    "x50,3 = 5.8 um; supported span ~2.1-12.9 um",
        certified_x50_um=CERT_X50_UM,
        r3_grid_um=GRID.tolist(),
        forward_operator="diffractomorph_pipeline.optics.mie_candidate manual-derived R3 annuli; "
                         "production operator optics/mie.py NOT modified",
        geometry_source="Sympatec HELOS/R Operating Instructions, 24 November 2009, item "
                        "BM00010E.W: R3 measuring-range table and Fraunhofer relation",
        geometry_source_sha256=MANUAL_SHA256,
        certificate_sha256=CERTIFICATE_SHA256,
    )

    return Objective1Result(
        measured=obs, predicted=predicted, n_cert=n_cert,
        feasibility=feasibility, boundary=boundary, monodisperse=mono,
        assumptions=assumptions, provenance=provenance,
        meta=dict(geometry_fitted=False),
    )


# ── Renderer ─────────────────────────────────────────────────────────────────────────────────────

def _apply_career_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from diffractomorph_pipeline import plot_styles as ps

    ps.apply_manuscript_style()          # Arial, white bg, embedded TrueType (pdf.fonttype 42)
    plt.rcParams.update({                # CAREER sizing: ~8-9 pt at 7.0 in wide, restrained
        "font.size": 8, "font.weight": "normal", "axes.labelweight": "normal",
        "axes.labelsize": 8.5, "axes.titlesize": 9, "axes.titleweight": "normal",
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.grid": False,
    })
    return plt


def render_career_figure(result: Objective1Result, out_dir: Path | str,
                         basename: str = FIGURE_BASENAME) -> dict:
    """Render the two-panel CAREER figure (vector PDF + preview PNG) and the source-data CSVs."""
    import csv

    plt = _apply_career_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ch = np.arange(1, mc.N_CHANNELS + 1)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.05))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.155, top=0.9, wspace=0.42)

    # ── Panel A — physical feasibility ──────────────────────────────────────────────────────────
    # Measured: solid dark line + filled circles. Predicted: dashed mid-blue + open squares. The
    # dash/solid + filled/open contrast keeps them distinct in grayscale as well as colour.
    axA.plot(ch, result.measured, "-o", color="#222222", lw=1.3, ms=3.4,
             label="measured (SRM 1021)")
    axA.plot(ch, result.predicted, "--s", color="#2166ac", lw=1.3, ms=3.4,
             markerfacecolor="none", markeredgecolor="#2166ac", markeredgewidth=0.9,
             label="predicted (certified PSD → operator)")
    f = result.feasibility
    metric_txt = f"cosine = {f['cosine']:.4f}"
    if "second_session_cosine" in f:
        # "second NIST session" (same SRM-1021 standard, re-measured; angular scale not refitted) —
        # avoids implying validation against a different particle-size distribution.
        metric_txt += f"\nsecond NIST session: {f['second_session_cosine']:.4f}"
    # The measured/predicted profile rises left→right, so the bottom-right corner is empty — put the
    # metric box there, clear of the upper-left legend.
    axA.text(0.965, 0.045, metric_txt, transform=axA.transAxes, ha="right", va="bottom", fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
    axA.set_xlabel("detector channel")
    axA.set_ylabel("normalized response")
    axA.set_xlim(0.5, 31.5)
    axA.set_ylim(bottom=0)
    axA.legend(loc="upper left", handlelength=1.9, borderaxespad=0.3)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)

    # ── Panel B — resolution boundary ───────────────────────────────────────────────────────────
    mono = result.monodisperse
    # Restrained sequential grays→one accent; distinct linestyle + marker per diameter for grayscale.
    styles = [("#000000", "-", "o"), ("#333333", "--", "s"), ("#666666", "-.", "^"),
              ("#999999", ":", "D"), ("#c1440e", (0, (3, 1, 1, 1)), "v")]
    for i, d in enumerate(mono["diams_um"]):
        color, ls, mk = styles[i % len(styles)]
        axB.plot(ch, mono["profiles"][:, i], color=color, ls=ls, marker=mk, lw=1.2, ms=3.0,
                 markevery=2, label=f"{d:g} µm")
    axB.set_xlabel("detector channel")
    axB.set_ylabel("normalized response")
    axB.set_xlim(0.5, 31.5)
    axB.set_ylim(bottom=0)
    axB.legend(title="simulated monodisperse response", loc="upper left", bbox_to_anchor=(0.1, 1.0),
               handlelength=2.2, borderaxespad=0.3, labelspacing=0.3)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    # Panel labels to the LEFT of each y-axis.
    for ax, letter in ((axA, "A"), (axB, "B")):
        ax.text(-0.2, 1.02, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="bottom", ha="right")

    pdf_path = out_dir / f"{basename}.pdf"
    png_path = out_dir / f"{basename}.png"
    fig.savefig(pdf_path)                       # vector; text/lines not rasterized
    fig.savefig(png_path, dpi=600)              # high-res preview
    plt.close(fig)

    # Source-data CSVs.
    prof_csv = out_dir / "objective1_measured_vs_predicted.csv"
    with open(prof_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "measured_norm", "predicted_norm"])
        for c, mval, pval in zip(ch, result.measured, result.predicted):
            w.writerow([int(c), f"{mval:.6e}", f"{pval:.6e}"])
    mono_csv = out_dir / "objective1_monodisperse_responses.csv"
    with open(mono_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel"] + [f"d_{d:g}um_norm" for d in mono["diams_um"]])
        for j, c in enumerate(ch):
            w.writerow([int(c)] + [f"{mono['profiles'][j, i]:.6e}" for i in range(len(mono["diams_um"]))])

    geom_csv = out_dir / "objective1_manual_r3_geometry.csv"
    radii = mc.r3_manual_radii(EDGES)
    rings = detector_rings()
    # Manual size boundaries are largest-to-smallest when mapped from inner-to-outer radius.
    manual_boundaries = np.asarray(EDGES, float)[::-1]
    with open(geom_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "manual_size_boundary_um", "radius_lo_mm", "radius_hi_mm",
                    "theta_lo_medium_deg", "theta_hi_medium_deg"])
        for j in range(mc.N_CHANNELS):
            w.writerow([j + 1, f"{manual_boundaries[j]:.8g}", f"{radii[j]:.8g}",
                        f"{radii[j + 1]:.8g}", f"{rings.theta_lo[j]:.8g}",
                        f"{rings.theta_hi[j]:.8g}"])

    return dict(pdf=pdf_path, png=png_path, measured_vs_predicted_csv=prof_csv,
                monodisperse_csv=mono_csv, manual_geometry_csv=geom_csv)


def write_metrics_json(result: Objective1Result, out_dir: Path | str,
                       name: str = "objective1_metrics.json") -> Path:
    """Write all metrics + geometry/RI assumptions + rank/overlap diagnostics + provenance to JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(
        analysis="objective1_operator_feasibility",
        question_A="physical feasibility: certified PSD -> aggregate detector response",
        question_B="resolution boundary: overlapping monodisperse signatures / limited size rank",
        feasibility=result.feasibility,
        resolution_boundary=result.boundary,
        assumptions=result.assumptions,
        provenance=result.provenance,
        nonclaims=[
            "the production optical operator is NOT claimed to be fully calibrated",
            "a single NIST standard does NOT validate every diameter-response column of A",
            "detector channels do NOT uniquely identify particle diameter",
            "quantitative size-resolved flux is NOT established",
            "the evaluated operator is NOT recommended to replace the production operator without "
            "further validation",
        ],
        conclusion="The physical architecture of the observation operator is feasible, but "
                   "monodisperse standards, mixtures, and controlled synthetic histories are still "
                   "required to establish the resolution and identifiability needed for dynamic "
                   "inference.",
    )
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=2))
    return path


# ── CLI ──────────────────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Fixed manual-geometry optical-operator evaluation + resolution boundary "
                    "(NIST SRM 1021 forward-operator).")
    p.add_argument("--study-root", type=Path,
                   help="pH-study root containing QC/NIST_weekly_QCs; used to resolve both RTFs.")
    p.add_argument("--nist-rtf", type=Path,
                   help="NIST SRM 1021 session .rtf used to evaluate the fixed operator.")
    p.add_argument("--second-session-rtf", "--held-out-rtf", dest="second_session_rtf", type=Path,
                   help="Second same-standard session .rtf for the cross-session check "
                        "(pass 'none' to skip).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory for the PDF/PNG/JSON/CSV outputs (e.g. the manuscript "
                        "figures folder). Required — deliverables are never written to a "
                        "default location.")
    p.add_argument("--no-figure", action="store_true", help="Compute + write JSON/CSV only.")
    args = p.parse_args(argv)

    if args.study_root is not None:
        weekly = args.study_root / "QC" / "NIST_weekly_QCs"
        default_nist = weekly / DEFAULT_NIST_RTF.name
        default_held = weekly / DEFAULT_HELDOUT_RTF.name
    else:
        default_nist, default_held = DEFAULT_NIST_RTF, DEFAULT_HELDOUT_RTF
    nist = args.nist_rtf or default_nist
    second = args.second_session_rtf or default_held
    second = None if str(second).lower() == "none" else second
    if not Path(nist).exists():
        p.error(f"NIST evaluation RTF not found: {nist}")

    print(f"[objective1] evaluating fixed manual-derived R3 geometry on {nist}")
    result = compute_objective1(nist, second)
    f = result.feasibility
    print(f"[objective1] angle span = {result.assumptions['theta_min_deg']:.2f}--"
          f"{result.assumptions['theta_max_deg']:.2f} deg   cosine = {f['cosine']:.4f}"
          + (f"   second session = {f['second_session_cosine']:.4f}"
             if "second_session_cosine" in f else ""))
    b = result.boundary
    print(f"[objective1] 2-15 um resolution: eff-rank {b['eff_rank_abs']} abs / "
          f"{b['eff_rank_colnorm']} col-norm; adjacent-column cosine median "
          f"{b['adjacent_col_cosine_median']:.3f}")

    json_path = write_metrics_json(result, args.output_dir)
    print(f"[objective1] wrote {json_path}")
    if not args.no_figure:
        outs = render_career_figure(result, args.output_dir)
        for k, v in outs.items():
            print(f"[objective1] wrote {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
