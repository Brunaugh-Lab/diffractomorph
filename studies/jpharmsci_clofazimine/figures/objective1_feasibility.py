"""Objective-1 grant figure: optical-operator *feasibility* and its *calibration boundary*.

Grant-facing (NSF CAREER) analysis + two-panel figure answering two questions about the
DiffractoMorph HELOS R3 observation operator, using the **candidate physical operator**
(:mod:`diffractomorph_pipeline.optics.mie_candidate`) — the production operator is not touched:

1. **Physical feasibility (Panel A).** Can known scattering physics propagate an *independently
   certified* particle population (NIST SRM 1021) into an aggregate 31-channel detector response
   consistent with measurement? Propagate the certified SRM-1021 Q3 distribution (Table 1) through
   the C_sca-inclusive Mie + annular-detector operator and compare the predicted normalized channel
   profile to the measured one.

2. **Calibration boundary (Panel B).** Why does agreement for one broad standard NOT establish
   unique diameter→channel localization? Show that monodisperse diameters produce *distributed,
   overlapping* channel signatures, and quantify the operator's limited independent size information
   (effective rank, adjacent-column overlap).

This is a **feasibility and identifiability** analysis. It does NOT claim the production operator is
calibrated, that a single standard validates every diameter-response column, that detector channels
uniquely identify diameter, or that quantitative size-resolved flux is established. The one geometry
degree of freedom (the angular scale ``theta_max``) is *fitted* to the measurement and is reported
as such — a **calibrated** angular scale, not a **measured** one.

The module is split into a pure compute core (operator build via ``mie_candidate``, the certified
number PSD, shape metrics, resolution diagnostics — all array-in/array-out and unit-testable with no
vault files), a thin data-loading layer (:func:`observed_net_shape` reads a PAQXOS ``.rtf`` via the
ingest layer), a renderer, and a ``main`` CLI that takes explicit input/output paths and never
depends on the working directory.

Reproduce / export::

    python -m diffractomorph_pipeline.figures.objective1_feasibility \
        --nist-rtf <clean_session.rtf> --held-out-rtf <second_session.rtf> --output-dir <dir>

``--output-dir`` is required; the input defaults resolve from the configured data corpus
(``DFM_DATA_ROOT`` / ``.dfm.toml``). Outputs (into ``--output-dir``):
``optical_operator_feasibility_and_calibration_boundary.pdf`` (canonical vector) + ``.png`` (preview),
``objective1_metrics.json`` (metrics + geometry/RI assumptions + rank/overlap diagnostics +
provenance), and two source-data CSVs.
"""
from __future__ import annotations

import argparse
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

# One geometry degree of freedom (the angular scale) is FITTED to the measured channel profile.
# theta_min is a fixed sensitivity endpoint; theta_max is profiled over this grid (spans below the
# ~10–15° optimum so the fit is interior, not railed). Same grid as the audit → same 9.9° optimum.
THETA_MIN_DEG = 0.19
THETA_MAX_GRID = np.geomspace(6.0, 55.0, 41)
PAQXOS_FIT_TMAX_DEG = 39.185      # old circular PAQXOS-fitted angular scale (reference point only)

# Panel-B representative monodisperse diameters (µm). 1 µm is below the certified span (deliberately —
# it shows the sub-2 µm signature the standard cannot constrain).
MONODISPERSE_UM = np.array([1.0, 2.0, 5.0, 10.0, 15.0])

# Resolution diagnostics are computed on the certified decision range only.
RESOLUTION_LO_UM = 2.0
RESOLUTION_HI_UM = 15.0

# Default measurement provenance (overridable on the CLI), resolved from the configured data corpus
# (DFM_DATA_ROOT / .dfm.toml) — never an absolute path. The clean session is used to calibrate the
# angular scale; the second same-standard session is a held-out transfer check. When the corpus is
# unconfigured these are non-existent sentinels, so `.exists()` guards skip rather than error.
_WEEKLY = corpus("disso_experiments", "ph_dependent_dissolution_study", "QC", "NIST_weekly_QCs")
DEFAULT_NIST_RTF = _WEEKLY / "NIST QC Week of 20260608.rtf"          # clean net (calibration)
DEFAULT_HELDOUT_RTF = _WEEKLY / "NIST QC Week of 20260601.rtf"       # 2nd session (held-out transfer)
FIGURE_BASENAME = "optical_operator_feasibility_and_calibration_boundary"


# ── Certified distribution → number PSD (certificate only, no PAQXOS) ────────────────────────────

def certified_number_psd(grid: np.ndarray = GRID, edges: np.ndarray = EDGES) -> np.ndarray:
    """Nominal certified SRM-1021 **number** PSD on the R3 grid (n ∝ q3 / d³).

    Monotone (PCHIP) interpolation of the certified cumulative *volume* in log-diameter, anchored at
    the certified span endpoints (2.1 µm → 0 %, 12.9 µm → 100 %), differenced onto the class edges to
    per-bin volume, then converted to number and normalized. No submicron PAQXOS mass is added. This
    reproduces the ``nominal`` reconstruction of the forward-operator audit exactly.
    """
    from scipy.interpolate import PchipInterpolator

    dk = np.r_[2.1, CERT_D, 12.9]
    ck = np.r_[0.0, CERT_PCT, 100.0]
    o = np.argsort(dk)
    dk, ck = dk[o], ck[o]
    dk, idx = np.unique(dk, return_index=True)
    ck = np.maximum.accumulate(ck[idx])
    f = PchipInterpolator(np.log(dk), ck, extrapolate=False)
    Qe = f(np.log(edges))
    Qe = np.where(edges < dk.min(), 0.0, Qe)
    Qe = np.where(edges > dk.max(), 100.0, Qe)
    Qe = np.maximum.accumulate(np.clip(np.nan_to_num(Qe, nan=0.0), 0.0, 100.0))
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


# ── Physical operator (the candidate C_sca + annular operator; NOT reimplemented here) ───────────

def build_physical_operator(theta_max_deg: float, theta_min_deg: float = THETA_MIN_DEG,
                            diams: np.ndarray = GRID, n_quad: int = 64) -> np.ndarray:
    """Diameter→channel response matrix ``A`` (31 × D) from the candidate physical operator.

    Delegates to :func:`mie_candidate.response_matrix` over log-spaced detector rings between the two
    angular endpoints — the C_sca-inclusive, annular-integrated Mie operator. Optics are fixed inside
    ``mie_candidate`` (He–Ne λ = 0.6328 µm, soda-lime glass n = 1.52, medium n = 1.331 nominal).
    """
    rings = mc.log_rings(theta_min_deg, theta_max_deg, mc.N_CHANNELS)
    return mc.response_matrix(diams, rings, mc.GLASS_RI, n_quad=n_quad)


def _current_operator(theta_max_deg: float, theta_min_deg: float = THETA_MIN_DEG,
                      diams: np.ndarray = GRID) -> np.ndarray:
    """The production *shortcut* operator (centre-angle × θ², **no** C_sca) — for the ablation only.

    This is not the production kernel object; it re-expresses the two shortcuts the production build
    takes (no scattering cross-section; single centre angle × θ² instead of an annular integral) so
    the C_sca contribution can be isolated. Used only to show C_sca is the load-bearing term.
    """
    import miepython as mp

    edges = np.logspace(np.log10(theta_min_deg), np.log10(theta_max_deg), mc.N_CHANNELS + 1)
    th = np.sqrt(edges[:-1] * edges[1:])
    m = complex(mc.GLASS_RI, 0.0) / mc.N_MED
    A = np.empty((mc.N_CHANNELS, len(diams)))
    for i, d in enumerate(diams):
        x = np.pi * d * mc.N_MED / mc.LAMBDA0_UM
        A[:, i] = mp.i_unpolarized(m, x, np.cos(np.deg2rad(th)), norm="one")
    return A * (th ** 2)[:, None]


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


def fit_theta_max(obs: np.ndarray, n_cert: np.ndarray, mode: str = "physical",
                  theta_min_deg: float = THETA_MIN_DEG, grid: np.ndarray = THETA_MAX_GRID) -> dict:
    """Fit the angular scale ``theta_max`` by maximizing cosine(predicted, observed) — blur = 0.

    ``mode='physical'`` uses the candidate C_sca + annular operator; ``mode='current'`` uses the
    no-C_sca production shortcut. Returns the best ``theta_max`` and its shape metrics.
    """
    build = build_physical_operator if mode == "physical" else _current_operator
    best = None
    for t in grid:
        A = build(float(t), theta_min_deg)
        met = shape_metrics(predict_channel_shape(A, n_cert), obs)
        if best is None or met["cosine"] > best["cosine"]:
            best = dict(theta_max_deg=float(t), **met)
    return best


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


def monodisperse_responses(theta_max_deg: float, diams: np.ndarray = MONODISPERSE_UM,
                           theta_min_deg: float = THETA_MIN_DEG, n_quad: int = 128) -> dict:
    """Per-diameter normalized channel responses + centroid, FWHM (channels), and adjacent overlap.

    Uses the same candidate operator / geometry as the feasibility panel. Each monodisperse diameter
    yields a distributed response (never one channel); neighbouring diameters overlap strongly.
    """
    rings = mc.log_rings(theta_min_deg, theta_max_deg, mc.N_CHANNELS)
    A = mc.response_matrix(diams, rings, mc.GLASS_RI, n_quad=n_quad)
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
    measured: np.ndarray                 # observed normalized channel shape (calibration session)
    predicted: np.ndarray                # predicted normalized channel shape (certified PSD)
    n_cert: np.ndarray                   # certified number PSD on the R3 grid
    theta_max_deg: float
    feasibility: dict                    # cosine/corr/tv (fit) + held-out transfer + C_sca ablation
    boundary: dict                       # resolution diagnostics
    monodisperse: dict                   # per-diameter responses
    assumptions: dict                    # geometry + refractive-index + optics
    provenance: dict                     # measurement files, certificate, grid
    meta: dict = field(default_factory=dict)


def compute_objective1(nist_rtf: Path | str = DEFAULT_NIST_RTF,
                       held_out_rtf: Path | str | None = DEFAULT_HELDOUT_RTF,
                       theta_min_deg: float = THETA_MIN_DEG) -> Objective1Result:
    """Full Objective-1 computation from measurement files + the certified certificate.

    Calibrates the angular scale on ``nist_rtf`` (the clean session), predicts the certified PSD,
    scores feasibility, runs the C_sca ablation, transfers the frozen operator to ``held_out_rtf``
    (same standard, different session), and computes the calibration-boundary diagnostics.
    """
    n_cert = certified_number_psd()
    obs = observed_net_shape(nist_rtf)

    fit = fit_theta_max(obs, n_cert, mode="physical", theta_min_deg=theta_min_deg)
    theta_max = fit["theta_max_deg"]
    A = build_physical_operator(theta_max, theta_min_deg)
    predicted = predict_channel_shape(A, n_cert)

    # C_sca ablation: best no-C_sca fit vs the physical fit, and the physical fit evaluated at the
    # old circular PAQXOS angular scale (39.2°) — both show C_sca is load-bearing and 39.2° is wrong.
    cur = fit_theta_max(obs, n_cert, mode="current", theta_min_deg=theta_min_deg)
    at_paqxos = shape_metrics(
        predict_channel_shape(build_physical_operator(PAQXOS_FIT_TMAX_DEG, theta_min_deg), n_cert), obs)

    feasibility = dict(
        theta_max_deg=round(theta_max, 2),
        cosine=fit["cosine"], corr=fit["corr"], tv=fit["tv"],
        no_csca_best_cosine=cur["cosine"], no_csca_best_theta_max_deg=round(cur["theta_max_deg"], 2),
        physical_at_paqxos_theta_max_cosine=at_paqxos["cosine"],
        csca_cosine_gain=round(fit["cosine"] - cur["cosine"], 4),
    )

    if held_out_rtf is not None and Path(held_out_rtf).exists():
        obs_ho = observed_net_shape(held_out_rtf)
        feasibility["held_out_transfer_cosine"] = shape_metrics(predicted, obs_ho)["cosine"]
        feasibility["held_out_rtf"] = str(held_out_rtf)

    boundary = resolution_diagnostics(A)
    mono = monodisperse_responses(theta_max, theta_min_deg=theta_min_deg)
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
        theta_min_deg=theta_min_deg,
        theta_min_note="fixed inner-angle sensitivity endpoint (not a measured ring geometry)",
        theta_max_deg=round(theta_max, 3),
        theta_max_note="FITTED angular scale (calibrated to this measurement) — a calibrated, not a "
                       "measured, geometry; not the old PAQXOS-fitted 39.2 deg",
        detector_geometry="log-spaced angular rings (assumed Fraunhofer boundaries; real HELOS R3 "
                          "ring radii not yet available)",
        scattering_model="Mie C_sca(d) = pi (d/2)^2 Q_sca(d), annular integral of the unpolarized "
                         "phase function over each ring's [theta_lo, theta_hi]",
        normalization="both predicted and measured profiles normalized to unit sum; agreement scored "
                      "by cosine / correlation / total variation (shape only, amplitude-invariant)",
    )

    provenance = dict(
        calibration_rtf=str(nist_rtf),
        certificate="NIST SRM 1021, 1021.pdf Table 1 (cumulative volume fraction finer); certified "
                    "x50,3 = 5.8 um; supported span ~2.1-12.9 um",
        certified_x50_um=CERT_X50_UM,
        r3_grid_um=GRID.tolist(),
        candidate_operator="diffractomorph_pipeline.optics.mie_candidate (isolated; production "
                           "operator optics/mie.py NOT modified)",
    )

    return Objective1Result(
        measured=obs, predicted=predicted, n_cert=n_cert, theta_max_deg=theta_max,
        feasibility=feasibility, boundary=boundary, monodisperse=mono,
        assumptions=assumptions, provenance=provenance,
        meta=dict(theta_max_grid=[round(float(t), 3) for t in THETA_MAX_GRID]),
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
    if "held_out_transfer_cosine" in f:
        # "second NIST session" (same SRM-1021 standard, re-measured; angular scale not refitted) —
        # avoids implying validation against a different particle-size distribution.
        metric_txt += f"\nsecond NIST session: {f['held_out_transfer_cosine']:.4f}"
    # The measured/predicted profile rises left→right, so the bottom-right corner is empty — put the
    # metric box there, clear of the upper-left legend.
    axA.text(0.965, 0.045, metric_txt, transform=axA.transAxes, ha="right", va="bottom", fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
    axA.set_xlabel("detector channel (low → high angle)")
    axA.set_ylabel("normalized response")
    axA.set_xlim(0.5, 31.5)
    axA.set_ylim(bottom=0)
    axA.legend(loc="upper left", handlelength=1.9, borderaxespad=0.3)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)

    # ── Panel B — calibration boundary ──────────────────────────────────────────────────────────
    mono = result.monodisperse
    # Restrained sequential grays→one accent; distinct linestyle + marker per diameter for grayscale.
    styles = [("#000000", "-", "o"), ("#333333", "--", "s"), ("#666666", "-.", "^"),
              ("#999999", ":", "D"), ("#c1440e", (0, (3, 1, 1, 1)), "v")]
    for i, d in enumerate(mono["diams_um"]):
        color, ls, mk = styles[i % len(styles)]
        axB.plot(ch, mono["profiles"][:, i], color=color, ls=ls, marker=mk, lw=1.2, ms=3.0,
                 markevery=2, label=f"{d:g} µm")
    axB.set_xlabel("detector channel (low → high angle)")
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

    return dict(pdf=pdf_path, png=png_path, measured_vs_predicted_csv=prof_csv,
                monodisperse_csv=mono_csv)


def write_metrics_json(result: Objective1Result, out_dir: Path | str,
                       name: str = "objective1_metrics.json") -> Path:
    """Write all metrics + geometry/RI assumptions + rank/overlap diagnostics + provenance to JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(
        analysis="objective1_operator_feasibility",
        question_A="physical feasibility: certified PSD -> aggregate detector response",
        question_B="calibration boundary: overlapping monodisperse signatures / limited size rank",
        feasibility=result.feasibility,
        calibration_boundary=result.boundary,
        assumptions=result.assumptions,
        provenance=result.provenance,
        nonclaims=[
            "the production optical operator is NOT claimed to be fully calibrated",
            "a single NIST standard does NOT validate every diameter-response column of A",
            "detector channels do NOT uniquely identify particle diameter",
            "quantitative size-resolved flux is NOT established",
            "the candidate operator is NOT recommended to replace the production operator without "
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
        description="Objective-1 grant figure: optical-operator feasibility + calibration boundary "
                    "(NIST SRM 1021 forward-operator).")
    p.add_argument("--nist-rtf", type=Path, default=DEFAULT_NIST_RTF,
                   help="Clean NIST session .rtf used to calibrate the angular scale.")
    p.add_argument("--held-out-rtf", type=Path, default=DEFAULT_HELDOUT_RTF,
                   help="Second same-standard session .rtf for the held-out transfer check "
                        "(pass 'none' to skip).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory for the PDF/PNG/JSON/CSV outputs (e.g. the manuscript "
                        "figures folder). Required — deliverables are never written to a "
                        "default location.")
    p.add_argument("--theta-min", type=float, default=THETA_MIN_DEG,
                   help="Fixed inner angular endpoint (deg).")
    p.add_argument("--no-figure", action="store_true", help="Compute + write JSON/CSV only.")
    args = p.parse_args(argv)

    held = None if (args.held_out_rtf is None or str(args.held_out_rtf).lower() == "none") \
        else args.held_out_rtf
    if not Path(args.nist_rtf).exists():
        p.error(f"calibration RTF not found: {args.nist_rtf}")

    print(f"[objective1] calibrating angular scale on {args.nist_rtf}")
    result = compute_objective1(args.nist_rtf, held, theta_min_deg=args.theta_min)
    f = result.feasibility
    print(f"[objective1] theta_max = {result.theta_max_deg:.2f} deg   cosine(fit) = {f['cosine']:.4f}"
          + (f"   held-out = {f['held_out_transfer_cosine']:.4f}" if "held_out_transfer_cosine" in f else ""))
    print(f"[objective1] C_sca gain: no-C_sca best {f['no_csca_best_cosine']:.4f} -> "
          f"physical {f['cosine']:.4f} (+{f['csca_cosine_gain']:.4f}); at PAQXOS 39.2 deg "
          f"{f['physical_at_paqxos_theta_max_cosine']:.4f}")
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
