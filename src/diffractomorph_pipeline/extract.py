"""Module 3 — Dissolution-kinetics extraction (KWW / stretched-exponential).

Fits the **magnitude** signal of a run that triage (Step 1) has already routed:
the cross-section-weighted signal mass ``a(t) = Σ_c I(c, t)`` over the dissolving
population. Triage decides *which* channels are legal to lump:

- **single-mode** → the whole admitted (active) population is one mode; fit the
  aggregate over ``triage.active_channels``.
- **multi-band** → fit only the **dissolution** band (the monotonic-decay,
  small-particle/high-channel band); the growth band is a separate process and is
  *not* fit here.

The form is Kohlrausch–Williams–Watts (KWW) / Weibull, the natural ensemble decay
of a polydisperse population under single-particle Noyes–Whitney dissolution:

    a(t) = a_inf + Δa · exp[ −(k t)^β ]

with first-order (β ≡ 1) as the null model. β is reported as a **descriptor** of
timescale dispersion, not a mechanism claim — and the fitted rate ``k`` is
delivered-dose-confounded (see the manuscript framing). Mapping β → σ_g / regime
and the mechanistic forward model are deliberately out of scope here.

Methodology reference: ``Stretched_Exponential_Fitting_for_Dissolution_Kinetics``
(decision tree §2.4, fitting recipe §3, β reading §4, Copt-divergence §6).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

OBSTRUCTION_COPT = 40.0      # frames above this Copt are beam-obstructed (mirror forward/model)
GAP_THRESHOLD_MIN = 2.0      # re-zero acquisition gaps larger than this (mirror triage)
_K_STARTS = (0.05, 0.1, 0.5, 1.0, 5.0)   # multi-start over k (KWW likelihood is multi-modal)


# ── KWW core (methodology §1, §3) ─────────────────────────────────────────────

def kww(t, a_inf, da, k, beta):
    """Stretched-exponential approach-to-plateau ``a_inf + da·exp[−(k t)^β]``."""
    return a_inf + da * np.exp(-(k * np.clip(t, 1e-9, None)) ** beta)


def _r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _durbin_watson(resid):
    """Durbin–Watson on fit residuals: ≈2 → flat/independent; ≪2 → systematic
    curvature (a monodisperse-fit-to-polydisperse-data signature, §2.4)."""
    d = np.diff(resid)
    denom = float(np.sum(resid ** 2))
    return float(np.sum(d ** 2) / denom) if denom > 0 else float("nan")


def _fit_first_order(t, y):
    """First-order (KWW with β ≡ 1) — the null model. Returns (params4, r2, resid)."""
    a_inf0 = float(y[-max(1, len(y) // 5):].mean())
    a0 = float(y[0])
    best = None
    for k0 in _K_STARTS:
        try:
            p, _ = curve_fit(
                lambda tt, a_inf, da, k: kww(tt, a_inf, da, k, 1.0),
                t, y, p0=[a_inf0, a0 - a_inf0, k0],
                bounds=([-1e9, -1e9, 1e-6], [1e9, 1e9, 1e6]), maxfev=50000,
            )
            yhat = kww(t, p[0], p[1], p[2], 1.0)
            r2 = _r2(y, yhat)
            if best is None or r2 > best[1]:
                best = (np.array([p[0], p[1], p[2], 1.0]), r2, y - yhat)
        except Exception:
            continue
    return best


def _fit_kww(t, y, beta0=0.7):
    """Multi-start KWW fit. Returns (params4, ses4, r2, resid) or ``None``."""
    a_inf0 = float(y[-max(1, len(y) // 5):].mean())
    a0 = float(y[0])
    best = None
    for k0 in _K_STARTS:
        try:
            p, cov = curve_fit(
                kww, t, y, p0=[a_inf0, a0 - a_inf0, k0, beta0],
                bounds=([-1e9, -1e9, 1e-6, 0.05], [1e9, 1e9, 1e6, 3.0]),
                maxfev=50000,
            )
            yhat = kww(t, *p)
            r2 = _r2(y, yhat)
            if best is None or r2 > best[1]:
                ses = np.sqrt(np.clip(np.diag(cov), 0, None))
                best = (p, ses, r2, y - yhat)
        except Exception:
            continue
    return best


# ── Signal construction — what triage hands us ───────────────────────────────

def _col_index(channels, wanted):
    """Column indices into ``I`` for the requested channel numbers (order preserved)."""
    pos = {c: i for i, c in enumerate(channels)}
    return [pos[c] for c in wanted if c in pos]


def _dissolution_channels(triage):
    """Channel numbers feeding the kinetic fit + the fit target label.

    Single-mode → all active channels (one population). Multi-band → the
    dissolution-role band(s); if none is labelled dissolution, fall back to the
    smallest-particle (highest-channel) band and flag it.
    """
    if triage.verdict != "multi_band" or not triage.bands:
        return list(triage.active_channels), "aggregate", None, []
    diss = [b for b in triage.bands if b.role == "dissolution"]
    flags = []
    if not diss:
        diss = [max(triage.bands, key=lambda b: np.mean(b.channel_range))]
        flags.append("no_dissolution_role")     # used the smallest-particle band as a proxy
    lo = min(b.channel_range[0] for b in diss)
    hi = max(b.channel_range[1] for b in diss)
    band_id = "+".join(b.id for b in diss)
    chans = [c for c in triage.active_channels if lo <= c <= hi]
    return chans, "dissolution_band", band_id, flags


def dissolution_signal(run, triage, window=None):
    """Build the kinetic fit target from a run + its triage verdict.

    Returns ``(t, a, copt, channels, target, band_id, flags)`` where ``a(t)`` is
    the summed signal mass over the dissolving channels on the clean frames
    (beam-obstruction frames dropped, the largest acquisition gap re-zeroed, and
    an optional ``window=(t_min, t_max)`` in minutes applied).
    """
    I = np.asarray(run.I, dtype=float)
    t = np.asarray(run.t_min, dtype=float)
    copt = np.asarray(run.copt, dtype=float)
    chans, target, band_id, flags = _dissolution_channels(triage)
    cols = _col_index(run.channels, chans)

    keep = np.isfinite(copt) & (copt <= OBSTRUCTION_COPT)
    if keep.sum() < 2:                              # obstruction killed the run → keep all
        keep = np.ones(t.shape[0], dtype=bool)
        flags = flags + ["obstruction_not_trimmed"]
    t, I, copt = t[keep], I[keep], copt[keep]

    # Re-zero the largest acquisition gap (drop pre-gap frames), like triage §4.2.
    if t.shape[0] >= 2:
        dt = np.diff(t)
        j = int(np.argmax(dt))
        if dt[j] > GAP_THRESHOLD_MIN:
            t, I, copt = t[j + 1:], I[j + 1:], copt[j + 1:]
            flags = flags + ["gap_rezeroed"]
    t = t - t[0] if t.shape[0] else t

    if window is not None:
        lo, hi = window
        wmask = (t >= lo) & (t <= hi)
        t, I, copt = t[wmask], I[wmask], copt[wmask]

    a = I[:, cols].sum(axis=1) if cols else np.zeros(t.shape[0])
    return t, a, copt, chans, target, band_id, flags


def copt_divergence(t, a, copt):
    """Late-time AUC-vs-Copt divergence (§6): a precipitation/deposition flag.

    ``R(t) = a(t)·Copt(0) / [Copt(t)·a(0)]`` stays ≈1 in single-mechanism
    dissolution; sustained late drift flags a secondary process. Returns the
    max ``|R−1|`` over the final third of the run (NaN if Copt is unusable).
    """
    if t.shape[0] < 6 or a[0] <= 0 or not np.isfinite(copt).all() or copt[0] <= 0:
        return float("nan")
    R = (a * copt[0]) / (copt * a[0])
    late = t >= (t[0] + 2.0 / 3.0 * (t[-1] - t[0]))
    return float(np.max(np.abs(R[late] - 1.0))) if late.any() else float("nan")


# ── Output contract ───────────────────────────────────────────────────────────

@dataclass
class KineticFit:
    """Result of :func:`fit_dissolution_kinetics` — the per-run kinetic descriptor."""
    target: str                  # "aggregate" (single-mode) | "dissolution_band" (multi-band)
    band_id: str | None          # which band(s) were summed (multi-band only)
    channels: list[int]          # channels summed into the signal mass
    verdict: str                 # triage verdict the fit was routed by
    model: str                   # recommended model: "kww" | "first_order" | "none"
    # KWW parameters
    a_inf: float                 # plateau signal mass
    da: float                    # a0 − a_inf (positive for a decay)
    k: float                     # rate (1/min)
    beta: float                  # stretching exponent (descriptor of dispersion)
    k_se: float
    beta_se: float
    r2: float
    half_life: float             # t_1/2 = (ln2)^(1/β)/k  (min)
    signal_drop_frac: float      # 1 − a_inf/a0  (fractional decay of the signal)
    # First-order benchmark
    fo_k: float
    fo_r2: float
    fo_durbin_watson: float      # residual structure of the null model (<~1.5 → curvature)
    # Diagnostics / provenance
    n_frames: int
    window: tuple[float, float]
    copt_divergence: float       # max |R−1| late (precipitation flag); NaN if N/A
    flags: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        """Flat dict for a per-run summary CSV."""
        return {
            "verdict": self.verdict, "target": self.target, "band_id": self.band_id,
            "model": self.model, "k": self.k, "beta": self.beta,
            "k_se": self.k_se, "beta_se": self.beta_se, "r2": self.r2,
            "half_life_min": self.half_life, "signal_drop_frac": self.signal_drop_frac,
            "a_inf": self.a_inf, "a0": self.a_inf + self.da,
            "fo_k": self.fo_k, "fo_r2": self.fo_r2, "fo_dw": self.fo_durbin_watson,
            "n_frames": self.n_frames, "copt_divergence": self.copt_divergence,
            "n_channels": len(self.channels), "flags": ",".join(self.flags) or "-",
        }


def _empty_fit(target, band_id, chans, verdict, n, window, flags):
    nan = float("nan")
    return KineticFit(
        target=target, band_id=band_id, channels=list(chans), verdict=verdict,
        model="none", a_inf=nan, da=nan, k=nan, beta=nan, k_se=nan, beta_se=nan,
        r2=nan, half_life=nan, signal_drop_frac=nan, fo_k=nan, fo_r2=nan,
        fo_durbin_watson=nan, n_frames=int(n), window=window,
        copt_divergence=nan, flags=list(flags),
    )


def fit_dissolution_kinetics(run, triage, window=None, beta0=0.7) -> KineticFit:
    """Fit KWW (+ first-order null) to the triage-routed dissolution signal.

    Parameters
    ----------
    run
        The ingested ``RawRun`` (provides ``I``, ``t_min``, ``copt``, ``channels``).
    triage
        Its ``ChannelTriage`` verdict — selects the dissolving channels (§above).
    window
        Optional ``(t_min, t_max)`` in minutes (from the clean frame 0) to restrict
        the fit to a single mechanism regime; use the ``copt_divergence`` flag to
        decide if late-time windowing is needed.
    """
    t, a, copt, chans, target, band_id, sig_flags = dissolution_signal(run, triage, window)
    win = (float(t[0]), float(t[-1])) if t.shape[0] else (float("nan"), float("nan"))
    flags = list(sig_flags)

    if t.shape[0] < 4 or a[0] <= 0 or np.allclose(a, a[0]):
        flags.append("insufficient_data")
        return _empty_fit(target, band_id, chans, triage.verdict, t.shape[0], win, flags)

    fo = _fit_first_order(t, a)
    kw = _fit_kww(t, a, beta0=beta0)
    if fo is None and kw is None:
        flags.append("fit_failed")
        return _empty_fit(target, band_id, chans, triage.verdict, t.shape[0], win, flags)

    fo_p, fo_r2, fo_resid = fo if fo else (np.full(4, np.nan), float("nan"), None)
    fo_dw = _durbin_watson(fo_resid) if fo_resid is not None else float("nan")

    if kw is None:                       # KWW failed; report the first-order null
        a_inf, da, k, beta = fo_p
        model = "first_order"
        k_se = beta_se = float("nan")
        r2 = fo_r2
        flags.append("kww_fit_failed")
    else:
        p, ses, r2, _ = kw
        a_inf, da, k, beta = p
        k_se, beta_se = float(ses[2]), float(ses[3])
        # Decision tree (§2.4): prefer KWW only if the null shows residual structure
        # AND KWW materially improves the fit; otherwise keep the parsimonious null.
        null_has_structure = np.isfinite(fo_dw) and fo_dw < 1.5
        kww_improves = np.isfinite(fo_r2) and (r2 - fo_r2) >= 0.001
        model = "kww" if (null_has_structure and kww_improves) else "first_order"
        if abs(beta - 1.0) < 0.05:
            flags.append("beta_near_1")          # indistinguishable from first-order
        if beta > 1.0:
            flags.append("compressed_exp")       # β>1: narrow-PSD / single-particle shape (§4)

    a0 = a_inf + da
    half_life = float((np.log(2.0)) ** (1.0 / beta) / k) if (k > 0 and beta > 0) else float("nan")
    drop_frac = float(1.0 - a_inf / a0) if a0 != 0 else float("nan")
    div = copt_divergence(t, a, copt)
    if np.isfinite(div) and div > 0.2:
        flags.append("late_copt_divergence")     # possible precipitation/deposition (§6)
    if target == "aggregate" and np.isfinite(drop_frac) and drop_frac < 0.05:
        flags.append("little_dissolution")       # signal barely decays (e.g. pH-5 slow runs)

    return KineticFit(
        target=target, band_id=band_id, channels=list(chans), verdict=triage.verdict,
        model=model, a_inf=float(a_inf), da=float(da), k=float(k), beta=float(beta),
        k_se=k_se, beta_se=beta_se, r2=float(r2), half_life=half_life,
        signal_drop_frac=drop_frac, fo_k=float(fo_p[2]), fo_r2=float(fo_r2),
        fo_durbin_watson=float(fo_dw), n_frames=int(t.shape[0]), window=win,
        copt_divergence=div, flags=flags,
        params={"beta0": beta0, "obstruction_copt": OBSTRUCTION_COPT,
                "gap_threshold_min": GAP_THRESHOLD_MIN},
    )


# Conventional Step-3 entry-point name (operates on the typed run + triage, since
# clean.py's tidy-table contract is not yet built).
extract_parameters = fit_dissolution_kinetics
