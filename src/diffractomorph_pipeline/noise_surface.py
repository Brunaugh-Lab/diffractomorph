"""Per-channel noise *surface* + directional-drift significance.

Implements ``Noise_Floor_and_Significance_Component_Spec``. The substrate is the
drug in its antisolvent (CFZ at pH 7), where it does **not** dissolve, so every
frame-to-frame change is pure instrument/sampling noise — "particles bumping
around." We characterize that per channel as a function of **signal level** (and
Copt), then use it as a physically-matched null: a change during a real
dissolution run is real only if its *directional trend* exceeds what bumping alone
produces.

Two pieces:

- :func:`build_noise_surface` → :class:`NoiseSurface` — robust per-channel σ from
  successive differences (MAD), pooled power-law fit ``σ = k·Sᵖ``, lag-1
  autocorrelation ``ρ`` (AR(1) SE inflation), and a per-(channel, Copt) p95 table.
- :meth:`NoiseSurface.is_significant` — per-channel windowed-trend z-test
  (``z = |slope| / SE``, AR(1)-inflated), with the slope sign distinguishing
  dissolution (falling) from growth/precipitation (rising).

This replaces the static ``drop ch1–4`` mask in triage (see
:func:`diffractomorph_pipeline.noise_filter.triage_channels`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from diffractomorph_pipeline.noise import detect_plateaus, flag_fouled

# ── Defaults (spec §8) ───────────────────────────────────────────────────────
OBSTRUCTION_COPT = 40.0      # drop frames above this (laser-obstruction spikes)
DT_OK_MIN = 0.30             # only difference time-contiguous pairs (Δt ≤ this)
# 0.01-count export resolution → MAD collapses to 0 in quiet channels; clamp σ.
SIGMA_QUANT = 1.4826 * 0.01 / np.sqrt(2.0)   # ≈ 0.0105 counts
PLATEAU_TOLERANCE = 0.6
MIN_STABLE_FRAMES = 20
RHO_DEFAULT = 0.2
WINDOW_MIN = 2.0
Z_THRESH = 4.0
DRUG_WINDOW_CH = (15, 31)    # PAQXOS channels for the drug-window summary


# ── Robust per-channel noise from successive differences ─────────────────────

def _successive_diffs(I, idx, t_min, bad, dt_ok=DT_OK_MIN):
    """Δ(c) = I(c,t+1) − I(c,t) over valid, time-contiguous consecutive pairs."""
    idx = np.asarray(idx)
    out = []
    for a, b in zip(idx[:-1], idx[1:]):
        if bad[a] or bad[b] or (t_min[b] - t_min[a]) > dt_ok:
            continue
        out.append(I[b] - I[a])
    return np.array(out)


def _mad_sigma(diffs, sigma_quant=SIGMA_QUANT):
    """Per-frame σ per channel from successive diffs (MAD → σ), with the
    quantization clamp. Returns ``(sigma, quantization_limited_mask)``."""
    med = np.median(diffs, axis=0)
    raw = 1.4826 * np.median(np.abs(diffs - med), axis=0) / np.sqrt(2.0)
    quant_limited = raw <= 0
    return np.maximum(raw, sigma_quant), quant_limited


# ── The surface object ───────────────────────────────────────────────────────

@dataclass
class SignificanceResult:
    channels: list[int]
    zmax: np.ndarray            # per-channel max windowed-trend z
    real: np.ndarray            # bool: zmax > z_thresh
    slope_sign: np.ndarray      # sign of the slope at the max-z window (−1 dissolution, +1 growth)
    z_thresh: float

    @property
    def real_channels(self) -> list[int]:
        return [self.channels[i] for i in np.where(self.real)[0]]


@dataclass
class NoiseSurface:
    """Per-channel σ(signal, Copt) noise surface measured on CFZ-pH-7."""
    k: float
    p: float
    rho: float
    sigma_quant: float
    channels: list[int]
    copt_levels: np.ndarray            # measured plateau Copt (sorted ascending)
    p95_table: np.ndarray              # (n_channels × n_copt) 95th-pct |Δ| per channel/Copt
    meta: dict = field(default_factory=dict)

    @property
    def infl(self) -> float:
        """AR(1) standard-error inflation, √((1+ρ)/(1−ρ))."""
        rho = float(np.clip(self.rho, -0.99, 0.99))
        return float(np.sqrt((1 + rho) / (1 - rho)))

    def sigma(self, signal_level, copt=None) -> np.ndarray:
        """Per-frame σ at a channel's signal level: ``k·Sᵖ`` clamped to σ_quant.

        ``copt`` is accepted for the v2 multiplicative term; v1 is signal-level only
        (the surface residual–Copt correlation is ≈ 0, so signal level captures it).
        """
        S = np.clip(np.asarray(signal_level, dtype=float), 1e-6, None)
        return np.maximum(self.k * S ** self.p, self.sigma_quant)

    def floor_p95(self, channel, copt) -> float:
        """Interpolated 95th-percentile |Δ| detection threshold for a channel at Copt."""
        ci = self.channels.index(int(channel))
        return float(np.interp(copt, self.copt_levels, self.p95_table[ci]))

    def frame_reliable(self, I, snr_min=5.0) -> np.ndarray:
        """Per-frame trust mask: is each channel above its own absolute noise floor *now*?

        ``reliable[t, c] = I[t, c] ≥ snr_min · σ(I[t, c])`` — a channel's signal at a frame is
        trustworthy when it clears its own signal-level σ by an SNR margin. Time-resolved and
        independent of the *other* channels (unlike a signal-fraction test), so it tracks a
        channel as it fills or empties: an emerging size lights up when its own signal rises
        above the floor, and a channel that dissolves into the noise drops out at the frame it
        sinks below it. Complements the whole-run admission in :func:`noise_filter`; used at the
        fit stage to fit each channel only over the frames it is real.
        """
        I = np.asarray(I, dtype=float)
        return I >= snr_min * self.sigma(np.clip(I, 1e-6, None))

    def is_significant(self, I, copt, t_min, window_min=WINDOW_MIN, z_thresh=Z_THRESH,
                       smooth_w=5) -> SignificanceResult:
        """Per-channel directional-drift test on a real run (spec §5).

        A channel is *really changing* when its windowed-trend z exceeds
        ``z_thresh`` for some ~``window_min`` window; σ is looked up at each
        window's own signal level so an emptying channel inherits the right floor.
        """
        I = np.asarray(I, dtype=float)
        t_min = np.asarray(t_min, dtype=float)
        n, C = I.shape
        dt = max(float(np.median(np.diff(t_min))), 1e-3)
        win = max(8, int(window_min / dt))
        s_local = np.column_stack([_smooth(I[:, c], smooth_w) for c in range(C)])

        zmax = np.zeros(C)
        slope_at = np.zeros(C)
        for c in range(C):
            best_z, best_slope = 0.0, 0.0
            for s in range(0, max(1, n - win), max(1, win // 2)):
                sl = slice(s, s + win)
                tt = t_min[sl] - t_min[sl].mean()
                if tt.std() < 1e-6 or (sl.stop - sl.start) < 5:
                    continue
                slope = np.polyfit(tt, I[sl, c], 1)[0]
                sig = float(self.sigma(max(s_local[sl, c].mean(), 1e-6)))
                se = self.infl * sig / (tt.std() * np.sqrt(min(win, n - s)))
                z = abs(slope) / se if se > 0 else 0.0
                if z > best_z:
                    best_z, best_slope = z, slope
            zmax[c], slope_at[c] = best_z, best_slope
        real = zmax > z_thresh
        return SignificanceResult(list(self.channels), zmax, real,
                                  np.sign(slope_at), z_thresh)

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "k": self.k, "p": self.p, "rho": self.rho, "sigma_quant": self.sigma_quant,
            "channels": list(self.channels), "copt_levels": self.copt_levels.tolist(),
            "p95_table": self.p95_table.tolist(), "meta": self.meta,
        }, indent=2))
        return path

    @classmethod
    def load(cls, path: Path | str) -> "NoiseSurface":
        d = json.loads(Path(path).read_text())
        return cls(k=d["k"], p=d["p"], rho=d["rho"], sigma_quant=d["sigma_quant"],
                   channels=[int(c) for c in d["channels"]],
                   copt_levels=np.array(d["copt_levels"]),
                   p95_table=np.array(d["p95_table"]), meta=d.get("meta", {}))


def _smooth(y, w=5):
    if len(y) < w or w < 2:
        return np.asarray(y, dtype=float)
    return np.convolve(y, np.ones(w) / w, mode="same")


# ── Build the surface from CFZ-pH-7 titration runs ───────────────────────────

def build_noise_surface(runs, obstruction_copt=OBSTRUCTION_COPT, dt_ok_min=DT_OK_MIN,
                        sigma_quant=SIGMA_QUANT, plateau_tolerance=PLATEAU_TOLERANCE,
                        min_stable_frames=MIN_STABLE_FRAMES, exclude_fouled=True,
                        rho_default=RHO_DEFAULT) -> NoiseSurface:
    """Build a :class:`NoiseSurface` from one or more CFZ-pH-7 titration runs.

    ``runs`` is a list of ``(label, RawRun)`` (``.I``, ``.copt``, ``.t_min``,
    ``.ref``). Fouled runs (elevated reference background) are excluded by default.
    """
    fouled = flag_fouled(runs) if exclude_fouled else []
    clean = [(lab, r) for lab, r in runs if lab not in fouled]
    if not clean:
        raise ValueError("No clean CFZ-pH-7 runs to build the surface from.")

    S_all, SIG_all, ac1 = [], [], []
    copt_pts, p95_cols, channels = [], [], None
    for lab, run in clean:
        I, copt, t = np.asarray(run.I, float), np.asarray(run.copt, float), np.asarray(run.t_min, float)
        channels = list(run.channels)
        bad = copt > obstruction_copt
        cc = copt.copy(); cc[bad] = np.nan
        for pl in detect_plateaus(np.nan_to_num(cc, nan=1e3), t,
                                  tolerance=plateau_tolerance, min_stable_frames=min_stable_frames):
            if np.isnan(cc[pl.indices]).all():
                continue
            diffs = _successive_diffs(I, pl.indices, t, bad, dt_ok_min)
            if diffs.shape[0] < 8:
                continue
            sig, _ = _mad_sigma(diffs, sigma_quant)
            good = np.array([i for i in pl.indices if not bad[i]])
            S = I[good].mean(axis=0)
            S_all.append(S); SIG_all.append(sig)
            copt_pts.append(float(np.nanmean(cc[pl.indices])))
            p95_cols.append(np.maximum(np.percentile(np.abs(diffs), 95, axis=0), sigma_quant))
            # lag-1 autocorrelation of detrended per-channel level series
            tt = np.arange(good.size)
            for c in range(I.shape[1]):
                y = I[good, c]
                if y.std() < 1e-9:
                    continue
                r = y - np.polyval(np.polyfit(tt, y, 1), tt)
                if r[:-1].std() > 1e-9 and r[1:].std() > 1e-9:
                    ac1.append(np.corrcoef(r[:-1], r[1:])[0, 1])

    S_all = np.concatenate(S_all); SIG_all = np.concatenate(SIG_all)
    m = np.isfinite(S_all) & np.isfinite(SIG_all) & (S_all > 1e-3) & (SIG_all > 1e-6)
    p, logk = np.polyfit(np.log(S_all[m]), np.log(SIG_all[m]), 1)
    k = float(np.exp(logk))
    rho = float(np.median(ac1)) if ac1 else rho_default

    # Per-(channel, Copt) p95 table, sorted ascending by Copt.
    order = np.argsort(copt_pts)
    copt_levels = np.array(copt_pts)[order]
    p95_table = np.column_stack(p95_cols)[:, order]    # (n_channels × n_plateaus)

    meta = {
        "k": k, "p": float(p), "rho": rho, "sigma_quant": float(sigma_quant),
        "n_reps": len(clean), "sources": [lab for lab, _ in clean],
        "excluded_fouled": fouled,
        "ref_sums": {lab: round(float(np.asarray(r.ref).sum()), 1) for lab, r in clean},
        "n_points": int(m.sum()), "rel_slope": float(p - 1),
        "params": {"obstruction_copt": obstruction_copt, "dt_ok_min": dt_ok_min,
                   "plateau_tolerance": plateau_tolerance, "min_stable_frames": min_stable_frames},
    }
    return NoiseSurface(k=k, p=float(p), rho=rho, sigma_quant=float(sigma_quant),
                        channels=channels, copt_levels=copt_levels,
                        p95_table=p95_table, meta=meta)


# ── Packaged default surface ─────────────────────────────────────────────────

def _surface_path() -> Path:
    from importlib import resources
    return Path(resources.files("diffractomorph_pipeline")) / "data" / "noise" / "cfz_ph7_surface.json"


def load_surface(path: Path | str | None = None) -> NoiseSurface:
    """Load a caller-selected surface or the optional legacy CFZ profile."""
    selected = Path(path) if path is not None else _surface_path()
    if not selected.exists():
        raise FileNotFoundError(
            "the optional CFZ noise surface is not installed; pass an explicit surface path"
        )
    return NoiseSurface.load(selected)
