"""Reliability-gated q3 shape descriptors at matched detector-signal extent.

q3 is treated only as PAQXOS-inverted relative composition from the same optical
acquisition. This module never de-normalizes q3 into particle mass.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffractomorph_pipeline import psd


@dataclass(frozen=True)
class MatchedExtentConfig:
    targets: tuple[float, ...]
    match_tolerance_s: float
    anchor_frames: int
    noise_multiplier: float
    acquisition_floor_absolute: float
    size_max_um: float
    tail_fraction_max_pct: float
    coarse_review_size_um: float
    coarse_review_fraction_pct: float
    exclude_coarse_review: bool


def _first_crossing(values: np.ndarray, target: float) -> int | None:
    below = np.where(np.asarray(values, float) <= target)[0]
    return int(below[0]) if below.size and below[0] > 0 else None


def _metrics(xo, cumulative, anchor, config: MatchedExtentConfig) -> dict:
    d10, d50, d90 = psd.q3_percentiles(xo, cumulative)
    tail = psd.q3_tail_fraction(xo, cumulative, config.size_max_um)
    coarse_tail = psd.q3_tail_fraction(xo, cumulative, config.coarse_review_size_um)
    tail_unstable = bool(
        tail > config.tail_fraction_max_pct or (np.isfinite(d90) and d90 > config.size_max_um)
    )
    if np.isfinite(d90) and d90 > config.size_max_um:
        d90 = np.nan
    span = (
        (d90 - d10) / d50
        if np.isfinite(d10) and np.isfinite(d50) and np.isfinite(d90) and d50 > 0
        else np.nan
    )
    xr, cr = psd.restrict_cumulative(xo, cumulative, config.size_max_um)
    d10r, d50r, d90r = psd.q3_percentiles(xr, cr)
    spanr = (
        (d90r - d10r) / d50r
        if np.isfinite(d10r) and np.isfinite(d50r) and np.isfinite(d90r) and d50r > 0
        else np.nan
    )
    anchor_d50 = psd.q3_percentiles(xo, anchor, (50.0,))[0]
    dlog = (
        np.log10(d50) - np.log10(anchor_d50)
        if np.isfinite(d50) and np.isfinite(anchor_d50) and d50 > 0 and anchor_d50 > 0
        else np.nan
    )
    fractions = np.diff(cumulative, prepend=0.0) / 100.0
    anchor_fractions = np.diff(anchor, prepend=0.0) / 100.0
    return {
        "D10": d10, "D50": d50, "D90": d90, "span": span,
        "D10_restricted": d10r, "D50_restricted": d50r,
        "D90_restricted": d90r, "span_restricted": spanr,
        "dlog10_D50": dlog,
        "tv_distance": 0.5 * float(np.abs(fractions - anchor_fractions).sum()),
        "tail_pct_gt_size_max": tail,
        "tail_unstable": tail_unstable,
        "coarse_tail_pct": coarse_tail,
        "coarse_review": bool(coarse_tail > config.coarse_review_fraction_pct),
    }


def matched_q3_extent(
    frames: psd.Q3Frames,
    *,
    detector_epoch_s,
    detector_remaining_fraction,
    detector_start_signal: float,
    detector_noise_sigma: float,
    acquisition_variable,
    config: MatchedExtentConfig | None = None,
) -> dict[float, dict]:
    """Return one explicit result or rejection reason for every requested extent."""
    if config is None:
        raise ValueError("matched-extent reliability config is required")
    epoch = np.asarray(detector_epoch_s, float)
    remaining = np.asarray(detector_remaining_fraction, float)
    acquisition = np.asarray(acquisition_variable, float)
    if epoch.shape != remaining.shape or epoch.shape != acquisition.shape:
        raise ValueError("detector epoch, remaining fraction, and acquisition arrays must align")
    if epoch.ndim != 1 or epoch.size < 2 or np.any(np.diff(epoch) < 0):
        raise ValueError("detector epoch must be a chronological one-dimensional series")
    invalid_remaining = not np.all(np.isfinite(remaining)) or np.any(np.diff(remaining) > 1e-9)
    if any(not 0.0 < target < 1.0 for target in config.targets):
        raise ValueError("matched-extent targets must lie strictly between zero and one")
    if not np.isfinite(acquisition).any() or np.nanmax(acquisition) <= 0:
        raise ValueError("acquisition reliability variable requires a positive finite value")
    pairs, unmatched_q3, unmatched_detector = psd.match_frames_by_time(
        frames.t_epoch, epoch, tol_s=config.match_tolerance_s,
    )
    base = {
        "n_matched": len(pairs),
        "n_unmatched_q3": len(unmatched_q3),
        "n_unmatched_detector": len(unmatched_detector),
        "q3_role": "model_inverted_relative_composition",
        "is_mass_measurement": False,
        "is_independent_of_detector": False,
    }
    if invalid_remaining:
        return {target: {**base, "reason": "invalid_anchor"} for target in config.targets}
    if detector_start_signal <= 0 or len(pairs) < 5:
        reason = "invalid_anchor" if detector_start_signal <= 0 else "few_frames"
        return {target: {**base, "reason": reason} for target in config.targets}

    detector_indices = np.array([pair[1] for pair in pairs])
    q3_indices = np.array([pair[0] for pair in pairs])
    matched_remaining = remaining[detector_indices]
    matched_cumulative = frames.Q3_cum[q3_indices]
    matched_epoch = epoch[detector_indices]
    matched_acquisition = acquisition[detector_indices]
    reliable = matched_acquisition >= config.acquisition_floor_absolute
    matched_dq3 = frames.dQ3[q3_indices]
    valid_q3 = (
        np.all(np.isfinite(matched_cumulative), axis=1)
        & np.all(np.isfinite(matched_dq3), axis=1)
        & (np.abs(np.sum(matched_dq3, axis=1) - 1.0) < 0.05)
    )
    anchor_eligible = reliable & valid_q3
    if anchor_eligible.sum() < config.anchor_frames:
        return {target: {**base, "reason": "invalid_anchor"} for target in config.targets}
    anchor = matched_cumulative[anchor_eligible][:config.anchor_frames].mean(axis=0)

    out: dict[float, dict] = {}
    for target in config.targets:
        record = dict(base)
        if target * detector_start_signal <= config.noise_multiplier * detector_noise_sigma:
            out[target] = {**record, "reason": "below_noise"}
            continue
        if _first_crossing(remaining, target) is None:
            out[target] = {**record, "reason": "not_reached"}
            continue
        crossing = _first_crossing(matched_remaining, target)
        if crossing is None:
            out[target] = {**record, "reason": "unmatched"}
            continue
        if not (reliable[crossing - 1] and reliable[crossing]):
            out[target] = {**record, "reason": "q3_unreliable"}
            continue
        if not (valid_q3[crossing - 1] and valid_q3[crossing]):
            out[target] = {**record, "reason": "q3_malformed"}
            continue
        interpolation = psd.interp_cumulative_at_g(matched_remaining, matched_cumulative, target)
        if interpolation is None:
            out[target] = {**record, "reason": "unmatched"}
            continue
        g_at, cumulative, index, fraction = interpolation
        time_min = (
            matched_epoch[index - 1]
            + fraction * (matched_epoch[index] - matched_epoch[index - 1])
            - epoch[0]
        ) / 60.0
        record.update(reason="ok", g=float(g_at), t_min=float(time_min))
        record.update(_metrics(frames.xo, cumulative, anchor, config))
        if config.exclude_coarse_review and record["coarse_review"]:
            record["reason"] = "coarse_review"
        out[target] = record
    return out
