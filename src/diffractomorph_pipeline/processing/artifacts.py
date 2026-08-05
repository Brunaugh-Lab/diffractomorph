"""Frame-artifact correction with a complete, frame-level repair ledger.

The synchronized detector/Copt rule and isolated-channel median rule reproduce the
manuscript workflow. They remain separate from the one-sided Hampel filter applied
to the aggregate trajectory before KWW fitting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from diffractomorph_pipeline.model import Run
from diffractomorph_pipeline.noise_filter import despike_frames


@dataclass(frozen=True)
class ArtifactCorrectionConfig:
    acquisition_variable: str = "copt"
    synchronized_intensity_z: float = 4.0
    synchronized_acquisition_z: float = 1.5
    synchronized_half_window: int = 2
    isolated_spike_mad: float = 5.0
    gap_threshold_min: float = 2.0

    @classmethod
    def from_profile(cls, parameters) -> "ArtifactCorrectionConfig":
        required = {
            "acquisition_variable", "synchronized_intensity_z",
            "synchronized_acquisition_z", "synchronized_half_window",
            "isolated_spike_mad", "gap_threshold_min",
        }
        missing = sorted(required - set(parameters))
        if missing:
            raise ValueError(f"artifact-correction profile missing: {', '.join(missing)}")
        return cls(
            acquisition_variable=str(parameters["acquisition_variable"]),
            synchronized_intensity_z=float(parameters["synchronized_intensity_z"]),
            synchronized_acquisition_z=float(parameters["synchronized_acquisition_z"]),
            synchronized_half_window=int(parameters["synchronized_half_window"]),
            isolated_spike_mad=float(parameters["isolated_spike_mad"]),
            gap_threshold_min=float(parameters["gap_threshold_min"]),
        )


@dataclass(frozen=True)
class ArtifactCorrectionResult:
    run: Run
    ledger: pd.DataFrame
    config: ArtifactCorrectionConfig


def _isolated_channel_spikes(signal: np.ndarray, threshold: float):
    """Return median-repaired signal and its frame-by-channel repair mask."""
    signal = np.asarray(signal, float)
    repaired = signal.copy()
    mask = np.zeros(signal.shape, dtype=bool)
    if signal.shape[0] < 5:
        return repaired, mask
    for channel in range(signal.shape[1]):
        values = signal[:, channel]
        local = np.median(
            np.stack([np.r_[values[0], values[:-1]], values, np.r_[values[1:], values[-1]]]),
            axis=0,
        )
        residual = np.abs(values - local)
        mad = float(np.median(np.abs(values - np.median(values)))) or 1.0
        channel_mask = residual > threshold * mad
        repaired[channel_mask, channel] = local[channel_mask]
        mask[:, channel] = channel_mask
    return repaired, mask


def correct_artifacts(run: Run, config: ArtifactCorrectionConfig | None = None) -> ArtifactCorrectionResult:
    """Correct manuscript-defined artifacts without changing the scientific observable.

    The required acquisition variable is explicit. Missing Copt-like data is an error,
    because silently skipping corroboration would implement a different QC rule.
    """
    config = config or ArtifactCorrectionConfig()
    acquisition = run.acquisition_variable(config.acquisition_variable)
    original_n = run.signal.shape[0]
    original_time = run.time_min.copy()
    actions: list[list[str]] = [[] for _ in range(original_n)]
    repaired_channels: list[list[str]] = [[] for _ in range(original_n)]

    signal, time_min, corrected_acq, info = despike_frames(
        run.signal,
        run.time_min,
        acquisition,
        z_int=config.synchronized_intensity_z,
        z_copt_corrob=config.synchronized_acquisition_z,
        w=config.synchronized_half_window,
    )
    lead = int(info.get("n_lead_dropped", 0))
    original_indices = np.arange(original_n)[lead:]
    for index in range(lead):
        actions[index].append("startup_dropped")
    for index in info.get("interior_fixed", []):
        actions[int(index)].append("synchronized_interpolated")

    # Align every frame-level acquisition variable with synchronized repairs.
    new_acquisition: dict[str, np.ndarray] = {}
    interior = np.asarray(info.get("interior_fixed", []), dtype=int)
    flagged = np.asarray(info.get("spike_frames", []), dtype=int)
    retained_original = np.arange(original_n)[lead:]
    for name, values in run.acquisition.items():
        values = np.asarray(values, float).copy()
        if name == config.acquisition_variable:
            aligned = np.asarray(corrected_acq, float)
        else:
            if interior.size:
                good = np.array([i for i in range(original_n) if i not in set(flagged)], dtype=int)
                if good.size >= 2:
                    values[interior] = np.interp(original_time[interior], original_time[good], values[good])
            aligned = values[lead:]
        new_acquisition[name] = aligned

    signal, isolated = _isolated_channel_spikes(signal, config.isolated_spike_mad)
    for frame_index, channel_index in zip(*np.where(isolated)):
        original_index = int(original_indices[frame_index])
        if "channel_median_replaced" not in actions[original_index]:
            actions[original_index].append("channel_median_replaced")
        repaired_channels[original_index].append(run.channel_ids[channel_index])

    gap_rezeroed = False
    gap_min = 0.0
    if time_min.size >= 2:
        gaps = np.diff(time_min)
        gap_index = int(np.argmax(gaps))
        gap_min = float(gaps[gap_index])
        if gap_min > config.gap_threshold_min:
            gap_rezeroed = True
            drop_count = gap_index + 1
            for original_index in original_indices[:drop_count]:
                actions[int(original_index)].append("pre_gap_dropped")
            signal = signal[drop_count:]
            time_min = time_min[drop_count:] - time_min[drop_count]
            original_indices = original_indices[drop_count:]
            new_acquisition = {name: values[drop_count:] for name, values in new_acquisition.items()}

    flags = dict(run.flags)
    flags["artifact_correction"] = {
        "startup_frames_dropped": lead,
        "synchronized_frames_interpolated": len(info.get("interior_fixed", [])),
        "isolated_channel_points_replaced": int(isolated.sum()),
        "gap_rezeroed": gap_rezeroed,
        "gap_min": gap_min,
    }
    corrected = Run(
        signal=signal,
        channel_ids=run.channel_ids,
        time_min=time_min,
        acquisition=new_acquisition,
        provenance=run.provenance,
        run_kind=run.run_kind,
        stored_reference=run.stored_reference,
        started_at=run.started_at,
        measurement_name=run.measurement_name,
        flags=flags,
    )
    ledger = pd.DataFrame({
        "original_frame": np.arange(original_n, dtype=int),
        "original_time_min": original_time,
        "action": [";".join(value) if value else "retained" for value in actions],
        "repaired_channels": [";".join(value) for value in repaired_channels],
        "retained_in_output": [int(index) in set(original_indices) for index in range(original_n)],
    })
    return ArtifactCorrectionResult(corrected, ledger, config)
