"""Manuscript-authoritative aggregate angular-signal KWW analysis."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from diffractomorph_pipeline import kinetics
from diffractomorph_pipeline.model import Run


ReferenceMode = Literal["raw_measured", "reference_adjusted"]
StartPolicy = Literal["first_frame", "concordant_early_maximum"]


@dataclass(frozen=True)
class AggregateKWWConfig:
    channel_ids: tuple[str, ...] | None = None
    reference_mode: ReferenceMode = "raw_measured"
    upward_hampel_z: float = 4.0
    upward_hampel_half_window: int = 3
    start_policy: StartPolicy = "first_frame"
    start_acquisition_variable: str = "copt"
    start_search_frames: int = 3
    start_maximum_time_min: float | None = None
    start_minimum_relative_increase: float | None = None
    start_minimum_spectral_cosine: float | None = None
    tau_bounds_min: tuple[float, float] = (0.05, 500.0)
    beta_bounds: tuple[float, float] = (0.2, 3.0)

    def validate_start_boundary(self) -> None:
        """Validate the optional start policy before any run is processed."""
        if self.start_policy == "first_frame":
            return
        if self.start_policy != "concordant_early_maximum":
            raise ValueError(f"unsupported start policy: {self.start_policy!r}")
        if self.start_search_frames < 2:
            raise ValueError("start_search_frames must be at least 2")
        if self.start_maximum_time_min is None or self.start_maximum_time_min < 0:
            raise ValueError("start_maximum_time_min must be declared and nonnegative")
        if (
            self.start_minimum_relative_increase is None
            or self.start_minimum_relative_increase < 0
        ):
            raise ValueError("start_minimum_relative_increase must be declared and nonnegative")
        if (
            self.start_minimum_spectral_cosine is None
            or not 0 <= self.start_minimum_spectral_cosine <= 1
        ):
            raise ValueError(
                "start_minimum_spectral_cosine must be declared between 0 and 1"
            )

    @classmethod
    def from_profile(cls, parameters) -> "AggregateKWWConfig":
        """Build from an explicit manifest analysis profile."""
        required = {
            "channel_set", "stored_reference_subtraction", "start_boundary",
            "upward_hampel", "tau_bounds_min", "beta_bounds",
        }
        missing = sorted(required - set(parameters))
        if missing:
            raise ValueError("analysis profile missing: " + ", ".join(missing))
        channel_set = parameters["channel_set"]
        if channel_set == "all_measured":
            channel_ids = None
        elif isinstance(channel_set, (list, tuple)) and channel_set:
            channel_ids = tuple(str(value) for value in channel_set)
        else:
            raise ValueError("analysis channel_set must be 'all_measured' or a non-empty list")
        hampel = parameters["upward_hampel"]
        missing_hampel = sorted({"threshold_mad", "half_window_frames"} - set(hampel))
        if missing_hampel:
            raise ValueError("upward_hampel missing: " + ", ".join(missing_hampel))
        reference_mode = (
            "reference_adjusted"
            if parameters["stored_reference_subtraction"]
            else "raw_measured"
        )
        start = parameters["start_boundary"]
        policy = str(start.get("policy", "first_frame"))
        if policy not in ("first_frame", "concordant_early_maximum"):
            raise ValueError(
                "start_boundary.policy must be 'first_frame' or "
                "'concordant_early_maximum'"
            )
        if policy == "concordant_early_maximum":
            required_start = {
                "acquisition_variable", "search_frames", "maximum_time_min",
                "minimum_relative_increase", "minimum_spectral_cosine",
            }
            missing_start = sorted(required_start - set(start))
            if missing_start:
                raise ValueError("start_boundary missing: " + ", ".join(missing_start))
        config = cls(
            channel_ids=channel_ids,
            reference_mode=reference_mode,
            upward_hampel_z=float(hampel["threshold_mad"]),
            upward_hampel_half_window=int(hampel["half_window_frames"]),
            start_policy=policy,
            start_acquisition_variable=str(start.get("acquisition_variable", "copt")),
            start_search_frames=int(start.get("search_frames", 3)),
            start_maximum_time_min=(
                float(start["maximum_time_min"])
                if "maximum_time_min" in start else None
            ),
            start_minimum_relative_increase=(
                float(start["minimum_relative_increase"])
                if "minimum_relative_increase" in start else None
            ),
            start_minimum_spectral_cosine=(
                float(start["minimum_spectral_cosine"])
                if "minimum_spectral_cosine" in start else None
            ),
            tau_bounds_min=tuple(float(value) for value in parameters["tau_bounds_min"]),
            beta_bounds=tuple(float(value) for value in parameters["beta_bounds"]),
        )
        config.validate_start_boundary()
        return config


@dataclass(frozen=True)
class AggregateKWWResult:
    run_id: str
    independent_unit_id: str | None
    time_min: np.ndarray
    aggregate_signal: np.ndarray
    upward_repair_mask: np.ndarray
    channel_ids: tuple[str, ...]
    reference_mode: ReferenceMode
    start_index: int
    selected_elapsed_time_min: float
    start_reason: str
    fit: dict
    config: dict
    observable: str = "angle-integrated detector signal"
    is_particle_mass: bool = False

    def to_row(self) -> dict:
        row = {
            "run_id": self.run_id,
            "independent_unit_id": self.independent_unit_id,
            "reference_mode": self.reference_mode,
            "start_policy": self.config["start_policy"],
            "start_index": self.start_index,
            "selected_elapsed_time_min": self.selected_elapsed_time_min,
            "start_reason": self.start_reason,
            "n_channels": len(self.channel_ids),
            "n_upward_repairs": int(self.upward_repair_mask.sum()),
            **self.fit,
        }
        row["optical_decay_depth_pct"] = 100.0 * float(self.fit["depth"])
        row["i0_fit"] = float(self.fit["i0"])
        return row


def aggregate_signal(run: Run, config: AggregateKWWConfig | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    """Sum the declared measured detector channels; never substitute Copt or q3."""
    config = config or AggregateKWWConfig()
    selected = config.channel_ids or run.channel_ids
    unknown = [channel for channel in selected if channel not in run.channel_ids]
    if unknown:
        raise ValueError(f"unknown aggregate channels: {', '.join(unknown)}")
    indices = [run.channel_ids.index(channel) for channel in selected]
    signal = run.signal[:, indices].sum(axis=1)
    if config.reference_mode == "reference_adjusted":
        if run.stored_reference is None:
            raise ValueError("reference_adjusted aggregate requires a stored reference")
        signal = signal - run.stored_reference[indices].sum()
    elif config.reference_mode != "raw_measured":
        raise ValueError("reference_mode must be 'raw_measured' or 'reference_adjusted'")
    return signal, tuple(selected)


def select_aggregate_start(
    run: Run,
    aggregate: np.ndarray,
    channel_ids: tuple[str, ...],
    config: AggregateKWWConfig,
) -> tuple[int, str]:
    """Select the first analysis frame under an explicit acquisition-start policy.

    ``first_frame`` preserves the released behavior. ``concordant_early_maximum``
    recognizes incomplete initial circulation only when the same later frame is
    the early maximum of both total angular signal and a declared acquisition
    variable within the declared startup interval, both increases meet or exceed
    the declared minimum, and the angular pattern remains shape-preserving. If
    those conditions are not met, frame 0 remains the start. The caller re-zeros
    time after selection.
    """
    if config.start_policy == "first_frame":
        return 0, "first_frame"
    config.validate_start_boundary()

    aggregate = np.asarray(aggregate, dtype=float)
    if aggregate.shape != run.time_min.shape:
        raise ValueError("aggregate must contain one value per run frame")
    acquisition = run.acquisition_variable(config.start_acquisition_variable)
    n_early = min(config.start_search_frames, aggregate.size)
    if not (
        np.all(np.isfinite(aggregate[:n_early]))
        and np.all(np.isfinite(acquisition[:n_early]))
    ):
        return 0, "nonfinite_early_start_variable"
    signal_index = int(np.argmax(aggregate[:n_early]))
    acquisition_index = int(np.argmax(acquisition[:n_early]))
    if signal_index == 0 or signal_index != acquisition_index:
        return 0, "no_concordant_early_maximum"
    selected_time = float(run.time_min[signal_index] - run.time_min[0])
    if selected_time > config.start_maximum_time_min:
        return 0, "early_maximum_after_startup_interval"

    baseline_signal = float(aggregate[0])
    baseline_acquisition = float(acquisition[0])
    if baseline_signal <= 0 or baseline_acquisition <= 0:
        return 0, "nonpositive_start_variable"
    signal_increase = float(aggregate[signal_index] / baseline_signal - 1.0)
    acquisition_increase = float(acquisition[signal_index] / baseline_acquisition - 1.0)
    minimum = config.start_minimum_relative_increase
    if signal_increase < minimum or acquisition_increase < minimum:
        return 0, "early_maximum_below_minimum_increase"

    indices = [run.channel_ids.index(channel) for channel in channel_ids]
    patterns = np.asarray(run.signal[:, indices], dtype=float)
    if config.reference_mode == "reference_adjusted":
        if run.stored_reference is None:
            raise ValueError("reference_adjusted start selection requires a stored reference")
        patterns = patterns - run.stored_reference[indices][None, :]
    first_pattern = patterns[0]
    peak_pattern = patterns[signal_index]
    denominator = float(np.linalg.norm(first_pattern) * np.linalg.norm(peak_pattern))
    cosine = float(np.dot(first_pattern, peak_pattern) / denominator) if denominator else float("nan")
    if not np.isfinite(cosine) or cosine < config.start_minimum_spectral_cosine:
        return 0, "early_maximum_changed_angular_pattern"
    return signal_index, "concordant_early_maximum"


def fit_aggregate_kww(run: Run, config: AggregateKWWConfig | None = None) -> AggregateKWWResult:
    """Fit the all-measured-channel aggregate with a free-amplitude KWW descriptor."""
    config = config or AggregateKWWConfig()
    signal, selected = aggregate_signal(run, config)
    start_index, start_reason = select_aggregate_start(run, signal, selected, config)
    original_start_time = float(run.time_min[start_index])
    selected_elapsed_time = original_start_time - float(run.time_min[0])
    analysis_time = run.time_min[start_index:] - original_start_time
    analysis_signal = signal[start_index:]
    time_min, cleaned, repaired = kinetics.despike_upward(
        analysis_time,
        analysis_signal,
        z=config.upward_hampel_z,
        half=config.upward_hampel_half_window,
        return_mask=True,
    )
    fit = kinetics.fit_signal(
        time_min,
        cleaned,
        tau_bounds=config.tau_bounds_min,
        beta_bounds=config.beta_bounds,
    )
    return AggregateKWWResult(
        run_id=run.provenance.run_id,
        independent_unit_id=run.provenance.independent_unit_id,
        time_min=time_min,
        aggregate_signal=cleaned,
        upward_repair_mask=repaired,
        channel_ids=selected,
        reference_mode=config.reference_mode,
        start_index=start_index,
        selected_elapsed_time_min=selected_elapsed_time,
        start_reason=start_reason,
        fit=fit,
        config=asdict(config),
    )
