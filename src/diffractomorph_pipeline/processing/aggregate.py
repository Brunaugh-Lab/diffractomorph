"""Manuscript-authoritative aggregate angular-signal KWW analysis."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from diffractomorph_pipeline import kinetics
from diffractomorph_pipeline.model import Run


ReferenceMode = Literal["raw_measured", "reference_adjusted"]


@dataclass(frozen=True)
class AggregateKWWConfig:
    channel_ids: tuple[str, ...] | None = None
    reference_mode: ReferenceMode = "raw_measured"
    upward_hampel_z: float = 4.0
    upward_hampel_half_window: int = 3
    tau_bounds_min: tuple[float, float] = (0.05, 500.0)
    beta_bounds: tuple[float, float] = (0.2, 3.0)

    @classmethod
    def from_profile(cls, parameters) -> "AggregateKWWConfig":
        """Build from an explicit manifest analysis profile."""
        required = {"channel_set", "stored_reference_subtraction", "upward_hampel", "tau_bounds_min", "beta_bounds"}
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
        return cls(
            channel_ids=channel_ids,
            reference_mode=reference_mode,
            upward_hampel_z=float(hampel["threshold_mad"]),
            upward_hampel_half_window=int(hampel["half_window_frames"]),
            tau_bounds_min=tuple(float(value) for value in parameters["tau_bounds_min"]),
            beta_bounds=tuple(float(value) for value in parameters["beta_bounds"]),
        )


@dataclass(frozen=True)
class AggregateKWWResult:
    run_id: str
    independent_unit_id: str | None
    time_min: np.ndarray
    aggregate_signal: np.ndarray
    upward_repair_mask: np.ndarray
    channel_ids: tuple[str, ...]
    reference_mode: ReferenceMode
    fit: dict
    config: dict
    observable: str = "angle-integrated detector signal"
    is_particle_mass: bool = False

    def to_row(self) -> dict:
        row = {
            "run_id": self.run_id,
            "independent_unit_id": self.independent_unit_id,
            "reference_mode": self.reference_mode,
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


def fit_aggregate_kww(run: Run, config: AggregateKWWConfig | None = None) -> AggregateKWWResult:
    """Fit the all-measured-channel aggregate with a free-amplitude KWW descriptor."""
    config = config or AggregateKWWConfig()
    signal, selected = aggregate_signal(run, config)
    time_min, cleaned, repaired = kinetics.despike_upward(
        run.time_min,
        signal,
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
        fit=fit,
        config=asdict(config),
    )
