"""Instrument-neutral data contracts for DiffractoMorph.

The public contract is :class:`Run`: a time-by-channel signal matrix plus explicit
channel identifiers, acquisition variables, stored-reference data, and provenance.
Legacy PAQXOS attribute names remain as read-only compatibility properties while the
rest of the package migrates to the neutral vocabulary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

_TIME_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class RunProvenance:
    """Identity and origin of one acquired run."""

    run_id: str
    source_path: str
    adapter: str
    sample_id: str
    independent_unit_id: str | None = None
    technical_replicate: str | None = None
    instrument_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("run_id", "source_path", "adapter", "sample_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"provenance.{name} must be non-empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass
class Run:
    """Chronologically ordered, instrument-neutral run.

    ``signal`` is shaped ``(n_frames, n_channels)``. Acquisition variables are
    named one-dimensional arrays with one value per frame. Stored reference is
    optional because not every instrument exports one.
    """

    signal: np.ndarray
    channel_ids: tuple[str, ...]
    time_min: np.ndarray
    acquisition: dict[str, np.ndarray]
    provenance: RunProvenance
    run_kind: str
    stored_reference: np.ndarray | None = None
    started_at: datetime | None = None
    measurement_name: str = ""
    flags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.signal = np.asarray(self.signal, dtype=float)
        self.time_min = np.asarray(self.time_min, dtype=float)
        self.channel_ids = tuple(str(c) for c in self.channel_ids)
        self.acquisition = {
            str(name): np.asarray(values, dtype=float)
            for name, values in self.acquisition.items()
        }
        if self.stored_reference is not None:
            self.stored_reference = np.asarray(self.stored_reference, dtype=float)

        if self.signal.ndim != 2:
            raise ValueError("signal must be a two-dimensional frame-by-channel matrix")
        n_frames, n_channels = self.signal.shape
        if n_frames < 1 or n_channels < 1:
            raise ValueError("signal must contain at least one frame and one channel")
        if len(self.channel_ids) != n_channels:
            raise ValueError("channel_ids length must match signal columns")
        if any(not channel.strip() for channel in self.channel_ids):
            raise ValueError("channel_ids must be non-empty")
        if len(set(self.channel_ids)) != len(self.channel_ids):
            raise ValueError("channel_ids must be unique")
        if self.time_min.shape != (n_frames,):
            raise ValueError("time_min length must match signal frames")
        if not np.all(np.isfinite(self.time_min)):
            raise ValueError("time_min must be finite")
        if np.any(np.diff(self.time_min) < 0):
            raise ValueError("time_min must be chronological")
        for name, values in self.acquisition.items():
            if values.shape != (n_frames,):
                raise ValueError(f"acquisition variable {name!r} must have one value per frame")
        if self.stored_reference is not None and self.stored_reference.shape != (n_channels,):
            raise ValueError("stored_reference length must match signal columns")
        if not str(self.run_kind).strip():
            raise ValueError("run_kind must be non-empty")

    def acquisition_variable(self, name: str) -> np.ndarray:
        """Return a named acquisition variable, failing explicitly when absent."""
        try:
            return self.acquisition[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.acquisition)) or "none"
            raise KeyError(f"acquisition variable {name!r} is unavailable; available: {available}") from exc

    # Legacy PAQXOS-facing compatibility properties. New code should use the
    # neutral names above.
    @property
    def I(self) -> np.ndarray:
        return self.signal

    @property
    def ref(self) -> np.ndarray:
        if self.stored_reference is None:
            raise AttributeError("this run has no stored reference")
        return self.stored_reference

    @ref.setter
    def ref(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        if values.shape != (self.signal.shape[1],):
            raise ValueError("ref length must match signal columns")
        self.stored_reference = values

    @property
    def copt(self) -> np.ndarray:
        try:
            return self.acquisition_variable("copt")
        except KeyError as exc:
            raise AttributeError("this run has no copt acquisition variable") from exc

    @property
    def t_min(self) -> np.ndarray:
        return self.time_min

    @property
    def t0(self) -> datetime | None:
        return self.started_at

    @property
    def channels(self) -> list[int | str]:
        out: list[int | str] = []
        for channel in self.channel_ids:
            out.append(int(channel) if channel.isdigit() else channel)
        return out

    @property
    def source_path(self) -> str:
        return self.provenance.source_path

    @property
    def I_bgsub(self) -> np.ndarray:
        """Stored-reference-subtracted signal, only when a reference exists."""
        return self.signal - self.ref[None, :]

    def to_frame(self) -> pd.DataFrame:
        """Return a wide, lossless table for interchange and inspection."""
        legacy_paqxos = self.provenance.adapter == "paqxos_rtf_legacy"
        data: dict[str, Any] = {"frame": np.arange(self.signal.shape[0])}
        if self.started_at is not None:
            data["time_iso"] = [
                (self.started_at + timedelta(minutes=float(m))).strftime(_TIME_FMT)
                for m in self.time_min
            ]
        data["t_min" if legacy_paqxos else "time_min"] = self.time_min
        for name, values in self.acquisition.items():
            data[name if legacy_paqxos and name == "copt" else f"acq_{name}"] = values
        for index, channel in enumerate(self.channel_ids):
            prefix = "I_ch" if legacy_paqxos else "signal_"
            data[f"{prefix}{channel}"] = self.signal[:, index]
        if self.stored_reference is not None and not legacy_paqxos:
            for index, channel in enumerate(self.channel_ids):
                data[f"ref_{channel}"] = np.repeat(
                    self.stored_reference[index], self.signal.shape[0]
                )
        return pd.DataFrame(data)

    def write_csv(self, csv_path: Path | str) -> Path:
        """Write a neutral CSV mirror and JSON provenance sidecar."""
        csv_path = Path(csv_path)
        self.to_frame().to_csv(csv_path, index=False)
        meta = {
            "schema_version": 1,
            "measurement_name": self.measurement_name,
            "run_kind": self.run_kind,
            "started_at": self.started_at.strftime(_TIME_FMT) if self.started_at else None,
            "channel_ids": list(self.channel_ids),
            "stored_reference": (
                self.stored_reference.tolist() if self.stored_reference is not None else None
            ),
            "acquisition_variables": sorted(self.acquisition),
            "provenance": {
                "run_id": self.provenance.run_id,
                "source_path": self.provenance.source_path,
                "adapter": self.provenance.adapter,
                "sample_id": self.provenance.sample_id,
                "independent_unit_id": self.provenance.independent_unit_id,
                "technical_replicate": self.provenance.technical_replicate,
                "instrument_id": self.provenance.instrument_id,
                "metadata": dict(self.provenance.metadata),
            },
            "flags": self.flags,
        }
        if self.provenance.adapter == "paqxos_rtf_legacy":
            meta.update({
                "source_path": self.provenance.source_path,
                "t0": self.started_at.strftime(_TIME_FMT) if self.started_at else None,
                "channels": self.channels,
                "ref": self.ref.tolist(),
            })
        csv_path.with_name(csv_path.stem + "_meta.json").write_text(json.dumps(meta, indent=2))
        return csv_path
