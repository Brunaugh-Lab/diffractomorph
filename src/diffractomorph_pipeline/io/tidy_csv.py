"""Vendor-neutral wide CSV adapter.

Required columns are ``time_min`` and one or more ``signal_<channel-id>`` columns.
Optional acquisition variables use ``acq_<name>``. Optional static references use
``ref_<channel-id>`` and must be constant down the file.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from diffractomorph_pipeline.model import Run, RunProvenance


class TidyCsvReader:
    adapter_id = "tidy_csv"

    def read(self, spec) -> Run:
        frame = pd.read_csv(spec.source)
        if "time_min" not in frame:
            raise ValueError(f"{spec.source}: missing required column 'time_min'")
        signal_columns = [c for c in frame if c.startswith("signal_")]
        if not signal_columns:
            raise ValueError(f"{spec.source}: expected at least one signal_<channel> column")
        channel_ids = tuple(c.removeprefix("signal_") for c in signal_columns)
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError(f"{spec.source}: duplicate channel identifiers")

        acquisition = {
            c.removeprefix("acq_"): frame[c].to_numpy(float)
            for c in frame if c.startswith("acq_")
        }
        ref_columns = [f"ref_{channel}" for channel in channel_ids]
        present_refs = [c for c in ref_columns if c in frame]
        if present_refs and len(present_refs) != len(ref_columns):
            raise ValueError(f"{spec.source}: stored reference columns must cover every channel")
        stored_reference = None
        ref_static = None
        if present_refs:
            ref_matrix = frame[ref_columns].to_numpy(float)
            ref_static = bool(np.nanmax(np.nanstd(ref_matrix, axis=0)) < 1e-6)
            if not ref_static:
                raise ValueError(f"{spec.source}: stored reference columns are not static")
            stored_reference = ref_matrix[0]

        started_at = None
        value = spec.metadata.get("started_at")
        if value:
            started_at = datetime.fromisoformat(str(value))
        flags = {"n_frames": int(len(frame))}
        if ref_static is not None:
            flags["ref_static"] = ref_static
        return Run(
            signal=frame[signal_columns].to_numpy(float),
            channel_ids=channel_ids,
            time_min=frame["time_min"].to_numpy(float),
            acquisition=acquisition,
            provenance=RunProvenance(
                run_id=spec.run_id,
                source_path=str(spec.source),
                adapter=self.adapter_id,
                sample_id=spec.sample_id,
                independent_unit_id=spec.independent_unit_id,
                technical_replicate=spec.technical_replicate,
                instrument_id=spec.instrument_id,
                metadata=spec.metadata,
            ),
            run_kind=spec.run_kind,
            stored_reference=stored_reference,
            started_at=started_at,
            measurement_name=spec.run_id,
            flags=flags,
        )
