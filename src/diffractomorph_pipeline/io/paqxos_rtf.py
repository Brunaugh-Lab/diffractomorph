"""Sympatec PAQXOS RTF adapter."""
from __future__ import annotations

from diffractomorph_pipeline import ingest
from diffractomorph_pipeline.model import Run, RunProvenance


class PaqxosRtfReader:
    adapter_id = "paqxos_rtf"

    def read(self, spec) -> Run:
        run = ingest.extract_run(spec.source, run_kind=spec.run_kind)
        run.provenance = RunProvenance(
            run_id=spec.run_id,
            source_path=str(spec.source),
            adapter=self.adapter_id,
            sample_id=spec.sample_id,
            independent_unit_id=spec.independent_unit_id,
            technical_replicate=spec.technical_replicate,
            instrument_id=spec.instrument_id,
            metadata=spec.metadata,
        )
        return run
