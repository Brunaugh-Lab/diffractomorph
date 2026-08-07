"""Sympatec PAQXOS RTF adapter."""
from __future__ import annotations

from diffractomorph_pipeline import ingest
from diffractomorph_pipeline.model import Run, RunProvenance


class PaqxosRtfReader:
    adapter_id = "paqxos_rtf"

    def read(self, spec) -> Run:
        return self._read(spec)

    def read_with_instrument_profile(self, spec, parameters) -> Run:
        """Read using structural expectations declared by the instrument profile."""
        expected = parameters.get("channel_ids")
        run = self._read(spec, expected_channel_ids=expected)
        if expected is not None:
            run.flags["expected_channel_ids_source"] = "instrument_profile"
        return run

    def _read(self, spec, expected_channel_ids=None) -> Run:
        run = ingest.extract_run(
            spec.source,
            run_kind=spec.run_kind,
            expected_channel_ids=expected_channel_ids,
        )
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
