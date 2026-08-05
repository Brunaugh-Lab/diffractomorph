"""Adapter protocol and explicit reader registry."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from diffractomorph_pipeline.model import Run


class RunSpecLike(Protocol):
    run_id: str
    source: Path
    adapter: str
    run_kind: str
    sample_id: str
    independent_unit_id: str | None
    technical_replicate: str | None
    instrument_id: str | None
    metadata: dict


class RunReader(Protocol):
    """Read one declared source into the neutral run contract."""

    adapter_id: str

    def read(self, spec: RunSpecLike) -> Run: ...


_READERS: dict[str, RunReader] = {}
_BUILTINS_LOADED = False


def register_reader(reader: RunReader, *, replace: bool = False) -> None:
    _ensure_builtin_readers()
    adapter_id = str(reader.adapter_id).strip()
    if not adapter_id:
        raise ValueError("reader adapter_id must be non-empty")
    if adapter_id in _READERS and not replace:
        raise ValueError(f"reader already registered: {adapter_id}")
    _READERS[adapter_id] = reader


def _ensure_builtin_readers() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from .paqxos_rtf import PaqxosRtfReader
    from .tidy_csv import TidyCsvReader

    for reader in (PaqxosRtfReader(), TidyCsvReader()):
        _READERS[reader.adapter_id] = reader
    _BUILTINS_LOADED = True


def get_reader(adapter_id: str) -> RunReader:
    """Return an explicitly selected adapter; no instrument is a silent default."""
    _ensure_builtin_readers()
    try:
        return _READERS[adapter_id]
    except KeyError as exc:
        available = ", ".join(sorted(_READERS))
        raise KeyError(f"unknown adapter {adapter_id!r}; available: {available}") from exc
