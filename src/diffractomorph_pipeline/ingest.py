"""Module 0 — LD Ingest / Extraction.

Parse one Sympatec PAQXOS ``.rtf`` export (measurement *or* blank) into a typed,
chronologically-sorted :class:`RawRun` and an optional tidy CSV mirror. Nothing
analytical happens here — the job is faithful, lossless extraction with the
per-instrument quirks handled once, in one place.

Per ``LD_Ingest_Extraction_Component_Spec.md`` (pipeline Step 0). This is the
corrected/extended half of the legacy ``diffractomorph_core`` (``parse_paqxos_rtf``
+ ``to_arrays``); the legacy Wasserstein/plateau code is deliberately excluded.

Three instrument quirks the parser respects (spec §2):

1. **Two intensity columns per channel** — ``Ref Value`` and ``Measured Value``.
   ``Measured`` is the per-frame signal; ``Ref`` is the stored reference spectrum.
   Both are captured (``Ref`` feeds optional static-baseline subtraction).
2. **``Ref`` is expected to be static within a file** — variation is retained
   and flagged for review rather than silently changing frame eligibility.
3. **Frames are listed newest-first** — the parser sorts by timestamp ascending
   and records ``reverse_order_detected``; document order is never trusted.

Spec module path was ``src/dissolution_optics/io/paqxos.py``; reconciled to this
package's flat layout as ``diffractomorph_pipeline/ingest.py``.
"""
from __future__ import annotations

import re
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from diffractomorph_pipeline.model import Run, RunProvenance
from diffractomorph_pipeline.utils import guess_from_filename

_TIME_FMT = "%Y-%m-%d %H:%M:%S"
_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_CHANNEL_ROW = re.compile(rf"^(\d+)\s*,\s*({_NUM})\s*,\s*({_NUM})\s*$")


# Backward-compatible public name. The implementation now lives in the neutral
# data-contract module.
RawRun = Run


# ── Parsing ──────────────────────────────────────────────────────────────────

def _parse_frames(text: str) -> list[dict]:
    """Split de-RTF'd text into per-frame dicts in *document order*.

    Each frame: ``name``, ``time`` (str), ``copt`` (float|nan), ``channels``
    (dict ``ch -> (ref, measured)``). A new frame begins at ``Measurement Time:``;
    the preceding ``Measurement Name:`` is attached.
    """
    frames: list[dict] = []
    cur: dict | None = None
    pending_name = ""
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("Measurement Name:"):
            pending_name = line.split(":", 1)[1].strip()
        elif line.startswith("Measurement Time:"):
            if cur is not None and cur["channels"]:
                frames.append(cur)
            cur = {"name": pending_name, "time": line.split(":", 1)[1].strip(),
                   "copt": float("nan"), "channels": {}, "duplicate_channels": set()}
        elif line.startswith("Optical Concentration:") and cur is not None:
            m = re.search(rf"({_NUM})", line.split(":", 1)[1])
            cur["copt"] = float(m.group(1)) if m else float("nan")
        elif cur is not None:
            m = _CHANNEL_ROW.match(line)
            if m:
                channel = int(m.group(1))
                if channel in cur["channels"]:
                    cur["duplicate_channels"].add(channel)
                cur["channels"][channel] = (float(m.group(2)), float(m.group(3)))
    if cur is not None and cur["channels"]:
        frames.append(cur)
    return frames


# ── Public API (§11) ─────────────────────────────────────────────────────────

def extract_run(
    path: Path | str,
    run_kind: str | None = None,
    ref_static_tol: float = 1e-6,
    emit_csv: bool = False,
    expected_channel_ids: Sequence[int | str] | None = None,
) -> RawRun:
    """Parse one PAQXOS RTF into a :class:`RawRun` (spec §4).

    Parameters
    ----------
    path
        Path to one ``.rtf`` export.
    run_kind
        ``"measurement"`` / ``"blank"``; inferred from the filename if ``None``.
    ref_static_tol
        Max per-channel std of ``ref`` across frames for the static check.
    emit_csv
        If true, also write the CSV mirror + meta sidecar next to the source.
    expected_channel_ids
        Detector-channel identifiers declared by the instrument profile. When
        supplied, output columns follow this declared order. When omitted, the
        parser infers the unique most frequently observed exact channel set and
        fails closed if equally supported sets are ambiguous.
    """
    from striprtf.striprtf import rtf_to_text

    path = Path(path)
    text = rtf_to_text(path.read_text())
    frames = _parse_frames(text)

    if not frames:
        raise ValueError(f"No frames parsed from {path}")

    for frame in frames:
        try:
            frame["parsed_time"] = datetime.strptime(frame["time"], _TIME_FMT)
        except ValueError:
            frame["parsed_time"] = None

    if expected_channel_ids is not None:
        try:
            canonical = [int(str(channel)) for channel in expected_channel_ids]
        except ValueError as exc:
            raise ValueError("expected_channel_ids must contain integer identifiers") from exc
        if not canonical:
            raise ValueError("expected_channel_ids must not be empty")
        if len(set(canonical)) != len(canonical):
            raise ValueError("expected_channel_ids must be unique")
        if canonical != sorted(canonical):
            raise ValueError("expected_channel_ids must be in strictly increasing order")
        channel_source = "argument"
    else:
        eligible_sets = [
            tuple(sorted(frame["channels"]))
            for frame in frames
            if not frame["duplicate_channels"] and frame["parsed_time"] is not None
        ]
        if not eligible_sets:
            raise ValueError(f"{path}: no structurally valid frames from which to infer channels")
        counts = Counter(eligible_sets)
        top_count = max(counts.values())
        leaders = [channel_set for channel_set, count in counts.items() if count == top_count]
        if len(leaders) != 1:
            rendered = ", ".join(str(list(channel_set)) for channel_set in sorted(leaders))
            raise ValueError(
                f"{path}: ambiguous detector-channel sets ({rendered}); "
                "declare expected_channel_ids in the instrument profile"
            )
        canonical = list(leaders[0])
        channel_source = "inferred"

    expected_set = set(canonical)
    dropped_reasons: Counter[str] = Counter()
    good = []
    for frame in frames:
        if frame["duplicate_channels"]:
            dropped_reasons["duplicate_channel_ids"] += 1
        elif frame["parsed_time"] is None:
            dropped_reasons["unparseable_timestamp"] += 1
        elif set(frame["channels"]) != expected_set:
            dropped_reasons["channel_set_mismatch"] += 1
        else:
            good.append(frame)
    dropped = sum(dropped_reasons.values())
    if not good:
        detail = ", ".join(f"{key}={value}" for key, value in sorted(dropped_reasons.items()))
        raise ValueError(f"{path}: no structurally valid frames ({detail})")

    # Document-order timestamps → detect reverse ordering, then sort ascending (§2.3).
    times_doc = [fr["parsed_time"] for fr in good]
    reverse_order_detected = len(times_doc) > 1 and times_doc[0] > times_doc[-1]
    order = np.argsort(times_doc)
    good = [good[i] for i in order]

    # Vectorize.
    I = np.array([[fr["channels"][c][1] for c in canonical] for fr in good], dtype=float)
    ref_all = np.array([[fr["channels"][c][0] for c in canonical] for fr in good], dtype=float)
    copt = np.array([fr["copt"] for fr in good], dtype=float)
    times = [fr["parsed_time"] for fr in good]
    t0 = times[0]
    t_min = np.array([(t - t0).total_seconds() / 60.0 for t in times])

    # Validate static ref (§4.5); keep the chronological frame-0 ref regardless.
    ref_std = ref_all.std(axis=0)
    ref_static = bool(ref_std.max() < ref_static_tol)
    ref = ref_all[0]
    if not ref_static:
        warnings.warn(f"{path.name}: Ref not static (max std {ref_std.max():.2e}); flagging run.")

    max_gap_min = float(np.diff(t_min).max()) if t_min.size > 1 else 0.0
    copt_nan = int(np.isnan(copt).sum())
    if copt_nan:
        warnings.warn(f"{path.name}: {copt_nan} frame(s) missing Optical Concentration.")

    # run_kind: supplied, else infer from filename (§3).
    inferred = run_kind is None
    if inferred:
        run_kind = guess_from_filename(path.name)["kind"]

    flags = {
        "ref_static": ref_static,
        "reverse_order_detected": bool(reverse_order_detected),
        "max_gap_min": max_gap_min,
        "n_frames": int(I.shape[0]),
        "dropped_frames": int(dropped),
        "dropped_frame_reasons": dict(sorted(dropped_reasons.items())),
        "expected_channel_ids_source": channel_source,
        "copt_nan": copt_nan,
        "run_kind_inferred": inferred,
    }

    guessed = guess_from_filename(path.name)
    run = Run(
        signal=I,
        channel_ids=tuple(str(channel) for channel in canonical),
        time_min=t_min,
        acquisition={"copt": copt},
        stored_reference=ref,
        started_at=t0,
        measurement_name=good[0]["name"],
        run_kind=run_kind,
        provenance=RunProvenance(
            run_id=path.stem,
            source_path=str(path),
            adapter="paqxos_rtf_legacy",
            sample_id=str(guessed.get("sample") or "unspecified"),
            independent_unit_id=None,
            metadata={"filename_metadata": guessed},
        ),
        flags=flags,
    )
    if emit_csv:
        run.write_csv(path.with_suffix(".csv"))
    return run


def load_run(path: Path | str) -> RawRun:
    """Convenience alias for :func:`extract_run` (RawRun only, no CSV)."""
    return extract_run(path)
