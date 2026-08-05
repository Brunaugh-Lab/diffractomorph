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
2. **``Ref`` is static within a file** — validated; a varying ``Ref`` flags a
   malformed/concatenated export.
3. **Frames are listed newest-first** — the parser sorts by timestamp ascending
   and records ``reverse_order_detected``; document order is never trusted.

Spec module path was ``src/dissolution_optics/io/paqxos.py``; reconciled to this
package's flat layout as ``diffractomorph_pipeline/ingest.py``.
"""
from __future__ import annotations

import re
import warnings
from datetime import datetime
from pathlib import Path

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
                   "copt": float("nan"), "channels": {}}
        elif line.startswith("Optical Concentration:") and cur is not None:
            m = re.search(rf"({_NUM})", line.split(":", 1)[1])
            cur["copt"] = float(m.group(1)) if m else float("nan")
        elif cur is not None:
            m = _CHANNEL_ROW.match(line)
            if m:
                cur["channels"][int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    if cur is not None and cur["channels"]:
        frames.append(cur)
    return frames


# ── Public API (§11) ─────────────────────────────────────────────────────────

def extract_run(
    path: Path | str,
    run_kind: str | None = None,
    ref_static_tol: float = 1e-6,
    emit_csv: bool = False,
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
    """
    from striprtf.striprtf import rtf_to_text

    path = Path(path)
    text = rtf_to_text(path.read_text())
    frames = _parse_frames(text)

    # Canonical channel set = the channel list of the most complete frame.
    if not frames:
        raise ValueError(f"No frames parsed from {path}")
    canonical = sorted(max(frames, key=lambda fr: len(fr["channels"]))["channels"].keys())
    n_ch = len(canonical)

    # Drop malformed frames (wrong channel count); fail only if <2 remain (§8).
    good = [fr for fr in frames if sorted(fr["channels"].keys()) == canonical]
    dropped = len(frames) - len(good)
    if len(good) < 2 and len(frames) >= 2:
        # too many dropped to trust — but a genuine single-frame file is allowed below
        good = [fr for fr in frames if len(fr["channels"]) == n_ch] or good
    if len(good) < 1:
        raise ValueError(f"{path}: no valid frames after channel-count filtering")

    # Document-order timestamps → detect reverse ordering, then sort ascending (§2.3).
    times_doc = [datetime.strptime(fr["time"], _TIME_FMT) for fr in good]
    reverse_order_detected = len(times_doc) > 1 and times_doc[0] > times_doc[-1]
    order = np.argsort(times_doc)
    good = [good[i] for i in order]

    # Vectorize.
    I = np.array([[fr["channels"][c][1] for c in canonical] for fr in good], dtype=float)
    ref_all = np.array([[fr["channels"][c][0] for c in canonical] for fr in good], dtype=float)
    copt = np.array([fr["copt"] for fr in good], dtype=float)
    times = [datetime.strptime(fr["time"], _TIME_FMT) for fr in good]
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
