"""Shared utilities for diffractomorph_pipeline."""

import re
import sys
from pathlib import Path

# ── Column / metadata naming ─────────────────────────────────────────────────
# Tidy long-format columns produced by ingest. One row per (file, frame, channel).
TIDY_COLUMNS = [
    "sample",        # sample / formulation name parsed from filename
    "kind",          # "measurement" or "blank"
    "ph",            # buffer pH parsed from filename (float or NaN)
    "day",           # study day (int or NaN)
    "rep",           # replicate number (int or NaN)
    "frame",         # frame index within the run (0-based, chronological)
    "t_min",         # elapsed minutes from frame 0
    "copt",          # optical concentration (%) for the frame
    "channel",       # Sympatec detector ring index (1..31)
    "intensity",     # measured intensity for that channel/frame
]


def guess_from_filename(fname: str) -> dict:
    """Extract sample metadata from a Sympatec/PAQXOS export filename.

    Lab convention (see the clofazimine_dissolution study) looks like::

        "CFZ disso pH = 4.5 measurement Day 1 Rep 2.rtf"
        "CFZ disso pH = 4.5 blank Day 1 Rep 3.rtf"
        "CFZ QC pH = 7.rtf"

    Returns a dict with keys: sample, kind, ph, day, rep. Missing fields are
    returned as ``None`` so the caller (or the interactive prompt) can fill them.
    """
    stem = Path(fname).stem

    # kind: blank vs measurement (default to measurement)
    kind = "blank" if re.search(r"\bblank\b", stem, re.IGNORECASE) else "measurement"

    # pH: "pH = 4.5", "pH=4.5", "pH 4.5"
    m = re.search(r"pH\s*=?\s*([\d.]+)", stem, re.IGNORECASE)
    ph = float(m.group(1)) if m else None

    # Day N
    m = re.search(r"\bDay\s*(\d+)", stem, re.IGNORECASE)
    day = int(m.group(1)) if m else None

    # Rep N
    m = re.search(r"\bRep\s*(\d+)", stem, re.IGNORECASE)
    rep = int(m.group(1)) if m else None

    # sample: leading token(s) before "disso"/"QC"/"pH"
    m = re.match(r"^(.*?)\s+(?:disso|QC|pH)\b", stem, re.IGNORECASE)
    sample = m.group(1).strip() if m else stem

    return {"sample": sample, "kind": kind, "ph": ph, "day": day, "rep": rep}


def is_interactive() -> bool:
    """Check whether the current environment supports interactive prompts."""
    if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
        return False
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return False
    except ImportError:
        pass
    return True
