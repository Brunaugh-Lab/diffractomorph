"""Calibration-standard inputs for the Mie kernel build.

The kernel is calibrated from two standards, each supplying a *pair*:

- a measured **channel-intensity** pattern (the ``.rtf`` export, read via the
  ingest layer), and
- a **number (q0) size distribution** (cumulative undersize), preferred as a
  **CSV**, with the instrument's q0-distribution PDF as a fallback.

File-naming convention (discovered automatically):

    data_<sample>_<type>.<ext>

    <sample>  short tag — the NIST/glass-bead standard and the drug, e.g.
              ``NIST`` (or ``glass``/``bead``) and ``CFZ``.
    <type>    ``intensity`` (channel data) or the distribution type ``q0`` /
              ``q1`` / ``q3``.
    <ext>     ``rtf`` for intensity; ``csv`` (preferred) or ``pdf`` for the
              distribution.

Examples in a QC folder::

    data_NIST_intensity.rtf      data_NIST_q0.csv
    data_CFZ_intensity.rtf       data_CFZ_q0.csv

The kernel needs the **number (q0)** distribution. q1 (length) and q3 (volume)
are converted to number (n ∝ qr / dᵣ); q0 is used directly. The CSV is two
columns — size (µm) and cumulative undersize (%) — at the class boundaries.
"""
from __future__ import annotations

import csv
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from diffractomorph_pipeline import ingest

# Sympatec R3 31-class grid (µm): bin geomeans (sizes) and class upper edges.
GRID = np.array([
    0.67, 0.99, 1.20, 1.40, 1.64, 1.99, 2.39, 2.84, 3.39, 3.99, 4.64, 5.48,
    6.71, 8.22, 9.72, 11.46, 13.69, 16.43, 19.44, 22.91, 27.39, 32.86, 39.34,
    46.83, 55.78, 66.73, 79.69, 94.66, 112.56, 134.47, 160.39,
])
EDGES = np.array([
    0.90, 1.10, 1.30, 1.50, 1.80, 2.20, 2.60, 3.10, 3.70, 4.30, 5.00, 6.00,
    7.50, 9.00, 10.50, 12.50, 15.00, 18.00, 21.00, 25.00, 30.00, 36.00, 43.00,
    51.00, 61.00, 73.00, 87.00, 103.00, 123.00, 147.00, 175.00,
])
N_CLASSES = len(GRID)
NIST_ALIASES = ("nist", "glass", "bead")
_DIST_R = {"q0": 0, "q1": 1, "q2": 2, "q3": 3}

# Distribution type (q0 number, q1 length, q2 area, q3 volume) is normally declared by
# the PAQXOS export — via the cumulative-column header ("Q3 / %", "Q₀ / %") or the
# filename ("..._q3.csv") — so the reader detects it rather than assuming one.
_SUBSCRIPT = {"₀": "0", "₁": "1", "₂": "2", "₃": "3"}


def _dist_from_label(label) -> str | None:
    """Distribution type ('q0'..'q3') from a PAQXOS cumulative-column header, else None."""
    s = "".join(_SUBSCRIPT.get(c, c) for c in str(label))
    m = re.search(r"Q\s*([0-3])", s, re.IGNORECASE)
    return f"q{m.group(1)}" if m else None


def _dist_from_name(path) -> str | None:
    """Distribution type from a filename token (e.g. '..._q3.csv'), else None."""
    m = re.search(r"(?:^|[ _-])q([0-3])(?=[ _.-]|$)", Path(path).name, re.IGNORECASE)
    return f"q{m.group(1)}" if m else None

_FNAME = re.compile(
    r"^data_(?P<sample>.+?)_(?P<type>intensity|q0|q1|q3)\.(?P<ext>rtf|csv|pdf)$",
    re.IGNORECASE,
)


# ── Intensity (from the .rtf, via ingest) ────────────────────────────────────

def read_qc_intensity(rtf_path: Path | str) -> np.ndarray:
    """Mean measured 31-channel QC pattern (averaged over the run's frames)."""
    run = ingest.extract_run(rtf_path)
    return run.I.mean(axis=0)


# ── Number PSD (from CSV or PDF) ─────────────────────────────────────────────

def _cumulative_to_number(size_um, cum_pct, dist_type, grid, edges):
    """Cumulative undersize (%) at boundaries → number fraction on the grid.

    Differences the cumulative onto the class edges to get the per-class amount of
    distribution type ``dist_type``, converts that to number (n ∝ amount / dᵣ),
    and normalizes.
    """
    size_um = np.asarray(size_um, dtype=float)
    cum_pct = np.asarray(cum_pct, dtype=float)
    order = np.argsort(size_um)
    size_um, cum_pct = size_um[order], cum_pct[order]
    # Cumulative undersize evaluated at each class upper edge (0 below data, 100 above).
    Qe = np.interp(edges, size_um, cum_pct, left=0.0, right=100.0)
    Qe = np.maximum.accumulate(np.clip(Qe, 0.0, 100.0))   # enforce monotonic
    amount = np.empty(len(edges))
    amount[0] = Qe[0]
    amount[1:] = np.diff(Qe)
    amount = np.clip(amount, 0.0, None)
    r = _DIST_R[dist_type.lower()]
    number = amount / (grid ** r) if r else amount
    s = number.sum()
    return number / s if s else number


def _read_csv_cumulative(path):
    """Read ``(size_um, cumulative_pct, dist_type)`` from a CSV.

    Tolerant of both the simple ``size_um, cumulative_pct`` form and the native PAQXOS
    export (BOM + metadata/header lines, then a table whose first two columns are
    ``xo / µm`` and ``Qn / %``). Any row whose first two cells parse as floats is taken as
    ``(boundary, cumulative)``. ``dist_type`` is auto-detected from the cumulative column's
    header label (``Qn / %``), or ``None`` if the CSV carries no such header.
    """
    size, cum = [], []
    dist_type = None
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.reader(f):
            if len(r) < 2:
                continue
            try:
                x, q = float(r[0]), float(r[1])
            except ValueError:
                if dist_type is None:              # header row → detect from the cumulative column
                    dist_type = _dist_from_label(r[1])
                continue
            size.append(x)
            cum.append(q)
    if not size:
        raise ValueError(f"No numeric (size, cumulative) rows found in {path}")
    return size, cum, dist_type


def _read_rtf_cumulative(path):
    """Read ``(size_um, cumulative_pct, dist_type)`` from a PAQXOS q-dist RTF export.

    The RTF carries the same ``CUMULATIVE DISTRIBUTION`` table as the CSV/PDF, laid out in two
    ``xo / µm | Qn / %`` column-pairs (fine then coarse); the later columns are the density
    (``q0 lg``) and are ignored. dist_type is detected from the ``Qn / %`` header.
    """
    from striprtf.striprtf import rtf_to_text
    text = rtf_to_text(Path(path).read_text())
    size, cum, dist_type, cum_cols, in_table = [], [], None, None, False
    for line in text.splitlines():
        if "CUMULATIVE DISTRIBUTION" in line.upper():
            in_table = True
            continue
        if not in_table:
            continue
        fields = line.split("\t")
        if cum_cols is None:                          # header: locate the 'Qn / %' columns
            cols = [i for i, f in enumerate(fields) if _dist_from_label(f)]
            if cols:
                cum_cols = cols
                dist_type = _dist_from_label(fields[cols[0]])
            continue
        for c in cum_cols:                            # each cumulative col is preceded by its size col
            if c >= 1 and c < len(fields):
                try:
                    x, q = float(fields[c - 1].strip()), float(fields[c].strip())
                except ValueError:
                    continue
                if 0.0 <= q <= 100.0:
                    size.append(x); cum.append(q)
    if not size:
        raise ValueError(f"No cumulative (size, Q) rows found in {path}")
    order = np.argsort(size)
    return list(np.asarray(size)[order]), list(np.asarray(cum)[order]), dist_type


def _read_pdf_cumulative(path):
    """Average cumulative undersize at class boundaries from a PAQXOS q-dist PDF."""
    import pdfplumber

    frames = []
    with pdfplumber.open(path) as pdf:
        for pg in pdf.pages:
            cum = {}
            for line in (pg.extract_text() or "").splitlines():
                nums = re.findall(r"\d+\.\d+", line)
                if len(nums) >= 2:
                    x = float(nums[0])
                    hit = np.where(np.abs(EDGES - x) < 1e-6)[0]
                    if hit.size:
                        cum[float(EDGES[hit[0]])] = float(nums[1])
            if len(cum) >= 14:
                frames.append([cum.get(float(e), np.nan) for e in EDGES])
    if not frames:
        raise ValueError(f"No distribution frames parsed from {path}")
    with warnings.catch_warnings():           # edges past the data are all-NaN — fine
        warnings.simplefilter("ignore", RuntimeWarning)
        Q = np.nanmean(np.array(frames), axis=0)
    mask = ~np.isnan(Q)
    return EDGES[mask].tolist(), Q[mask].tolist()


def read_number_psd(path: Path | str, dist_type: str | None = None,
                    grid=GRID, edges=EDGES) -> np.ndarray:
    """Read a size-distribution export → number (q0) fraction per grid bin.

    ``path`` may be:

    - a **directory** of per-frame PAQXOS CSVs — each frame is converted and averaged;
    - a single **CSV** (PAQXOS export or simple ``size_um, cumulative_pct``); or
    - a PAQXOS q-dist **PDF** (fallback).

    The source distribution type is resolved per file by precedence: an explicit
    ``dist_type`` argument > the ``Qn / %`` header label > the filename token (``_q3``) >
    ``q3`` (the usual volume export). It is then converted to number (n ∝ amount / dᵣ).
    """
    path = Path(path)

    def _to_number(size, cum, header, src):
        dt = dist_type or header or _dist_from_name(src) or "q3"
        return _cumulative_to_number(size, cum, dt, grid, edges)

    if path.is_dir():
        csvs = sorted(path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No per-frame CSVs in {path}")
        return np.mean([_to_number(*_read_csv_cumulative(c), c) for c in csvs], axis=0)
    if path.suffix.lower() == ".csv":
        size, cum, header = _read_csv_cumulative(path)
    elif path.suffix.lower() == ".rtf":
        size, cum, header = _read_rtf_cumulative(path)
    elif path.suffix.lower() == ".pdf":
        (size, cum), header = _read_pdf_cumulative(path), None
    else:
        raise ValueError(f"Unsupported PSD format: {path.suffix} (use a folder, .csv, .rtf, or .pdf)")
    return _to_number(size, cum, header, path)


# ── QC file discovery ────────────────────────────────────────────────────────

@dataclass
class StandardFiles:
    intensity: Path
    psd: Path
    psd_type: str           # "q0" | "q1" | "q3"


@dataclass
class QCFiles:
    nist: StandardFiles
    drug: StandardFiles | None      # None when RI comes from the library (weekly rebuild)


def _parse(name):
    m = _FNAME.match(name)
    return (m.group("sample").lower(), m.group("type").lower(), m.group("ext").lower()) if m else None


def discover_qc_files(qc_dir: Path | str, drug: str, need_drug: bool = True,
                      nist_aliases=NIST_ALIASES) -> QCFiles:
    """Find the NIST (+ drug, if ``need_drug``) ``data_<sample>_<type>`` QC files.

    Pairs each standard's ``intensity`` (.rtf) with its distribution file
    (q0 preferred; .csv preferred over .pdf). NIST is always required (geometry);
    the drug standard is only required when fitting refractive index. Raises a
    clear error if a required standard is incomplete.
    """
    qc_dir = Path(qc_dir)
    by_sample: dict[str, dict] = {}
    for p in sorted(qc_dir.iterdir()):
        parsed = _parse(p.name)
        if not parsed:
            continue
        sample, typ, ext = parsed
        by_sample.setdefault(sample, {}).setdefault(typ, []).append(p)

    def pick_standard(sample_key, role):
        files = by_sample[sample_key]
        if "intensity" not in files:
            raise FileNotFoundError(
                f"{role} standard '{sample_key}': missing data_{sample_key}_intensity.rtf in {qc_dir}")
        intensity = next((f for f in files["intensity"] if f.suffix.lower() == ".rtf"), None)
        if intensity is None:
            raise FileNotFoundError(f"{role} standard '{sample_key}': intensity file must be .rtf")
        for dist in ("q0", "q1", "q3"):
            if dist in files:
                # prefer CSV over PDF
                csvs = [f for f in files[dist] if f.suffix.lower() == ".csv"]
                pdfs = [f for f in files[dist] if f.suffix.lower() == ".pdf"]
                chosen = (csvs or pdfs)[0]
                return StandardFiles(intensity=intensity, psd=chosen, psd_type=dist)
        raise FileNotFoundError(
            f"{role} standard '{sample_key}': no q0/q1/q3 distribution (.csv or .pdf) in {qc_dir}")

    samples = list(by_sample)
    nist_key = next((s for s in samples if any(a in s for a in nist_aliases)), None)
    drug_key = next((s for s in samples if drug.lower() in s or s in drug.lower()), None)
    if nist_key is None:
        raise FileNotFoundError(
            f"No NIST/glass-bead standard found in {qc_dir} "
            f"(expected a sample tag containing one of {nist_aliases}). Found: {samples}")
    drug_files = None
    if need_drug:
        if drug_key is None:
            raise FileNotFoundError(
                f"No '{drug}' standard found in {qc_dir} (needed to fit refractive "
                f"index). Found samples: {samples}")
        drug_files = pick_standard(drug_key, drug)
    return QCFiles(nist=pick_standard(nist_key, "NIST"), drug=drug_files)
