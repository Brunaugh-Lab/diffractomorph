"""Discovery and design bookkeeping for the Arm B in-medium Tween 80 (micelle) study.

Layout: ``<condition>/<YYYYMMDD>/{Raw Data, UV Data, q3 Data, q0 Data, QC}``.

**The replication unit is the independent suspension prep**, and a prep is identified by its
**injection date from the volume log — not by the folder name**. The two differ: the ``20260702``
folder holds replicate 1 injected on 2026-07-02 and replicates 2–3 injected on 2026-07-07 off a
*fresh* suspension, each with its own QC. Resolving preps properly therefore gives **n = 4
independent replicates per condition**, with unequal technical repeats behind them (1, 2, 3, 3):

    0.5x CMC   07-02 (1 run) · 07-07 (2) · 07-22 (3) · 07-23 (3)
    1.0x CMC   07-02 (1 run) · 07-07 (2) · 07-22 (3) · 07-24 (3)
    10x CMC    07-02 (1 run) · 07-07 (2) · 07-23 (3) · 07-24 (3)

The ``Rep N`` runs sharing a prep are repeat cuvette measurements, not independent n. They are
averaged to one value per prep; a condition is n=4, never n=9. Because each prep is one
independent observation whatever number of repeats supports it, prep means are combined
*unweighted* — weighting by run count would assume no prep-to-prep variance, which is the
pseudo-replication it exists to avoid. The 07-02 prep rests on a single run, so its mean is the
least precisely measured of the four.

One prep serves every condition injected from it: the suspension QC exports are byte-identical
across the condition folders of a shared prep (and where a QC workbook was saved twice under
different names, both read to the same value). So each pair of conditions shares three preps
(07-02, 07-07, and one later date), which :func:`condition_pairs_sharing_preps` reports — making a
prep-paired contrast available as a variance-reduced companion to the across-prep comparison.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import openpyxl
import pandas as pd

from diffractomorph_pipeline.config import data_root

ARM_B_REL = Path("disso_experiments/dissolution_media_diagnostic/micelle_effects_tween80_arm_b")
# Canonical condition label (the folder name) → nominal in-cuvette Tween level, × CMC.
CONDITIONS = {"0.5x CMC": 0.5, "1.0x CMC": 1.0, "10x CMC": 10.0}
REPS = (1, 2, 3)
VOLUME_WORKBOOK = "CFZ volume added.xlsx"
# Arm B rows in the shared volume log. Whitespace around the level and "Day N" is irregular, and
# the same log carries other studies (Arm A concentrations, the Copt ladder) that must not match.
_LABEL = re.compile(
    r"4\.5\s+0\.02%\s*w/v\s*Tween\s*\(\s*(?P<level>[0-9.]+x)\s*(?:-\s*Day\s*(?P<day>\d+)\s*)?\)",
    re.IGNORECASE)
_LEVEL_TO_CONDITION = {level.split()[0]: level for level in CONDITIONS}


def default_study_root() -> Path:
    return data_root() / ARM_B_REL


def _one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise FileNotFoundError(f"{label}: expected one match, found {len(paths)}: "
                                f"{[p.name for p in paths]}")
    return paths[0]


def _reps_for_qc_name(name: str) -> set[int]:
    """Which replicates a QC export covers, from its filename.

    The 20260702 cell was QC'd twice — ``… Rep 1`` on the preparation date and
    ``… Reps 2 and 3`` on 20260707 — so QC provenance there is per-replicate. A name carrying
    no replicate token covers the whole cell.
    """
    found = set()
    for match in re.finditer(r"[Rr]eps?[ _-]*([123])(?:[ _-]*and[ _-]*([123]))?", name):
        found.update(int(g) for g in match.groups() if g)
    return found or set(REPS)


def _qc_for_rep(qc_dir: Path, pattern: str, rep: int, label: str) -> Path:
    """The QC export in ``qc_dir`` covering ``rep`` (single-file cells cover every replicate)."""
    candidates = [p for p in sorted(qc_dir.glob(pattern)) if rep in _reps_for_qc_name(p.name)]
    return _one(candidates, f"{label} rep {rep} QC ({pattern})")


def _qc_psd_for_rep(qc_dir: Path, rep: int, label: str) -> Path:
    """The starting-suspension ``q0`` frame folder covering ``rep``.

    Folder naming varies (``q0 Data``, ``q0_Data``, ``q0 Data Rep 1``, ``q0 Data - Reps 2 and 3``);
    the replicate token routes it exactly as the QC files are routed.
    """
    candidates = [p for p in sorted(qc_dir.iterdir())
                  if p.is_dir() and p.name.lower().startswith("q0")
                  and rep in _reps_for_qc_name(p.name)]
    return _one(candidates, f"{label} rep {rep} QC q0 folder")


def _uv_for_rep(uv_dir: Path, rep: int, label: str) -> Path:
    return _one([p for p in sorted(uv_dir.glob("*.xlsx"))
                 if re.search(rf"rep[ _-]*{rep}\b", p.stem, re.IGNORECASE)],
                f"{label} rep {rep} UV plate")


def _day_token(stem: str) -> int:
    """The ``dayN`` ordinal in a UV filename; day 1 files carry no token."""
    m = re.search(r"day[ _-]*([0-9]+)", stem, re.IGNORECASE)
    return int(m.group(1)) if m else 1


def _volume_uL(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and (m := re.search(r"([0-9]+(?:\.[0-9]+)?)\s*u[lL]", value)):
        return float(m.group(1))
    return None


def read_run_volumes(path: Path | str) -> pd.DataFrame:
    """Select the Arm B rows from the shared injection-volume log.

    One row per (condition, folder_index, rep) carrying the injected volume and the
    ``injection_date`` the log records — which is what identifies the suspension prep.
    ``folder_index`` is the ``Day N`` the label carries (absent → 1) and addresses the on-disk
    folder, so it is *not* the prep: the day-1 folder spans two injection dates.
    """
    path = Path(path)
    rows = openpyxl.load_workbook(path, data_only=True).worksheets[0].iter_rows(values_only=True)
    date = condition = folder_index = None
    records = []
    for row in rows:
        if not row:
            continue
        if isinstance(row[0], dt.datetime):
            date = row[0].date()
        label = row[1] if len(row) > 1 else None
        if isinstance(label, str):
            if match := _LABEL.search(label):
                condition = _LEVEL_TO_CONDITION.get(match.group("level"))
                folder_index = int(match.group("day") or 1)
            elif label.strip():
                condition = folder_index = None     # a different study's block — stop collecting
        rep = row[2] if len(row) > 2 else None
        volume = _volume_uL(row[3] if len(row) > 3 else None)
        if condition and isinstance(rep, (int, float)) and volume is not None:
            records.append({"condition": condition, "folder_index": folder_index, "rep": int(rep),
                            "injected_uL": volume, "injection_date": date.strftime("%Y-%m-%d")})
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(f"{path.name}: no Arm B injection-volume rows found")
    keys = ["condition", "folder_index", "rep"]
    if frame.duplicated(keys).any():
        dupes = frame.loc[frame.duplicated(keys, keep=False), keys].to_dict("records")
        raise ValueError(f"{path.name}: duplicate Arm B volume rows: {dupes}")
    expected = {(c, f, r) for c in CONDITIONS for f in (1, 2, 3) for r in REPS}
    observed = set(frame[keys].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError(f"{path.name}: incomplete Arm B volume design; "
                         f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}")
    return frame.sort_values(keys).reset_index(drop=True)


def discover_runs(study_root: Path | str | None = None) -> pd.DataFrame:
    """Every Arm B run with its measurement, UV, q3 and QC sources; validates the design.

    One row per cuvette run. ``prep`` is the **injection date** from the volume log and is the
    independent replicate; ``rep`` is the technical repeat within it. ``folder`` is the on-disk
    date folder, which is *not* the prep — the day-1 folder spans two injection dates. Raises if
    any condition lacks three folders or three repeats per folder, or if a UV filename's ``dayN``
    token disagrees with the folder ordinal.
    """
    root = Path(study_root) if study_root is not None else default_study_root()
    records = []
    for condition, xcmc in CONDITIONS.items():
        cell = root / condition
        if not cell.is_dir():
            raise FileNotFoundError(f"Arm B: missing condition folder {cell}")
        dates = sorted(d.name for d in cell.iterdir() if d.is_dir() and re.fullmatch(r"\d{8}", d.name))
        if len(dates) != 3:
            raise ValueError(f"Arm B {condition}: expected 3 date folders, found {dates}")
        for folder_index, date in enumerate(dates, start=1):
            folder = cell / date
            label = f"Arm B {condition} {date}"
            for rep in REPS:
                uv = _uv_for_rep(folder / "UV Data", rep, label)
                if (token := _day_token(uv.stem)) != folder_index:
                    raise ValueError(
                        f"{label} rep {rep}: UV filename says day {token} but {date} is folder "
                        f"{folder_index} for {condition} ({uv.name})")
                q3 = folder / "q3 Data" / f"Rep {rep}"
                if not q3.is_dir():
                    raise FileNotFoundError(f"{label} rep {rep}: missing q3 frame folder {q3}")
                records.append({
                    "condition": condition, "xcmc": xcmc, "folder": date,
                    "folder_index": folder_index, "rep": rep,
                    "run_id": f"{condition.replace(' ', '')}_{date}_rep{rep}",
                    "rtf": _one(sorted((folder / "Raw Data").glob(f"*Measurement Rep {rep}.rtf")),
                                f"{label} rep {rep} measurement RTF"),
                    "uv_file": uv,
                    "q3_dir": q3,
                    "qc_rtf": _qc_for_rep(folder / "QC", "*.rtf", rep, label),
                    "qc_xlsx": _qc_for_rep(folder / "QC", "*.xlsx", rep, label),
                    "qc_psd_dir": _qc_psd_for_rep(folder / "QC", rep, label),
                })
    frame = pd.DataFrame(records)
    expected = len(CONDITIONS) * 3 * len(REPS)
    if len(frame) != expected:
        raise ValueError(f"Arm B: expected {expected} runs, discovered {len(frame)}")

    # The prep is the injection date, not the folder — attach it from the volume log.
    volumes = read_run_volumes(root / VOLUME_WORKBOOK)
    frame = frame.merge(volumes, on=["condition", "folder_index", "rep"], how="left", validate="1:1")
    if frame["injection_date"].isna().any():
        missing = frame.loc[frame["injection_date"].isna(), ["condition", "folder", "rep"]]
        raise ValueError(f"runs with no injection-volume row:\n{missing.to_string(index=False)}")
    frame["prep"] = frame["injection_date"].str.replace("-", "", regex=False)
    order = {c: {p: i for i, p in enumerate(sorted(g["prep"].unique()), start=1)}
             for c, g in frame.groupby("condition")}
    frame["prep_index"] = [order[r.condition][r.prep] for r in frame.itertuples()]
    return frame.sort_values(["xcmc", "prep", "rep"]).reset_index(drop=True)


def replication_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Per condition: the independent preps behind it and the technical repeats within each.

    ``n_preps`` is the condition's true n (preps = injection dates). ``n_runs`` counts cuvette
    runs and is deliberately *not* an n — it counts technical repeats too. ``n_folders`` is the
    on-disk folder count, which under-counts preps because the day-1 folder spans two.
    """
    rows = []
    for (condition, xcmc), group in runs.groupby(["condition", "xcmc"]):
        preps = sorted(group["prep"].unique())
        per_prep = group.groupby("prep").size()
        rows.append({"condition": condition, "xcmc": xcmc,
                     "n_preps": len(preps), "preps": " | ".join(preps),
                     "reps_per_prep": " | ".join(str(int(n)) for n in per_prep.loc[preps]),
                     "n_folders": group["folder"].nunique(), "n_runs": len(group)})
    return pd.DataFrame(rows).sort_values("xcmc").reset_index(drop=True)


def condition_pairs_sharing_preps(runs: pd.DataFrame) -> pd.DataFrame:
    """For each condition pair, the preps both were spiked from.

    A shared prep means the two conditions came off one suspension, so a within-prep contrast
    removes prep-to-prep variation. Conditions are not run on a common set of preps, so this is
    an optional companion to the primary across-prep comparison, not a replacement for it.
    """
    preps_for = {c: set(g["prep"]) for c, g in runs.groupby("condition")}
    names = list(CONDITIONS)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sorted(preps_for.get(a, set()) & preps_for.get(b, set()))
            rows.append({"condition_a": a, "condition_b": b,
                         "n_shared_preps": len(shared), "shared_preps": " | ".join(shared)})
    return pd.DataFrame(rows)
