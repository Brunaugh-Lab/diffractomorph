"""Discovery and provenance helpers for the antisolvent Tween 80 concentration study."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from diffractomorph_pipeline.assay import read_qc
from diffractomorph_pipeline.config import data_root


ARM_A_REL = Path("disso_experiments/dissolution_media_diagnostic/tween80_suspension_wetting_arm_a")
CONDITIONS = ("0.01", "0.03")


def default_study_root() -> Path:
    return data_root() / ARM_A_REL


def _condition(value) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*4\.5\s+(0\.0[13])%\s*w/v\s*Tween\s*", value, re.IGNORECASE)
    return match.group(1) if match else None


def _date(value) -> str | None:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return pd.to_datetime(value, unit="D", origin="1899-12-30").strftime("%Y-%m-%d")
    return None


def _volume_ul(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and (match := re.search(r"([0-9]+(?:\.[0-9]+)?)\s*u[lL]", value)):
        return float(match.group(1))
    return None


def read_run_volumes(path: Path | str) -> pd.DataFrame:
    """Select the replicated 0.01%/0.03% concentration rows from the shared dose workbook."""
    path = Path(path)
    rows = openpyxl.load_workbook(path, data_only=True).worksheets[0].iter_rows(values_only=True)
    current_date = current_condition = None
    records = []
    for row in rows:
        if not row:
            continue
        if (parsed_date := _date(row[0])) is not None:
            current_date = parsed_date
        if len(row) > 1 and row[1] is not None:
            current_condition = _condition(row[1])
        rep = row[2] if len(row) > 2 else None
        volume = _volume_ul(row[3] if len(row) > 3 else None)
        if current_date and current_condition and isinstance(rep, (int, float)) and volume is not None:
            records.append({
                "date": current_date,
                "tween_pct_wv": current_condition,
                "rep": int(rep),
                "injected_uL": volume,
            })
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(f"{path.name}: no replicated antisolvent Tween 80 dose rows found")
    keys = ["date", "tween_pct_wv", "rep"]
    if frame.duplicated(keys).any():
        duplicates = frame.loc[frame.duplicated(keys, keep=False), keys].to_dict("records")
        raise ValueError(f"{path.name}: duplicate antisolvent Tween 80 dose rows: {duplicates}")
    dates = sorted(frame["date"].unique())
    if len(dates) != 3:
        raise ValueError(f"{path.name}: expected 3 antisolvent Tween 80 preparation dates, found {dates}")
    day_for_date = {date: day for day, date in enumerate(dates, start=1)}
    frame.insert(1, "day", frame["date"].map(day_for_date).astype(int))
    expected = {(date, condition, rep) for date in dates for condition in CONDITIONS for rep in (1, 2, 3)}
    observed = set(frame[keys].itertuples(index=False, name=None))
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{path.name}: incomplete antisolvent Tween 80 dose design; missing={missing}, extra={extra}")
    return frame.sort_values(["date", "tween_pct_wv", "rep"]).reset_index(drop=True)


def _mapped_qc(path: Path) -> dict[str, object]:
    reads = read_qc(path)
    if len(reads) != 2:
        raise ValueError(f"{path.name}: expected 2 suspension reads, found {len(reads)}")
    # The plate layout places 0.01% first and 0.03% second on every preparation date.
    # Preserve that documented layout instead of assigning conditions from measured magnitude.
    return dict(zip(CONDITIONS, reads))


def read_day_qc(study_root: Path | str, day: int) -> tuple[dict[str, object], list[Path]]:
    """Read and cross-check the duplicated two-condition QC workbooks for one day."""
    study_root = Path(study_root)
    paths = sorted(study_root.glob(f"*_ Tween 80/Day {day}/QC/*.xlsx"))
    if not paths:
        raise FileNotFoundError(f"Day {day}: no QC workbook found under {study_root}")
    mapped = [_mapped_qc(path) for path in paths]
    reference = mapped[0]
    for path, candidate in zip(paths[1:], mapped[1:]):
        for condition in CONDITIONS:
            if not np.isclose(candidate[condition].abs_bs, reference[condition].abs_bs, atol=1e-9):
                raise ValueError(f"Day {day}: duplicated QC workbooks disagree for {condition}% ({path})")
    return reference, paths


def parse_uv_path(path: Path | str, study_root: Path | str) -> tuple[str, int, int]:
    path, study_root = Path(path), Path(study_root)
    relative = path.relative_to(study_root)
    condition_match = re.fullmatch(r"(0\.0[13])_ Tween 80", relative.parts[0])
    day_match = re.fullmatch(r"Day (\d+)", relative.parts[1])
    rep_match = re.search(r"rep\s*([123])", path.stem, re.IGNORECASE)
    if not (condition_match and day_match and rep_match):
        raise ValueError(f"unrecognized antisolvent Tween 80 UV path: {relative}")
    return condition_match.group(1), int(day_match.group(1)), int(rep_match.group(1))


def discover_uv(study_root: Path | str) -> pd.DataFrame:
    study_root = Path(study_root)
    records = []
    for path in sorted(study_root.glob("*_ Tween 80/Day */UV Data/*.xlsx")):
        condition, day, rep = parse_uv_path(path, study_root)
        records.append({"tween_pct_wv": condition, "day": day, "rep": rep, "uv_file": path})
    frame = pd.DataFrame(records)
    keys = ["tween_pct_wv", "day", "rep"]
    expected = {(condition, day, rep) for condition in CONDITIONS for day in (1, 2, 3) for rep in (1, 2, 3)}
    observed = set(frame[keys].itertuples(index=False, name=None)) if not frame.empty else set()
    if observed != expected or len(frame) != len(expected):
        raise ValueError(
            f"incomplete or duplicated antisolvent Tween 80 UV design; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}, files={len(frame)}"
        )
    return frame.sort_values(keys).reset_index(drop=True)


def arm_a_day_root(study_root: Path | str, condition: str, day: int) -> Path:
    """Return the canonical folder for one condition and preparation day."""
    condition = f"{float(condition):.2f}"
    if condition not in CONDITIONS or int(day) not in (1, 2, 3):
        raise ValueError(f"invalid antisolvent Tween 80 condition/day: {condition}, {day}")
    return Path(study_root) / f"{condition}_ Tween 80" / f"Day {int(day)}"


def _one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise FileNotFoundError(f"{label}: expected one match, found {len(paths)}: {paths}")
    return paths[0]


def find_qc_psd(study_root: Path | str, condition: str, day: int) -> Path:
    """Find the daily starting-suspension q0 source (RTF or per-frame CSV folder)."""
    qc = arm_a_day_root(study_root, condition, day) / "QC"
    folders = [qc / "q0 Data"] if (qc / "q0 Data").is_dir() else []
    if folders:
        return folders[0]
    return _one(sorted(qc.glob("*q0.rtf")), f"antisolvent Tween 80 {condition}% day {day} QC q0")


def find_measurement_rtf(study_root: Path | str, condition: str, day: int, rep: int) -> Path:
    """Find one raw diffraction measurement without admitting blank or q0 exports."""
    raw = arm_a_day_root(study_root, condition, day) / "Raw Data"
    paths = sorted(raw.glob(f"*Measurement Rep {int(rep)}.rtf"))
    return _one(paths, f"antisolvent Tween 80 {condition}% day {day} rep {rep} raw RTF")


def find_q3_source(study_root: Path | str, condition: str, day: int, rep: int) -> Path:
    """Find the q3 trajectory source (stacked CSV on day 1, frame folder thereafter)."""
    q3 = arm_a_day_root(study_root, condition, day) / "q3 Data"
    folder = q3 / f"Rep {int(rep)}"
    if folder.is_dir():
        return folder
    return _one(sorted(q3.glob(f"*Measurement Rep {int(rep)} q3.csv")),
                f"antisolvent Tween 80 {condition}% day {day} rep {rep} q3")
