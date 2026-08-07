"""The Arm C Cs ladders Arm B divides by — one primary, two secondary method sensitivities.

**Primary: ``filtered_48h``.** The 48 h syringe-filtered ladder is the operational
saturation-solubility measurement for Arm B, selected *prospectively* on the strength of the 48 h
centrifuge-vs-filter experiment (recorded 2026-07-04, commit ``9ed2cb9``) — not chosen after
seeing an Arm B result:

    0x    0.5x   1x     2x     10x     20x  (× CMC)
    5.95  7.77   8.39   8.85   12.97   20.78   µg/mL

That experiment was the resolving one. Centrifuged supernatant read 2–3.5 µg/mL **high** at the
intermediate levels (C2–C4), where a one-minute spin does not clear fine particulate, which then
reads as dissolved drug. Filtered and centrifuged **agree** at 0×, 10× and 20× CMC — where
particulate carryover is minimal — and that agreement is what validates the fixed additive filter
correction rather than leaving it assumed. So the filtered ladder is the clean phase-solubility
result across the range, and the apparent centrifuged "plateau" near 12 µg/mL at intermediate
levels was largely artifact.

**Secondary: ``centrifuged_48h`` and ``centrifuged_24h``.** Retained as *method* sensitivities.
They are not competing candidates for the operational Cs; they are the particulate-containing
preparation, and comparing against them demonstrates how much the mid-ladder answer moves when
particulate is not excluded. Report them as such, never as an equally plausible primary.

The 0.5× CMC level is the deliberate **below-CMC, monomer-only anchor** and is central to the
design — it separates a monomer-level solubility effect from a micellar one, so it stays in every
comparison.

Residual assay limitation (not a gate on interpretation): no CFZ standard curve exists at these
in-medium Tween levels, so absolute UV concentrations carry an uncalibrated matrix term. Arm B
conclusions rest on condition *comparisons* under the primary ladder, which is where that term is
least influential; see :mod:`arm_b_uv_timecourse`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from diffractomorph_pipeline.config import data_root

ARM_C_REL = Path("physicochemical_characterization/saturation_solubility"
                 "/Cs_CMC_surfactant_media_effects_pH_4.5")
SUMMARY_48H = "arm_c_48h_summary.csv"
LADDER_24H = "arm_c_cs_ladder_24h.csv"

# Arm C ladder level → Arm B condition. C1/C4/C6 have no Arm B counterpart.
LEVEL_TO_CONDITION = {"C2": "0.5x CMC", "C3": "1.0x CMC", "C5": "10x CMC"}

PRIMARY_LADDER = "filtered_48h"          # prospectively selected; see the module docstring
SECONDARY_LADDERS = ("centrifuged_48h", "centrifuged_24h")   # particulate-containing, method only
LADDER_NAMES = (PRIMARY_LADDER, *SECONDARY_LADDERS)
DEFAULT_LADDER = PRIMARY_LADDER

LADDER_ROLE = {PRIMARY_LADDER: "primary",
               **{name: "secondary_method_sensitivity" for name in SECONDARY_LADDERS}}


def arm_c_dir() -> Path:
    return data_root() / ARM_C_REL


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Arm C output missing: {path}. Regenerate with analysis/arm_c_48h.py "
            f"and analysis/arm_c_cs_ladder.py.")
    return pd.read_csv(path)


def ladders(arm_c: Path | str | None = None) -> pd.DataFrame:
    """Tidy Cs ladders for the three Arm B conditions, one row per (ladder, condition).

    Carries ``role`` (primary vs secondary method sensitivity), ``sd_ugml`` where Arm C reports
    it, and the source file, so downstream outputs state which solubility they used and why.
    """
    root = Path(arm_c) if arm_c is not None else arm_c_dir()
    h48 = _read(root / SUMMARY_48H).set_index("level")
    h24 = _read(root / LADDER_24H).set_index("level")
    spec = [
        (PRIMARY_LADDER, h48, "filt48", "filt48_sd", SUMMARY_48H,
         "48 h syringe-filtered supernatant + fixed additive filter offset; the operational "
         "saturation solubility, selected prospectively (commit 9ed2cb9)"),
        ("centrifuged_48h", h48, "cent48", "cent48_sd", SUMMARY_48H,
         "48 h centrifuged supernatant — retains fine particulate at intermediate Tween; "
         "secondary method sensitivity only"),
        ("centrifuged_24h", h24, "cs_avg", "sd_avg", LADDER_24H,
         "24 h centrifuged supernatant — pre-plateau and particulate-containing; "
         "secondary method sensitivity only"),
    ]
    rows = []
    for name, table, col, sd_col, source, description in spec:
        for level, condition in LEVEL_TO_CONDITION.items():
            rows.append({"ladder": name, "role": LADDER_ROLE[name], "condition": condition,
                         "arm_c_level": level,
                         "cs_ugml": float(table.loc[level, col]),
                         "sd_ugml": float(table.loc[level, sd_col]),
                         "source_file": source, "description": description})
    return pd.DataFrame(rows)


def cs_map(ladder: str = PRIMARY_LADDER, arm_c: Path | str | None = None) -> dict[str, float]:
    """``{condition: Cs µg/mL}`` for one named ladder (defaults to the primary)."""
    if ladder not in LADDER_NAMES:
        raise ValueError(f"unknown Cs ladder {ladder!r}; expected one of {LADDER_NAMES}")
    table = ladders(arm_c)
    return dict(table.loc[table["ladder"].eq(ladder), ["condition", "cs_ugml"]]
                .itertuples(index=False, name=None))


def centrifuge_filter_gap(arm_c: Path | str | None = None) -> pd.DataFrame:
    """Centrifuged − filtered at 48 h, per Arm B condition.

    A *diagnostic of particulate carryover in the centrifuged preparation*, not a defect in the
    selected filtered Cs: it is large at the intermediate levels the spin cannot clear and small
    where the two methods agree.
    """
    table = ladders(arm_c).pivot(index="condition", columns="ladder", values="cs_ugml")
    gap = (table["centrifuged_48h"] - table[PRIMARY_LADDER]).rename("gap_ugml").reset_index()
    gap["particulate_carryover_suspected"] = gap["gap_ugml"] > 1.0
    return gap


def provenance(ladder: str = PRIMARY_LADDER, arm_c: Path | str | None = None) -> dict:
    """Machine-readable record of which solubility a result was computed against."""
    table = ladders(arm_c)
    sub = table[table["ladder"].eq(ladder)]
    if sub.empty:
        raise ValueError(f"unknown Cs ladder {ladder!r}")
    return {"cs_ladder": ladder,
            "cs_role": LADDER_ROLE[ladder],
            "cs_is_primary": ladder == PRIMARY_LADDER,
            "cs_selection": ("operational saturation solubility, selected prospectively from the "
                             "48 h centrifuge-vs-filter experiment (commit 9ed2cb9)"
                             if ladder == PRIMARY_LADDER else
                             "secondary method sensitivity (particulate-containing preparation); "
                             "not the operational Cs"),
            "cs_source_file": sub["source_file"].iloc[0],
            "cs_description": sub["description"].iloc[0],
            "cs_ugml": {r.condition: r.cs_ugml for r in sub.itertuples()},
            "cs_arm_c_dir": str(arm_c_dir())}
