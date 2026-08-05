"""Hierarchical summaries that preserve the declared independent unit."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HierarchicalSummary:
    independent_units: pd.DataFrame
    conditions: pd.DataFrame
    independent_unit_column: str
    run_id_column: str


def summarize_hierarchy(
    rows: pd.DataFrame,
    *,
    condition_columns: tuple[str, ...] = ("condition",),
    value_columns: tuple[str, ...],
    independent_unit_column: str = "independent_unit_id",
    run_id_column: str = "run_id",
) -> HierarchicalSummary:
    """Average technical runs within units, then weight independent units equally."""
    required = set(condition_columns) | set(value_columns) | {
        independent_unit_column, run_id_column,
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"hierarchical summary missing columns: {', '.join(missing)}")
    if rows.empty:
        raise ValueError("hierarchical summary requires at least one row")
    identity = rows[independent_unit_column].astype("string")
    if identity.isna().any() or identity.str.strip().eq("").any():
        raise ValueError("every row requires an independent-unit identity")
    if rows[run_id_column].astype("string").duplicated().any():
        raise ValueError("run identifiers must be unique")

    unit_keys = [*condition_columns, independent_unit_column]
    unit_values = rows.groupby(unit_keys, dropna=False, as_index=False)[list(value_columns)].mean()
    unit_counts = (
        rows.groupby(unit_keys, dropna=False)[run_id_column].nunique().rename("n_runs").reset_index()
    )
    unit_values = unit_values.merge(unit_counts, on=unit_keys, validate="one_to_one")
    for column in value_columns:
        contributing = (
            rows.assign(_finite=np.isfinite(pd.to_numeric(rows[column], errors="coerce")))
            .groupby(unit_keys, dropna=False)["_finite"].sum()
            .rename(f"{column}_n_runs")
            .reset_index()
        )
        unit_values = unit_values.merge(contributing, on=unit_keys, validate="one_to_one")

    condition_counts = (
        unit_values.groupby(list(condition_columns), dropna=False)
        .agg(n_independent_units=(independent_unit_column, "nunique"), n_runs=("n_runs", "sum"))
        .reset_index()
    )
    summaries = []
    for condition_key, group in unit_values.groupby(list(condition_columns), dropna=False):
        condition_key = condition_key if isinstance(condition_key, tuple) else (condition_key,)
        row = dict(zip(condition_columns, condition_key))
        for column in value_columns:
            values = group[column].to_numpy(float)
            finite = values[np.isfinite(values)]
            row[f"{column}_mean"] = float(np.mean(finite)) if finite.size else np.nan
            row[f"{column}_sd"] = float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan
            row[f"{column}_n_independent_units"] = int(finite.size)
            row[f"{column}_n_runs"] = int(
                group.loc[np.isfinite(values), f"{column}_n_runs"].sum()
            )
        summaries.append(row)
    conditions = pd.DataFrame(summaries).merge(
        condition_counts, on=list(condition_columns), validate="one_to_one",
    )
    return HierarchicalSummary(unit_values, conditions, independent_unit_column, run_id_column)
