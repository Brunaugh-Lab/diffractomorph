"""pH-study experimental design + restricted-permutation bookkeeping.

The pH study is an **incomplete block design**: a fresh suspension **date** is the independent
**date block**, and the three runs within a (date × condition) cell are nested runs. (It is *not*
described as a randomized block — condition-to-date randomization was not documented.) The realized
design has four date blocks:

    20260608 : pH 4.0, pH 4.5              (2 conditions)
    20260609 : pH 4.0, pH 4.5, pH 5.0      (all 3 conditions)
    20260610 : pH 4.0, pH 5.0              (2 conditions)
    20260611 : pH 4.5, pH 5.0              (2 conditions)

so every condition appears on three date blocks, and **20260609 is the one date block that carries all
three conditions** while the other three blocks each carry two. This module reads the realized design
straight from the run inventory (it does not hard-code the cells) and provides the combinatorics needed
to report — rather than over-claim — a within-date permutation test on so few blocks:

* :func:`design_table`                — condition × date coverage (n reps per cell), from ``iter_runs``.
* :func:`distinct_restricted_permutations` — number of DISTINCT within-date condition-label
  arrangements available to a permutation test given the **date × condition units** actually present
  (identity included). This is the honest ceiling on permutation resolution; ``min_attainable_p = 1/N``.

Date × condition units: each (date × condition) cell collapses its nested runs to one vector, so within
a date block the labels are distinct and a block contributes ``n_conditions!`` arrangements; the product
over blocks is the total. The general multiset form (used if ever applied at run level) divides by the
per-label multiplicities.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import permutations, product
from math import factorial

import numpy as np
import pandas as pd

from psd_evolution_common import iter_runs


def design_cells():
    """``{(ph, date): n_reps}`` for the realized pH study, read from the run inventory."""
    cells = defaultdict(int)
    for ph, date, rep, rtf, fo in iter_runs():
        cells[(float(ph), int(date))] += 1
    return dict(cells)


def design_table() -> pd.DataFrame:
    """Condition (rows) × date (cols) reps-per-cell table, with per-condition ``n_days`` and per-date
    ``n_conditions`` margins. Empty cells are 0."""
    cells = design_cells()
    phs = sorted({ph for ph, _ in cells})
    dates = sorted({d for _, d in cells})
    rows = []
    for ph in phs:
        row = {"condition": f"pH {ph}"}
        n_dates = 0
        for d in dates:
            n = cells.get((ph, d), 0)
            row[str(d)] = n
            n_dates += int(n > 0)
        row["n_dates"] = n_dates
        row["n_runs"] = sum(cells.get((ph, d), 0) for d in dates)
        rows.append(row)
    df = pd.DataFrame(rows)
    # per-date condition count (blocks usable for a within-date between-condition contrast)
    margin = {"condition": "n_conditions"}
    for d in dates:
        margin[str(d)] = sum(1 for ph in phs if cells.get((ph, d), 0) > 0)
    margin["n_dates"] = np.nan
    margin["n_runs"] = sum(cells.values())
    return pd.concat([df, pd.DataFrame([margin])], ignore_index=True)


def restricted_label_arrangements(dates, conds):
    """Every DISTINCT within-date condition-label arrangement (identity included), as a list of label
    lists. Labels are permuted only among units sharing a date block, so the block structure is exact.

    ``len(restricted_label_arrangements(d, c)) == distinct_restricted_permutations(d, c)``. Use this to
    build an EXACT permutation null (enumeration), rather than Monte-Carlo draws, when the restricted
    group is small — which it always is here (four date blocks)."""
    dates = list(dates)
    conds = list(conds)
    by_date = defaultdict(list)
    for i, d in enumerate(dates):
        by_date[d].append(i)
    per_date = []
    for idxs in by_date.values():
        labels = [conds[i] for i in idxs]
        per_date.append((idxs, sorted(set(permutations(labels)))))
    out = []
    for combo in product(*[p[1] for p in per_date]):
        lab = [None] * len(conds)
        for (idxs, _), labels in zip(per_date, combo):
            for i, l in zip(idxs, labels):
                lab[i] = l
        out.append(lab)
    return out


def distinct_restricted_permutations(dates, conds) -> int:
    """Number of DISTINCT within-date label arrangements (identity included) for units carrying the
    given ``dates`` and condition ``conds`` (parallel sequences, one entry per unit).

    A date contributes ``n_d! / ∏_c m_{d,c}!`` distinct label arrangements (``n_d`` units in the date,
    ``m_{d,c}`` of them labelled ``c``); dates with a single condition contribute 1. The product over
    all dates is the size of the restricted-permutation group actually available. At day level (one
    unit per date×condition) this reduces to ``∏_d n_d!`` over dates with ≥2 conditions.
    """
    dates = list(dates)
    conds = list(conds)
    by_date = defaultdict(list)
    for d, c in zip(dates, conds):
        by_date[d].append(c)
    total = 1
    for d, cs in by_date.items():
        n_d = len(cs)
        denom = 1
        for c in set(cs):
            denom *= factorial(cs.count(c))
        total *= factorial(n_d) // denom
    return int(total)


def main():
    df = design_table()
    n_blocks = df[df.condition != "n_conditions"].shape[0] and \
        len([c for c in df.columns if c.isdigit()])
    print("pH study — incomplete block design; condition × date coverage (reps per cell):")
    print(df.to_string(index=False))
    print(f"\n{n_blocks} unique preparation-date blocks; "
          f"{int(df[df.condition != 'n_conditions'].n_runs.sum() // 3)} date × condition units "
          f"(3 reps each).")
    # day-level restricted-permutation ceiling over the full design (all cells present)
    cells = design_cells()
    day_dates = [d for (ph, d) in cells]
    day_conds = [ph for (ph, d) in cells]
    n = distinct_restricted_permutations(day_dates, day_conds)
    print(f"Full-design date × condition-unit distinct restricted permutations (identity incl.): {n}")
    print(f"min attainable permutation p = 1/{n} = {1.0 / n:.4f}")
    try:
        from psd_evolution_common import BASE
        out = BASE / "psd_evolution" / "design_coverage.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nwrote {out}")
    except Exception as e:  # pragma: no cover - reporting convenience only
        print(f"(design table not written: {e})")


if __name__ == "__main__":
    main()
