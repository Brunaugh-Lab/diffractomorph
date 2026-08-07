"""Design-table + restricted-permutation bookkeeping for the pH study (`analysis/study_design.py`).

Locks the realized incomplete block design and the terminology the manuscript depends on: four unique
preparation-date blocks, nine date × condition units, and 48 distinct within-date condition-label
arrangements (identity included). The pure-combinatoric tests use hand-built inputs; the data-gated test
checks the design table against the real run inventory when the corpus is configured.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
sys.path.insert(0, str(ANALYSIS))

sd = pytest.importorskip("study_design")

# The realized pH-study blocks (from the run inventory) — the design the manuscript reports.
REALIZED = {
    20260608: ("pH 4.0", "pH 4.5"),
    20260609: ("pH 4.0", "pH 4.5", "pH 5.0"),
    20260610: ("pH 4.0", "pH 5.0"),
    20260611: ("pH 4.5", "pH 5.0"),
}


def _day_units():
    dates, conds = [], []
    for d, cs in REALIZED.items():
        for c in cs:
            dates.append(d)
            conds.append(c)
    return dates, conds


# ── combinatorics (no corpus needed) ────────────────────────────────────────────────
def test_distinct_restricted_permutations_is_48():
    dates, conds = _day_units()
    # one block with all 3 conditions (3!) and three blocks with 2 conditions (2! each): 6*2*2*2 = 48
    assert sd.distinct_restricted_permutations(dates, conds) == 48


def test_enumerated_arrangements_match_count_and_are_distinct():
    dates, conds = _day_units()
    arr = sd.restricted_label_arrangements(dates, conds)
    assert len(arr) == sd.distinct_restricted_permutations(dates, conds) == 48
    assert len({tuple(a) for a in arr}) == 48                      # all distinct
    # every arrangement preserves each block's condition multiset (within-date permutation only)
    idx_by_date = {}
    for i, d in enumerate(dates):
        idx_by_date.setdefault(d, []).append(i)
    for a in arr:
        for d, idxs in idx_by_date.items():
            assert sorted(a[i] for i in idxs) == sorted(conds[i] for i in idxs)


def test_identity_arrangement_is_present():
    dates, conds = _day_units()
    arr = sd.restricted_label_arrangements(dates, conds)
    assert conds in arr                                            # identity (observed labels) included


def test_single_condition_block_contributes_no_freedom():
    # a block with one condition cannot be permuted → factor 1
    dates = [1, 1, 2, 2]
    conds = ["a", "a", "a", "b"]      # date 1 all 'a' (no freedom), date 2 has {a,b} (2! = 2)
    assert sd.distinct_restricted_permutations(dates, conds) == 2


# ── terminology present in the module documentation ──────────────────────────────────
def test_docstring_uses_corrected_terminology():
    doc = " ".join(sd.__doc__.split())                            # normalize line-wrapping whitespace
    assert "incomplete block design" in doc
    assert "incomplete randomized block" not in doc               # the original over-claim is gone
    assert "20260609 is the one date block that carries all three conditions" in doc
    for term in ("date block", "date × condition unit", "nested run"):
        assert term in doc


# ── data-gated: the realized design table matches the corpus inventory ────────────────
def test_design_table_matches_realized_design():
    try:
        cells = sd.design_cells()
    except Exception as e:                                          # corpus not configured
        pytest.skip(f"run inventory unavailable: {e}")
    if not cells:
        pytest.skip("empty run inventory")
    got = {}
    for (ph, date), n in cells.items():
        got.setdefault(int(date), []).append(f"pH {ph:.1f}")
    got = {d: tuple(sorted(v)) for d, v in got.items()}
    assert got == {d: tuple(sorted(cs)) for d, cs in REALIZED.items()}

    df = sd.design_table()
    assert "n_dates" in df.columns and "n_days" not in df.columns  # renamed away from "days"
    body = df[df.condition != "n_conditions"]
    assert set(body.n_dates) == {3}                                # every condition on 3 date blocks
    assert int(body.n_runs.sum()) == 27                            # 9 units × 3 nested runs
