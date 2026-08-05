# Test tiers and release checks

The private development repository contains several test tiers with different data and time
requirements. Public CI runs only redistributable tests from the clean snapshot.

| Tier | Purpose | Data boundary | Typical command |
|---|---|---|---|
| Public | Installed API, synthetic example, metadata and disclosure boundary | Redistributable only | `pytest -q tests/public` |
| Core | Algorithms, schemas, adapters and scientific contracts | Synthetic/frozen fixtures | `pytest -q -m "core and not real_corpus"` |
| Manuscript regression | Frozen estimands, figure source data and claim boundaries | Frozen derived fixtures; some tests may discover a private corpus | `pytest -q -m "manuscript and not real_corpus"` |
| Real corpus | End-to-end ingestion and study reconstruction | Private and optional | `pytest -q -m real_corpus` with an explicit `DFM_DATA_ROOT` |

Tests that require the private corpus must skip clearly when it is absent. A skip does not
prove manuscript reproduction. Release evidence must state which tiers ran, which skipped,
the wheel or source artifact tested, and whether the workflow was executed from an isolated
installation.

The public release check is:

1. build and verify the allowlisted snapshot;
2. build both source distribution and wheel;
3. install the wheel into an empty environment;
4. run `tests/public` against that installation;
5. execute the bundled synthetic workflow; and
6. scan the snapshot and distributions for secrets and disclosure markers.

The final manuscript archive adds a seventh check: install the public wheel, attach only the
licensed DOI-pinned data bundle, run the reproduction command, and compare numerical outputs
and frozen checksums.
