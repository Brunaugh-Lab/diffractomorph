# Clofazimine JPharmSci reference application

This directory contains the study-specific application of the generic DiffractoMorph package used
for the clofazimine manuscript. It is intentionally outside `src/diffractomorph_pipeline`: drug,
instrument, calibration, experimental-design, and manuscript-figure choices are not generic package
defaults.

## Current gate

The migrated code is a **current-manuscript equivalence draft**. The pH analysis explicitly
recomputes aggregate KWW endpoints from raw RTF exports with the circulation-start policy used by
the current manuscript. It preserves the Copt-based Arm B reference analysis solely to verify that
moving code between repositories does not change that existing result. The planned
summed-angular-intensity Arm B analysis is a separate scientific revision and must not be conflated
with this migration.

`manifest.json` describes the current local corpus layout and manuscript reference artifacts.
`reproduce.py` preflights declared inputs, runs the study recipes, records hashes and code state, and
checks manuscript artifacts. It does not copy or redistribute source data.

The `manual_r3_operator` recipe reconstructs nominal HELOS R3 annular boundaries from the
manufacturer's 31 class limits, 100-mm focal length, and Fraunhofer first-minimum relation. It then
propagates the certified NIST SRM 1021 distribution through a cross-section-weighted annular Mie
operator. No detector-geometry parameter is fitted to the NIST profiles. The exported geometry CSV
records every radius and medium-angle boundary; the metrics JSON records the signal-transform,
certificate-tail, angle-mapping, and channel-orientation sensitivities. This is an evaluation of a
nominal optical-equivalent geometry, not mechanical detector metrology or a detector-gain
calibration.

## External study profiles

The generic wheel does not contain clofazimine calibration payloads. The study bundle must supply
the following files, selected by the runner through explicit environment variables:

| Variable | Bundle path |
|---|---|
| `DFM_ASSAY_PROFILE` | `profiles/assay/calibration.json` |
| `DFM_SOLUBILITY_PROFILE` | `profiles/solubility/cfz_cs_ph.json` |
| `DFM_NOISE_SURFACE` | `profiles/noise/cfz_ph7_surface.json` |
| `DFM_OPTICAL_KERNEL` | `profiles/kernels/R3_CFZ_20260601_1.71.npz` |

These data-derived profiles belong in the separately licensed data archive, not under the MIT
software license. The currently deposited archive will need a new version containing the declared
profiles and archive-layout manifest before clean-room reproduction can pass.

## Local verification

With a compatible local corpus and the four profile files available:

```bash
python studies/jpharmsci_clofazimine/reproduce.py \
  --data-root /path/to/corpus \
  --manuscript-root /path/to/Manuscripts/DiffractoMorph_JPharmSci \
  --output-dir /path/to/reproduced \
  --mode all
```

The study tests are separate from the public package tests:

```bash
pytest -q studies/jpharmsci_clofazimine/tests
```

Tests needing the real corpus or external profiles skip when those inputs are unavailable. The
generic public suite remains independent of all clofazimine data.
