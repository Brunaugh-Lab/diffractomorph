# Calibration and profile guide

DiffractoMorph does not have a scientifically meaningful universal calibration. Every
calibration is tied to an instrument geometry, material, medium, acquisition regime, and
purpose. The project manifest makes that selection explicit.

## Profile roles

| Role | What it establishes | When required |
|---|---|---|
| `instrument` | Adapter, channel identities, acquisition variables and geometry | Every project |
| `material` | Material identity and parameters used by a named model | Every project |
| `medium` | Vessel and medium conditions | Every project |
| `analysis` | Channel set, artifact rules and fitted-model bounds | Every analysis |
| `noise` | Supported signal range and per-channel variability | Channel admission or reliability gating |
| `optical_operator` | Forward mapping and its allowed inference | Optical forward prediction or inversion QC |
| `assay` | Wavelengths, blanks, curves, dilution and offsets | Independent concentration/mass assay |
| `solubility` | Condition-specific saturation concentration and provenance | Mass-domain dissolution model |

A profile may be inline, file-backed, or explicitly not applicable. File-backed profiles
should carry SHA-256 checksums in frozen work.

## Noise calibration

Build a noise model from non-transforming material measured over the intended signal and
acquisition range. Record source hashes, instrument/lens, material/medium, temperature,
cadence, concentration range, exclusions, fit diagnostics, and a calibration ID. A CFZ
pH-7 surface cannot be assumed valid for another material or instrument.

## Optical operators

An operator record must include its immutable ID, purpose, input standards, geometry,
refractive-index assumptions, dimensions, size grid, fit diagnostics, build recipe, source
hashes, and artifact checksum. Detector channels are angular measurements, not size bins.
One broad standard can support a feasibility or geometry check without establishing unique
diameter-to-channel localization.

`dfm-build-kernel` requires an explicit external registry and a `YYYY-MM-DD` calibration
date. The installed package is not a writable calibration store.

## Assay calibration

An assay profile declares the calibrated wavelengths, curve parameters, blank convention,
dilution, filter or recovery offsets, units, calibration range, and calibration identity.
Missing condition-specific offsets fail closed. Passing an offset of zero is an explicit
statement that no offset applies.

## Solubility and forward parameters

Use measured, condition-specific solubility when the scientific analysis requires it and
record the preparation and method that produced it. Unknown materials must supply explicit
forward-model parameters. A model fit in total dissolved mass does not validate a
size-specific rate law or optical inversion.

## Validation checklist

Before freezing a profile:

1. confirm ownership and redistribution terms for every source;
2. calculate and verify source and artifact checksums;
3. define supported and unsupported ranges;
4. test a held-out or independent check where scientifically available;
5. state the allowed inference and explicit nonclaims;
6. give the artifact a stable ID and version; and
7. select it explicitly in a project manifest.
