# Scientific Contracts

Gate 2 separates measured observables, fitted descriptors, model-inverted
quantities, and forward predictions. These categories are not interchangeable.

## Aggregate angular KWW

`processing.fit_aggregate_kww` is the manuscript-authoritative particle-side
endpoint. It sums the explicitly selected measured detector channels, applies
the one-sided upward Hampel repair, and fits a free-amplitude KWW curve. The
default profile selects every measured channel and retains the stored reference
in the signal. A reference-adjusted fit is available only through the explicit
`reference_mode="reference_adjusted"` sensitivity setting.

The returned mean relaxation time, beta, half-time, and optical decay depth are
empirical detector-space descriptors. Copt is never substituted for the
aggregate, and optical decay depth is not dissolved mass.

`dfm-aggregate-kww MANIFEST --output-dir OUTPUT` writes per-run fits, technical-
run means within each declared independent unit, and condition summaries that
weight independent units equally. Overall and per-endpoint contributing run and
independent-unit counts are stored separately so missing values cannot inflate the
reported replication for an endpoint.

## Artifact processing

`processing.correct_artifacts` requires a named Copt-like acquisition variable.
It records startup removal, synchronized interpolation, isolated-channel median
replacement, and gap re-zeroing in a frame-level ledger. The separate aggregate
Hampel repair remains separate in both code and provenance.

## Matched-extent q3

`processing.matched_q3_extent` timestamp-matches q3 and detector frames, uses a
monotone detector remaining-signal fraction as the progress coordinate, applies
signal and absolute acquisition-reliability gates, validates finite normalized
q3 frames, and returns an explicit rejection reason for every target. For the
JPharmSci profile, Copt >= 0.79 is the inclusive/provisional rule and Copt >= 4%
is the supported sensitivity rule; the retired threshold of 30% of each run's
maximum is not used. A q3 volume fraction above 100 micrometers greater than 1%
is recorded as a review flag and is not an automatic exclusion in the inclusive
summary. The JPharmSci recipe writes three distinct profiles: inclusive
(Copt >= 0.79, review frames retained), primary coarse-excluded (Copt >= 0.79,
review frames omitted), and stringent (Copt >= 4%, review frames omitted). The
separate legacy `psd.frame_mask(copt_floor_frac=...)` helper is not part of this
manuscript-authoritative path. q3 remains normalized, PAQXOS-inverted relative
composition from the same optical acquisition. The API never multiplies q3 by
Copt, UV recovery, or another implied mass scale.

The JPharmSci compatibility profile also retains a separate 15-micrometer
inversion-support diagnostic: D90 and span are reported as missing when D90
falls beyond that boundary, while restricted-range descriptors and the explicit
`tail_unstable` flag remain available. This diagnostic is distinct from the
manuscript's 100-micrometer coarse-tail review rule.

## UV assay

Generic assay work begins with `AssayCalibration.from_mapping` or
`load_assay_profile`. Curves, blanks, filter offsets, dilution, and the paired
wavelengths are selected explicitly by the caller. `uv_timecourse_profiled`
additionally requires the material-specific solubility value and fails closed
when a condition-specific filter offset or wavelength calibration is absent; it does not select CFZ calibration or
solubility implicitly. The older CFZ constants and `uv_timecourse` defaults
remain compatibility interfaces for the frozen manuscript scripts.

Callers may pass `filter_offset_ugml=0.0` for a filter-free sample. The dosing-
suspension assay is filter-free by default, matching the manuscript; its former
borrowed filter correction remains only as an explicitly requested legacy
sensitivity.

UV-derived dissolved mass is an independent evidence stream. Paired wavelengths
are analytical estimates of one sample, not independent replicates.

## Forward-model identities

Forward engines are selected by immutable IDs:

- `mass_surface_ph_nb_v1`: primary independent mass-domain Nernst--Brunner model;
- `mass_morphology_diagnostic_v1`: exploratory morphology/rate sensitivity;
- `optical_fitted_g_v1`: legacy optical fitted-rate engine, which uses detector
  signal and an optical operator and therefore is not independent mass validation.

`forward.run_named_model` returns the selected model specification with the
numerical result so the domain and allowed inference remain attached.

The convenience `forward.predict` interface does not select a material implicitly.
Generic callers pass an explicit `Parameters` object; the optional legacy CFZ profile is
selected only by writing `drug="CFZ"`. Snapshot-anchored predictions additionally require the
dissolution volume. Optical-kernel construction requires explicit material, lens, calibration
date, and external registry arguments.
