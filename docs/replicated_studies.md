# Replicated-study tutorial

This tutorial extends the bundled single-run example to a study with independent
preparations and nested technical runs. The central rule is that technical runs do not
increase the number of independent experimental units.

## 1. Arrange portable inputs

Convert each run to the tidy interchange format or retain the original PAQXOS RTF and
select `paqxos_rtf` explicitly. Keep raw files immutable. Put the manifest beside a
project-local data directory so every source can be expressed as a relative path.

```text
example-study/
  project.yaml
  data/
    prep-a-run-1.csv
    prep-a-run-2.csv
    prep-b-run-1.csv
    prep-b-run-2.csv
```

The tidy format requires `time_min` plus `signal_<channel-id>` columns. Optional
acquisition variables use `acq_<name>`, and an optional stored reference uses
`ref_<channel-id>`.

## 2. Declare the hierarchy

```yaml
schema_version: 1
project_id: compound-y-buffer-screen
data_root: data
independent_unit: preparation

runs:
  - id: prep-a-run-1
    source: prep-a-run-1.csv
    adapter: tidy_csv
    kind: measurement
    sample_id: compound-y
    independent_unit_id: prep-a
    technical_replicate: "1"
    metadata: {condition: buffer-a}
  - id: prep-a-run-2
    source: prep-a-run-2.csv
    adapter: tidy_csv
    kind: measurement
    sample_id: compound-y
    independent_unit_id: prep-a
    technical_replicate: "2"
    metadata: {condition: buffer-a}
```

Add the required explicit profiles shown in the bundled example. Use
`not_applicable_reason` only when the corresponding evidence stream is genuinely absent;
it is not a way to bypass a required calibration for an analysis that consumes it.

Each measurement run requires `independent_unit_id`. DiffractoMorph will not infer it
from a filename or directory. If one preparation contributes more technical runs than
another, those runs are averaged within preparation before preparation means are combined.

## 3. Validate before analysis

```bash
dfm-manifest project.yaml --inspect-runs
```

Validation checks the schema, adapter IDs, source containment, optional SHA-256 values,
profile declarations, channel identities, and independent-unit identity.

## 4. Run aggregate detector-space kinetics

```bash
dfm-aggregate-kww project.yaml --output-dir results/aggregate-kww
```

The output contains run-level fits, independent-unit means, and condition summaries.
Review the endpoint-specific contributor counts before reporting a result. A missing fit
must reduce the count for that endpoint rather than silently borrowing another endpoint's
replication.

## 5. Interpret at the correct level

- Aggregate angular-signal loss is a detector-space observation, not dissolved mass.
- KWW parameters describe trajectory shape and timescale; they do not identify mechanism.
- q3 is model-inverted relative composition from the same optical acquisition.
- Independent UV assay observations may test dissolved mass, but paired wavelengths are
  analytical estimates of the same sample rather than replicates.

Commit the manifest, code version, calibration identifiers, and output provenance. Keep
restricted raw data and local absolute paths outside the public code repository.
