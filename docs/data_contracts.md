# Public Data Contracts

DiffractoMorph separates study description, source-file parsing, and analysis.
This keeps instrument conventions and manuscript-specific assumptions out of the
public run model.

## Project manifest

A project begins with a YAML manifest. Relative paths resolve from the manifest;
run sources resolve inside its explicit `data_root`. Every manifest declares:

- a project identifier and independent-unit type;
- instrument, material, medium, analysis, noise, optical-operator, assay, and
  solubility profiles;
- each run's source, parser adapter, role, sample identity, and (for
  measurements) independent-unit identity.

Each profile must be supplied inline, loaded from a YAML or JSON file, or marked
not applicable with a reason. Optional SHA-256 values protect file-backed
profiles and run sources against silent replacement. Unknown fields, adapters,
and run roles are rejected rather than inferred.

The packaged schema is
`diffractomorph_pipeline/data/schema/project_manifest_v1.json`. Runtime profile
roles and run kinds are derived from that schema so the documented vocabulary
and validator cannot drift independently.

## Neutral run model

All adapters return `diffractomorph_pipeline.model.Run`, whose primary fields are:

- `signal`: frames by channels;
- `channel_ids`: stable, user-defined channel names;
- `time_min`: chronological elapsed time;
- `acquisition`: optional named frame-level variables;
- `stored_reference`: an optional static channel vector;
- `provenance`: explicit source, adapter, sample, and independent-unit identity.

No fixed channel count, PAQXOS field, compound name, or preparation-date naming
scheme is required by this contract. Legacy PAQXOS properties remain available
only as a migration aid for existing analyses.

## Adapters

Adapters are selected explicitly by ID. Gate 1 includes:

- `tidy_csv`: a vendor-neutral wide CSV with `time_min`, one or more
  `signal_<channel>` columns, optional `acq_<name>` columns, and optional static
  `ref_<channel>` columns;
- `paqxos_rtf`: the existing Sympatec PAQXOS export reader mapped into the neutral
  run model.

Run `dfm-manifest --example --inspect-runs` to validate and inspect the packaged
four-channel synthetic example without external data.
