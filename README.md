# DiffractoMorph

DiffractoMorph is a Python package for using time-resolved, multichannel light
scattering as a proxy for particle dissolution. It provides an instrument-neutral
run model, a PAQXOS RTF adapter, artifact and noise handling, aggregate KWW
descriptors, matched-extent q3 analysis, UV assay primitives, hierarchical study
summaries, and explicitly named forward models.

Version 0.1.0 is an alpha research release. APIs and schemas may still change.

## Scientific scope

DiffractoMorph keeps four evidence types separate:

- measured detector-channel signal;
- empirical fitted descriptors such as mean relaxation time and optical decay depth;
- model-inverted q3 relative composition from the same optical acquisition; and
- independent UV-derived dissolved mass and forward-model predictions.

Detector channels are angular measurements, not particle-size bins. Optical signal
loss is not automatically dissolved mass. q3 is supporting model-inverted evidence,
not an independent mass measurement or a biological replicate.

## Installation

```bash
git clone https://github.com/Brunaugh-Lab/diffractomorph.git
cd diffractomorph
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

This project has not yet been published to PyPI. DiffractoMorph supports Python
3.10 through 3.13.

## Five-minute synthetic example

The installed package contains a four-channel synthetic compound with an explicit
project manifest and no private paths or real study data.

```bash
EXAMPLE_MANIFEST=$(python -c "from diffractomorph_pipeline.study import bundled_example_manifest; print(bundled_example_manifest())")

dfm-manifest "$EXAMPLE_MANIFEST"
dfm-aggregate-kww "$EXAMPLE_MANIFEST" --output-dir example-output
```

The aggregate workflow writes:

- `aggregate_kww_by_run.csv`;
- `aggregate_kww_by_independent_unit.csv`; and
- `aggregate_kww_by_condition.csv`.

Technical runs are averaged within the independent unit declared in the manifest;
independent units are then weighted equally.

## Input model

Every public workflow begins with a project manifest. It declares:

- run files and adapters;
- channel identities and acquisition variables;
- sample and independent-unit identities;
- material, medium, assay, solubility, noise, and optical profiles; and
- analysis parameters and QC choices.

The initial instrument adapter supports Sympatec PAQXOS RTF exports. Other
instruments require adapters; the package does not claim universal file-format
compatibility. A tidy-CSV adapter is included for portable examples and converted
data.

See [data contracts](docs/data_contracts.md), [file format](docs/file_format.md),
[scientific contracts](docs/scientific_contracts.md), and the
[calibration guide](docs/calibration.md). New users can continue with the
[replicated-study tutorial](docs/replicated_studies.md) or inspect the
[clofazimine JPharmSci reference application](studies/jpharmsci_clofazimine/README.md). The latter
keeps manuscript-specific parameters and figure workflows outside the generic package and requires
the separately licensed data archive.

## Calibration and manuscript data

The clean public snapshot contains only schemas and synthetic example data. Real
CFZ/NIST standard exports, material-specific assay and solubility calibrations,
noise profiles, and optical kernels are not bundled until their provenance and data
license are approved. Supply explicit profiles for your own material and instrument.

The clofazimine/JPharmSci reproduction bundle will be distributed separately as a
versioned archive with checksums. See [the public-release boundary](docs/public_release.md).

## Main commands

```text
dfm-manifest       validate and summarize a project manifest
dfm-run            run a manifest-driven study workflow
dfm-ingest         convert PAQXOS RTF exports to tidy data
dfm-noise-filter   evaluate detector-channel signal above a supplied noise model
dfm-build-kernel   build an explicitly configured optical operator
dfm-qc             run size-consistency quality control
dfm-noise-floor    estimate a scalar noise floor
dfm-noise-surface  build a per-channel noise surface
dfm-extract        run classifier-routed exploratory extraction
dfm-aggregate-kww  run the manuscript-authoritative aggregate KWW workflow
dfm-diagnostics    write a privacy-safe environment and manifest diagnostic record
```

Commands that require material- or instrument-specific calibration do not silently
select a generic scientific default.

## Support and contributing

Search the public issue tracker before opening a bug report. Include the package
version, operating system, command, traceback, and a minimal synthetic reproduction.
Do not attach raw or confidential research data.

`dfm-diagnostics MANIFEST --output diagnostic.json` creates a support record without
raw values, sample identities, independent-unit identities, or local paths. Review it
before attaching it to an issue. See [support guidance](docs/support.md).

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the
[MIT License](LICENSE).

## Citation

Brunaugh, A., & Al-Gousous, J. (2026). *DiffractoMorph: Multichannel
Light-Scattering Analysis as a Proxy for Particle Dissolution* (v0.1.0). Zenodo.
https://doi.org/10.5281/zenodo.21810748
