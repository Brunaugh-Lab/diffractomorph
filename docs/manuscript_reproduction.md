# JPharmSci manuscript reproduction

The generic public package and the CFZ/JPharmSci reference application are separate release
layers. The code snapshot contains no real CFZ, PAQXOS, or NIST-derived payload.

## Current verified state

The private development checkout has a restricted local manifest and one-command driver.
It verifies the manuscript-facing reference PDFs, preflights required raw and derived inputs,
runs nine declared recipes, records source/output hashes, scans generated files for disclosure,
and writes a code-state-bound report covering the manifest, analysis modules, reusable package
source, package version, and declared/actual Git commit. This local path passed independent review.

That result is not yet a public reproduction archive. The local manifest consumes a private
corpus and selected derived tables. Some manuscript panels are assembled outside the generic
pipeline and are checksum-verified rather than regenerated.

## Final archive command

The eventual archive must provide an equivalent command of the form:

```bash
python -m pip install diffractomorph_pipeline-<VERSION>-py3-none-any.whl
dfm-reproduce-jpharmsci \
  --bundle <downloaded-archive> \
  --output-dir reproduced
```

The exact command, version, DOI and checksum remain placeholders until the archive is
approved and deposited. Documentation must be updated with real identifiers before release.

## Required archive contents

- an archive manifest with DOI/version, license, file roles and SHA-256 values;
- the study manifest and preparation/technical-run hierarchy;
- every redistributable raw or derived input required by a recipe;
- calibration/profile IDs and payloads permitted for redistribution;
- manuscript-only analysis recipes excluded from the generic wheel;
- expected numerical tables and manuscript-facing reference checksums; and
- a machine-readable execution report schema.

Externally assembled physicochemical and composite optical panels must be labeled as such.
The archive must not imply that DiffractoMorph regenerates work owned by another pipeline.

## Scientific boundaries

UV-derived dissolved mass is independent evidence. q3 is model-inverted relative
composition from the same optical acquisition. Detector channels are angular measurements,
and preparation—not the number of technical cuvette runs—is the replicated unit.

The restricted development repository maintains availability-language and archive-manifest
templates. They are not shipped in the generic code snapshot because their identifiers, rights,
and checksums remain unresolved. Reviewed copies will accompany the versioned data archive.
