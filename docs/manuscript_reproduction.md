# JPharmSci manuscript reproduction

The generic public package and the CFZ/JPharmSci reference application are separate release
layers. The code snapshot contains no real CFZ, PAQXOS, or NIST-derived payload.

## Current verified state

The public repository now contains a restricted migration-equivalence study layer at
`studies/jpharmsci_clofazimine`. Its manifest and one-command driver preserve the current
manuscript science while the archive layout is finalized.
It verifies the manuscript-facing reference PDFs, preflights required raw and derived inputs,
runs eleven declared recipes, records source/output hashes, scans generated files for disclosure,
and writes a code-state-bound report covering the manifest, analysis modules, reusable package
source, package version, and declared/actual Git commit. The migration code and release boundary
passed independent Gate 2 review; clean-room data-archive completion remains a separate gate.

This is not yet a clean-room public reproduction archive. The migration manifest still consumes
the established local corpus layout and selected derived tables. It also declares four external,
data-derived profiles (assay, solubility, noise surface, and optical kernel) that must be added to a
new version of the separately licensed data archive. Some manuscript panels are assembled in the
writing repository and are checksum-verified rather than regenerated.

## Final archive command

The eventual archive must provide an equivalent command of the form:

```bash
python -m pip install diffractomorph_pipeline-<VERSION>-py3-none-any.whl
python studies/jpharmsci_clofazimine/reproduce.py \
  --data-root <unpacked-archive> \
  --manuscript-root <manuscript-checkout> \
  --output-dir reproduced \
  --mode all
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
