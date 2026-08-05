# Public-release boundary

The public repository is built from a clean reviewed snapshot. It does not inherit the
private development repository's Git history, abandoned branches, raw study data, or
local-path provenance.

The default snapshot contains the generic package, schemas, documentation, public
tests, and synthetic example. CFZ/NIST calibration payloads and manuscript data are
excluded until they are released through an explicitly licensed, versioned archive.
Generic workflows must therefore supply their own assay, solubility, noise, and optical
profiles. Legacy CFZ convenience functions fail with a clear message when the optional
profile bundle is absent.

Building a snapshot does not create a repository, push a branch, change visibility,
publish a package, or upload data. Those actions require explicit authorization.

Run locally:

```bash
python scripts/build_public_snapshot.py /tmp/diffractomorph-public
cd /tmp/diffractomorph-public
python -m pytest -q tests/public
python -m pip wheel --no-deps .
```

The builder writes `PUBLIC_SNAPSHOT_MANIFEST.sha256`, ignores files outside the
reviewed allowlist, and stops before copying if a selected file has an unreviewed
suffix, forbidden private path, operator initials, credential-shaped token, or
symlink.
