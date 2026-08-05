# Privacy-safe support and diagnostics

Public bug reports must not contain raw research data, confidential filenames, local paths,
credentials, unpublished collaborator material, or sample and preparation identities.

Create a diagnostic record with:

```bash
dfm-diagnostics project.yaml --output diagnostic.json
```

Add `--inspect-runs` to confirm that each adapter can read its source. The command records
only software/environment information, manifest structure, profile IDs, adapter IDs, run
counts, and optional frame/channel dimensions. It does not copy the manifest, raw values,
local paths, sample IDs, run IDs, or independent-unit IDs.

Review the JSON manually before sharing it. A useful issue also includes:

- the exact command that failed;
- the relevant traceback with local paths removed;
- the smallest synthetic reproduction possible; and
- the expected versus observed behavior.

Use private vulnerability reporting for suspected credentials, path traversal, unintended
file writes, or disclosure of input data. Scientific interpretation questions without a
security impact belong in the normal issue tracker.
