# Release engineering

`public_snapshot_policy.json` defines the clean-snapshot boundary. The builder copies
only reviewed public files and intentionally excludes Git history, research analysis
scripts, real calibration payloads, and manuscript data.

This public directory records snapshot policy only. It does not authorize publication.
The restricted development repository also maintains three archive-preparation templates that
are intentionally withheld until the licensed manuscript bundle is approved:

- `manuscript-constraints.txt` pins the direct environment used for the verified local
  Gate 4 run; a transitive lock must be frozen with the final licensed archive.
- `archive-manifest-template.json` lists the identifiers, rights fields, roles and checksums
  required before deposit.
- `manuscript-availability-template.md` provides manuscript wording with conspicuous
  placeholders that must be replaced from live, logged-out-verified deposits.

These three files are not part of the generic code snapshot and should not be expected in this
directory. The eventual archive will carry reviewed, versioned copies.
