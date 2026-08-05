# Release engineering

`public_snapshot_policy.json` defines the clean-snapshot boundary. The builder copies
only reviewed public files and intentionally excludes Git history, research analysis
scripts, real calibration payloads, and manuscript data.

This directory records preparation policy only. It does not authorize publication.

- `manuscript-constraints.txt` pins the direct environment used for the verified local
  Gate 4 run; a transitive lock must be frozen with the final licensed archive.
- `archive-manifest-template.json` lists the identifiers, rights fields, roles and checksums
  required before deposit.
- `manuscript-availability-template.md` provides manuscript wording with conspicuous
  placeholders that must be replaced from live, logged-out-verified deposits.
