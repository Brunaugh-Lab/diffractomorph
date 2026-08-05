# Examples

The package includes a redistributable, four-channel, non-CFZ example used to
verify the generic manifest and vendor-neutral CSV adapter. Validate and inspect it
from an installed package with:

```bash
dfm-manifest --example --inspect-runs
```

Real or private datasets remain ignored. Public example data must be deliberately
placed in the package-data example directory and reviewed, rather than allow-listed
wholesale under this folder.
