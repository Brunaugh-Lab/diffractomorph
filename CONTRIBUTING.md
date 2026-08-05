# Contributing

Thank you for improving DiffractoMorph.

1. Open an issue before a large behavioral or scientific-contract change.
2. Use synthetic or explicitly redistributable fixtures. Never commit raw study data,
   credentials, private paths, participant information, or unpublished collaborator material.
3. Preserve the distinction among measured detector signal, fitted descriptors,
   model-inverted q3, UV-derived mass, and forward-model predictions.
4. Add tests for scientific behavior and provenance, not only execution.
5. Run:

   ```bash
   python -m pytest -q tests/public
   python -m pip wheel --no-deps .
   ```

The clean public repository ships the public test tier. Maintainers working in the restricted
development repository also run the core, manuscript, and optional real-corpus tiers described in
`docs/testing.md`.

By contributing, you agree that your code contribution is licensed under the MIT License.
Do not contribute third-party data unless its redistribution terms are documented.
