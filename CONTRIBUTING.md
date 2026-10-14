# Contributing

Use Python 3.12 and uv. Routine development requires no model or CT downloads.

```sh
uv sync --locked --group dev
uv run ruff check src tests scripts packaging
uv run ruff format --check src tests scripts packaging
uv run pytest -q
```

For desktop interaction, rendering, or scene restoration changes, run the native smoke test on macOS:

```sh
uv run python scripts/smoke-desktop.py
```

For package metadata, structure, or distribution changes, also run:

```sh
uv build
uv run python scripts/check-distribution.py
```

## Code style

Ruff formats Python with an 88-character line length and sorts imports. Public classes and functions require docstrings. Complex APIs use Google-style `Args:`, `Returns:`, and `Raises:` sections. Document physical units, coordinate order, array shapes, and behavioral constraints. Simple callbacks use a concise description. Place necessary algorithm explanations in docstrings and avoid inline comments that repeat the code.

Keep UI text, CLI help, errors, documentation, and contributor instructions in English. Keep new code within the existing module boundaries. Remove unused code, duplicate paths, and machine-specific policies. Preserve DICOM geometry validation, unit conversions, cancellation isolation, empty-segmentation rejection, and export integrity checks. Compatibility handling needs a reproducible failure and appropriate validation.

Update `uv.lock` when changing application dependencies. Optional inference uses the separate `requirements-segmentation.lock`; changes to that environment should be validated with actual model inference. State explicitly when model access prevents validation.

## Submission scope

Commit source, tests, documentation, and build configuration. Exclude images, weights, virtual environments, caches, screenshots, and historical outputs. `.gitignore` and the source-distribution allowlist jointly control distribution contents. Do not include patient or personal data in issues, fixtures, or screenshots.

Keep machine paths, account details, credentials, local model symlinks, and personal preferences outside the repository. Examples should use generic paths or environment variables.

Describe the specific problem, resulting behavior, and completed validation in each PR. Contributions use this project's Apache-2.0 license. Third-party files must retain their original licenses and attribution.
