# Releasing

## GitHub Actions

`ci.yml` runs on `main` pushes, pull requests, and manual dispatches, and is reusable by the release workflow. It uses Python 3.12 on `ubuntu-24.04` and `macos-15`, installing dependencies with `uv sync --locked`. Checks cover Ruff, pytest, wheels and source distributions, distribution contents, and wheel installation outside the source tree. macOS also runs the native desktop smoke test.

`release.yml` runs on `v*` tag pushes and waits for CI on the same commit. It then builds the application on macOS arm64. The tag must equal `v` followed by the package version. A successful bundle launch and screenshot precede ZIP, wheel, source archive, and SHA-256 manifest creation. A separate job with `contents: write` creates a **draft** GitHub Release; other jobs have read-only repository permissions.

Actions are pinned to full commit SHAs. Dependabot checks Actions and uv updates monthly. Pull requests do not receive publishing secrets or publish Releases. The workflow does not publish to PyPI.

## Release steps

1. Initialize and push the project to a GitHub repository with `main` as its default branch and Actions enabled. The configuration does not assume a particular account or repository name.
2. Require both CI platform checks in branch protection. Branch protection is managed in repository settings rather than workflow YAML.
3. Update the versions in `pyproject.toml` and `src/openkalolumen/__init__.py`, run `uv lock`, and commit the change.
4. After `main` passes CI, create and push the matching tag; for example, use `v0.1.0` for package version `0.1.0`.
5. Wait for Draft release to complete, inspect the attachments, SHA-256 manifest, and notes, then publish the draft manually.

Repeated runs do not overwrite an existing Release. Inspect failure logs and, if necessary, delete an unpublished draft before rerunning. Do not move tags for versions that have already been published.

## Local validation

```sh
uv sync --locked --group dev
uv run ruff check src tests scripts packaging
uv run ruff format --check src tests scripts packaging
uv run pytest -q
uv run python scripts/smoke-desktop.py
uv build
uv run python scripts/check-distribution.py
scripts/build-app.sh
mkdir -p outputs
./dist/openKaloLumen.app/Contents/MacOS/openKaloLumen \
  --demo --capture outputs/bundle.png --quit-after 20
```

`check-distribution.py` requires project license texts in wheels and source archives and rejects models, images, caches, virtual environments, and source-archive symlinks. Desktop bundles include project notices, the third-party source index, runtime dependency metadata, and the Qt LGPL/GPL texts and attribution stored in `packaging/licenses/`.

The application currently uses an ad-hoc signature. Developer ID signing and Apple notarization are not configured, and no Apple developer credentials are stored in the repository. Release notes should state this signing status. Models and sample images remain separately obtained resources and must not be attached to Releases.
