# openKaloLumen

[![CI](https://github.com/ydy0615/openKaloLumen/actions/workflows/ci.yml/badge.svg)](https://github.com/ydy0615/openKaloLumen/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

A local CT/MR desktop viewer built with Python, Qt, and VTK. It supports layered isosurfaces, finite tissue volumes, lumen clipping, PBR materials, volume rendering, MPR, DRR, measurements, and mesh export. Optional organ segmentation runs in a separate Python process.

The supported desktop target is **macOS Apple Silicon with Python 3.12**. CI also checks image computation and package installation on Linux. Windows and Linux desktop operation have not been validated. This independent implementation is intended for education and research and has not undergone clinical validation.

## Install and launch

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the project directory:

```sh
uv sync --locked
uv run openkalolumen
```

The default view uses a generated synthetic lumen. It requires no CT download, model weights, or pre-existing scenes.

```sh
uv run openkalolumen --image /path/to/image.nii.gz
uv run openkalolumen --scene /path/to/scene.json
uv run openkalolumen --demo
uv run openkalolumen --help
```

Supported inputs include DICOM folders and static 3D scalar NIfTI, NRRD, and MHA/MHD images. DICOM slices are sorted by physical position; missing slices and irregular geometry are rejected. MRI and uncalibrated images use relative intensities.

Public CT samples can be selected from the toolbar or downloaded explicitly:

```sh
uv run openkalolumen-samples CTACardio
uv run openkalolumen-samples all --directory /path/to/samples
```

Downloads are verified with SHA-256 and receive a provenance sidecar. Sample images are excluded from source and application distributions.

## Features

- Layers represent individual isosurfaces, interval boundaries, finite tissue, semantic labels, or volume intensity intervals.
- Threshold changes use a GPU preview while detailed meshes are computed in the background. Exterior, interior, and section colors and roughness are configurable.
- Physical clipping supports open lumens, grayscale slices, and illuminated tissue sections. Finite materials retain tissue thickness and cavity openings.
- Orthogonal and oblique MPR sample the original image. Display smoothing uses millimeters and affects only the 3D display field.
- ROI boxes, connected regions, and labels restrict visibility. Distances, angles, and annotations use LPS coordinates in millimeters.
- DRR integrates reliably HU-calibrated CT using a single-energy attenuation approximation and exports 16-bit PNG images.
- Scenes save cameras, layers, clipping, MPR, and measurements and reference external images, labels, and masks.
- STL and colored PLY export use original scalar geometry. Clipped finite materials must be closed and free of nonmanifold edges.

## Application paths

Caches use the platform's application data directory: `~/Library/Application Support/openKaloLumen` on macOS, `$XDG_DATA_HOME/openkalolumen` on Linux (defaulting to `~/.local/share/openkalolumen`), and `%LOCALAPPDATA%/openKaloLumen` on Windows.

| Resource | CLI option | Environment variable |
| --- | --- | --- |
| Application data | `--data-dir` | `OPENKALOLUMEN_HOME` |
| Segmentation model | `--model` | `OPENKALOLUMEN_MODEL` |
| Segmentation interpreter | `--segmentation-python` | `OPENKALOLUMEN_SEGMENTATION_PYTHON` |

CLI options take precedence over environment variables. The default model directory is `models/NV-Segment-CTMR` beneath the application data directory. The default inference interpreter is the current Python executable; packaged applications require an explicitly configured external Python interpreter. Application resources do not need to reside in a source checkout.

## Optional local segmentation

Obtain a compatible model directory from [NVIDIA NV-Segment-CTMR](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR). It must contain `metadata.json` and `vista3d_pretrained_model/model.pt`.

**Model weights have separate noncommercial terms and are not covered by this project's Apache-2.0 license.** See [third-party resources and licenses](THIRD_PARTY_NOTICES.md).

```sh
scripts/setup-segmentation.sh
export OPENKALOLUMEN_SEGMENTATION_PYTHON="$PWD/.venv-segmentation/bin/python"
export OPENKALOLUMEN_MODEL="/path/to/NV-Segment-CTMR"
uv run openkalolumen
```

Enter class IDs from the model metadata, such as heart `115` and aorta `6`. IDs are passed directly to the model without substitutions. FP32 inference supports CPU and MPS. Automatic mode validates an actual-weight forward pass before selecting MPS and retries on CPU if inference fails. Nearest-neighbor inverse transforms restore the original physical grid. Empty segmentations are rejected.

The standalone worker can also be invoked directly:

```sh
.venv-segmentation/bin/python src/openkalolumen/segmentation/worker.py \
  --image /path/to/image.nii.gz --model "$OPENKALOLUMEN_MODEL" \
  --classes 115,6 --device auto --output /path/to/results
```

`requirements-segmentation.lock` pins the isolated inference environment. `uv.lock` pins the main application, which does not install PyTorch or model dependencies.

## Development and builds

```sh
uv sync --locked --group dev
uv run ruff check src tests scripts packaging
uv run ruff format --check src tests scripts packaging
uv run pytest -q
uv run python scripts/smoke-desktop.py
uv build
uv run python scripts/check-distribution.py
scripts/build-app.sh
```

The macOS desktop smoke test uses a real Qt/VTK window and synthetic data to check finite geometry, clipping, MPR, scene round trips, STL export, and screenshots. Generated results go to the ignored `outputs/` directory. The build produces `dist/openKaloLumen.app` with an ad-hoc signature; Developer ID signing and Apple notarization are not configured.

GitHub Actions check formatting, lint, tests, clean wheel installation, and distribution contents on PRs and `main` pushes. macOS also runs the desktop smoke test. A version-matched `v*` tag triggers validation, application ZIP/wheel/source builds, a SHA-256 manifest, and a draft GitHub Release.

See the [architecture](docs/architecture.md), [contribution guide](CONTRIBUTING.md), and [release process](docs/releasing.md).

## License

Project source is licensed under [Apache License 2.0](LICENSE), with attribution in [NOTICE](NOTICE). Qt, other dependencies, sample images, and models retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
