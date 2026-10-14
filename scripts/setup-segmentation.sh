#!/bin/sh
set -eu
project_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$project_dir"
uv venv --allow-existing --python 3.12 .venv-segmentation
uv pip sync --python .venv-segmentation/bin/python requirements-segmentation.lock
printf 'Set OPENKALOLUMEN_SEGMENTATION_PYTHON=%s/.venv-segmentation/bin/python\n' "$project_dir"
printf 'Set OPENKALOLUMEN_MODEL to a separately downloaded NV-Segment-CTMR directory.\n'
