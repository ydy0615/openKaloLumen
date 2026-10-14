#!/bin/sh
set -eu
project_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$project_dir"
if [ "$(uname -s)" != Darwin ]; then
  echo "The desktop bundle currently supports macOS only." >&2
  exit 1
fi
uv sync --locked --group build
uv run --no-sync python -m PyInstaller --noconfirm --clean packaging/openKaloLumen.spec
