"""Resolve writable application data and optional inference resources."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def default_data_dir() -> Path:
    """Return the platform's per-user application data directory."""
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/openKaloLumen"
    if sys.platform == "win32":
        return (
            Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
            / "openKaloLumen"
        )
    return (
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
        / "openkalolumen"
    )


@dataclass(frozen=True)
class AppPaths:
    """Locations for writable data, model files, and the inference interpreter.

    Attributes:
        data: Writable per-user application directory.
        model: External NV-Segment-CTMR directory containing metadata and weights.
        segmentation_python: Python executable with the inference dependencies.
    """

    data: Path
    model: Path
    segmentation_python: Path

    @classmethod
    def resolve(cls, data=None, model=None, segmentation_python=None):
        """Resolve explicit paths, environment variables, and per-user defaults.

        Args:
            data: Application directory; overrides OPENKALOLUMEN_HOME.
            model: Model directory; overrides OPENKALOLUMEN_MODEL.
            segmentation_python: Interpreter; overrides OPENKALOLUMEN_SEGMENTATION_PYTHON.

        Returns:
            Absolute application paths without creating directories.
        """
        directory = (
            Path(data or os.environ.get("OPENKALOLUMEN_HOME") or default_data_dir())
            .expanduser()
            .resolve()
        )
        model_path = (
            Path(
                model
                or os.environ.get("OPENKALOLUMEN_MODEL")
                or directory / "models/NV-Segment-CTMR"
            )
            .expanduser()
            .resolve()
        )
        python = (
            Path(
                segmentation_python
                or os.environ.get("OPENKALOLUMEN_SEGMENTATION_PYTHON")
                or sys.executable
            )
            .expanduser()
            .resolve()
        )
        return cls(directory, model_path, python)
