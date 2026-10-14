"""Verify that release archives contain only public project files."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

REQUIRED_NOTICES = {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"}
PRIVATE_PARTS = {
    "data",
    "outputs",
    "models",
    "external",
    "MatLat",
    "NV-Reason-CT",
    "NV-Segment-CTMR",
    "__pycache__",
    ".git",
    ".venv",
    ".venv-segmentation",
}


def validate(path):
    """Reject private resources and require license texts in a release archive.

    Args:
        path: Wheel or source-distribution archive to inspect.

    Raises:
        ValueError: Private files, symlinks, or required-notice omissions exist.
    """
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    else:
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            if any(member.issym() or member.islnk() for member in members):
                raise ValueError(f"Archive contains symlinks: {path}")
            names = [member.name for member in members]
    for name in names:
        parts = Path(name).parts
        if any(part in PRIVATE_PARTS or part.startswith(".env") for part in parts):
            raise ValueError(f"Private resource in {path}: {name}")
        if name.endswith((".pt", ".nrrd", ".nii", ".nii.gz", ".pyc")):
            raise ValueError(f"Binary data or model in {path}: {name}")
    found = {Path(name).name for name in names}
    if not REQUIRED_NOTICES <= found:
        raise ValueError(f"Missing notices in {path}: {REQUIRED_NOTICES - found}")
    print(f"Verified {path.name}: {len(names)} entries, required notices present")


def main():
    """Check every wheel and source distribution in the selected directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("dist"))
    args = parser.parse_args()
    paths = list(args.directory.glob("*.whl")) + list(args.directory.glob("*.tar.gz"))
    if not any(path.suffix == ".whl" for path in paths) or not any(
        path.name.endswith(".tar.gz") for path in paths
    ):
        raise ValueError(
            "Build both a wheel and a source distribution before checking."
        )
    for path in paths:
        validate(path)


if __name__ == "__main__":
    main()
