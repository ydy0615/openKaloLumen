from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from openkalolumen.config import AppPaths
from openkalolumen.errors import Cancelled

SAMPLES = {
    "CTACardio": {
        "file": "CTA-cardio.nrrd",
        "title": "Cardiac CTA",
        "sha256": "3b0d4eb1a7d8ebb0c5a89cc0504640f76a030b4e869e33ff34c564c3d3b88ad2",
    },
    "CTChest": {
        "file": "CT-chest.nrrd",
        "title": "Chest CT",
        "sha256": "4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e",
    },
    "CTLiver": {
        "file": "CT-liver.nrrd",
        "title": "Contrast-enhanced abdominal CT",
        "sha256": "e16eae0ae6fefa858c5c11e58f0f1bb81834d81b7102e021571056324ef6f37e",
    },
    "PDAHeart": {
        "file": "PDA_CT.nrrd",
        "title": "High-resolution cardiac CTA · SlicerHeart PDA",
        "sha256": "00dfd092e10fcddc009671361b808da9324a3f122e72c27dc50d280f4e7ef242",
        "url": "https://github.com/SlicerHeart/SlicerHeart/releases/download/TestingData/PDA_CT.nrrd",
        "definition": "https://github.com/SlicerHeart/SlicerHeart/blob/master/ValveAnnulusAnalysis/ValveAnnulusAnalysis.py",
    },
}
SOURCE = "https://raw.githubusercontent.com/Slicer/Slicer/main/Modules/Scripted/SampleData/SampleData.py"
BASE = "https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/"


def default_data_dir():
    """Return the configured sample cache directory."""
    return AppPaths.resolve().data / "samples"


def checksum(path):
    """Return a file SHA-256 digest using bounded-memory reads."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_sample(name, directory=None, progress=None, cancelled=lambda: False):
    """Download a public sample with checksum validation and provenance.

    Args:
        name: Key in SAMPLES.
        directory: Destination directory, or None for the application sample cache.
        progress: Optional callback accepting a percentage and status message.
        cancelled: Callback reporting cooperative cancellation.

    Returns:
        The path to the validated image.

    Raises:
        ValueError: The downloaded checksum differs from the sample definition.
        Cancelled: The cancellation callback returns True.
    """
    item = SAMPLES[name]
    directory = Path(directory or default_data_dir())
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / item["file"]
    if destination.exists() and checksum(destination) == item["sha256"]:
        return str(destination)
    url = item.get("url", BASE + item["sha256"])
    request = urllib.request.Request(
        url, headers={"User-Agent": "openKaloLumen/0.1 educational research"}
    )
    temporary = destination.with_suffix(".download")
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as f,
        ):
            total = int(response.headers.get("Content-Length", 0))
            received = 0
            while block := response.read(1024 * 1024):
                if cancelled():
                    raise Cancelled()
                f.write(block)
                received += len(block)
                if progress:
                    progress(
                        int(90 * received / total) if total else 0,
                        f"Downloading {item['title']} · {received / 1024**2:.1f} MiB",
                    )
        if checksum(temporary) != item["sha256"]:
            raise ValueError(
                "Sample SHA-256 verification failed; the file was not enabled."
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    destination.with_suffix(".source.json").write_text(
        json.dumps(
            {
                **item,
                "url": url,
                "definition": item.get("definition", SOURCE),
                "usage": "Public 3D Slicer sample; use under the source terms. Not redistributed with the application.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return str(destination)


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sample", choices=list(SAMPLES) + ["all"], default="all", nargs="?"
    )
    parser.add_argument("--directory")
    args = parser.parse_args()
    for name in SAMPLES if args.sample == "all" else [args.sample]:
        print(
            download_sample(
                name, args.directory, lambda n, s: print(f"{n}% {s}", flush=True)
            )
        )


if __name__ == "__main__":
    main()
