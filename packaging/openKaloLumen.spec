"""Build the macOS desktop bundle with external inference resources."""

from importlib.metadata import distributions
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

project_dir = Path(SPECPATH).parent
runtime_names = {
    "numpy",
    "scipy",
    "pyside6-essentials",
    "shiboken6",
    "vtk",
    "simpleitk",
    "pillow",
    "matplotlib",
    "contourpy",
    "cycler",
    "fonttools",
    "kiwisolver",
    "packaging",
    "pyparsing",
    "python-dateutil",
    "six",
}
data_files = collect_data_files("vtkmodules")
for distribution in distributions():
    if distribution.metadata["Name"].lower().replace("_", "-") in runtime_names:
        data_files += copy_metadata(distribution.metadata["Name"])
data_files += [
    (str(project_dir / name), "licenses")
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")
]
data_files += [
    (
        str(project_dir / "src/openkalolumen/segmentation/worker.py"),
        "openkalolumen/segmentation",
    )
]
data_files += [
    (str(path), "licenses/Qt")
    for path in sorted((project_dir / "packaging/licenses").glob("*"))
]

analysis = Analysis(
    [str(project_dir / "packaging/desktop_entry.py")],
    pathex=[str(project_dir / "src")],
    datas=data_files,
    hiddenimports=collect_submodules("vtkmodules"),
    excludes=[
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebEngineQuick",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtQuickWidgets",
        "torch",
        "monai",
    ],
)
archive = PYZ(analysis.pure)
executable = EXE(
    archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="openKaloLumen",
    console=False,
    upx=False,
)
collection = COLLECT(
    executable, analysis.binaries, analysis.datas, name="openKaloLumen", upx=False
)
app = BUNDLE(
    collection, name="openKaloLumen.app", bundle_identifier="org.openkalolumen.desktop"
)
