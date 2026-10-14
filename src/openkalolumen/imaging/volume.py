"""Image IO and one explicit physical coordinate convention: LPS millimetres."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np
import SimpleITK as sitk
import vtk
from scipy import ndimage
from vtk.util.numpy_support import numpy_to_vtk


@dataclass
class VolumeData:
    """A scalar voxel grid with LPS geometry and source metadata."""

    array: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]
    source: str = ""
    modality: str = "CT"
    hu_calibrated: bool = True
    name: str = "Image"
    fingerprint: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.array = np.ascontiguousarray(self.array)
        if self.array.ndim != 3 or min(self.array.shape) < 2:
            raise ValueError(
                "Only static 3D scalar images with at least two voxels per axis are supported."
            )
        if not np.isfinite(self.array).all():
            raise ValueError("The image contains NaN or infinite values.")
        if len(self.spacing) != 3 or min(self.spacing) <= 0:
            raise ValueError("Voxel spacing must contain three positive values.")
        direction = np.asarray(self.direction).reshape(3, 3)
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
            raise ValueError(
                "The image direction matrix is not orthogonal. Resample onto a regular grid first."
            )
        if not self.fingerprint:
            h = sha256()
            h.update(memoryview(self.array).cast("B"))
            h.update(json.dumps([self.spacing, self.origin, self.direction]).encode())
            self.fingerprint = h.hexdigest()

    @property
    def shape_xyz(self):
        """Return voxel counts in x, y, z order."""
        return np.array(self.array.shape[::-1], dtype=int)

    @property
    def index_to_lps(self):
        """Return the homogeneous transform from voxel indices to LPS millimeters."""
        affine = np.eye(4)
        affine[:3, :3] = np.asarray(self.direction).reshape(3, 3) @ np.diag(
            self.spacing
        )
        affine[:3, 3] = self.origin
        return affine

    def to_world(self, xyz):
        """Transform x, y, z voxel indices to LPS coordinates."""
        return np.asarray(xyz) @ self.index_to_lps[:3, :3].T + self.origin

    def to_index(self, world):
        """Transform LPS coordinates to continuous x, y, z voxel indices."""
        return (np.asarray(world) - self.origin) @ np.linalg.inv(
            self.index_to_lps[:3, :3]
        ).T

    @property
    def center(self):
        """Return the center of the voxel grid in LPS millimeters."""
        return self.to_world((self.shape_xyz - 1) / 2)

    @property
    def bounds(self):
        """Return lower and upper physical bounds of the voxel-center grid."""
        corners = self.to_world(
            list(product(*[(0, int(n) - 1) for n in self.shape_xyz]))
        )
        return np.stack([corners.min(0), corners.max(0)])

    @property
    def scalar_range(self):
        """Return the minimum and maximum original scalar values."""
        return float(self.array.min()), float(self.array.max())

    def to_sitk(self):
        """Return a SimpleITK image preserving physical geometry."""
        image = sitk.GetImageFromArray(self.array)
        image.SetSpacing(self.spacing)
        image.SetOrigin(self.origin)
        image.SetDirection(self.direction)
        return image

    def to_vtk(self):
        """Return an axis-aligned VTK scalar grid for direction-aware rendering."""
        image = vtk.vtkImageData()
        image.SetDimensions(*map(int, self.shape_xyz))
        image.SetSpacing(*self.spacing)
        image.SetOrigin(0, 0, 0)
        scalars = numpy_to_vtk(self.array.ravel(), deep=True)
        scalars.SetName("intensity")
        image.GetPointData().SetScalars(scalars)
        return image

    def actor_matrix(self):
        """Return the VTK transform from local millimeters to LPS coordinates."""
        m = vtk.vtkMatrix4x4()
        d = np.asarray(self.direction).reshape(3, 3)
        for i in range(3):
            for j in range(3):
                m.SetElement(i, j, float(d[i, j]))
            m.SetElement(i, 3, float(self.origin[i]))
        return m


def from_sitk(image, source="", modality="CT", hu_calibrated=True, name=""):
    """Construct a volume from a SimpleITK image and source metadata."""
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(
            "Only static 3D scalar images are supported. Convert multi-frame, 4D, or vector data first."
        )
    return VolumeData(
        sitk.GetArrayFromImage(image),
        image.GetSpacing(),
        image.GetOrigin(),
        image.GetDirection(),
        source,
        modality,
        hu_calibrated,
        name or Path(source).name or "Image",
    )


def dicom_series(directory: str) -> list[dict]:
    """List scalar DICOM series with their physical slice positions."""
    candidates = []
    root = Path(directory)
    folders = [root] + sorted(
        p for p in root.rglob("*") if p.is_dir() and not p.name.startswith(".")
    )
    for folder in folders:
        for uid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)) or []:
            files = list(
                sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), uid)
            )
            if not files:
                continue
            reader = sitk.ImageFileReader()
            reader.SetFileName(files[0])
            reader.ReadImageInformation()
            description = (
                reader.GetMetaData("0008|103e")
                if reader.HasMetaDataKey("0008|103e")
                else "DICOM"
            )
            modality = (
                reader.GetMetaData("0008|0060").strip()
                if reader.HasMetaDataKey("0008|0060")
                else "UNKNOWN"
            )
            candidates.append(
                {
                    "uid": uid,
                    "files": files,
                    "description": description,
                    "modality": modality,
                }
            )
    return candidates


def read_dicom(series: dict, progress: Callable | None = None) -> VolumeData:
    """Load an ordered DICOM series and apply intensity calibration once."""
    positions, orientation, files = [], None, []
    for i, file in enumerate(series["files"]):
        r = sitk.ImageFileReader()
        r.SetFileName(file)
        r.ReadImageInformation()
        if r.HasMetaDataKey("0028|0008") and int(r.GetMetaData("0028|0008")) > 1:
            raise ValueError(
                "Multi-frame DICOM is not supported. Convert it to static NIfTI first."
            )
        if not all(r.HasMetaDataKey(k) for k in ("0020|0032", "0020|0037")):
            raise ValueError(
                "DICOM slice position or orientation is missing; a reliable 3D coordinate system cannot be established."
            )
        current = np.fromstring(r.GetMetaData("0020|0037"), sep="\\")
        if orientation is None:
            orientation = current
        if not np.allclose(orientation, current, atol=1e-4):
            raise ValueError(
                "Slice orientations differ within the series; it cannot form a regular volume."
            )
        position = np.fromstring(r.GetMetaData("0020|0032"), sep="\\")
        positions.append(position)
        files.append(file)
        if progress:
            progress(
                int(40 * (i + 1) / len(series["files"])), "Checking DICOM geometry"
            )
    normal = np.cross(orientation[:3], orientation[3:])
    order = np.argsort(np.asarray(positions) @ normal)
    positions = np.asarray(positions)[order]
    distances = np.diff(positions @ normal)
    if len(distances) == 0 or distances.min() <= 1e-3:
        raise ValueError(
            "The series has duplicate slice positions or fewer than two slices."
        )
    if not np.allclose(
        distances,
        np.median(distances),
        atol=max(0.05, float(np.median(distances)) * 0.02),
    ):
        raise ValueError(
            "The series has irregular spacing or missing slices. Resample it first."
        )
    drift = (
        positions - positions[0] - np.outer((positions - positions[0]) @ normal, normal)
    )
    if np.max(np.linalg.norm(drift, axis=1)) > 0.1:
        raise ValueError(
            "The series has gantry tilt or in-plane offsets. Correct the tilt first."
        )
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([files[int(i)] for i in order])
    reader.MetaDataDictionaryArrayUpdateOn()
    image = reader.Execute()
    modality = series["modality"]
    result = from_sitk(
        image,
        str(Path(files[0]).parent),
        modality,
        modality == "CT",
        series["description"],
    )
    result.metadata = {"series_uid": series["uid"]}
    return result


def load_volume(
    path: str, modality="CT", calibrated=True, series_uid=None
) -> VolumeData:
    """Read a static scalar image or selected DICOM series."""
    path = str(Path(path).resolve())
    if Path(path).is_dir():
        series = dicom_series(path)
        if series_uid:
            series = [s for s in series if s["uid"] == series_uid]
        if len(series) != 1:
            raise ValueError(
                f"The folder contains {len(series)} series. Select one in the interface."
            )
        return read_dicom(series[0])
    return from_sitk(
        sitk.ReadImage(path), path, modality, calibrated and modality == "CT"
    )


def save_volume(volume: VolumeData, path: str):
    """Write a volume with its original physical geometry."""
    sitk.WriteImage(volume.to_sitk(), str(path), True)


def aligned_labels(path: str, volume: VolumeData) -> np.ndarray:
    """Read labels and align them to a volume using nearest-neighbor sampling."""
    labels = sitk.ReadImage(path)
    if labels.GetDimension() != 3 or labels.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Labels must be a 3D scalar image.")
    raw = sitk.GetArrayViewFromImage(labels)
    if raw.min() < 0 or raw.max() > 65535 or not np.allclose(raw, np.round(raw)):
        raise ValueError("Labels must contain integer class IDs between 0 and 65535.")
    labels = sitk.Resample(
        labels,
        volume.to_sitk(),
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt16,
    )
    result = sitk.GetArrayFromImage(labels)
    if not result.any():
        raise ValueError(
            "Resampled labels are empty. Check that the labels use the same image and physical space."
        )
    return result


def connected_region(
    volume: VolumeData, point, lower: float, upper: float, visible_mask=None
):
    """Find an intensity-connected component near a physical seed.

    Args:
        volume: Original scalar voxel grid.
        point: Seed point in LPS millimeters.
        lower: Inclusive lower intensity bound.
        upper: Inclusive upper intensity bound.
        visible_mask: Optional boolean mask limiting eligible voxels.

    Returns:
        A boolean mask on the original voxel grid.

    Raises:
        ValueError: The seed has no eligible voxel in its immediate cell neighborhood.

    A picked surface point may lie between voxel centers. The seed search stays
    within one neighboring voxel and never jumps to a distant component.
    """
    index = np.rint(volume.to_index(point)).astype(int)
    if np.any(index < 0) or np.any(index >= volume.shape_xyz):
        raise ValueError("The seed point is outside the image.")
    mask = (volume.array >= lower) & (volume.array <= upper)
    if visible_mask is not None:
        mask &= visible_mask
    components, _ = ndimage.label(mask)
    label = components[tuple(index[::-1])]
    if label == 0:
        lower_index = np.maximum(index - 1, 0)
        upper_index = np.minimum(index + 2, volume.shape_xyz)
        slices = tuple(
            slice(int(a), int(b)) for a, b in zip(lower_index[::-1], upper_index[::-1])
        )
        candidates = np.argwhere(mask[slices]) + lower_index[::-1]
        if not len(candidates):
            raise ValueError(
                "No voxel near the seed point is within the requested intensity range."
            )
        worlds = volume.to_world(candidates[:, ::-1])
        closest = candidates[np.argmin(np.linalg.norm(worlds - point, axis=1))]
        label = components[tuple(closest)]
    return components == label


def synthetic_volume(size=112) -> VolumeData:
    """Continuous hollow sphere, crossing tube, and bone-like rod in known HU."""
    spacing = (1.0, 1.2, 1.4)
    z, y, x = np.indices((size, size, size), dtype=np.float32)
    x = (x - (size - 1) / 2) * spacing[0]
    y = (y - (size - 1) / 2) * spacing[1]
    z = (z - (size - 1) / 2) * spacing[2]
    radius = np.sqrt(x * x + y * y + z * z)

    def sigmoid(a):
        return 1 / (1 + np.exp(np.clip(-a, -50, 50)))

    wall = sigmoid((size * 0.34 - radius) / 0.65) * sigmoid(
        (radius - size * 0.25) / 0.65
    )
    tube_r = np.sqrt((x - size * 0.38) ** 2 + y * y)
    tube = (
        sigmoid((size * 0.09 - tube_r) / 0.5)
        * sigmoid((tube_r - size * 0.045) / 0.5)
        * sigmoid((size * 0.48 - np.abs(z)) / 0.7)
    )
    rod = sigmoid(
        (size * 0.045 - np.sqrt((x + size * 0.38) ** 2 + (y - size * 0.2) ** 2)) / 0.5
    ) * sigmoid((size * 0.42 - np.abs(z)) / 0.5)
    intensity = -1000 + 1100 * np.maximum(wall, tube)
    intensity = np.maximum(intensity, -1000 + 2050 * rod)
    return VolumeData(
        intensity.astype(np.float32),
        spacing,
        tuple(-((size - 1) / 2) * np.array(spacing)),
        tuple(np.eye(3).ravel()),
        "synthetic:hollow",
        "CT",
        True,
        "Synthetic lumen · geometry validation",
    )
