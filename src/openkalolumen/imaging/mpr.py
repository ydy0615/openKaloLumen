from __future__ import annotations

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

BASES = {
    "axial": (np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),
    "coronal": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    "sagittal": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
}


def plane_basis(normal):
    """Return orthonormal in-plane vectors for a physical plane normal."""
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    reference = (
        np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(reference, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def reslice(
    volume, point, u, v, resolution=320, extent=None, array=None, image_override=None
):
    """Sample a physical plane through a scalar image or label grid.

    Args:
        volume: Volume providing physical geometry and scalar values.
        point: Plane center in LPS millimeters.
        u: Horizontal in-plane direction in LPS coordinates.
        v: Vertical in-plane direction in LPS coordinates.
        resolution: Number of pixels on each image axis.
        extent: Physical image width in millimeters, or None to fit the volume.
        array: Optional label array sampled with nearest-neighbor interpolation.
        image_override: Optional VTK scalar field for display filtering.

    Returns:
        A vertically flipped array and its pixel spacing in millimeters.
    """
    image = (
        image_override
        if image_override is not None
        else (volume.to_vtk() if array is None else None)
    )
    if array is not None:
        from openkalolumen.render.geometry import _image_for_array

        image = _image_for_array(volume, array)
    u, v = np.array(u, dtype=float, copy=True), np.array(v, dtype=float, copy=True)
    u /= np.linalg.norm(u)
    v -= u * (u @ v)
    v /= np.linalg.norm(v)
    n = np.cross(u, v)
    inverse_direction = np.asarray(volume.direction).reshape(3, 3).T
    axes = vtk.vtkMatrix4x4()
    for j, vector in enumerate((u, v, n)):
        vector = inverse_direction @ vector
        for i in range(3):
            axes.SetElement(i, j, float(vector[i]))
    origin = inverse_direction @ (np.asarray(point) - volume.origin)
    for i in range(3):
        axes.SetElement(i, 3, float(origin[i]))
    spans = volume.bounds[1] - volume.bounds[0]
    extent = float(extent or max(np.abs(u) @ spans, np.abs(v) @ spans) * 1.03)
    pixels = int(resolution)
    spacing = extent / pixels
    sampler = vtk.vtkImageReslice()
    sampler.SetInputData(image)
    sampler.SetResliceAxes(axes)
    sampler.SetOutputDimensionality(2)
    sampler.SetOutputExtent(0, pixels - 1, 0, pixels - 1, 0, 0)
    sampler.SetOutputSpacing(spacing, spacing, 1)
    sampler.SetOutputOrigin(-(pixels - 1) / 2 * spacing, -(pixels - 1) / 2 * spacing, 0)
    sampler.SetBackgroundLevel(image.GetScalarRange()[0] if array is None else 0)
    if array is None:
        sampler.SetInterpolationModeToLinear()
        sampler.SetOutputScalarType(vtk.VTK_FLOAT)
    else:
        sampler.SetInterpolationModeToNearestNeighbor()
    sampler.Update()
    result = vtk_to_numpy(sampler.GetOutput().GetPointData().GetScalars()).reshape(
        pixels, pixels
    )
    return np.flipud(result).copy(), spacing


def display_slice(array, level, width, labels=None):
    """Map scalar intensities to RGB using a window and optional label overlay."""
    rgb = np.repeat(
        (np.clip((array - (level - width / 2)) / max(width, 1), 0, 1) * 255).astype(
            np.uint8
        )[..., None],
        3,
        axis=-1,
    )
    if labels is not None:
        palette = np.array(
            [
                [0, 0, 0],
                [255, 125, 100],
                [100, 190, 255],
                [200, 170, 250],
                [110, 240, 190],
                [255, 220, 100],
            ],
            dtype=float,
        )
        mask = labels > 0
        rgb[mask] = (
            rgb[mask] * 0.5 + palette[labels[mask] % len(palette)] * 0.5
        ).astype(np.uint8)
    return np.ascontiguousarray(rgb)
