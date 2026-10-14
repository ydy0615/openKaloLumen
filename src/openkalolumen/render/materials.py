"""Deterministic studio illumination; no image synthesis or geometry modification."""

from __future__ import annotations

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def studio_environment():
    """Linear HDR radiance of broad softboxes, sampled on an equirectangular sphere."""
    h, w = 128, 256
    theta = (np.arange(h) + 0.5) * np.pi / h
    phi = (np.arange(w) + 0.5) * 2 * np.pi / w
    directions = np.stack(
        np.broadcast_arrays(
            np.sin(theta[:, None]) * np.cos(phi),
            np.cos(theta[:, None]) + np.zeros((1, w)),
            np.sin(theta[:, None]) * np.sin(phi),
        ),
        axis=-1,
    )
    radiance = np.broadcast_to(np.array([0.055, 0.065, 0.08]), (h, w, 3)).copy()
    for direction, width, strength, color in [
        ((-0.5, 0.8, -1), 0.36, 2.0, (1.0, 0.94, 0.88)),
        ((1, 0.4, 0.1), 0.5, 0.65, (0.83, 0.9, 1.0)),
        ((-0.2, 0.7, 1), 0.35, 1.2, (1.0, 0.97, 0.93)),
    ]:
        d = np.asarray(direction, dtype=float)
        d /= np.linalg.norm(d)
        angle = np.arccos(np.clip(directions @ d, -1, 1))
        radiance += (
            np.exp(-0.5 * (angle / width) ** 2)[..., None]
            * strength
            * np.asarray(color)
        )
    image = vtk.vtkImageData()
    image.SetDimensions(w, h, 1)
    image.GetPointData().SetScalars(
        numpy_to_vtk(radiance.astype(np.float32).reshape(-1, 3), deep=True)
    )
    texture = vtk.vtkTexture()
    texture.SetInputData(image)
    texture.SetColorModeToDirectScalars()
    texture.InterpolateOn()
    texture.MipmapOn()
    texture.UseSRGBColorSpaceOff()
    return texture


def tissue_property(layer, inside=False):
    """Create a PBR property from a layer and its boundary color."""
    prop = vtk.vtkProperty()
    prop.SetInterpolationToPBR()
    prop.SetMetallic(0.0)
    prop.SetBaseIOR(1.4)
    prop.SetRoughness(layer.roughness)
    prop.SetCoatStrength(layer.coat)
    prop.SetCoatRoughness(0.32)
    prop.SetCoatIOR(1.33)
    color = (
        layer.interior_color
        if inside and layer.interior_color is not None
        else layer.color
    )
    prop.SetColor(*color)
    prop.SetOpacity(layer.opacity)
    prop.BackfaceCullingOff()
    prop.FrontfaceCullingOff()
    return prop


def apply_material(actor, layer, layers=()):
    """Apply material colors without mutating cached geometry.

    Args:
        actor: VTK actor displaying a material mesh.
        layer: Layer supplying exterior and interior appearance.
        layers: Adjacent material layers used to hide opaque internal contacts.

    Boundary identity determines color independently of camera orientation.
    Opaque tissue contacts are internal to the material union; the closed meshes
    remain available in the cache for sectioning and export.
    """
    actor.SetProperty(tissue_property(layer))
    actor.SetBackfaceProperty(tissue_property(layer, layer.kind != "solid_band"))
    if layer.kind == "solid_band":
        actor.GetProperty().BackfaceCullingOn()
        actor.SetBackfaceProperty(None)
    mapper = actor.GetMapper()
    mesh = mapper.GetInput()
    if (
        layer.kind == "solid_band"
        and mesh is not None
        and mesh.GetPointData().GetArray("BoundaryHU") is not None
    ):
        hu = vtk_to_numpy(mesh.GetPointData().GetArray("BoundaryHU"))
        weight = np.clip(
            (hu - layer.threshold) / max(layer.upper - layer.threshold, 1), 0, 1
        )
        weight = weight * weight * (3 - 2 * weight)
        outer = np.asarray(layer.color)
        inner = np.asarray(layer.interior_color or layer.color)
        rgb = np.clip(
            (outer[None, :] * (1 - weight[:, None]) + inner[None, :] * weight[:, None])
            * 255,
            0,
            255,
        ).astype(np.uint8)
        colors = numpy_to_vtk(rgb, deep=True)
        colors.SetName("TissueColor")

        local = vtk.vtkPolyData()
        local.ShallowCopy(mesh)
        local.GetPointData().SetScalars(colors)

        contacts = []
        for other in layers:
            if (
                other is layer
                or other.kind != "solid_band"
                or not other.visible
                or other.opacity != 1
                or layer.opacity != 1
            ):
                continue
            if other.smoothing_mm != layer.smoothing_mm:
                continue
            if abs(other.threshold - layer.upper) < 1e-6:
                contacts.append(layer.upper)
            if abs(other.upper - layer.threshold) < 1e-6:
                contacts.append(layer.threshold)
        if contacts:
            cells = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3)
            keep = np.ones(len(cells), dtype=bool)
            for value in contacts:
                keep &= ~np.all(np.abs(hu[cells] - value) < 0.02, axis=1)
            from vtk.util.numpy_support import numpy_to_vtkIdTypeArray

            polys = vtk.vtkCellArray()
            polys.SetData(
                numpy_to_vtkIdTypeArray(
                    np.arange(int(keep.sum()) + 1, dtype=np.int64) * 3, deep=True
                ),
                numpy_to_vtkIdTypeArray(
                    np.ascontiguousarray(cells[keep]).ravel().astype(np.int64),
                    deep=True,
                ),
            )
            local.SetPolys(polys)
        mapper.SetInputData(local)
        mapper.SetColorModeToDirectScalars()
        mapper.ScalarVisibilityOn()
        actor.GetProperty().SetColor(1, 1, 1)
    else:
        mapper.ScalarVisibilityOff()
