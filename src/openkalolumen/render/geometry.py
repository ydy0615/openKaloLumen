"""Surface/finite-material geometry in physical coordinates; display filters explicit."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from threading import Lock

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from openkalolumen.errors import Cancelled


class MeshCache:
    """A thread-safe least-recently-used cache bounded by mesh memory."""

    def __init__(self, max_bytes=2 * 1024**3):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.lock = Lock()

    def get(self, key):
        """Return a cached mesh and mark it recently used, or return None."""
        with self.lock:
            if key not in self.entries:
                return None
            self.entries.move_to_end(key)
            return self.entries[key][0]

    def put(self, key, mesh):
        """Cache a mesh and evict older entries until the memory budget is met."""
        size = int(mesh.GetActualMemorySize()) * 1024
        with self.lock:
            if key in self.entries:
                self.bytes -= self.entries.pop(key)[1]
            self.entries[key] = (mesh, size)
            self.bytes += size
            while self.bytes > self.max_bytes and len(self.entries) > 1:
                _, (_, n) = self.entries.popitem(last=False)
                self.bytes -= n


MESH_CACHE = MeshCache()
FIELD_CACHE = OrderedDict()
FIELD_LOCK = Lock()
SOLID_FIELDS = MeshCache(max_bytes=1024**3)


def display_field(volume, smoothing_mm=0.0):
    """Optional physical-mm Gaussian display field; original HU data stays untouched."""
    if smoothing_mm <= 0:
        return volume.to_vtk()
    key = (volume.fingerprint, float(smoothing_mm))
    with FIELD_LOCK:
        if key in FIELD_CACHE:
            FIELD_CACHE.move_to_end(key)
            return FIELD_CACHE[key]
    cast = vtk.vtkImageCast()
    cast.SetInputData(volume.to_vtk())
    cast.SetOutputScalarTypeToFloat()
    blur = vtk.vtkImageGaussianSmooth()
    blur.SetInputConnection(cast.GetOutputPort())
    blur.SetStandardDeviations(*(smoothing_mm / np.asarray(volume.spacing)))
    blur.SetRadiusFactors(3, 3, 3)
    blur.Update()
    image = vtk.vtkImageData()
    image.DeepCopy(blur.GetOutput())
    with FIELD_LOCK:
        FIELD_CACHE[key] = image
        while (
            sum(v.GetActualMemorySize() * 1024 for v in FIELD_CACHE.values()) > 1024**3
            and len(FIELD_CACHE) > 1
        ):
            FIELD_CACHE.popitem(last=False)
    return image


def geometry_key(volume, layers, mask=None, labels=None):
    """Hash volume identity, layer geometry, and visibility masks."""
    h = sha256(volume.fingerprint.encode())
    h.update(
        json.dumps(
            [
                (
                    layer.kind,
                    layer.threshold,
                    layer.upper,
                    layer.smoothing_mm,
                    layer.label_id,
                    layer.visible,
                )
                for layer in layers
            ]
        ).encode()
    )
    for a in (mask, labels):
        if a is not None:
            h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return h.hexdigest()


def _image_for_array(volume, array):
    image = vtk.vtkImageData()
    image.SetDimensions(*map(int, volume.shape_xyz))
    image.SetSpacing(*volume.spacing)
    image.GetPointData().SetScalars(
        numpy_to_vtk(np.ascontiguousarray(array).ravel(), deep=True)
    )
    return image


def extract_surface(volume, layer, mask=None, labels=None):
    """Extract a scalar boundary, interval, label, or finite material.

    Args:
        volume: Scalar voxel grid supplying geometry and physical intensities.
        layer: Boundary or material definition.
        mask: Optional visibility mask on the original voxel grid.
        labels: Optional integer labels for a semantic layer.

    Returns:
        A VTK polygon mesh in LPS millimeters.

    Raises:
        ValueError: A label layer has no corresponding label data.

    Interval surfaces reverse the upper boundary so its normals point into the
    excluded lumen. Mask clipping does not add artificial caps.
    """
    if layer.kind == "solid_band":
        return extract_solid(volume, layer, mask)
    if layer.label_id is not None:
        if labels is None:
            raise ValueError("This organ layer has no corresponding label data.")
        image = _image_for_array(volume, (labels == layer.label_id).astype(np.uint8))
        contour = vtk.vtkFlyingEdges3D()
        contour.SetInputData(image)
        contour.SetValue(0, 0.5)
    else:
        image = display_field(volume, layer.smoothing_mm)
        contour = vtk.vtkFlyingEdges3D()
        contour.SetInputData(image)
        contour.SetValue(0, layer.threshold)
    contour.ComputeNormalsOn()
    contour.Update()
    poly = contour.GetOutput()
    if (
        layer.kind == "surface_band"
        and layer.upper is not None
        and layer.label_id is None
    ):
        upper = vtk.vtkFlyingEdges3D()
        upper.SetInputData(image)
        upper.SetValue(0, layer.upper)
        upper.ComputeNormalsOn()
        upper.Update()
        reverse = vtk.vtkReverseSense()
        reverse.SetInputData(upper.GetOutput())
        reverse.ReverseCellsOn()
        reverse.ReverseNormalsOn()
        reverse.Update()
        append = vtk.vtkAppendPolyData()
        append.AddInputData(poly)
        append.AddInputData(reverse.GetOutput())
        append.Update()
        poly = append.GetOutput()
    if mask is not None:
        probe = vtk.vtkProbeFilter()
        probe.SetInputData(poly)
        probe.PassPointArraysOn()
        mask_image = _image_for_array(volume, mask.astype(np.float32))
        mask_image.GetPointData().GetScalars().SetName("visibility")
        probe.SetSourceData(mask_image)
        probe.Update()
        probe.GetOutput().GetPointData().SetActiveScalars("visibility")
        clip = vtk.vtkClipPolyData()
        clip.SetInputData(probe.GetOutput())
        clip.SetValue(0.5)
        clip.Update()
        poly = clip.GetOutput()
    transform = vtk.vtkTransform()
    transform.SetMatrix(volume.actor_matrix())
    transformed = vtk.vtkTransformFilter()
    transformed.SetInputData(poly)
    transformed.SetTransform(transform)
    transformed.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(transformed.GetOutput())
    result.GetPointData().SetScalars(None)
    return result


def solid_field(volume, layer, mask=None):
    """Construct the implicit field of a finite material interval.

    Args:
        volume: Scalar volume defining the material.
        layer: Lower and upper material thresholds.
        mask: Optional boolean region of interest.

    Returns:
        A VTK scalar field whose zero contour bounds the retained material.

    A sub-intensity numerical epsilon resolves exact-threshold plateaus and
    prevents coincident or degenerate contour triangles.
    """
    if layer.upper is None or layer.upper <= layer.threshold:
        raise ValueError(
            "Finite tissue requires valid lower and upper intensity limits."
        )
    key = geometry_key(volume, [layer], mask)
    cached = SOLID_FIELDS.get(key)
    if cached is not None:
        return cached
    image = display_field(volume, layer.smoothing_mm)
    hu = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(volume.array.shape)
    span = layer.upper - layer.threshold
    field = np.minimum(hu.astype(np.float32) - layer.threshold, layer.upper - hu)

    field[field == 0] = 1e-3
    if mask is not None:
        np.minimum(field, np.where(mask, span * 0.5, -span * 0.5), out=field)
    field = np.pad(field, 1, constant_values=-span * 0.5)
    solid = vtk.vtkImageData()
    solid.SetDimensions(*field.shape[::-1])
    solid.SetSpacing(*volume.spacing)
    solid.SetOrigin(*(-np.asarray(volume.spacing)))
    solid.GetPointData().SetScalars(numpy_to_vtk(field.ravel(), deep=True))
    SOLID_FIELDS.put(key, solid)
    return solid


def extract_solid(volume, layer, mask=None):
    """One oriented boundary of the finite material, including the ROI envelope.

    No extrusion, dilation or hole filling. Acquisition guard voxels close only
    the data extent; the excluded upper HU interval remains a void.
    """
    solid = solid_field(volume, layer, mask)
    image = display_field(volume, layer.smoothing_mm)
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(solid)
    contour.SetValue(0, 0)
    contour.ComputeNormalsOn()
    contour.Update()

    probe = vtk.vtkProbeFilter()
    probe.SetInputData(contour.GetOutput())
    probe.PassPointArraysOn()
    probe.SetSourceData(image)
    probe.Update()
    poly = probe.GetOutput()
    boundary = vtk.vtkFloatArray()
    boundary.DeepCopy(poly.GetPointData().GetScalars())
    boundary.SetName("BoundaryHU")
    poly.GetPointData().AddArray(boundary)
    poly.GetPointData().SetScalars(None)
    transform = vtk.vtkTransform()
    transform.SetMatrix(volume.actor_matrix())
    transformed = vtk.vtkTransformFilter()
    transformed.SetInputData(poly)
    transformed.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    transformed.SetTransform(transform)
    transformed.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(transformed.GetOutput())
    if np.linalg.det(np.asarray(volume.direction).reshape(3, 3)) < 0:
        reverse = vtk.vtkReverseSense()
        reverse.SetInputData(result)
        reverse.ReverseCellsOn()
        reverse.ReverseNormalsOff()
        reverse.Update()
        result.DeepCopy(reverse.GetOutput())
    return result


def solid_cut(mesh, clip):
    """Intersect an oriented closed material with the retained half-space.

    Clip faces connect outer and inner walls and preserve oriented cavity loops.
    The separate face output is used for its own cut-tissue material.
    """
    plane = vtk.vtkPlane()
    plane.SetOrigin(*clip.origin)
    plane.SetNormal(*clip.effective_normal)
    planes = vtk.vtkPlaneCollection()
    planes.AddItem(plane)
    cutter = vtk.vtkClipClosedSurface()
    cutter.SetInputData(mesh)
    cutter.SetClippingPlanes(planes)
    cutter.PassPointDataOn()
    cutter.SetTolerance(1e-7)
    cutter.GenerateClipFaceOutputOn()
    errors = []
    cutter.AddObserver(vtk.vtkCommand.ErrorEvent, lambda *_: errors.append(True))
    cutter.TriangulationErrorDisplayOn()
    cutter.Update()
    if errors:
        raise ValueError(
            "The clipped material could not be closed reliably. No incomplete mesh was exported; move the clipping plane and retry."
        )

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(cutter.GetOutput())
    clean.ToleranceIsAbsoluteOn()
    clean.SetAbsoluteTolerance(1e-9)
    clean.Update()
    output = vtk.vtkPolyData()
    output.DeepCopy(clean.GetOutput())
    clean_faces = vtk.vtkCleanPolyData()
    clean_faces.SetInputData(cutter.GetClipFaceOutput())
    clean_faces.ToleranceIsAbsoluteOn()
    clean_faces.SetAbsoluteTolerance(1e-5)
    clean_faces.Update()
    faces = vtk.vtkPolyData()
    faces.DeepCopy(clean_faces.GetOutput())
    return output, faces


def validate_solid_export(poly):
    """Validate the coordinates actually representable by STL/PLY serialization.

    CT contact/saddle points or a failed cap can be geometrically coincident
    even when distinct point IDs make the in-memory mesh appear manifold.
    """
    copy = vtk.vtkPolyData()
    copy.ShallowCopy(poly)
    points = vtk.vtkPoints()
    points.SetData(
        numpy_to_vtk(
            vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32), deep=True
        )
    )
    copy.SetPoints(points)
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(copy)
    clean.ToleranceIsAbsoluteOn()
    clean.SetAbsoluteTolerance(0)
    clean.Update()
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(clean.GetOutput())
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.Update()
    count = edges.GetOutput().GetNumberOfCells()
    if count:
        raise ValueError(
            f"The material mesh contains {count} open or nonmanifold edges. No mesh was written; adjust the thresholds or clipping plane, or export an original isosurface."
        )
    result = vtk.vtkPolyData()
    result.DeepCopy(clean.GetOutput())
    return result


def solid_faces(mesh, clip):
    """Triangulate just the oriented section loops, without copying the 3D mesh."""
    plane = vtk.vtkPlane()
    plane.SetOrigin(*clip.origin)
    plane.SetNormal(*clip.effective_normal)
    cutter = vtk.vtkPlaneCutter()
    cutter.SetInputData(mesh)
    cutter.SetPlane(plane)
    cutter.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(cutter.GetOutput())
    clean.ToleranceIsAbsoluteOn()
    clean.SetAbsoluteTolerance(1e-5)
    clean.Update()
    triangulate = vtk.vtkContourTriangulator()
    triangulate.SetInputData(clean.GetOutput())
    triangulate.TriangulationErrorDisplayOff()
    triangulate.Update()
    if triangulate.GetTriangulationError():
        raise ValueError(
            "The material section could not be triangulated reliably. Move the clipping plane slightly."
        )
    faces = vtk.vtkPolyData()
    faces.DeepCopy(triangulate.GetOutput())
    if faces.GetNumberOfCells():
        points = vtk_to_numpy(faces.GetPoints().GetData())
        cells = vtk_to_numpy(faces.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        cross = np.cross(
            points[cells[:, 1]] - points[cells[:, 0]],
            points[cells[:, 2]] - points[cells[:, 0]],
        )
        if np.sum(cross @ clip.effective_normal) > 0:
            reverse = vtk.vtkReverseSense()
            reverse.SetInputData(faces)
            reverse.ReverseCellsOn()
            reverse.Update()
            faces.DeepCopy(reverse.GetOutput())
    return faces


def sampled_solid_faces(volume, layer, mask, clip, resolution=768):
    """Robust cut-face fallback sampled from the SAME positive-material field.

    This local grid never triangulates across negative-valued lumen/air regions.
    Its edge accuracy is limited by the slice grid and CT sampling, explicitly
    distinguished from exact mesh-loop faces in the rendering diagnostics.
    """
    from openkalolumen.imaging.mpr import plane_basis, reslice

    u, v = plane_basis(clip.effective_normal)
    values, spacing = reslice(
        volume,
        clip.origin,
        u,
        v,
        resolution,
        image_override=solid_field(volume, layer, mask),
    )
    image = vtk.vtkImageData()
    image.SetDimensions(resolution, resolution, 1)
    image.SetSpacing(spacing, spacing, 1)
    image.SetOrigin(-(resolution - 1) * spacing / 2, -(resolution - 1) * spacing / 2, 0)
    image.GetPointData().SetScalars(
        numpy_to_vtk(np.flipud(values).copy().ravel(), deep=True)
    )
    geometry = vtk.vtkGeometryFilter()
    geometry.SetInputData(image)
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(geometry.GetOutputPort())
    cut = vtk.vtkClipPolyData()
    cut.SetInputConnection(triangles.GetOutputPort())
    cut.SetValue(0)
    triangulated = vtk.vtkTriangleFilter()
    triangulated.SetInputConnection(cut.GetOutputPort())
    transform = vtk.vtkTransform()
    matrix = vtk.vtkMatrix4x4()
    for j, axis in enumerate((u, v, clip.effective_normal)):
        for i in range(3):
            matrix.SetElement(i, j, float(axis[i]))
    for i in range(3):
        matrix.SetElement(i, 3, float(clip.origin[i]))
    transform.SetMatrix(matrix)
    transformed = vtk.vtkTransformFilter()
    transformed.SetInputConnection(triangulated.GetOutputPort())
    transformed.SetTransform(transform)
    transformed.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    transformed.Update()
    faces = vtk.vtkPolyData()
    faces.DeepCopy(transformed.GetOutput())
    faces.GetPointData().SetScalars(None)
    return faces


def build_surfaces(
    volume, layers, mask=None, labels=None, progress=None, cancelled=lambda: False
):
    """Build visible layer meshes with cache reuse and cooperative cancellation."""
    base = geometry_key(volume, layers, mask, labels)
    content = geometry_key(volume, [], mask, labels)
    output = {}
    visible = [
        (i, layer)
        for i, layer in enumerate(layers)
        if layer.visible and layer.kind in ("isosurface", "surface_band", "solid_band")
    ]
    for count, (i, layer) in enumerate(visible):
        if cancelled():
            raise Cancelled()
        key = (
            content,
            layer.kind,
            layer.threshold,
            layer.upper,
            layer.smoothing_mm,
            layer.label_id,
        )
        mesh = MESH_CACHE.get(key)
        if mesh is None:
            mesh = extract_surface(volume, layer, mask, labels)
            MESH_CACHE.put(key, mesh)
        output[i] = mesh
        if progress:
            progress(
                int(100 * (count + 1) / max(len(visible), 1)),
                f"Generating detailed surfaces · {layer.name}",
            )
    return base, output


def clipped_mesh(mesh, clip):
    """Return a mesh clipped by the physical half-space without capping."""
    if not clip or not clip.enabled:
        return mesh
    plane = vtk.vtkPlane()
    plane.SetOrigin(*clip.origin)
    plane.SetNormal(*clip.effective_normal)
    cutter = vtk.vtkClipPolyData()
    cutter.SetInputData(mesh)
    cutter.SetClipFunction(plane)
    cutter.SetValue(0)
    cutter.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(cutter.GetOutput())
    return result


def export_mesh(path, mesh, layer, volume, clip=None):
    """Write an unfiltered STL or colored PLY in LPS millimeters."""
    path = Path(path)
    if layer.smoothing_mm > 0:
        raise ValueError(
            "Geometry with display smoothing cannot be exported as original CT geometry. Disable smoothing and extract again."
        )
    if mesh.GetNumberOfPoints() == 0:
        raise ValueError(
            "The current threshold or filter produces an empty surface; export is unavailable."
        )
    poly = vtk.vtkPolyData()
    if layer.kind == "solid_band" and clip and clip.enabled:
        poly.DeepCopy(solid_cut(mesh, clip)[0])
    else:
        poly.DeepCopy(clipped_mesh(mesh, clip))
    if poly.GetNumberOfPoints() == 0:
        raise ValueError("The surface is empty after applying the current clip.")
    if layer.kind == "solid_band":
        poly = validate_solid_export(poly)
    if path.suffix.lower() == ".stl":
        writer = vtk.vtkSTLWriter()
        writer.SetFileTypeToBinary()
    elif path.suffix.lower() == ".ply":
        colors = np.tile(
            (np.asarray(layer.color) * 255).astype(np.uint8),
            (poly.GetNumberOfPoints(), 1),
        )
        array = numpy_to_vtk(colors, deep=True)
        array.SetName("Colors")
        poly.GetPointData().SetScalars(array)
        writer = vtk.vtkPLYWriter()
        writer.SetFileTypeToBinary()
        writer.SetArrayName("Colors")
    else:
        raise ValueError("Only STL and PLY are supported.")
    writer.SetInputData(poly)
    writer.SetFileName(str(path))
    if not writer.Write():
        raise OSError("Mesh writing failed.")
    metadata = {
        "unit": "mm",
        "coordinate_system": "LPS",
        "fingerprint": volume.fingerprint,
        "source": volume.source,
        "layer": asdict(layer),
        "clip": asdict(clip) if clip else None,
        "points": poly.GetNumberOfPoints(),
        "triangles": poly.GetNumberOfCells(),
        "smoothing": False,
        "hole_filling": False,
        "material_definition": "finite HU interval intersected with ROI"
        if layer.kind == "solid_band"
        else "threshold boundary",
        "closed_cut_faces": bool(layer.kind == "solid_band" and clip and clip.enabled),
    }
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)
    )
    return metadata
