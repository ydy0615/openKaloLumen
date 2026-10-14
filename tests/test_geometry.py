import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from openkalolumen.imaging.drr import project_drr
from openkalolumen.imaging.mpr import reslice
from openkalolumen.imaging.volume import (
    VolumeData,
    aligned_labels,
    load_volume,
    save_volume,
    synthetic_volume,
)
from openkalolumen.render.geometry import (
    clipped_mesh,
    display_field,
    export_mesh,
    extract_surface,
    geometry_key,
)
from openkalolumen.state import ClipState, Layer, Measurement, read_scene, save_scene


def rotated_volume(array):
    angle = 0.4
    direction = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1.0],
        ]
    )
    return VolumeData(
        array, (1.2, 1.8, 2.4), (17.0, -31.0, 9.0), tuple(direction.ravel())
    )


def test_physical_transform_nifti_roundtrip(tmp_path):
    volume = rotated_volume(np.arange(6 * 7 * 8, dtype=np.int16).reshape(6, 7, 8))
    points = np.array([[0.0, 0.0, 0.0], [3.3, 2.1, 1.7], [7, 6, 5]])
    np.testing.assert_allclose(
        volume.to_index(volume.to_world(points)), points, atol=1e-10
    )
    path = tmp_path / "orientation.nii.gz"
    save_volume(volume, path)
    restored = load_volume(path)
    np.testing.assert_array_equal(volume.array, restored.array)
    np.testing.assert_allclose(volume.index_to_lps, restored.index_to_lps, atol=2e-6)


def test_mpr_rotated_anisotropic_ramp():
    z, y, x = np.indices((25, 25, 25), dtype=np.float32)
    volume = rotated_volume(x + 10 * y + 100 * z)
    direction = np.asarray(volume.direction).reshape(3, 3)
    sampled, spacing = reslice(
        volume, volume.center, direction[:, 0], direction[:, 1], 21, extent=12.6
    )
    coordinates = (np.arange(21) - 10) * spacing
    expected = (
        12 + 120 + 1200 + coordinates[None, :] / 1.2 - 10 * coordinates[:, None] / 1.8
    )
    np.testing.assert_allclose(sampled, expected, atol=0.003)


def test_contour_export_keeps_physical_mm(tmp_path):
    z, y, x = np.indices((9, 10, 11), dtype=np.float32)
    volume = rotated_volume(x)
    layer = Layer("known plane", 4.5, (0.3, 0.7, 0.8))
    mesh = extract_surface(volume, layer)
    xyz = volume.to_index(vtk_to_numpy(mesh.GetPoints().GetData()))
    np.testing.assert_allclose(xyz[:, 0], 4.5, atol=3e-6)
    for extension, reader_class in [
        ("stl", vtk.vtkSTLReader),
        ("ply", vtk.vtkPLYReader),
    ]:
        path = tmp_path / f"plane.{extension}"
        metadata = export_mesh(path, mesh, layer, volume)
        reader = reader_class()
        reader.SetFileName(str(path))
        reader.Update()
        np.testing.assert_allclose(
            reader.GetOutput().GetBounds(), mesh.GetBounds(), atol=5e-5
        )
        assert metadata["coordinate_system"] == "LPS"
        assert metadata["unit"] == "mm"


def test_hollow_clip_has_open_rim_and_inner_surface():
    volume = synthetic_volume(72)
    mesh = extract_surface(volume, Layer("wall", 50, (0.8, 0.3, 0.3)))
    clipped = clipped_mesh(mesh, ClipState(True, [0, 0, 0], [0, 1, 0]))
    points = vtk_to_numpy(clipped.GetPoints().GetData())
    assert points[:, 1].min() >= -1e-5

    sphere = points[np.abs(points[:, 0]) < 5]
    radius = np.linalg.norm(sphere, axis=1)
    assert (radius < 72 * 0.28).any() and (radius > 72 * 0.30).any()
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(clipped)
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.NonManifoldEdgesOff()
    edges.Update()
    assert edges.GetOutput().GetNumberOfCells() > 100


def test_roi_retains_scalar_gradient_normals():
    volume = synthetic_volume(64)
    mask = np.ones(volume.array.shape, dtype=bool)
    mask[:10] = False
    mesh = extract_surface(volume, Layer("wall", 50, (0.8, 0.3, 0.3)), mask=mask)
    assert mesh.GetPointData().GetNormals() is not None
    normals = vtk_to_numpy(mesh.GetPointData().GetNormals())
    assert np.isfinite(normals).all()
    assert (np.linalg.norm(normals, axis=1) > 0.9).mean() > 0.99


def test_interval_boundaries_have_opposite_outward_normals():
    z, y, x = np.indices((9, 10, 11), dtype=np.float32)
    volume = rotated_volume(x)
    layer = Layer(
        "tissue interval", 2.5, (0.7, 0.4, 0.2), kind="surface_band", upper=6.5
    )
    mesh = extract_surface(volume, layer)
    xyz = volume.to_index(vtk_to_numpy(mesh.GetPoints().GetData()))
    normals = vtk_to_numpy(mesh.GetPointData().GetNormals())
    axis = np.asarray(volume.direction).reshape(3, 3)[:, 0]
    low = np.isclose(xyz[:, 0], 2.5, atol=3e-6)
    high = np.isclose(xyz[:, 0], 6.5, atol=3e-6)
    assert low.any() and high.any() and (low | high).all()
    assert np.all(normals[low] @ axis < -0.99)
    assert np.all(normals[high] @ axis > 0.99)
    before = geometry_key(volume, [layer])
    layer.upper = 7
    assert before != geometry_key(volume, [layer])


def test_lit_interval_section_keeps_the_excluded_lumen_open():
    from openkalolumen.imaging.mpr import plane_basis
    from openkalolumen.render.engine import RenderEngine

    z, y, x = np.indices((21, 21, 21), dtype=np.float32)
    volume = VolumeData(
        np.hypot(x - 10, y - 10),
        (1, 1, 1),
        (-10, -10, -10),
        (1, 0, 0, 0, 1, 0, 0, 0, 1),
    )
    engine = RenderEngine()
    engine.set_volume(volume)
    layer = Layer("annular wall", 2, (0.7, 0.4, 0.2), kind="surface_band", upper=5)
    engine.configure_layers([layer])
    clip = ClipState(True, [0, 0, 0], [0, 0, 1], section="tissue")
    u, v = plane_basis(clip.effective_normal)
    engine._tissue_section(clip, volume.array[10], 1, u, v)
    actors = engine.section_actor.GetParts()
    actors.InitTraversal()
    actor = actors.GetNextProp3D()
    assert actor is not None
    points = vtk_to_numpy(actor.GetMapper().GetInput().GetPoints().GetData())
    radius = np.linalg.norm(points[:, :2], axis=1)
    assert radius.min() > 1.9 and radius.max() < 5.01
    assert actor.GetProperty().GetInterpolation() == vtk.VTK_PBR
    assert actors.GetNextProp3D() is None
    engine.window.Finalize()


def test_display_filter_uses_mm_preserves_raw_and_blocks_filtered_export(tmp_path):
    values = np.zeros((25, 25, 25), dtype=np.float32)
    values[12, 12, 12] = 1000
    volume = VolumeData(values, (1.0, 2.0, 1.0), (0, 0, 0), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    original = values.copy()
    image = display_field(volume, 2.0)
    filtered = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(values.shape)

    np.testing.assert_allclose(filtered[12, 12, 14], filtered[12, 13, 12], rtol=0.001)
    np.testing.assert_array_equal(volume.array, original)
    assert filtered[12, 12, 12] < 1000 and filtered.sum() == pytest.approx(
        1000, rel=0.005
    )
    assert display_field(volume, 2.0) is image
    layer = Layer("filtered", 10, (0.7, 0.3, 0.2), smoothing_mm=2.0)
    key = geometry_key(volume, [layer])
    layer.smoothing_mm = 0
    assert geometry_key(volume, [layer]) != key
    layer.smoothing_mm = 2
    with pytest.raises(ValueError, match="display smoothing"):
        export_mesh(tmp_path / "unsafe.stl", vtk.vtkPolyData(), layer, volume)


@pytest.mark.parametrize("reflected", [False, True])
def test_solid_wall_closed_roi_oblique_clip_and_open_lumen(tmp_path, reflected):
    from openkalolumen.render.geometry import solid_cut, solid_faces

    z, y, x = np.indices((33, 33, 33), dtype=np.float32)
    volume = rotated_volume(np.hypot((x - 16) * 1.2, (y - 16) * 1.8))
    if reflected:
        direction = np.asarray(volume.direction).reshape(3, 3).copy()
        direction[:, 0] *= -1
        volume = VolumeData(
            volume.array, volume.spacing, volume.origin, tuple(direction.ravel())
        )
    layer = Layer("finite wall", 5, (0.8, 0.6, 0.4), kind="solid_band", upper=9)
    roi = np.zeros(volume.array.shape, dtype=bool)
    roi[4:29, 3:30, 3:30] = True
    original = volume.array.copy()
    mesh = extract_surface(volume, layer, roi)

    def open_edges(poly):
        e = vtk.vtkFeatureEdges()
        e.SetInputData(poly)
        e.BoundaryEdgesOn()
        e.NonManifoldEdgesOn()
        e.FeatureEdgesOff()
        e.ManifoldEdgesOff()
        e.Update()
        return e.GetOutput().GetNumberOfCells()

    assert open_edges(mesh) == 0
    direction = np.asarray(volume.direction).reshape(3, 3)
    normal = direction @ np.array([0.2, 0.1, 1.0])
    normal /= np.linalg.norm(normal)
    clip = ClipState(True, list(volume.center), list(normal))
    closed, faces = solid_cut(mesh, clip)
    assert open_edges(closed) == 0 and faces.GetNumberOfCells() > 0
    points = vtk_to_numpy(faces.GetPoints().GetData())
    np.testing.assert_allclose((points - volume.center) @ normal, 0, atol=8e-6)

    cells = vtk_to_numpy(faces.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    xyz = volume.to_index(points[cells].mean(1))
    radius = np.hypot((xyz[:, 0] - 16) * 1.2, (xyz[:, 1] - 16) * 1.8)
    assert radius.min() > 4.7 and radius.max() < 9.01
    fast_faces = solid_faces(mesh, clip)
    p = vtk_to_numpy(fast_faces.GetPoints().GetData())
    cells = vtk_to_numpy(fast_faces.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    xyz = volume.to_index(p[cells].mean(1))
    assert np.hypot((xyz[:, 0] - 16) * 1.2, (xyz[:, 1] - 16) * 1.8).min() > 4.7
    np.testing.assert_allclose((p - volume.center) @ normal, 0, atol=8e-6)

    ray = vtk.vtkStaticCellLocator()
    ray.SetDataSet(mesh)
    ray.BuildLocator()
    hits = vtk.vtkPoints()
    ids = vtk.vtkIdList()
    ray.IntersectWithLine(
        volume.to_world([16, 16.01, 16.02]),
        volume.to_world([28, 16.01, 16.02]),
        1e-5,
        hits,
        ids,
    )
    distances = np.linalg.norm(vtk_to_numpy(hits.GetData()) - volume.center, axis=1)
    np.testing.assert_allclose(distances, [5, 9], atol=0.04)
    np.testing.assert_array_equal(volume.array, original)
    metadata = export_mesh(tmp_path / "closed-wall.stl", mesh, layer, volume, clip)
    assert metadata["closed_cut_faces"] and not metadata["hole_filling"]
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(tmp_path / "closed-wall.stl"))
    reader.Update()
    assert open_edges(reader.GetOutput()) == 0


def test_solid_material_is_boundary_based_and_does_not_recolor_mesh_cache():
    from openkalolumen.render.materials import apply_material

    z, y, x = np.indices((13, 13, 13), dtype=np.float32)
    volume = VolumeData(x, (1, 1, 1), (0, 0, 0), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    layer = Layer(
        "slab",
        3.5,
        (0.8, 0.6, 0.4),
        kind="solid_band",
        upper=8.5,
        interior_color=(0.5, 0.2, 0.15),
    )
    mesh = extract_surface(volume, layer)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    apply_material(actor, layer)
    assert mesh.GetPointData().GetScalars() is None
    colors = vtk_to_numpy(mapper.GetInput().GetPointData().GetScalars())
    hu = vtk_to_numpy(mesh.GetPointData().GetArray("BoundaryHU"))
    np.testing.assert_allclose(
        colors[np.isclose(hu, 3.5)][0], np.array(layer.color) * 255, atol=1
    )
    np.testing.assert_allclose(
        colors[np.isclose(hu, 8.5)][0], np.array(layer.interior_color) * 255, atol=1
    )


def test_sampled_solid_cut_keeps_cavity_and_opaque_contacts_are_internal():
    from openkalolumen.render.geometry import sampled_solid_faces
    from openkalolumen.render.materials import apply_material

    z, y, x = np.indices((31, 31, 31), dtype=np.float32)
    volume = VolumeData(
        np.hypot(x - 15, y - 15),
        (1, 1, 1),
        (-15, -15, -15),
        (1, 0, 0, 0, 1, 0, 0, 0, 1),
    )
    layer = Layer("inner", 5, (0.5, 0.2, 0.1), kind="solid_band", upper=9)
    clip = ClipState(True, [0, 0, 0], [0.1, 0.2, 1])
    face = sampled_solid_faces(volume, layer, None, clip, 151)
    points = vtk_to_numpy(face.GetPoints().GetData())
    cells = vtk_to_numpy(face.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    assert np.linalg.norm(points[cells].mean(1)[:, :2], axis=1).min() > 4.8
    np.testing.assert_allclose(points @ clip.effective_normal, 0, atol=1e-7)
    outer = Layer("outer", 9, (0.8, 0.7, 0.6), kind="solid_band", upper=12)
    mesh = extract_surface(volume, layer)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    apply_material(actor, layer, [layer, outer])
    assert mapper.GetInput().GetNumberOfCells() < mesh.GetNumberOfCells()

    outer.visible = False
    mapper.SetInputData(mesh)
    apply_material(actor, layer, [layer, outer])
    assert mapper.GetInput().GetNumberOfCells() == mesh.GetNumberOfCells()


def test_solid_export_rejects_coincident_nonmanifold_geometry_before_writing(tmp_path):
    volume = synthetic_volume(32)
    sphere = vtk.vtkSphereSource()
    sphere.Update()
    append = vtk.vtkAppendPolyData()
    append.AddInputData(sphere.GetOutput())
    append.AddInputData(sphere.GetOutput())
    append.Update()
    layer = Layer(
        "invalid coincident solid", 0, (1, 1, 1), kind="solid_band", upper=200
    )
    path = tmp_path / "invalid.stl"
    with pytest.raises(ValueError, match="nonmanifold"):
        export_mesh(path, append.GetOutput(), layer, volume)
    assert not path.exists()


def test_drr_analytic_water_anisotropic_oblique():
    volume = rotated_volume(np.zeros((12, 14, 16), dtype=np.float32))
    direction = np.asarray(volume.direction).reshape(3, 3)
    axis = direction[:, 0]
    camera = {
        "position": list(volume.center - axis * 200),
        "focal_point": list(volume.center),
        "view_up": [0, 0, 1],
        "parallel": True,
        "parallel_scale": 0.1,
    }
    result = project_drr(volume, camera, 5, mu_water=0.02, step=0.7)
    np.testing.assert_allclose(result, np.exp(-0.02 * 16 * 1.2), atol=1e-6)
    clipped = project_drr(
        volume,
        camera,
        5,
        mu_water=0.02,
        step=0.1,
        clip=ClipState(True, list(volume.center), list(axis)),
    )
    np.testing.assert_allclose(clipped, np.exp(-0.02 * 16 * 1.2 / 2), atol=1e-5)
    volume.hu_calibrated = False
    with pytest.raises(ValueError, match="HU"):
        project_drr(volume, camera, 5)


def test_labels_use_nearest_neighbour_physical_alignment(tmp_path):
    volume = rotated_volume(np.zeros((8, 9, 10), dtype=np.int16))
    labels = np.zeros_like(volume.array, dtype=np.uint16)
    labels[2:5, 3:6, 4:7] = 115
    image = rotated_volume(labels)
    path = tmp_path / "labels.nii.gz"
    save_volume(image, path)
    aligned = aligned_labels(path, volume)
    np.testing.assert_array_equal(labels, aligned)
    assert set(np.unique(aligned)) == {0, 115}


def test_scene_and_measurement_roundtrip(tmp_path):
    volume = synthetic_volume(48)
    layers = [Layer("soft", 50, (0.8, 0.2, 0.2))]
    clip = ClipState(True, list(volume.center), [0.3, 0.4, 0.5], True, "tissue")
    measurement = Measurement(
        "distance",
        [[0, 0, 0], [3, 4, 12]],
        layer="soft",
        geometry_key=geometry_key(volume, layers),
    )
    assert measurement.value == 13
    assert Measurement("angle", [[1, 0, 0], [0, 0, 0], [0, 1, 0]]).value == 90
    path = tmp_path / "scene.json"
    save_scene(
        path,
        volume=volume,
        layers=layers,
        clip=clip,
        camera={},
        crosshair=volume.center,
        measurements=[measurement],
    )
    scene = read_scene(path)
    assert scene["layers"] == layers
    assert scene["clip"] == clip
    assert scene["measurements"][0].value == 13
    layers[0].color = (0.1, 0.2, 0.3)
    assert geometry_key(volume, layers) == measurement.geometry_key
    layers[0].threshold = 51
    assert geometry_key(volume, layers) != measurement.geometry_key
