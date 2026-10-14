"""OpenGL objects live exclusively on the caller's GUI/render thread."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from openkalolumen.imaging.mpr import display_slice, plane_basis, reslice
from openkalolumen.render.geometry import (
    _image_for_array,
    clipped_mesh,
    display_field,
    sampled_solid_faces,
    solid_faces,
)
from openkalolumen.render.materials import apply_material, studio_environment


class RenderEngine:
    """Coordinate VTK volume, surface, section, lighting, and picking passes."""

    def __init__(self, window=None):
        self.window = window or vtk.vtkRenderWindow()
        self.window.SetAlphaBitPlanes(1)
        self.window.SetMultiSamples(0)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.035, 0.044, 0.06)
        self.renderer.SetBackground2(0.09, 0.105, 0.135)
        self.renderer.GradientBackgroundOn()
        self.renderer.SetUseDepthPeeling(True)
        self.renderer.SetMaximumNumberOfPeels(100)
        self.renderer.SetOcclusionRatio(0.0)
        self.renderer.SetUseFXAA(True)
        self.renderer.SetTwoSidedLighting(True)
        self.renderer.SetAutomaticLightCreation(False)
        self.environment = studio_environment()
        self.renderer.SetEnvironmentTexture(self.environment)
        self.renderer.UseImageBasedLightingOn()
        self.renderer.SetEnvironmentUp(0, 0, 1)
        self.renderer.SetEnvironmentRight(1, 0, 0)
        self.window.AddRenderer(self.renderer)
        self.volume = None
        self.image = None
        self.mask = None
        self.iso_mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.iso_mapper.SetBlendModeToIsoSurface()
        self.iso_mapper.SetComputeNormalFromOpacity(False)
        self.iso = vtk.vtkVolume()
        self.iso.SetMapper(self.iso_mapper)
        self.iso_property = vtk.vtkVolumeProperty()
        self.iso_property.ShadeOn()
        self.iso_property.SetAmbient(0.18)
        self.iso_property.SetDiffuse(0.85)
        self.iso_property.SetSpecular(0.22)
        self.iso_property.SetSpecularPower(24)
        self.iso_property.SetInterpolationTypeToLinear()
        self.iso.SetProperty(self.iso_property)
        self.renderer.AddVolume(self.iso)
        self.composite_mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.composite_mapper.SetBlendModeToComposite()
        self.composite = vtk.vtkVolume()
        self.composite.SetMapper(self.composite_mapper)
        self.composite_property = vtk.vtkVolumeProperty()
        self.composite_property.ShadeOn()
        self.composite_property.SetAmbient(0.12)
        self.composite_property.SetDiffuse(0.8)
        self.composite_property.SetSpecular(0.2)
        self.composite_property.SetSpecularPower(25)
        self.composite_property.SetInterpolationTypeToLinear()
        self.composite.SetProperty(self.composite_property)
        self.renderer.AddVolume(self.composite)
        self.iso.VisibilityOff()
        self.composite.VisibilityOff()
        self.surface_actors = {}
        self.meshes = {}
        self.layers = []
        self.clip = None
        self.mode = "surface"
        self.geometry_key = ""
        self.geometry_current = False
        self.annotations = []
        self.section_actor = None
        self.section_sources = {}
        self.section_signature = None
        self.plane = vtk.vtkPlane()
        self.outline_actor = None
        self.ssao = True
        self.camera_callback = None

    def set_volume(self, volume):
        """Replace the volume, clear derived geometry, and initialize physical transforms."""
        self.volume = volume
        self.clip = None
        if self.section_actor:
            self.renderer.RemoveActor(self.section_actor)
            self.section_actor = None
        self.image = volume.to_vtk()
        for mapper, actor in (
            (self.iso_mapper, self.iso),
            (self.composite_mapper, self.composite),
        ):
            mapper.SetInputData(self.image)
            mapper.RemoveAllClippingPlanes()
            mapper.SetAutoAdjustSampleDistances(False)
            mapper.SetSampleDistance(min(volume.spacing) * 0.5)
            mapper.SetImageSampleDistance(1.0)
            mapper.SetUseJittering(True)
            mapper.SetMaxMemoryInBytes(1024**3)
            mapper.SetMaskInput(None)
            actor.SetUserMatrix(volume.actor_matrix())
        self.mask = None
        self.clear_surfaces()
        for actor in self.annotations:
            self.renderer.RemoveViewProp(actor)
        self.annotations = []
        if self.outline_actor:
            self.renderer.RemoveActor(self.outline_actor)
        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(self.image)
        outline.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(outline.GetOutputPort())
        self.outline_actor = vtk.vtkActor()
        self.outline_actor.SetMapper(mapper)
        self.outline_actor.SetUserMatrix(volume.actor_matrix())
        self.outline_actor.GetProperty().SetColor(0.19, 0.26, 0.34)
        self.outline_actor.GetProperty().SetOpacity(0.45)
        self.outline_actor.PickableOff()
        self.outline_actor.VisibilityOff()
        self.renderer.AddActor(self.outline_actor)
        self._set_lights()
        camera = self.renderer.GetActiveCamera()
        center = volume.center
        diagonal = float(np.linalg.norm(volume.bounds[1] - volume.bounds[0]))
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center + np.array([0.8, -1.6, 0.55]) * diagonal))
        camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()

    def _set_lights(self, strength=1.0):
        """Create directional camera-relative fill lights."""
        self.renderer.RemoveAllLights()
        for offset, intensity, color in [
            ((-0.6, 0.8, 1.0), 0.8, (1.0, 0.93, 0.85)),
            ((0.8, 0.2, 0.5), 0.3, (0.76, 0.86, 1.0)),
            ((0.2, -0.6, -1.0), 0.5, (1.0, 0.9, 0.8)),
        ]:
            light = vtk.vtkLight()
            light.SetLightTypeToCameraLight()
            light.SetPosition(*offset)
            light.SetFocalPoint(0, 0, 0)
            light.SetColor(*color)
            light.SetIntensity(intensity * strength)
            self.renderer.AddLight(light)

    def clear_surfaces(self):
        """Remove derived surface actors and invalidate current geometry."""
        for actor in self.surface_actors.values():
            self.renderer.RemoveActor(actor)
        self.surface_actors = {}
        self.meshes = {}
        self.geometry_current = False

    def set_mask(self, mask):
        """Restrict visible rendering to a voxel mask while preserving source data."""
        self.mask = mask
        image = (
            None
            if mask is None
            else _image_for_array(self.volume, mask.astype(np.uint8) * 255)
        )
        for mapper in (self.iso_mapper, self.composite_mapper):
            mapper.SetMaskTypeToBinary()
            mapper.SetMaskInput(image)
        self.geometry_current = False

    def configure_layers(self, layers):
        """Update scalar previews, materials, and visible material contacts."""
        self.layers = layers
        visible = sorted(
            (
                layer
                for layer in layers
                if layer.visible
                and layer.kind in ("isosurface", "surface_band", "solid_band")
                and layer.label_id is None
            ),
            key=lambda layer: layer.threshold,
        )
        thresholds = [layer.threshold for layer in visible]
        preview = [(layer.threshold, layer) for layer in visible]
        preview.extend(
            (layer.upper, layer)
            for layer in visible
            if layer.kind in ("surface_band", "solid_band") and layer.upper is not None
        )
        preview = list({value: (value, layer) for value, layer in preview}.values())
        if len(preview) > 8:
            raise ValueError(
                "Fast preview supports at most eight intensity boundaries; each tissue interval has two boundaries."
            )
        if len(set(thresholds)) != len(thresholds):
            raise ValueError("Visible intensity layers must have distinct thresholds.")
        iso_colors = vtk.vtkColorTransferFunction()
        iso_opacity = vtk.vtkPiecewiseFunction()
        contours = self.iso_property.GetIsoSurfaceValues()
        contours.SetNumberOfContours(len(preview))
        for i, (value, layer) in enumerate(sorted(preview, key=lambda pair: pair[0])):
            contours.SetValue(i, value)
            iso_colors.AddRGBPoint(value, *layer.color)
            iso_opacity.AddPoint(value, layer.opacity)
        self.iso_property.SetColor(iso_colors)
        self.iso_property.SetScalarOpacity(iso_opacity)
        self._composite_transfer(layers)
        for i, actor in self.surface_actors.items():
            if i < len(layers):
                layer = layers[i]
                actor.SetVisibility(layer.visible)

                actor.GetMapper().SetInputData(self.meshes[i])
                apply_material(actor, layer, self.layers)

    def _composite_transfer(self, layers):
        """Build a combined color and opacity transfer function for visible layers."""
        colors = vtk.vtkColorTransferFunction()
        opacity = vtk.vtkPiecewiseFunction()
        lo, hi = self.volume.scalar_range
        colors.AddRGBPoint(lo, 0.1, 0.1, 0.13)
        opacity.AddPoint(lo, 0)
        nodes = []
        for layer in layers:
            if not layer.visible or layer.label_id is not None:
                continue
            ramp = max((hi - lo) * 0.003, 2.0)
            upper = (
                layer.upper
                if layer.kind in ("volume_band", "solid_band")
                and layer.upper is not None
                else layer.threshold + max(70, (hi - lo) * 0.035)
            )
            nodes.extend(
                [
                    (layer.threshold - ramp, layer.color, 0.0),
                    (layer.threshold, layer.color, layer.opacity * 0.16),
                    (upper, layer.color, layer.opacity * 0.16),
                    (upper + ramp, layer.color, 0.0),
                ]
            )
        for x, color, a in sorted(nodes, key=lambda x: x[0]):
            colors.AddRGBPoint(x, *color)
            opacity.AddPoint(x, a)
        self.composite_property.SetColor(colors)
        self.composite_property.SetScalarOpacity(opacity)
        self.composite_property.SetScalarOpacityUnitDistance(min(self.volume.spacing))

    def show_preview(self):
        """Display interactive scalar boundaries while detailed meshes are pending."""
        self.geometry_current = False
        self.renderer.SetPass(None)
        if self.section_actor:
            self.section_actor.VisibilityOff()
        for actor in self.surface_actors.values():
            actor.VisibilityOff()
        self.iso.SetVisibility(
            self.mode == "surface"
            and any(
                layer.visible
                and layer.label_id is None
                and layer.kind in ("isosurface", "surface_band", "solid_band")
                for layer in self.layers
            )
        )
        self.composite.SetVisibility(self.mode in ("volume", "hybrid"))
        self.renderer.SetUseDepthPeelingForVolumes(self.mode == "hybrid")
        self.render()

    def apply_surfaces(self, key, meshes):
        """Install detailed meshes and mark their geometry generation current."""
        self.clear_surfaces()
        self.geometry_key = key
        self.meshes = meshes
        for i, mesh in meshes.items():
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(mesh)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            apply_material(actor, self.layers[i], self.layers)
            self.renderer.AddActor(actor)
            self.surface_actors[i] = actor
        self.geometry_current = True
        self.iso.VisibilityOff()
        self.set_mode(self.mode)
        self.set_clip(self.clip)

    def set_mode(self, mode):
        """Select surface, volume, or combined rendering."""
        self.mode = mode
        if self.geometry_current:
            self.iso.VisibilityOff()
            for i, actor in self.surface_actors.items():
                actor.SetVisibility(
                    mode in ("surface", "hybrid") and self.layers[i].visible
                )
        else:
            self.iso.SetVisibility(
                mode == "surface"
                and any(
                    layer.visible
                    and layer.kind in ("isosurface", "surface_band", "solid_band")
                    and layer.label_id is None
                    for layer in self.layers
                )
            )
        self.composite.SetVisibility(mode in ("volume", "hybrid"))
        self.renderer.SetUseDepthPeelingForVolumes(mode == "hybrid")
        self._apply_ssao()
        self.render()

    def _apply_ssao(self):
        """Enable anatomy-scaled ambient occlusion for opaque surface scenes."""
        self.renderer.SetPass(None)

        if (
            self.ssao
            and self.mode == "surface"
            and self.geometry_current
            and all(layer.opacity == 1 for layer in self.layers if layer.visible)
        ):
            steps = vtk.vtkRenderStepsPass()
            pass_ = vtk.vtkSSAOPass()
            pass_.SetDelegatePass(steps)

            bounds = np.asarray(self.renderer.ComputeVisiblePropBounds()).reshape(3, 2)
            radius = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0])) * 0.045
            pass_.SetRadius(min(30.0, max(radius * 2, min(self.volume.spacing) * 5)))
            pass_.SetBias(min(self.volume.spacing) * 0.2)
            pass_.SetKernelSize(128)
            pass_.BlurOn()
            self.renderer.SetPass(pass_)

    def set_quality(self, high=True, scattering=False):
        """Select interactive or detailed volume sampling and scattering."""
        if self.volume is None:
            return
        for mapper in (self.iso_mapper, self.composite_mapper):
            mapper.SetSampleDistance(min(self.volume.spacing) * (0.5 if high else 1.0))
            mapper.SetImageSampleDistance(1.0 if high else 2.0)
        self.composite_mapper.SetVolumetricScatteringBlending(
            0.6 if scattering and high else 0.0
        )
        self.composite_mapper.SetGlobalIlluminationReach(
            0.3 if scattering and high else 0.0
        )

    def set_clip(self, clip):
        """Apply a physical clipping half-space to all rendering passes."""
        self.clip = clip
        for mapper in (self.iso_mapper, self.composite_mapper):
            mapper.RemoveAllClippingPlanes()
        for actor in self.surface_actors.values():
            actor.GetMapper().RemoveAllClippingPlanes()
        if clip and clip.enabled:
            self.plane.SetOrigin(*clip.origin)
            self.plane.SetNormal(*clip.effective_normal)
            for mapper in (self.iso_mapper, self.composite_mapper):
                mapper.AddClippingPlane(self.plane)
            for actor in self.surface_actors.values():
                actor.GetMapper().AddClippingPlane(self.plane)
        if self.section_actor:
            self.section_actor.SetVisibility(
                bool(
                    clip
                    and clip.enabled
                    and self.section_signature == self._section_key(clip)
                    and (
                        clip.section != "open"
                        or any(
                            layer.visible and layer.kind == "solid_band"
                            for layer in self.layers
                        )
                    )
                )
            )
        self.render()

    def _section_key(self, clip):
        """Identify a section by its physical plane and display geometry."""
        return (
            self.geometry_key,
            tuple(clip.origin),
            tuple(clip.effective_normal),
            clip.section,
        )

    def set_section(self, clip, level, width, resolution=512):
        """Render an open, grayscale, or illuminated tissue section."""
        self.section_sources = {}
        self.section_signature = None
        if self.section_actor:
            self.renderer.RemoveActor(self.section_actor)
            self.section_actor = None
        if not clip or not clip.enabled:
            return
        self.section_signature = self._section_key(clip)
        solids = [
            i
            for i, layer in enumerate(self.layers)
            if layer.visible and layer.kind == "solid_band"
        ]
        if solids and clip.section in ("open", "tissue"):
            self._solid_section(clip, solids)
            self._apply_ssao()
            self.render()
            return
        if clip.section == "open":
            return
        u, v = plane_basis(clip.effective_normal)
        values, spacing = reslice(self.volume, clip.origin, u, v, resolution)
        if clip.section == "tissue":
            self._tissue_section(clip, values, spacing, u, v)
            self._apply_ssao()
            self.render()
            return
        rgba = np.empty((*values.shape, 4), dtype=np.uint8)
        rgba[..., :3] = display_slice(values, level, width)
        rgba[..., 3] = 255
        if clip.section == "tissue":
            rgba[..., 3] = 0
            for layer in sorted(self.layers, key=lambda layer: layer.threshold):
                if layer.visible and layer.label_id is None:
                    mask = values >= layer.threshold
                    if layer.upper is not None:
                        mask &= values <= layer.upper
                    rgba[mask, :3] = (np.asarray(layer.color) * 255).astype(np.uint8)
                    rgba[mask, 3] = int(layer.opacity * 255)
        if self.mask is not None:
            mask, _ = reslice(
                self.volume,
                clip.origin,
                u,
                v,
                resolution,
                array=self.mask.astype(np.uint8),
            )
            rgba[mask < 0.5, 3] = 0
        image = vtk.vtkImageData()
        image.SetDimensions(resolution, resolution, 1)
        image.SetSpacing(spacing, spacing, 1)
        image.SetOrigin(
            -(resolution - 1) * spacing / 2, -(resolution - 1) * spacing / 2, 0
        )
        scalars = numpy_to_vtk(
            np.ascontiguousarray(np.flipud(rgba)).reshape(-1, 4), deep=True
        )
        image.GetPointData().SetScalars(scalars)
        actor = vtk.vtkImageActor()
        actor.GetMapper().SetInputData(image)
        matrix = vtk.vtkMatrix4x4()
        for j, axis in enumerate((u, v, np.cross(u, v))):
            for i in range(3):
                matrix.SetElement(i, j, float(axis[i]))
        for i in range(3):
            matrix.SetElement(i, 3, float(clip.origin[i]))
        actor.SetUserMatrix(matrix)
        actor.PickableOff()
        self.section_actor = actor
        self.renderer.AddActor(actor)
        self.render()

    def _solid_section(self, clip, indices):
        """Exact plane/material intersection; voids receive no cut polygon.

        The cap comes from the SAME closed mesh as its inner and outer walls,
        so reslice resolution cannot invent thickness or misalign the rim.
        """
        from openkalolumen.render.materials import tissue_property

        assembly = vtk.vtkAssembly()
        self.section_sources = {}
        u, v = plane_basis(clip.effective_normal)
        values, spacing = reslice(self.volume, clip.origin, u, v, 768)
        resolution = values.shape[0]
        for index in indices:
            if index not in self.meshes:
                continue
            layer = self.layers[index]
            try:
                faces = solid_faces(self.meshes[index], clip)
                self.section_sources[index] = "mesh_loops"
            except ValueError:
                faces = sampled_solid_faces(self.volume, layer, self.mask, clip)
                self.section_sources[index] = "sampled_material_field"
            if not faces.GetNumberOfCells():
                continue
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(faces)
            normals.SplittingOff()
            normals.ConsistencyOn()
            normals.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = tissue_property(layer)

            density = np.clip(
                (values - layer.threshold) / (layer.upper - layer.threshold), 0, 1
            )
            tint = 0.88 + 0.12 * density
            rgb = np.clip(
                np.asarray(layer.cut_color or layer.color)[None, None, :]
                * tint[..., None]
                * 255,
                0,
                255,
            ).astype(np.uint8)
            texture_image = vtk.vtkImageData()
            texture_image.SetDimensions(resolution, resolution, 1)
            texture_image.GetPointData().SetScalars(
                numpy_to_vtk(np.flipud(rgb).copy().reshape(-1, 3), deep=True)
            )
            texture = vtk.vtkTexture()
            texture.SetInputData(texture_image)
            texture.SetColorModeToDirectScalars()
            texture.InterpolateOn()
            texture.UseSRGBColorSpaceOn()
            points = vtk_to_numpy(
                normals.GetOutput().GetPoints().GetData()
            ) - np.asarray(clip.origin)
            uv = (
                np.column_stack((points @ u, points @ v)) / ((resolution - 1) * spacing)
                + 0.5
            )
            normals.GetOutput().GetPointData().SetTCoords(
                numpy_to_vtk(uv.astype(np.float32), deep=True)
            )
            surface = vtk.vtkPolyData()
            surface.ShallowCopy(normals.GetOutput())
            mapper.SetInputData(surface)
            prop.SetBaseColorTexture(texture)
            prop.SetColor(1, 1, 1)
            prop.SetRoughness(max(0.65, layer.roughness))
            prop.SetCoatStrength(min(0.08, layer.coat))
            actor.SetProperty(prop)
            actor.PickableOff()
            assembly.AddPart(actor)
        self.section_actor = assembly
        self.renderer.AddActor(assembly)

    def _tissue_section(self, clip, values, spacing, u, v):
        """Lit cut faces clipped from the sampled scalar field, including interval holes.

        This is a display-only section. Original surface/export coordinates are unchanged.
        Unlike a solid polygon cap it preserves high-HU exclusions and the ROI boundary.
        """
        from openkalolumen.render.materials import tissue_property

        resolution = values.shape[0]
        image = vtk.vtkImageData()
        image.SetDimensions(resolution, resolution, 1)
        image.SetSpacing(spacing, spacing, 1)
        image.SetOrigin(
            -(resolution - 1) * spacing / 2, -(resolution - 1) * spacing / 2, 0
        )
        hu = numpy_to_vtk(np.ascontiguousarray(np.flipud(values)).ravel(), deep=True)
        hu.SetName("HU")
        image.GetPointData().SetScalars(hu)
        if self.mask is not None:
            mask, _ = reslice(
                self.volume,
                clip.origin,
                u,
                v,
                resolution,
                array=self.mask.astype(np.uint8),
            )
            roi = numpy_to_vtk(
                np.ascontiguousarray(np.flipud(mask)).astype(np.float32).ravel(),
                deep=True,
            )
            roi.SetName("ROI")
            image.GetPointData().AddArray(roi)
        geometry = vtk.vtkGeometryFilter()
        geometry.SetInputData(image)
        geometry.Update()
        triangle = vtk.vtkTriangleFilter()
        triangle.SetInputData(geometry.GetOutput())
        triangle.Update()
        assembly = vtk.vtkAssembly()
        matrix = vtk.vtkMatrix4x4()
        for j, axis in enumerate((u, v, np.cross(u, v))):
            for i in range(3):
                matrix.SetElement(i, j, float(axis[i]))

        for i in range(3):
            matrix.SetElement(
                i, 3, float(clip.origin[i] - clip.effective_normal[i] * 0.02)
            )
        displayed = sorted(
            (
                layer
                for layer in self.layers
                if layer.visible and layer.label_id is None
            ),
            key=lambda layer: layer.threshold,
        )
        for index, layer in enumerate(displayed):
            poly = triangle.GetOutput()
            if layer.smoothing_mm > 0:
                sampled, _ = reslice(
                    self.volume,
                    clip.origin,
                    u,
                    v,
                    resolution,
                    image_override=display_field(self.volume, layer.smoothing_mm),
                )
                filtered = vtk.vtkImageData()
                filtered.DeepCopy(image)
                array = numpy_to_vtk(
                    np.ascontiguousarray(np.flipud(sampled)).ravel(), deep=True
                )
                array.SetName("HU")
                filtered.GetPointData().SetScalars(array)
                geometry_ = vtk.vtkGeometryFilter()
                geometry_.SetInputData(filtered)
                geometry_.Update()
                triangles = vtk.vtkTriangleFilter()
                triangles.SetInputData(geometry_.GetOutput())
                triangles.Update()
                poly = triangles.GetOutput()
            poly.GetPointData().SetActiveScalars("HU")
            lower = vtk.vtkClipPolyData()
            lower.SetInputData(poly)
            lower.SetValue(layer.threshold)
            lower.Update()
            poly = lower.GetOutput()

            upper = layer.upper
            if index + 1 < len(displayed):
                upper = (
                    min(upper, displayed[index + 1].threshold)
                    if upper is not None
                    else displayed[index + 1].threshold
                )
            if upper is not None:
                cut = vtk.vtkClipPolyData()
                cut.SetInputData(poly)
                cut.SetValue(upper)
                cut.InsideOutOn()
                cut.Update()
                poly = cut.GetOutput()
            if self.mask is not None:
                poly.GetPointData().SetActiveScalars("ROI")
                cut = vtk.vtkClipPolyData()
                cut.SetInputData(poly)
                cut.SetValue(0.5)
                cut.Update()
                poly = cut.GetOutput()
            if not poly.GetNumberOfPoints():
                continue
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(poly)
            normals.SplittingOff()
            normals.Update()
            surface = vtk.vtkPolyData()
            surface.DeepCopy(normals.GetOutput())
            hu = vtk_to_numpy(surface.GetPointData().GetArray("HU"))
            span = max((upper or layer.threshold + 400) - layer.threshold, 1)
            tint = 0.85 + 0.15 * np.clip((hu - layer.threshold) / span, 0, 1)
            rgb = (np.asarray(layer.color)[None, :] * tint[:, None] * 255).astype(
                np.uint8
            )
            color = numpy_to_vtk(rgb, deep=True)
            color.SetName("TissueColor")
            surface.GetPointData().SetScalars(color)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(surface)
            mapper.SetColorModeToDirectScalars()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetUserMatrix(matrix)
            prop = tissue_property(layer)
            prop.SetRoughness(0.65)
            prop.SetCoatStrength(0.12)
            actor.SetProperty(prop)
            actor.PickableOff()
            assembly.AddPart(actor)
        self.section_actor = assembly
        self.renderer.AddActor(assembly)

    def pick(self, x, y, layer_index=None):
        """Pick visible detailed geometry with physical clipping applied."""
        if not self.geometry_current or self.mode == "volume":
            raise ValueError(
                "Surface picking requires current detailed geometry. Wait for generation and select surface or hybrid rendering."
            )
        if any(
            layer.visible
            and layer.smoothing_mm > 0
            and (layer_index is None or i == layer_index)
            for i, layer in enumerate(self.layers)
        ):
            raise ValueError(
                "This layer uses display smoothing, which changes threshold boundaries. Disable it before measuring or picking original image geometry."
            )
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.003)
        picker.PickFromListOn()
        for i, actor in self.surface_actors.items():
            if actor.GetVisibility() and (layer_index is None or i == layer_index):
                mesh = clipped_mesh(self.meshes[i], self.clip)
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(mesh)
                candidate = vtk.vtkActor()
                candidate.SetMapper(mapper)
                candidate.GetProperty().DeepCopy(actor.GetProperty())
                picker.AddPickList(candidate)
        if not picker.Pick(x, y, 0, self.renderer):
            raise ValueError(
                "No visible surface was hit. Check the locked layer or click position."
            )
        point = np.asarray(picker.GetPickPosition())
        return point

    def add_measurement(self, measurement, stale=False):
        """Add a picked-point measurement and its physical label to the scene."""
        points = measurement.points
        for point in points:
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(*point)
            sphere.SetRadius(min(self.volume.spacing) * 1.2)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.75, 0.25)
            actor.PickableOff()
            self.renderer.AddActor(actor)
            self.annotations.append(actor)
        if len(points) > 1:
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(points))
            vtk_points = vtk.vtkPoints()
            for i, point in enumerate(points):
                vtk_points.InsertNextPoint(point)
                line.GetPointIds().SetId(i, i)
            cells = vtk.vtkCellArray()
            cells.InsertNextCell(line)
            poly = vtk.vtkPolyData()
            poly.SetPoints(vtk_points)
            poly.SetLines(cells)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.75, 0.25)
            actor.GetProperty().SetLineWidth(2.0)
            actor.PickableOff()
            self.renderer.AddActor(actor)
            self.annotations.append(actor)
        text = (
            measurement.text
            if measurement.kind == "annotation"
            else f"{measurement.value:.2f} {'mm' if measurement.kind == 'distance' else '°'}"
        )
        label = vtk.vtkBillboardTextActor3D()
        label.SetInput(text + (" *" if stale else ""))
        label.SetPosition(*points[-1])
        label.GetTextProperty().SetFontSize(15)
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        if font.exists():
            label.GetTextProperty().SetFontFamily(vtk.VTK_FONT_FILE)
            label.GetTextProperty().SetFontFile(str(font))
        label.GetTextProperty().SetColor(1.0, 0.8, 0.35)
        label.PickableOff()
        self.renderer.AddActor(label)
        self.annotations.append(label)
        self.render()

    def camera_state(self):
        """Return serializable physical camera parameters."""
        c = self.renderer.GetActiveCamera()
        return {
            "position": list(c.GetPosition()),
            "focal_point": list(c.GetFocalPoint()),
            "view_up": list(c.GetViewUp()),
            "parallel": bool(c.GetParallelProjection()),
            "parallel_scale": c.GetParallelScale(),
            "view_angle": c.GetViewAngle(),
            "aspect_ratio": self.window.GetSize()[0] / max(self.window.GetSize()[1], 1),
        }

    def restore_camera(self, state):
        """Restore physical camera parameters and fit the current viewport."""
        c = self.renderer.GetActiveCamera()
        c.SetPosition(*state["position"])
        c.SetFocalPoint(*state["focal_point"])
        c.SetViewUp(*state["view_up"])
        c.SetParallelProjection(state.get("parallel", False))
        c.SetParallelScale(state.get("parallel_scale", 100.0))
        angle = state.get("view_angle", 30.0)
        aspect = self.window.GetSize()[0] / max(self.window.GetSize()[1], 1)
        ratio = max(1.0, state.get("aspect_ratio", aspect) / max(aspect, 0.01))
        c.SetViewAngle(
            float(np.degrees(2 * np.arctan(np.tan(np.radians(angle) / 2) * ratio)))
        )
        if state.get("parallel", False):
            c.SetParallelScale(state.get("parallel_scale", 100.0) * ratio)
        self.renderer.ResetCameraClippingRange()
        self.render()

    def render(self):
        """Render the current VTK scene."""
        if self.volume is not None:
            self.renderer.ResetCameraClippingRange()
            self.window.Render()

    def screenshot(self, path, scale=2):
        """Save a scaled RGB capture of the VTK framebuffer."""
        self.render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(self.window)
        capture.SetScale(scale)
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(capture.GetOutputPort())
        writer.Write()

    def capabilities(self):
        """Report the active OpenGL renderer and transparency capabilities."""
        return {
            "gpu_volume": bool(
                self.iso_mapper.IsRenderSupported(self.window, self.iso_property)
            ),
            "depth_peeling_used": bool(
                self.renderer.GetLastRenderingUsedDepthPeeling()
            ),
            "opengl": self.window.ReportCapabilities(),
        }
