"""Persist scenes, export meshes, and capture the application."""

from __future__ import annotations

import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import vtk
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
)

from openkalolumen.imaging.volume import (
    aligned_labels,
    synthetic_volume,
)
from openkalolumen.render.geometry import (
    export_mesh,
    extract_surface,
)
from openkalolumen.state import (
    read_scene,
    save_scene,
)


class SceneMixin:
    """Persist scenes, export meshes, and capture the application."""

    def capture(self):
        """Choose a destination for a detailed three-dimensional screenshot."""
        if not self.require_volume():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save detailed 3D screenshot", "openkalolumen.png", "PNG (*.png)"
        )
        if path:
            self.engine.set_quality(True, self.scattering.isChecked())
            self.engine.screenshot(path, 2)
            self.statusBar().showMessage(f"Screenshot saved: {path}")

    def capture_interface(self, path):
        """Capture Qt chrome plus its native OpenGL child from the real framebuffer."""
        from vtk.util.numpy_support import vtk_to_numpy

        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(self.engine.window)
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.Update()
        data = capture.GetOutput()
        w, h, _ = data.GetDimensions()
        rgb = np.ascontiguousarray(
            np.flipud(vtk_to_numpy(data.GetPointData().GetScalars()).reshape(h, w, 3))
        )
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pixmap = self.grab()
        painter = QPainter(pixmap)
        rect = self.vtk_widget.rect()
        rect.moveTopLeft(self.vtk_widget.mapTo(self, rect.topLeft()))
        painter.drawImage(rect, image)
        painter.end()
        pixmap.save(str(path))

    def export_selected(self):
        """Export the selected original-data mesh with optional physical clipping."""
        if not self.require_volume():
            return
        index = self.layer_list.currentRow()
        if index not in self.engine.meshes or not self.engine.geometry_current:
            self.statusBar().showMessage(
                "Select an isosurface or organ layer with completed detailed geometry."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export selected layer",
            self.layers[index].name + ".stl",
            "STL (*.stl);;Colored PLY (*.ply)",
        )
        if not path:
            return
        apply_clip = False
        if self.clip.enabled:
            choice, ok = QInputDialog.getItem(
                self,
                "Export geometry",
                "Apply the current clipping plane?",
                ["Complete structure", "Apply current clip"],
                0,
                False,
            )
            if not ok:
                return
            apply_clip = choice.startswith("Apply")
        mesh = self.engine.meshes[index]
        layer = replace(self.layers[index])
        volume = self.volume
        clip = replace(self.clip) if apply_clip else None
        mask, labels = self.mask, self.labels

        def export_original(progress, cancelled):
            original = replace(layer, smoothing_mm=0)
            geometry = mesh
            if layer.smoothing_mm > 0:
                progress(
                    5,
                    "Disabling display smoothing and extracting original CT geometry for export",
                )
                geometry = extract_surface(volume, original, mask, labels)
            return export_mesh(path, geometry, original, volume, clip)

        self.tasks.submit(
            "export",
            export_original,
            lambda _: self.statusBar().showMessage(
                f"Exported original scalar geometry: {path}(LPS / mm)"
            ),
            self.error,
            self.report_progress,
        )

    def save_current_scene(self):
        """Choose a destination and save the current scene."""
        if not self.require_volume():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image scene", "scene.json", "Scenes (*.json)"
        )
        if not path:
            return
        try:
            self.save_scene_to(path)
            self.statusBar().showMessage(f"Scene saved: {path}")
        except Exception:
            self.error(traceback.format_exc())

    def save_scene_to(self, path):
        """Persist the scene and any derived visibility mask beside it."""
        directory = Path(path).resolve().parent
        if self.mask is not None:
            import SimpleITK as sitk

            mask_path = directory / (Path(path).stem + ".mask.nrrd")
            image = sitk.GetImageFromArray(self.mask.astype(np.uint8))
            image.CopyInformation(self.volume.to_sitk())
            sitk.WriteImage(image, str(mask_path), True)
            self.mask_path = str(mask_path)
        save_scene(
            path,
            volume=self.volume,
            layers=self.layers,
            clip=self.clip,
            camera=self.engine.camera_state(),
            crosshair=self.crosshair,
            measurements=self.measurements,
            mode=self.engine.mode,
            labels_path=self.labels_path,
            mask_path=self.mask_path,
            drr=self.drr_settings,
            window={
                "level": self.level.value(),
                "width": self.width.value(),
                "oblique": self.oblique_check.isChecked(),
            },
            quality={
                "scattering": self.scattering.isChecked(),
                "ssao": self.ssao_check.isChecked(),
            },
        )

    def open_scene(self):
        """Choose a scene file and load its referenced image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open scene", str(self.root), "Scenes (*.json)"
        )
        if path:
            self.open_scene_path(path)

    def open_scene_path(self, path):
        """Read a validated scene and restore its synthetic or external image."""
        try:
            scene = read_scene(path)
            source = scene["source"]
            if source.startswith("synthetic:"):
                self.set_volume(synthetic_volume(), scene)
            else:
                self.open_path(
                    source,
                    scene.get("modality", "CT"),
                    scene.get("hu_calibrated", True),
                    scene,
                )
        except Exception:
            self.error(traceback.format_exc())

    def _restore_scene(self, scene):
        """Restore desktop state and reapply the camera after native viewport resizing."""
        self.layers = scene["layers"]
        self.clip = scene["clip"]
        self.crosshair = np.asarray(
            scene.get("crosshair", self.volume.center), dtype=float
        )
        self.measurements = scene["measurements"]
        self._populate_layers()
        self.engine.configure_layers(self.layers)
        self.mode_combo.setCurrentIndex(
            max(0, self.mode_combo.findData(scene.get("mode", "surface")))
        )
        self.engine.restore_camera(scene["camera"])

        fingerprint = self.volume.fingerprint
        camera = scene["camera"]
        self._pending_camera_state = camera
        QTimer.singleShot(
            150,
            lambda: (
                self.engine.restore_camera(camera)
                if self._pending_camera_state is camera
                and self.volume
                and self.volume.fingerprint == fingerprint
                else None
            ),
        )
        for widget, value in [
            (self.clip_check, self.clip.enabled),
            (self.clip_flip, self.clip.flip),
        ]:
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)
        self.section_combo.blockSignals(True)
        self.section_combo.setCurrentIndex(
            max(0, self.section_combo.findData(self.clip.section))
        )
        self.section_combo.blockSignals(False)
        self.plane_rep.SetOrigin(*self.clip.origin)
        self.plane_rep.SetNormal(*self.clip.normal)
        self.plane_widget.SetEnabled(
            self.clip.enabled and self.clip_handles.isChecked()
        )
        normal = np.asarray(self.clip.normal, dtype=float)
        normal /= np.linalg.norm(normal)
        axis = "free"
        for name, vector in (
            ("sagittal", [1, 0, 0]),
            ("coronal", [0, 1, 0]),
            ("axial", [0, 0, 1]),
        ):
            if np.allclose(normal, vector):
                axis = name
        self.clip_axis.blockSignals(True)
        self.clip_axis.setCurrentIndex(self.clip_axis.findData(axis))
        self.clip_axis.blockSignals(False)
        self.engine.set_clip(self.clip)
        self.labels_path = scene.get("labels_path", "")
        if self.labels_path:
            self.labels = aligned_labels(self.labels_path, self.volume)
        self.mask_path = scene.get("mask_path", "")
        if self.mask_path:
            self.mask = aligned_labels(self.mask_path, self.volume) > 0
            self.engine.set_mask(self.mask)
        self._place_clip_widget()
        for m in self.measurements:
            self.engine.add_measurement(m, True)
        self.drr_settings = scene.get("drr", self.drr_settings)
        self.mu_water.setValue(self.drr_settings.get("mu_water", 0.02))
        self.drr_clip.setChecked(self.drr_settings.get("apply_clip", False))
        settings = scene.get("window", {})
        self.level.setValue(settings.get("level", 60))
        self.width.setValue(settings.get("width", 400))
        self.oblique_check.setChecked(settings.get("oblique", False))
        quality = scene.get("quality", {})
        self.scattering.setChecked(quality.get("scattering", False))
        self.ssao_check.setChecked(quality.get("ssao", False))
        self.section_timer.start()
