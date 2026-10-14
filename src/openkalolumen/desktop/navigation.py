"""Coordinate slice navigation, clipping, and projection controls."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QFileDialog,
)

from openkalolumen.imaging.drr import project_drr
from openkalolumen.imaging.mpr import BASES, display_slice, plane_basis, reslice


class NavigationMixin:
    """Coordinate slice navigation, clipping, and projection controls."""

    def _clip_changed(self):
        if not self.volume:
            return
        self.clip.enabled = self.clip_check.isChecked()
        self.clip.flip = self.clip_flip.isChecked()
        self.clip.section = self.section_combo.currentData()
        self.plane_widget.SetEnabled(
            self.clip.enabled and self.clip_handles.isChecked()
        )
        self.engine.set_clip(self.clip)
        self.section_timer.start()
        self.mpr_timer.start()

    def _clip_axis_changed(self):
        axis = self.clip_axis.currentData()
        if axis == "free" or not self.volume:
            return
        self.clip.normal = list(
            {"axial": (0, 0, 1), "coronal": (0, 1, 0), "sagittal": (1, 0, 0)}[axis]
        )
        self.plane_rep.SetNormal(*self.clip.normal)
        self._clip_position_changed(self.clip_slider.value())

    def _clip_position_changed(self, value):
        if not self.volume:
            return
        n = np.asarray(self.clip.normal)
        n = n / np.linalg.norm(n)
        lo, hi = self._clip_bounds()
        a = float(np.where(n >= 0, lo, hi) @ n)
        b = float(np.where(n >= 0, hi, lo) @ n)
        target = a + (b - a) * value / 1000
        point = np.asarray(self.clip.origin) + n * (
            target - np.asarray(self.clip.origin) @ n
        )
        self.clip.origin = list(point)
        self.plane_rep.SetOrigin(*point)
        self.engine.set_clip(self.clip)
        self.section_timer.start()
        self.mpr_timer.start()

    def _clip_bounds(self):
        if self.mask is None or not self.mask.any():
            return self.volume.bounds
        from itertools import product

        spans = [
            np.flatnonzero(self.mask.any(axis=tuple(a for a in range(3) if a != axis)))
            for axis in (2, 1, 0)
        ]
        corners = self.volume.to_world(
            list(product(*[(int(a[0]), int(a[-1])) for a in spans]))
        )
        return np.stack((corners.min(0), corners.max(0)))

    def _place_clip_widget(self):
        lo, hi = self._clip_bounds()
        self.plane_rep.PlaceWidget([lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])
        self.plane_rep.SetOrigin(*self.clip.origin)
        self.plane_rep.SetNormal(*self.clip.normal)
        n = np.asarray(self.clip.normal)
        n = n / np.linalg.norm(n)
        a = float(np.where(n >= 0, lo, hi) @ n)
        b = float(np.where(n >= 0, hi, lo) @ n)
        value = round(1000 * (np.asarray(self.clip.origin) @ n - a) / max(b - a, 1))
        self.clip_slider.blockSignals(True)
        self.clip_slider.setValue(value)
        self.clip_slider.blockSignals(False)

    def _plane_changed(self, *_):
        self.clip.origin = list(self.plane_rep.GetOrigin())
        self.clip.normal = list(self.plane_rep.GetNormal())
        self.clip_axis.blockSignals(True)
        self.clip_axis.setCurrentIndex(3)
        self.clip_axis.blockSignals(False)
        self.engine.set_clip(self.clip)
        self.section_timer.start()
        self.mpr_timer.start()

    def update_section(self):
        """Refresh the current physical section after control changes settle."""
        if self.volume:
            self.engine.set_section(self.clip, self.level.value(), self.width.value())
            if "sampled_material_field" in self.engine.section_sources.values():
                self.statusBar().showMessage(
                    "Material sections sample the volume and retain cavities. Edge precision depends on sampling resolution."
                )

    def _window_changed(self):
        self.mpr_timer.start()
        self.section_timer.start()

    def refresh_mpr(self):
        """Schedule orthogonal or oblique slices using the current crosshair."""
        if not self.volume:
            return
        volume = self.volume
        point = self.crosshair.copy()
        clip = replace(self.clip)
        oblique = self.oblique_check.isChecked()
        labels = self.labels
        level, width = self.level.value(), self.width.value()

        def compute(progress, cancelled):
            output = {}
            for name, (u, v) in BASES.items():
                origin = point
                if name == "axial" and oblique:
                    u, v = plane_basis(clip.effective_normal)
                    origin = np.asarray(clip.origin)
                values, spacing = reslice(volume, origin, u, v, 320)
                label_slice = (
                    reslice(volume, origin, u, v, 320, array=labels)[0]
                    if labels is not None
                    else None
                )
                output[name] = (
                    display_slice(values, level, width, label_slice),
                    spacing,
                    u,
                    v,
                    origin,
                )
            return output

        self.tasks.submit("mpr", compute, self._mpr_ready, self.error)

    def _mpr_ready(self, output):
        self.mpr_state = output
        for name, (rgb, spacing, u, v, origin) in output.items():
            panel = self.slice_panels[name]
            panel.canvas.set_image(
                rgb, f"LPS {origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f} mm", True
            )

    def sync_slice_sliders(self):
        """Synchronize slice sliders with the physical crosshair."""
        if not self.volume:
            return
        axes = {"axial": 2, "coronal": 1, "sagittal": 0}
        lo, hi = self.volume.bounds
        for name, panel in self.slice_panels.items():
            axis = axes[name]
            value = int(
                1000 * (self.crosshair[axis] - lo[axis]) / max(hi[axis] - lo[axis], 1)
            )
            panel.slider.blockSignals(True)
            panel.slider.setValue(value)
            panel.slider.blockSignals(False)

    def move_slice(self, name, value):
        """Move the crosshair along a principal LPS axis using a 0–1000 slider."""
        if not self.volume:
            return
        axis = {"axial": 2, "coronal": 1, "sagittal": 0}[name]
        lo, hi = self.volume.bounds
        self.crosshair[axis] = lo[axis] + (hi[axis] - lo[axis]) * value / 1000
        if self.clip.enabled:
            self.clip.origin = list(self.crosshair)
            self.plane_rep.SetOrigin(*self.crosshair)
            self.engine.set_clip(self.clip)
            self.section_timer.start()
        self.mpr_timer.start()

    def click_slice(self, name, x, y):
        """Move the physical crosshair to a normalized slice-image location."""
        if not hasattr(self, "mpr_state") or name not in self.mpr_state:
            return
        rgb, spacing, u, v, origin = self.mpr_state[name]
        self.crosshair = (
            np.asarray(origin)
            + u * (x - 0.5) * rgb.shape[1] * spacing
            + v * (0.5 - y) * rgb.shape[0] * spacing
        )
        self.sync_slice_sliders()
        if self.clip.enabled:
            self.clip.origin = list(self.crosshair)
            self.plane_rep.SetOrigin(*self.crosshair)
            self.engine.set_clip(self.clip)
            self.section_timer.start()
        self.mpr_timer.start()

    def _end_camera_interaction(self, *_):
        self.engine.set_quality(True, self.scattering.isChecked())
        self.engine.render()
        if self.drr_result is not None:
            self.drr_canvas.caption = "Camera changed · recompute the projection"
            self.drr_canvas.update()

    def camera_direction(self, direction):
        """Orient the camera along a principal physical direction."""
        if not self.require_volume():
            return
        c = self.engine.renderer.GetActiveCamera()
        distance = np.linalg.norm(np.asarray(c.GetPosition()) - c.GetFocalPoint())
        vector = (
            np.array([0.0, -1.0, 0.0])
            if direction == "front"
            else np.array([-1.0, 0.0, 0.0])
        )
        c.SetFocalPoint(*self.volume.center)
        c.SetPosition(*(self.volume.center + vector * distance))
        c.SetViewUp(0, 0, 1)
        self.engine.render()

    def reset_camera(self):
        """Fit the camera to the currently displayed anatomy."""
        if self.volume:
            self.engine.renderer.ResetCamera()
            self.engine.render()

    def calculate_drr(self):
        """Schedule a calibrated projection from the current camera."""
        if not self.require_volume():
            return
        volume = self.volume
        camera = self.engine.camera_state()
        clip = replace(self.clip) if self.drr_clip.isChecked() else None
        mu = self.mu_water.value()
        settings = {
            "mu_water": mu,
            "apply_clip": self.drr_clip.isChecked(),
            "camera": camera,
            "fingerprint": volume.fingerprint,
            "clip": vars(clip) if clip else None,
        }

        def ready(result):
            self.drr_settings = settings
            self.drr_result = result
            rgb = np.repeat(
                (np.clip(1 - result, 0, 1) * 255).astype(np.uint8)[..., None],
                3,
                axis=-1,
            )
            self.drr_canvas.set_image(
                rgb, "Attenuation projection · single-energy approximation"
            )
            self.report_progress(100, "DRR completed")

        self.tasks.submit(
            "drr",
            lambda p, c: project_drr(
                volume, camera, 256, mu, clip=clip, progress=p, cancelled=c
            ),
            ready,
            self.error,
            self.report_progress,
        )

    def export_drr(self):
        """Save the completed projection as a 16-bit PNG."""
        if self.drr_result is None:
            self.statusBar().showMessage("Compute a DRR first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export DRR", "drr.png", "PNG (*.png)"
        )
        if path:
            from PIL import Image

            Image.fromarray(
                (np.clip(1 - self.drr_result, 0, 1) * 65535).astype(np.uint16)
            ).save(path)
            Path(path + ".json").write_text(
                json.dumps(
                    {**self.drr_settings, "approximation": "monoenergetic"}, indent=2
                )
            )
