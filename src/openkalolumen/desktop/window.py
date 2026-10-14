from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import vtk
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from openkalolumen.config import AppPaths
from openkalolumen.desktop.navigation import NavigationMixin
from openkalolumen.desktop.scenes import SceneMixin
from openkalolumen.desktop.segmentation import SegmentationMixin
from openkalolumen.desktop.tasks import TaskManager
from openkalolumen.desktop.widgets import ImageCanvas, SlicePanel
from openkalolumen.imaging.volume import (
    aligned_labels,
    connected_region,
    dicom_series,
    load_volume,
    read_dicom,
    synthetic_volume,
)
from openkalolumen.render.engine import RenderEngine
from openkalolumen.render.geometry import (
    build_surfaces,
)
from openkalolumen.samples import SAMPLES, download_sample
from openkalolumen.state import (
    ClipState,
    Layer,
    Measurement,
    default_layers,
)

STYLE = """
QMainWindow,QWidget { background: #121a26; color: #dae4f1; font-size: 12px; }
QToolBar { background: #182231; border: 0; spacing: 6px; padding: 7px; }
QToolButton,QPushButton { background: #233248; border: 1px solid #33445c; border-radius: 5px; padding: 7px 11px; }
QToolButton:hover,QPushButton:hover { background: #314764; }
QPushButton:disabled { color: #64758a; }
QGroupBox { border: 1px solid #29394e; border-radius: 7px; margin-top: 12px; padding: 14px 8px 8px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #83d4d1; }
QListWidget { background: #101722; border: 1px solid #29394e; border-radius: 5px; padding: 3px; }
QListWidget::item { padding: 10px 6px; border-radius: 4px; }
QListWidget::item:selected { background: #294159; color: #efffff; }
QComboBox,QDoubleSpinBox,QLineEdit { background: #1b2839; border: 1px solid #344760; border-radius: 4px; padding: 5px; }
QComboBox::drop-down { border: 0; width: 20px; }
QSlider::groove:horizontal { background: #2a3b50; height: 5px; border-radius: 2px; }
QSlider::handle:horizontal { background: #69c9c1; width: 12px; margin: -4px 0; border-radius: 6px; }
QProgressBar { border: 1px solid #2d4155; background: #192638; border-radius: 3px; text-align: center; }
QProgressBar::chunk { background: #2d938e; }
QScrollArea { border: 0; }
QSplitter::handle { background: #233247; }
QCheckBox { spacing: 6px; }
QStatusBar { background: #0e1621; }
QLabel#muted { color: #8a9cb2; }
QLabel#brand { color: #ecf3ff; font-size: 21px; font-weight: 700; }
"""


def spin(low, high, value, decimals=1):
    """Create a bounded decimal spin box with an initial value."""
    widget = QDoubleSpinBox()
    widget.setRange(low, high)
    widget.setDecimals(decimals)
    widget.setValue(value)
    return widget


def button(text, callback):
    """Create a push button connected to a callback."""
    widget = QPushButton(text)
    widget.clicked.connect(callback)
    return widget


class VolumeWidget(QVTKRenderWindowInteractor):
    """Avoid Cocoa expose -> swapBuffer -> expose storms in Qt's native VTK widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._resize_render = QTimer(self)
        self._resize_render.setSingleShot(True)
        self._resize_render.setInterval(80)
        self._resize_render.timeout.connect(self.Render)

    def paintEvent(self, event):
        """Let explicit VTK renders handle native buffer updates.

        Cocoa emits an expose event after each buffer swap. Rendering from
        that event would recurse and starve the Qt event loop.
        """
        pass

    def resizeEvent(self, event):
        """Resize the native VTK widget and defer its next render."""
        super().resizeEvent(event)
        if hasattr(self, "_resize_render"):
            self._resize_render.start()


class MainWindow(NavigationMixin, SceneMixin, SegmentationMixin, QMainWindow):
    """Own application state and connect desktop controls to imaging tasks."""

    def __init__(self, paths=None):
        super().__init__()
        self.paths = paths or AppPaths.resolve()
        self.root = self.paths.data
        self.root.mkdir(parents=True, exist_ok=True)
        self.setWindowTitle("openKaloLumen · 3D imaging and lumen inspection")
        self.resize(1500, 1000)
        self.setStyleSheet(STYLE)
        self.volume = None
        self.layers = []
        self.labels = None
        self.labels_path = ""
        self.mask = None
        self.mask_path = ""
        self.crosshair = np.zeros(3)
        self.clip = ClipState(normal=[0.0, 1.0, 0.0])
        self.measurements = []
        self.pick_points = []
        self.pick_kind = None
        self.tasks = TaskManager(self)
        self.seg_process = None
        self.seg_buffer = ""
        self.drr_result = None
        self.drr_settings = {"mu_water": 0.02, "apply_clip": False}
        self.last_error = ""
        self._pending_camera_state = None
        self.mesh_timer = QTimer(self)
        self.mesh_timer.setSingleShot(True)
        self.mesh_timer.setInterval(250)
        self.mesh_timer.timeout.connect(self.rebuild_meshes)
        self.section_timer = QTimer(self)
        self.section_timer.setSingleShot(True)
        self.section_timer.setInterval(80)
        self.section_timer.timeout.connect(self.update_section)
        self.mpr_timer = QTimer(self)
        self.mpr_timer.setSingleShot(True)
        self.mpr_timer.setInterval(60)
        self.mpr_timer.timeout.connect(self.refresh_mpr)
        self._toolbar()
        self._build_ui()
        self._statusbar()
        self.vtk_widget.Initialize()
        self.engine = RenderEngine(self.vtk_widget.GetRenderWindow())
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        interactor.AddObserver("LeftButtonPressEvent", self._vtk_click, 1.0)
        interactor.AddObserver(
            "StartInteractionEvent", lambda *_: self.engine.set_quality(False)
        )
        interactor.AddObserver("EndInteractionEvent", self._end_camera_interaction)
        self.plane_rep = vtk.vtkImplicitPlaneRepresentation()
        self.plane_rep.SetPlaceFactor(1.0)
        self.plane_rep.DrawPlaneOff()
        self.plane_rep.DrawOutlineOff()
        self.plane_rep.OutlineTranslationOff()
        self.plane_rep.ScaleEnabledOff()
        self.plane_rep.GetOutlineProperty().SetColor(0.2, 0.8, 0.8)
        self.plane_rep.GetOutlineProperty().SetOpacity(0.35)
        self.plane_rep.GetEdgesProperty().SetColor(0.2, 0.8, 0.8)
        self.plane_rep.GetNormalProperty().SetColor(0.35, 0.85, 0.8)
        self.plane_rep.GetSelectedNormalProperty().SetColor(0.65, 1.0, 0.95)
        self.plane_widget = vtk.vtkImplicitPlaneWidget2()
        self.plane_widget.SetInteractor(interactor)
        self.plane_widget.SetRepresentation(self.plane_rep)
        self.plane_widget.AddObserver("InteractionEvent", self._plane_changed)
        axes = vtk.vtkAxesActor()
        axes.SetXAxisLabelText("L")
        axes.SetYAxisLabelText("P")
        axes.SetZAxisLabelText("S")
        self.orientation = vtk.vtkOrientationMarkerWidget()
        self.orientation.SetOrientationMarker(axes)
        self.orientation.SetInteractor(interactor)
        self.orientation.SetViewport(0, 0, 0.16, 0.16)
        self.orientation.SetEnabled(1)
        self.orientation.InteractiveOff()

    def _toolbar(self):
        toolbar = QToolBar("Main toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        brand = QLabel("openKaloLumen")
        brand.setObjectName("brand")
        toolbar.addWidget(brand)
        toolbar.addSeparator()
        for title, callback in [
            ("Open image", self.open_file),
            ("DICOM folder", self.open_dicom),
            ("Public CT samples", self.choose_sample),
            ("Synthetic lumen", self.open_synthetic),
            ("Open scene", self.open_scene),
            ("Save scene", self.save_current_scene),
            ("Detailed screenshot", self.capture),
            ("Export mesh", self.export_selected),
        ]:
            action = QAction(title, self)
            action.triggered.connect(callback)
            toolbar.addAction(action)

    def _build_ui(self):
        splitter = QSplitter()
        self.setCentralWidget(splitter)
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(12, 12, 12, 12)
        title = QLabel("Data and structures")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)
        self.data_info = QLabel(
            "Load CT/MRI or open a public sample.\nImages and computation stay on this device."
        )
        self.data_info.setObjectName("muted")
        self.data_info.setWordWrap(True)
        layout.addWidget(self.data_info)
        self.layer_list = QListWidget()
        self.layer_list.setMinimumHeight(210)
        self.layer_list.currentRowChanged.connect(self._selected_layer_changed)
        self.layer_list.itemChanged.connect(self._layer_visibility)
        layout.addWidget(self.layer_list, 1)
        row = QHBoxLayout()
        row.addWidget(button("Add layer", self.add_layer))
        row.addWidget(button("Remove", self.remove_layer))
        layout.addLayout(row)
        layout.addWidget(button("Import organ labels", self.import_labels))
        layout.addWidget(button("Segment organs", self.start_segmentation))
        layout.addWidget(button("Filter by label", self.filter_labels))
        self.region_lower = spin(-100000, 100000, 40)
        self.region_upper = spin(-100000, 100000, 2000)
        group = QGroupBox("Connected region")
        form = QFormLayout(group)
        form.addRow("Lower", self.region_lower)
        form.addRow("Upper", self.region_upper)
        form.addRow(button("Pick region to retain", lambda: self.start_pick("keep")))
        form.addRow(button("Pick region to remove", lambda: self.start_pick("remove")))
        form.addRow(button("Limit region with ROI box", self.edit_roi))
        form.addRow(button("Show all structures", self.clear_mask))
        layout.addWidget(group)
        self.instructions = QLabel(
            "Left drag: rotate · Middle drag: pan · Wheel: zoom\nDrag the clipping widget to inspect the lumen\nSelect a layer to adjust thresholds and materials"
        )
        self.instructions.setObjectName("muted")
        self.instructions.setWordWrap(True)
        layout.addWidget(self.instructions)
        left.setMinimumWidth(235)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(235)
        splitter.addWidget(left_scroll)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 8, 0, 0)
        view_row = QHBoxLayout()
        self.mode_combo = QComboBox()
        for text, value in [
            ("Surface view", "surface"),
            ("Volume view", "volume"),
            ("Surface + volume", "hybrid"),
        ]:
            self.mode_combo.addItem(text, value)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        view_row.addWidget(self.mode_combo)
        self.four_views = QCheckBox("3D + three slices")
        self.four_views.setChecked(True)
        self.four_views.toggled.connect(self._layout_changed)
        view_row.addWidget(self.four_views)
        self.oblique_check = QCheckBox("Oblique MPR")
        self.oblique_check.toggled.connect(lambda _: self.mpr_timer.start())
        view_row.addWidget(self.oblique_check)
        view_row.addWidget(button("Front", lambda: self.camera_direction("front")))
        view_row.addWidget(button("Side", lambda: self.camera_direction("side")))
        view_row.addWidget(button("Reset", self.reset_camera))
        view_row.addStretch()
        center_layout.addLayout(view_row)
        self.views_grid = QGridLayout()
        self.views_grid.setSpacing(5)
        self.vtk_widget = VolumeWidget(center)
        self.vtk_widget.setMinimumSize(350, 320)
        self.views_grid.addWidget(self.vtk_widget, 0, 0, 3, 1)
        self.slice_panels = {}
        for name, title, color, pos in [
            ("axial", "Axial", "#e9a884", (0, 1)),
            ("coronal", "Coronal", "#8cdbb2", (1, 1)),
            ("sagittal", "Sagittal", "#8aaeea", (2, 1)),
        ]:
            panel = SlicePanel(title, color)
            panel.canvas.directions = {
                "axial": ("R", "L", "A", "P"),
                "coronal": ("R", "L", "S", "I"),
                "sagittal": ("A", "P", "S", "I"),
            }[name]
            panel.moved.connect(lambda n, name=name: self.move_slice(name, n))
            panel.clicked.connect(lambda x, y, name=name: self.click_slice(name, x, y))
            self.slice_panels[name] = panel
            self.views_grid.addWidget(panel, *pos)
        self.views_grid.setColumnStretch(0, 3)
        self.views_grid.setColumnStretch(1, 1)
        self.views_grid.setRowStretch(0, 1)
        self.views_grid.setRowStretch(1, 1)
        self.views_grid.setRowStretch(2, 1)
        center_layout.addLayout(self.views_grid, 1)
        self.view_status = QLabel("Waiting for an image")
        self.view_status.setStyleSheet("padding: 6px; color: #8ecbc9;")
        center_layout.addWidget(self.view_status)
        splitter.addWidget(center)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 3, 10, 10)
        layer_group = QGroupBox("Layer parameters")
        form = QFormLayout(layer_group)
        self.layer_kind = QComboBox()
        self.layer_kind.addItem("Isosurface", "isosurface")
        self.layer_kind.addItem("Volume interval", "volume_band")
        self.layer_kind.addItem("Surface interval", "surface_band")
        self.layer_kind.addItem("Finite tissue", "solid_band")
        self.layer_kind.currentIndexChanged.connect(self._layer_attributes)
        form.addRow("Representation", self.layer_kind)
        self.threshold = spin(-100000, 100000, 50)
        self.threshold.valueChanged.connect(self._threshold_spin_changed)
        form.addRow("Threshold", self.threshold)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 10000)
        self.threshold_slider.valueChanged.connect(self._threshold_slider_changed)
        form.addRow(self.threshold_slider)
        self.upper = spin(-100000, 100000, 300)
        self.upper.valueChanged.connect(self._layer_attributes)
        form.addRow("Upper limit", self.upper)
        self.opacity = spin(0, 1, 0.92, 2)
        self.opacity.setSingleStep(0.05)
        self.opacity.valueChanged.connect(self._layer_attributes)
        form.addRow("Opacity", self.opacity)
        self.roughness = spin(0, 1, 0.55, 2)
        self.roughness.setSingleStep(0.05)
        self.roughness.valueChanged.connect(self._layer_attributes)
        form.addRow("Roughness", self.roughness)
        self.surface_smoothing = spin(0, 3, 0, 1)
        self.surface_smoothing.setSingleStep(0.1)
        self.surface_smoothing.setToolTip(
            "Changes only the 3D display field, preserving original CT and MPR. Filtering moves threshold boundaries. Disable smoothing for measurements; export always extracts original scalar data."
        )
        self.surface_smoothing.valueChanged.connect(self._layer_attributes)
        form.addRow("Smoothing / mm", self.surface_smoothing)
        self.color_button = button("Choose...", self.choose_color)
        form.addRow("Exterior color", self.color_button)
        self.inside_color_button = button("Choose...", lambda: self.choose_color(True))
        form.addRow("Inner-wall color", self.inside_color_button)
        self.cut_color_button = button("Choose...", lambda: self.choose_color(cut=True))
        form.addRow("Section color", self.cut_color_button)
        self.coat = spin(0, 1, 0.35, 2)
        self.coat.setSingleStep(0.05)
        self.coat.valueChanged.connect(self._layer_attributes)
        form.addRow("Clearcoat", self.coat)
        self.scattering = QCheckBox("Volume scattering")
        self.scattering.toggled.connect(self._quality_changed)
        form.addRow(self.scattering)
        self.ssao_check = QCheckBox("Ambient occlusion")
        self.ssao_check.setToolTip("Available for opaque surface rendering.")
        self.ssao_check.setChecked(True)
        self.ssao_check.toggled.connect(self._quality_changed)
        form.addRow(self.ssao_check)
        right_layout.addWidget(layer_group)
        clipping = QGroupBox("Clipping and lumen")
        form = QFormLayout(clipping)
        self.clip_check = QCheckBox("Enable clipping")
        self.clip_check.toggled.connect(self._clip_changed)
        form.addRow(self.clip_check)
        self.clip_handles = QCheckBox("Show 3D clipping widget")
        self.clip_handles.toggled.connect(self._clip_changed)
        form.addRow(self.clip_handles)
        self.clip_axis = QComboBox()
        for text, axis in [
            ("Coronal", "coronal"),
            ("Axial", "axial"),
            ("Sagittal", "sagittal"),
            ("Free orientation", "free"),
        ]:
            self.clip_axis.addItem(text, axis)
        self.clip_axis.currentIndexChanged.connect(self._clip_axis_changed)
        form.addRow("Orientation", self.clip_axis)
        self.clip_slider = QSlider(Qt.Orientation.Horizontal)
        self.clip_slider.setRange(0, 1000)
        self.clip_slider.setValue(500)
        self.clip_slider.valueChanged.connect(self._clip_position_changed)
        form.addRow("Position", self.clip_slider)
        self.clip_flip = QCheckBox("Reverse retained side")
        self.clip_flip.toggled.connect(self._clip_changed)
        form.addRow(self.clip_flip)
        self.section_combo = QComboBox()
        for text, key in [
            ("Open lumen · retain material sections", "open"),
            ("Grayscale slice", "gray"),
            ("Tissue section", "tissue"),
        ]:
            self.section_combo.addItem(text, key)
        self.section_combo.currentIndexChanged.connect(self._clip_changed)
        form.addRow("Section display", self.section_combo)
        right_layout.addWidget(clipping)
        window = QGroupBox("Slices and measurements")
        form = QFormLayout(window)
        self.level = spin(-100000, 100000, 60)
        self.width = spin(1, 100000, 400)
        for widget in (self.level, self.width):
            widget.valueChanged.connect(self._window_changed)
        form.addRow("Window level", self.level)
        form.addRow("Window width", self.width)
        self.lock_layer = QCheckBox("Pick only the selected layer")
        form.addRow(self.lock_layer)
        row = QHBoxLayout()
        row.addWidget(button("Distance", lambda: self.start_pick("distance")))
        row.addWidget(button("Angle", lambda: self.start_pick("angle")))
        row.addWidget(button("Annotation", lambda: self.start_pick("annotation")))
        form.addRow(row)
        form.addRow(button("Cancel picking", self.cancel_pick))
        self.measurement_list = QListWidget()
        self.measurement_list.setMaximumHeight(110)
        form.addRow(self.measurement_list)
        form.addRow(button("Clear measurements", self.clear_measurements))
        right_layout.addWidget(window)
        drr_group = QGroupBox("Simulated X-ray · DRR")
        form = QFormLayout(drr_group)
        self.mu_water = spin(0.001, 0.1, 0.02, 3)
        self.mu_water.setSingleStep(0.001)
        form.addRow("Attenuation / mm", self.mu_water)
        self.drr_clip = QCheckBox("Apply current clip")
        form.addRow(self.drr_clip)
        form.addRow(button("Compute from current camera", self.calculate_drr))
        self.drr_canvas = ImageCanvas()
        self.drr_canvas.setMinimumSize(180, 180)
        self.drr_canvas.setMaximumHeight(240)
        self.drr_canvas.caption = "Generated by integrating CT attenuation"
        form.addRow(self.drr_canvas)
        form.addRow(button("Export DRR", self.export_drr))
        right_layout.addWidget(drr_group)
        right_layout.addStretch()
        right_scroll.setWidget(right)
        right_scroll.setMinimumWidth(330)
        splitter.addWidget(right_scroll)
        splitter.setSizes([270, 890, 340])

    def _statusbar(self):
        self.progress = QProgressBar()
        self.progress.setFixedWidth(180)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(button("Cancel tasks", self.cancel_tasks))
        self.statusBar().showMessage("Local educational and research tool")

    def report_progress(self, n, text):
        """Display task progress and the latest status message."""
        self.progress.setValue(n)
        self.statusBar().showMessage(text)

    def error(self, error):
        """Display an application error and retain it for desktop verification."""
        self.last_error = error
        message = error.strip().splitlines()[-1] if error else "Unknown error"
        self.statusBar().showMessage(message)
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Operation failed")
        dialog.setText(message)
        dialog.setDetailedText(error)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.exec()

    def require_volume(self):
        """Return whether a volume is loaded, displaying a status message if absent."""
        if self.volume is None:
            self.statusBar().showMessage("Load an image first.")
            return False
        return True

    def open_file(self):
        """Choose a scalar image and start loading it."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open CT / MRI",
            str(self.root),
            "3D images (*.nii *.nii.gz *.nrrd *.mha *.mhd);;All files (*)",
        )
        if not path:
            return
        modality, ok = QInputDialog.getItem(
            self,
            "Image intensity",
            "Select modality and calibration",
            ["CT (reliable HU calibration)", "MRI", "CBCT / uncalibrated intensity"],
            0,
            False,
        )
        if ok:
            self.open_path(
                path, "MR" if modality == "MRI" else "CT", modality.startswith("CT(")
            )

    def open_path(self, path, modality="CT", calibrated=True, scene=None):
        """Load a scalar image in the background and optionally restore its scene."""
        self.tasks.cancel_all()
        self.mesh_timer.stop()
        self.report_progress(0, "Reading image")
        uid = (scene or {}).get("source_metadata", {}).get("series_uid")
        self.tasks.submit(
            "load",
            lambda p, c: load_volume(path, modality, calibrated, uid),
            lambda volume: self.set_volume(volume, scene),
            self.error,
            self.report_progress,
        )

    def open_synthetic(self):
        """Load the deterministic hollow-structure demonstration volume."""
        self.set_volume(synthetic_volume())

    def open_dicom(self):
        """Choose a DICOM folder and enumerate its series."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select DICOM folder", str(self.root)
        )
        if not folder:
            return
        self.tasks.submit(
            "discover",
            lambda p, c: dicom_series(folder),
            self._select_series,
            self.error,
        )

    def _select_series(self, series):
        if not series:
            self.error("No DICOM series found.")
            return
        names = [
            f"{i + 1}. {s['description']} · {s['modality']} · {len(s['files'])} slices"
            for i, s in enumerate(series)
        ]
        selected, ok = QInputDialog.getItem(
            self, "DICOM series", "Select a static 3D series", names, 0, False
        )
        if ok:
            s = series[names.index(selected)]
            self.tasks.cancel_all()
            self.tasks.submit(
                "load",
                lambda p, c: read_dicom(s, p),
                self.set_volume,
                self.error,
                self.report_progress,
            )

    def choose_sample(self):
        """Choose a public sample and download it into the application cache."""
        names = [f"{SAMPLES[k]['title']} · {k}" for k in SAMPLES]
        choice, ok = QInputDialog.getItem(
            self,
            "Public CT samples",
            "Source: 3D Slicer; available offline after downloading",
            names,
            0,
            False,
        )
        if not ok:
            return
        key = list(SAMPLES)[names.index(choice)]
        directory = self.paths.data / "samples"
        self.tasks.submit(
            "download",
            lambda p, c: download_sample(key, directory, p, c),
            self.open_path,
            self.error,
            self.report_progress,
        )

    def set_volume(self, volume, scene=None):
        """Replace the volume, clear derived geometry, and initialize physical transforms."""
        self.tasks.cancel_all()
        self.cancel_segmentation()
        self.volume = volume
        self._pending_camera_state = None
        self.labels = None
        self.labels_path = ""
        self.mask = None
        self.mask_path = ""
        self.crosshair = volume.center.copy()
        self.clip = ClipState(origin=list(volume.center), normal=[0.0, 1.0, 0.0])
        self.measurements = []
        self.measurement_list.clear()
        self.drr_result = None
        self.drr_canvas.image = None
        self.drr_canvas.update()
        self.layers = default_layers(
            volume.modality if volume.hu_calibrated else "MR", volume.scalar_range
        )
        self.engine.set_volume(volume)
        self.engine.set_quality(True, self.scattering.isChecked())
        self._populate_layers()
        self.engine.configure_layers(self.layers)
        self.engine.mode = self.mode_combo.currentData()
        self.engine.show_preview()
        self.data_info.setText(
            f"{volume.name}\n{volume.shape_xyz[0]} × {volume.shape_xyz[1]} × {volume.shape_xyz[2]}\n"
            f"Spacing {volume.spacing[0]:.3f} / {volume.spacing[1]:.3f} / {volume.spacing[2]:.3f} mm\n"
            f"{volume.modality} · {'HU' if volume.hu_calibrated else 'Relative intensity'}"
        )
        self.threshold.setRange(*volume.scalar_range)
        if volume.modality != "CT":
            lo, hi = volume.scalar_range
            self.level.setValue((lo + hi) / 2)
            self.width.setValue(max(hi - lo, 1))
        else:
            self.level.setValue(60)
            self.width.setValue(400)
        self.clip_check.blockSignals(True)
        self.clip_check.setChecked(False)
        self.clip_check.blockSignals(False)
        for widget, value in ((self.clip_flip, False), (self.clip_slider, 500)):
            widget.blockSignals(True)
            widget.setChecked(value) if isinstance(
                widget, QCheckBox
            ) else widget.setValue(value)
            widget.blockSignals(False)
        for widget, value in (
            (self.clip_axis, "coronal"),
            (self.section_combo, "open"),
        ):
            widget.blockSignals(True)
            widget.setCurrentIndex(widget.findData(value))
            widget.blockSignals(False)
        self._place_clip_widget()
        self.plane_widget.SetEnabled(False)
        if scene:
            if scene["fingerprint"] != volume.fingerprint:
                self.error(
                    "The image fingerprint does not match the scene. Camera, labels, and annotations were not restored."
                )
            else:
                self._restore_scene(scene)
        self.sync_slice_sliders()
        self.refresh_mpr()
        self.mesh_timer.start()
        self.report_progress(100, "Image loaded; generating detailed surfaces")

    def _populate_layers(self, selected=0):
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for layer in self.layers:
            suffix = (
                f"Label {layer.label_id}"
                if layer.label_id is not None
                else f"{layer.threshold:.0f} {'HU' if self.volume and self.volume.hu_calibrated else ''}"
            )
            item = QListWidgetItem(f"{layer.name}\n{suffix}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked
            )
            item.setForeground(QColor.fromRgbF(*layer.color))
            self.layer_list.addItem(item)
        self.layer_list.setCurrentRow(min(selected, len(self.layers) - 1))
        self.layer_list.blockSignals(False)
        self._selected_layer_changed(self.layer_list.currentRow())

    def _selected_layer_changed(self, index):
        if not 0 <= index < len(self.layers):
            return
        layer = self.layers[index]
        for widget, value in [
            (self.threshold, layer.threshold),
            (self.opacity, layer.opacity),
            (self.roughness, layer.roughness),
            (self.coat, layer.coat),
            (self.surface_smoothing, layer.smoothing_mm),
            (
                self.upper,
                layer.upper if layer.upper is not None else layer.threshold + 200,
            ),
        ]:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self.layer_kind.blockSignals(True)
        self.layer_kind.setCurrentIndex(self.layer_kind.findData(layer.kind))
        self.layer_kind.blockSignals(False)
        self.upper.setEnabled(
            layer.kind in ("volume_band", "surface_band", "solid_band")
            and layer.label_id is None
        )
        self.surface_smoothing.setEnabled(
            layer.label_id is None and layer.kind != "volume_band"
        )
        for widget in (self.threshold, self.threshold_slider, self.layer_kind):
            widget.setEnabled(layer.label_id is None)
        color = QColor.fromRgbF(*layer.color)
        self.color_button.setStyleSheet(f"background: {color.name()}; color: #ffffff;")
        inside = QColor.fromRgbF(*(layer.interior_color or layer.color))
        self.inside_color_button.setStyleSheet(
            f"background: {inside.name()}; color: #ffffff;"
        )
        cut = QColor.fromRgbF(*(layer.cut_color or layer.color))
        self.cut_color_button.setStyleSheet(
            f"background: {cut.name()}; color: #ffffff;"
        )
        self.cut_color_button.setEnabled(layer.kind == "solid_band")
        self._sync_threshold_slider()

    def _sync_threshold_slider(self):
        if not self.volume:
            return
        lo, hi = self.volume.scalar_range
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(
            round(10000 * (self.threshold.value() - lo) / max(hi - lo, 1))
        )
        self.threshold_slider.blockSignals(False)

    def _threshold_slider_changed(self, value):
        if self.volume:
            lo, hi = self.volume.scalar_range
            self.threshold.setValue(lo + (hi - lo) * value / 10000)

    def _threshold_spin_changed(self):
        self._sync_threshold_slider()
        self._layer_attributes(geometry_changed=True)

    def _layer_attributes(self, *_, geometry_changed=False):
        i = self.layer_list.currentRow()
        if not self.volume or not 0 <= i < len(self.layers):
            return
        layer = self.layers[i]
        old_kind = layer.kind
        old_threshold = layer.threshold
        old_upper = layer.upper
        old_smoothing = layer.smoothing_mm
        layer.threshold = self.threshold.value()
        layer.kind = self.layer_kind.currentData()
        layer.upper = (
            self.upper.value()
            if layer.kind in ("volume_band", "surface_band", "solid_band")
            else None
        )
        layer.opacity = self.opacity.value()
        layer.roughness = self.roughness.value()
        layer.coat = self.coat.value()
        layer.smoothing_mm = (
            self.surface_smoothing.value()
            if layer.label_id is None and layer.kind != "volume_band"
            else 0
        )
        self.upper.setEnabled(
            layer.kind in ("volume_band", "surface_band", "solid_band")
            and layer.label_id is None
        )
        if layer.upper is not None and layer.upper <= layer.threshold:
            layer.threshold, layer.kind, layer.upper = (
                old_threshold,
                old_kind,
                old_upper,
            )
            self._selected_layer_changed(i)
            self.statusBar().showMessage(
                "The upper intensity limit must exceed the lower limit."
            )
            return
        try:
            self.engine.configure_layers(self.layers)
        except ValueError as e:
            layer.threshold = old_threshold
            layer.kind = old_kind
            layer.upper = old_upper
            self._selected_layer_changed(i)
            self.statusBar().showMessage(str(e))
            return
        geometry_changed |= (
            old_kind != layer.kind
            or old_threshold != layer.threshold
            or old_upper != layer.upper
            or old_smoothing != layer.smoothing_mm
        )
        if geometry_changed:
            self.tasks.cancel("mesh")
            self.engine.show_preview()
            self.view_status.setText(
                "Original-intensity preview · detailed surfaces and display smoothing follow after adjustment"
            )
            self.mesh_timer.start()
        else:
            self.engine._apply_ssao()
            self.engine.render()
        self.section_timer.start()
        item = self.layer_list.item(i)
        self.layer_list.blockSignals(True)
        item.setText(
            f"{layer.name}\n"
            + (
                f"Label {layer.label_id}"
                if layer.label_id is not None
                else f"{layer.threshold:.0f} {'HU' if self.volume.hu_calibrated else ''}"
            )
        )
        self.layer_list.blockSignals(False)

    def _layer_visibility(self, item):
        i = self.layer_list.row(item)
        if not 0 <= i < len(self.layers):
            return
        self.layers[i].visible = item.checkState() == Qt.CheckState.Checked
        self.tasks.cancel("mesh")
        try:
            self.engine.configure_layers(self.layers)
            self.engine.show_preview()
            self.mesh_timer.start()
        except ValueError as e:
            self.layers[i].visible = False
            self.statusBar().showMessage(str(e))
        self.section_timer.start()

    def add_layer(self):
        """Add an intensity layer initialized from the current scalar range."""
        if not self.require_volume():
            return
        name, ok = QInputDialog.getText(self, "Add layer", "Layer name")
        if ok and name.strip():
            threshold = self.threshold.value() + 10
            occupied = {layer.threshold for layer in self.layers}
            while threshold in occupied:
                threshold += 10
            self.layers.append(Layer(name.strip(), threshold, (0.7, 0.8, 0.9), 0.7))
            self._populate_layers(len(self.layers) - 1)
            self._layer_attributes(geometry_changed=True)

    def remove_layer(self):
        """Remove the selected rendering layer and rebuild detailed geometry."""
        i = self.layer_list.currentRow()
        if 0 <= i < len(self.layers):
            self.layers.pop(i)
            self.tasks.cancel("mesh")
            self._populate_layers()
            self.engine.configure_layers(self.layers)
            self.engine.show_preview()
            self.mesh_timer.start()

    def choose_color(self, inside=False, cut=False):
        """Update the selected exterior, interior, or cut-surface color."""
        i = self.layer_list.currentRow()
        if not 0 <= i < len(self.layers):
            return
        from PySide6.QtWidgets import QColorDialog

        layer = self.layers[i]
        initial = (
            (layer.cut_color or layer.color)
            if cut
            else ((layer.interior_color or layer.color) if inside else layer.color)
        )
        color = QColorDialog.getColor(
            QColor.fromRgbF(*initial),
            self,
            "Choose section color"
            if cut
            else ("Choose inner-wall color" if inside else "Choose exterior color"),
        )
        if color.isValid():
            if cut:
                layer.cut_color = color.getRgbF()[:3]
            elif inside:
                layer.interior_color = color.getRgbF()[:3]
            else:
                layer.color = color.getRgbF()[:3]
            self._selected_layer_changed(i)
            self._layer_attributes()

    def rebuild_meshes(self):
        """Schedule detailed geometry for the current volume and visible layers."""
        if not self.volume:
            return
        volume = self.volume
        layers = [replace(layer) for layer in self.layers]
        mask, labels = self.mask, self.labels
        started = time.perf_counter()
        self.tasks.submit(
            "mesh",
            lambda p, c: build_surfaces(volume, layers, mask, labels, p, c),
            lambda result: self._meshes_ready(result, started),
            self.error,
            self.report_progress,
        )

    def _meshes_ready(self, result, started):
        key, meshes = result
        self.engine.apply_surfaces(key, meshes)
        self.engine.set_quality(True, self.scattering.isChecked())
        triangles = sum(m.GetNumberOfCells() for m in meshes.values())
        filtered = sorted(
            {
                layer.smoothing_mm
                for layer in self.layers
                if layer.visible and layer.smoothing_mm > 0
            }
        )
        note = (
            (
                " · display smoothing "
                + "/".join(f"{s:g}" for s in filtered)
                + " mm; disable for measurements"
            )
            if filtered
            else ""
        )
        self.view_status.setText(
            f"Detailed surfaces · {triangles:,} triangles · {time.perf_counter() - started:.2f} s"
            + note
        )
        self.report_progress(100, "Detailed surfaces ready")
        self._refresh_measurement_list()
        self.section_timer.start()

    def _mode_changed(self):
        if self.volume:
            self.engine.set_mode(self.mode_combo.currentData())
            if self.mode_combo.currentData() == "hybrid":
                self.engine.render()
                translucent = any(
                    layer.visible and 0 < layer.opacity < 1
                    for layer in self.layers
                    if layer.kind != "volume_band"
                )
                if (
                    translucent
                    and not self.engine.renderer.GetLastRenderingUsedDepthPeeling()
                ):
                    self.mode_combo.blockSignals(True)
                    self.mode_combo.setCurrentIndex(self.mode_combo.findData("surface"))
                    self.mode_combo.blockSignals(False)
                    self.engine.set_mode("surface")
                    self.statusBar().showMessage(
                        "Hybrid transparency compositing is unavailable on this device. Use separate surface or volume rendering."
                    )

    def _quality_changed(self):
        if self.volume:
            self.engine.ssao = self.ssao_check.isChecked()
            self.engine.set_quality(True, self.scattering.isChecked())
            self.engine._apply_ssao()
            self.engine.render()

    def _layout_changed(self, checked):
        for panel in self.slice_panels.values():
            panel.setVisible(checked)
        self.views_grid.setColumnStretch(1, 1 if checked else 0)

    def start_pick(self, kind):
        """Begin a distance, angle, annotation, or connected-region selection."""
        if not self.require_volume():
            return
        if not self.engine.geometry_current or self.engine.mode == "volume":
            self.statusBar().showMessage(
                "Wait for detailed surfaces and select surface rendering."
            )
            return
        self.pick_kind = kind
        self.pick_points = []
        names = {
            "distance": "Pick two surface points in order",
            "angle": "Pick three surface points; the middle point is the angle vertex",
            "annotation": "Pick a surface point to add an annotation",
            "keep": "Pick a surface to retain its intensity-connected region",
            "remove": "Pick a surface to remove its intensity-connected region",
        }
        self.statusBar().showMessage(names[kind])

    def cancel_pick(self):
        """Clear unfinished picked points and restore navigation mode."""
        self.pick_kind = None
        self.pick_points = []
        self.statusBar().showMessage("Surface picking cancelled")

    def _vtk_click(self, interactor, event):
        if not self.pick_kind:
            return
        try:
            x, y = interactor.GetEventPosition()
            index = (
                self.layer_list.currentRow() if self.lock_layer.isChecked() else None
            )
            point = self.engine.pick(x, y, index)
            if self.pick_kind in ("keep", "remove"):
                action = self.pick_kind
                lower, upper = self.region_lower.value(), self.region_upper.value()
                if lower >= upper:
                    raise ValueError(
                        "The connected-region upper limit must exceed the lower limit."
                    )
                volume, mask = self.volume, self.mask

                def compute(p, c):
                    region = connected_region(volume, point, lower, upper, mask)
                    return (
                        region
                        if action == "keep"
                        else (
                            (np.ones_like(region) if mask is None else mask) & ~region
                        )
                    )

                self.tasks.submit("region", compute, self.apply_mask, self.error)
                self.cancel_pick()
                return
            self.pick_points.append(list(map(float, point)))
            required = {"distance": 2, "angle": 3, "annotation": 1}[self.pick_kind]
            self.statusBar().showMessage(
                f"Selected {len(self.pick_points)} / {required} points"
            )
            if len(self.pick_points) == required:
                text = ""
                if self.pick_kind == "annotation":
                    text, ok = QInputDialog.getText(
                        self, "Surface annotation", "Annotation text"
                    )
                    if not ok:
                        self.cancel_pick()
                        return
                measurement = Measurement(
                    self.pick_kind,
                    self.pick_points,
                    text,
                    self.layers[index].name
                    if index is not None and index >= 0
                    else "Visible surface",
                    self.engine.geometry_key,
                )
                measurement.value
                self.measurements.append(measurement)
                self.engine.add_measurement(measurement)
                self._refresh_measurement_list()
                self.cancel_pick()
        except Exception as e:
            self.statusBar().showMessage(str(e))

    def _refresh_measurement_list(self):
        self.measurement_list.clear()
        for m in self.measurements:
            stale = m.geometry_key != self.engine.geometry_key
            text = (
                m.text
                if m.kind == "annotation"
                else f"{m.value:.2f} {'mm' if m.kind == 'distance' else '°'}"
            )
            self.measurement_list.addItem(
                text + (" · geometry has changed" if stale else "")
            )

    def clear_measurements(self):
        """Remove all measurement actors and recorded points."""
        self.measurements = []
        for actor in self.engine.annotations:
            self.engine.renderer.RemoveViewProp(actor)
        self.engine.annotations = []
        self.measurement_list.clear()
        self.engine.render()

    def apply_mask(self, mask):
        """Replace the region-of-interest mask and rebuild visible geometry."""
        self.tasks.cancel("mesh")
        self.mask = mask
        self.mask_path = ""
        self.engine.set_mask(mask)
        self._place_clip_widget()
        self.engine.show_preview()
        self.mesh_timer.start()
        self.section_timer.start()
        self.report_progress(100, "Structure filter updated")

    def clear_mask(self):
        """Restore visibility across the original voxel grid."""
        if self.volume:
            self.apply_mask(None)

    def edit_roi(self):
        """Choose a physical bounding box and apply it as a visibility mask."""
        if not self.require_volume():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("ROI box · LPS millimeters")
        form = QFormLayout(dialog)
        inputs = []
        lo, hi = self.volume.bounds
        for i, axis in enumerate("XYZ"):
            a, b = spin(lo[i], hi[i], lo[i]), spin(lo[i], hi[i], hi[i])
            row = QHBoxLayout()
            row.addWidget(a)
            row.addWidget(b)
            form.addRow(f"{axis} min / max", row)
            inputs.append((a, b))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        bounds = np.asarray([(a.value(), b.value()) for a, b in inputs])
        if np.any(bounds[:, 1] <= bounds[:, 0]):
            self.statusBar().showMessage(
                "ROI maximum bounds must exceed the minimum bounds."
            )
            return
        volume = self.volume
        previous = self.mask

        def create(p, c):
            result = np.ones(volume.array.shape, dtype=bool)
            x, y = np.meshgrid(
                np.arange(volume.shape_xyz[0]), np.arange(volume.shape_xyz[1])
            )
            for z in range(volume.array.shape[0]):
                points = volume.to_world(np.stack([x, y, np.full_like(x, z)], axis=-1))
                result[z] = np.all(
                    (points >= bounds[:, 0]) & (points <= bounds[:, 1]), axis=-1
                )
            return result if previous is None else result & previous

        self.tasks.submit("region", create, self.apply_mask, self.error)

    def import_labels(self):
        """Choose an integer label image aligned to the current volume."""
        if not self.require_volume():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import organ labels for this image",
            str(self.root),
            "Labels (*.nii *.nii.gz *.nrrd *.mha)",
        )
        if path:
            self.load_labels(path)

    def load_labels(self, path):
        """Align labels in the background without modifying the source volume."""
        volume = self.volume
        self.tasks.submit(
            "labels",
            lambda p, c: aligned_labels(path, volume),
            lambda labels: self._labels_ready(labels, path),
            self.error,
        )

    def _labels_ready(self, labels, path):
        self.labels = labels
        self.labels_path = str(Path(path).resolve())
        colors = [
            (0.86, 0.36, 0.28),
            (0.96, 0.62, 0.31),
            (0.34, 0.66, 0.9),
            (0.64, 0.53, 0.9),
            (0.36, 0.78, 0.59),
        ]
        names = self.label_names()
        existing = {layer.label_id for layer in self.layers}
        for i, label in enumerate(int(x) for x in np.unique(labels) if x > 0):
            if label not in existing:
                self.layers.append(
                    Layer(
                        names.get(label, f"Structure {label}"),
                        0.0,
                        colors[i % len(colors)],
                        0.85,
                        False,
                        label_id=label,
                    )
                )
        self._populate_layers()
        self.mpr_timer.start()
        self.report_progress(
            100, "Organ labels aligned; enable a structure to display its surface"
        )

    def filter_labels(self):
        """Apply one selected semantic label as the visibility mask."""
        if self.labels is None:
            self.statusBar().showMessage("Import or generate organ labels first.")
            return
        names = self.label_names()
        choices = [
            f"{i}: {names.get(i, 'Structure')}" for i in np.unique(self.labels) if i > 0
        ]
        selection, ok = QInputDialog.getItem(
            self,
            "Filter organ",
            "Retain original intensity boundaries within one organ region",
            choices,
            0,
            False,
        )
        if ok:
            self.apply_mask(self.labels == int(selection.split(":")[0]))

    def cancel_tasks(self):
        """Cancel timers, background computations, and isolated inference."""
        self.mesh_timer.stop()
        self.tasks.cancel_all()
        self.cancel_segmentation()
        self.report_progress(
            0, "Background tasks cancelled; completed results retained"
        )

    def closeEvent(self, event):
        """Cancel computations and release native VTK resources when closing."""
        self.cancel_tasks()
        self.plane_widget.SetEnabled(False)
        self.orientation.SetEnabled(False)
        self.vtk_widget.Finalize()
        event.accept()
