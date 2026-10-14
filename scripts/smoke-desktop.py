"""Exercise the native desktop with synthetic data and no external resources."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import QApplication

from openkalolumen.config import AppPaths
from openkalolumen.desktop.window import MainWindow
from openkalolumen.render.geometry import export_mesh
from openkalolumen.state import Layer


def main():
    """Check geometry, navigation, scene round trips, export, and native capture."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    with tempfile.TemporaryDirectory() as directory:
        window = MainWindow(AppPaths.resolve(data=directory))
        window.show()
        window.open_synthetic()

        def wait_ready():
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                app.processEvents()
                if window.last_error:
                    raise RuntimeError(window.last_error)
                if (
                    window.engine.geometry_current
                    and not window.tasks.active
                    and not window.mesh_timer.isActive()
                    and not window.section_timer.isActive()
                    and not window.mpr_timer.isActive()
                ):
                    return
                time.sleep(0.01)
            raise TimeoutError("Desktop geometry did not settle within 45 seconds.")

        try:
            wait_ready()
            window.layers = [
                Layer("material", 40, (0.7, 0.5, 0.4), kind="solid_band", upper=160)
            ]
            window._populate_layers()
            window.engine.configure_layers(window.layers)
            window.rebuild_meshes()
            wait_ready()
            window.clip_check.setChecked(True)
            window.section_combo.setCurrentIndex(
                window.section_combo.findData("tissue")
            )
            wait_ready()
            window.move_slice("axial", 600)
            wait_ready()
            lower, upper = window.volume.bounds
            assert (
                abs(window.crosshair[2] - (lower[2] + (upper[2] - lower[2]) * 0.6))
                < 1e-6
            )
            window.move_slice("axial", 500)
            wait_ready()
            assert window.engine.meshes[0].GetNumberOfCells() > 0
            scene = args.output / "synthetic.scene.json"
            window.save_scene_to(scene)
            expected_camera = window.engine.camera_state()
            window.open_scene_path(scene)
            wait_ready()
            assert window.layers[0].kind == "solid_band"
            assert window.clip.enabled
            assert (
                window.engine.camera_state()["position"] == expected_camera["position"]
            )
            export_mesh(
                args.output / "synthetic.stl",
                window.engine.meshes[0],
                replace(window.layers[0], smoothing_mm=0),
                window.volume,
                window.clip,
            )
            window.capture_interface(args.output / "desktop.png")
            assert (args.output / "desktop.png").stat().st_size > 1000
            report = {
                "checks": [
                    "synthetic startup",
                    "finite-material geometry",
                    "physical clipping",
                    "MPR navigation",
                    "scene round trip",
                    "clipped STL export",
                    "native framebuffer capture",
                ],
                "triangles": window.engine.meshes[0].GetNumberOfCells(),
                "error": window.last_error,
            }
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print(json.dumps(report))
        finally:
            window.close()
            window.tasks.pool.waitForDone(5000)
            app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
