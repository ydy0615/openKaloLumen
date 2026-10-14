from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from openkalolumen.config import AppPaths
from openkalolumen.desktop.window import MainWindow


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="openKaloLumen local 3D medical image viewer"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image")
    source.add_argument("--scene")
    source.add_argument(
        "--demo", action="store_true", help="Open the synthetic lumen demo (default)"
    )
    parser.add_argument("--data-dir", help="Writable application data directory")
    parser.add_argument("--model", help="NV-Segment-CTMR model directory")
    parser.add_argument(
        "--segmentation-python",
        help="Python interpreter with segmentation dependencies",
    )
    parser.add_argument("--capture", help="Save a desktop screenshot after startup")
    parser.add_argument(
        "--quit-after",
        type=float,
        help="Close automatically after the specified number of seconds",
    )
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("openKaloLumen")
    app.setOrganizationName("openKaloLumen")
    window = MainWindow(
        AppPaths.resolve(args.data_dir, args.model, args.segmentation_python)
    )
    window.show()
    if args.scene:
        QTimer.singleShot(200, lambda: window.open_scene_path(args.scene))
    elif args.image:
        QTimer.singleShot(200, lambda: window.open_path(args.image))
    else:
        QTimer.singleShot(200, window.open_synthetic)
    if args.capture:

        def capture_when_ready():
            if window.engine.geometry_current and not window.section_timer.isActive():
                path = Path(args.capture).resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                window.capture_interface(path)
            else:
                QTimer.singleShot(500, capture_when_ready)

        QTimer.singleShot(3000, capture_when_ready)
    if args.quit_after:
        QTimer.singleShot(int(args.quit_after * 1000), window.close)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
