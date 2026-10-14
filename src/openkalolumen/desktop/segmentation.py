"""Manage the isolated segmentation process and model label metadata."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QInputDialog,
)

from openkalolumen.imaging.volume import (
    save_volume,
)


class SegmentationMixin:
    """Manage the isolated segmentation process and model label metadata."""

    def label_names(self):
        """Read label names directly from the configured model metadata."""
        path = self.paths.model / "metadata.json"
        if path.exists():
            raw = json.loads(path.read_text())["network_data_format"]["outputs"][
                "pred"
            ]["channel_def"]
            result = {int(k): v for k, v in raw.items()}
        else:
            result = {}
        return result

    def start_segmentation(self):
        """Start isolated inference with configured resources and class prompts."""
        if not self.require_volume():
            return
        python = self.paths.segmentation_python
        worker = Path(__file__).resolve().parents[1] / "segmentation" / "worker.py"
        model = self.paths.model
        if (
            not python.is_file()
            or not (model / "vista3d_pretrained_model/model.pt").is_file()
        ):
            self.error(
                "Configure the segmentation Python and model directory, and confirm that weights have been downloaded."
            )
            return
        selected, ok = QInputDialog.getText(
            self,
            "Local organ segmentation",
            "Enter comma-separated class IDs (heart 115, aorta 6, lung lobes 28–32)",
            text="115,6",
        )
        if not ok:
            return
        try:
            ids = [int(x.strip()) for x in selected.split(",")]
            names = self.label_names()
            if not ids or any(i not in names or "deprecated" in names[i] for i in ids):
                raise ValueError("Unknown or deprecated class ID.")
        except ValueError as e:
            self.statusBar().showMessage(str(e))
            return
        self.cancel_segmentation()
        cache = self.paths.data / "segmentation"
        cache.mkdir(parents=True, exist_ok=True)
        image_path = cache / (self.volume.fingerprint + ".nii.gz")
        save_volume(self.volume, image_path)
        process = QProcess(self)
        self.seg_process = process
        self.seg_buffer = ""
        process.setProgram(str(python))
        process.setArguments(
            [
                str(worker),
                "--image",
                str(image_path),
                "--model",
                str(model),
                "--classes",
                ",".join(map(str, ids)),
                "--output",
                str(cache),
                "--device",
                "auto",
            ]
        )
        process.readyReadStandardOutput.connect(self._seg_stdout)
        process.readyReadStandardError.connect(lambda: self._seg_stderr(process))
        process.finished.connect(lambda code, status: self._seg_finished(process, code))
        process.errorOccurred.connect(lambda _: self.error(process.errorString()))
        process.start()
        self.report_progress(
            0, "Background segmentation started; image viewing remains available"
        )

    def _seg_stdout(self):
        process = self.seg_process
        if not process:
            return
        self.seg_buffer += bytes(process.readAllStandardOutput()).decode(
            errors="replace"
        )
        while "\n" in self.seg_buffer:
            line, self.seg_buffer = self.seg_buffer.split("\n", 1)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "progress":
                self.report_progress(
                    event.get("percent", 0), event.get("message", "Segmenting")
                )
            elif event.get("event") == "result":
                process.setProperty("result_path", event["path"])
            elif event.get("event") == "error":
                process.setProperty("failure", event["message"])

    def _seg_stderr(self, process):
        text = bytes(process.readAllStandardError()).decode(errors="replace")
        process.setProperty(
            "stderr", (str(process.property("stderr") or "") + text)[-5000:]
        )

    def _seg_finished(self, process, code):
        if process is not self.seg_process:
            return
        self._seg_stdout()
        self.seg_process = None
        if code == 0 and process.property("result_path"):
            self.load_labels(process.property("result_path"))
        else:
            self.error(
                str(
                    process.property("failure")
                    or process.property("stderr")
                    or "Segmentation did not complete."
                )
            )

    def cancel_segmentation(self):
        """Terminate the active inference process and invalidate its delivery."""
        if self.seg_process:
            process = self.seg_process
            self.seg_process = None
            process.kill()
            process.waitForFinished(2000)
