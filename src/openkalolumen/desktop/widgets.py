from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QLabel, QSlider, QVBoxLayout, QWidget


class ImageCanvas(QWidget):
    """Display a slice or projection and emit normalized click coordinates."""

    clicked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(170, 160)
        self.image = None
        self.crosshair = False
        self.image_rect = QRectF()
        self.caption = ""
        self.directions = ("", "", "", "")
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_image(self, rgb, caption="", crosshair=False):
        """Display an RGB image, caption, and optional central crosshair."""
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        self.image = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        self.caption = caption
        self.crosshair = crosshair
        self.update()

    def paintEvent(self, event):
        """Paint the image canvas and its overlays."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0b1018"))
        if self.image:
            size = self.image.size().scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio
            )
            self.image_rect = QRectF(
                (self.width() - size.width()) / 2,
                (self.height() - size.height()) / 2,
                size.width(),
                size.height(),
            )
            painter.drawImage(self.image_rect, self.image)
            if self.crosshair:
                painter.setPen(QPen(QColor(70, 200, 220, 160), 1))
                c = self.image_rect.center()
                painter.drawLine(
                    QPointF(c.x(), self.image_rect.top()),
                    QPointF(c.x(), self.image_rect.bottom()),
                )
                painter.drawLine(
                    QPointF(self.image_rect.left(), c.y()),
                    QPointF(self.image_rect.right(), c.y()),
                )
            painter.setPen(QColor("#91a3b9"))
            for text, x, y in zip(
                self.directions,
                [
                    self.image_rect.left() + 8,
                    self.image_rect.right() - 18,
                    self.width() / 2,
                    self.width() / 2,
                ],
                [
                    self.height() / 2,
                    self.height() / 2,
                    self.image_rect.top() + 18,
                    self.image_rect.bottom() - 8,
                ],
            ):
                painter.drawText(int(x), int(y), text)
        else:
            painter.setPen(QColor("#697b91"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                self.caption or "Displayed after an image is loaded",
            )
        if self.caption:
            painter.setPen(QColor("#a6b5c9"))
            painter.drawText(12, 24, self.caption)

    def mousePressEvent(self, event):
        """Emit normalized image coordinates for an in-bounds left click."""
        if self.image and self.image_rect.contains(event.position()):
            point = event.position()
            self.clicked.emit(
                (point.x() - self.image_rect.left()) / self.image_rect.width(),
                (point.y() - self.image_rect.top()) / self.image_rect.height(),
            )


class SlicePanel(QWidget):
    """Combine a slice canvas with a physical-position slider."""

    moved = Signal(int)
    clicked = Signal(float, float)

    def __init__(self, title, color, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.title = QLabel(title)
        self.title.setStyleSheet(f"color: {color}; font-weight: 600; padding: 5px;")
        self.canvas = ImageCanvas()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(500)
        self.slider.valueChanged.connect(self.moved)
        self.canvas.clicked.connect(self.clicked)
        layout.addWidget(self.title)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.slider)
