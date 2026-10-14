from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Layer:
    """Rendering parameters for one scalar boundary, interval, or label."""

    name: str
    threshold: float
    color: tuple[float, float, float]
    opacity: float = 1.0
    visible: bool = True
    roughness: float = 0.55
    kind: str = "isosurface"
    upper: float | None = None
    label_id: int | None = None
    interior_color: tuple[float, float, float] | None = None
    coat: float = 0.35
    smoothing_mm: float = 0.0
    cut_color: tuple[float, float, float] | None = None

    def __post_init__(self):
        self.color = tuple(self.color)
        if self.interior_color is not None:
            self.interior_color = tuple(self.interior_color)
        if self.cut_color is not None:
            self.cut_color = tuple(self.cut_color)


def default_layers(modality="CT", scalar_range=(-1000, 1500)):
    """Create intensity layers for CT or the supplied relative scalar range."""
    if modality == "CT":
        return [
            Layer(
                "Soft-tissue boundary",
                40.0,
                (0.67, 0.50, 0.38),
                1.0,
                roughness=0.43,
                interior_color=(0.43, 0.24, 0.18),
                coat=0.4,
            ),
            Layer(
                "Contrast-enhanced structures",
                160.0,
                (0.82, 0.75, 0.60),
                1.0,
                roughness=0.38,
                interior_color=(0.43, 0.24, 0.18),
                coat=0.5,
            ),
            Layer(
                "Bone",
                600.0,
                (0.86, 0.79, 0.62),
                1.0,
                roughness=0.5,
                interior_color=(0.76, 0.66, 0.49),
                coat=0.18,
            ),
        ]
    lo, hi = scalar_range
    return [
        Layer("Low-intensity boundary", lo + (hi - lo) * 0.25, (0.58, 0.32, 0.48), 0.3),
        Layer("Mid-intensity boundary", lo + (hi - lo) * 0.5, (0.83, 0.58, 0.40), 0.8),
        Layer(
            "High-intensity boundary", lo + (hi - lo) * 0.75, (0.95, 0.88, 0.73), 1.0
        ),
    ]


@dataclass
class ClipState:
    """A clipping plane in LPS millimeters and its section display mode."""

    enabled: bool = False
    origin: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    normal: list[float] = field(default_factory=lambda: [0.0, -1.0, 0.0])
    flip: bool = False
    section: str = "open"

    @property
    def effective_normal(self):
        """Return a finite unit normal with the requested clipping orientation."""
        n = np.asarray(self.normal, dtype=float)
        length = np.linalg.norm(n)
        if not np.isfinite(length) or length < 1e-8:
            raise ValueError("Invalid clipping plane normal.")
        return n / length * (-1 if self.flip else 1)


@dataclass
class Measurement:
    """Picked LPS points bound to the geometry used during acquisition."""

    kind: str
    points: list[list[float]]
    text: str = ""
    layer: str = ""
    geometry_key: str = ""

    @property
    def value(self):
        """Return a distance in millimeters or an angle in degrees."""
        p = np.asarray(self.points, dtype=float)
        if self.kind == "distance":
            return float(np.linalg.norm(p[1] - p[0]))
        if self.kind == "angle":
            a, b = p[0] - p[1], p[2] - p[1]
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            if denominator < 1e-8:
                raise ValueError(
                    "An angle measurement cannot contain coincident points."
                )
            return float(np.degrees(np.arccos(np.clip(a @ b / denominator, -1, 1))))
        return 0.0


def save_scene(
    path,
    *,
    volume,
    layers,
    clip,
    camera,
    crosshair,
    measurements,
    mode="surface",
    labels_path="",
    mask_path="",
    drr=None,
    window=None,
    quality=None,
):
    """Atomically serialize volume references, geometry, and desktop state."""
    payload = {
        "schema_version": 1,
        "source": volume.source,
        "fingerprint": volume.fingerprint,
        "modality": volume.modality,
        "hu_calibrated": volume.hu_calibrated,
        "layers": [asdict(layer) for layer in layers],
        "clip": asdict(clip),
        "camera": camera,
        "crosshair": list(map(float, crosshair)),
        "measurements": [asdict(m) for m in measurements],
        "mode": mode,
        "labels_path": labels_path,
        "mask_path": mask_path,
        "drr": drr or {},
        "window": window or {},
        "source_metadata": volume.metadata,
        "quality": quality or {"scattering": False, "ssao": True},
    }
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(destination)


def read_scene(path):
    """Read a scene and validate its schema, layer parameters, and clipping plane."""
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    if p.get("schema_version") != 1:
        raise ValueError("Unsupported scene version.")
    if not all(k in p for k in ("source", "fingerprint", "layers", "clip", "camera")):
        raise ValueError("The scene is missing required fields.")
    if len(p["layers"]) > 40:
        raise ValueError("The scene contains too many layers.")
    p["layers"] = [Layer(**layer) for layer in p["layers"]]
    p["clip"] = ClipState(**p["clip"])
    p["measurements"] = [Measurement(**m) for m in p.get("measurements", [])]
    for layer in p["layers"]:
        if (
            not np.isfinite(layer.threshold)
            or not 0 <= layer.opacity <= 1
            or not 0 <= layer.roughness <= 1
        ):
            raise ValueError("Invalid scene layer parameters.")
        if (
            layer.kind
            not in ("isosurface", "surface_band", "solid_band", "volume_band")
            or not 0 <= layer.coat <= 1
        ):
            raise ValueError("Invalid scene layer material or type.")
        if not np.isfinite(layer.smoothing_mm) or not 0 <= layer.smoothing_mm <= 3:
            raise ValueError("Display smoothing must be between 0 and 3 mm.")
        if layer.kind in ("surface_band", "solid_band", "volume_band") and (
            layer.upper is None
            or not np.isfinite(layer.upper)
            or layer.upper <= layer.threshold
        ):
            raise ValueError("The upper intensity limit must exceed the lower limit.")
        for color in (layer.color, layer.interior_color, layer.cut_color):
            if color is not None and (
                len(color) != 3
                or not np.isfinite(color).all()
                or not all(0 <= c <= 1 for c in color)
            ):
                raise ValueError("Invalid scene color.")
    p["clip"].effective_normal
    return p
