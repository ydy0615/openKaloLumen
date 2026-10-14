"""Monoenergetic teaching DRR; integrates attenuation in physical millimetres."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from openkalolumen.errors import Cancelled


def project_drr(
    volume,
    camera,
    resolution=256,
    mu_water=0.02,
    step=None,
    clip=None,
    progress=None,
    cancelled=lambda: False,
):
    """Integrate a calibrated CT volume along camera rays.

    Args:
        volume: CT volume with calibrated Hounsfield units.
        camera: Camera parameters defining parallel or perspective rays.
        resolution: Number of pixels along each output axis.
        mu_water: Water attenuation coefficient in inverse millimeters.
        step: Ray step in millimeters, or None for the default sampling interval.
        clip: Optional physical clipping plane.
        progress: Optional callback accepting a percentage and status message.
        cancelled: Callback reporting cooperative cancellation.

    Returns:
        A float32 array of transmitted intensities in [0, 1].

    Raises:
        ValueError: The volume does not contain calibrated CT intensities.
        Cancelled: The cancellation callback returns True.

    Ray intersections include voxel-cell bounds, half a voxel beyond the first
    and last sample centers.
    """
    if volume.modality != "CT" or not volume.hu_calibrated:
        raise ValueError(
            "DRR requires reliably HU-calibrated CT. MRI and uncalibrated CBCT are unsupported."
        )
    if mu_water <= 0 or not np.isfinite(mu_water):
        raise ValueError("The water attenuation coefficient must be positive.")
    eye = np.asarray(camera["position"], dtype=float)
    focal = np.asarray(camera["focal_point"], dtype=float)
    forward = focal - eye
    distance = np.linalg.norm(forward)
    forward /= distance
    right = np.cross(forward, np.asarray(camera["view_up"], dtype=float))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    h = w = int(resolution)
    uv = (np.arange(w) + 0.5) / w * 2 - 1
    u, v = np.meshgrid(uv, -uv)
    if camera.get("parallel", False):
        scale = camera.get("parallel_scale", 100.0)
        origins = eye + u[..., None] * right * scale + v[..., None] * up * scale
        directions = np.broadcast_to(forward, origins.shape).copy()
    else:
        scale = np.tan(np.radians(camera.get("view_angle", 30.0) / 2))
        directions = forward + u[..., None] * right * scale + v[..., None] * up * scale
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        origins = np.broadcast_to(eye, directions.shape).copy()

    origin_idx = volume.to_index(origins)
    direction_idx = directions @ np.linalg.inv(volume.index_to_lps[:3, :3]).T
    lower, upper = np.full(3, -0.5), volume.shape_xyz - 0.5
    near = np.full((h, w), -np.inf)
    far = np.full((h, w), np.inf)
    hit = np.ones((h, w), dtype=bool)
    for axis in range(3):
        component = direction_idx[..., axis]
        parallel = np.abs(component) < 1e-10
        hit &= ~(
            parallel
            & (
                (origin_idx[..., axis] < lower[axis])
                | (origin_idx[..., axis] > upper[axis])
            )
        )
        safe = np.where(parallel, 1.0, component)
        a = (lower[axis] - origin_idx[..., axis]) / safe
        b = (upper[axis] - origin_idx[..., axis]) / safe
        near = np.maximum(near, np.where(parallel, -np.inf, np.minimum(a, b)))
        far = np.minimum(far, np.where(parallel, np.inf, np.maximum(a, b)))
    near = np.maximum(near, 0)
    if clip is not None and clip.enabled:
        normal = clip.effective_normal
        signed = (origins - np.asarray(clip.origin)) @ normal
        slope = directions @ normal
        parallel = np.abs(slope) < 1e-10
        hit &= ~(parallel & (signed < 0))
        crossing = -signed / np.where(parallel, 1.0, slope)
        near = np.maximum(near, np.where((~parallel) & (slope > 0), crossing, -np.inf))
        far = np.minimum(far, np.where((~parallel) & (slope < 0), crossing, np.inf))
    hit &= far > near
    lengths = np.where(hit, far - near, 0.0)
    ds = float(step or min(volume.spacing))
    if ds <= 0:
        raise ValueError("The integration step must be positive.")
    steps = np.ceil(lengths / ds).astype(int)
    local_ds = lengths / np.maximum(steps, 1)
    optical_depth = np.zeros((h, w), dtype=np.float64)
    max_steps = int(steps.max())
    for i in range(max_steps):
        if cancelled():
            raise Cancelled()
        active = i < steps
        t = near[active] + (i + 0.5) * local_ds[active]
        coords = origin_idx[active] + direction_idx[active] * t[:, None]
        hu = map_coordinates(
            volume.array, coords[:, ::-1].T, order=1, mode="nearest", prefilter=False
        )
        attenuation = np.maximum(0.0, mu_water * (1.0 + hu / 1000.0))
        optical_depth[active] += attenuation * local_ds[active]
        if progress and i % 20 == 0:
            progress(
                int(100 * (i + 1) / max_steps), "Computing the attenuation projection"
            )
    return np.exp(-optical_depth).astype(np.float32)
