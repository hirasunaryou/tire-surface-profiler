"""Cylinder fitting helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from pyransac3d import Cylinder


@dataclass
class CylinderModel:
    point_on_axis: np.ndarray
    axis_direction: np.ndarray
    radius: float
    inliers: np.ndarray


class CylinderFitError(RuntimeError):
    """Raised when the cylinder cannot be estimated."""


def fit_cylinder(
    points: np.ndarray,
    *,
    threshold: float = 0.003,
    max_iterations: int = 5000,
) -> CylinderModel:
    """Fit a cylinder model to the provided point cloud."""
    cyl = Cylinder()

    def _try_fit():
        attempts = [
            ((points, threshold, max_iterations), {}),
            ((points,), {"threshold": threshold, "maxIteration": max_iterations}),
            ((points,), {"threshold": threshold, "maxIterations": max_iterations}),
            ((points,), {"thresh": threshold, "maxIteration": max_iterations}),
            ((points,), {"thresh": threshold, "maxIterations": max_iterations}),
        ]

        last_err: TypeError | None = None
        for args, kwargs in attempts:
            try:
                return cyl.fit(*args, **kwargs)
            except TypeError as exc:  # pragma: no cover - depends on pyransac3d build
                last_err = exc
                continue

        raise last_err if last_err else TypeError("Cylinder.fit signature mismatch")

    center, axis_dir, radius, inliers = _try_fit()

    if radius is None or np.isnan(radius):
        raise CylinderFitError("Cylinder RANSAC did not converge")

    axis_dir = np.asarray(axis_dir, dtype=float)
    norm = np.linalg.norm(axis_dir)
    if norm == 0:
        raise CylinderFitError("Cylinder axis direction is zero")

    return CylinderModel(
        point_on_axis=np.asarray(center, dtype=float),
        axis_direction=axis_dir / norm,
        radius=float(radius),
        inliers=np.asarray(inliers, dtype=int),
    )
