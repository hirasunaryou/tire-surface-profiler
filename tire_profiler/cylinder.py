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
    center, axis_dir, radius, inliers = cyl.fit(
        points,
        threshold,
        max_iterations,
    )

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
