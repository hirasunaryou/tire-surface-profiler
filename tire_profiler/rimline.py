"""Helpers for picking rim-line points and building the baseline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import open3d as o3d

from .io_glb import from_numpy


class RimLineError(RuntimeError):
    """Raised when the rim line cannot be estimated."""


def pick_rim_points(
    points: np.ndarray,
    *,
    window_name: str = "Pick rim points (Shift + click, press Q when done)",
) -> np.ndarray:
    """Launch an Open3D editor so the user can pick rim points."""
    pcd = from_numpy(points)
    picked_indices = o3d.visualization.draw_geometries_with_editing(
        [pcd], window_name=window_name
    )
    pts = np.asarray(pcd.points)
    if not picked_indices:
        raise RimLineError("No rim points were picked")
    return pts[picked_indices]


def save_rim_points(points: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"points": points.tolist()}
    path.write_text(json.dumps(payload, indent=2))


def load_rim_points(path: str | Path) -> np.ndarray:
    path = Path(path)
    payload = json.loads(path.read_text())
    return np.asarray(payload["points"], dtype=float)


def fit_rimline(points: np.ndarray, arc_lengths: np.ndarray) -> tuple[float, float]:
    """Fit Z0(Y) = alpha + beta * Y."""
    if len(points) < 3:
        raise RimLineError("At least three rim points are required")
    y = arc_lengths.reshape(-1, 1)
    z = points[:, 2]
    A = np.hstack([np.ones_like(y), y])
    alpha, beta = np.linalg.lstsq(A, z, rcond=None)[0]
    return float(alpha), float(beta)
