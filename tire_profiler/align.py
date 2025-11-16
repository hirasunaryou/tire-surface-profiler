"""Coordinate alignment helpers."""
from __future__ import annotations

import numpy as np


def _rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.isclose(c, 1):
        return np.eye(3)
    s = np.linalg.norm(v)
    kmat = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
        dtype=float,
    )
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))


def _rotation_x(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def align_points(
    points: np.ndarray,
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align the tire axis to +X and 12 o'clock to +Z."""
    translated = points - axis_point
    rot_axis = _rotation_matrix_from_vectors(axis_direction, np.array([1.0, 0.0, 0.0]))
    aligned = translated @ rot_axis.T

    top_idx = np.argmax(aligned[:, 2])
    yz = aligned[top_idx, 1:3]
    angle = np.arctan2(yz[0], yz[1])
    rot_top = _rotation_x(angle)
    aligned = aligned @ rot_top.T

    rotation = rot_top @ rot_axis
    translation = -axis_point
    return aligned, rotation, translation


def apply_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply an affine transform (R, t)."""
    return (points + translation) @ rotation.T
