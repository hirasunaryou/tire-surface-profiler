"""Utilities for loading GLB files into Open3D point clouds."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


def load_point_cloud(
    glb_path: str | Path,
    *,
    voxel_size: Optional[float] = None,
    sample_points: int = 200_000,
) -> o3d.geometry.PointCloud:
    """Return a point cloud sampled from a GLB mesh."""
    glb_path = Path(glb_path)
    if not glb_path.exists():
        raise FileNotFoundError(f"GLB not found: {glb_path}")

    mesh = o3d.io.read_triangle_mesh(str(glb_path))
    if mesh.is_empty():
        raise ValueError(f"Mesh at {glb_path} is empty")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    if sample_points:
        pcd = mesh.sample_points_poisson_disk(sample_points)
    else:
        pcd = mesh.sample_points_poisson_disk(100_000)

    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size)

    return pcd


def to_numpy(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """Convert a point cloud to a numpy array."""
    return np.asarray(pcd.points)


def from_numpy(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Create an Open3D point cloud from numpy points."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def save_point_cloud(points: np.ndarray, path: str | Path) -> None:
    """Save a numpy point cloud to disk for debugging."""
    pcd = from_numpy(points)
    o3d.io.write_point_cloud(str(path), pcd)
