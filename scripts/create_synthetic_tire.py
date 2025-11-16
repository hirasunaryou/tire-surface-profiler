"""Generate a small synthetic tire-like GLB for demos."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


OUT = Path(__file__).resolve().parents[1] / "sample_data" / "synthetic_tire.glb"


def main() -> None:
    radius = 0.35
    height = 0.25
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=80)
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    x = vertices[:, 2].copy()
    vertices[:, 2] = vertices[:, 1]
    vertices[:, 1] = x

    # imprint a rim line on the side walls
    for sign in (-1, 1):
        mask = np.isclose(vertices[:, 0], sign * height / 2, atol=0.01)
        bump = 0.01 * np.exp(-((vertices[mask, 1]) ** 2) / 0.02)
        vertices[mask, 2] += bump

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(OUT), mesh, write_triangle_uvs=False, write_vertex_normals=True)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
