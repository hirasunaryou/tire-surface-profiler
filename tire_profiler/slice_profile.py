"""Slice the aligned point cloud and build an axial profile."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_glb import save_point_cloud


@dataclass
class ProfileResult:
    profile: pd.DataFrame
    sliced_points: np.ndarray
    mask: np.ndarray


def cylindrical_features(points: np.ndarray, radius: float) -> dict[str, np.ndarray]:
    r = np.linalg.norm(points[:, 1:3], axis=1)
    theta = np.arctan2(points[:, 1], points[:, 2])
    arc = theta * radius
    delta_r = r - radius
    return {
        "x": points[:, 0],
        "z": points[:, 2],
        "arc": arc,
        "delta_r": delta_r,
        "theta": theta,
        "radius": r,
    }


def slice_band(
    points: np.ndarray,
    *,
    features: dict[str, np.ndarray],
    tape_width: float,
    outer_band: float,
) -> np.ndarray:
    arc = features["arc"]
    delta_r = features["delta_r"]
    z = features["z"]
    mask = (
        (np.abs(arc) <= tape_width / 2)
        & (z > 0)
        & (np.abs(delta_r) <= outer_band)
    )
    return mask


def compute_profile(
    points: np.ndarray,
    *,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    rimline: tuple[float, float],
    nbins: int,
    out_dir: Optional[Path] = None,
    save_debug: bool = False,
) -> ProfileResult:
    if not np.any(mask):
        raise ValueError("Slice mask is empty; adjust tape width or outer band")
    x = features["x"][mask]
    arc = features["arc"][mask]
    delta_r = features["delta_r"][mask]
    z = features["z"][mask]
    alpha, beta = rimline
    z0 = alpha + beta * arc
    zprime = z - z0

    bins = np.linspace(x.min(), x.max(), nbins + 1)
    cats = pd.cut(x, bins=bins, include_lowest=True)
    df = pd.DataFrame({"x": x, "arc": arc, "zprime": zprime, "delta_r": delta_r, "z": z})
    grouped = df.groupby(cats)
    profile = grouped.agg(
        x_center=("x", "mean"),
        z_mean=("zprime", "mean"),
        z_std=("zprime", "std"),
        delta_r_mean=("delta_r", "mean"),
        samples=("zprime", "count"),
    ).dropna()

    if save_debug and out_dir:
        save_point_cloud(points[mask], out_dir / "slice_points.ply")

    return ProfileResult(profile=profile, sliced_points=points[mask], mask=mask)


def plot_profile(profile: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(profile["x_center"], profile["z_mean"], label="Z' mean")
    ax.fill_between(
        profile["x_center"],
        profile["z_mean"] - profile["z_std"],
        profile["z_mean"] + profile["z_std"],
        color="C0",
        alpha=0.2,
        label="±1σ",
    )
    ax.set_xlabel("X (axial)")
    ax.set_ylabel("Z' (radial, rim-zero)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
