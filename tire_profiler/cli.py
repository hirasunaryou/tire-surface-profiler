"""Command line interface for tire profiling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .align import align_points
from .cylinder import CylinderFitError, fit_cylinder
from .io_glb import load_point_cloud, to_numpy
from .rimline import (
    RimLineError,
    fit_rimline,
    load_rim_points,
    pick_rim_points,
    save_rim_points,
)
from .slice_profile import cylindrical_features, compute_profile, plot_profile, slice_band


def _process_single(args: argparse.Namespace, glb_path: Path) -> Path:
    out_dir = Path(args.out) / glb_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[tireprof] Loading {glb_path}")
    pcd = load_point_cloud(glb_path, voxel_size=args.voxel, sample_points=args.sample_points)
    points = to_numpy(pcd)

    print("[tireprof] Fitting cylinder …")
    try:
        model = fit_cylinder(points, threshold=args.ransac_thresh)
    except CylinderFitError as exc:
        raise SystemExit(str(exc)) from exc

    aligned, rotation, translation = align_points(
        points,
        model.point_on_axis,
        model.axis_direction,
    )

    features = cylindrical_features(aligned, model.radius)
    mask = slice_band(
        aligned,
        features=features,
        tape_width=args.tape_width,
        outer_band=args.outer_band,
    )

    rim_points = None
    rim_path = args.rim_json
    if rim_path:
        rim_points = load_rim_points(rim_path)
    else:
        if args.non_interactive:
            raise SystemExit("Rim points required in non-interactive mode")
        print("[tireprof] Launching Open3D picker for rim points …")
        rim_points = pick_rim_points(aligned)
        if args.save_rim_points:
            save_rim_points(rim_points, args.save_rim_points)

    arc = np.arctan2(rim_points[:, 1], rim_points[:, 2]) * model.radius
    try:
        rimline = fit_rimline(rim_points, arc)
    except RimLineError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        result = compute_profile(
            aligned,
            features=features,
            mask=mask,
            rimline=rimline,
            nbins=args.nbins,
            out_dir=out_dir,
            save_debug=args.save_debug,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    csv_path = out_dir / "profile.csv"
    result.profile.to_csv(csv_path, index=False)
    png_path = out_dir / "profile.png"
    plot_profile(result.profile, png_path)

    summary = {
        "rimline": {"alpha": rimline[0], "beta": rimline[1]},
        "cylinder": {
            "radius": model.radius,
            "axis_point": model.point_on_axis.tolist(),
            "axis_dir": model.axis_direction.tolist(),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[tireprof] Saved {csv_path} and {png_path}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tire rim-line profiler")
    parser.add_argument("--glb", type=Path, help="Path to a GLB file", default=None)
    parser.add_argument("--batch", type=Path, help="Process all *.glb in folder", default=None)
    parser.add_argument("--tape-width", type=float, required=True)
    parser.add_argument("--voxel", type=float, default=None, help="Voxel size for down-sampling")
    parser.add_argument("--sample-points", type=int, default=200_000)
    parser.add_argument("--ransac-thresh", type=float, default=0.003)
    parser.add_argument("--outer-band", type=float, default=0.05)
    parser.add_argument("--nbins", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rim-json", type=Path, help="Load rim picks from JSON")
    parser.add_argument("--save-rim-points", type=Path, help="Where to store picked rim points")
    parser.add_argument("--non-interactive", action="store_true", help="Fail if rim points missing")
    parser.add_argument("--save-debug", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    targets: Iterable[Path]
    if args.batch:
        targets = sorted(args.batch.glob("*.glb"))
    elif args.glb:
        targets = [args.glb]
    else:
        parser.error("Specify either --glb or --batch")
        return

    for glb_path in targets:
        _process_single(args, glb_path)


if __name__ == "__main__":
    main()
