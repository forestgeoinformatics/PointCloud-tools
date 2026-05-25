"""
compute_metrics_jakteristics.py
--------------------------------
Fast variant of compute_metrics_selected.py that uses the jakteristics
library (parallelised C++ backend) for geometric feature computation.
Typically 10-50× faster than the pure-Python scipy variant.

Features computed (per radius)
-------------------------------
  sphericity      — from jakteristics
  Density         — number_of_neighbors / sphere volume
  roughness       — surface_variation from jakteristics
  intensity_mean  — mean intensity in neighbourhood via cKDTree
                    (only if an Intensity field is present;
                     jakteristics does not carry radiometric attributes)

Each feature is suffixed with the radius, e.g. sphericity_r0.05.

Requirements
------------
  pip install jakteristics

Usage
-----
  python compute_metrics_jakteristics.py \\
      --input_folder /path/to/las/files \\
      --radii 0.05 0.10 0.15           \\
      [--output_folder /path/to/output]
"""

import os
import argparse
import numpy as np
import laspy

try:
    from jakteristics import compute_features
except ImportError:
    raise ImportError(
        "jakteristics is not installed. Run:  pip install jakteristics"
    )


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

DESIRED_FEATURES = ["sphericity", "number_of_neighbors", "surface_variation"]


def compute_jakteristics_metrics(
    points: np.ndarray,
    intensity: np.ndarray | None,
    radius: float,
) -> dict[str, np.ndarray]:

    features = compute_features(
        points.astype(np.float64),
        search_radius=radius,
        feature_names=DESIRED_FEATURES,
    )

    feat_idx    = {name: i for i, name in enumerate(DESIRED_FEATURES)}
    sphericity  = features[:, feat_idx["sphericity"]].astype(np.float32)
    point_count = features[:, feat_idx["number_of_neighbors"]].astype(np.float32)
    roughness   = features[:, feat_idx["surface_variation"]].astype(np.float32)

    volume  = (4.0 / 3.0) * np.pi * (radius ** 3)
    density = (point_count / volume).astype(np.float32)

    metrics = {
        'sphericity': sphericity,
        'Density'   : density,
        'roughness' : roughness,
    }

    if intensity is not None:
        from scipy.spatial import cKDTree
        tree     = cKDTree(points)
        idxs     = tree.query_ball_point(points, r=radius, workers=-1)
        int_mean = np.array(
            [intensity[ix].mean() if len(ix) else 0.0 for ix in idxs],
            dtype=np.float32,
        )
        metrics['intensity_mean'] = int_mean

    return metrics


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute selected point-cloud features via jakteristics (fast C++ backend).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input_folder", required=True,
                        help="Folder containing input LAS/LAZ files.")
    parser.add_argument("--radii", nargs="+", type=float, required=True,
                        help="One or more neighbourhood radii in metres, e.g. 0.05 0.10 0.15")
    parser.add_argument("--output_folder", default=None,
                        help="Output folder (default: input folder, files suffixed '_metrics').")
    args = parser.parse_args()

    out_dir = args.output_folder or args.input_folder
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([
        os.path.join(args.input_folder, f)
        for f in os.listdir(args.input_folder)
        if f.lower().endswith((".las", ".laz"))
    ])

    if not files:
        print("[ERROR] No LAS/LAZ files found.")
        return

    print(f"Files   : {len(files)}")
    print(f"Radii   : {args.radii}")
    print(f"Output  : {out_dir}\n")

    for idx, filepath in enumerate(files, 1):
        fname = os.path.basename(filepath)
        print(f"[{idx}/{len(files)}] {fname}")

        las    = laspy.read(filepath)
        points = np.column_stack((las.x, las.y, las.z))

        dim_names_lower = [d.lower() for d in las.point_format.dimension_names]
        if 'intensity' in dim_names_lower:
            intensity = np.asarray(las.intensity, dtype=np.float32)
            print("  Intensity field found.")
        else:
            intensity = None
            print("  [WARN] No intensity field — intensity_mean will be skipped.")

        all_metrics: dict[str, np.ndarray] = {}

        for radius in args.radii:
            print(f"  Computing features at radius {radius} …")
            metrics = compute_jakteristics_metrics(points, intensity, radius)
            for k, v in metrics.items():
                all_metrics[f"{k}_r{radius:.2f}"] = v

        out_las = laspy.LasData(las.header)
        out_las.points = las.points.copy()

        for k, v in all_metrics.items():
            out_las.add_extra_dim(laspy.ExtraBytesParams(name=k, type=np.float32))
            out_las[k] = v

        out_path = os.path.join(out_dir, fname.replace(".las", "_metrics.las").replace(".laz", "_metrics.las"))
        out_las.write(out_path)
        print(f"  Saved → {out_path}\n")

    print("All files processed.")


if __name__ == "__main__":
    main()
