"""
compute_metrics.py
------------------
Compute 3D geometric and radiometric point-cloud features for every point
in each LAS file in a folder, then save the results as new scalar fields
in output LAS files.

Features computed (per radius)
-------------------------------
Eigenvalue-based  : linearity, planarity, scattering, omnivariance,
                    anisotropy, eigentropy, eigensum, curvature,
                    sphericity, verticality, 3d_eigen_1/2/3
Neighbourhood     : Density, ZRange, ZStd, number,
                    min_distance, max_distance, roughness
Radiometric       : intensity_mean, intensity_std, intensity_range
                    (only if an Intensity field is present)

Each feature is suffixed with the radius, e.g. sphericity_r0.05.

Usage
-----
  python compute_metrics.py \\
      --input_folder /path/to/las/files \\
      --radii 0.05 0.10 0.15           \\
      [--output_folder /path/to/output] \\
      [--min_neighbors 5]
"""

import os
import argparse
import numpy as np
from scipy.spatial import cKDTree
import laspy
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_3d_eigen_metrics(
    points: np.ndarray,
    intensity: np.ndarray | None,
    tree: cKDTree,
    radius: float,
    min_neighbors: int = 5,
) -> dict[str, np.ndarray]:

    N = len(points)
    volume = (4.0 / 3.0) * np.pi * (radius ** 3)

    # initialise output arrays
    linearity     = np.zeros(N, np.float32)
    planarity     = np.zeros(N, np.float32)
    scattering    = np.zeros(N, np.float32)
    omnivariance  = np.zeros(N, np.float32)
    anisotropy    = np.zeros(N, np.float32)
    eigentropy    = np.zeros(N, np.float32)
    eigensum      = np.zeros(N, np.float32)
    curvature     = np.zeros(N, np.float32)
    sphericity    = np.zeros(N, np.float32)
    verticality   = np.zeros(N, np.float32)
    ev1_          = np.zeros(N, np.float32)
    ev2_          = np.zeros(N, np.float32)
    ev3_          = np.zeros(N, np.float32)
    density       = np.zeros(N, np.float32)
    z_range       = np.zeros(N, np.float32)
    z_std         = np.zeros(N, np.float32)
    number        = np.zeros(N, np.float32)
    roughness     = np.zeros(N, np.float32)
    min_dist      = np.zeros(N, np.float32)
    max_dist      = np.zeros(N, np.float32)
    intensity_mean  = np.zeros(N, np.float32)
    intensity_std   = np.zeros(N, np.float32)
    intensity_range = np.zeros(N, np.float32)

    for i in tqdm(range(N), desc=f"  radius={radius}", leave=False):
        idx = tree.query_ball_point(points[i], r=radius)
        if len(idx) < min_neighbors:
            continue

        neighbors = points[idx]
        centered  = neighbors - neighbors.mean(axis=0)

        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)          # ascending order
        ev1, ev2, ev3 = eigvals[2], eigvals[1], eigvals[0]   # ev1 ≥ ev2 ≥ ev3

        total = ev1 + ev2 + ev3
        if total == 0 or ev1 == 0:
            continue

        normal = eigvecs[:, 0]   # smallest eigenvector ≈ surface normal

        linearity[i]   = (ev1 - ev2) / ev1
        planarity[i]   = (ev2 - ev3) / ev1
        scattering[i]  = ev3 / total
        omnivariance[i]= np.cbrt(ev1 * ev2 * ev3) if ev1 * ev2 * ev3 > 0 else 0.0
        anisotropy[i]  = (ev1 - ev3) / ev1
        eigentropy[i]  = -np.sum(eigvals * np.log(eigvals + 1e-10))
        eigensum[i]    = total
        curvature[i]   = ev3 / total
        sphericity[i]  = ev3 / ev1
        verticality[i] = 1.0 - abs(normal[2])
        ev1_[i], ev2_[i], ev3_[i] = ev1, ev2, ev3

        number[i]  = len(idx)
        density[i] = len(idx) / volume
        z_vals     = neighbors[:, 2]
        z_range[i] = z_vals.ptp()
        z_std[i]   = z_vals.std()

        dists      = np.linalg.norm(neighbors - points[i], axis=1)
        min_dist[i]= dists.min()
        max_dist[i]= dists.max()

        # roughness = std of distances to the best-fit plane
        _, _, vh    = np.linalg.svd(np.cov(centered.T))
        roughness[i]= np.std(np.dot(centered, vh[-1]))

        if intensity is not None:
            intens          = intensity[idx]
            intensity_mean[i]  = intens.mean()
            intensity_std[i]   = intens.std()
            intensity_range[i] = intens.ptp()

    metrics = {
        'linearity'   : linearity,
        'planarity'   : planarity,
        'scattering'  : scattering,
        'omnivariance': omnivariance,
        'anisotropy'  : anisotropy,
        'eigentropy'  : eigentropy,
        'eigensum'    : eigensum,
        'curvature'   : curvature,
        'sphericity'  : sphericity,
        'verticality' : verticality,
        '3d_eigen_1'  : ev1_,
        '3d_eigen_2'  : ev2_,
        '3d_eigen_3'  : ev3_,
        'Density'     : density,
        'ZRange'      : z_range,
        'ZStd'        : z_std,
        'number'      : number,
        'roughness'   : roughness,
        'min_distance': min_dist,
        'max_distance': max_dist,
    }

    if intensity is not None:
        metrics['intensity_mean']  = intensity_mean
        metrics['intensity_std']   = intensity_std
        metrics['intensity_range'] = intensity_range

    return metrics


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute 3D geometric and radiometric point-cloud features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input_folder", required=True,
                        help="Folder containing input LAS/LAZ files.")
    parser.add_argument("--radii", nargs="+", type=float, required=True,
                        help="One or more neighbourhood radii in metres, e.g. 0.05 0.10 0.15")
    parser.add_argument("--output_folder", default=None,
                        help="Output folder (default: input folder, files suffixed '_metrics').")
    parser.add_argument("--min_neighbors", type=int, default=5,
                        help="Minimum neighbours required to compute metrics (default: 5).")
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

    print(f"Files     : {len(files)}")
    print(f"Radii     : {args.radii}")
    print(f"Output    : {out_dir}\n")

    for idx, filepath in enumerate(files, 1):
        fname = os.path.basename(filepath)
        print(f"[{idx}/{len(files)}] {fname}")

        las    = laspy.read(filepath)
        points = np.column_stack((las.x, las.y, las.z))

        # intensity — try both 'Intensity' (extra dim) and 'intensity' (standard)
        dim_names_lower = [d.lower() for d in las.point_format.dimension_names]
        if 'intensity' in dim_names_lower:
            intensity = np.asarray(las.intensity, dtype=np.float32)
            print("  Intensity field found.")
        else:
            intensity = None
            print("  [WARN] No intensity field — intensity metrics will be skipped.")

        tree = cKDTree(points)
        all_metrics: dict[str, np.ndarray] = {}

        for radius in args.radii:
            metrics = compute_3d_eigen_metrics(
                points, intensity, tree, radius, args.min_neighbors
            )
            for k, v in metrics.items():
                all_metrics[f"{k}_r{radius:.2f}"] = v.astype(np.float32)

        # preserve all original data and append new metrics
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
