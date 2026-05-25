"""
compute_metrics_selected.py
----------------------------
Faster variant of compute_metrics.py that computes only the features you
specify via --features, rather than the full set.  Useful when you only
need a subset of features downstream or when processing time is a
constraint.

Available features
------------------
Eigenvalue-based (require covariance matrix — computed together for free):
  linearity       (ev1 - ev2) / ev1
  planarity       (ev2 - ev3) / ev1
  scattering      ev3 / (ev1 + ev2 + ev3)
  omnivariance    (ev1 * ev2 * ev3) ^ (1/3)
  anisotropy      (ev1 - ev3) / ev1
  eigentropy      -sum(evi * log(evi))
  eigensum        ev1 + ev2 + ev3
  curvature       ev3 / (ev1 + ev2 + ev3)
  sphericity      ev3 / ev1
  verticality     1 - |normal_z|
  3d_eigen_1      largest eigenvalue
  3d_eigen_2      middle eigenvalue
  3d_eigen_3      smallest eigenvalue

Neighbourhood:
  Density         point count / sphere volume
  ZRange          max(z) - min(z) in neighbourhood
  ZStd            std(z) in neighbourhood
  number          neighbour count
  roughness       std of distances to best-fit plane
  min_distance    distance to nearest neighbour
  max_distance    distance to farthest neighbour in radius

Radiometric (only if an Intensity field is present):
  intensity_mean  mean intensity in neighbourhood
  intensity_std   std of intensity in neighbourhood
  intensity_range max - min intensity in neighbourhood

Each output field is suffixed with the radius, e.g. sphericity_r0.05.

Usage
-----
  # Compute just three features at two radii
  python compute_metrics_selected.py \\
      --input_folder /path/to/las/files \\
      --radii        0.05 0.10          \\
      --features     sphericity Density roughness

  # Compute eigenvalue features + intensity mean
  python compute_metrics_selected.py \\
      --input_folder /path/to/las/files \\
      --radii        0.05               \\
      --features     linearity planarity sphericity verticality intensity_mean

  # List all available feature names and exit
  python compute_metrics_selected.py --list_features
"""

import os
import argparse
import numpy as np
from scipy.spatial import cKDTree
import laspy
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------

# Features that need the covariance / eigendecomposition
EIGEN_FEATURES = {
    'linearity', 'planarity', 'scattering', 'omnivariance',
    'anisotropy', 'eigentropy', 'eigensum', 'curvature',
    'sphericity', 'verticality', '3d_eigen_1', '3d_eigen_2', '3d_eigen_3',
}

# Features that need only the neighbour positions
NEIGHBOURHOOD_FEATURES = {
    'Density', 'ZRange', 'ZStd', 'number',
    'roughness', 'min_distance', 'max_distance',
}

# Features that need intensity values
INTENSITY_FEATURES = {
    'intensity_mean', 'intensity_std', 'intensity_range',
}

ALL_FEATURES = EIGEN_FEATURES | NEIGHBOURHOOD_FEATURES | INTENSITY_FEATURES


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_selected_metrics(
    points: np.ndarray,
    intensity: np.ndarray | None,
    tree: cKDTree,
    radius: float,
    features: set[str],
    min_neighbors: int = 5,
) -> dict[str, np.ndarray]:

    N      = len(points)
    volume = (4.0 / 3.0) * np.pi * (radius ** 3)

    # allocate only what was requested
    arrays: dict[str, np.ndarray] = {f: np.zeros(N, np.float32) for f in features}

    need_eigen  = bool(features & EIGEN_FEATURES)
    need_rough  = 'roughness' in features
    need_intens = bool(features & INTENSITY_FEATURES) and intensity is not None

    for i in tqdm(range(N), desc=f"  radius={radius}", leave=False):
        idx = tree.query_ball_point(points[i], r=radius)
        if len(idx) < min_neighbors:
            continue

        neighbors = points[idx]
        centered  = neighbors - neighbors.mean(axis=0)

        # --- eigendecomposition (shared cost for all eigen features) ---
        ev1 = ev2 = ev3 = 0.0
        normal = None
        if need_eigen:
            cov              = np.cov(centered, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)          # ascending
            ev1, ev2, ev3    = eigvals[2], eigvals[1], eigvals[0]
            total            = ev1 + ev2 + ev3
            if total == 0 or ev1 == 0:
                continue
            normal = eigvecs[:, 0]

            if 'linearity'    in features: arrays['linearity'][i]    = (ev1 - ev2) / ev1
            if 'planarity'    in features: arrays['planarity'][i]    = (ev2 - ev3) / ev1
            if 'scattering'   in features: arrays['scattering'][i]   = ev3 / total
            if 'omnivariance' in features: arrays['omnivariance'][i] = np.cbrt(ev1 * ev2 * ev3) if ev1 * ev2 * ev3 > 0 else 0.0
            if 'anisotropy'   in features: arrays['anisotropy'][i]   = (ev1 - ev3) / ev1
            if 'eigentropy'   in features: arrays['eigentropy'][i]   = -np.sum(eigvals * np.log(eigvals + 1e-10))
            if 'eigensum'     in features: arrays['eigensum'][i]     = total
            if 'curvature'    in features: arrays['curvature'][i]    = ev3 / total
            if 'sphericity'   in features: arrays['sphericity'][i]   = ev3 / ev1
            if 'verticality'  in features: arrays['verticality'][i]  = 1.0 - abs(normal[2])
            if '3d_eigen_1'   in features: arrays['3d_eigen_1'][i]   = ev1
            if '3d_eigen_2'   in features: arrays['3d_eigen_2'][i]   = ev2
            if '3d_eigen_3'   in features: arrays['3d_eigen_3'][i]   = ev3

        # --- neighbourhood features ---
        if 'Density'      in features: arrays['Density'][i]      = len(idx) / volume
        if 'number'       in features: arrays['number'][i]       = len(idx)
        if 'ZRange'       in features:
            z = neighbors[:, 2]
            arrays['ZRange'][i] = z.ptp()
        if 'ZStd'         in features:
            z = neighbors[:, 2] if 'ZRange' not in features else z
            arrays['ZStd'][i]   = z.std()

        if 'min_distance' in features or 'max_distance' in features:
            dists = np.linalg.norm(neighbors - points[i], axis=1)
            if 'min_distance' in features: arrays['min_distance'][i] = dists.min()
            if 'max_distance' in features: arrays['max_distance'][i] = dists.max()

        if need_rough:
            _, _, vh         = np.linalg.svd(np.cov(centered.T))
            arrays['roughness'][i] = np.std(np.dot(centered, vh[-1]))

        # --- intensity features ---
        if need_intens:
            intens = intensity[idx]
            if 'intensity_mean'  in features: arrays['intensity_mean'][i]  = intens.mean()
            if 'intensity_std'   in features: arrays['intensity_std'][i]   = intens.std()
            if 'intensity_range' in features: arrays['intensity_range'][i] = intens.ptp()

    return arrays


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute a user-defined subset of 3D point-cloud features (faster variant).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input_folder", default=None,
                        help="Folder containing input LAS/LAZ files.")
    parser.add_argument("--radii", nargs="+", type=float, default=None,
                        help="One or more neighbourhood radii in metres, e.g. 0.05 0.10 0.15")
    parser.add_argument("--features", nargs="+", default=None,
                        help="Features to compute. Run --list_features to see all options.")
    parser.add_argument("--output_folder", default=None,
                        help="Output folder (default: input folder, files suffixed '_metrics').")
    parser.add_argument("--min_neighbors", type=int, default=5,
                        help="Minimum neighbours required to compute metrics (default: 5).")
    parser.add_argument("--list_features", action="store_true",
                        help="Print all available feature names and exit.")
    args = parser.parse_args()

    # --- list features and exit ---
    if args.list_features:
        print("\nAvailable features for --features\n")
        print("Eigenvalue-based:")
        for f in sorted(EIGEN_FEATURES):
            print(f"  {f}")
        print("\nNeighbourhood:")
        for f in sorted(NEIGHBOURHOOD_FEATURES):
            print(f"  {f}")
        print("\nRadiometric (requires Intensity field in LAS):")
        for f in sorted(INTENSITY_FEATURES):
            print(f"  {f}")
        print()
        return

    # --- validate required args ---
    if not args.input_folder:
        parser.error("--input_folder is required.")
    if not args.radii:
        parser.error("--radii is required.")
    if not args.features:
        parser.error("--features is required. Run --list_features to see options.")

    # --- validate feature names ---
    unknown = [f for f in args.features if f not in ALL_FEATURES]
    if unknown:
        parser.error(
            f"Unknown feature(s): {unknown}\n"
            f"Run --list_features to see all valid names."
        )

    requested = set(args.features)
    out_dir   = args.output_folder or args.input_folder
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
    print(f"Features  : {sorted(requested)}")
    print(f"Output    : {out_dir}\n")

    for idx, filepath in enumerate(files, 1):
        fname = os.path.basename(filepath)
        print(f"[{idx}/{len(files)}] {fname}")

        las    = laspy.read(filepath)
        points = np.column_stack((las.x, las.y, las.z))

        dim_names_lower = [d.lower() for d in las.point_format.dimension_names]
        has_intensity   = 'intensity' in dim_names_lower

        if requested & INTENSITY_FEATURES:
            if has_intensity:
                intensity = np.asarray(las.intensity, dtype=np.float32)
                print("  Intensity field found.")
            else:
                intensity = None
                print("  [WARN] No intensity field — intensity features will be skipped.")
        else:
            intensity = None

        tree        = cKDTree(points)
        all_metrics: dict[str, np.ndarray] = {}

        for radius in args.radii:
            metrics = compute_selected_metrics(
                points, intensity, tree, radius, requested, args.min_neighbors
            )
            for k, v in metrics.items():
                all_metrics[f"{k}_r{radius:.2f}"] = v.astype(np.float32)

        out_las        = laspy.LasData(las.header)
        out_las.points = las.points.copy()

        for k, v in all_metrics.items():
            out_las.add_extra_dim(laspy.ExtraBytesParams(name=k, type=np.float32))
            out_las[k] = v

        out_path = os.path.join(
            out_dir,
            fname.replace(".las", "_metrics.las").replace(".laz", "_metrics.las")
        )
        out_las.write(out_path)
        print(f"  Saved → {out_path}\n")

    print("All files processed.")


if __name__ == "__main__":
    main()
