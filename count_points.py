"""
count_points.py
---------------
Summarise LAS/LAZ files in a folder:
  • Number of points per file
  • Average k-nearest-neighbour distance (proxy for point spacing)

Results are printed to the console and optionally saved to a CSV.

Usage
-----
  python count_points.py \\
      --input_folder /path/to/las/files  \\
      [--output_csv  /path/to/report.csv] \\
      [--k_neighbors 32]                  \\
      [--max_points  2000000]
"""

import os
import argparse
import numpy as np
import laspy
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def analyse_file(
    filepath: str,
    k: int,
    max_points: int | None,
) -> dict:

    las       = laspy.read(filepath)
    n_total   = len(las.points)
    points    = np.column_stack((las.x, las.y, las.z))

    sampled = n_total
    if max_points and n_total > max_points:
        idx    = np.random.choice(n_total, max_points, replace=False)
        points = points[idx]
        sampled = max_points

    nbrs      = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(points)
    dists, _  = nbrs.kneighbors(points)
    avg_dist  = float(np.mean(dists[:, 1:]))   # skip self (column 0)

    return {
        "filename"       : os.path.basename(filepath),
        "total_points"   : n_total,
        "sampled_points" : sampled,
        f"avg_{k}nn_dist": round(avg_dist, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count points and compute average NN distance for LAS files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input_folder", required=True,
                        help="Folder containing LAS/LAZ files.")
    parser.add_argument("--output_csv", default=None,
                        help="Optional path to save results as a CSV file.")
    parser.add_argument("--k_neighbors", type=int, default=32,
                        help="Number of nearest neighbours for distance computation (default: 32).")
    parser.add_argument("--max_points", type=int, default=2_000_000,
                        help="Subsample large clouds to this many points before computing "
                             "NN distances (default: 2,000,000).  Set 0 to disable.")
    args = parser.parse_args()

    max_pts = args.max_points if args.max_points > 0 else None

    files = sorted([
        os.path.join(args.input_folder, f)
        for f in os.listdir(args.input_folder)
        if f.lower().endswith((".las", ".laz"))
    ])

    if not files:
        print("[ERROR] No LAS/LAZ files found.")
        return

    print(f"Files       : {len(files)}")
    print(f"k neighbors : {args.k_neighbors}")
    if max_pts:
        print(f"Max points  : {max_pts:,} (for NN computation)")
    print()

    results = []
    for idx, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        print(f"[{idx}/{len(files)}] {fname}")
        row = analyse_file(fpath, args.k_neighbors, max_pts)
        results.append(row)
        print(f"  Total points   : {row['total_points']:,}")
        if row['sampled_points'] < row['total_points']:
            print(f"  Sampled        : {row['sampled_points']:,}")
        print(f"  Avg {args.k_neighbors}-NN dist : {row[f'avg_{args.k_neighbors}nn_dist']:.4f} m\n")

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved → {args.output_csv}")


if __name__ == "__main__":
    main()
