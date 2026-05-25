"""
combine_las.py
--------------
Combine all LAS/LAZ files in a folder into a single LAS file.
Each source file is assigned a unique integer cluster_id so that
the origin of every point can be traced back after merging.

Usage
-----
  python combine_las.py \\
      --input_folder /path/to/las/files \\
      --output_las   /path/to/combined.las
"""

import os
import argparse
import numpy as np
import laspy


def combine_las_files(input_folder: str, output_path: str) -> None:
    las_files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".las", ".laz"))
    ])

    if not las_files:
        raise ValueError(f"No LAS/LAZ files found in: {input_folder}")

    print(f"Found {len(las_files)} file(s) to combine.\n")

    all_x, all_y, all_z, all_ids = [], [], [], []

    for cluster_id, fname in enumerate(las_files):
        fpath = os.path.join(input_folder, fname)
        las   = laspy.read(fpath)
        n     = len(las.x)
        print(f"  [{cluster_id:>3}] {fname}  ({n:,} points)")

        all_x.append(np.asarray(las.x))
        all_y.append(np.asarray(las.y))
        all_z.append(np.asarray(las.z))
        all_ids.append(np.full(n, cluster_id, dtype=np.uint16))

    header      = laspy.LasHeader(point_format=3, version="1.3")
    combined    = laspy.LasData(header)
    combined.add_extra_dim(laspy.ExtraBytesParams(name="cluster_id", type=np.uint16))

    combined.x          = np.concatenate(all_x)
    combined.y          = np.concatenate(all_y)
    combined.z          = np.concatenate(all_z)
    combined.cluster_id = np.concatenate(all_ids)

    combined.write(output_path)
    print(f"\nTotal points : {len(combined.x):,}")
    print(f"Saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine multiple LAS/LAZ files and assign a unique cluster_id per source file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input_folder", required=True,
                        help="Folder containing LAS/LAZ files to combine.")
    parser.add_argument("--output_las", required=True,
                        help="Path for the combined output LAS file.")
    args = parser.parse_args()
    combine_las_files(args.input_folder, args.output_las)


if __name__ == "__main__":
    main()
