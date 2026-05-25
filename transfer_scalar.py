"""
transfer_scalar.py
------------------
Transfer one or more scalar fields from a SOURCE point cloud to a TARGET
point cloud using nearest-neighbour matching (cKDTree).

For each point in the target cloud, the closest point in the source cloud
(within --radius) is found and the requested field values are copied.
Points in the target that have no source neighbour within the radius are
assigned a fill value (default: 0).

Typical use-cases
-----------------
* Copy intensity from an original full-density scan to a subsampled cloud.
* Transfer predicted labels (Class, pred, BO …) computed on a subsampled
  cloud back to the original full-density cloud.
* Any situation where a field exists in one cloud and needs to be projected
  onto another cloud of different density.

Usage
-----
  python transfer_scalar.py \\
      --source_folder  /path/to/source_las_files \\
      --target_folder  /path/to/target_las_files \\
      --output_folder  /path/to/output            \\
      --fields         Intensity Class BO          \\
      --radius         0.08                        \\
      --fill           0

File matching
-------------
Each target file is matched to a source file by the longest common
substring of their base names (case-insensitive).  A warning is printed
when no match is found and that target file is skipped.
"""

import os
import argparse
import gc
import numpy as np
import laspy
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _best_match(target_name: str, source_names: list[str]) -> str | None:
    """Return the source filename that shares the longest common prefix with
    target_name (case-insensitive base names, without extension)."""
    target_base = os.path.splitext(target_name)[0].lower()
    best, best_len = None, 0
    for sname in source_names:
        sbase = os.path.splitext(sname)[0].lower()
        common = len(os.path.commonprefix([target_base, sbase]))
        if common > best_len:
            best, best_len = sname, common
    return best if best_len > 0 else None


def _read_field(las: laspy.LasData, field: str) -> np.ndarray | None:
    """Return the numpy array for *field* from *las*, or None if absent."""
    available = [d.lower() for d in las.point_format.dimension_names]
    if field.lower() in available:
        return np.asarray(las[field])
    return None


def _field_dtype(arr: np.ndarray) -> type:
    """Map a numpy dtype to a laspy-compatible extra-bytes type."""
    if np.issubdtype(arr.dtype, np.floating):
        return np.float32
    if arr.dtype.itemsize == 1:
        return np.uint8
    if arr.dtype.itemsize == 2:
        return np.uint16
    return np.int32


# ---------------------------------------------------------------------------
# Core transfer function
# ---------------------------------------------------------------------------

def transfer_scalar(
    source_path: str,
    target_path: str,
    output_path: str,
    fields: list[str],
    radius: float,
    fill: float,
) -> None:

    print(f"  Source : {os.path.basename(source_path)}")
    print(f"  Target : {os.path.basename(target_path)}")

    # --- load source ---
    src = laspy.read(source_path)
    src_xyz = np.column_stack((src.x, src.y, src.z))

    # validate requested fields exist in source
    valid_fields = []
    for f in fields:
        arr = _read_field(src, f)
        if arr is None:
            print(f"  [WARN] Field '{f}' not found in source — skipping.")
        else:
            valid_fields.append(f)

    if not valid_fields:
        print("  [ERROR] No valid fields to transfer. Skipping file pair.")
        return

    # --- load target ---
    tgt = laspy.read(target_path)
    tgt_xyz = np.column_stack((tgt.x, tgt.y, tgt.z))
    n_tgt = len(tgt_xyz)

    print(f"  Source points : {len(src_xyz):,}")
    print(f"  Target points : {n_tgt:,}")

    # --- nearest-neighbour query ---
    print("  Building KDTree …")
    tree = cKDTree(src_xyz)
    print("  Querying nearest neighbours …")
    distances, indices = tree.query(tgt_xyz, k=1, distance_upper_bound=radius, workers=-1)

    within_radius = np.isfinite(distances)
    print(f"  Matched {within_radius.sum():,} / {n_tgt:,} target points within radius {radius} m")

    del src_xyz, tgt_xyz
    gc.collect()

    # --- build output LAS (copy all target data first) ---
    out_las = laspy.LasData(tgt.header)
    out_las.points = tgt.points.copy()

    # --- transfer each field ---
    for f in valid_fields:
        src_arr = np.asarray(src[f])
        transferred = np.full(n_tgt, fill, dtype=np.float32)
        transferred[within_radius] = src_arr[indices[within_radius]].astype(np.float32)

        # add as extra dimension (skip if already present in target)
        existing = [d.lower() for d in out_las.point_format.dimension_names]
        if f.lower() not in existing:
            las_dtype = _field_dtype(src_arr)
            out_las.add_extra_dim(laspy.ExtraBytesParams(name=f, type=las_dtype))

        out_las[f] = transferred.astype(out_las[f].dtype)
        print(f"  Transferred field '{f}'")

    out_las.write(output_path)
    print(f"  Saved  → {output_path}\n")

    del src, tgt, out_las
    gc.collect()


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def batch_transfer(args: argparse.Namespace) -> None:
    os.makedirs(args.output_folder, exist_ok=True)

    target_files = sorted([
        f for f in os.listdir(args.target_folder)
        if f.lower().endswith((".las", ".laz"))
    ])
    source_files = sorted([
        f for f in os.listdir(args.source_folder)
        if f.lower().endswith((".las", ".laz"))
    ])

    if not target_files:
        print("[ERROR] No LAS/LAZ files found in target folder.")
        return
    if not source_files:
        print("[ERROR] No LAS/LAZ files found in source folder.")
        return

    print(f"Target files : {len(target_files)}")
    print(f"Source files : {len(source_files)}")
    print(f"Fields       : {args.fields}")
    print(f"Radius       : {args.radius} m")
    print(f"Fill value   : {args.fill}\n")

    matched = skipped = 0

    for tname in target_files:
        sname = _best_match(tname, source_files)
        if sname is None:
            print(f"[WARN] No source match for '{tname}' — skipping.\n")
            skipped += 1
            continue

        src_path = os.path.join(args.source_folder, sname)
        tgt_path = os.path.join(args.target_folder, tname)
        out_path = os.path.join(
            args.output_folder,
            os.path.splitext(tname)[0] + "_transferred.las"
        )

        print(f"[{matched + 1}/{len(target_files)}]")
        transfer_scalar(src_path, tgt_path, out_path, args.fields, args.radius, args.fill)
        matched += 1

    print(f"Done. Processed {matched} file(s), skipped {skipped}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer scalar field(s) from source LAS clouds to target LAS clouds via nearest-neighbour matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source_folder", required=True,
                        help="Folder containing source LAS/LAZ files (fields are read from here).")
    parser.add_argument("--target_folder", required=True,
                        help="Folder containing target LAS/LAZ files (fields are written here).")
    parser.add_argument("--output_folder", required=True,
                        help="Folder where output LAS files are written.")
    parser.add_argument("--fields", nargs="+", required=True,
                        help="One or more field names to transfer, e.g. --fields Intensity Class BO")
    parser.add_argument("--radius", type=float, default=0.08,
                        help="Maximum search radius in metres (default: 0.08).")
    parser.add_argument("--fill", type=float, default=0.0,
                        help="Fill value for target points with no source neighbour within radius (default: 0).")

    args = parser.parse_args()
    batch_transfer(args)


if __name__ == "__main__":
    main()
