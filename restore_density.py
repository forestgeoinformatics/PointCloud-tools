"""
restore_density.py
------------------
Restore the original point density of a subsampled point cloud by pulling
back matching points from the original full-density scan.

Workflow
--------
1. Read the subsampled (reference) cloud.
2. Expand its bounding box by --padding metres and use it to crop the
   original large cloud in memory — avoiding loading the entire file.
3. From the cropped region, keep only points within --radius of at least
   one reference point (proximity filter via cKDTree).
4. Write the restored points to an output LAS file.

The output contains the original-density points from the region covered
by the subsampled cloud — NOT a merge of both clouds.  If you need the
subsampled points included as well, use --include_reference.

This bbox pre-filter makes the script viable for very large original
clouds (full forest scans, full flight strips) without loading the
entire file into RAM.

Usage
-----
  python restore_density.py \\
      --original_folder   /path/to/original_dense_las   \\
      --reference_folder  /path/to/subsampled_las        \\
      --output_folder     /path/to/output                \\
      --radius            0.08                           \\
      --padding           0.5                            \\
      [--include_reference]

File matching
-------------
Each reference file is matched to an original file by the longest common
substring of their base names (case-insensitive).  A warning is printed
when no match is found.
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

def _best_match(ref_name: str, original_names: list[str]) -> str | None:
    """Return the original filename that shares the longest common prefix
    with ref_name (case-insensitive, without extension)."""
    ref_base = os.path.splitext(ref_name)[0].lower()
    best, best_len = None, 0
    for oname in original_names:
        obase = os.path.splitext(oname)[0].lower()
        common = len(os.path.commonprefix([ref_base, obase]))
        if common > best_len:
            best, best_len = oname, common
    return best if best_len > 0 else None


def _copy_extra_dims(src_las: laspy.LasData,
                     dst_las: laspy.LasData,
                     mask: np.ndarray) -> None:
    """Copy all extra dimensions from src_las[mask] into dst_las."""
    standard = {
        'x', 'y', 'z', 'X', 'Y', 'Z',
        'intensity', 'return_number', 'number_of_returns',
        'scan_direction_flag', 'edge_of_flight_line', 'classification',
        'synthetic', 'key_point', 'withheld', 'scan_angle_rank',
        'user_data', 'point_source_id', 'gps_time', 'red', 'green', 'blue',
    }
    for dim in src_las.point_format.dimension_names:
        if dim in standard:
            continue
        arr = np.asarray(src_las[dim])[mask]
        dst_las.add_extra_dim(laspy.ExtraBytesParams(name=dim, type=arr.dtype.type))
        dst_las[dim] = arr


# ---------------------------------------------------------------------------
# Core restoration function
# ---------------------------------------------------------------------------

def restore_density(
    original_path: str,
    reference_path: str,
    output_path: str,
    radius: float,
    padding: float,
    include_reference: bool,
) -> None:

    print(f"  Original  : {os.path.basename(original_path)}")
    print(f"  Reference : {os.path.basename(reference_path)}")

    # --- load reference (subsampled) cloud ---
    ref_las = laspy.read(reference_path)
    ref_xyz = np.column_stack((ref_las.x, ref_las.y, ref_las.z))
    n_ref = len(ref_xyz)
    print(f"  Reference points : {n_ref:,}")

    # --- compute padded bounding box ---
    min_x = ref_xyz[:, 0].min() - padding
    max_x = ref_xyz[:, 0].max() + padding
    min_y = ref_xyz[:, 1].min() - padding
    max_y = ref_xyz[:, 1].max() + padding
    min_z = ref_xyz[:, 2].min() - padding
    max_z = ref_xyz[:, 2].max() + padding

    # --- load and crop original cloud ---
    print("  Loading and cropping original cloud …")
    orig_las = laspy.read(original_path)
    bbox_mask = (
        (orig_las.x >= min_x) & (orig_las.x <= max_x) &
        (orig_las.y >= min_y) & (orig_las.y <= max_y) &
        (orig_las.z >= min_z) & (orig_las.z <= max_z)
    )
    orig_xyz = np.column_stack((
        orig_las.x[bbox_mask],
        orig_las.y[bbox_mask],
        orig_las.z[bbox_mask],
    ))
    print(f"  Original points in bbox : {len(orig_xyz):,}")

    # --- proximity filter: keep original points within radius of any reference point ---
    print("  Building KDTree on reference cloud …")
    ref_tree = cKDTree(ref_xyz)

    print("  Running proximity filter …")
    # For each original point, find distance to nearest reference point
    dists, _ = ref_tree.query(orig_xyz, k=1, workers=-1)
    proximity_mask = dists <= radius
    n_kept = proximity_mask.sum()
    print(f"  Points within radius {radius} m : {n_kept:,} / {len(orig_xyz):,}")

    del ref_xyz, dists
    gc.collect()

    if n_kept == 0:
        print("  [WARN] No original points found within radius. Check --radius and --padding values.")
        return

    # --- build output LAS ---
    header = laspy.LasHeader(point_format=orig_las.header.point_format.id,
                             version=orig_las.header.version)
    out_las = laspy.LasData(header)

    # indices into the full original array
    full_indices = np.where(bbox_mask)[0][proximity_mask]

    if include_reference:
        # merge: restored original points + reference points
        out_x = np.concatenate([orig_las.x[full_indices], ref_las.x])
        out_y = np.concatenate([orig_las.y[full_indices], ref_las.y])
        out_z = np.concatenate([orig_las.z[full_indices], ref_las.z])
        print(f"  Output points (restored + reference) : {len(out_x):,}")
    else:
        out_x = orig_las.x[full_indices]
        out_y = orig_las.y[full_indices]
        out_z = orig_las.z[full_indices]
        print(f"  Output points (restored only) : {len(out_x):,}")

    out_las.x = out_x
    out_las.y = out_y
    out_las.z = out_z

    # copy extra dims from original (proximity-filtered portion)
    _copy_extra_dims(orig_las, out_las, full_indices)

    del orig_las, ref_las
    gc.collect()

    out_las.write(output_path)
    print(f"  Saved → {output_path}\n")


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def batch_restore(args: argparse.Namespace) -> None:
    os.makedirs(args.output_folder, exist_ok=True)

    ref_files = sorted([
        f for f in os.listdir(args.reference_folder)
        if f.lower().endswith((".las", ".laz"))
    ])
    orig_files = sorted([
        f for f in os.listdir(args.original_folder)
        if f.lower().endswith((".las", ".laz"))
    ])

    if not ref_files:
        print("[ERROR] No LAS/LAZ files found in reference folder.")
        return
    if not orig_files:
        print("[ERROR] No LAS/LAZ files found in original folder.")
        return

    print(f"Reference files : {len(ref_files)}")
    print(f"Original files  : {len(orig_files)}")
    print(f"Radius          : {args.radius} m")
    print(f"Padding         : {args.padding} m")
    print(f"Include reference in output : {args.include_reference}\n")

    processed = skipped = 0

    for rname in ref_files:
        oname = _best_match(rname, orig_files)
        if oname is None:
            print(f"[WARN] No original match for '{rname}' — skipping.\n")
            skipped += 1
            continue

        orig_path = os.path.join(args.original_folder, oname)
        ref_path  = os.path.join(args.reference_folder, rname)
        out_path  = os.path.join(
            args.output_folder,
            os.path.splitext(rname)[0] + "_restored.las"
        )

        print(f"[{processed + 1}/{len(ref_files)}]")
        restore_density(orig_path, ref_path, out_path,
                        args.radius, args.padding, args.include_reference)
        processed += 1

    print(f"Done. Processed {processed} file(s), skipped {skipped}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore original point density from a subsampled LAS cloud.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--original_folder", required=True,
                        help="Folder containing original full-density LAS/LAZ files.")
    parser.add_argument("--reference_folder", required=True,
                        help="Folder containing subsampled (reference) LAS/LAZ files.")
    parser.add_argument("--output_folder", required=True,
                        help="Folder where output LAS files are written.")
    parser.add_argument("--radius", type=float, default=0.08,
                        help="Proximity radius in metres: original points within this distance "
                             "of any reference point are kept (default: 0.08).")
    parser.add_argument("--padding", type=float, default=0.5,
                        help="Bounding box padding in metres for initial crop of original cloud "
                             "(default: 0.5).")
    parser.add_argument("--include_reference", action="store_true",
                        help="If set, merge the reference (subsampled) points into the output "
                             "alongside the restored original points.")

    args = parser.parse_args()
    batch_restore(args)


if __name__ == "__main__":
    main()
