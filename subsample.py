"""
subsample.py
------------
Height-based weighted voxel subsampling for LAS point clouds.

Points at higher elevations (canopy, upper branches) are subsampled more
finely (smaller voxel) than lower points (stem, ground), preserving fine
structural detail where it matters most.

The file is divided into an n_chunks × n_chunks XY grid and each tile is
processed independently to keep RAM usage manageable on large scans.
Height normalisation uses the global Z min/max so voxel sizes are
consistent across tiles.

Weighting functions
-------------------
  linear      : voxel_size decreases linearly with height
  exponential : aggressive fine detail at the top
  sigmoid     : smooth transition concentrated around mid-height

Usage
-----
  python subsample.py \\
      --input_folder    /path/to/las/files \\
      --output_folder   /path/to/output    \\
      --base_voxel_size 0.03               \\
      --min_voxel_size  0.003              \\
      --weighting       linear             \\
      --steepness       3.0                \\
      --n_chunks        4
"""

import os
import argparse
import gc
from collections import defaultdict
import numpy as np
import laspy
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Adaptive voxel subsampler
# ---------------------------------------------------------------------------

def adaptive_voxel_subsample(points: np.ndarray, voxel_sizes: np.ndarray) -> np.ndarray:
    """Return indices of points selected by per-point adaptive voxel size."""
    base_voxel = voxel_sizes.min()
    voxel_coords = np.floor(points / base_voxel).astype(np.int64)

    voxel_dict: dict = defaultdict(list)
    for idx, vc in enumerate(voxel_coords):
        voxel_dict[tuple(vc)].append(idx)

    kept      = []
    processed = set()

    for voxel_key, indices in voxel_dict.items():
        if voxel_key in processed:
            continue

        local_vs = float(np.median(voxel_sizes[indices]))
        scale    = max(1, int(round(local_vs / base_voxel)))
        av_key   = tuple(np.array(voxel_key) // scale)

        region_indices = []
        to_mark        = []
        for dx in range(scale):
            for dy in range(scale):
                for dz in range(scale):
                    ck = (av_key[0] * scale + dx,
                          av_key[1] * scale + dy,
                          av_key[2] * scale + dz)
                    if ck in voxel_dict:
                        region_indices.extend(voxel_dict[ck])
                        to_mark.append(ck)

        processed.update(to_mark)

        if region_indices:
            rp       = points[region_indices]
            centroid = rp.mean(axis=0)
            dist     = np.linalg.norm(rp - centroid, axis=1)
            kept.append(region_indices[int(np.argmin(dist))])

    return np.array(sorted(kept), dtype=np.int64)


# ---------------------------------------------------------------------------
# Per-file subsampler
# ---------------------------------------------------------------------------

STANDARD_DIMS = {
    'X', 'Y', 'Z', 'x', 'y', 'z',
    'intensity', 'return_number', 'number_of_returns',
    'scan_direction_flag', 'edge_of_flight_line', 'classification',
    'synthetic', 'key_point', 'withheld', 'scan_angle_rank',
    'user_data', 'point_source_id', 'gps_time', 'red', 'green', 'blue',
}


def subsample_file(
    las_file: str,
    output_file: str,
    base_voxel_size: float,
    min_voxel_size: float,
    weighting: str,
    steepness: float,
    n_chunks: int,
) -> None:

    las = laspy.read(las_file)
    x   = np.asarray(las.x)
    y   = np.asarray(las.y)
    z   = np.asarray(las.z)

    z_min, z_max = z.min(), z.max()
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    print(f"  Points    : {len(x):,}")
    print(f"  Z range   : {z_min:.2f} – {z_max:.2f} m")
    print(f"  Grid      : {n_chunks}×{n_chunks}")

    extra_dims = [f for f in las.point_format.dimension_names if f not in STANDARD_DIMS]

    x_edges = np.linspace(x_min, x_max, n_chunks + 1)
    y_edges = np.linspace(y_min, y_max, n_chunks + 1)

    all_x, all_y, all_z = [], [], []
    all_extra            = {name: [] for name in extra_dims}
    total_in = total_out = 0

    for i in tqdm(range(n_chunks), desc="  Chunks", leave=False):
        for j in range(n_chunks):
            x0, x1 = x_edges[i], x_edges[i + 1]
            y0, y1 = y_edges[j], y_edges[j + 1]

            last_x = (i == n_chunks - 1)
            last_y = (j == n_chunks - 1)
            mask   = (
                (x >= x0) & (x <= x1 if last_x else x < x1) &
                (y >= y0) & (y <= y1 if last_y else y < y1)
            )

            n_in = int(mask.sum())
            if n_in == 0:
                continue

            cx, cy, cz = x[mask], y[mask], z[mask]

            z_range  = z_max - z_min
            norm_h   = (cz - z_min) / (z_range if z_range > 0 else 1.0)

            if weighting == 'linear':
                weights = norm_h
            elif weighting == 'exponential':
                weights = np.exp(steepness * norm_h) - 1
                weights /= weights.max() if weights.max() > 0 else 1
            elif weighting == 'sigmoid':
                weights = 1.0 / (1.0 + np.exp(-steepness * (norm_h - 0.5)))
            else:
                raise ValueError(f"Unknown weighting function: {weighting}")

            voxel_sizes = base_voxel_size - weights * (base_voxel_size - min_voxel_size)
            points_tile = np.stack([cx, cy, cz], axis=1)
            kept_idx    = adaptive_voxel_subsample(points_tile, voxel_sizes)

            total_in  += n_in
            total_out += len(kept_idx)

            all_x.append(cx[kept_idx])
            all_y.append(cy[kept_idx])
            all_z.append(cz[kept_idx])

            orig_idx = np.where(mask)[0][kept_idx]
            for name in extra_dims:
                all_extra[name].append(np.asarray(getattr(las, name))[orig_idx])

            del cx, cy, cz, points_tile, voxel_sizes, weights, mask
            gc.collect()

    # write output
    out_las   = laspy.LasData(laspy.LasHeader(point_format=3, version="1.3"))
    out_las.x = np.concatenate(all_x)
    out_las.y = np.concatenate(all_y)
    out_las.z = np.concatenate(all_z)

    for name in extra_dims:
        dtype = np.asarray(getattr(las, name)).dtype
        out_las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
        setattr(out_las, name, np.concatenate(all_extra[name]))

    out_las.write(output_file)
    pct = 100.0 * total_out / total_in if total_in else 0
    print(f"  {total_in:,} → {total_out:,} points ({pct:.1f}%)")
    print(f"  Saved → {output_file}\n")


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Height-based weighted voxel subsampling for LAS point clouds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input_folder",    required=True,
                        help="Folder containing input LAS/LAZ files.")
    parser.add_argument("--output_folder",   default=None,
                        help="Output folder (default: input folder, files suffixed '_hwl').")
    parser.add_argument("--base_voxel_size", type=float, default=0.03,
                        help="Voxel size at the base (ground level) in metres (default: 0.03).")
    parser.add_argument("--min_voxel_size",  type=float, default=0.003,
                        help="Voxel size at the top (canopy) in metres (default: 0.003).")
    parser.add_argument("--weighting",       default="linear",
                        choices=["linear", "exponential", "sigmoid"],
                        help="Height-weighting function (default: linear).")
    parser.add_argument("--steepness",       type=float, default=3.0,
                        help="Steepness for exponential/sigmoid weighting (default: 3.0).")
    parser.add_argument("--n_chunks",        type=int, default=4,
                        help="XY grid divisions for chunked processing (default: 4). "
                             "Increase if RAM is limited.")
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

    print(f"Files            : {len(files)}")
    print(f"Base voxel size  : {args.base_voxel_size} m")
    print(f"Min voxel size   : {args.min_voxel_size} m")
    print(f"Weighting        : {args.weighting}")
    print(f"Chunks           : {args.n_chunks}×{args.n_chunks}\n")

    for idx, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        print(f"[{idx}/{len(files)}] {fname}")
        out_path = os.path.join(
            out_dir,
            fname.replace(".las", "_hwl.las").replace(".laz", "_hwl.las")
        )
        subsample_file(
            fpath, out_path,
            args.base_voxel_size, args.min_voxel_size,
            args.weighting, args.steepness, args.n_chunks,
        )

    print("All files processed.")


if __name__ == "__main__":
    main()
