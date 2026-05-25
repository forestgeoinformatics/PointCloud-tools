"""
remove_duplicates.py
---------------------
Remove duplicate points (identical XYZ coordinates) from LAS files while
preserving a preferred point when duplicates exist — determined by a
priority value in a chosen scalar field.

When two points share the same XYZ location, the point whose scalar field
equals --priority is kept.  If neither or both have the priority value,
the first occurrence is kept.

Usage
-----
  python remove_duplicates.py \\
      --input_folder  /path/to/las/files \\
      --output_folder /path/to/output    \\
      --field         cluster_id         \\
      --priority      1                  \\
      [--precision 6]
"""

import os
import argparse
import numpy as np
import laspy


def remove_duplicates(
    input_path: str,
    output_path: str,
    field_name: str,
    priority_value: int,
    precision: int = 6,
) -> None:

    las          = laspy.read(input_path)
    coords       = np.column_stack((las.x, las.y, las.z))
    field_values = np.asarray(las[field_name])
    n_in         = len(coords)

    # Build coordinate hash keys
    fmt = f"{{:.{precision}f}}_{{:.{precision}f}}_{{:.{precision}f}}"
    coord_keys = [fmt.format(x, y, z) for x, y, z in coords]

    coord_map: dict[str, int] = {}
    for i, key in enumerate(coord_keys):
        if key not in coord_map:
            coord_map[key] = i
        else:
            existing = coord_map[key]
            if (field_values[i] == priority_value and
                    field_values[existing] != priority_value):
                coord_map[key] = i

    kept_indices = np.array(sorted(coord_map.values()))
    n_removed    = n_in - len(kept_indices)

    out_las = las[kept_indices]
    out_las.write(output_path)

    print(f"  Input points   : {n_in:,}")
    print(f"  Removed        : {n_removed:,}")
    print(f"  Output points  : {len(kept_indices):,}")
    print(f"  Saved → {output_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove duplicate XYZ points from LAS files with priority-based selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input_folder", required=True,
                        help="Folder containing input LAS/LAZ files.")
    parser.add_argument("--output_folder", required=True,
                        help="Folder where deduplicated files are written.")
    parser.add_argument("--field", required=True,
                        help="Scalar field used to determine which duplicate to keep.")
    parser.add_argument("--priority", type=int, required=True,
                        help="Value of --field that is preferred when a duplicate exists.")
    parser.add_argument("--precision", type=int, default=6,
                        help="Decimal places used for coordinate matching (default: 6).")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    files = sorted([
        f for f in os.listdir(args.input_folder)
        if f.lower().endswith((".las", ".laz"))
    ])

    if not files:
        print("[ERROR] No LAS/LAZ files found.")
        return

    print(f"Files : {len(files)}\n")

    for idx, fname in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {fname}")
        remove_duplicates(
            input_path     = os.path.join(args.input_folder, fname),
            output_path    = os.path.join(args.output_folder, fname),
            field_name     = args.field,
            priority_value = args.priority,
            precision      = args.precision,
        )

    print("All files processed.")


if __name__ == "__main__":
    main()
