"""
scalar_field_delete.py
-----------------------
Retain only a specified set of scalar fields in LAS files, dropping all
others.  Useful for reducing file size before sharing or archiving.

Usage
-----
  python scalar_field_delete.py \\
      --input_folder  /path/to/las/files \\
      --output_folder /path/to/output    \\
      --keep_fields   Intensity Class    \\
      [--overwrite]
"""

import os
import argparse
import numpy as np
import laspy


def filter_fields(
    input_path: str,
    output_path: str,
    keep_fields: list[str],
) -> None:

    las = laspy.read(input_path)

    header  = laspy.LasHeader(point_format=3, version="1.3")
    out_las = laspy.LasData(header)
    out_las.x = las.x
    out_las.y = las.y
    out_las.z = las.z

    all_dims = list(las.point_format.dimension_names)
    kept = []

    for field in all_dims:
        if field not in keep_fields:
            continue

        arr   = np.asarray(getattr(las, field))
        dtype = arr.dtype

        if np.issubdtype(dtype, np.floating):
            las_type = np.float32
        elif dtype.itemsize == 1:
            las_type = np.uint8
        elif dtype.itemsize == 2:
            las_type = np.uint16
        else:
            las_type = np.int32

        out_las.add_extra_dim(laspy.ExtraBytesParams(name=field, type=las_type))
        setattr(out_las, field, arr)
        kept.append(field)

    missing = [f for f in keep_fields if f not in kept]
    if missing:
        print(f"  [WARN] Fields not found in file: {missing}")

    out_las.write(output_path)
    print(f"  Kept fields : {kept}")
    print(f"  Saved → {output_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retain only specified scalar fields in LAS files, drop all others.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input_folder", required=True,
                        help="Folder containing input LAS/LAZ files.")
    parser.add_argument("--output_folder", default=None,
                        help="Output folder. If omitted and --overwrite is set, "
                             "files are overwritten in place.")
    parser.add_argument("--keep_fields", nargs="+", required=True,
                        help="Scalar field names to retain, e.g. --keep_fields Intensity Class")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite input files (only used when --output_folder is not set).")
    args = parser.parse_args()

    if args.output_folder:
        out_dir = args.output_folder
        os.makedirs(out_dir, exist_ok=True)
    elif args.overwrite:
        out_dir = None   # signal: write back to input path
    else:
        parser.error("Provide --output_folder or --overwrite.")

    files = sorted([
        f for f in os.listdir(args.input_folder)
        if f.lower().endswith((".las", ".laz"))
    ])

    if not files:
        print("[ERROR] No LAS/LAZ files found.")
        return

    print(f"Files        : {len(files)}")
    print(f"Keep fields  : {args.keep_fields}\n")

    for idx, fname in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {fname}")
        in_path  = os.path.join(args.input_folder, fname)
        out_path = in_path if out_dir is None else os.path.join(out_dir, fname)
        filter_fields(in_path, out_path, args.keep_fields)

    print("All files processed.")


if __name__ == "__main__":
    main()
