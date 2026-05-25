[README.md](https://github.com/user-attachments/files/28233440/README.md)
# PointCloud-tools#

A collection of Python command-line tools for **batch processing LAS/LAZ point cloud files**.

Each task here can be performed manually in CloudCompare or similar software, but these scripts automate the process across entire folders of files — essential when working with large datasets of individual tree scans, forest plots, or flight strips.

---

## Tools

| Script | Purpose |
|---|---|
| [`combine_las.py`](#combine_laspy) | Merge multiple LAS files into one; assign a unique `cluster_id` per source file |
| [`remove_duplicates.py`](#remove_duplicatespy) | Remove duplicate XYZ points with priority-based selection |
| [`subsample.py`](#samplepy) | Height-based weighted voxel subsampling (finer voxels at canopy) |
| [`scalar_field_delete.py`](#scalar_field_deletepy) | Strip unwanted scalar fields to reduce file size |
| [`transfer_scalar.py`](#transfer_scalarpy) | Copy one or more scalar fields between clouds via nearest-neighbour matching |
| [`restore_density.py`](#restore_densitypy) | Restore original point density from a subsampled cloud |
| [`compute_metrics.py`](#compute_metricspy) | Compute full set of 3D geometric + radiometric features |
| [`compute_metrics_selected.py`](#compute_metrics_selectedpy) | Faster variant — computes only the features you specify via `--features` |
| [`compute_metrics_jakteristics.py`](#compute_metrics_jakteristicspy) | Fastest variant — uses C++ backend (jakteristics) |
| [`count_points.py`](#count_pointspy) | Count points and compute average NN spacing per file |

---

## Installation

```bash
pip install laspy[lazrs] numpy scipy tqdm pandas scikit-learn
```

Optional dependencies:
```bash
pip install open3d        # restore_density.py
pip install jakteristics  # compute_metrics_jakteristics.py
```

---

## Tool Reference

### `combine_las.py`
Combine all LAS/LAZ files in a folder into a single LAS file. A `cluster_id` field is added so every point can be traced back to its source file.

```bash
python tools/combine_las.py \
    --input_folder /path/to/las/files \
    --output_las   /path/to/combined.las
```

---

### `remove_duplicates.py`
Remove points with identical XYZ coordinates. When a duplicate exists, the point whose scalar field matches `--priority` is kept.

```bash
python tools/remove_duplicates.py \
    --input_folder  /path/to/las/files \
    --output_folder /path/to/output    \
    --field         cluster_id         \
    --priority      1                  \
    --precision     6
```

---

### `subsample.py`
Height-based weighted voxel subsampling. Points near the canopy are subsampled with a finer voxel than points near the ground, preserving structural detail where it matters.

```bash
python tools/subsample.py \
    --input_folder    /path/to/las/files \
    --output_folder   /path/to/output    \
    --base_voxel_size 0.03               \
    --min_voxel_size  0.003              \
    --weighting       linear             \
    --n_chunks        4
```

`--n_chunks` controls the XY grid used for chunked processing. Increase it (e.g. `8`) if you run out of RAM on large files.

---

### `scalar_field_delete.py`
Keep only the specified scalar fields and discard the rest. Useful before sharing or archiving files.

```bash
python tools/scalar_field_delete.py \
    --input_folder  /path/to/las/files \
    --output_folder /path/to/output    \
    --keep_fields   Intensity Class
```

Add `--overwrite` instead of `--output_folder` to modify files in place.

---

### `transfer_scalar.py`
For each file in `--target_folder`, find the best-matching file in `--source_folder` by name, then copy the specified scalar fields via nearest-neighbour (cKDTree) matching.

Typical uses:
- Copy `Intensity` from an original full-density scan onto a subsampled cloud.
- Transfer predicted labels (`Class`, `pred`) from a subsampled cloud back to the original.

```bash
python tools/transfer_scalar.py \
    --source_folder /path/to/source \
    --target_folder /path/to/target \
    --output_folder /path/to/output \
    --fields        Intensity Class BO \
    --radius        0.08
```

`--fill` sets the value assigned to target points with no source neighbour within `--radius` (default: `0`).

---

### `restore_density.py`
Restore the original point density of a subsampled cloud by pulling back points from the full-density original scan.

The original cloud is first cropped to the bounding box of the reference (subsampled) cloud (+ `--padding`), then only points within `--radius` of any reference point are retained. This bbox pre-filter makes it viable for very large original scans without loading the entire file into RAM.

```bash
python tools/restore_density.py \
    --original_folder  /path/to/original_dense \
    --reference_folder /path/to/subsampled      \
    --output_folder    /path/to/output          \
    --radius           0.08                     \
    --padding          0.5
```

Add `--include_reference` to merge the subsampled points into the output alongside the restored original points.

---

### `compute_metrics.py`
Compute a comprehensive set of 3D geometric and radiometric features for every point. Results are appended as new scalar fields, one set per radius (e.g. `sphericity_r0.05`).

Features: `linearity`, `planarity`, `scattering`, `omnivariance`, `anisotropy`, `eigentropy`, `eigensum`, `curvature`, `sphericity`, `verticality`, `3d_eigen_1/2/3`, `Density`, `ZRange`, `ZStd`, `number`, `roughness`, `min_distance`, `max_distance`, `intensity_mean`, `intensity_std`, `intensity_range`.

```bash
python tools/compute_metrics.py \
    --input_folder /path/to/las/files \
    --radii        0.05 0.10 0.15     \
    --output_folder /path/to/output
```

---

### `compute_metrics_selected.py`
Faster variant of `compute_metrics.py` that computes only the features you specify via `--features`. Useful when you need a subset of features for a specific classifier, or when processing time is a constraint.

```bash
# Compute three features at two radii
python tools/compute_metrics_selected.py \
    --input_folder /path/to/las/files \
    --radii        0.05 0.10          \
    --features     sphericity Density roughness

# Compute eigenvalue features + intensity statistics
python tools/compute_metrics_selected.py \
    --input_folder /path/to/las/files \
    --radii        0.05               \
    --features     linearity planarity sphericity verticality intensity_mean intensity_std
```

To see all available feature names:
```bash
python tools/compute_metrics_selected.py --list_features
```

**Available features:**

| Group | Features |
|---|---|
| Eigenvalue-based | `linearity`, `planarity`, `scattering`, `omnivariance`, `anisotropy`, `eigentropy`, `eigensum`, `curvature`, `sphericity`, `verticality`, `3d_eigen_1`, `3d_eigen_2`, `3d_eigen_3` |
| Neighbourhood | `Density`, `ZRange`, `ZStd`, `number`, `roughness`, `min_distance`, `max_distance` |
| Radiometric | `intensity_mean`, `intensity_std`, `intensity_range` *(skipped if no Intensity field)* |

Eigenvalue-based features share a single covariance decomposition per point, so requesting more of them costs very little extra time compared to requesting just one.

---

### `compute_metrics_jakteristics.py`
Fastest variant using the [jakteristics](https://github.com/jakarto3d/jakteristics) C++ backend. Computes the same four features as `compute_metrics_selected.py` but typically 10–50× faster on large clouds. Requires `pip install jakteristics`.

```bash
python tools/compute_metrics_jakteristics.py \
    --input_folder /path/to/las/files \
    --radii        0.05 0.10 0.15
```

---

### `count_points.py`
Print a summary table of point count and average nearest-neighbour spacing for each file. Optionally save to CSV.

```bash
python tools/count_points.py \
    --input_folder /path/to/las/files \
    --output_csv   /path/to/report.csv \
    --k_neighbors  32
```

---

## File Matching in Batch Tools

Scripts that operate on pairs of files (`transfer_scalar.py`, `restore_density.py`) match files by the **longest common prefix** of their base names (case-insensitive). For example:

| Source | Target | Matched? |
|---|---|---|
| `Pine_4_original.las` | `Pine_4_hwl.las` | ✅ (`Pine_4`) |
| `Kanasar_Tree5.las` | `Kanasar_Tree5_subsampled.las` | ✅ |
| `SiteA_plot1.las` | `SiteB_plot2.las` | ⚠️ weak match — review output |

A warning is printed when no confident match is found.

---

## Notes

- All scripts accept `.las` and `.laz` files.
- Output files are never written to the input folder by default — always specify `--output_folder` or check the default suffix (`_metrics`, `_hwl`, `_transferred`, `_restored`).
- Scripts preserve all existing scalar fields from the input unless explicitly filtering them (e.g. `scalar_field_delete.py`).
