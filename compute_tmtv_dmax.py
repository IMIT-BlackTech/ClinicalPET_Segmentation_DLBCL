#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：IMIT

"""
Compute metabolic tumor volume (TMTV), dissemination (Dmax), SUV-based metrics,
and export per-lesion and per-patient CSVs from 3D PET mask(.nii.gz) + SUV(.nii.gz).

Key outputs (per-patient):
- TMTV (cm^3)
- SUV41% TMTV (cm^3)
- TLG (TMTV * SUVmean), SUV41%TLG
- D_patient (max centroid-to-centroid distance across all lesions, mm)
- D_bulk   (max distance from the largest lesion's centroid to others, mm)
- lesion count
- global mean SUVpeak / SUV41peak across lesions
- largest-lesion MTV, size (mm in z/x/y), center (voxel center in z/x/y)

Key outputs (per-lesion):
- SUVmean / SUVmax
- 41% of lesion’s SUVmax → SUV41%mean / SUV41%max / SUV41% TMTV
- TLG (using lesion’s own SUVmean) / SUV41%TLG
- SUVpeak / SUV41peak (12 mm sphere, ellipsoid in voxel space)
- lesion center & bbox size (mm in z/x/y)

NOTE on axis & spacing:
- Arrays from nibabel / SimpleITK are read as shape [Z, Y, X] when converted to NumPy by SimpleITK,
  but in this script we keep nibabel arrays directly: nibabel returns array shaped [X, Y, Z] or [Z, Y, X]
  depending on affine. To avoid confusion, we **treat arrays as [Z, Y, X]** after loading by nibabel via asanyarray().
- We take spacing from SimpleITK (GetSpacing) which returns (sx, sy, sz) in **X, Y, Z (mm)**.
- Therefore, when mapping voxel counts to physical sizes:
    dim_z_size_mm = size_in_vox_z * sz
    dim_y_size_mm = size_in_vox_y * sy
    dim_x_size_mm = size_in_vox_x * sx

Dependencies:
    numpy, nibabel, SimpleITK, cc3d, scikit-image, scipy, tqdm (optional)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import nibabel as nib
import cc3d
import SimpleITK as sitk
from scipy.spatial import distance
from skimage.measure import label, regionprops
from skimage import util
from tqdm import tqdm
import warnings


# ----------------------------- Logging ----------------------------- #

def setup_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ----------------------------- Dataclasses ----------------------------- #

@dataclass
class CasePaths:
    case_dir: Path
    mask_path: Path
    suv_path: Path


@dataclass
class Spacing:
    sx: float
    sy: float
    sz: float  # 注意：与数组轴 [Z, Y, X] 的对应关系：Z<-sz, Y<-sy, X<-sx

    @property
    def voxel_volume_mm3(self) -> float:
        return float(self.sx * self.sy * self.sz)


# ----------------------------- Utilities ----------------------------- #

def find_cases(
    input_dir: Path,
    mask_filename: str = "PRE_251_cohort2_epoch2000_2_PET_best.nii.gz",
    suv_filename: str = "SUV.nii.gz",
) -> List[CasePaths]:
    cases: List[CasePaths] = []
    for entry in sorted(input_dir.iterdir()):
        if not entry.is_dir():
            continue
        mask = entry / mask_filename
        suv = entry / suv_filename
        if mask.exists() and suv.exists():
            cases.append(CasePaths(entry, mask, suv))
        else:
            logging.warning("Skip %s (missing %s or %s)", entry.name, mask.name, suv.name)
    return cases


def write_csv(rows: List[List], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    logging.info("Saved: %s", str(out_path))


# ----------------------------- Core math ----------------------------- #

def connected_components_3d(binary: np.ndarray, connectivity: int = 18) -> np.ndarray:
    """Return labeled connected components of a binary array."""
    return cc3d.connected_components(binary.astype(np.uint8), connectivity=connectivity)


def binarize(mask: np.ndarray) -> np.ndarray:
    """Return a binarized copy (avoid in-place modification)."""
    out = (mask > 0).astype(np.uint8)
    return out


def compute_voxel_count(mask_bin: np.ndarray) -> int:
    """Count positive voxels in a binary mask."""
    return int(mask_bin.sum(dtype=np.int64))


def compute_tmtv_mm3(mask_bin: np.ndarray, spacing: Spacing) -> float:
    """TMTV in mm^3 (positive voxels * voxel-volume)."""
    return compute_voxel_count(mask_bin) * spacing.voxel_volume_mm3


def suv_sum_max(arr: np.ndarray) -> Tuple[float, float]:
    """Sum and max over an array (float)."""
    return float(np.nansum(arr)), float(np.nanmax(arr)) if arr.size else (0.0, 0.0)


def suv_mean_in_mask(suv: np.ndarray, mask_bin: np.ndarray) -> float:
    cnt = compute_voxel_count(mask_bin)
    if cnt == 0:
        return 0.0
    return float(np.nansum(suv * mask_bin) / cnt)


def threshold_41_percent(suv_voi: np.ndarray, suvmax: float) -> np.ndarray:
    """Return a binary mask of voxels >= 41% of suvmax (no in-place)."""
    if suvmax <= 0:
        return np.zeros_like(suv_voi, dtype=np.uint8)
    thr = 0.41 * suvmax
    return (suv_voi > thr).astype(np.uint8)


def compute_dmax_mm(mask_bin: np.ndarray, spacing: Spacing) -> Tuple[float, float]:
    """
    D_patient: max pairwise centroid distance across *all* lesions (mm)
    D_bulk   : max distance from the largest lesion centroid to all others (mm)
    """
    if mask_bin.max() == 0:
        return 0.0, 0.0

    labeled = label(util.img_as_ubyte(mask_bin) > 0, connectivity=mask_bin.ndim)
    props = regionprops(labeled)
    if len(props) == 0:
        return 0.0, 0.0

    # voxel centroids (z, y, x) -> physical mm using (sz, sy, sx)
    centroids_mm = []
    areas = []
    for p in props:
        z, y, x = p.centroid
        centroids_mm.append([z * spacing.sz, y * spacing.sy, x * spacing.sx])
        areas.append(p.area)

    # max pairwise distance among all centroids
    d_patient = 0.0
    for i in range(len(centroids_mm)):
        for j in range(i + 1, len(centroids_mm)):
            d = distance.euclidean(centroids_mm[i], centroids_mm[j])
            if d > d_patient:
                d_patient = d

    # D_bulk: from largest area centroid to others
    bulk_idx = int(np.argmax(np.asarray(areas)))
    bulk_centroid = centroids_mm[bulk_idx]
    d_bulk = 0.0
    for j in range(len(centroids_mm)):
        d = distance.euclidean(bulk_centroid, centroids_mm[j])
        if d > d_bulk:
            d_bulk = d

    return float(d_patient), float(d_bulk)


def ellipsoid_mask(radius_vox: Tuple[int, int, int]) -> np.ndarray:
    """
    Create a 3D ellipsoid mask (1 inside, 0 outside) with radii (rz, ry, rx) in voxels.
    """
    rz, ry, rx = radius_vox
    grid = np.zeros((2 * rz + 1, 2 * ry + 1, 2 * rx + 1), dtype=np.uint8)
    z = np.arange(-rz, rz + 1)
    y = np.arange(-ry, ry + 1)
    x = np.arange(-rx, rx + 1)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    inside = (zz ** 2) / (rz ** 2 + 1e-8) + (yy ** 2) / (ry ** 2 + 1e-8) + (xx ** 2) / (rx ** 2 + 1e-8) <= 1.0
    grid[inside] = 1
    return grid


def suv_peak_12mm(
    mask_bin: np.ndarray,
    suv_voi: np.ndarray,
    spacing: Spacing,
    sphere_diameter_mm: float = 12.0,
) -> float:
    """
    SUVpeak: mean SUV in a 12 mm sphere (ellipsoid in voxels) centered at the max SUV voxel inside the lesion.
    Returns np.nan if lesion too small or at border causing index issues.
    """
    positions = np.where(mask_bin == 1)
    if len(positions[0]) <= 3:
        return np.nan

    # Find the voxel index of suv_voi max inside mask
    try:
        flat_idx = np.argmax(suv_voi[positions])
    except ValueError:
        return np.nan

    zc, yc, xc = positions[0][flat_idx], positions[1][flat_idx], positions[2][flat_idx]

    # Radius in vox for (z,y,x): r_vox = (6mm / s?)
    rz = max(1, int(round((sphere_diameter_mm / 2.0) / spacing.sz)))
    ry = max(1, int(round((sphere_diameter_mm / 2.0) / spacing.sy)))
    rx = max(1, int(round((sphere_diameter_mm / 2.0) / spacing.sx)))

    z0, z1 = zc - rz, zc + rz
    y0, y1 = yc - ry, yc + ry
    x0, x1 = xc - rx, xc + rx

    # Border check
    Z, Y, X = suv_voi.shape
    if z0 < 0 or y0 < 0 or x0 < 0 or z1 >= Z or y1 >= Y or x1 >= X:
        # too close to border
        warnings.warn("SUVpeak neighborhood out of bounds; consider padding input.")
        return np.nan

    neighborhood = suv_voi[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    ell = ellipsoid_mask((rz, ry, rx))

    try:
        vals = neighborhood[ell.astype(bool)]
        if vals.size == 0:
            return np.nan
        return float(np.nanmean(vals))
    except Exception:
        return np.nan


# ----------------------------- Case processing ----------------------------- #

def load_arrays(mask_path: Path, suv_path: Path) -> Tuple[np.ndarray, np.ndarray, Spacing]:
    """
    Load mask (binarized) and SUV arrays as [Z, Y, X] and spacing (sx, sy, sz).
    Also checks orientation consistency (basic).
    """
    # Nibabel arrays; we will treat them as [Z, Y, X] after np.asanyarray()
    mask_img = nib.load(str(mask_path))
    suv_img = nib.load(str(suv_path))

    # Quick affine-based code check (best-effort)
    mask_orient = nib.aff2axcodes(mask_img.affine)
    suv_orient = nib.aff2axcodes(suv_img.affine)
    if mask_orient != suv_orient:
        logging.warning("Orientation mismatch %s vs %s for %s",
                        mask_orient, suv_orient, mask_path.parent.name)

    mask = np.asanyarray(mask_img.dataobj)
    suv = np.asanyarray(suv_img.dataobj)

    # Coerce to 3D if a trailing singleton exists
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if suv.ndim == 4 and suv.shape[-1] == 1:
        suv = suv[..., 0]

    # Convert via SimpleITK for spacing (X,Y,Z)
    suv_itk = sitk.ReadImage(str(suv_path))
    sx, sy, sz = map(float, suv_itk.GetSpacing())  # (X, Y, Z)
    spacing = Spacing(sx=sx, sy=sy, sz=sz)

    # Binarize mask
    mask_bin = binarize(mask)

    # Ensure shapes match
    if mask_bin.shape != suv.shape:
        raise ValueError(f"Shape mismatch: mask {mask_bin.shape} vs suv {suv.shape}")

    # Reorder assumption: treat arrays as [Z, Y, X].
    # Many PET NIfTI are actually saved as [X, Y, Z]; if so and affine differs,
    # a proper reorientation to a common canonical space is recommended.
    # Here we assume consistent voxel-wise indices for both volumes.

    return mask_bin, suv.astype(np.float32), spacing


def crop_to_bbox(arr: np.ndarray, mask_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """
    Crop array & mask to tight bbox of mask==1.
    Return cropped arrays and center (in voxel coords) as (zc, yc, xc).
    """
    idx = np.where(mask_bin > 0)
    if len(idx[0]) == 0:
        return arr, mask_bin, (np.nan, np.nan, np.nan)

    zmin, ymin, xmin = np.min(idx[0]), np.min(idx[1]), np.min(idx[2])
    zmax, ymax, xmax = np.max(idx[0]), np.max(idx[1]), np.max(idx[2])

    arr_c = arr[zmin: zmax + 1, ymin: ymax + 1, xmin: xmax + 1]
    msk_c = mask_bin[zmin: zmax + 1, ymin: ymax + 1, xmin: xmax + 1]

    zc = zmin + (zmax - zmin) / 2.0
    yc = ymin + (ymax - ymin) / 2.0
    xc = xmin + (xmax - xmin) / 2.0

    return arr_c, msk_c, (zc, yc, xc)


# ----------------------------- Pipeline ----------------------------- #

def process_case(case: CasePaths) -> Tuple[List[List], List]:
    """
    Returns:
      lesion_rows (per-lesion), patient_row (per-patient)
    """
    mask_bin, suv, spacing = load_arrays(case.mask_path, case.suv_path)
    Z, Y, X = mask_bin.shape

    # Patient-level accumulators
    tmtv_mm3_total = 0.0
    suv41_tmtv_mm3_total = 0.0
    tlg_total = 0.0
    suv41_tlg_total = 0.0
    suvpeak_list: List[float] = []
    suv41peak_list: List[float] = []
    vol_suv_max = 0.0
    vol_suv41_max = 0.0

    lesion_mtv_list: List[float] = []
    lesion_size_mm_list: List[Tuple[float, float, float]] = []
    lesion_center_vox_list: List[Tuple[float, float, float]] = []

    # Connected components
    conn = connected_components_3d(mask_bin, connectivity=18)
    n_lesions = int(conn.max())

    lesion_rows: List[List] = []
    case_id = case.case_dir.name

    for i in range(1, n_lesions + 1):
        roi_mask = (conn == i).astype(np.uint8)
        if roi_mask.sum() == 0:
            continue

        # Crop to speed up per-lesion metrics
        suv_c, roi_c, center_vox = crop_to_bbox(suv, roi_mask)
        zc, yc, xc = center_vox

        # lesion size in mm (z,y,x)
        lz, ly, lx = roi_c.shape
        size_mm = (lz * spacing.sz, ly * spacing.sy, lx * spacing.sx)

        # MTV (mm^3)
        mtv_mm3 = compute_tmtv_mm3(roi_mask, spacing)
        lesion_mtv_list.append(mtv_mm3)
        lesion_size_mm_list.append(size_mm)
        lesion_center_vox_list.append((zc, yc, xc))

        # SUV-based metrics within lesion
        vox_count = compute_voxel_count(roi_mask)
        if vox_count == 0:
            continue
        roi_suv_sum, roi_suv_max = suv_sum_max(suv * roi_mask)
        vol_suv_max = max(vol_suv_max, roi_suv_max)

        roi_suv_mean = roi_suv_sum / vox_count if vox_count > 0 else 0.0

        # 41% threshold mask (per-lesion)
        roi41_mask = threshold_41_percent(suv * roi_mask, roi_suv_max)
        roi41_cnt = compute_voxel_count(roi41_mask)
        roi41_mtv_mm3 = roi41_cnt * spacing.voxel_volume_mm3
        suv41_tmtv_mm3_total += roi41_mtv_mm3

        roi41_sum, roi41_max = suv_sum_max(suv * roi41_mask)
        vol_suv41_max = max(vol_suv41_max, roi41_max)
        roi41_mean = (roi41_sum / roi41_cnt) if roi41_cnt > 0 else 0.0

        # TLG variants
        tlg_total += (mtv_mm3 * roi_suv_mean)
        suv41_tlg_total += (roi41_mtv_mm3 * roi41_mean)

        # SUVpeak variants
        suvpeak_val = suv_peak_12mm(roi_mask, suv * roi_mask, spacing)
        suv41peak_val = suv_peak_12mm(roi41_mask, suv * roi_mask, spacing)
        if not np.isnan(suvpeak_val):
            suvpeak_list.append(float(suvpeak_val))
        if not np.isnan(suv41peak_val):
            suv41peak_list.append(float(suv41peak_val))

        # Per-lesion row (mm^3 for volumes)
        lesion_rows.append([
            f"{case_id}_{i}",              # ID
            spacing.sx, spacing.sy, spacing.sz,  # spacing X,Y,Z
            roi_suv_mean, roi_suv_max,
            roi41_mean, roi41_max,
            roi41_mtv_mm3,                 # SUV41%_TMTV (mm^3)
            mtv_mm3,                       # pred_TMTV (mm^3)
            roi41_mtv_mm3 * roi41_mean,    # SUV41%TLG
            mtv_mm3 * roi_suv_mean,        # pred_TLG
            suvpeak_val,                   # suv_peak
            suv41peak_val,                 # suv41_peak
            (zc, yc, xc),                  # lesion center (voxel)
            size_mm                        # lesion size (mm, z,y,x)
        ])

        # accumulate TMTV
        tmtv_mm3_total += mtv_mm3

    # Dmax family
    d_patient_mm, d_bulk_mm = compute_dmax_mm(mask_bin, spacing)

    # Whole-body "relative" TMTV ratio (sum MTV / whole SUV>mean volume), keep your logic
    suv_mean_global = float(np.nanmean(suv)) if suv.size else 0.0
    suv_global_mask = (suv > suv_mean_global).astype(np.uint8)
    whole_vol_mm3 = compute_tmtv_mm3(suv_global_mask, spacing)
    improve_tmtv_ratio = (tmtv_mm3_total / whole_vol_mm3) if whole_vol_mm3 > 0 else 0.0

    # Largest lesion info
    if len(lesion_mtv_list) > 0:
        idx_max = int(np.argmax(np.asarray(lesion_mtv_list)))
        max_lesion_mtv = lesion_mtv_list[idx_max]
        max_lesion_size = lesion_size_mm_list[idx_max]
        max_lesion_center = lesion_center_vox_list[idx_max]
    else:
        max_lesion_mtv = np.nan
        max_lesion_size = (np.nan, np.nan, np.nan)
        max_lesion_center = (np.nan, np.nan, np.nan)

    # patient-level row; convert mm^3 → cm^3 for TMTV columns
    mm3_to_cm3 = 1.0 / 1000.0
    suvpeak_mean = float(np.nanmean(suvpeak_list)) if len(suvpeak_list) else np.nan
    suv41peak_mean = float(np.nanmean(suv41peak_list)) if len(suv41peak_list) else np.nan

    patient_row = [
        case_id,
        spacing.sx, spacing.sy, spacing.sz,
        vol_suv41_max, vol_suv_max,
        suv41_tmtv_mm3_total * mm3_to_cm3,   # SUV41%_TMTV (cm^3)
        tmtv_mm3_total * mm3_to_cm3,         # pred_TMTV (cm^3)
        suv41_tlg_total,                     # SUV41%TLG
        tlg_total,                           # pred_TLG
        d_patient_mm, d_bulk_mm,
        int(n_lesions),
        suvpeak_mean, suv41peak_mean,
        improve_tmtv_ratio,
        max_lesion_mtv * mm3_to_cm3,         # cm^3
        max_lesion_size,                     # (mm_z, mm_y, mm_x)
        max_lesion_center                    # (z,y,x in voxel)
    ]

    return lesion_rows, patient_row


# ----------------------------- Main ----------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute TMTV / Dmax / SUV metrics from mask+SUV NIfTI files."
    )
    parser.add_argument("--input_dir", type=Path, required=False, default=Path("/NII_dir"),
                        help="Root dir: each case is a subfolder with mask & SUV files.")
    parser.add_argument("--output_dir", type=Path, required=False, default=Path("/output_dir"),
                        help="Output directory for CSVs.")
    parser.add_argument("--mask_name", type=str, default="PRE.nii.gz",
                        help="Mask NIfTI filename within each case folder.")
    parser.add_argument("--suv_name", type=str, default="SUV.nii.gz",
                        help="SUV NIfTI filename within each case folder.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level.")
    args = parser.parse_args()

    setup_logging(args.log_level)

    cases = find_cases(args.input_dir, args.mask_name, args.suv_name)
    logging.info("Total cases found: %d", len(cases))

    lesion_header = [
        "ID", "X", "Y", "Z",
        "suv_mean", "suv_max",
        "suv41%_mean", "suv41%_max",
        "SUV41%_TMTV(mm^3)",
        "pred_TMTV(mm^3)",
        "SUV41%TLG",
        "pred_TLG",
        "suv_peak",
        "suv41_peak",
        "lesion_center_voxel(z,y,x)",
        "lesion_size_mm(z,y,x)"
    ]
    patient_header = [
        "ID", "X", "Y", "Z",
        "suv41%_max", "suv_max",
        "SUV41%_TMTV(cm^3)",
        "pred_TMTV(cm^3)",
        "SUV41%TLG",
        "pred_TLG",
        "D_patient(mm)", "D_bulk(mm)",
        "num_lesions",
        "suvpeak_mean",
        "suv41peak_mean",
        "improve_tmtv_ratio",
        "max_lesion_mtv(cm^3)",
        "max_lesion_size_mm(z,y,x)",
        "max_lesion_center_voxel(z,y,x)"
    ]

    lesion_rows_all: List[List] = [lesion_header]
    patient_rows_all: List[List] = [patient_header]

    for case in tqdm(cases, desc="Processing cases"):
        try:
            lesion_rows, patient_row = process_case(case)
            lesion_rows_all.extend(lesion_rows)
            patient_rows_all.append(patient_row)
        except Exception as e:
            logging.exception("Failed on case %s: %s", case.case_dir.name, str(e))

    # Filenames (keep your originals but more generic)
    out_lesion = args.output_dir / "data_xyz_mtv.csv"
    out_patient = args.output_dir / "data_xyz_tmtv.csv"

    write_csv(lesion_rows_all, out_lesion)
    write_csv(patient_rows_all, out_patient)

    logging.info("Patients processed successfully: %d", len(patient_rows_all) - 1)
    logging.info("Done.")


if __name__ == "__main__":
    main()
