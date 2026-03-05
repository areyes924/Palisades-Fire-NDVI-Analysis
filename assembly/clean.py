import numpy as np
import pandas as pd
import xarray as xr

# ==========================================================================
# Data Loading
# ==========================================================================

ctrl = xr.load_dataset("data/raw/APPEARS_data/Control.nc")
pali = xr.load_dataset("data/processed/palisades_ndvi_unclean.nc")

# ==========================================================================
# Standardize Formatting / Variable Names
# ==========================================================================

ctrl = ctrl.rename({
    "_250m_16_days_NDVI": "ndvi",
    "_250m_16_days_VI_Quality": "qa"
})

# ==========================================================================
# Time Coordinate Fix (cftime -> datetime64[ns])
# ==========================================================================

def ensure_datetime(ds: xr.Dataset) -> xr.Dataset:
    t = ds["time"].values

    # Already in pandas datetime?
    if np.issubdtype(ds["time"].dtype, np.datetime64):
        return ds

    # cftime objects (e.g., cftime.DatetimeJulian)
    # Convert each to ISO date string, then parse with pandas
    t_str = [f"{x.year:04d}-{x.month:02d}-{x.day:02d}" for x in t]
    t_dt = pd.to_datetime(t_str)

    return ds.assign_coords(time=("time", t_dt.values))

ctrl = ensure_datetime(ctrl)
pali = ensure_datetime(pali)

print(ctrl["time"].dtype, pali["time"].dtype)

# ==========================================================================
# Quality-Control Rules (MOD13Q1 VI_Quality bitfield)
# ==========================================================================

def mod13q1_good_pixel_mask(qa_u16: xr.DataArray) -> xr.DataArray:
    """
    Returns boolean mask (time, ydim, xdim) for pixels that are:
      - Land only (Land/Water bits 11-13 == 001)
      - MODLAND_QA is good or acceptable (bits 0-1 in {00, 01})
      - Not snow/ice and not shadow (bits 14 and 15 are 0)
    """
    # Land/water bits 11–13
    lw = (qa_u16 >> 11) & 0b111
    land = (lw == 1)

    # MODLAND_QA bits 0–1
    modland = qa_u16 & 0b11
    ok_modland = (modland == 0) | (modland == 1)

    # Snow & shadow bits 14 and 15
    snow = (((qa_u16 >> 14) & 0b1) == 1)
    shadow = (((qa_u16 >> 15) & 0b1) == 1)
    no_snow_shadow = (~snow) & (~shadow)

    return land & ok_modland & no_snow_shadow

# ==========================================================================
# Apply QC Masking
# ==========================================================================

# NDVI nodata is -3000
ctrl["ndvi"] = ctrl["ndvi"].where(ctrl["ndvi"] != -3000)
pali["ndvi"] = pali["ndvi"].where(pali["ndvi"] != -3000)

# QA nodata is 65535; remove it before casting
ctrl["qa"] = ctrl["qa"].where(ctrl["qa"] != 65535)
pali["qa"] = pali["qa"].where(pali["qa"] != 65535)

# Cast QA to uint16, fill NaNs to avoid invalid cast warnings
ctrl_qa_u16 = ctrl["qa"].fillna(0).astype("uint16")
pali_qa_u16 = pali["qa"].fillna(0).astype("uint16")

# Build good-pixel masks (time, ydim, xdim)
ctrl_good = mod13q1_good_pixel_mask(ctrl_qa_u16)
pali_good = mod13q1_good_pixel_mask(pali_qa_u16)

# Apply masks to NDVI (preserves grid; invalid pixels become NaN)
ctrl["ndvi"] = ctrl["ndvi"].where(ctrl_good)
pali["ndvi"] = pali["ndvi"].where(pali_good)

print("Palisades ds info:")
print(pali)
print("Control ds info:")
print(ctrl)

# ==========================================================================
# Export
# ==========================================================================

DO_EXPORT = True

if DO_EXPORT:
    # Export analysis-ready NDVI cubes (QA-masked) + region definition
    ctrl_out = ctrl[["ndvi", "qa", "crs"]] if "crs" in ctrl.data_vars else ctrl[["ndvi", "qa"]]
    ctrl_out.to_netcdf("data/processed/control_ndvi_qa.nc")

    pali_out = pali[["ndvi", "qa", "burn_mask"]]
    pali_out.to_netcdf("data/processed/palisades_ndvi_qa_burnmask.nc")