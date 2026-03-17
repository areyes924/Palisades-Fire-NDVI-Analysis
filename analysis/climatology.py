import xarray as xr
import numpy as np

# Data Loading
pali = xr.open_dataset("data/processed/palisades_ndvi_qa_burnmask.nc")
ctrl = xr.open_dataset("data/processed/control_ndvi_qa.nc")

# Grab ndvi means by calendar year
pali_yearly = pali["ndvi"].groupby("time.year").mean(dim="time", skipna=True)
ctrl_yearly = ctrl["ndvi"].groupby("time.year").mean(dim="time", skipna=True)

# Historical baseline: 2015-2024
pali_baseline = pali_yearly.sel(year=slice(2015, 2024))
ctrl_baseline = ctrl_yearly.sel(year=slice(2015, 2024))

# For each pixel, baseline mean and std
pali_mean = pali_baseline.mean(dim="year", skipna=True)
pali_std = pali_baseline.std(dim="year", skipna=True)

ctrl_mean = ctrl_baseline.mean(dim="year", skipna=True)
ctrl_std = ctrl_baseline.std(dim="year", skipna=True)

# Count number of years with data per pixel
pali_valid_years = pali_baseline.notnull().sum(dim="year")
ctrl_valid_years = ctrl_baseline.notnull().sum(dim="year")

# 2025 seasonal mean
pali_2025 = pali_yearly.sel(year=2025)
ctrl_2025 = ctrl_yearly.sel(year=2025)

# Anomaly relative to the baseline
pali_anom = pali_2025 - pali_mean
ctrl_anom = ctrl_2025 - ctrl_mean

# Standardized anomaly (z-score), with guard against zero variance (divide by zero)
eps = 1e-6
pali_z = pali_anom / xr.where(pali_std > eps, pali_std, np.nan)
ctrl_z = ctrl_anom / xr.where(ctrl_std > eps, ctrl_std, np.nan)

# Keep only pixels with enough years
min_years = 8
pali_mask = pali_valid_years >= min_years
ctrl_mask = ctrl_valid_years >= min_years

pali_z = pali_z.where(pali_mask)
ctrl_z = ctrl_z.where(ctrl_mask)

pali_anom = pali_anom.where(pali_mask)
ctrl_anom = ctrl_anom.where(ctrl_mask)

# Restrict Palisades analysis to burned pixels only
burn_mask = pali["burn_mask"] == 1
pali_burn_z = pali_z.where(burn_mask)
pali_burn_anom = pali_anom.where(burn_mask)

# Flatten to 1D and drop NaNs for summary stats
pali_burn_z_vals = pali_burn_z.values.flatten()
pali_burn_z_vals = pali_burn_z_vals[~np.isnan(pali_burn_z_vals)]

ctrl_z_vals = ctrl_z.values.flatten()
ctrl_z_vals = ctrl_z_vals[~np.isnan(ctrl_z_vals)]

pali_burn_anom_vals = pali_burn_anom.values.flatten()
pali_burn_anom_vals = pali_burn_anom_vals[~np.isnan(pali_burn_anom_vals)]

ctrl_anom_vals = ctrl_anom.values.flatten()
ctrl_anom_vals = ctrl_anom_vals[~np.isnan(ctrl_anom_vals)]


def summarize_z(arr):
    return {
        "n": arr.size,
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "frac_below_neg1": np.mean(arr < -1),
        "frac_below_neg2": np.mean(arr < -2),
    }


def summarize_raw(arr):
    return {
        "n": arr.size,
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
    }


pali_z_summary = summarize_z(pali_burn_z_vals)
ctrl_z_summary = summarize_z(ctrl_z_vals)

pali_anom_summary = summarize_raw(pali_burn_anom_vals)
ctrl_anom_summary = summarize_raw(ctrl_anom_vals)

print("Palisades burned z:", pali_z_summary)
print("Control z:", ctrl_z_summary)
print("Palisades burned raw anomaly:", pali_anom_summary)
print("Control raw anomaly:", ctrl_anom_summary)