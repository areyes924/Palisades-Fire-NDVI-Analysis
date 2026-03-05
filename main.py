import numpy as np
import pandas as pd
import xarray as xr

ctrl = xr.load_dataset("data/raw/APPEARS_data/Control.nc")
pali = xr.load_dataset("data/processed/palisades_ndvi_julian.nc")

print("==========================================================================")
print("Control Info:")
print("==========================================================================")

print(ctrl.info())

print("==========================================================================")
print("Palisades Info:")
print("==========================================================================")
print(pali.info())