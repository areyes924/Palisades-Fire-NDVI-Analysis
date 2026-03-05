import xarray as xr
import pandas as pd
import numpy as np

pali = xr.open_dataset("data/processed/palisades_ndvi_qa_burnmask.nc")
ctrl = xr.open_dataset("data/processed/control_ndvi_qa.nc")

print("Palisades Info:")
print(pali.info())
print("Control Info:")
print(ctrl.info())