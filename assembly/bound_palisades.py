import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from rasterio import features
from rasterio.transform import from_origin

# ==========================================================================
# Data Loading
# ==========================================================================

unbound = xr.open_dataset("data/raw/APPEARS_data/Palisades_Bounding_Box.nc")
perimeter = gpd.read_file("data/raw/shape_files/Palisades_perimeter.geojson")

# print(perimeter.head())
# print(perimeter.crs)

# ==========================================================================
# Create Mask
# ==========================================================================

# convert crs (coordinate reference system) of Palisades Perimeter
raster_crs = unbound["crs"].attrs["spatial_ref"]
print(raster_crs)
perimeter_proj = perimeter.to_crs(raster_crs)

# Define parameters for Rasterization
x = unbound["xdim"].values
y = unbound["ydim"].values

# width of pixel
dx = x[1]-x[0]
dy = y[0]-y[1]

# find top left
x0 = float(x.min() - dx/2)
y0 = float(y.max() + dy/2)

# Define 

transform = from_origin(x0, y0, dx, dy)
out_shape = (len(y), len(x))

# Apply Mask
geoms = [(geom, 1) for geom in perimeter_proj.geometry if geom is not None]
mask = features.rasterize(
    geoms,
    out_shape=out_shape,
    transform=transform,
    fill=0,
    dtype="uint8",
    all_touched=False
).astype(bool)

burn_mask = xr.DataArray(
    mask.astype(np.uint8),            
    dims=("ydim", "xdim"),
    coords={"ydim": unbound["ydim"], "xdim": unbound["xdim"]},
    name="burn_mask",
    attrs={
        "description": "1 = inside Palisades fire perimeter, 0 = outside",
        "source": "Rasterized from Palisades_perimeter.geojson onto MODIS Sinusoidal grid",
    },
)

print(burn_mask.shape, burn_mask.dtype)
print("Burn pixels:", int(burn_mask.sum()))

ndvi = unbound["_250m_16_days_NDVI"].where(unbound["_250m_16_days_NDVI"] != -3000)
# ndvi_burned = ndvi.where(burn_mask)

# import matplotlib.pyplot as plt

# burn_mask.plot()
# plt.show()

# ndvi.isel(time=0).plot()
# plt.show()

# ndvi_burned.isel(time=0).plot()
# plt.show()

qa = unbound["_250m_16_days_VI_Quality"].where(unbound["_250m_16_days_VI_Quality"] != 65535)

processed = xr.Dataset({
    "ndvi": ndvi,
    "qa": qa,
    "burn_mask": burn_mask
})

print("PROCESSED INFO")
print(processed)
print(processed["ndvi"].shape, processed["burn_mask"].shape)
print(processed["ndvi"].dtype, processed["burn_mask"].dtype)

processed.to_netcdf("data/processed/palisades_ndvi_unclean.nc")