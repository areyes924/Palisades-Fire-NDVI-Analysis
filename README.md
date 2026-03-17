# Palisades Fire NDVI Analysis (2025)



The January 2025 fires in the Palisades and Altadena had a major impact across the region. Personally, I had to evacuate campus, and take classes online for a week and a half.



When I revisited the Palisades, the vegetation appeared to have recovered more than I expected. I wanted to look more closely at the data and quantify how unusual post-fire vegetation conditions were using NDVI.



## What I’m doing



* Comparing Jan–Apr 2025 NDVI to 2015–2024 baseline
* Computing raw anomaly + z-scores per pixel
* Comparing burned region vs nearby control in Santa Monica Mountains



Once data comes out: Compare post-fire recovery against the baseline + 2025 anomaly



Data:

* MODIS NDVI (MOD13Q1, 250m, 16-day), 2015–2025, Jan–Apr window (https://appeears.earthdatacloud.nasa.gov/)
* Palisades burn mask + control region (https://geohub.lacity.org/datasets/lacounty::palisades-and-eaton-dissolved-fire-perimeters-2025/about?layer=1)



Libraries:



* numpy -> array operations, flattening, summary stats
* pandas -> datetime handling (cftime → standard time)
* xarray -> working with gridded NDVI (NetCDF, masking, grouping, etc)



## Pipeline



bound\_palisades.py

* Load fire perimeter + MODIS bounding box
* Reproject and rasterize perimeter to make burn mask aligned to grid
* Export masked Palisades dataset



clean.py

* Load raw NDVI + QA data
* Standardize variable names + fix time (cftime to datetime)
* Apply MODIS QA bitmask (remove ocean, clouds, snow, bad pixels)
* Keep grid structure by masking invalid values (NaNs)
* Export cleaned burned + control datasets



climatology.py

* Aggregate NDVI to yearly Jan–Apr means
* Build baseline (2015–2024 mean + std per pixel)
* Compute 2025 seasonal mean -> raw anomaly + z-score
* Filter pixels with low baseline support
* Apply burn mask and compare burned vs control distributions



## (Current) Results



Burned region: mean z ≈ -7.5

Control: ~0, normal spread



So yes, the burned area is way outside its historical range.



## Next up



* maps (baseline / anomaly / z)
* distribution plots (burned vs control)



Then (the exciting part), **look at how it recovers once the rest of March data comes in**



Documenting my progress and specifics of what I learned here, while project is still IP:
https://docs.google.com/document/d/1vp3YeO\_BotoLv4Rv\_sCDwZ32wd70y2MGMvlDy5dQ6Ek/edit?tab=t.0

