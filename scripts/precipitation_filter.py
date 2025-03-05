# scripts/wet_filter.py

import xarray as xr

def apply_wet_filter(ds, var_name, threshold):
    """
    Mask sub-threshold precipitation *without* converting the Dataset into a DataArray.
    
    - ds:        xarray.Dataset
    - var_name:  e.g. "precipitation" or "RR"
    - threshold: float, e.g. 0.1
    
    Returns another xarray.Dataset, preserving shape and coords,
    but setting all values < threshold to NaN in *each* data variable.
    """

    # If the specified variable doesn't exist, do nothing
    if var_name not in ds.data_vars:
        print(f"[Warning] {var_name} not found in dataset. Skipping wet filter.")
        return ds

    # Create a boolean mask from the user-specified variable
    # e.g. mask is True where precip >= threshold, else False
    mask = ds[var_name] >= threshold

    # Copy the original dataset structure
    ds_filtered = ds.copy(deep=False)

    # For each data variable in the dataset, apply the same mask
    for dv in ds.data_vars:
        ds_filtered[dv] = ds[dv].where(mask)

    return ds_filtered


def apply_precip_range(ds, var_name, min_val, max_val):
    """
    Mask precipitation outside [min_val, max_val), 
    preserving the dataset structure (no dimension dropping).
    
    - ds:       xarray.Dataset
    - var_name: e.g. "precipitation" or "RR"
    - min_val:  float
    - max_val:  float
    """
    if var_name not in ds:
        print(f"[Warning] '{var_name}' not found in dataset. Skipping precip range filter.")
        return ds
    
    # Build a boolean mask: True where the data is in [min_val, max_val)
    mask = (ds[var_name] >= min_val) & (ds[var_name] < max_val)
    
    # We do the same approach as the "wet_filter" that keeps a Dataset
    ds_filtered = ds.copy(deep=False)
    for dv in ds.data_vars:
        ds_filtered[dv] = ds[dv].where(mask)
    
    return ds_filtered
