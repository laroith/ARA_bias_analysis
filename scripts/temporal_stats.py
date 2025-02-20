"""
Module: temporal_stats.py

This module provides functions for temporal aggregation and grouping of climate data using xarray.
It is designed to be modular and scalable for:
    - Converting hourly data to daily, monthly, seasonal, and annual aggregations.
    - Handling missing values (e.g., converting -999 to NaN).
    - Grouping by custom categories such as weather types (stub provided).

Future extensions (e.g., bias metrics, distribution analysis, SAL, fraction skill score)
should be implemented in separate modules.
"""

import xarray as xr
import numpy as np

def fix_missing_values(da, missing_value=-999):
    """
    Replace missing values in the DataArray with NaN.
    
    Args:
        da (xarray.DataArray): Input data array.
        missing_value (numeric, optional): Value indicating missing data. Default is -999.
    
    Returns:
        xarray.DataArray: DataArray with missing values replaced by NaN.
    """
    return da.where(da != missing_value, np.nan)

def aggregate_to_daily(ds, var_name, method="sum", missing_value=None, compute_ens_mean=False):
    """
    Aggregate (presumably) hourly data to daily resolution and return a Dataset.
    
    Returns
    -------
    xarray.Dataset
        A new dataset containing the daily-aggregated var_name and
        any other non-time-dependent variables (e.g. altitude).
        The daily time dimension replaces the original hourly time in var_name.
    """
    if var_name not in ds:
        raise ValueError(f"Variable '{var_name}' not found in dataset.")

    da = ds[var_name]
    # Optionally handle missing
    if missing_value is not None:
        da = fix_missing_values(da, missing_value)

    if compute_ens_mean and "member" in da.dims:
        da = da.mean(dim="member")

    # Perform daily aggregation
    if method == "sum":
        da_daily = da.resample(time="1D").sum()
    elif method == "mean":
        da_daily = da.resample(time="1D").mean()
    else:
        raise ValueError("Unsupported aggregation method. Choose 'sum' or 'mean'.")

    # --- (1) Build a new Dataset with the daily-aggregated variable
    ds_out = da_daily.to_dataset(name=var_name)

    # --- (2) Reattach any other variables that do NOT depend on time.
    # For example, altitude might be [lat_coord, lon_coord] with no time dimension.
    for var in ds.data_vars:
        # if var == var_name, we already included it in ds_out
        if var == var_name:
            continue

        # Only attach if it doesn't have the 'time' dimension
        dims_var = ds[var].dims
        if "time" not in dims_var:
            ds_out[var] = ds[var]

    return ds_out



def aggregate_by_month(ds, var_name, method="sum", missing_value=None, compute_ens_mean=False):
    """
    Aggregate daily data to monthly resolution.
    
    Args:
        ds (xarray.Dataset): Input dataset with a time coordinate.
        var_name (str): Variable name to aggregate.
        method (str, optional): Aggregation method: "sum" or "mean". Default is "sum".
        missing_value (numeric, optional): If provided, fix missing values before aggregation.
        
    Returns:
        xarray.DataArray: Monthly aggregated data.
    """
    if var_name not in ds:
        raise ValueError(f"Variable '{var_name}' not found in dataset.")

    da = ds[var_name]
    if missing_value is not None:
        da = fix_missing_values(da, missing_value)
    
    if compute_ens_mean and "member" in da.dims:
        da = da.mean(dim="member")

    if method == "sum":
        da_monthly = da.resample(time="1M").sum()
    elif method == "mean":
        da_monthly = da.resample(time="1M").mean()
    else:
        raise ValueError("Unsupported aggregation method. Choose 'sum' or 'mean'.")
    
    ds_out = da_monthly.to_dataset(name=var_name)

    for var in ds.data_vars:
        if var == var_name:
            continue

        dims_var = ds[var].dims
        if "time" not in dims_var:
            ds_out[var] = ds[var]

    return ds_out


def aggregate_by_season(ds, var_name, method="sum", missing_value=None, compute_ens_mean=False):
    """
    Aggregate daily data to seasonal resolution.
    
    Uses the xarray 'time.season' attribute for grouping.
    
    Args:
        ds (xarray.Dataset): Input dataset with a time coordinate.
        var_name (str): Variable name to aggregate.
        method (str, optional): Aggregation method: "sum" or "mean". Default is "sum".
        missing_value (numeric, optional): If provided, fix missing values before aggregation.
        
    Returns:
        xarray.DataArray: Seasonal aggregated data.
    """
    if var_name not in ds:
        raise ValueError(f"Variable '{var_name}' not found in dataset.")

    da = ds[var_name]
    if missing_value is not None:
        da = fix_missing_values(da, missing_value)
    
    if compute_ens_mean and "member" in da.dims:
        da = da.mean(dim="member")

    if method == "sum":
        # Sum across all times belonging to each season
        da_season = da.groupby("time.season").sum(dim="time")
    elif method == "mean":
        da_season = da.groupby("time.season").mean(dim="time")
    else:
        raise ValueError("Unsupported aggregation method. Choose 'sum' or 'mean'.")
    
    ds_out = da_season.to_dataset(name=var_name)

    for var in ds.data_vars:
        if var == var_name:
            continue

        dims_var = ds[var].dims
        if "time" not in dims_var:
            ds_out[var] = ds[var]

    return ds_out
 

def aggregate_by_year(ds, var_name, method="sum", missing_value=None, compute_ens_mean=False):
    """
    Aggregate daily data to annual resolution.
    
    Args:
        ds (xarray.Dataset): Input dataset with a time coordinate.
        var_name (str): Variable name to aggregate.
        method (str, optional): Aggregation method: "sum" or "mean". Default is "sum".
        missing_value (numeric, optional): If provided, fix missing values before aggregation.
        
    Returns:
        xarray.DataArray: Annual aggregated data.
    """
    da = ds[var_name]
    if missing_value is not None:
        da = fix_missing_values(da, missing_value)
    
    if compute_ens_mean and "member" in da.dims:
        da = da.mean(dim="member")

    if method == "sum":
        da_annual = da.resample(time="1Y").sum()
    elif method == "mean":
        da_annual = da.resample(time="1Y").mean()
    else:
        raise ValueError("Unsupported aggregation method. Choose 'sum' or 'mean'.")
    
    ds_out = da_annual.to_dataset(name=var_name)

    for var in ds.data_vars:
        if var == var_name:
            continue

        dims_var = ds[var].dims
        if "time" not in dims_var:
            ds_out[var] = ds[var]

    return ds_out


def group_by_weather_type(ds, var_name, weather_type_array, compute_ens_mean=False):
    """
    Group data by weather type.
    
    This function is a placeholder for grouping data based on weather type classifications.
    It assumes that a corresponding weather type array (or DataArray) is provided, aligned
    with the time coordinate of the dataset.
    
    Args:
        ds (xarray.Dataset): Input dataset with a time coordinate.
        var_name (str): Variable name to group.
        weather_type_array (xarray.DataArray or np.array): Array of weather type classifications.
        
    Returns:
        xarray.DataArray: Data grouped by weather type (currently returns the mean for each group).
    """
    da = ds[var_name]

    if compute_ens_mean and "member" in da.dims:
        da = da.mean(dim="member")

    # Example grouping – this should be refined based on how weather types are defined
    grouped = da.groupby(weather_type_array).mean()
    return grouped

# Example test block (only runs if executed directly)
if __name__ == "__main__":
    # For testing, load a small test dataset.
    import xarray as xr
    test_path = "../data/testdata_prec_201611_short.nc"
    ds = xr.open_dataset(test_path)
    
    # Example: Aggregate precipitation to daily, monthly, seasonal, and annual values,
    # handling missing values (assuming missing values are indicated by -999).
    daily = aggregate_to_daily(ds, "precipitation", method="sum", missing_value=-999)
    monthly = aggregate_by_month(ds, "precipitation", method="sum", missing_value=-999)
    seasonal = aggregate_by_season(ds, "precipitation", method="sum", missing_value=-999)
    annual = aggregate_by_year(ds, "precipitation", method="sum", missing_value=-999)
    
    print("Daily Aggregation:\n", daily)
    print("Monthly Aggregation:\n", monthly)
    print("Seasonal Aggregation:\n", seasonal)
    print("Annual Aggregation:\n", annual)
