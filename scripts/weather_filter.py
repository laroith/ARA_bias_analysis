# scripts/weather_filter.py
"""
Module: weather_filter.py

Implements functions for loading and filtering datasets by weather type.
Intended use:
1. Load weather types from a CSV, each row corresponding to a day with a date column and a weather type index (1–9).
2. Convert this to an xarray DataArray aligned on time.
3. Filter or group a Dataset or DataArray by specific weather types.

Example usage:
    >>> from weather_filter import load_weather_types_csv, filter_by_weather_types
    >>> wt_da = load_weather_types_csv("weather_types.csv", date_col="date", wt_col="slwt")
    >>> ds_filtered = filter_by_weather_types(ds, wt_da, include_types=[1,3])
"""

import pandas as pd
import xarray as xr

def load_weather_types_csv(csv_path, date_col="date", wt_col="slwt"):
    """
    Load a CSV file containing daily (or sub-daily) weather type info.

    The CSV must have:
      - A column named `date_col` with parseable datetimes (default: "date").
      - A column named `wt_col` with the weather type index (default: "slwt").

    Returns
    -------
    wt_da : xarray.DataArray
        DataArray with dimension "time", coordinate "time".
    """
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df.sort_values(by=date_col, inplace=True)
    df.set_index(date_col, inplace=True)

    # Rename the index to "time"
    df.index.name = "time"

    wt_da = xr.DataArray(
        data=df[wt_col].values,
        coords={"time": df.index},
        dims=["time"],
        name="weather_type"
    )
    
    return wt_da

def filter_by_weather_types(ds, wt_da, include_types=None):
    """
    Filter a dataset by weather types in `include_types`.
    ...
    """
    if not include_types:
        return ds
    
    wt_da_aligned = wt_da.reindex(time=ds.time, method="nearest")
    mask = wt_da_aligned.isin(include_types)
    ds_filtered = ds.where(mask, drop=True)
    return ds_filtered
