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
import numpy as np
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
#    df = df["time"].dt.floor("D")

    wt_da = xr.DataArray(
        data=df[wt_col].values,
        coords={"time": df.index},
        dims=["time"],
        name="weather_type"
    )
    
    return wt_da

def filter_by_weather_types(ds, wt_da, include_types=None):
    """
    Keep all hours from any day whose weather type is in `include_types`,
    without shifting or changing actual data times.

    ds:   xarray.Dataset with hourly resolution
    wt_da: xarray.DataArray with daily resolution (~11:00 each day)
    include_types: list of types (e.g. [1,2,3,4,5,6,7,8,9])
    """
    if not include_types:
        return ds

    # 1) Build the set of "included day labels" from the CSV
    included_days = get_included_days(wt_da, include_types)
    if not included_days:
        print("No days matched the requested weather types!")
        return ds.isel(time=0).drop_isel(time=range(ds.sizes["time"]))  # returns an empty dataset


    # 2) For each *hourly* time in ds, floor to day => e.g. 2017-10-01T00:00 => 2017-10-01
    ds_day = ds.time.dt.floor("D").values.astype("datetime64[D]")

    # 3) Build a boolean array: True if this hour's day is in included_days
    mask_arr = np.isin(ds_day, list(included_days))  # shape=(time,)

    # (C) Build a boolean mask that is True if this hour's day is in days_included:
    #     np.isin(...) returns True/False for each element of ds_day vs. the array of days_included
    mask = xr.DataArray(
        mask_arr,
        coords={"time": ds.time},
        dims=["time"]
    )

    # (D) Filter out any hours that are not in the included days:
    ds_filtered = ds.where(mask, drop=True)

    return ds_filtered


def get_included_days(wt_da, include_types):
    """
    From a daily weather-type DataArray (times at ~11:00 each day),
    return a Python set of datetime64[D] day labels for the days we want to include.
    """
    # This will store e.g. {numpy.datetime64('2017-10-01'), numpy.datetime64('2017-10-02'), ...}
    included_days = set()

    # Make sure we handle all timesteps of wt_da
    for i in range(wt_da.time.size):
        # The actual timestamp might be e.g. 2017-10-01T11:00:00
        # We 'floor' it to the day boundary => 2017-10-01 (midnight)
        day_label = wt_da.time[i].dt.floor("D").values.astype("datetime64[D]")

        # Check if the weather type at that day is in our list
        # (Cast to plain Python int or str if needed)
        weather_type_value = wt_da[i].item()  # e.g. 6, 7, etc.
        if str(weather_type_value) in include_types or weather_type_value in include_types:
            included_days.add(day_label)

    return included_days
