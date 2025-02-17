"""
Module: bias_metrics.py

Implements common bias metrics for comparing climate or weather datasets:
    - Mean Error (ME) or Bias
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)

These metrics assume you've already:
  1. Aligned the datasets spatially and temporally (same time, lat, lon, etc.).
  2. Handled any ensemble dimension (e.g., compute ensemble mean first,
     or pass matching member dimension in both model and reference arrays).

Dimension Handling:
- By default, dim=None => the metrics compute a single scalar across *all* dimensions.
- You can pass a list of dimension names to control how data is collapsed:
   e.g. dim=["time", "lat", "lon"] will produce one result per ensemble member if "member" is also in the data.
   e.g. dim="time" will produce a result for each (lat, lon, member).

Examples:
    >>> # Suppose model_da and ref_da share dims (time, lat, lon).
    >>> # (a) Single scalar RMSE across everything:
    >>> single_value = root_mean_squared_error(model_da, ref_da)
    >>> # (b) RMSE for each ensemble member, ignoring lat/lon:
    >>> rmse_per_member = root_mean_squared_error(model_da, ref_da, dim=["time", "lat", "lon"])
    >>> # (c) RMSE for each grid cell (collapsing only time):
    >>> rmse_map = root_mean_squared_error(model_da, ref_da, dim="time")

You can also use compute_all_bias_metrics() to get ME, MAE, and RMSE in a single Dataset.
"""

import xarray as xr

def mean_error(model: xr.DataArray, reference: xr.DataArray, dim=None) -> xr.DataArray:
    """
    Mean Error (ME) or Bias = average(model - reference)
    
    Args:
        model (xarray.DataArray): Model or ensemble data array with dims (time, lat, lon, [member]).
        reference (xarray.DataArray): Reference data array with matching coords/dims.
        dim (str or list of str, optional): Dimension(s) over which to average.
                                           If None, average across all dims.
    Returns:
        xarray.DataArray: Mean Error.
    """
    return (model - reference).mean(dim=dim)

def mean_absolute_error(model: xr.DataArray, reference: xr.DataArray, dim=None) -> xr.DataArray:
    """
    Mean Absolute Error (MAE) = average(|model - reference|)
    """
    return abs(model - reference).mean(dim=dim)

def root_mean_squared_error(model: xr.DataArray, reference: xr.DataArray, dim=None) -> xr.DataArray:
    """
    Root Mean Squared Error (RMSE) = sqrt( average( (model - reference)^2 ) )
    """
    return ((model - reference) ** 2).mean(dim=dim) ** 0.5

def compute_all_bias_metrics(model: xr.DataArray,
                             reference: xr.DataArray,
                             dim=None) -> xr.Dataset:
    """
    Convenience function returning ME, MAE, RMSE as a single xarray.Dataset.
    
    Returns a dataset with:
      - mean_error
      - mean_absolute_error
      - root_mean_squared_error
    """
    me = mean_error(model, reference, dim=dim)
    mae = mean_absolute_error(model, reference, dim=dim)
    rmse = root_mean_squared_error(model, reference, dim=dim)
    
    return xr.Dataset({
        "mean_error": me,
        "mean_absolute_error": mae,
        "root_mean_squared_error": rmse
    })
