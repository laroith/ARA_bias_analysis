# precip_analysis/subsetting.py
import xarray as xr
import numpy as np

def subset_by_lat_lon(ds, lat_bounds, lon_bounds, lat_var='lat', lon_var='lon'):
    """
    Subset an xarray dataset based on latitude and longitude bounds.
    
    If the dataset has lat and lon as coordinates, it uses .sel.
    Otherwise, it uses a boolean mask via .where(..., drop=True).
    
    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to subset.
    lat_bounds : tuple of float
        (min_lat, max_lat)
    lon_bounds : tuple of float
        (min_lon, max_lon)
    lat_var : str, optional
        Name of the latitude variable or coordinate (default is 'lat').
    lon_var : str, optional
        Name of the longitude variable or coordinate (default is 'lon').
    
    Returns
    -------
    ds_subset : xarray.Dataset
        The subset dataset.
    """
    if not isinstance(lat_bounds, (tuple, list)) or not isinstance(lon_bounds, (tuple, list)):
        raise TypeError(f"lat_bounds and lon_bounds must be tuples/lists, but received {type(lat_bounds)} and {type(lon_bounds)}.")

    if lat_var not in ds.variables or lon_var not in ds.variables:
        raise KeyError(f"Dataset variables: {list(ds.variables.keys())}. Expected: '{lat_var}', '{lon_var}'")

    if lat_var in ds.coords and lon_var in ds.coords:
        # Force the coordinate arrays to be computed in memory.
        lat_da = ds[lat_var].load()
        lon_da = ds[lon_var].load()

        # Create a boolean mask using the in-memory NumPy arrays.
        mask = ((lat_da.values >= lat_bounds[0]) & (lat_da.values <= lat_bounds[1]) &
                (lon_da.values >= lon_bounds[0]) & (lon_da.values <= lon_bounds[1]))
        # Wrap the mask in a DataArray using the dims and coords of lat_da.
        mask_da = xr.DataArray(mask, dims=lat_da.dims, coords=lat_da.coords)
        ds_subset = ds.where(mask_da, drop=True)
    else:
        lat_arr = ds[lat_var].values  # Convert to a NumPy array
        lon_arr = ds[lon_var].values

        mask = ((ds[lat_var] >= lat_bounds[0]) & (ds[lat_var] <= lat_bounds[1]) &
                (ds[lon_var] >= lon_bounds[0]) & (ds[lon_var] <= lon_bounds[1]))
        ds_subset = ds.where(mask, drop=True)
    return ds_subset


def subset_time(ds, start_time, end_time):
    """
    Subset an xarray dataset along the time dimension.
    
    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to subset.
    start_time : str or np.datetime64
        Start time (e.g., '2010-01-01')
    end_time : str or np.datetime64
        End time (e.g., '2020-12-31')
    
    Returns
    -------
    ds_time_subset : xarray.Dataset
        The time-subset dataset.
    """
    return ds.sel(time=slice(start_time, end_time))
