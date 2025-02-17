# precip_analysis/utils.py

def prepare_ensemble_grid(ds, lat_var='latitude', lon_var='longitude'):
    """
    Prepare the ensemble dataset for regridding by converting 2D spatial variables 
    into proper 1D coordinate variables and swapping dimensions.
    
    This function implements the following steps:
    
    # --- Step 1: Extract 1D coordinates from the ensemble dataset ---
    lat_1d = ds[lat_var][:, 0].data   # each row is assumed constant in latitude
    lon_1d = ds[lon_var][0, :].data    # each column is assumed constant in longitude
    
    # --- Step 2: Assign these as coordinates (and drop the original variables) ---
    ds1 = ds.assign_coords(
        lat_coord=("lat", lat_1d),
        lon_coord=("lon", lon_1d)
    ).drop_vars([lat_var, lon_var])
    
    # Rename (optional) and swap dims so that 'lat_coord' and 'lon_coord' become the dimensions
    ds1_swapped = ds1.swap_dims({"lat": "lat_coord", "lon": "lon_coord"})
    
    Parameters
    ----------
    ds : xarray.Dataset
        The ensemble dataset to prepare. Expected to have dimensions 'lat' and 'lon'.
    lat_var : str, optional
        Name of the 2D variable storing latitude (default 'latitude').
    lon_var : str, optional
        Name of the 2D variable storing longitude (default 'longitude').
    
    Returns
    -------
    ds_prepared : xarray.Dataset
        The dataset with proper 1D coordinate variables and swapped dimensions.
    """
    # Extract 1D coordinates from the 2D variables
    lat_1d = ds[lat_var][0, :, 0, 0].data
    lon_1d = ds[lon_var][0, 0, :, 0].data

    # Assign these as new coordinates and drop the original 2D variables
    ds1 = ds.assign_coords(
        lat_coord=("lat", lat_1d),
        lon_coord=("lon", lon_1d)
    ).drop_vars([lat_var, lon_var])
    
    # Swap dimensions so that lat_coord and lon_coord become the indexing dimensions
    ds_prepared = ds1.swap_dims({"lat": "lat_coord", "lon": "lon_coord"})
    
    return ds_prepared

def prepare_reference_grid(ds, lat_var='lat', lon_var='lon', dim_lat='y', dim_lon='x'):
    """
    Prepare the reference dataset for regridding by converting 2D coordinate variables 
    into 1D coordinate variables and swapping dimensions.
    
    This function implements:
    
    # --- Step 3: Prepare the reference dataset ---
    lat_ref = ds[lat_var][:, 0].data
    lon_ref = ds[lon_var][0, :].data
    
    ds2 = ds.assign_coords(
        lat_coord=(dim_lat, lat_ref),
        lon_coord=(dim_lon, lon_ref)
    ).rename({dim_lat: "lat_dim", dim_lon: "lon_dim"})
    
    ds2_prepared = ds2.swap_dims({"lat_dim": "lat_coord", "lon_dim": "lon_coord"})
    
    Parameters
    ----------
    ds : xarray.Dataset
        The reference dataset to prepare. Expected to have spatial coordinate variables 
        (e.g., 'lat' and 'lon') defined on dimensions given by `dim_lat` and `dim_lon`.
    lat_var : str, optional
        Name of the latitude coordinate variable (default 'lat').
    lon_var : str, optional
        Name of the longitude coordinate variable (default 'lon').
    dim_lat : str, optional
        Name of the dimension associated with latitude (default 'y').
    dim_lon : str, optional
        Name of the dimension associated with longitude (default 'x').
    
    Returns
    -------
    ds_prepared : xarray.Dataset
        The reference dataset with new 1D coordinates ('lat_coord', 'lon_coord') and swapped dims.
    """
    # Extract 1D coordinate arrays from the 2D coordinate variables
    lat_1d = ds[lat_var][:, 0].data
    lon_1d = ds[lon_var][0, :].data

    # Assign new coordinates; note that we use the original spatial dimensions (e.g., 'y' and 'x')
    ds2 = ds.assign_coords(
        lat_coord=(dim_lat, lat_1d),
        lon_coord=(dim_lon, lon_1d)
    ).rename({dim_lat: "lat_dim", dim_lon: "lon_dim"})
    
    # Swap dims so that the new coordinates become the dimensions
    ds_prepared = ds2.swap_dims({"lat_dim": "lat_coord", "lon_dim": "lon_coord"})
    
    return ds_prepared
