# precip_analysis/regridding.py

def regrid_to_target(source, target, source_dim_names=('lat_coord', 'lon_coord'),
                       target_coord_names=('lat_coord', 'lon_coord'), method='nearest'):
    """
    Regrid the source DataArray or Dataset onto the target grid.
    
    This function assumes that the source dataset already uses its coordinate variables as the
    dimension indexes (e.g. via swap_dims), and that the target dataset has matching coordinate names.
    
    Parameters
    ----------
    source : xarray.DataArray or xarray.Dataset
        The source data (e.g. ensemble precipitation) with dimensions given by source_dim_names.
    target : xarray.DataArray or xarray.Dataset
        The target grid (e.g. reference precipitation) with coordinates named in target_coord_names.
    source_dim_names : tuple of str, optional
        The dimension names in source data (default is ('lat_coord', 'lon_coord')).
    target_coord_names : tuple of str, optional
        The coordinate names in target data (default is ('lat_coord', 'lon_coord')).
    method : str, optional
        Interpolation method (default is 'nearest').
    
    Returns
    -------
    regridded : xarray.DataArray or xarray.Dataset
        The source data interpolated to the target grid.
    """
    regrid_kwargs = {
        source_dim_names[0]: target[target_coord_names[0]],
        source_dim_names[1]: target[target_coord_names[1]]
    }
    regridded = source.interp(**regrid_kwargs, method=method)
    return regridded
