# scripts/data_loader.py

import xarray as xr

def load_nc_files(file_pattern, concat_dim=None, combine='by_coords', chunks=None):
    """
    Load multiple NetCDF files using xarray.open_mfdataset.
    
    Parameters
    ----------
    file_pattern : str or list
        File pattern (e.g. "data/INCAL_HOURLY_RR_*.nc") or a list of file paths.
    concat_dim : str, optional
        Dimension along which to concatenate datasets (default is 'time').
    combine : str, optional
        How to combine datasets (default is 'by_coords').
    
    Returns
    -------
    ds : xarray.Dataset
        The combined dataset.
    """
    ds = xr.open_mfdataset(file_pattern, concat_dim=concat_dim, combine=combine, chunks=chunks)
    return ds
