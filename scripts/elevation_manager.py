# scripts/elevation_manager.py

import xarray as xr
from scripts.subsetting import subset_by_lat_lon
from scripts.utils import prepare_reference_grid
from scripts.regridding import regrid_to_target

def load_and_subset_dem(
    dem_path,
    dem_var="ZH",
    lat_bounds=None,
    lon_bounds=None,
    lat_var="lat",
    lon_var="lon",
    dim_lat="y",
    dim_lon="x"
):
    """
    Load the DEM dataset from NetCDF, rename the variable to 'altitude',
    subset in space (lat_bounds/lon_bounds),
    and prepare to 1D lat/lon coordinates.
    
    Parameters
    ----------
    dem_path : str
        Path to the DEM netCDF file.
    dem_var : str, optional
        Name of the DEM variable (default 'ZH').
    lat_bounds : tuple of float, optional
        (min_lat, max_lat). If None, no spatial lat subsetting is done.
    lon_bounds : tuple of float, optional
        (min_lon, max_lon). If None, no spatial lon subsetting is done.
    lat_var : str, optional
        Name of the latitude variable (default 'lat').
    lon_var : str, optional
        Name of the longitude variable (default 'lon').
    dim_lat : str, optional
        Name of the latitude dimension (default 'y').
    dim_lon : str, optional
        Name of the longitude dimension (default 'x').

    Returns
    -------
    ds_dem_prepared : xarray.Dataset
        DEM dataset with 'altitude' variable, 1D lat/lon coordinates, 
        and optional subsetting applied.
    """
    # 1. Load DEM
    ds_dem = xr.open_dataset(dem_path)
    if dem_var not in ds_dem:
        raise ValueError(f"Variable '{dem_var}' not found in {dem_path}.")
    ds_dem = ds_dem.rename({dem_var: "altitude"})
    
    # 2. Subset by space
    if lat_bounds and lon_bounds:
        ds_dem = subset_by_lat_lon(ds_dem, lat_bounds, lon_bounds, lat_var=lat_var, lon_var=lon_var)
    
    # 3. Prepare to 1D lat/lon
    #    (If your DEM has lat(y,x) and lon(y,x), this should work.)
    ds_dem_prepared = prepare_reference_grid(
        ds_dem, 
        lat_var=lat_var, 
        lon_var=lon_var, 
        dim_lat=dim_lat, 
        dim_lon=dim_lon
    )
    return ds_dem_prepared


def bin_by_altitude(ds, alt_var="altitude", bins=None):
    """
    Bin dataset grid cells by altitude ranges.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing a variable named alt_var.
    alt_var : str, optional
        Name of altitude variable (default "altitude").
    bins : list of float, optional
        E.g. [0, 500, 1000, 2000]. If None, a default list is used.

    Returns
    -------
    dict
        Keys are (min_bin, max_bin), values are subsets of ds in that bin.
    """
    if alt_var not in ds:
        raise ValueError(f"Variable '{alt_var}' not found in dataset.")
    
    if bins is None:
        bins = [0, 500, 1000, 1500, 2000, 3000]

    bin_dict = {}
    alt_da = ds[alt_var]
    for i in range(len(bins) - 1):
        min_b = bins[i]
        max_b = bins[i + 1]
        mask = (alt_da >= min_b) & (alt_da < max_b)
        bin_ds = ds.where(mask, drop=True)
        bin_dict[(min_b, max_b)] = bin_ds

    return bin_dict


def add_alt_to_ds(ds, dem_ds):
    """
    Add altitude to the dataset.
    
    Parameters
    ----------
    ds : xarray.Dataset
        The dataset (with lat_coord/lon_coord).
    dem_ds : xarray.Dataset
        The prepared DEM dataset (with lat_coord/lon_coord).
    
    Returns
    -------
    ds_with_alt : xarray.Dataset
        Dataset with 'altitude' variable on the matching grid.
    """
    # Attach the altitude from dem_ds
    ds["altitude"] = dem_ds["altitude"]
    
    return ds
