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
        # If it's stored as a coordinate, you can do:
    if "lambert_conformal_conic" in ds.coords:
        ds = ds.reset_coords("lambert_conformal_conic", drop=True)
    if "lambert_conformal_conic" in dem_ds.coords:
        dem_ds = dem_ds.reset_coords("lambert_conformal_conic", drop=True)
    # Attach the altitude from dem_ds
    ds["altitude"] = dem_ds["altitude"]
    
    return ds


def mask_altitude_bins(ds_or_da, alt_var="altitude", bins=[0, 500, 1000, 2000]):
    """
    Manually create a dictionary of masked DataArrays (or Datasets) 
    for each altitude bin range. The shape remains (lat_coord, lon_coord[, time,...])
    but cells not in the bin are set to NaN.

    Parameters
    ----------
    ds_or_da : xarray.Dataset or xarray.DataArray
        Must have 'altitude' as shape (lat_coord, lon_coord) (or broadcastable).
    alt_var : str
        Name of the altitude variable in ds_or_da.
    bins : list of float
        e.g. [0, 500, 1000, 2000, ...]

    Returns
    -------
    bin_dict : dict
        Keys are tuples (bin_min, bin_max), 
        values are masked DataArrays (or Datasets) of the same shape, 
        except out-of-bin cells are NaN.
    """
    # Ensure altitude is a coordinate:
    if alt_var not in ds_or_da.coords:
        if alt_var in ds_or_da:
            ds_or_da = ds_or_da.set_coords(alt_var)
        else:
            raise KeyError(f"'{alt_var}' not found in the provided data.")

    bin_dict = {}
    for i in range(len(bins) - 1):
        min_b = bins[i]
        max_b = bins[i+1]
        # create a label or a tuple
        bin_label = (min_b, max_b)

        # create a mask for cells in [min_b, max_b)
        mask = (ds_or_da[alt_var] >= min_b) & (ds_or_da[alt_var] < max_b)
        
        # apply the mask with .where(...) => out-of-bin cells become NaN
        ds_bin = ds_or_da.where(mask, drop=False)  # keep shape, fill with NaN

        bin_dict[bin_label] = ds_bin
    return bin_dict
