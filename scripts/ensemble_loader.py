# precip_analysis/ensemble_loader.py

import os
import re
from glob import glob
import xarray as xr

def parse_member_id(filepath):
    """
    Parse the two-digit ensemble member from filenames like:
        total_precipitation_20170101_00.nc
        total_precipitation_20170101_01.nc
    Returns the string member ID (e.g. "00", "01", ...).

    Parameters
    ----------
    filepath : str
        The full file path.

    Returns
    -------
    str
        The ensemble member identifier.
    """
    basename = os.path.basename(filepath)
    match = re.search(r"_(\d{2})\.nc$", basename)
    if not match:
        raise ValueError(f"Could not parse member ID from filename: {filepath}")
    return match.group(1)

def load_ensemble_files(file_pattern, chunks=None):
    """
    Loads multiple daily netCDF files for different ensemble members, 
    concatenates them along 'time', then merges them along a new 'member' dimension.

    Parameters
    ----------
    file_pattern : str
        Glob pattern to find daily ensemble files 
        (e.g. 'data/20170102/total_precipitation_20170102_*.nc').
    chunks : dict, optional
        Dictionary for Dask chunking 
        (e.g. {'time': 24, 'lat': 100, 'lon': 100}). If provided, the files will be
        lazy loaded using dask.

    Returns
    -------
    xarray.Dataset
        A dataset with dimensions (time, lat, lon, member).
    """
    filepaths = sorted(glob(file_pattern))
    if not filepaths:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

    ds_list = []
    for fp in filepaths:
        member_id = parse_member_id(fp)
        # Lazy load the file using dask (if chunks is provided)
        ds = xr.open_dataset(fp, chunks=chunks)
        # Expand a new dimension 'member' with the parsed ID
        ds = ds.expand_dims({"member": [member_id]})
        ds_list.append(ds)

    # Combine datasets along the existing time coordinate and the new member dimension.
    ds_ensemble = xr.combine_by_coords(ds_list, combine_attrs="override")
    # Reorder dimensions (if needed) to (time, lat, lon, member)
    # (Assumes that spatial dimensions are named 'lat' and 'lon' in the individual files.)
    ds_ensemble = ds_ensemble.transpose("time", "lat", "lon", "member")
    
    return ds_ensemble
