import pytest
import numpy as np
import xarray as xr
import os

from scripts.read_dataset import read_precipitation_dataset


@pytest.fixture
def mock_reanalysis_file(tmp_path):
    """
    Creates a minimal in-memory NetCDF file resembling the *reanalysis* data:
      - Dimensions: time (24), lat, lon
      - Variable name: 'precipitation'
      - Units: mm
      - No scale_factor or missing_value
      - time: 'hours since 1800-01-01'
    But we'll reduce lat/lon to smaller shape for quick tests, e.g. lat=2, lon=3
    """
    time_len, lat_len, lon_len = 4, 2, 3  # tiny example
    data = np.random.randint(0, 10, size=(time_len, lat_len, lon_len))
    
    ds = xr.Dataset(
        {
            "precipitation": (("time", "lat", "lon"), data)
        },
        coords={
            "time": ("time", np.arange(time_len)),
            "lat": (("lat"), [45.0, 46.0]),
            "lon": (("lon"), [14.0, 15.0, 16.0])
        }
    )
    
    # Add relevant attributes
    ds["time"].attrs["units"] = "hours since 1800-01-01"
    ds["precipitation"].attrs["units"] = "mm"
    
    # Save to temporary file
    nc_file = tmp_path / "reanalysis_mock.nc"
    ds.to_netcdf(nc_file)
    return nc_file


@pytest.fixture
def mock_daily_ref_file(tmp_path):
    """
    Creates a minimal in-memory NetCDF file resembling the *first reference dataset*:
      - daily resolution: time=365, but we'll reduce to 5 for speed
      - variable name 'RR'
      - scale_factor=0.1
      - missing_value=-999
      - shape (time, y, x), e.g. y=2, x=3
      - time units = 'days since 1961-01-01'
    """
    time_len, y_len, x_len = 5, 2, 3
    data = np.array([
        # random data but let's embed some -999
        [[0, 2, -999], [3, 5, 1]],
        [[-999, 2, 3], [9, -999, 4]],
        [[1, 0, 2], [2, 3, 5]],
        [[-999, -999, 0], [3, 4, 1]],
        [[2, 2, 2], [9, 9, -999]],
    ], dtype=int)
    
    ds = xr.Dataset(
        {
            "RR": (("time", "y", "x"), data)
        },
        coords={
            "time": ("time", np.arange(time_len)),
            "y": ("y", [1000, 2000]),
            "x": ("x", [1100, 1200, 1300])
        }
    )
    
    # Add daily-specific attributes
    ds["time"].attrs["units"] = "days since 1961-01-01"
    ds["RR"].attrs["scale_factor"] = 0.1
    ds["RR"].attrs["missing_value"] = -999
    ds["RR"].attrs["units"] = "kg m-2"
    ds["RR"].attrs["cell_method"] = "time: sum (7:00 CET to 7:00 CET day+1)"
    
    nc_file = tmp_path / "daily_ref_mock.nc"
    ds.to_netcdf(nc_file)
    return nc_file


@pytest.fixture
def mock_hourly_ref_file(tmp_path):
    """
    Creates a minimal in-memory NetCDF file resembling the *second reference dataset*:
      - time=744 (hourly for a 31-day month), but we'll reduce to, say, 6 for speed
      - variable name 'RR'
      - scale_factor=0.001
      - missing_value=-999
      - shape (time, y, x)
      - time units = 'seconds since 1961-01-01'
    """
    time_len, y_len, x_len = 6, 2, 2
    data = np.array([
        [[0, -999],
         [2, 3]],
        [[100, 200],
         [-999, 400]],
        [[500, 600],
         [700, -999]],
        [[-999, -999],
         [900, 1000]],
        [[1100, 1200],
         [1300, 1400]],
        [[-999, 2],
         [5, 7]]
    ], dtype=int)  # shape (6,2,2)
    
    ds = xr.Dataset(
        {
            "RR": (("time", "y", "x"), data)
        },
        coords={
            "time": ("time", np.arange(time_len)),
            "y": ("y", [3000, 4000]),
            "x": ("x", [5000, 6000])
        }
    )
    
    # Add hourly-specific attributes
    ds["time"].attrs["units"] = "seconds since 1961-01-01"
    ds["RR"].attrs["scale_factor"] = 0.001
    ds["RR"].attrs["missing_value"] = -999
    ds["RR"].attrs["units"] = "kg m-2"
    ds["RR"].attrs["cell_method"] = "time: sum"
    
    nc_file = tmp_path / "hourly_ref_mock.nc"
    ds.to_netcdf(nc_file)
    return nc_file


def test_read_reanalysis(mock_reanalysis_file):
    """
    Tests that we can read the mock reanalysis data 
    and that the shape, variable, and naming are correct.
    """
    ds = read_precipitation_dataset(file_path=str(mock_reanalysis_file), var_name="precipitation")
    assert "precip" in ds.data_vars, "Expected the variable to be renamed to 'precip'"
    
    # The shape should match (time_len=4, lat_len=2, lon_len=3)
    assert ds["precip"].shape == (4, 2, 3)
    # No scale_factor or missing_value, so just check it's not turned into NaNs
    assert not np.isnan(ds["precip"]).any(), "Reanalysis mock data had no missing values, so no NaNs expected."
    print("Test reanalysis dataset: PASSED")


def test_read_daily_reference(mock_daily_ref_file):
    """
    Test the daily reference dataset scenario: 'RR' variable, 
    scale_factor=0.1, missing_value=-999
    """
    ds = read_precipitation_dataset(file_path=str(mock_daily_ref_file), var_name="RR")
    
    # Should be renamed to "precip"
    assert "precip" in ds.data_vars
    assert "RR" not in ds.data_vars
    
    # shape: (time_len=5, y_len=2, x_len=3)
    assert ds["precip"].shape == (5, 2, 3)
    
    # Because we set scale_factor=0.1, and missing_value=-999
    # let's check some known positions
    # We'll check the first data point: originally 0 => 0.0 after scale
    np.testing.assert_allclose(ds["precip"].isel(time=0, y=0, x=0), 0.0)
    
    # The original -999 should be replaced by NaN
    # e.g. ds["precip"].isel(time=0, y=0, x=2)
    # after scale => -99.9, but that should become NaN
    val = ds["precip"].isel(time=0, y=0, x=2).values
    assert np.isnan(val), "Missing value -999 should become NaN"
    
    # Another check: 2 => 0.2 if scale_factor=0.1
    # e.g. ds["precip"].isel(time=0, y=0, x=1)
    np.testing.assert_allclose(ds["precip"].isel(time=0, y=0, x=1), 0.2)
    
    print("Test daily reference dataset: PASSED")


def test_read_hourly_reference(mock_hourly_ref_file):
    """
    Test the second (hourly) reference dataset scenario: 
    'RR' var, scale_factor=0.001, missing_value=-999
    """
    ds = read_precipitation_dataset(file_path=str(mock_hourly_ref_file), var_name="RR")
    
    # renamed to "precip"
    assert "precip" in ds.data_vars
    assert "RR" not in ds.data_vars
    
    # shape: (time=6, y=2, x=2)
    assert ds["precip"].shape == (6, 2, 2)
    
    # scale_factor=0.001 => e.g. original 100 => 0.1
    # missing_value -999 => NaN
    val = ds["precip"].isel(time=1, y=0, x=0).values  # originally 100 => 100 * 0.001 = 0.1
    np.testing.assert_allclose(val, 0.1)
    
    # check -999 => NaN
    val_missing = ds["precip"].isel(time=1, y=0, x=1).values  # originally 200 => 0.2
    # let's pick a location that was -999
    val_missing2 = ds["precip"].isel(time=1, y=1, x=0).values  # originally -999 => NaN
    assert np.isnan(val_missing2)
    
    print("Test hourly reference dataset: PASSED")
