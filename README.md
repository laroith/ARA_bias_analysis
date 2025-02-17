# README.md

# Climate Analysis

## Overview
This package provides tools for analyzing and validating high-resolution climate data, including:
- Data loading from NetCDF files
- Preprocessing and aggregation
- Regridding and subsetting
- Bias metric calculations
- Temporal statistics
- Command-line execution support

## Installation
```sh
pip install .
```

## Command-Line Usage
Run a complete analysis using a configuration file:
```sh
python cli.py --config config.yml
```

## Example Analysis Workflow
A typical analysis involves:
1. **Loading ensemble and reference datasets**
2. **Subsetting by space and time**
3. **Regridding ensemble data to match reference grid**
4. **Computing daily precipitation sums**
5. **Calculating bias metrics (e.g., RMSE)**
6. **Saving outputs (NetCDF, plots, statistics)**

## Configuration File
Users define parameters in a `config.yml` file:
```yaml
input:
  ensemble_pattern: "../data/total_precipitation_2017*.nc"
  reference_file: "../data/INCA_RR_20171*.nc"
subset:
  lat_bounds: [46, 48]
  lon_bounds: [10, 13]
  start_time: "2017-10-01T00:00:00"
  end_time: "2017-10-02T23:00:00"
output:
  save_netcdf: "output/bias_metrics.nc"
  save_plot: "output/rmse_map.png"
```

## Requirements
- Python 3.7+
- numpy
- xarray
- matplotlib
- scipy
- dask
- netCDF4
- h5py
- pyyaml

## License
MIT License
