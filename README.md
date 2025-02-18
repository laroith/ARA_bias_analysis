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

## Project Structure

- **cli.py**  
  Main entry point, using `argparse` for command-line options. Pulls in configuration from a YAML file plus CLI overrides, then calls `run_analysis`.
- **scripts/**  
  Contains modular Python scripts for subsetting, regridding, loading data, computing metrics, etc.
- **tests/**  
  Pytest-based tests verifying the loading, subsetting, filtering, and bias calculations.  
  - `test_cli_ens_selection.py` verifies ensemble member selection.  
  - `test_cli_weathertypes.py` verifies weather-type filtering.

## Installation
```sh
pip install .
```

## Command-Line Usage
Run a complete analysis using a configuration file:
```sh
python cli.py --config config.yaml
```

## Example Analysis Workflow
A typical analysis involves:
1. **Loading ensemble and reference datasets**
2. **Subsetting by space and time**
3. **Filtering by Weather Type using a CSV file that defines weather types for each date. This feature now supports automatically filtering both ensemble and reference datasets via the CLI (`--weather-type-file`, `--include-weather-types 1 2 3`, etc.).**
4. **Regridding ensemble data to match reference grid**
5. **Computing daily, monthly, seasonal, ... precipitation sums/means**
6. **Calculating bias metrics (e.g., RMSE)**
7. **Saving outputs (NetCDF, plots, statistics)**

## Configuration File
Users define parameters in a `config.yaml` file:
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
