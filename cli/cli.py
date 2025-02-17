# cli.py
import sys
import os

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import os
import xarray as xr
from scripts.ensemble_loader import load_ensemble_files
from scripts.subsetting import subset_by_lat_lon, subset_time
from scripts.utils import prepare_ensemble_grid, prepare_reference_grid
from scripts.data_loader import load_nc_files
from scripts.regridding import regrid_to_target
from scripts.temporal_stats import aggregate_to_daily
from scripts.bias_metrics import root_mean_squared_error

def run_analysis(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load ensemble
    ds_ensemble = load_ensemble_files(config['input']['ensemble_pattern'])
    ds_ensemble = subset_by_lat_lon(ds_ensemble, 
                                lat_bounds=config['subset']['lat_bounds'], 
                                lon_bounds=config['subset']['lon_bounds'], 
                                lat_var='latitude', lon_var='longitude')
    ds_ensemble = subset_time(ds_ensemble, config['subset']['start_time'], config['subset']['end_time'])
    ds_ensemble = prepare_ensemble_grid(ds_ensemble)
    
    # Load reference
    ds_ref = load_nc_files(config['input']['reference_file'])
    ds_ref = subset_by_lat_lon(ds_ref, 
                           lat_bounds=config['subset']['lat_bounds'], 
                           lon_bounds=config['subset']['lon_bounds'], 
                           lat_var='lat', lon_var='lon')  # Correct variable names
    ds_ref = subset_time(ds_ref, config['subset']['start_time'], config['subset']['end_time'])
    ds_ref = prepare_reference_grid(ds_ref)
    
    # Regrid ensemble to match reference grid
    ds_ensemble_interp = regrid_to_target(ds_ensemble, ds_ref)
    
    # Aggregate to daily values
    daily_ens = aggregate_to_daily(ds_ensemble_interp, "precipitation")
    daily_ref = aggregate_to_daily(ds_ref, "RR", missing_value=-999)
    
    # Compute RMSE
    rmse_map = root_mean_squared_error(daily_ens, daily_ref, dim="time")
    
    # Save outputs
    if 'save_netcdf' in config['output']:
        rmse_map.to_netcdf(config['output']['save_netcdf'])
    
    if 'save_plot' in config['output']:
        rmse_map.sel(member='00').plot()
        import matplotlib.pyplot as plt
        plt.savefig(config['output']['save_plot'])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run climate analysis')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    run_analysis(args.config)
