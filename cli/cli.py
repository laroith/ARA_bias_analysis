# cli.py selection, aggregation, bias
import sys
import os

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import os
import xarray as xr
import numpy as np
from scripts.ensemble_loader import load_ensemble_files
from scripts.subsetting import subset_by_lat_lon, subset_time
from scripts.utils import prepare_ensemble_grid, prepare_reference_grid
from scripts.data_loader import load_nc_files
from scripts.regridding import regrid_to_target
from scripts.temporal_stats import (
    aggregate_to_daily, aggregate_by_month, aggregate_by_year,
    aggregate_by_season, group_by_weather_type
)
from scripts.bias_metrics import (
    root_mean_squared_error, mean_absolute_error, mean_error
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run climate analysis")

    parser.add_argument("--config", type=str, required=False, help="Path to YAML config file.")

    parser.add_argument("--inspect", action="store_true",
                        help="If set, load data, print info, and exit.")

    # Ensemble selection
    parser.add_argument("--member", type=str, nargs='*',
                        help="One or more ensemble member IDs to analyze.")
    parser.add_argument("--ensemble-mean", action='store_true',
                        help="Compute the ensemble mean across selected members.")

    # Example for overriding variable name
    parser.add_argument("--var-name", type=str,
                        help="Override the default ensemble variable name.")
    parser.add_argument("--ref-var-name", type=str,
                        help="Override the default reference variable name.")

    # Example for temporal aggregation
    parser.add_argument("--aggregation", type=str, choices=["daily", "monthly", "seasonal", "yearly", "none"],
                        help="Temporal aggregation period.")

    # Example for bias dimension and metrics
    parser.add_argument("--bias-dim", type=str, nargs='*',
                        help="Dimensions to average over, e.g. time lat lon.")
    parser.add_argument("--bias-metrics", type=str, nargs='*', choices=["RMSE", "MAE", "ME"],
                        help="Which bias metrics to compute.")

    return parser.parse_args()



def run_analysis(config):
    """
    Run the core climate analysis based on the provided configuration dictionary.
    """

    print("Loading ensemble dataset...")
    ds_ensemble = load_ensemble_files(config['input']['ensemble_pattern'])

    # Subset lat/lon
    ds_ensemble = subset_by_lat_lon(
        ds_ensemble,
        lat_bounds=config['subset']['lat_bounds'],
        lon_bounds=config['subset']['lon_bounds'],
        lat_var=config['input'].get('lat_name', 'latitude'),
        lon_var=config['input'].get('lon_name', 'longitude')
    )

    # Subset time
    ds_ensemble = subset_time(ds_ensemble,
                              config['subset']['start_time'],
                              config['subset']['end_time'])
    print(ds_ensemble)

    ds_ensemble = prepare_ensemble_grid(ds_ensemble,
                                        lat_var=config['input'].get('lat_name', 'latitude'),
                                        lon_var=config['input'].get('lon_name', 'longitude'))
    print(f"Ensemble dataset loaded with dimensions: {ds_ensemble.dims}")

    var_name = config['input'].get('var_name', 'precipitation')

    if var_name not in ds_ensemble.data_vars:
        print(f"Warning: '{var_name}' not found in ensemble dataset. "
              f"Available variables: {list(ds_ensemble.data_vars)}")

    lat_name = config['input'].get('lat_name', 'latitude')
    if lat_name not in ds_ensemble.coords and lat_name not in ds_ensemble.data_vars:
        print(f"Warning: '{lat_name}' not found. Default might be invalid.")


    # Handle ensemble-member selection
    member_list = config['ensemble'].get('members')  # Could be a list or None
    use_mean = config['ensemble'].get('mean', False)

    if member_list:
        print(f"Selecting ensemble members: {member_list}")
        ds_ensemble = ds_ensemble.sel(member=member_list)
        print(ds_ensemble)

    if use_mean:
        print("Computing ensemble mean across selected members...")
        ds_ensemble = ds_ensemble.mean(dim='member', keep_attrs=True)
        print(ds_ensemble)

    # Load reference
    print("Loading reference dataset...")
    ds_ref = load_nc_files(config['input']['reference_file'])

    ds_ref_sub = subset_by_lat_lon(
        ds_ref,
        lat_bounds=config['subset']['lat_bounds'],
        lon_bounds=config['subset']['lon_bounds']
    )
    ds_ref_temp = subset_time(ds_ref_sub,
                         config['subset']['start_time'],
                         config['subset']['end_time'])
    print(f"Reference dataset loaded with dimensions: {ds_ref.dims}")

    ds_ref_prepared = prepare_reference_grid(ds_ref_temp,
                                    lat_var=config['input'].get('ref_lat_name', 'lat'),
                                    lon_var=config['input'].get('ref_lon_name', 'lon'),
                                    dim_lat=config['input'].get('ref_lat_dim', 'y'),
                                    dim_lon=config['input'].get('ref_lon_dim', 'x'))


    # Regrid ensemble to match reference
    print("Regridding ensemble to match reference grid...")
    ds_ensemble_interp = regrid_to_target(ds_ensemble, ds_ref_prepared)
    print("Regridding complete.")
    print(ds_ensemble_interp)

    # Next steps (aggregation, bias metrics, etc.) will be shown in the following sections.
    # Decide the variable name and missing_value from config
    var_name = config['input'].get('var_name', 'precipitation')
    missing_value = config['input'].get('missing_value', -999)

    aggregation_method = config['aggregation'].get('period', 'daily')  
    # possible values: "daily", "monthly", "seasonal", "yearly", "none" (or direct)

    print(f"Performing {aggregation_method} aggregation on ensemble data...")

    if aggregation_method == "daily":
        aggregated_ens = aggregate_to_daily(ds_ensemble_interp, var_name,
                                            missing_value=missing_value)
    elif aggregation_method == "monthly":
        aggregated_ens = aggregate_by_month(ds_ensemble_interp, var_name,
                                            missing_value=missing_value)
    elif aggregation_method == "seasonal":
        aggregated_ens = aggregate_by_season(ds_ensemble_interp, var_name,
                                             missing_value=missing_value)
    elif aggregation_method == "yearly":
        aggregated_ens = aggregate_by_year(ds_ensemble_interp, var_name,
                                           missing_value=missing_value)
    else:
        print("No valid aggregation method selected. Using the data as-is.")
        aggregated_ens = ds_ensemble_interp[var_name]

    print(f"Aggregation complete. Resulting data dims: {aggregated_ens.dims}")

    # Next: we load reference, which may or may not be aggregated similarly:
    ref_var_name = config['input'].get('ref_var_name', 'RR')
    aggregated_ref = None

    if aggregation_method == "daily":
        aggregated_ref = aggregate_to_daily(ds_ref, ref_var_name,
                                            missing_value=missing_value)
    elif aggregation_method == "monthly":
        aggregated_ref = aggregate_by_month(ds_ref, ref_var_name,
                                            missing_value=missing_value)
    elif aggregation_method == "seasonal":
        aggregated_ref = aggregate_by_season(ds_ref, ref_var_name,
                                             missing_value=missing_value)
    elif aggregation_method == "yearly":
        aggregated_ref = aggregate_by_year(ds_ref, ref_var_name,
                                           missing_value=missing_value)
    elif aggregation_method == "weathertype":
        # Pass a weather_type_array from the config or elsewhere
        weather_type_array = ds_ref['weather_type']  # just as an example
        aggregated_ens = group_by_weather_type(ds_ensemble_interp, var_name, weather_type_array)
    else:
        aggregated_ref = ds_ref[ref_var_name]

    print(f"Reference data also aggregated with dims: {aggregated_ref.dims}")


    bias_config = config.get('bias', {})
    requested_metrics = bias_config.get('metrics', ["RMSE"])  # default to RMSE
    dims_to_average = bias_config.get('dimensions', None)     # e.g., ["time"]

    results = {}

    if "RMSE" in requested_metrics:
        print("Computing RMSE...")
        rmse_vals = root_mean_squared_error(aggregated_ens, aggregated_ref,
                                            dim=dims_to_average)
        results["RMSE"] = rmse_vals

    if "MAE" in requested_metrics:
        print("Computing MAE...")
        mae_vals = mean_absolute_error(aggregated_ens, aggregated_ref,
                                       dim=dims_to_average)
        results["MAE"] = mae_vals

    if "ME" in requested_metrics:
        print("Computing Mean Error (ME)...")
        me_vals = mean_error(aggregated_ens, aggregated_ref,
                             dim=dims_to_average)
        results["ME"] = me_vals

    print("Bias metric calculations complete.")

    # You can save or plot them as needed:
    if 'save_netcdf' in config['output']:
        # Example: if only RMSE is selected, results["RMSE"] is an xarray.DataArray
        # If multiple metrics were selected, you may want to combine them into a single Dataset
        combined = xr.Dataset(results)
        combined.to_netcdf(config['output']['save_netcdf'])

    if 'save_plot' in config['output']:
        # As an example, plot the first computed metric:
        metric_to_plot = list(results.keys())[0]
        print(f"Plotting metric: {metric_to_plot}")
        results[metric_to_plot].plot()
        import matplotlib.pyplot as plt
        plt.savefig(config['output']['save_plot'])

def main():
    args = parse_args()

    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Provide minimal default config
        config = {
            'input': {
                'ensemble_pattern': 'data/total_precipitation_20*.nc',
                'reference_file':  'data/INCA_RR_20*.nc',
                'var_name': 'precipitation',
                'ref_var_name': 'RR',
            },
            'subset': {
                'lat_bounds': [46, 48],
                'lon_bounds': [11, 13],
                'start_time': '2017-10-01T00:00:00',
                'end_time':   '2017-10-02T23:00:00'
            },
            'ensemble': {
                'members': None,
                'mean': False
            },
            'aggregation': {
                'period': 'daily'
            },
            'bias': {
                'dimensions': ['time'],
                'metrics': ['RMSE']
            }
        }

    # If --inspect, do that first, then exit
    if args.inspect:
        ds_temp = load_ensemble_files(config['input']['ensemble_pattern'])
        ds_ref_temp = load_nc_files(config['input']['reference_file'])

        print("=== Ensemble Dataset Info ===")
        print(ds_temp)

        print("\n=== Reference Dataset Info ===")
        print(ds_ref_temp)

        sys.exit(0)

    # Overwrite config from CLI if provided
    if args.member:
        config['ensemble']['members'] = args.member
    if args.ensemble_mean:
        config['ensemble']['mean'] = True
    if args.var_name:
        config['input']['var_name'] = args.var_name
    if args.ref_var_name:
        config['input']['ref_var_name'] = args.ref_var_name
    if args.aggregation:
        config['aggregation']['period'] = args.aggregation
    if args.bias_dim:
        config['bias']['dimensions'] = args.bias_dim
    if args.bias_metrics:
        config['bias']['metrics'] = args.bias_metrics

    # Now run analysis once, with updated config
    run_analysis(config)




if __name__ == "__main__":
    main()



