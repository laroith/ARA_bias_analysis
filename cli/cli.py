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
from scripts.elevation_manager import (
    load_and_subset_dem,
    add_alt_to_ds,
    bin_by_altitude
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

    parser.add_argument("--weather-type-file", type=str,
                        help="Path to a CSV containing weather types (with columns date, slwt, etc.).")
    parser.add_argument("--include-weather-types", type=int, nargs='*',
                        help="List of integer weather types to include (e.g. 6 7).")

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

#    print("Subsetted ensemble dataset:", ds_ensemble)

    ds_ensemble = prepare_ensemble_grid(ds_ensemble,
                                        lat_var=config['input'].get('lat_name', 'latitude'),
                                        lon_var=config['input'].get('lon_name', 'longitude'))

#    print(f"Ensemble dataset loaded with dimensions: {ds_ensemble.dims}")

    var_name = config['input'].get('var_name', 'precipitation')

    if var_name not in ds_ensemble.data_vars:
        print(f"Warning: '{var_name}' not found in ensemble dataset. "
              f"Available variables: {list(ds_ensemble.data_vars)}")

    lat_name = config['input'].get('lat_name', 'latitude')
#    if lat_name not in ds_ensemble.coords and lat_name not in ds_ensemble.data_vars:
#        print(f"Warning: '{lat_name}' not found. Default might be invalid.")


    # Handle ensemble-member selection
    member_list = config['ensemble'].get('members')  # Could be a list or None
    use_mean = config['ensemble'].get('mean', False)

    if member_list:
        print(f"Selecting ensemble members: {member_list}")
        ds_ensemble = ds_ensemble.sel(member=member_list)
#        print(ds_ensemble)

    if use_mean:
        print("Computing ensemble mean across selected members...")
        ds_ensemble = ds_ensemble.mean(dim='member', keep_attrs=True)
#        print(ds_ensemble)

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
#    print(f"Reference dataset loaded with dimensions: {ds_ref.dims}")

    ds_ref_prepared = prepare_reference_grid(ds_ref_temp,
                                    lat_var=config['input'].get('ref_lat_name', 'lat'),
                                    lon_var=config['input'].get('ref_lon_name', 'lon'),
                                    dim_lat=config['input'].get('ref_lat_dim', 'y'),
                                    dim_lon=config['input'].get('ref_lon_dim', 'x'))

    print("Reference dataset after subsetting:", ds_ref_prepared)

    # If the user has 'dem' block in config and enabled it, we load the DEM
    dem_config = config.get('dem', {})
    dem_enabled = dem_config.get('enabled', False)
    
    if dem_enabled:
        print("DEM support is enabled. Loading DEM dataset...")
        dem_path = dem_config['dem_path']
        dem_var = dem_config.get('dem_var', 'ZH')
        
        # e.g. from config: lat_var='lat', lon_var='lon', dim_lat='y', dim_lon='x'
        lat_var = dem_config.get('lat_var', 'lat')
        lon_var = dem_config.get('lon_var', 'lon')
        dim_lat = dem_config.get('dim_lat', 'y')
        dim_lon = dem_config.get('dim_lon', 'x')
        
        # Bins could also be in dem_config. We'll handle them later for binning:
        alt_bins = dem_config.get('bins', [0, 500, 1000, 1500, 2000, 3000])
        
        # Load DEM
        ds_dem = load_and_subset_dem(
            dem_path,
            dem_var=dem_var,
            lat_bounds=config['subset']['lat_bounds'],
            lon_bounds=config['subset']['lon_bounds'],
            lat_var=lat_var,
            lon_var=lon_var,
            dim_lat=dim_lat,
            dim_lon=dim_lon
        )

    else:
        ds_dem = None
        alt_bins = []
    ### END NEW DEM LOADING
    print("Altitude dataset:", ds_dem)

    # weather type filtering
    weather_config = config.get('weather_types', {})
    if 'file' in weather_config and 'include' in weather_config and weather_config['include']:
        from scripts.weather_filter import load_weather_types_csv, filter_by_weather_types
        print(f"Filtering datasets by weather types: {weather_config['include']}")
        # For example, if your CSV has columns date, slwt
        wt_da = load_weather_types_csv(csv_path=weather_config['file'],
                                       date_col='date',
                                       wt_col='slwt')
        ds_ensemble = filter_by_weather_types(ds_ensemble, wt_da, include_types=weather_config['include']) 
        print("Ensemble dataset after weather type filtering:\n", ds_ensemble)
        ds_ref_prepared = filter_by_weather_types(ds_ref_prepared, wt_da, include_types=weather_config['include']) 
        print("Reference dataset after weather type filtering:\n", ds_ref_prepared)

    # Regrid ensemble to match reference
    print("Regridding ensemble to match reference grid...")
    ds_ensemble_interp = regrid_to_target(ds_ensemble, ds_ref_prepared)
    print("Regridding complete.")
    print(ds_ensemble_interp.time.values)
    print("Ensemble data after regridding:", ds_ensemble_interp)

    # Attach altitude to the datasets
    if ds_dem is not None:
        # if user wants to regrid DEM to reference
        if dem_config.get('regrid_to_reference', False):
            print("Regridding DEM to reference grid before attaching altitude...")
            ds_dem = regrid_to_target(ds_dem, ds_ref_prepared)
        
        # Now attach altitude to ensemble
        ds_ensemble_interp = add_alt_to_ds(ds_ensemble_interp, ds_dem)
        
        # Also attach altitude to reference
        ds_ref_prepared = add_alt_to_ds(ds_ref_prepared, ds_dem)
        
        print("Altitude successfully attached to ensemble & reference datasets.")

    print("Ensemble data with altitude:", ds_ensemble_interp)

    # Next steps (aggregation, bias metrics, etc.) will be shown in the following sections.
    # Decide the variable name and missing_value from config
    var_name = config['input'].get('var_name', 'precipitation')
    missing_value = config['input'].get('missing_value', -999)

    aggregation_method = config['aggregation'].get('period', 'daily')  
    # possible values: "daily", "monthly", "seasonal", "yearly", "none" (or direct)
    print(f"Performing {aggregation_method} aggregation on ensemble data...")

    aggregated_ens = None

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
    print(aggregated_ens)

    # Next: we load reference, which may or may not be aggregated similarly:
    ref_var_name = config['input'].get('ref_var_name', 'RR')
    aggregated_ref = None

    if aggregation_method == "daily":
        aggregated_ref = aggregate_to_daily(ds_ref_prepared, ref_var_name,
                                            missing_value=missing_value)
    elif aggregation_method == "monthly":
        aggregated_ref = aggregate_by_month(ds_ref_prepared, ref_var_name,
                                            missing_value=missing_value)
    elif aggregation_method == "seasonal":
        aggregated_ref = aggregate_by_season(ds_ref_prepared, ref_var_name,
                                             missing_value=missing_value)
    elif aggregation_method == "yearly":
        aggregated_ref = aggregate_by_year(ds_ref_prepared, ref_var_name,
                                           missing_value=missing_value)
    else:
        aggregated_ref = ds_ref_prepared[ref_var_name]

    print(f"Reference data also aggregated with dims: {aggregated_ref.dims}")
    print(aggregated_ref)

    ### NEW: BIN BY ALTITUDE (IF ENABLED)
    altbin_config = config.get('altitude_binning', {})
    bins_enabled = altbin_config.get('enabled', False) and ds_dem is not None
    bin_results = {}  # We'll store the bin-based metrics here

    if bins_enabled:
        print("Altitude binning is enabled. Binning the aggregated data.")
        user_bins = altbin_config.get('bins', [0, 500, 1000, 1500, 2000, 3000])
    
        # bin_by_altitude expects a Dataset with altitude, so ensure aggregated_ens and aggregated_ref
        # are still Datasets, not DataArrays. If they are DataArrays, convert them to a Dataset or
        # rework aggregator to always return a Dataset.

        ens_binned_dict = bin_by_altitude(aggregated_ens, alt_var="altitude", bins=user_bins)
        ref_binned_dict = bin_by_altitude(aggregated_ref, alt_var="altitude", bins=user_bins)

        # Now each of these is a dict: {(min_bin, max_bin): subset_dataset}

    print("Ensemble dataset with altitude bins:", ens_binned_dict)

    bias_config = config.get('bias', {})
    requested_metrics = bias_config.get('metrics', ["RMSE"])  # default to RMSE
    dims_to_average = bias_config.get('dimensions', None)     # e.g., ["time"]

    results = {}

    # domain-wide
    for metric in requested_metrics:
        if metric == "RMSE":
            rmse_vals = root_mean_squared_error(aggregated_ens[var_name],
                                                aggregated_ref[ref_var_name],
                                                dim=dims_to_average)
            results["RMSE"] = rmse_vals
        elif metric == "MAE":
            mae_vals = mean_absolute_error(aggregated_ens[var_name],
                                           aggregated_ref[ref_var_name],
                                           dim=dims_to_average)
            results["MAE"] = mae_vals
        elif metric == "ME":
            me_vals = mean_error(aggregated_ens[var_name],
                                 aggregated_ref[ref_var_name],
                                 dim=dims_to_average)
            results["ME"] = me_vals

    ### NEW BIN LOGIC ###
    if bins_enabled:
        print("Computing bias metrics in each altitude bin.")
        # e.g. results_bins = { (min_bin, max_bin): { "RMSE": <da>, "MAE": <da> } }
        results_bins = {}
        for bin_range, ds_ens_bin in ens_binned_dict.items():
            ds_ref_bin = ref_binned_dict[bin_range]
            # create an inner dict for metrics in this bin
            bin_metrics = {}
        
            # each subset is still a dataset with "altitude", plus [time, lat_coord, lon_coord,...]
            for metric in requested_metrics:
                if metric == "RMSE":
                    rmse_da = root_mean_squared_error(
                        ds_ens_bin[var_name],
                        ds_ref_bin[ref_var_name],
                        dim=dims_to_average
                    )
                    bin_metrics["RMSE"] = rmse_da
                elif metric == "MAE":
                    mae_da = mean_absolute_error(
                        ds_ens_bin[var_name],
                        ds_ref_bin[ref_var_name],
                        dim=dims_to_average
                    )
                    bin_metrics["MAE"] = mae_da
                elif metric == "ME":
                    me_da = mean_error(
                        ds_ens_bin[var_name],
                        ds_ref_bin[ref_var_name],
                        dim=dims_to_average
                    )
                    bin_metrics["ME"] = me_da
        
            results_bins[bin_range] = bin_metrics

        # store the bin-based results in 'results' or a separate top-level var
        results["altitude_bins"] = results_bins

    print("Bias metric calculations complete.")

    # You can save or plot them as needed:
    if 'save_netcdf' in config['output']:
        combined = xr.Dataset()
        # domain-wide metrics:
        for k,v in results.items():
            if k == "altitude_bins":
                # skip or handle separately
                continue
            combined[k] = v
        # you may want to convert bin-based metrics to a single dataset if they are all the same shape
        # or else you store them separately
        combined.to_netcdf(config['output']['save_netcdf'])

    if 'save_plot' in config['output'] and config['output']['save_plot']:
        # This means we do some plotting.
        # Next, check if we want to plot altitude bins:
        plot_bin_config = config['output'].get('plot_altitude_bins', {})
        if plot_bin_config.get('enabled', False):
            print("Plotting altitude-bin-based metrics...")

            metric_to_plot = plot_bin_config.get('metric', 'RMSE')
            bins_to_plot = plot_bin_config.get('bins_to_plot', [])
            filename_pattern = plot_bin_config.get('filename_pattern', "bin_{min}_{max}.png")

            # We assume results["altitude_bins"] exists from earlier:
            if "altitude_bins" not in results:
                print("No altitude bin metrics found in 'results'. Skipping bin plotting.")
            else:
                # Let's loop over the user-specified bin ranges
                for bin_range in bins_to_plot:
                    bin_range_tuple = tuple(bin_range)  # e.g., [0, 500] -> (0, 500)
                    if bin_range_tuple not in results["altitude_bins"]:
                        print(f"No data for bin range {bin_range_tuple}. Skipping.")
                        continue
                    bin_metrics_dict = results["altitude_bins"][bin_range_tuple]

                    if metric_to_plot not in bin_metrics_dict:
                        print(f"Metric '{metric_to_plot}' not found in bin {bin_range_tuple}. Skipping.")
                        continue

                    da_metric = bin_metrics_dict[metric_to_plot]
                    if da_metric is None:
                        print(f"No metric data for bin {bin_range_tuple}, metric={metric_to_plot}.")
                        continue

                    # Now do the actual plotting
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(6, 4))
                    da_metric.plot(cmap="viridis")
                    plt.title(f"{metric_to_plot} for altitude bin {bin_range_tuple}")
                
                    out_fname = filename_pattern.format(
                        min=bin_range_tuple[0],
                        max=bin_range_tuple[1]
                    )
                    plt.savefig(out_fname, dpi=150)
                    plt.close()
                    print(f"Saved plot '{out_fname}' for bin {bin_range_tuple}")
        else:
            # Optionally do domain-wide plotting (like your old logic):
            if len(results) > 0:
                # For example, pick the first top-level metric to plot
                top_metrics = [k for k in results.keys() if k != "altitude_bins"]
                if len(top_metrics) > 0:
                    metric_to_plot = top_metrics[0]
                    da_metric = results[metric_to_plot]
                    if hasattr(da_metric, "plot"):
                        print(f"Plotting top-level metric: {metric_to_plot}")
                        da_metric.plot()
                        import matplotlib.pyplot as plt
                        plt.savefig("output/domain_wide_metric.png")
                        plt.close()


#    if 'save_plot' in config['output']:
        # As an example, plot the first computed metric:
#        metric_to_plot = list(results.keys())[0]
#        print(f"Plotting metric: {metric_to_plot}")
#        results[metric_to_plot].plot()
#        import matplotlib.pyplot as plt
#        plt.savefig(config['output']['save_plot'])

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
                'end_time':   '2017-10-02T12:00:00'
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

    if args.weather_type_file:
        # Create or update a 'weather_types' subsection in config
        if 'weather_types' not in config:
            config['weather_types'] = {}
        config['weather_types']['file'] = args.weather_type_file

    if args.include_weather_types:
        if 'weather_types' not in config:
            config['weather_types'] = {}
        config['weather_types']['include'] = args.include_weather_types

    # Now run analysis once, with updated config
    run_analysis(config)




if __name__ == "__main__":
    main()



