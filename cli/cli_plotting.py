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
    bin_by_altitude,
    groupby_altitude_bins
)
from scripts.plotting import (
    plot_spatial_map,
    plot_time_series_multi_line,
    plot_cycle_multi,
    plot_distribution_multi,
    plot_member_subplots,
    plot_alt_bin_subplots
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run climate analysis + optional plotting")

    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")

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
    parser.add_argument("--aggregation", type=str, choices=["none", "daily", "monthly", "seasonal", "yearly", "none"],
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

#    print("Reference dataset after subsetting:", ds_ref_prepared)

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
        print("Elevation dataset:", ds_dem)
    else:
        ds_dem = None
        alt_bins = []
       
    
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
#    print(ds_ensemble_interp.time.values)
#    print("Ensemble data after regridding:", ds_ensemble_interp)

    # Attach altitude to the datasets
    if ds_dem is not None:
        # if user wants to regrid DEM to reference
        if dem_config.get('regrid_to_reference', False):
            print("Regridding DEM to reference grid before attaching altitude...")
            ds_dem = regrid_to_target(ds_dem, ds_ref_prepared)
            print("Regridded Elevation data:", ds_dem)
        # Now attach altitude to ensemble
        ds_ensemble_interp = add_alt_to_ds(ds_ensemble_interp, ds_dem)
        
        # Also attach altitude to reference
        ds_ref_prepared = add_alt_to_ds(ds_ref_prepared, ds_dem)
        
        print("Altitude successfully attached to ensemble & reference datasets.")
        print("Ensemble data with altitude:", ds_ensemble_interp)

    # Decide the variable name and missing_value from config
    var_name = config['input'].get('var_name', 'precipitation')
    missing_value = config['input'].get('missing_value', -999)

    aggregation_method = config['aggregation'].get('period', 'daily')  
    # possible values: "daily", "monthly", "seasonal", "yearly", "none" (or direct)

    # Aggregation 
    print(f"Performing {aggregation_method} aggregation on ensemble data...")

    aggregated_ens = None

    if aggregation_method == 'none' or aggregation_method == 'hourly':
        aggregated_ens = ds_ensemble_interp[var_name]
    elif aggregation_method == "daily":
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

    print("Aggregation complete.")
    print("Aggregated ensemble data:", aggregated_ens)

    if isinstance(aggregated_ens, xr.DataArray):
        if "altitude" in ds_ensemble_interp:
            aggregated_ens = aggregated_ens.to_dataset(name=var_name)
            aggregated_ens["altitude"] = ds_ensemble_interp["altitude"]
            print("Aggregated ens with altitude", aggregated_ens)

    # Next: we load reference, which may or may not be aggregated similarly:
    ref_var_name = config['input'].get('ref_var_name', 'RR')
    aggregated_ref = None

    if aggregation_method == 'none' or aggregation_method == 'hourly':
        aggregated_ref = ds_ref_prepared[ref_var_name]
    elif aggregation_method == "daily":
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

    print("Reference data also aggregated")
    print("Reference data after aggregation:", aggregated_ref)

    if isinstance(aggregated_ref, xr.DataArray):
        if "altitude" in ds_ref_prepared:
            aggregated_ref = aggregated_ref.to_dataset(name=ref_var_name)
            aggregated_ref["altitude"] = ds_ref_prepared["altitude"]
            print("Aggregated Ref with altitude:", aggregated_ref)

    altbin_config = config.get('altitude_binning', {})
    if altbin_config.get('enabled', False):
        # By default, let's do dimension-based binning, so we can handle alt_bin
        user_bins = altbin_config.get('bins', [0, 500, 1000, 1500, 2000, 3000])
    
        # Convert aggregated_ens (Dataset) into an alt_bin dimension
        aggregated_ens = groupby_altitude_bins(aggregated_ens, alt_var="altitude", bins=user_bins)
        aggregated_ref = groupby_altitude_bins(aggregated_ref, alt_var="altitude", bins=user_bins)

    # Bias Calculations
    bias_config = config.get('bias', {})
    requested_metrics = bias_config.get('metrics', ["RMSE"])  # default to RMSE
    dims_to_average = bias_config.get('dimensions', None)     # e.g., ["time"]

    results = {}

    for metric in requested_metrics:
        if metric == "RMSE":
            rmse_vals = root_mean_squared_error(aggregated_ens,
                                                aggregated_ref,
                                                dim=dims_to_average)
            results["RMSE"] = rmse_vals
        elif metric == "MAE":
            mae_vals = mean_absolute_error(aggregated_ens,
                                           aggregated_ref,
                                           dim=dims_to_average)
            results["MAE"] = mae_vals
        elif metric == "ME":
            me_vals = mean_error(aggregated_ens,
                                 aggregated_ref,
                                 dim=dims_to_average)
            results["ME"] = me_vals

    # (A) If altitude binning is enabled:
    altbin_config = config.get('altitude_binning', {})
    if altbin_config.get('enabled', False) and ds_dem is not None:
        user_bins = altbin_config.get('bins', [0, 500, 1000, 1500, 2000, 3000])

        for metric in requested_metrics:
            da_metric = results[metric]
            # e.g. results["ME"] might be shape (lat_coord, lon_coord)
            # group it by altitude:
            da_binned = groupby_altitude_bins(da_metric, alt_var="altitude", bins=user_bins)
            # Now da_binned has dims: (alt_bin, lat_coord, lon_coord) if lat/lon were kept
            # Replace the old data in results[metric]:
            results[metric] = da_binned

    print("Bias metric calculations complete.")
    print("results:", results)  # debugging

    # return all plotable data
    return results, aggregated_ens, aggregated_ref, ds_ensemble_interp, ds_ref_prepared

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



def run_plots(config, results, aggregated_ens, aggregated_ref):
    """
    Parse the 'plot' section of config and generate requested plots.
    """
    if not config.get('plot', {}).get('enabled', False):
        print("Plotting is disabled in config.")
        return

    plot_cfg = config['plot']

    # EXAMPLE: Spatial map of the first metric
    if plot_cfg.get('spatial_map', {}).get('enabled', False):
        m_cfg = plot_cfg['spatial_map']
        metric_name = m_cfg.get('metric', 'RMSE')
        if metric_name in results:
            out_path = m_cfg.get('out_file', 'output/spatial_map.png')
            plot_spatial_map(results[metric_name],
                             out_path=out_path,
                             title=f"{metric_name} Spatial Map",
                             cmap=m_cfg.get('cmap', 'coolwarm'),
                             vmin=m_cfg.get('vmin', None),
                             vmax=m_cfg.get('vmax', None))
        else:
            print(f"Spatial map skipping: either {metric_name} not in results or not 2D.")

    # EXAMPLE: Time series with optional trend
    if plot_cfg.get('time_series', {}).get('enabled', False):
        ts_cfg = plot_cfg['time_series']
        out_path = ts_cfg.get('out_file', 'output/time_series.png')
        show_trend = ts_cfg.get('show_trend', False)

        # Gather line config
        lines_cfg = ts_cfg.get('lines', [])

        # Pass them to our new multi-line plotting function
        plot_time_series_multi_line(
            lines_cfg=lines_cfg,
            ds_ens=aggregated_ens,
            ds_ref=aggregated_ref,
            out_path=out_path,
            title=ts_cfg.get('title', 'Time Series'),
            ylabel=ts_cfg.get('ylabel', 'mm'),
            show_trend=show_trend,
        )


    # CYCLE - new multi approach
    if plot_cfg.get('cycle', {}).get('enabled', False):
        c_cfg = plot_cfg['cycle']
        out_path = c_cfg.get('out_file', 'output/cycle.png')
        cycle_type = c_cfg.get('type', 'monthly')
        lines_cfg = c_cfg.get('lines', [])

        plot_cycle_multi(
            lines_cfg=lines_cfg,
            ds_ens=aggregated_ens,
            ds_ref=aggregated_ref,
            cycle_type=cycle_type,
            out_path=out_path,
            title=c_cfg.get('title', 'Cycle Plot'),
            ylabel=c_cfg.get('ylabel', 'mm')
        )

    # DISTRIBUTION - new multi approach
    if plot_cfg.get('distribution', {}).get('enabled', False):
        dist_cfg = plot_cfg['distribution']
        out_path = dist_cfg.get('out_file', 'output/distribution.png')
        kind = dist_cfg.get('kind', 'hist')
        bins = dist_cfg.get('bins', 20)
        lines_cfg = dist_cfg.get('lines', [])

        plot_distribution_multi(
            lines_cfg=lines_cfg,
            ds_ens=aggregated_ens,
            ds_ref=aggregated_ref,
            kind=kind,
            bins=bins,
            out_path=out_path,
            title=dist_cfg.get('title', 'Distribution'),
            # Optionally specify x/y labels
            xlabel=dist_cfg.get('xlabel', 'Value'),
            ylabel=dist_cfg.get('ylabel', None)
        )

    # EXAMPLE: side-by-side subplots for multiple ensemble members
    if plot_cfg.get('member_subplots', {}).get('enabled', False):
        ms_cfg = plot_cfg['member_subplots']
        out_path = ms_cfg.get('out_file', 'output/member_subplots.png')
        # Suppose we want to plot the same metric for each member (2D lat-lon, member):
        metric_name = ms_cfg.get('metric', 'RMSE')
        if metric_name in results and 'member' in results[metric_name].dims:
            da_metric = results[metric_name]
            plot_member_subplots(da_metric,
                                 out_path,
                                 title=ms_cfg.get('title', f"{metric_name} by member"),
                                 ncols=ms_cfg.get('ncols', 2),
                                 cmap=ms_cfg.get('cmap', 'coolwarm'),
                                 vmin=ms_cfg.get('vmin', None),
                                 vmax=ms_cfg.get('vmax', None))
        else:
            print(f"Skipping member_subplots: {metric_name} not found or no 'member' dim.")

    # (B) If altitude_bin_subplots is enabled
    if plot_cfg.get('altitude_bin_subplots', {}).get('enabled', False):
        ab_cfg = plot_cfg['altitude_bin_subplots']
        metric_name = ab_cfg.get('metric', 'RMSE')
        out_path = ab_cfg.get('out_file', 'output/rmse_by_altbin.png')

        if metric_name in results:
            # Now results[metric_name] should be a DataArray with dims (alt_bin, lat_coord, lon_coord)
            da_metric = results[metric_name]
            plot_alt_bin_subplots(
                da_metric,
                out_path=out_path,
                title=ab_cfg.get('title', "By-Altitude Subplots"),
                ncols=ab_cfg.get('ncols', 2),
                cmap=ab_cfg.get('cmap', 'viridis'),
                vmin=ab_cfg.get('vmin', None),
                vmax=ab_cfg.get('vmax', None),
                )
        else:
            print(f"[Warning] {metric_name} not in results. Skipping altitude_bin_subplots.")

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 1) Run analysis
    results, aggregated_ens, aggregated_ref, ds_ensemble_interp, ds_ref_prepared = run_analysis(config)

    # 2) Optionally run plotting
    run_plots(config, results, aggregated_ens, aggregated_ref)


if __name__ == "__main__":
    main()

