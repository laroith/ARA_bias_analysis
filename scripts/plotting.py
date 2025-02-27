# scripts/plotting.py

"""
Module: plotting.py

Provides modular plotting routines using Matplotlib and (optionally) Cartopy for:
  - Spatial maps (with coastlines, borders, lakes)
  - Time series (with optional linear trend)
  - Cycles (diurnal, seasonal, monthly)
  - Distribution plots (histogram, boxplot)
  - Subplots for comparing multiple ensemble members/bins side by side

All functions assume your dataset has coordinates named 'lat_coord' and 'lon_coord'
for 2D spatial plots, and 'time' for time-based plots.

Usage:
  from scripts.plotting import (
      plot_spatial_map, plot_time_series, plot_cycle, plot_distribution,
      plot_member_subplots
  )
"""

import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# For coastlines, borders, etc.
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# OPTIONAL: If you have a local ECMWF .mplstyle, load it:
# plt.style.use("ecmwf_style.mplstyle")

# Otherwise, set some defaults here:
matplotlib.rcParams.update({
    'figure.figsize': (8, 6),
    'axes.grid': True,
    'axes.labelsize': 12,
    'font.size': 11
})


def plot_spatial_map(da,
                     out_path,
                     title=None,
                     cmap='viridis',
                     vmin=None,
                     vmax=None,
                     show_coastlines=True,
                     show_borders=True,
                     show_lakes=True,
                     subplot_ax=None,
                     **kwargs):
    """
    Plot a 2D spatial field (lat_coord vs. lon_coord) with Cartopy features.

    Parameters
    ----------
    da : xarray.DataArray
        2D data with coordinates: da[lat_coord, lon_coord].
    out_path : str
        File path to save the resulting figure (e.g. "output/spatial_map.png").
        If subplot_ax is given, out_path can be None or optional.
    title : str, optional
        Plot title.
    cmap : str, optional
        Colormap (default 'viridis').
    vmin, vmax : float, optional
        Color scale limits.
    show_coastlines : bool, optional
        Whether to show coastlines (default True).
    show_borders : bool, optional
        Whether to show country borders (default True).
    show_lakes : bool, optional
        Whether to show lakes (Cartopy feature) (default True).
    subplot_ax : matplotlib Axes, optional
        If provided, plot into this subplot. Otherwise create a new figure.
    **kwargs : dict
        Additional arguments passed to da.plot().

    Returns
    -------
    None (saves figure or draws on subplot).
    """
    #if 'member' is present, squeeze it if it is size=1:
    if 'member' in da.dims:
        if da.sizes['member'] == 1:
            da = da.squeeze('member', drop=True)
        else:
            raise ValueError(
                "plot_spatial_map expects 2D data but found multiple members; "
                "use plot_member_subplots instead!"
            )

    # If not using subplots, create a new figure
    if subplot_ax is None:
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
    else:
        ax = subplot_ax
        ax.set_global()
        ax.set_projection(ccrs.PlateCarree())

    
    # Plot
    p = da.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                transform=ccrs.PlateCarree(),  # data are in lat/lon coords
                **kwargs)

    # Add features
    if show_coastlines:
        ax.coastlines(resolution='50m', color='black')
    if show_borders:
        ax.add_feature(cfeature.BORDERS, edgecolor='black')
    if show_lakes:
        ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none')

    # Set axis labels (Cartopy doesn't label lat/lon automatically)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if title:
        ax.set_title(title)

    # If we made a new figure, save and close
    if subplot_ax is None:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[plot_spatial_map] Saved figure to {out_path}")


import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

def plot_time_series_multi_line(lines_cfg,
                                ds_ens,
                                ds_ref,
                                out_path,
                                title=None,
                                ylabel=None,
                                show_trend=False):
    """
    Plot multiple lines from different sources/datasets on the same figure.

    If both 'member' and 'alt_bin' remain, we do a nested loop, 
    so each line is (bin=..., mem=...).

    Config usage (YAML example):
        lines:
          - source: "ensemble"
            variable: "precipitation"
            member_selection: "all"  # or "mean", or single name
            alt_bin_selection: ["(500.0, 1000.0]", "(1000.0, 1500.0]"]  # or "all", "mean"
            reduce_dims: ["lat_coord","lon_coord"]  # domain mean
            color: "blue"
            label: "Ens alt-bins"

    The final dimension for time-series must include 'time' 
    but can also have leftover 'member' or 'alt_bin' if user wants multiple lines.
    Parameters
    ----------
    lines_cfg : list of dict
        Each item must specify:
          - 'source': which dataset to pull from (e.g. "ensemble" or "reference")
          - 'variable': which variable name (e.g. "precip" or "RR")
          - member_selection: e.g. "all", "mean", or ["00","01"]
          - alt_bin_selection: e.g. "all", "mean", or ["(500.0, 1000.0]", ...]
          - reduce_dims: dims to average if desired
          - optionally 'label', 'color', 'reduce_dims', etc.
    ds_ens : xarray.Dataset
        The aggregated ensemble dataset (passed in from CLI).
    ds_ref : xarray.Dataset
        The aggregated reference dataset.
    out_path : str
        Where to save the figure.
    title : str, optional
        Figure title.
    ylabel : str, optional
        Y‐axis label.
    show_trend : bool, optional
        Whether to draw a simple linear trend line for each data series.

    Returns
    -------
    None (the figure is saved to out_path).
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    for line_cfg in lines_cfg:
        # 1) Extract line config
        source_key = line_cfg.get('source', 'ensemble')  # "ensemble" or "reference"
        var_name = line_cfg.get('variable')
        base_color = line_cfg.get('color', None)
        label = line_cfg.get('label', f"{source_key}-{var_name}")
        reduce_dims = line_cfg.get('reduce_dims', [])
        member_sel = line_cfg.get('member_selection', None)
        alt_bin_sel = line_cfg.get('alt_bin_selection', None)

        # 2) Pick the actual dataset or dataarray
        if source_key == 'ensemble':
            ds = ds_ens
        elif source_key == 'reference':
            ds = ds_ref
        else:
            raise ValueError(f"Unknown data source: {source_key}")

        # 3) Check if ds is a Dataset or a DataArray
        if isinstance(ds, xr.Dataset):
            # membership check in data_vars
            if var_name not in ds.data_vars:
                print(f"[Warning] Variable '{var_name}' not found in {source_key} dataset. Skipping.")
                continue
            da_line = ds[var_name]

        elif isinstance(ds, xr.DataArray):
            # It's already a single DataArray
            da_line = ds
            # (Optional) Check name if you want:
            # if da_line.name != var_name:
            #    print(f"[Warning] data_array name '{da_line.name}' != config var '{var_name}'")
        else:
            # If it's neither, we can't proceed
            print(f"[Warning] {source_key} is neither Dataset nor DataArray. Skipping.")
            continue

        # 4) alt_bin selection (like 'member')
        if alt_bin_sel is not None:
            if isinstance(alt_bin_sel, str):
                if alt_bin_sel == "all":
                    pass
                elif alt_bin_sel == "mean" and "alt_bin" in da_line.dims:
                    da_line = da_line.mean(dim="alt_bin")
                else:
                    # Single bin label
                    da_line = da_line.sel(alt_bin=alt_bin_sel)
            elif isinstance(alt_bin_sel, list):
                da_line = da_line.sel(alt_bin=alt_bin_sel)
            else:
                raise ValueError(f"Unknown alt_bin_selection: {alt_bin_sel}")


        # 5) Member selection logic
        if member_sel is not None:
            if isinstance(member_sel, str):
                if member_sel == "all":
                    pass
                elif member_sel == "mean":
                    if "member" in da_line.dims:
                        da_line = da_line.mean(dim="member")
                else:
                    da_line = da_line.sel(member=member_sel)
            elif isinstance(member_sel, list):
                da_line = da_line.sel(member=member_sel)
            else:
                raise ValueError(f"Unknown member_selection: {member_sel}")

        # --- 4) reduce_dims => average over user-specified dims ---
        if reduce_dims:
            da_line = da_line.mean(dim=reduce_dims)
        else:
            # Fallback: average all leftover dims EXCEPT time, member
            leftover_dims = [d for d in da_line.dims if d not in ("time","alt_bin","member")]
            if leftover_dims:
                da_line = da_line.mean(dim=leftover_dims)

        # Now we produce lines. If alt_bin & member both remain, do nested loops
        if "alt_bin" in da_line.dims and "member" in da_line.dims:
            for bin_val in da_line.alt_bin.values:
                sub_da_bin = da_line.sel(alt_bin=bin_val)
                for mem_val in sub_da_bin.member.values:
                    sub_da_mem = sub_da_bin.sel(member=mem_val)
                    line_label = f"{label} (bin={bin_val}, mem={mem_val})"
                    _plot_time_series_one_line(sub_da_mem, line_label, ax, show_trend, base_color)
        elif "alt_bin" in da_line.dims:
            # Single loop over alt_bin
            for bin_val in da_line.alt_bin.values:
                sub_da_bin = da_line.sel(alt_bin=bin_val)
                line_label = f"{label} (bin={bin_val})"
                _plot_time_series_one_line(sub_da_bin, line_label, ax, show_trend, base_color)
        elif "member" in da_line.dims:
            # Single loop over member
            for mem_val in da_line.member.values:
                sub_da_mem = da_line.sel(member=mem_val)
                line_label = f"{label} (mem={mem_val})"
                _plot_time_series_one_line(sub_da_mem, line_label, ax, show_trend, base_color)
        else:
            # single line
            _plot_time_series_one_line(da_line, label, ax, show_trend, base_color)


    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel if ylabel else "Value")
    if title:
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot_time_series_multi_line] Saved figure to {out_path}")

def _plot_time_series_one_line(da_line, line_label, ax, show_trend, color):
    """
    Helper that expects a 1D time dimension and plots a single line. 
    Optionally adds a linear trend.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # We assume time is the only dimension left
    if "time" not in da_line.dims:
        print(f"[Warning] No 'time' dimension for {line_label}, skipping line.")
        return

    time_vals = da_line["time"].values
    y = da_line.values

    ax.plot(time_vals, y, label=line_label, color=color)

    if show_trend:
        time_numeric = time_vals.astype('datetime64[D]').astype(float)
        mask = ~np.isnan(y)
        if mask.sum() >= 2:
            slope, intercept = np.polyfit(time_numeric[mask], y[mask], 1)
            y_fit = slope * time_numeric + intercept
            ax.plot(time_vals, y_fit, linestyle='--', color='black')


def plot_cycle_multi(lines_cfg,
                     ds_ens,
                     ds_ref,
                     cycle_type="monthly",
                     out_path="output/cycle.png",
                     title=None,
                     ylabel=None):
    """
    Plot multiple "cycle" lines (diurnal, daily, monthly, seasonal, annual) 
    from different datasets.

    For each entry in lines_cfg:
      - source: "ensemble" or "reference"
      - variable: e.g. "precip"
      - member_selection: "all", "mean", or a list like ["00","01"]
      - reduce_dims: which dims to average out (optional)
      - color, label, etc.
    If after these selections there's still a "member" dim, we loop over each 
    member separately, so each line is plotted and labeled (mem 00, mem 01, etc.).

    Recognized cycle_type options: "diurnal", "daily", "monthly", "seasonal", "annual".
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr

    fig, ax = plt.subplots(figsize=(8, 5))

    for line_cfg in lines_cfg:
        # --- 1) Basic parameters ---
        source_key = line_cfg.get('source', 'ensemble')
        var_name = line_cfg.get('variable')
        reduce_dims = line_cfg.get('reduce_dims', [])
        color = line_cfg.get('color', None)
        base_label = line_cfg.get('label', f"{source_key}-{var_name}")
        member_sel = line_cfg.get('member_selection', None)
        alt_bin_sel = line_cfg.get('alt_bin_selection', None)

        # --- 2) Pick the dataset or dataarray ---
        if source_key == 'ensemble':
            ds = ds_ens
        elif source_key == 'reference':
            ds = ds_ref
        else:
            print(f"[Warning] Unknown source '{source_key}' in cycle lines config.")
            continue

        # If ds is a Dataset, extract var_name
        if isinstance(ds, xr.Dataset):
            if var_name not in ds.data_vars:
                print(f"[Warning] '{var_name}' not found in {source_key} dataset. Skipping.")
                continue
            da = ds[var_name]
        elif isinstance(ds, xr.DataArray):
            da = ds
        else:
            print(f"[Warning] {source_key} is neither Dataset nor DataArray. Skipping.")
            continue

        # (B) alt_bin selection logic
        if alt_bin_sel is not None:
            if isinstance(alt_bin_sel, str):
                if alt_bin_sel=="all":
                    pass
                elif alt_bin_sel=="mean" and "alt_bin" in da.dims:
                    da = da.mean(dim="alt_bin")
                else:
                    # single bin label
                    da = da.sel(alt_bin=alt_bin_sel)
            elif isinstance(alt_bin_sel, list):
                da = da.sel(alt_bin=alt_bin_sel)
            else:
                raise ValueError(f"Unknown alt_bin_selection: {alt_bin_sel}")

        # --- 3) Member selection logic ---
        if member_sel is not None:
            if isinstance(member_sel, str):
                if member_sel == "all":
                    pass
                elif member_sel == "mean":
                    if "member" in da.dims:
                        da = da.mean(dim="member")
                else:
                    da = da.sel(member=member_sel)
            elif isinstance(member_sel, list):
                da = da.sel(member=member_sel)
            else:
                raise ValueError(f"Unknown member_selection: {member_sel}")

        print(da.dims)
        # --- 4) reduce_dims => average out user-specified dims ---
        if reduce_dims:
            da = da.mean(dim=reduce_dims)
            print(da)
        else:
            # fallback: average over everything except time
            leftover = [d for d in da.dims if d not in ("time","member","alt_bin")]
            if leftover:
                da = da.mean(dim=leftover)

        # --- 5) If there's a "member"or "alt_bin" dimension left, we plot each one separately ---
        if "alt_bin" in da.dims and "member" in da.dims:
            # double loop
            for bin_val in da.alt_bin.values:
                sub_da_bin = da.sel(alt_bin=bin_val)
                for mem_val in sub_da_bin.member.values:
                    sub_da = sub_da_bin.sel(member=mem_val)
                    lab = f"{base_label} (bin={bin_val}, mem={mem_val})"
                    xvals, yvals = _group_cycle_data(sub_da, cycle_type)
                    ax.plot(xvals, yvals, label=lab, color=color)
        elif "alt_bin" in da.dims:
            # single loop alt_bin
            for bin_val in da.alt_bin.values:
                sub_da = da.sel(alt_bin=bin_val)
                lab = f"{base_label} (bin={bin_val})"
                xvals, yvals = _group_cycle_data(sub_da, cycle_type)
                ax.plot(xvals, yvals, label=lab, color=color)
        elif "member" in da.dims:
            for mem_val in da.member.values:
                sub_da = da.sel(member=mem_val)
                mem_label = f"{base_label} (mem {mem_val})"

                xvals, yvals = _group_cycle_data(sub_da, cycle_type)
                # Plot
                if cycle_type == "seasonal":
                    ax.bar(xvals, yvals, label=mem_label, color=color, alpha=0.5)
                else:
                    ax.plot(xvals, yvals, label=mem_label, color=color)
        else:
            # Single line (no member dim)
            xvals, yvals = _group_cycle_data(da, cycle_type)
            if cycle_type == "seasonal":
                ax.bar(xvals, yvals, label=base_label, color=color, alpha=0.5)
            else:
                ax.plot(xvals, yvals, label=base_label, color=color)

    # --- 6) Finalize axes and save ---
    _setup_cycle_axes(ax, cycle_type, ylabel, title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot_cycle_multi] Saved figure to {out_path}")


def _group_cycle_data(da, cycle_type):
    """
    Groups 'da' by the given cycle_type and returns (xvals, yvals).
    Supported: 'diurnal', 'daily', 'monthly', 'seasonal', 'annual'.

    diurnal => groupby time.hour (0..23)
    daily   => groupby time.dayofyear (1..365/366)
    monthly => groupby time.month (1..12)
    seasonal=> groupby time.season (DJF/MAM/JJA/SON)
    annual  => groupby time.year
    """
    import numpy as np

    if cycle_type == "diurnal":
        cycle_data = da.groupby("time.hour").mean()
        xvals = cycle_data["hour"].values
        yvals = cycle_data.values
    elif cycle_type == "daily":
        cycle_data = da.groupby("time.dayofyear").mean()
        xvals = cycle_data["dayofyear"].values
        yvals = cycle_data.values
    elif cycle_type == "monthly":
        cycle_data = da.groupby("time.month").mean()
        xvals = cycle_data["month"].values
        yvals = cycle_data.values
    elif cycle_type == "seasonal":
        cycle_data = da.groupby("time.season").mean()
        xvals = cycle_data["season"].values
        yvals = cycle_data.values
    elif cycle_type == "annual":
        cycle_data = da.groupby("time.year").mean()
        xvals = cycle_data["year"].values
        yvals = cycle_data.values
    else:
        print(f"[Warning] Unknown cycle_type '{cycle_type}'. Must be one of "
              f"diurnal/daily/monthly/seasonal/annual.")
        return [], []
    return xvals, yvals


def _setup_cycle_axes(ax, cycle_type, ylabel, title):
    """
    Helper to label x-axis depending on cycle_type, plus set y-label/title if given.
    """
    import numpy as np

    if cycle_type == "diurnal":
        ax.set_xticks(np.arange(0, 24))
        ax.set_xlabel("Hour of Day")
    elif cycle_type == "daily":
        ax.set_xlabel("Day of Year")
    elif cycle_type == "monthly":
        ax.set_xticks(np.arange(1, 13))
        ax.set_xlabel("Month")
    elif cycle_type == "seasonal":
        ax.set_xlabel("Season")
    elif cycle_type == "annual":
        ax.set_xlabel("Year")

    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)



def plot_distribution_multi(lines_cfg,
                            ds_ens,
                            ds_ref,
                            kind="hist",
                            bins=20,
                            out_path="output/distribution.png",
                            title=None,
                            xlabel=None,
                            ylabel=None):
    """
    Plot multiple distributions from different datasets in one figure.
    
    This updated version can also handle an 'alt_bin' dimension
    (created by groupby_altitude_bins or similar).
    
    The logic now mirrors 'member' logic: we parse 'alt_bin_selection'
    from lines_cfg, apply selection or averaging, and if 'alt_bin'
    remains, we loop over each bin for separate distributions.

    [CHANGE: The new alt_bin handling is marked below!]

    Parameters
    ----------
    lines_cfg : list of dict
        Each item might have:
          - source: "ensemble" or "reference"
          - variable: "precip" etc.
          - member_selection: e.g. "all", "mean", ["00","01"] ...
          - alt_bin_selection: e.g. "all", "mean", ["(500.0, 1000.0]","(1000.0, 2000.0]"], or single bin
          - reduce_dims: dims to collapse
          - color, label, alpha ...
        etc.
    ds_ens : xarray.Dataset
    ds_ref : xarray.Dataset
    kind : str
        "hist" or "kde" or possibly other distribution types
    bins : int
        # of bins if 'hist'
    out_path : str
        Where to save the figure
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(8,5))

    for line_cfg in lines_cfg:
        source = line_cfg.get('source', 'ensemble')
        var_name = line_cfg.get('variable')
        reduce_dims = line_cfg.get('reduce_dims', [])
        color = line_cfg.get('color', None)
        label = line_cfg.get('label', f"{source}-{var_name}")
        alpha = line_cfg.get('alpha', 0.5)
        member_sel = line_cfg.get('member_selection', None)
        alt_bin_sel = line_cfg.get('alt_bin_selection', None)

        # Pick dataset
        if source == 'ensemble':
            ds = ds_ens
        elif source == 'reference':
            ds = ds_ref
        else:
            print(f"[Warning] Unknown source '{source}' in distribution lines config.")
            continue

        if isinstance(ds, xr.Dataset):
            if var_name not in ds.data_vars:
                print(f"[Warning] Variable '{var_name}' not in {source} dataset. Skipping.")
                continue
            da = ds[var_name]
        elif isinstance(ds, xr.DataArray):
            da = ds
        else:
            print(f"[Warning] {source} is neither Dataset nor DataArray. Skipping.")
            continue


        # 1) Member selection logic
        member_sel = line_cfg.get('member_selection', None)
        if member_sel is not None:
            if isinstance(member_sel, str):
                if member_sel == "all":
                    pass  # do nothing, keep all members
                elif member_sel == "mean":
                    if "member" in da.dims:
                        da = da.mean(dim="member")
                else:
                    # If it's a single member name like "00", you could do:
                    da = da.sel(member=member_sel)
            elif isinstance(member_sel, list):
                # e.g. ["00","01"]
                da = da.sel(member=member_sel)
            else:
                raise ValueError(f"Unknown member_selection: {member_sel}")


        # (B) alt_bin selection logic
        if alt_bin_sel is not None:
            if isinstance(alt_bin_sel, str):
                if alt_bin_sel=="all":
                    pass
                elif alt_bin_sel=="mean" and "alt_bin" in da.dims:
                    da = da.mean(dim="alt_bin")
                else:
                    # single bin label
                    da = da.sel(alt_bin=alt_bin_sel)
            elif isinstance(alt_bin_sel, list):
                da = da.sel(alt_bin=alt_bin_sel)
            else:
                raise ValueError(f"Unknown alt_bin_selection: {alt_bin_sel}")

        # Reduce dims (default to flatten everything if not provided)
        if reduce_dims:
            da = da.mean(dim=reduce_dims)
        else:
            pass

        if "alt_bin" in da.dims and "member" in da.dims:
            for bin_val in da.alt_bin.values:
                sub_da_bin = da.sel(alt_bin=bin_val)
                for mem_val in sub_da_bin.member.values:
                    sub_da_mem = sub_da_bin.sel(member=mem_val)
                    dist_label = f"{label} (bin={bin_val}, mem={mem_val})"
                    _plot_distribution_1d(sub_da_mem, dist_label, kind, bins, alpha, color, ax)
        elif "alt_bin" in da.dims:
            # Single loop over each alt_bin
            for bin_val in da.alt_bin.values:
                sub_da_bin = da.sel(alt_bin=bin_val)
                dist_label = f"{label} (bin={bin_val})"
                _plot_distribution_1d(sub_da_bin, dist_label, kind, bins, alpha, color, ax)
        elif "member" in da.dims:
            # Single loop over each member
            for mem_val in da.member.values:
                sub_da_mem = da.sel(member=mem_val)
                dist_label = f"{label} (mem={mem_val})"
                _plot_distribution_1d(sub_da_mem, dist_label, kind, bins, alpha, color, ax)
        else:
            # Single distribution
            _plot_distribution_1d(da, label, kind, bins, alpha, color, ax)


    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot_distribution_multi] Saved figure to {out_path}")

def _to_1d_array(da):
    """
    Flatten an xarray.DataArray to 1D and drop NaNs.
    """
    # either stack or simply do .values.ravel():
    arr = da.values.ravel()
    return arr[~np.isnan(arr)]

def _plot_distribution_1d(da, dist_label, kind, bins, alpha, color, ax):
    """
    [CHANGE: This is just factored out logic so we can handle leftover dims or flatten]
    Flatten to 1D and plot distribution (hist or kde).
    """
    import numpy as np
    from scipy.stats import gaussian_kde

    arr = da.values.ravel()  # flatten
    arr = arr[~np.isnan(arr)]  # drop NaNs
    if len(arr) < 1:
        print(f"[Warning] No valid data for {dist_label}")
        return

    if kind == "hist":
        ax.hist(arr, bins=bins, alpha=alpha, color=color, label=dist_label, density=False)
    elif kind == "kde":
        if len(arr) < 2:
            print(f"[Warning] Not enough data for KDE in {dist_label}")
            return
        kde_func = gaussian_kde(arr)
        x_grid = np.linspace(np.min(arr), np.max(arr), 200)
        y_kde = kde_func(x_grid)
        ax.plot(x_grid, y_kde, color=color, label=dist_label)
    else:
        print(f"[Warning] Unknown distribution kind '{kind}'. Only 'hist'/'kde'.")


def plot_member_subplots(da,
                         out_path,
                         title=None,
                         ncols=2,
                         cmap='viridis',
                         vmin=None,
                         vmax=None,
                         **kwargs):
    """
    Generate side-by-side subplots for each ensemble member (2D lat-lon maps).
    Each member's map is plotted in a separate subplot.

    Parameters
    ----------
    da : xarray.DataArray
        Data with dims ['lat_coord', 'lon_coord', 'member'] 
        or at least a 'member' dimension.
    out_path : str
        Where to save the subplot figure.
    title : str, optional
        Overall figure title.
    ncols : int, optional
        Number of columns in the subplot grid (default 2).
    cmap : str, optional
        Colormap for each subplot.
    vmin, vmax : float, optional
        Color scale min/max.
    **kwargs : dict
        Passed to the underlying xarray plot call.
    """
    if 'member' not in da.dims:
        raise ValueError("plot_member_subplots requires a 'member' dimension in the DataArray.")

    members = da.member.values
    n_members = len(members)
    nrows = int(np.ceil(n_members / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(5 * ncols, 4 * nrows),
                            subplot_kw={'projection': ccrs.PlateCarree()})

    if title:
        fig.suptitle(title, fontsize=16)
    
    # Flatten axes in case nrows > 1
    axs = np.array(axs).reshape(-1)

    for i, member_val in enumerate(members):
        ax = axs[i]
        sub_da = da.sel(member=member_val)
        # We pass transform=ccrs.PlateCarree(), as data are lat/lon
        p = sub_da.plot(
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,  # single colorbar logic below
            **kwargs
        )
        ax.set_title(f"Member {member_val}")
        ax.coastlines('50m', color='black')
        ax.add_feature(cfeature.BORDERS, edgecolor='black')
        ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # Optionally add a single colorbar for all subplots
    cb = fig.colorbar(p, ax=axs.tolist(), orientation='vertical', shrink=0.7)
    cb.set_label(da.name or 'Value')

    # Hide any extra unused subplots
    for j in range(i+1, len(axs)):
        axs[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot_member_subplots] Saved figure to {out_path}")



def plot_alt_bin_subplots(da,
                          out_path,
                          title=None,
                          ncols=2,
                          cmap='viridis',
                          vmin=None,
                          vmax=None,
                          **kwargs):
    """
    Generate side-by-side subplots for each altitude bin (2D lat-lon maps).
    Each bin's map is plotted in a separate subplot.

    Parameters
    ----------
    da : xarray.DataArray
        Data with dims ['lat_coord', 'lon_coord', 'alt_bin'] 
        or at least an 'alt_bin' dimension.
    out_path : str
        Where to save the subplot figure (e.g. "output/spatial_me_by_bin.png").
    title : str, optional
        Overall figure title.
    ncols : int, optional
        Number of columns in the subplot grid (default 2).
    cmap : str, optional
        Colormap (e.g. "viridis").
    vmin, vmax : float, optional
        Color scale limits.
    **kwargs : dict
        Passed to the underlying xarray plot call (e.g. transform=ccrs.PlateCarree()).
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # 1) Check if alt_bin in dims
    if 'alt_bin' not in da.dims:
        raise ValueError("plot_alt_bin_subplots requires an 'alt_bin' dimension in the DataArray.")

    # 2) Get the bin labels
    bins = da.alt_bin.values
    n_bins = len(bins)

    # 3) Figure layout
    nrows = int(np.ceil(n_bins / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(5 * ncols, 4 * nrows),
                            subplot_kw={'projection': ccrs.PlateCarree()})

    if title:
        fig.suptitle(title, fontsize=16)

    # Flatten axes if multi-row
    axs = np.array(axs).ravel()

    # 4) For each bin
    for i, bin_val in enumerate(bins):
        ax = axs[i]
        sub_da = da.sel(alt_bin=bin_val)

        # If there's leftover dims besides lat/lon, e.g. time or member, you can do:
        leftover_dims = [d for d in sub_da.dims if d not in ('lat_coord','lon_coord')]
        if leftover_dims:
            sub_da = sub_da.mean(dim=leftover_dims)

        # Plot
        p = sub_da.plot(
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,  
            **kwargs
        )

        ax.set_title(f"Bin {bin_val}")
        ax.coastlines('50m', color='black')
        ax.add_feature(cfeature.BORDERS, edgecolor='black')
        ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # 5) Hide any extra unused subplots
    for j in range(i+1, len(axs)):
        axs[j].set_visible(False)

    # 6) Single colorbar for all subplots
    cb = fig.colorbar(p, ax=axs.tolist(), orientation='vertical', shrink=0.7)
    cb.set_label(da.name or 'Value')

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot_alt_bin_subplots] Saved figure to {out_path}")



def plot_altitude_bin_subplots_old(bin_dict,
                               metric="RMSE",
                               out_path="output/altbin_subplots.png",
                               title=None,
                               ncols=2,
                               cmap='viridis',
                               vmin=None,
                               vmax=None,
                               **kwargs):
    """
    Generate side-by-side subplots for each altitude bin, 
    plotting a 2D lat-lon map of 'metric'.

    Parameters
    ----------
    bin_dict : dict
        Dictionary from bin_by_altitude. Keys are (min_b, max_b), 
        values are xarray.Datasets that contain 'metric' and coordinates.
    metric : str
        Which variable to plot (e.g. "RMSE", "precipitation").
    out_path : str
        File path to save the figure (e.g. "output/altbin_subplots.png").
    title : str, optional
        Overall figure title.
    ncols : int, optional
        Number of columns in the subplot grid (default 2).
    cmap : str, optional
        Colormap for each subplot.
    vmin, vmax : float, optional
        Color scale min/max.
    **kwargs : dict
        Passed to xarray's .plot().

    Returns
    -------
    None
    """
    # Sort bin_dict by lower bound, so subplots appear in ascending altitude
    sorted_bins = sorted(bin_dict.keys(), key=lambda x: x[0])
    n_bins = len(sorted_bins)

    nrows = int(np.ceil(n_bins / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(5 * ncols, 4 * nrows),
                            subplot_kw={'projection': ccrs.PlateCarree()})

    if title:
        fig.suptitle(title, fontsize=16)

    axs = np.array(axs).ravel()  # flatten in case of multiple rows

    for i, bin_range in enumerate(sorted_bins):
        ds_bin = bin_dict[bin_range]
        # Convert ds_bin to a DataArray for plotting
        if metric not in ds_bin:
            print(f"[Warning] '{metric}' not in bin {bin_range}. Skipping.")
            continue
        da_bin = ds_bin[metric]

        ax = axs[i]
        # Example: squeeze if there's a leftover 'member' dimension from binning
        # or if the user had not averaged. Otherwise .plot() might throw an error
        # if it sees a leftover dimension. Let's just do a fallback .mean() if leftover dims not lat/lon:
        leftover_dims = [d for d in da_bin.dims if d not in ('lat_coord','lon_coord')]
        if leftover_dims:
            da_bin = da_bin.mean(dim=leftover_dims)

        print("da_bin dims:", da_bin.dims)
        print("da_bin dtype:", da_bin.dtype)
        print("da_bin shape:", da_bin.shape)


        p = da_bin.plot(
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,  # single colorbar logic
            **kwargs
        )

        ax.set_title(f"{metric} for alt {bin_range[0]}-{bin_range[1]}m")
        ax.coastlines('50m', color='black')
        ax.add_feature(cfeature.BORDERS, edgecolor='black')
        ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # Hide any extra unused subplots
    for j in range(i+1, len(axs)):
        axs[j].set_visible(False)

    # single colorbar for all subplots
    cb = fig.colorbar(p, ax=axs.tolist(), orientation='vertical', shrink=0.7)
    cb.set_label(metric)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot_altitude_bin_subplots] Saved figure to {out_path}")
