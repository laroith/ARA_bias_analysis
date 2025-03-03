import matplotlib.pyplot as plt
from itertools import cycle

def build_cycle_lines(cycle_cfg):
    """
    Build a list of line_cfg dicts for plot_cycle_multi(), given a minimal config.
    
    Parameters
    ----------
    cycle_cfg : dict
        Expected fields:
          - alt_bin_ranges: list of [min, max] altitude pairs
          - ensemble_members: "all", "mean", or list like ["08","09"]
          - include_reference: bool
          - label_template: str, optional
          - auto_color_cycle: bool, optional
            (if True, picks different colors from matplotlib's cycle)
    
    Returns
    -------
    lines_cfg : list of dict
        Each dict includes keys needed by plot_cycle_multi, e.g.:
          - source, variable, member_selection, alt_bin_range, color, label
    """
    lines_cfg = []

    alt_bin_ranges = cycle_cfg.get("alt_bin_ranges", [])
    members_config = cycle_cfg.get("ensemble_members", "all")
    include_ref = cycle_cfg.get("include_reference", True)

    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    label_template = cycle_cfg.get("label_template",
                                   "{source}-{member}_{alt_min}-{alt_max}")

    # 1) Figure out the member selection approach
    if isinstance(members_config, str):
        if members_config in ["all", "mean"]:
            members_list = [members_config]
        else:
            # single ID
            members_list = [members_config]
    elif isinstance(members_config, list):
        members_list = members_config
    else:
        raise ValueError(f"Unknown ensemble_members setting: {members_config}")

    # 2) Build lines for ensemble
    for (alt_min, alt_max) in alt_bin_ranges:
        for mem_id in members_list:
            color = next(color_cycle)
            label = label_template.format(
                source="ens",
                member=mem_id,
                alt_min=alt_min,
                alt_max=alt_max
            )
            line_cfg = {
                "source": "ensemble",
                "variable": "precipitation", # if you always name ensemble var "precipitation"
                "member_selection": mem_id,
                "alt_bin_range": [alt_min, alt_max],
                "color": color,
                "label": label
            }
            lines_cfg.append(line_cfg)

    # 3) Build lines for reference
    if include_ref:
        for (alt_min, alt_max) in alt_bin_ranges:
            color = next(color_cycle)
            label = label_template.format(
                source="ref",
                member="na",
                alt_min=alt_min,
                alt_max=alt_max
            )
            line_cfg = {
                "source": "reference",
                "variable": "RR",  # if reference is always "RR"
                "alt_bin_range": [alt_min, alt_max],
                "color": color,
                "label": label
            }
            lines_cfg.append(line_cfg)

    return lines_cfg

def build_time_series_lines(ts_cfg):
    """
    Build a list of line_cfg dicts for plot_time_series_multi_line()
    from a more minimal config in ts_cfg.

    Returns: list of dict
    """
    lines_cfg = []

    alt_bin_ranges = ts_cfg.get("alt_bin_ranges", [])
    members_config = ts_cfg.get("ensemble_members", "all")
    include_ref = ts_cfg.get("include_reference", True)

    # For coloring, we can use either a user-specified colormap 
    # or the default matplotlib color cycle:
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    # Or if the user sets a "color_map", we can do something more advanced.

    label_template = ts_cfg.get("label_template",
                                "{source}_{member}_{alt_min}-{alt_max}")

    # 1) Ensemble lines
    #    if members_config is "mean", we produce just 1 line,
    #    if it's "all", we produce multiple lines for each available member,
    #    if it's a list, produce lines for that list, etc.
    #    For each altitude bin in alt_bin_ranges, produce a line.

    members_list = []
    if isinstance(members_config, str):
        if members_config in ["all", "mean"]:
            members_list = [members_config]  # single placeholder
        else:
            members_list = [members_config]  # single specific ID
    elif isinstance(members_config, list):
        members_list = members_config
    else:
        raise ValueError(f"Unknown ensemble_members config: {members_config}")

    for alt_min, alt_max in alt_bin_ranges:
        for mem_id in members_list:
            # pick a color from the cycle
            c = next(color_cycle)
            # build a label
            lbl = label_template.format(
                source="ens",
                member=mem_id,
                alt_min=alt_min,
                alt_max=alt_max
            )
            line_entry = {
                "source": "ensemble",
                "variable": "precipitation",  # Hardcode ensemble variable
                "member_selection": mem_id,
                "alt_bin_range": [alt_min, alt_max],
                "color": c,
                "label": lbl
            }
            lines_cfg.append(line_entry)

    # 2) Reference lines (if enabled)
    if include_ref:
        for alt_min, alt_max in alt_bin_ranges:
            c = next(color_cycle)
            lbl = label_template.format(
                source="ref",
                member="na",
                alt_min=alt_min,
                alt_max=alt_max
            )
            line_entry = {
                "source": "reference",
                "variable": "RR",    # Hardcode reference variable
                "alt_bin_range": [alt_min, alt_max],
                "color": c,
                "label": lbl
            }
            lines_cfg.append(line_entry)

    return lines_cfg


# Suppose in your plotting or a new utils_distribution.py module

import matplotlib.pyplot as plt
from itertools import cycle

def build_distribution_lines(dist_cfg):
    """
    Build a list of line_cfg dicts for plot_distribution_multi()
    from a simpler distribution config in dist_cfg.

    Returns: list of dict
    """
    lines_cfg = []

    alt_bin_ranges = dist_cfg.get("alt_bin_ranges", [])
    members_config = dist_cfg.get("ensemble_members", "all")
    include_ref = dist_cfg.get("include_reference", True)

    # For coloring, either pick from default color cycle
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    label_template = dist_cfg.get("label_template",
                                  "{source}_{member}_{alt_min}-{alt_max}")

    # 1) Determine which members to iterate over
    if isinstance(members_config, str):
        if members_config in ["all", "mean"]:
            members_list = [members_config]  # a single placeholder
        else:
            # single ID like "08"
            members_list = [members_config]
    elif isinstance(members_config, list):
        members_list = members_config
    else:
        raise ValueError(f"Unknown ensemble_members: {members_config}")

    # 2) Build lines for ensemble
    for (alt_min, alt_max) in alt_bin_ranges:
        for mem_id in members_list:
            color = next(color_cycle)
            label = label_template.format(
                source="ens",
                member=mem_id,
                alt_min=alt_min,
                alt_max=alt_max
            )
            line_entry = {
                "source": "ensemble",
                "variable": "precipitation",  # Hardcode if always "precip" for ensemble
                "member_selection": mem_id,
                "alt_bin_range": [alt_min, alt_max],
                "color": color,
                "label": label,
                # If you want to force flatten across e.g. time dimension, you can do:
                # "reduce_dims": ["time"]
            }
            lines_cfg.append(line_entry)

    # 3) Build lines for reference (if enabled)
    if include_ref:
        for (alt_min, alt_max) in alt_bin_ranges:
            color = next(color_cycle)
            label = label_template.format(
                source="ref",
                member="na",
                alt_min=alt_min,
                alt_max=alt_max
            )
            line_entry = {
                "source": "reference",
                "variable": "RR",  # Hardcode if always "RR" for reference
                "alt_bin_range": [alt_min, alt_max],
                "color": color,
                "label": label
            }
            lines_cfg.append(line_entry)

    return lines_cfg
