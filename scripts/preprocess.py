import xarray as xr

def aggregate_to_daily(ds, var_name):
    """Aggregate hourly precipitation to daily sums."""
    return ds[var_name].resample(time="1D").sum()

def preprocess_datasets():
    # Load datasets
    ensemble_ds = xr.open_dataset("data/ensemble_subset.nc")
    reference_ds = xr.open_dataset("data/SPARTACUS2-DAILY_RR_2011.nc")

    # Aggregate ensemble dataset to daily sums
    ensemble_ds["precipitation_daily"] = aggregate_to_daily(ensemble_ds, "precipitation")

    # Ensure reference dataset is properly formatted
    reference_ds["RR"] = reference_ds["RR"] * reference_ds["RR"].scale_factor  # Apply scaling factor

    # Save processed datasets (optional)
    ensemble_ds.to_netcdf("data/processed_ensemble.nc")
    reference_ds.to_netcdf("data/processed_reference.nc")

if __name__ == "__main__":
    preprocess_datasets()
