import xarray as xr
import numpy as np

def check_missing_values(ds, var_name):
    """Check for missing values in a dataset."""
    missing = ds[var_name].isnull().sum().values
    print(f"Missing values in {var_name}: {missing}")

def check_outliers(ds, var_name, threshold=500):
    """Identify potential outliers in precipitation data (e.g., extreme values)."""
    outlier_count = np.sum(ds[var_name] > threshold).values
    print(f"Number of outliers in {var_name} above {threshold} mm: {outlier_count}")

def main():
    # Load dataset
    ensemble_ds = xr.open_dataset("data/ensemble_subset.nc")
    reference_ds = xr.open_dataset("data/SPARTACUS2-DAILY_RR_2011.nc")

    # Check quality of ensemble dataset
    check_missing_values(ensemble_ds, "precipitation")
    check_outliers(ensemble_ds, "precipitation")

    # Check quality of reference dataset
    check_missing_values(reference_ds, "RR")
    check_outliers(reference_ds, "RR")

if __name__ == "__main__":
    main()
