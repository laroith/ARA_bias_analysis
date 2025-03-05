# scripts/unit_conversion.py
import xarray as xr

def unify_temperature_units(ds, var_name, target="degree_Celsius"):
    """
    If ds[var_name] has units not matching 'target', convert them.
    By default, we unify everything to degrees Celsius ('degree_Celsius').
    This function modifies ds in-place and returns it.

    - ds: xarray.Dataset
    - var_name: str
    - target: e.g. 'degree_Celsius'
    """
    if var_name not in ds:
        # not present => do nothing
        return ds

    # Check attribute "units" if it exists
    existing_units = ds[var_name].attrs.get("units", "").lower()

    # If we are unifying to Celsius:
    if target.lower() in ["degc", "c", "celsius", "degree_celsius"]:
        if "c" in existing_units:
            # Already in Celsius => do nothing
            pass
        elif "k" in existing_units:
            # It's in Kelvin => convert to Celsius
            ds[var_name] = ds[var_name] - 273.15
            ds[var_name].attrs["units"] = "degree_Celsius"
            print(f"[Info] Converted {var_name} from Kelvin to Celsius.")
        elif existing_units == "":
            print(f"[Warning] {var_name} has no 'units' attribute. Assuming it's Kelvin.")
            # Optionally set:
            ds[var_name] = ds[var_name] - 273.15
            ds[var_name].attrs["units"] = "degree_Celsius"
        else:
            # Another unknown unit => warn user
            print(f"[Warning] {var_name} has unrecognized units: '{existing_units}'. "
                  f"Not converting.")
    else:
        print(f"[Warning] unify_temperature_units only handles degree_Celsius. target='{target}' is not supported.")
    
    return ds
