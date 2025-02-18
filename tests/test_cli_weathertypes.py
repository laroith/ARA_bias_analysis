# tests/test_cli_wethertypes.py

import pytest
import subprocess
import sys
import os

@pytest.fixture
def base_args():
    """
    Base command for calling cli.py with a known test_config.yaml.
    Adjust 'test_config.yaml' to ensure it references minimal test data.
    """
    # Point to your cli.py. Adjust if it's in a different location.
    cli_path = os.path.abspath("cli/cli.py")
    # Use a test-specific config or a general config as a baseline.
    config_path = "tests/test_config.yaml"
    return [sys.executable, cli_path, "--config", config_path]

def test_weather_type_filtering(base_args):
    """
    Test that specifying a weather type file and include-weather-types
    filters both the ensemble and reference datasets. 
    We expect the script to run successfully and produce relevant log lines.
    """
    # Use a minimal CSV that has 'date' and 'slwt' columns, plus a small subset of times
    weather_csv = "data/ERA-5_historical.csv"

    cmd = base_args + [
        "--weather-type-file", weather_csv,
        "--include-weather-types", "1", "6",
        "--aggregation", "daily"
    ]
    
    # Example: also specify a single member
    cmd += ["--member", "00"]

    print("Running command:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # We expect success
    assert result.returncode == 0, f"CLI failed with code {result.returncode}\nSTDERR:\n{result.stderr}"
    
    # Check if the filtering step took place
    assert "Filtering datasets by weather types: [1, 6]" in result.stdout, (
        f"Did not see weather type filtering message. Output:\n{result.stdout}"
    )
    # Check that it also attempts monthly aggregation
    assert "Performing daily aggregation" in result.stdout
    assert "Ensemble dataset after weather type filtering:" in result.stdout

    # Optionally check for references to single-member selection
    assert "Selecting ensemble members: ['00']" in result.stdout
