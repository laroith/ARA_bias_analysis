# tests/test_cli.py

import subprocess
import sys
import os
import pytest

@pytest.fixture
def cli_script_path():
    # Construct a full path to the cli.py script, relative to the test file
    current_dir = os.path.dirname(__file__)           # tests/ directory
    project_root = os.path.abspath(os.path.join(current_dir, ".."))  # bias_analysis/ directory
    cli_script = os.path.join(project_root, "cli/cli.py")
    return cli_script

@pytest.fixture
def base_args(cli_script_path):
    """
    Provide a minimal set of arguments that generally work for all tests.
    You can adjust ensemble_pattern, reference_file, etc. in test_config.yaml
    if you want to keep data paths there. Or override them on the CLI.
    """
    return [
        sys.executable,  # e.g. "python"
        cli_script_path,        # path to your cli script
        "--config", "tests/test_config.yaml"
    ]

def test_inspect_mode(base_args):
    """
    Test that --inspect prints dataset info and exits before running the normal workflow.
    We expect certain lines to appear in the output (like "Ensemble Dataset Info").
    """
    cmd = base_args + ["--inspect"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "=== Ensemble Dataset Info ===" in result.stdout
    assert "=== Reference Dataset Info ===" in result.stdout
    # Since the script should exit early, we also expect NOT to see certain messages:
    assert "Regridding ensemble to match reference grid..." not in result.stdout

def test_invalid_var_name_warning(base_args):
    """
    If we pass a bad variable name, we expect the script to warn us about it.
    """
    # We'll override the var_name in the config by passing --var-name from CLI
    cmd = base_args + ["--var-name", "non_existent_variable"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # We don't necessarily want it to fail: maybe the script logs a warning but continues
    assert "Warning: 'non_existent_variable' not found in ensemble dataset." in result.stdout

def test_single_ensemble_member(base_args):
    """
    Test that we can specify a single ensemble member and see the expected output.
    """
    cmd = base_args + ["--member", "00"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    # Check the stdout for a known message
    # e.g. "Selecting ensemble members: ['00']"
    assert "Selecting ensemble members: ['00']" in result.stdout

def test_multiple_ensemble_members(base_args):
    """
    Test specifying multiple members: 00, 01.
    """
    cmd = base_args + ["--member", "00", "01"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "Selecting ensemble members: ['00', '01']" in result.stdout

def test_ensemble_mean(base_args):
    """
    Test computing the ensemble mean (using all members).
    """
    cmd = base_args + ["--ensemble-mean"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "Computing ensemble mean across selected members..." in result.stdout

def test_daily_aggregation(base_args):
    """
    Test daily aggregation step.
    """
    cmd = base_args + [
        "--aggregation", "daily"  # if you have that in your CLI, or else specify in the config
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    # Check for a log line that indicates daily aggregation
    # e.g. "Performing daily aggregation on ensemble data..."
    assert "Performing daily aggregation on ensemble data..." in result.stdout

def test_monthly_aggregation(base_args):
    """
    Test monthly aggregation step.
    """
    cmd = base_args + [
        "--aggregation", "monthly"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "Performing monthly aggregation on ensemble data..." in result.stdout

def test_bias_dimensions(base_args):
    """
    Test passing a dimension list for bias metrics.
    For example, we might pass --bias-dim time lat lon, etc.
    """
    cmd = base_args + [
        "--bias-dim", "time", # "lat_coord", "lon_coord",  # or however your CLI is set up
        "--bias-metrics", "RMSE"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    # Suppose we expect a line about computing RMSE
    assert "Computing RMSE..." in result.stdout
