import os
import sys

# Add root to sys path so we can import from scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from scripts.validate_data import validate

def test_synthetic_data_integrity():
    """
    Test that the synthetic data generation maintains referential integrity,
    has no duplicates, and properly aligns with the ground truth.
    """
    try:
        validate()
    except AssertionError as e:
        assert False, f"Data validation failed: {str(e)}"
