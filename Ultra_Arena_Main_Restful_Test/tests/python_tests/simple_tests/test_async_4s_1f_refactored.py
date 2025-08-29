#!/usr/bin/env python3
"""
Refactored Test: Asynchronous Combo Processing - 4 Strategies with 1 File

This script creates its own TestConfig and uses the test_async_utils module to run tests
with minimal code duplication.
"""

from test_async_utils import run_async_test, TestConfig
import sys


def main():
    """Run async test with 4 strategies and 1 file using the utility module."""
    
    # Create the configuration for 4 strategies with 1 file
    config = TestConfig(
        combo_name="combo_potential_top_4_strategies",
        file_name="1_file",
        max_wait_time=300,
        poll_interval=5
    )
    
    # Run the test
    result = run_async_test(config)
    
    # Note: result can be an empty dict {} which is still successful
    if result is not None:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
