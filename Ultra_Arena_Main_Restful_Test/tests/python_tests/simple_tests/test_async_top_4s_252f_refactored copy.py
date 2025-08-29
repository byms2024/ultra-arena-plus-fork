#!/usr/bin/env python3
"""
Refactored Test: Asynchronous Combo Processing - Top 4 Strategies with 252 Files

This script creates its own TestConfig and uses the test_async_utils module to run tests
with minimal code duplication.
"""

from test_async_utils import run_async_test, TestConfig
import sys


def main():
    """Run async test with top 4 strategies and 252 files using the utility module."""
    
    # Create the configuration for top 4 strategies with 252 files (using 200_files directory)
    config = TestConfig(
        combo_name="combo_test_top_4_strategies_2",
        file_name="200_files",
        max_wait_time=1800,  # 30 minutes for large test
        poll_interval=10,    # 10 seconds for large test
        max_files_per_request=10
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
