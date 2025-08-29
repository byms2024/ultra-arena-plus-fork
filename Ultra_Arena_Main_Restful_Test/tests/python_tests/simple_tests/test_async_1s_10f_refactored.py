#!/usr/bin/env python3
"""
Refactored Test: Asynchronous Combo Processing - 1 Strategy with 10 Files

This script creates its own TestConfig and uses the test_async_utils module to run tests
with minimal code duplication.
"""

from test_async_utils import run_async_test, TestConfig
import sys


def main():
    """Run async test with 1 strategy and 10 files using the utility module."""
    
    # Create the configuration for 1 strategy with 10 files
    config = TestConfig(
        combo_name="single_strategy_direct_file_google",
        file_name="10_files",
        max_wait_time=3000,
        poll_interval=10,
        max_files_per_request=2,
        max_cc_filegroups=2
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
