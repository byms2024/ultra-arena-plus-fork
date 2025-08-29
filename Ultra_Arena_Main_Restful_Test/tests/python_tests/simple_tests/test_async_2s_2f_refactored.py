#!/usr/bin/env python3
"""
Refactored Test: Asynchronous Combo Processing - 2 Strategies with 2 Files

This script creates its own TestConfig and uses the test_async_utils module to run tests
with minimal code duplication.
"""

from test_async_utils import run_async_test, TestConfig
import sys


def main():
    """Run async test with 2 strategies and 2 files using the utility module."""
    
    # Create the configuration for 2 strategies with 2 files
    config = TestConfig(
        combo_name="test_both_strategies_openai",
        file_name="2_files",
        max_wait_time=3000,
        poll_interval=10
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
