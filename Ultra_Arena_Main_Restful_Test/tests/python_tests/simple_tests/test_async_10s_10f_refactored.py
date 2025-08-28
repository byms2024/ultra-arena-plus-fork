#!/usr/bin/env python3
"""
Refactored Test: Asynchronous Combo Processing - 10 Strategies with 10 Files

This script uses the test_async_utils module to run tests with minimal code duplication.
"""

from test_async_utils import get_test_config_10s_10f, run_async_test
import sys


def main():
    """Run async test with 10 strategies and 10 files using the utility module."""
    
    # Get the configuration for 10 strategies with 10 files
    config = get_test_config_10s_10f("combo_test_10_strategies")
    
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
