#!/usr/bin/env python3
"""
Refactored Test: Asynchronous Combo Processing - 5 Strategies with 1 File

This script uses the test_async_utils module to run tests with minimal code duplication.
"""

from test_async_utils import get_test_config_5s_1f, run_async_test
import sys


def main():
    """Run async test with 5 strategies and 1 file using the utility module."""
    
    # Get the configuration for 5 strategies with 1 file
    config = get_test_config_5s_1f("combo_test_5_sub_strategies")
    
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
