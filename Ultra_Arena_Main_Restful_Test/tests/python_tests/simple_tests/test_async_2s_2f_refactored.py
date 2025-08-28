#!/usr/bin/env python3
"""
Refactored Test: Asynchronous Combo Processing - 2 Strategies with 2 Files

This script uses the test_async_utils module to run tests with minimal code duplication.
"""

from test_async_utils import get_test_config_2s_2f, run_async_test
import sys


def main():
    """Run async test with 2 strategies and 2 files using the utility module."""
    
    # Get the configuration for 2 strategies with 2 files
    config = get_test_config_2s_2f("test_both_strategies_openai")
    
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
