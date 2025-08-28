#!/usr/bin/env python3
"""
Refactored Test: Asynchronous Combo Processing - Top 4 Strategies with 252 Files

This script uses the test_async_utils module to run tests with minimal code duplication.
"""

from test_async_utils import get_test_config_top_4s_252f, run_async_test
import sys


def main():
    """Run async test with top 4 strategies and 252 files using the utility module."""
    
    # Get the configuration for top 4 strategies with 252 files
    config = get_test_config_top_4s_252f("combo_test_top_4_strategies_2")
    
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
