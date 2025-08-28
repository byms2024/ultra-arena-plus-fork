#!/usr/bin/env python3
"""
Refactored Test: Asynchronous Combo Processing - 1 Strategy with 10 Files

This script uses the test_async_utils module to run tests with minimal code duplication.
"""

from test_async_utils import get_test_config_1s_10f, run_async_test
import sys


def main():
    """Run async test with 1 strategy and 10 files using the utility module."""
    
    # Get the configuration for 1 strategy with 10 files
    config = get_test_config_1s_10f("test_textF_openai_only")
    
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
