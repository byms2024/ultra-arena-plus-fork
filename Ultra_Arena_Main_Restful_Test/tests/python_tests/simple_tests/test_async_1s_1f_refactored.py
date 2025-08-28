#!/usr/bin/env python3
"""
Refactored Test: Asynchronous Combo Processing - 1 Strategy with 1 File

This script demonstrates how to use the test_async_utils module to run tests
with minimal code duplication.
"""

from test_async_utils import get_test_config_1s_1f, run_async_test, TestConfig
import sys


# Predefined test configurations for common scenarios
def get_test_config_1s_1f(combo_name: str = "single_strategy_text_first_google") -> TestConfig:
    """Get configuration for 1 strategy with 1 file."""
    return TestConfig(
        combo_name=combo_name,
        file_name="1_file",
        max_wait_time=300,
        poll_interval=5
    )

def main():
    """Run async test with 1 strategy and 1 file using the utility module."""
    
    # Get the configuration for 1 strategy with 1 file
    config = get_test_config_1s_1f("single_strategy_text_first_google")
    
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
