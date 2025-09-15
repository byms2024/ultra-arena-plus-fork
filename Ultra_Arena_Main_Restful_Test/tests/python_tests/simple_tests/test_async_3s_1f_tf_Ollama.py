#!/usr/bin/env python3
"""
Refactored Test: Asynchronous Combo Processing - 1 Strategy with 1 File

This script creates its own TestConfig and uses the test_async_utils module to run tests
with minimal code duplication.
"""

from test_async_utils import run_async_test, TestConfig
import sys


def main():
    """Run async test with 1 strategy and 1 file using the utility module."""
    
    # Create the configuration for 1 strategy with 1 file
    config = TestConfig(
        combo_name="combo_test_textF_ollama_3s",
        file_name="1_file",
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
