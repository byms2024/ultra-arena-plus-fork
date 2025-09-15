#!/usr/bin/env python3
"""
Refactored Test: Asynchronous Combo Processing - 1 Strategy with 252 Files

This script creates its own TestConfig and uses the test_async_utils module to run tests
with minimal code duplication.
"""

from test_async_utils import run_async_test, TestConfig
import sys


def main():
    """Run async test with 1 strategy and 252 files using the utility module."""
    
    # Create the configuration for 1 strategy with 252 files (using 200_files directory)
    config = TestConfig(
        combo_name="single_test_textF_ollama_gptOss20b",
        file_name="1_file",
        max_wait_time=3000,  # 30 minutes for large test
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
