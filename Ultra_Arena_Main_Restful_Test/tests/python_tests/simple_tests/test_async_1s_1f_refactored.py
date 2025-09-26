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
    
    chain_config = {
        "chains": {
                "subchains": [
                    {
                        "censor": True,
                        "metadata_fields": [
                            "claim_id",
                            "claim_no",
                            "vin",
                            "dealer_cnpj",
                            "part_amount_dms",
                            "labour_amount_dms"
                        ],
                        "subchain_name": "subchain_regex",
                        "fileNumberPerFile": 1,
                        "pre-processing": {
                            "pre-type": "regex"
                        },
                        "processing": {
                            "proc-type": "regex",
                        },
                        "post-processing": {
                            "post-type": "regex",
                            "retries": {
                                "pre_retry": {
                                    "retry_count": 0
                                },
                                "proc_retry": {
                                    "retry_count": 0
                                }
                            }
                        }
                    }
                ]
            }
        }

    # Create the configuration for 1 strategy with 1 file
    config = TestConfig(
        # combo_name="single_strategy_text_first_google",
        chain_name="chain_strategy",
        chain_config = chain_config,
        file_name="1_file",
        max_wait_time=3000,
        poll_interval=10,
        desensitization= False,
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
