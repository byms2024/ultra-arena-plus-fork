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
                            "pre-type": "regex",
                            "enable_pdf_metadata": True,
                            "filename_pattern_case_sensitive": False
                        },
                        "processing": {
                            "proc-type": "regex",
                        },
                        "post-processing": {
                            "post-type": "metadata",
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
                    # {
                    #     "censor": True,
                    #     "metadata_fields": [
                    #         "claim_id",
                    #         "claim_no",
                    #         "vin",
                    #         "dealer_cnpj",
                    #         "part_amount_dms",
                    #         "labour_amount_dms"
                    #     ],
                    #     "subchain_name": "text_first_gemini",
                    #     "fileNumberPerFile": 1,
                    #     "pre-processing": {
                    #         "pre-type": "text",
                    #         "enable_pdf_metadata": True,
                    #         "filename_pattern_case_sensitive": False
                    #     },
                    #     "processing": {
                    #         "proc-type": "text_first",
                    #     },
                    #     "post-processing": {
                    #         "post-type": "metadata",
                    #         "retries": {
                    #             "pre_retry": {
                    #                 "retry_count": 0
                    #             },
                    #             "proc_retry": {
                    #                 "retry_count": 0
                    #             }
                    #         }
                    #     }
                    # }
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

        # Analyze the results for passthrough functionality
        print("\n📊 Analyzing results for passthrough functionality:")

        # Check if result has the expected structure
        if isinstance(result, dict):
            print(f"📈 Result structure: {list(result.keys())}")

            # Look for results in different possible locations
            results_data = None
            if 'results' in result:
                results_data = result['results']
                print("✅ Found results in 'results' key")
            elif isinstance(result, list):
                results_data = result
                print("✅ Result is a list (direct results)")
            else:
                # Check if result itself contains file results
                results_data = []
                for key, value in result.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
                        results_data = value
                        print(f"✅ Found results in '{key}' key")
                        break

            if results_data:
                print(f"📁 Processing {len(results_data)} file results:")
                for i, file_result in enumerate(results_data):
                    if isinstance(file_result, list) and len(file_result) == 2:
                        file_path, file_data = file_result
                        print(f"\n📄 File {i+1}: {file_path}")
                    else:
                        file_data = file_result
                        print(f"\n📄 Result {i+1}:")

                    if isinstance(file_data, dict):
                        print(f"   Status: {file_data.get('status', 'N/A')}")
                        
                        # Check for pattern validation status
                        pattern_status = file_data.get('pattern_validation_status')
                        if pattern_status:
                            print(f"   🔍 Pattern Validation: {pattern_status}")
                            pattern_reason = file_data.get('pattern_validation_reason')
                            if pattern_reason:
                                print(f"      Reason: {pattern_reason}")

                        # Check for DMS data (should be preserved from pre-processing)
                        dms_fields = {k: v for k, v in file_data.items() if k.startswith('dms.')}
                        if dms_fields:
                            print(f"   ✅ DMS data preserved: {len(dms_fields)} fields")
                            for k, v in list(dms_fields.items())[:3]:  # Show first 3
                                print(f"      {k}: {v}")
                            if len(dms_fields) > 3:
                                print(f"      ... and {len(dms_fields) - 3} more DMS fields")
                        else:
                            print("   ❌ No DMS data found")

                        # Check for processing data
                        proc_fields = {k: v for k, v in file_data.items() if k.startswith('proc.')}
                        if proc_fields:
                            print(f"   ✅ Processing data: {len(proc_fields)} fields")
                        else:
                            print("   ❌ No processing data found")

                        # Check for passthrough continuity indicators
                        if 'match_statuses' in file_data:
                            print(f"   🔗 Match statuses: {file_data['match_statuses']}")

                        # Check for match flags
                        match_flags = {k: v for k, v in file_data.items() if k.startswith('match_')}
                        if match_flags:
                            print(f"   🎯 Match flags: {match_flags}")
                    else:
                        print(f"   Unexpected data type: {type(file_data)}")
            else:
                print("❌ Could not find results data in response")
                print(f"📋 Available keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        else:
            print(f"❌ Unexpected result type: {type(result)}")
    else:
        print("❌ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
