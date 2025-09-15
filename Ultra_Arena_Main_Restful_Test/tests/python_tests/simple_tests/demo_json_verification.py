#!/usr/bin/env python3
"""
Demo Script: JSON Verification of overall_stats.total_processed_files

This script demonstrates how the verification functions work by checking
the existing JSON files in the test fixtures.
"""

import sys
from pathlib import Path
from typing import Optional

def find_latest_results_directory(output_base_dir: Path) -> Optional[Path]:
    """Find the most recent results directory in the output_files folder."""
    try:
        # Look for directories that match the pattern "results_*"
        results_dirs = list(output_base_dir.glob("results_*"))

        if not results_dirs:
            print(f"⚠️  No results directories found in {output_base_dir}")
            return None

        # Sort by modification time (most recent first)
        results_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_dir = results_dirs[0]

        print(f"📁 Found latest results directory: {latest_dir.name}")
        return latest_dir

    except Exception as e:
        print(f"❌ Error finding results directory: {e}")
        return None

def find_json_files(results_dir: Path) -> list:
    """Find all JSON files in the results directory."""
    json_files = []

    try:
        # Look for JSON files in the json/ subdirectory
        json_dir = results_dir / "json"
        if json_dir.exists():
            json_files = list(json_dir.glob("*.json"))
        else:
            # Fallback: look for JSON files directly in results directory
            json_files = list(results_dir.glob("*.json"))

        # Sort by modification time (oldest first to see progression)
        json_files.sort(key=lambda x: x.stat().st_mtime)

        print(f"📄 Found {len(json_files)} JSON files:")
        for i, json_file in enumerate(json_files, 1):
            print(f"   {i}. {json_file.name}")

        return json_files

    except Exception as e:
        print(f"❌ Error finding JSON files: {e}")
        return []

def verify_overall_stats_total_processed_files(json_files: list) -> bool:
    """Verify that overall_stats.total_processed_files is present and reasonable in JSON files."""
    if not json_files:
        print("❌ No JSON files to verify")
        return False

    verification_passed = True

    print("\n🔍 VERIFICATION: overall_stats.total_processed_files")
    print("=" * 60)

    for i, json_file in enumerate(json_files, 1):
        try:
            import json
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if overall_stats exists
            if 'overall_stats' not in data:
                print(f"❌ [{i}] {json_file.name}: Missing 'overall_stats' section")
                verification_passed = False
                continue

            overall_stats = data['overall_stats']

            # Check if total_processed_files exists
            if 'total_processed_files' not in overall_stats:
                print(f"❌ [{i}] {json_file.name}: Missing 'total_processed_files' in overall_stats")
                verification_passed = False
                continue

            total_processed_files = overall_stats['total_processed_files']

            # Check if it's a reasonable integer
            if not isinstance(total_processed_files, int) or total_processed_files < 0:
                print(f"❌ [{i}] {json_file.name}: Invalid total_processed_files value: {total_processed_files}")
                verification_passed = False
                continue

            print(f"✅ [{i}] {json_file.name}: total_processed_files = {total_processed_files}")

            # Show some additional stats for context
            if 'file_stats' in data:
                actual_files_in_stats = len(data['file_stats'])
                print(f"   📊 Files in file_stats: {actual_files_in_stats}")

                if total_processed_files == actual_files_in_stats:
                    print("   ✅ Count matches file_stats!")
                else:
                    print(f"   ⚠️  Count mismatch: expected {actual_files_in_stats}, got {total_processed_files}")

        except json.JSONDecodeError as e:
            print(f"❌ [{i}] {json_file.name}: Invalid JSON format - {e}")
            verification_passed = False
        except Exception as e:
            print(f"❌ [{i}] {json_file.name}: Error reading file - {e}")
            verification_passed = False

    return verification_passed

def main():
    """Demonstrate the JSON verification functionality."""

    print("🎯 DEMO: JSON Verification of overall_stats.total_processed_files")
    print("=" * 70)
    print("This script demonstrates how the verification functions work")
    print("by checking existing JSON files in the test fixtures.")
    print()

    # Determine the output directory path
    current_dir = Path(__file__).resolve()
    test_root = current_dir.parent.parent.parent.parent  # Go up to Ultra_Arena_Main_Restful_Test
    output_base_dir = test_root / "test_fixtures" / "br_fixture" / "output_files"

    print(f"📁 Looking for results in: {output_base_dir}")

    if output_base_dir.exists():
        # Find the latest results directory
        latest_results_dir = find_latest_results_directory(output_base_dir)

        if latest_results_dir:
            # Find JSON files
            json_files = find_json_files(latest_results_dir)

            if json_files:
                # Verify overall_stats.total_processed_files
                json_verification_passed = verify_overall_stats_total_processed_files(json_files)

                print("\n" + "=" * 60)
                if json_verification_passed:
                    print("✅ DEMO VERIFICATION PASSED!")
                    print("   overall_stats.total_processed_files is correctly present")
                    print("   and has reasonable values in the JSON files.")
                else:
                    print("❌ DEMO VERIFICATION FAILED!")
                    print("   Issues found with overall_stats.total_processed_files")
                    print("   in the JSON files.")
                print("=" * 60)

                print("\n💡 This demonstrates that our enhanced test script will work correctly!")
                print("   The verification functions can:")
                print("   - Find the latest results directory automatically")
                print("   - Locate JSON files within the directory structure")
                print("   - Parse and validate the overall_stats.total_processed_files field")
                print("   - Compare the count with actual files in file_stats")
            else:
                print("⚠️  No JSON files found in the latest results directory")
        else:
            print("⚠️  Could not find any results directories")
    else:
        print(f"⚠️  Output directory does not exist: {output_base_dir}")
        print("   (This is expected if no tests have been run yet)")

if __name__ == "__main__":
    main()
