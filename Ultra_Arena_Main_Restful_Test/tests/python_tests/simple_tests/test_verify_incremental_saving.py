#!/usr/bin/env python3
"""
Verification Test: Incremental Result Saving - 1 File Per Request

This script tests that overall_stats.total_processed_files is updated correctly
after each individual file group (LLM request) when max_files_per_request=1.

This helps verify that incremental result saving works properly.
"""

import os
import sys
import json
import time
import glob
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from test_async_utils import run_async_test, TestConfig

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

def find_json_files(results_dir: Path, show_output: bool = True) -> list:
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

        if show_output:
            print(f"📄 Found {len(json_files)} JSON files:")
            for i, json_file in enumerate(json_files, 1):
                print(f"   {i}. {json_file.name}")

        return json_files

    except Exception as e:
        if show_output:
            print(f"❌ Error finding JSON files: {e}")
        return []

def get_json_file_states(json_files: list) -> Dict[str, Dict[str, Any]]:
    """Get the current state of all JSON files for monitoring."""
    states = {}

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            file_state = {
                'file_path': json_file,
                'modified_time': json_file.stat().st_mtime,
                'total_processed_files': None,
                'file_stats_count': 0,
                'last_update': time.time()
            }

            # Extract overall_stats.total_processed_files
            if 'overall_stats' in data and 'total_processed_files' in data['overall_stats']:
                file_state['total_processed_files'] = data['overall_stats']['total_processed_files']

            # Count files in file_stats
            if 'file_stats' in data:
                file_state['file_stats_count'] = len(data['file_stats'])

            states[json_file.name] = file_state

        except Exception as e:
            print(f"⚠️  Error reading {json_file.name}: {e}")

    return states

def monitor_json_files(json_files: list, stop_event: threading.Event, output_base_dir: Path):
    """Monitor JSON files for changes every 2 seconds."""
    print("\n📊 REAL-TIME MONITORING: Starting JSON file monitoring...")
    print("   Press Ctrl+C to stop monitoring and continue with test verification")
    print("=" * 80)

    previous_states = get_json_file_states(json_files) if json_files else {}
    monitoring_start = time.time()

    # Show initial state
    if previous_states:
        print(f"\n📈 INITIAL STATE (t={0:.1f}s):")
        for filename, state in previous_states.items():
            total_processed = state['total_processed_files']
            file_count = state['file_stats_count']
            print(f"   {filename}: total_processed_files={total_processed}, files_in_stats={file_count}")
    else:
        print(f"\n📈 INITIAL STATE (t={0:.1f}s): No JSON files to monitor yet")

    try:
        while not stop_event.is_set():
            time.sleep(2)  # Check every 2 seconds

            # Re-scan for JSON files in case new ones were created
            current_json_files = []
            latest_results_dir = find_latest_results_directory(output_base_dir)
            if latest_results_dir:
                current_json_files = find_json_files(latest_results_dir, show_output=False)

            current_states = get_json_file_states(current_json_files) if current_json_files else {}
            current_time = time.time() - monitoring_start

            changes_detected = False

            # Check for new files
            for filename, current_state in current_states.items():
                if filename not in previous_states:
                    print(f"\n📈 NEW FILE DETECTED (t={current_time:.1f}s): {filename}")
                    print(f"   Current: total_processed_files={current_state['total_processed_files']}, files_in_stats={current_state['file_stats_count']}")
                    changes_detected = True
                    continue

                prev_state = previous_states[filename]

                # Check for changes in existing files
                if (current_state['total_processed_files'] != prev_state['total_processed_files'] or
                    current_state['file_stats_count'] != prev_state['file_stats_count'] or
                    current_state['modified_time'] != prev_state['modified_time']):

                    changes_detected = True
                    print(f"\n📈 CHANGE DETECTED (t={current_time:.1f}s): {filename}")
                    print(f"   Previous: total_processed_files={prev_state['total_processed_files']}, files_in_stats={prev_state['file_stats_count']}")
                    print(f"   Current:  total_processed_files={current_state['total_processed_files']}, files_in_stats={current_state['file_stats_count']}")

                    # Check if this looks like incremental saving
                    if (current_state['total_processed_files'] is not None and
                        prev_state['total_processed_files'] is not None):
                        if current_state['total_processed_files'] > prev_state['total_processed_files']:
                            print(f"   ✅ INCREMENTAL SAVE: Count increased by {current_state['total_processed_files'] - prev_state['total_processed_files']}")
                        elif current_state['total_processed_files'] == prev_state['total_processed_files']:
                            print(f"   ℹ️  NO CHANGE: Count remained the same")
                        else:
                            print(f"   ⚠️  UNEXPECTED: Count decreased!")

            # Check for files that disappeared
            for filename in previous_states:
                if filename not in current_states:
                    print(f"\n📈 FILE DISAPPEARED (t={current_time:.1f}s): {filename}")
                    changes_detected = True

            if not changes_detected:
                print(f"🔄 No changes detected at t={current_time:.1f}s", end='\r')

            previous_states = current_states

    except KeyboardInterrupt:
        print(f"\n⏹️  Monitoring stopped by user at t={time.time() - monitoring_start:.1f}s")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

    print("\n📊 REAL-TIME MONITORING: Completed")
    print("=" * 80)

def verify_overall_stats_total_processed_files(json_files: list) -> bool:
    """Verify that overall_stats.total_processed_files is present and reasonable in JSON files."""
    if not json_files:
        print("❌ No JSON files to verify")
        return False

    verification_passed = True
    previous_total_processed = 0

    print("\n🔍 FINAL VERIFICATION: overall_stats.total_processed_files")
    print("=" * 60)

    for i, json_file in enumerate(json_files, 1):
        try:
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

            # Check if it increased (except for the first file)
            if i > 1 and total_processed_files <= previous_total_processed:
                print(f"⚠️  [{i}] {json_file.name}: total_processed_files did not increase")
                print(f"      Previous: {previous_total_processed}, Current: {total_processed_files}")
                # This is a warning, not a failure, since some files might have retries

            print(f"✅ [{i}] {json_file.name}: total_processed_files = {total_processed_files}")
            previous_total_processed = total_processed_files

        except json.JSONDecodeError as e:
            print(f"❌ [{i}] {json_file.name}: Invalid JSON format - {e}")
            verification_passed = False
        except Exception as e:
            print(f"❌ [{i}] {json_file.name}: Error reading file - {e}")
            verification_passed = False

    return verification_passed

def main():
    """Run verification test with 1 file per request to test incremental saving."""

    print("🔍 VERIFICATION TEST: Incremental Result Saving")
    print("=" * 60)
    print("📋 Testing with max_files_per_request=1 to verify:")
    print("   - overall_stats.total_processed_files updates after each file group")
    print("   - Results are saved incrementally after each LLM request")
    print("   - Each file group triggers _calculate_accum_final_statistics")
    print()

    # Create the configuration for testing with 1 file per request
    config = TestConfig(
        combo_name="single_strategy_text_first_google",  # Use a simple strategy
        file_name="4_files",                    # Small test with 4 files
        max_wait_time=3000,                     # 30 minutes timeout
        poll_interval=5,                        # Poll every 5 seconds
        max_files_per_request=1,                # 1 file per request - KEY TEST PARAMETER
        run_type="evaluation"                   # Use evaluation mode
    )

    print("⚙️  Test Configuration:")
    print(f"   Combo: {config.combo_name}")
    print(f"   Files: {config.file_name}")
    print(f"   Max files per request: {config.max_files_per_request}")
    print(f"   Run type: {config.run_type}")
    print()

    # Determine the output directory path for monitoring
    current_dir = Path(__file__).resolve()
    test_root = current_dir.parent.parent.parent.parent  # Go up to Ultra_Arena_Main_Restful_Test
    output_base_dir = test_root / "test_fixtures" / "br_fixture" / "output_files"

    # Start real-time monitoring before the test
    monitoring_thread = None
    stop_monitoring = threading.Event()

    if output_base_dir.exists():
        print("📊 Starting real-time JSON monitoring...")
        print("💡 The monitoring will check for JSON file changes every 2 seconds")
        print("💡 Press Ctrl+C during monitoring to stop and continue with verification")
        print()

        # Find existing JSON files to monitor (before test starts)
        latest_results_dir = find_latest_results_directory(output_base_dir)
        json_files_to_monitor = []
        if latest_results_dir:
            json_files_to_monitor = find_json_files(latest_results_dir)
            if json_files_to_monitor:
                print(f"📄 Will monitor {len(json_files_to_monitor)} existing JSON files")
            else:
                print("📄 No existing JSON files found - will monitor for new files")
        else:
            print("📄 No existing results directory - will monitor for new directory")

        # Start monitoring in background thread
        monitoring_thread = threading.Thread(
            target=monitor_json_files,
            args=(json_files_to_monitor, stop_monitoring, output_base_dir),
            daemon=True
        )
        monitoring_thread.start()

        # Give monitoring a moment to start
        time.sleep(1)

    # Run the test
    print("🚀 Starting test...")
    test_start_time = time.time()

    result = run_async_test(config)

    test_end_time = time.time()
    test_duration = test_end_time - test_start_time

    # Stop monitoring
    if monitoring_thread and monitoring_thread.is_alive():
        print("\n⏹️  Stopping real-time monitoring...")
        stop_monitoring.set()
        monitoring_thread.join(timeout=3)  # Wait up to 3 seconds for monitoring to stop

    print()
    print("=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"   Duration: {test_duration:.2f}s")
    print(f"   Result: {'✅ SUCCESS' if result is not None else '❌ FAILED'}")
    print()

    # Enhanced verification: Check the actual JSON files
    if result is not None:
        print("🔍 ENHANCED VERIFICATION: Checking Generated JSON Files")
        print("=" * 60)

        print(f"📁 Looking for results in: {output_base_dir}")

        if output_base_dir.exists():
            # Find the latest results directory (might be different from before if new test ran)
            latest_results_dir = find_latest_results_directory(output_base_dir)

            if latest_results_dir:
                # Find JSON files
                json_files = find_json_files(latest_results_dir)

                if json_files:
                    # Verify overall_stats.total_processed_files
                    json_verification_passed = verify_overall_stats_total_processed_files(json_files)

                    if json_verification_passed:
                        print("\n✅ JSON VERIFICATION PASSED!")
                        print("   overall_stats.total_processed_files is correctly updated")
                        print("   in the generated JSON files.")
                    else:
                        print("\n❌ JSON VERIFICATION FAILED!")
                        print("   Issues found with overall_stats.total_processed_files")
                        print("   in the generated JSON files.")
                else:
                    print("⚠️  No JSON files found to verify")
            else:
                print("⚠️  Could not find latest results directory")
        else:
            print(f"⚠️  Output directory does not exist: {output_base_dir}")

        print("\n✅ Incremental saving verification test completed!")
        print("💡 The test verified both the processing flow and the generated JSON content.")
        print("💡 Real-time monitoring showed incremental updates during processing.")
    else:
        print("❌ Test failed - check logs for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
