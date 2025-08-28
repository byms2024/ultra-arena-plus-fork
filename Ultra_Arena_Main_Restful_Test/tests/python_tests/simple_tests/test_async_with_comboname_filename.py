#!/usr/bin/env python3
"""
Simple Test: Asynchronous Combo Processing with Configurable Combo Name and File Directory

This script tests the asynchronous combo processing endpoint against the REST API
with configurable file directory in evaluation mode using the br_fixture configuration.
Both combo_name and file_name can be specified as command line arguments.
The script submits a task and then polls for status updates until completion.
"""

import requests
import json
import sys
import os
import time
import argparse
from pathlib import Path

def main():
    """Run async test with configurable combo name and file directory."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test async processing with configurable combo name and file directory')
    parser.add_argument('--combo-name', 
                       default='combo_potential_top_4_strategies',
                       help='Combo name to use (default: combo_potential_top_4_strategies)')
    parser.add_argument('--file-name',
                       default='2_files',
                       help='File directory name to use (default: 2_files)')
    parser.add_argument('--poll-interval',
                       type=int,
                       default=5,
                       help='Polling interval in seconds (default: 5)')
    parser.add_argument('--max-wait',
                       type=int,
                       default=3000,
                       help='Maximum wait time in seconds (default: 300)')
    args = parser.parse_args()
    
    combo_name = args.combo_name
    file_name = args.file_name
    poll_interval = args.poll_interval
    max_wait = args.max_wait
    
    print(f"🚀 Ultra Arena Main - Async Test with Combo: {combo_name}, Files: {file_name}")
    print("=" * 75)
    
    # Define paths for br_fixture with configurable file directory
    base_dir = Path(__file__).parent.parent.parent.parent
    fixture_dir = base_dir / "test_fixtures" / "br_fixture"
    
    # Use configurable file directory
    input_pdf_dir_path = fixture_dir / "input_files" / file_name
    benchmark_file_path = fixture_dir / "benchmark_files" / "benchmark_252.csv"
    
    # Use fixture's output directory
    output_dir = fixture_dir / "output_files"
    
    print(f"📁 Input Directory: {input_pdf_dir_path}")
    print(f"📁 Output Directory: {output_dir}")
    print(f"📊 Benchmark File: {benchmark_file_path}")
    print(f"🎯 Combo Name: {combo_name}")
    print(f"📄 File Directory: {file_name}")
    print(f"⚙️  Run Type: Evaluation")
    print(f"⏱️  Poll Interval: {poll_interval}s")
    print(f"⏰ Max Wait Time: {max_wait}s")
    print()
    
    MAX_NUM_FILES_PER_REQUEST = 1
    MAX_CC_STRATEGIES = 5
    MAX_CC_FILEGROUPS = 1
    
    # Request data with configurable combo name and file directory
    data = {
        'combo_name': combo_name,
        'input_pdf_dir_path': str(input_pdf_dir_path),
        'output_dir': str(output_dir),
        'benchmark_file_path': str(benchmark_file_path),
        'run_type': 'evaluation',  # Benchmark evaluation mode
        'streaming': False,
        'max_cc_strategies': MAX_CC_STRATEGIES,
        'max_cc_filegroups': MAX_CC_FILEGROUPS,
        'max_files_per_request': MAX_NUM_FILES_PER_REQUEST  # Process files based on directory
    }
    
    try:
        # Step 1: Submit async task
        print("🔄 Submitting async task to /api/process/combo/async...")
        response = requests.post(
            'http://localhost:5002/api/process/combo/async',
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=30  # 30 seconds timeout for submission
        )
        
        print(f"✅ Response Status: {response.status_code}")
        
        if response.status_code == 202:
            result = response.json()
            print("📄 Async Response:")
            print(json.dumps(result, indent=2))
            
            # Extract request ID
            request_id = result.get('request_id')
            if not request_id:
                print("❌ Error: No request_id in response")
                return
            
            print(f"\n🎯 Request ID: {request_id}")
            print(f"📊 Status: {result.get('status', 'unknown')}")
            print()
            
            # Step 2: Poll for status updates
            print(f"⏳ Polling for status updates every {poll_interval} seconds...")
            start_time = time.time()
            
            while True:
                # Check if we've exceeded max wait time
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait:
                    print(f"⏰ Timeout: Exceeded maximum wait time of {max_wait} seconds")
                    break
                
                # Get status
                status_response = requests.get(
                    f'http://localhost:5002/api/requests/{request_id}',
                    timeout=10
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get('status', 'unknown')
                    progress = status_data.get('progress', {})
                    
                    # Print progress
                    print(f"⏱️  Elapsed: {elapsed_time:.1f}s | Status: {status}")
                    if isinstance(progress, dict):
                        total_files = progress.get('total_files_of_all_strategies_to_process', 0)
                        processed_files = progress.get('total_files_of_all_strategies_processed', 0)
                        if total_files and total_files > 0:
                            percentage = (processed_files / total_files) * 100
                            print(f"📊 Progress: {processed_files}/{total_files} files ({percentage:.1f}%)")
                    elif isinstance(progress, (int, float)):
                        print(f"📊 Progress: {progress}%")
                    
                    # Check if completed
                    if status == 'complete':
                        print("\n🎉 Task completed successfully!")
                        print("📄 Final Results:")
                        print(json.dumps(status_data, indent=2))
                        break
                    elif status == 'failed':
                        print(f"\n❌ Task failed: {status_data.get('error', 'Unknown error')}")
                        break
                    elif status in ['queued', 'processing', 'incomplete']:
                        print(f"⏳ Still {status}...")
                    else:
                        print(f"❓ Unknown status: {status}")
                        break
                else:
                    print(f"❌ Error getting status: {status_response.status_code}")
                    print(f"Response: {status_response.text}")
                    break
                
                # Wait before next poll
                time.sleep(poll_interval)
                
        else:
            print(f"❌ Error submitting task: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the REST server is running on port 5002")
        print("   Start it with: cd Ultra_Arena_Main_Restful && export RUN_PROFILE=br_profile_restful && nohup python server.py > server.log 2>&1 &")
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: Request took too long to complete")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
    finally:
        # Output files are saved to fixture's output directory
        print(f"\n📁 Output files generated in: {output_dir}")
        print("💡 Files are saved to the fixture's output directory for inspection")

if __name__ == "__main__":
    main()
