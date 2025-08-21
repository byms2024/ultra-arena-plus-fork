#!/usr/bin/env python3
"""
Simple Test: Asynchronous Combo Processing - Combo1 with 2 Files

This script tests the asynchronous combo processing endpoint against the REST API
with combo1 combo and 2_files directory in evaluation mode.
"""

import requests
import json
import sys
import os
import time
from pathlib import Path

def main():
    """Run async test with combo1 and 2 files."""
    
    # Hardcoded configuration for combo1 with 2 files
    combo_name = "combo1"
    file_name = "2_files"
    
    print(f"🚀 Ultra Arena Main - Async Test: Combo1 with 2 Files")
    print(f"Combo: {combo_name}")
    print(f"Files: {file_name}")
    print("=" * 80)
    
    # Configuration
    base_url = "http://localhost:5002"
    api_endpoint = f"{base_url}/api/process/combo/async"
    
    # Get the current script directory
    script_dir = Path(__file__).parent.parent.parent.parent
    test_fixtures_dir = script_dir / "test_fixtures" / "br_fixture"
    input_pdf_dir_path = test_fixtures_dir / "input_files" / file_name
    
    if not input_pdf_dir_path.exists():
        print(f"❌ Error: Input directory not found: {input_pdf_dir_path}")
        sys.exit(1)
    
    # Prepare the request payload
    payload = {
        'combo_name': combo_name,
        'input_pdf_dir_path': str(input_pdf_dir_path),
        'run_type': 'evaluation',  # Benchmark evaluation mode
        'output_dir': str(test_fixtures_dir / "output_files"),
        'benchmark_file_path': str(test_fixtures_dir / "benchmark_files" / "benchmark_252.csv")
    }
    
    print(f"📁 Input Directory: {input_pdf_dir_path}")
    print(f"📊 Benchmark File: {payload['benchmark_file_path']}")
    print(f"📤 Output Directory: {payload['output_dir']}")
    print()
    
    try:
        # Submit the async task
        print("🔄 Submitting async task...")
        response = requests.post(api_endpoint, json=payload, timeout=30)
        
        if response.status_code == 202:
            print("✅ Task submitted successfully!")
            task_id = response.json().get('request_id')
            print(f"📋 Task ID: {task_id}")
            
            # Poll for status updates
            status_endpoint = f"{base_url}/api/requests/{task_id}"
            max_wait_time = 300  # 5 minutes
            poll_interval = 5    # 5 seconds
            elapsed_time = 0
            
            print("\n🔄 Polling for task completion...")
            while elapsed_time < max_wait_time:
                try:
                    status_response = requests.get(status_endpoint, timeout=10)
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data.get('status')
                        progress = status_data.get('progress', 0)
                        
                        print(f"⏱️  Elapsed: {elapsed_time}s | Status: {status} | Progress: {progress}%")
                        
                        if status == 'completed':
                            print("✅ Task completed successfully!")
                            result = status_data.get('result', {})
                            print(f"📊 Results: {json.dumps(result, indent=2)}")
                            break
                        elif status == 'failed':
                            print("❌ Task failed!")
                            error = status_data.get('error', 'Unknown error')
                            print(f"🚨 Error: {error}")
                            break
                        elif status == 'running':
                            # Continue polling
                            pass
                        else:
                            print(f"⚠️  Unknown status: {status}")
                            break
                    else:
                        print(f"⚠️  Status check failed: {status_response.status_code}")
                        break
                        
                except requests.exceptions.RequestException as e:
                    print(f"⚠️  Status check error: {e}")
                    break
                
                time.sleep(poll_interval)
                elapsed_time += poll_interval
            
            if elapsed_time >= max_wait_time:
                print("⏰ Timeout: Task did not complete within the expected time")
                
        else:
            print(f"❌ Task submission failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
