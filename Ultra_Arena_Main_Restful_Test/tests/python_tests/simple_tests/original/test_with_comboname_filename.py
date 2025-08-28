#!/usr/bin/env python3
"""
Simple Test: Configurable Combo Name and File Directory

This script tests any combo strategy against the REST endpoint
with configurable file directory in evaluation mode using the br_fixture configuration.
Both combo_name and file_name can be specified as command line arguments.
"""

import requests
import json
import sys
import os
import tempfile
import argparse
from pathlib import Path

from Ultra_Arena_Main.config.config_base import MAX_NUM_FILES_PER_REQUEST

def main():
    """Run test with configurable combo name and file directory."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test with configurable combo name and file directory')
    parser.add_argument('--combo-name', 
                       default='combo_test_deepseek_strategies',
                       help='Combo name to use (default: combo_test_deepseek_strategies)')
    parser.add_argument('--file-name',
                       default='1_file',
                       help='File directory name to use (default: 1_file)')
    args = parser.parse_args()
    
    combo_name = args.combo_name
    file_name = args.file_name
    
    print(f"🚀 Ultra Arena Main - Test with Combo: {combo_name}, Files: {file_name}")
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
    print()
    
    MAX_NUM_FILES_PER_REQUEST = 10
    MAX_CC_STRATEGIES = 5
    MAX_CC_FILEGROUPS = 5
    
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
        # Make the request to the combo processing endpoint
        print("🔄 Making request to /api/process/combo...")
        response = requests.post(
            'http://localhost:5002/api/process/combo',
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=300  # 5 minutes timeout
        )
        
        print(f"✅ Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("📄 Response:")
            print(json.dumps(result, indent=2))
            
            # Extract key information
            if 'combo_name' in result:
                print(f"\n🎯 Combo Used: {result['combo_name']}")
            if 'output_dir' in result:
                print(f"📁 Output Directory: {result['output_dir']}")
            if 'status' in result:
                print(f"📊 Status: {result['status']}")
            if 'benchmark_evaluation' in result:
                print(f"📈 Benchmark Evaluation: {result['benchmark_evaluation']}")
            if 'processing_summary' in result:
                print(f"📋 Processing Summary: {result['processing_summary']}")
            if 'num_files_to_process' in result:
                print(f"📄 Files to Process: {result['num_files_to_process']}")
                
        else:
            print(f"❌ Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the REST server is running on port 5002")
        print("   Start it with: cd Ultra_Arena_Main_Restful && export RUN_PROFILE=br_profile_restful && nohup python server.py > server.log 2>&1 &")
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: Request took too long to complete")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
    finally:
        # Output files are saved to fixture's output directory
        print(f"📁 Output files generated in: {output_dir}")
        print("💡 Files are saved to the fixture's output directory for inspection")

if __name__ == "__main__":
    main()
