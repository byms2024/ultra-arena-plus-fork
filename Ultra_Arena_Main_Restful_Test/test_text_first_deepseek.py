#!/usr/bin/env python3
"""
Test Script for Text First Strategy with DeepSeek

This script tests the text first strategy using DeepSeek against the REST endpoint
with 1 file in evaluation mode using the br_fixture.
"""

import requests
import json
import sys
import os
import tempfile
from pathlib import Path

def main():
    """Run text first strategy with DeepSeek test."""
    
    print("🚀 Ultra Arena Main - Text First Strategy with DeepSeek Test")
    print("=" * 70)
    
    # Define paths for br_fixture
    base_dir = Path(__file__).parent
    fixture_dir = base_dir / "test_fixtures" / "br_fixture"
    
    input_pdf_dir_path = fixture_dir / "input_files" / "1_file"
    benchmark_file_path = fixture_dir / "benchmark_files" / "benchmark_252.csv"
    
    # Create temporary output directory
    output_dir = tempfile.mkdtemp(prefix="test_text_first_deepseek_")
    
    print(f"📁 Input Directory: {input_pdf_dir_path}")
    print(f"📁 Output Directory: {output_dir}")
    print(f"📊 Benchmark File: {benchmark_file_path}")
    print(f"🎯 Strategy: Text First with DeepSeek")
    print(f"⚙️  Run Type: Evaluation")
    print(f"📄 Files: 1 file")
    print()
    
    # Request data for text first strategy with DeepSeek
    data = {
        'combo_name': 'combo_test_deepseek_strategies',  # Text first + DeepSeek combo
        'input_pdf_dir_path': str(input_pdf_dir_path),
        'output_dir': str(output_dir),
        'benchmark_file_path': str(benchmark_file_path),
        'run_type': 'evaluation',  # Benchmark evaluation mode
        'streaming': False,
        'max_cc_strategies': 1,  # Single strategy
        'max_cc_filegroups': 1,  # Single file group
        'max_files_per_request': 1  # Single file
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
                
        else:
            print(f"❌ Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the REST server is running on port 5002")
        print("   Start it with: cd Ultra_Arena_Main_Restful && python server.py")
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: Request took too long to complete")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
    finally:
        # Clean up temporary directory
        try:
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                print(f"🧹 Cleaned up temporary output directory: {output_dir}")
        except Exception as e:
            print(f"⚠️ Failed to clean up temporary output directory {output_dir}: {e}")

if __name__ == "__main__":
    main()
