#!/usr/bin/env python3
"""
Direct Test: Combo Processing with 4 Top Strategies and 2 Files

This script directly tests the combo processing using the main modular processing
with combo_potential_top_4_strategies and 2 files in evaluation mode.
"""

import sys
import os
import time
from pathlib import Path

# Add the main Ultra Arena module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Ultra_Arena_Main'))

from main_modular import run_file_processing

def main():
    """Run direct combo test with 4 top strategies and 2 files."""
    
    print("🚀 Ultra Arena Main - Direct Combo Test with 4 Top Strategies and 2 Files")
    print("=" * 80)
    
    # Define paths for br_fixture
    base_dir = Path(__file__).parent
    fixture_dir = base_dir / "test_fixtures" / "br_fixture"
    
    input_pdf_dir_path = fixture_dir / "input_files" / "2_files"
    benchmark_file_path = fixture_dir / "benchmark_files" / "benchmark_252.csv"
    output_dir = fixture_dir / "output_files"
    
    print(f"📁 Input Directory: {input_pdf_dir_path}")
    print(f"📁 Output Directory: {output_dir}")
    print(f"📊 Benchmark File: {benchmark_file_path}")
    print(f"🎯 Combo Name: combo_potential_top_4_strategies")
    print(f"📄 Files: 2 files")
    print(f"⚙️  Run Type: Evaluation")
    print()
    
    # Verify files exist
    if not input_pdf_dir_path.exists():
        print(f"❌ Error: Input directory does not exist: {input_pdf_dir_path}")
        return
    
    if not benchmark_file_path.exists():
        print(f"❌ Error: Benchmark file does not exist: {benchmark_file_path}")
        return
    
    # List the input files
    pdf_files = list(input_pdf_dir_path.glob("*.pdf"))
    print(f"📄 Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"   - {pdf_file.name}")
    print()
    
    try:
        print("🔄 Starting combo processing...")
        start_time = time.time()
        
        # Run the combo processing
        result = run_file_processing(
            input_pdf_dir_path=input_pdf_dir_path,
            benchmark_eval_mode=True,
            output_file=str(output_dir / "combo_results_4_strategies_2_files.json"),
            checkpoint_file=str(output_dir / "combo_checkpoint_4_strategies_2_files.pkl"),
            streaming=False,
            max_files_per_request=10
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n✅ Combo processing completed in {elapsed_time:.2f} seconds")
        print("📄 Results:")
        print(f"   - Status: {result.get('status', 'unknown')}")
        print(f"   - Total files processed: {result.get('total_files_processed', 0)}")
        print(f"   - Successful files: {result.get('successful_files', 0)}")
        print(f"   - Failed files: {result.get('failed_files', 0)}")
        
        if 'results' in result:
            print(f"   - Detailed results: {len(result['results'])} entries")
        
        print(f"\n📁 Output files saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
