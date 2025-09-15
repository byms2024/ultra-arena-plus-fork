"""
Benchmark tracker for comparing extracted values against benchmark data.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


class BenchmarkTracker:
    """Tracks benchmark comparisons and errors."""
    
    def __init__(self, benchmark_comparator=None, csv_output_file: str | None = None):
        self.benchmark_comparator = benchmark_comparator
        self.csv_output_file = csv_output_file
        
        # Benchmark error tracking
        self.total_unmatched_fields = 0
        self.total_unmatched_files = 0
        self.unmatched_fields_data = []  # List to store unmatched field details
    
    def track_benchmark_error(self, file_path: str, field_name: str, benchmark_value: str, extracted_value: str):
        """Track a benchmark error for a specific field."""
        self.total_unmatched_fields += 1
        
        error_data = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'field_name': field_name,
            'benchmark_value': benchmark_value,
            'extracted_value': extracted_value,
            'error_type': 'value_mismatch' if extracted_value else 'missing_field'
        }
        
        self.unmatched_fields_data.append(error_data)
        
        logging.info(f"🔍 Benchmark error: {os.path.basename(file_path)} - {field_name}: "
                    f"benchmark='{benchmark_value}' vs extracted='{extracted_value}'")
    
    def track_file_benchmark_errors(self, file_path: str):
        """Track that a file has benchmark errors."""
        self.total_unmatched_files += 1
    
    def check_file_benchmark_errors(self, file_path: str, result: Dict[str, Any]):
        """Check for benchmark errors in a processed file and track them."""
        if not self.benchmark_comparator:
            return
        
        # Get the model output from the result
        model_output = result.get('file_model_output', result)
        if not model_output:
            return
        
        # Get mandatory keys from active profile via config_base
        try:
            from config.config_base import MANDATORY_KEYS as _MANDATORY_KEYS
            mandatory_keys = _MANDATORY_KEYS or []
        except Exception:
            mandatory_keys = []
        
        # Check if this is a failed file (has failure_reason in file_process_result)
        file_process_result = result.get('file_process_result', {})
        if file_process_result.get('success') == False:
            # This is a failed file - check each mandatory field individually
            file_has_errors = False
            for key in mandatory_keys:
                # Get benchmark value for this file and field
                benchmark_value = self.benchmark_comparator.get_benchmark_value(file_path, key)
                if benchmark_value is None:
                    continue
                
                # Get the actual extracted value from model output
                extracted_value = model_output.get(key)
                
                # If the field was successfully extracted, use the actual value
                # Otherwise use empty string for null/missing values
                if extracted_value is not None and extracted_value != "" and extracted_value != "Not found":
                    # Field was extracted but may not match benchmark
                    if not self.benchmark_comparator._values_match(extracted_value, benchmark_value):
                        self.track_benchmark_error(file_path, key, benchmark_value, extracted_value)
                        file_has_errors = True
                else:
                    # Field was not extracted - use empty string
                    self.track_benchmark_error(file_path, key, benchmark_value, "")
                    file_has_errors = True
            
            # Track file if it has any errors
            if file_has_errors:
                self.track_file_benchmark_errors(file_path)
            
            logging.info(f"🔍 File failed processing, checked {len([k for k in mandatory_keys if self.benchmark_comparator.get_benchmark_value(file_path, k) is not None])} mandatory fields: {os.path.basename(file_path)}")
            return
        
        # Check each mandatory key against benchmark for successfully processed files
        file_has_errors = False
        for key in mandatory_keys:
            extracted_value = model_output.get(key)
            
            # Get benchmark value for this file and field
            benchmark_value = self.benchmark_comparator.get_benchmark_value(file_path, key)
            if benchmark_value is None:
                continue
            
            # Check if extracted value is missing/null/empty
            if extracted_value is None or extracted_value == "" or extracted_value == "Not found":
                # Track missing mandatory field as benchmark error
                self.track_benchmark_error(file_path, key, benchmark_value, "")
                file_has_errors = True
                continue
            
            # Check if values match
            if not self.benchmark_comparator._values_match(extracted_value, benchmark_value):
                self.track_benchmark_error(file_path, key, benchmark_value, extracted_value)
                file_has_errors = True
        
        # Track file if it has any errors
        if file_has_errors:
            self.track_file_benchmark_errors(file_path)
    
    def generate_error_csv(self):
        """Generate error CSV file with unmatched field details."""
        if not self.unmatched_fields_data:
            logging.info("📊 No benchmark errors to save")
            return
        
        try:
            # Create errors directory as sibling to csv directory
            csv_dir = Path(self.csv_output_file).parent if self.csv_output_file else Path("output/results/non-batch/csv")
            errors_dir = csv_dir.parent / "errors"
            errors_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate errors CSV filename
            csv_filename = Path(self.csv_output_file).name if self.csv_output_file else "errors.csv"
            errors_csv_path = errors_dir / f"errors_{csv_filename}"
            
            # Create DataFrame and save to CSV
            df_errors = pd.DataFrame(self.unmatched_fields_data)
            df_errors.to_csv(errors_csv_path, index=False)
            
            logging.info(f"💾 Error CSV file saved: {errors_csv_path}")
            
        except Exception as e:
            logging.error(f"❌ Failed to save error CSV: {e}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get benchmark error statistics."""
        return {
            'total_unmatched_fields': self.total_unmatched_fields,
            'total_unmatched_files': self.total_unmatched_files,
            'unmatched_fields_data': self.unmatched_fields_data
        } 