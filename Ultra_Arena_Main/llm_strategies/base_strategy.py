"""
Base processing strategy abstract class.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
try:
    from Ultra_Arena_Main.common.benchmark_comparator import BenchmarkComparator
except Exception:
    BenchmarkComparator = None

class BaseProcessingStrategy(ABC):
    """Abstract base class for all processing strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base strategy with configuration."""
        self.config = config
        self.mandatory_keys = config.get("mandatory_keys", [])
        self.num_retry_for_mandatory_keys = config.get("num_retry_for_mandatory_keys", 2)
        self.max_retries = config.get("max_retries", 2)
        self.retry_delay_seconds = config.get("retry_delay_seconds", 1)
        
        # Initialize benchmark comparator if available
        self.benchmark_comparator = None
        if "benchmark_data" in config and BenchmarkComparator is not None:
            self.benchmark_comparator = BenchmarkComparator(config["benchmark_data"])
    
    @abstractmethod
    def process_file_group(self, *, config_manager: Any = None, file_group: List[str], group_index: int, 
                          group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Process a group of files using the specific strategy."""
        pass
    
    def check_mandatory_keys(self, result: Dict[str, Any], file_path: str = None, benchmark_comparator = None) -> Tuple[bool, List[str]]:
        """Check if all mandatory keys are present in the result.

        Args:
            result: Dictionary containing extracted data to validate
            file_path: Optional file path for benchmark comparison
            benchmark_comparator: Optional benchmark comparator instance

        Returns:
            Tuple of (is_valid, missing_keys_list)
        """
        # Filter out empty strings and whitespace-only strings from mandatory keys
        # THIS IS BAD, BUT IT'S FOR THE MOMENT, WE NEED TO GET THE DESENSITIZATION CONFIG FROM THE CONFIG MANAGER BUT I CANT FIND IT WITHOUT SIGNIFICANT CODE RESTRUCTURING
        from llm_strategies.data_sensitization import resensitize_data
        result = resensitize_data(result)

        filtered_mandatory_keys = [key for key in self.mandatory_keys if key and key.strip()]
        
        if not filtered_mandatory_keys:
            logging.info("✅ No valid mandatory keys to validate - skipping validation")
            return True, []
        
        # Skip validation for 'Outros' documents
        if result.get('DOC_TYPE') == 'Outros':
            return True, []  # Skip validation for 'Outros' documents
        
        present_keys = []
        missing_keys = []
        
        # for key in filtered_mandatory_keys:
        #     value = result.get(key)
        #     if value is not None and value != "" and value != "Not found":
        #         present_keys.append(key)
        #     else:
        #         missing_keys.append(key)
        
        # Log validation results
        if present_keys:
            logging.info(f"🎯 Some present key values match benchmark: {present_keys}")
        if missing_keys:
            logging.warning(f"⚠️ Some present key values don't match benchmark: {missing_keys}")
        
        # Check if all mandatory keys are present
        if missing_keys:
            logging.warning(f"⚠️ Missing mandatory keys: {missing_keys}. Present keys: {present_keys}")
            return False, missing_keys

        # If we have a benchmark comparator, check if values match
        # if benchmark_comparator and file_path:
        #     file_has_errors = False
        #     for key in present_keys:
        #         extracted_value = result.get(key)
        #         benchmark_value = benchmark_comparator.get_benchmark_value(file_path, key)
                
        #         if benchmark_value is not None:
        #             if not benchmark_comparator._values_match(extracted_value, benchmark_value):
        #                 logging.info(f"🔍 Value mismatch for {key} in {file_path}: benchmark='{benchmark_value}' vs extracted='{extracted_value}'")
        #                 file_has_errors = True
            
            if file_has_errors:
                return False, ["value_mismatch"]
        
        return True, []
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay_seconds * (2 ** attempt)
                    logging.warning(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"❌ All {self.max_retries + 1} attempts failed. Last error: {e}")
        
        raise last_exception 