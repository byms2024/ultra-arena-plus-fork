"""
Base processing strategy abstract class.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from common.benchmark_comparator import BenchmarkComparator

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
        if "benchmark_data" in config:
            self.benchmark_comparator = BenchmarkComparator(config["benchmark_data"])
    
    @abstractmethod
    def process_file_group(self, *, file_group: List[str], group_index: int, 
                          group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Process a group of files using the specific strategy."""
        pass
    
    def check_mandatory_keys(self, result: Dict[str, Any], file_path: str = None, benchmark_comparator = None, database_ops = None) -> Tuple[bool, List[str]]:
        """Check if all mandatory keys are present in the result.

        Args:
            result: Dictionary containing extracted data to validate
            file_path: Optional file path for benchmark comparison
            benchmark_comparator: Optional benchmark comparator instance
            database_ops: Optional database operations instance for DMS validation

        Returns:
            Tuple of (is_valid, missing_keys_list)
        """
        # Filter out empty strings and whitespace-only strings from mandatory keys
        # THIS IS BAD, BUT IT'S FOR THE MOMENT, WE NEED TO GET THE DESENSITIZATION CONFIG FROM THE CONFIG MANAGER BUT I CANT FIND IT WITHOUT SIGNIFICANT CODE RESTRUCTURING
        from llm_strategies.data_sensitization import resensitize_data
        result = resensitize_data(result)

        # Get DMS values if database_ops is provided and we can extract claim_id
        dms_values = {}
        if database_ops:
            claim_id = None
            # Try to extract claim_id from result
            if 'CLAIM_ID' in result:
                claim_id = result['CLAIM_ID']
            elif 'CLAIM_NUMBER' in result:
                # Try to lookup claim_id from claim number
                claim_no = result['CLAIM_NUMBER']
                if hasattr(database_ops, 'get_claim_id_by_claim_no'):
                    claim_id = database_ops.get_claim_id_by_claim_no(str(claim_no))
            elif 'CLAIM_NO' in result:
                # Alternative field name
                claim_no = result['CLAIM_NO']
                if hasattr(database_ops, 'get_claim_id_by_claim_no'):
                    claim_id = database_ops.get_claim_id_by_claim_no(str(claim_no))

            if claim_id:
                try:
                    dms_values = database_ops.get_dms_key_values(int(claim_id))
                    if dms_values:
                        logging.info(f"✅ Retrieved DMS values for claim {claim_id}: {list(dms_values.keys())}")
                    else:
                        logging.warning(f"⚠️ No DMS values found for claim {claim_id}")
                except Exception as e:
                    logging.error(f"❌ Error retrieving DMS values for claim {claim_id}: {e}")
            else:
                logging.debug("No claim_id found in result for DMS validation")

        filtered_mandatory_keys = [key for key in self.mandatory_keys if key and key.strip()]
        
        if not filtered_mandatory_keys:
            logging.info("✅ No valid mandatory keys to validate - skipping validation")
            return True, []
        
        # Skip validation for 'Outros' documents
        if result.get('DOC_TYPE') == 'Outros':
            return True, []  # Skip validation for 'Outros' documents
        
        present_keys = []
        missing_keys = []
        
        for key in filtered_mandatory_keys:
            value = result.get(key)
            if value is not None and value != "" and value != "Not found":
                present_keys.append(key)
            else:
                missing_keys.append(key)
        
        # Log validation results
        if present_keys:
            logging.info(f"🎯 Some present key values match benchmark: {present_keys}")
        if missing_keys:
            logging.warning(f"⚠️ Some present key values don't match benchmark: {missing_keys}")
        
        # Check if all mandatory keys are present
        if missing_keys:
            logging.warning(f"⚠️ Missing mandatory keys: {missing_keys}. Present keys: {present_keys}")
            return False, missing_keys

        # If we have DMS values, validate extracted values against DMS data
        if dms_values:
            dms_validation_errors = []
            for key in present_keys:
                extracted_value = result.get(key)
                dms_value = dms_values.get(key)

                if dms_value is not None and extracted_value is not None:
                    # Compare values (with some tolerance for numeric values)
                    if key in ['GROSS_CREDIT', 'LABOUR_AMOUNT', 'PART_AMOUNT']:
                        try:
                            # Convert to float for comparison with small tolerance
                            extracted_num = float(str(extracted_value).replace('R$', '').replace(',', '.').replace(' ', ''))
                            dms_num = float(str(dms_value))
                            tolerance = 0.01  # 1% tolerance

                            if abs(extracted_num - dms_num) > (dms_num * tolerance):
                                logging.info(f"🔍 DMS value mismatch for {key}: extracted='{extracted_value}' vs DMS='{dms_value}' (tolerance: {tolerance*100}%)")
                                dms_validation_errors.append(f"DMS_{key}_MISMATCH")
                        except (ValueError, TypeError) as e:
                            # If numeric comparison fails, do string comparison
                            if str(extracted_value).strip() != str(dms_value).strip():
                                logging.info(f"🔍 DMS value mismatch for {key}: extracted='{extracted_value}' vs DMS='{dms_value}'")
                                dms_validation_errors.append(f"DMS_{key}_MISMATCH")
                    elif str(extracted_value).strip() != str(dms_value).strip():
                        logging.info(f"🔍 DMS value mismatch for {key}: extracted='{extracted_value}' vs DMS='{dms_value}'")
                        dms_validation_errors.append(f"DMS_{key}_MISMATCH")
                    else:
                        logging.debug(f"✅ DMS validation passed for {key}: '{extracted_value}'")

            if dms_validation_errors:
                logging.warning(f"⚠️ DMS validation found {len(dms_validation_errors)} mismatches")
                # Note: We're not failing validation here, just logging for now
                # Could be made configurable to fail on DMS mismatches if needed

        # If we have a benchmark comparator, check if values match
        if benchmark_comparator and file_path:
            file_has_errors = False
            for key in present_keys:
                extracted_value = result.get(key)
                benchmark_value = benchmark_comparator.get_benchmark_value(file_path, key)
                
                if benchmark_value is not None:
                    if not benchmark_comparator._values_match(extracted_value, benchmark_value):
                        logging.info(f"🔍 Value mismatch for {key} in {file_path}: benchmark='{benchmark_value}' vs extracted='{extracted_value}'")
                        file_has_errors = True
            
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