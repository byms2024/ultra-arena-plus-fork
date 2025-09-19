import os
import logging
import hashlib
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple

from llm_strategies.data_sensitization import resensitize_data

# Import default output directory and mandatory keys
try:
    from config.config_base import DEFAULT_OUTPUT_DIR, MANDATORY_KEYS
except ImportError:
    DEFAULT_OUTPUT_DIR = "output/results/csv"
    MANDATORY_KEYS = []

class CSVResultDumper:
    """Handles dumping processing results to CSV files."""
    
    def __init__(self, output_dir: str = None, custom_filename: str = None):
        """Initialize the CSV dumper."""
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR
        
        self.output_dir = output_dir
        self.global_df = pd.DataFrame()
        self.output_file = None
        self.file_retry_rounds = {}  # Track retry rounds for each file
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate output filename
        if custom_filename:
            self.output_file = os.path.join(self.output_dir, custom_filename)
        else:
            # Generate output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = os.path.join(self.output_dir, f"processing_results_{timestamp}.csv")
        
        logging.info(f"📊 Initialized CSV dumper with output file: {self.output_file}")
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate MD5 hash for file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logging.warning(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def _check_mandatory_keys(self, result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if all mandatory keys are present and non-empty."""
        print("=========result===========\n")
        print(result)
        result = resensitize_data(result)
        print("=========Sensitized result===========\n")
        print(result)
        if not result or not isinstance(result, dict):
            return False, MANDATORY_KEYS
        
        # Skip validation for 'Outros' documents
        if result.get('DOC_TYPE') == 'Outros':
            return True, []  # Skip validation for 'Outros' documents
        
        # Filter out empty strings and whitespace-only strings from mandatory keys
        filtered_mandatory_keys = [key for key in MANDATORY_KEYS if key and key.strip()]
        
        # If no valid mandatory keys, return success
        if not filtered_mandatory_keys:
            logging.info("✅ No valid mandatory keys to validate - skipping validation")
            return True, []
        
        missing_keys = []
        present_keys = []
        for key in filtered_mandatory_keys:
            value = result.get(key)
            if value is None or value == "" or value == "Not found":
                missing_keys.append(key)
            else:
                present_keys.append(key)
        
        # Log the status of mandatory keys
        if len(missing_keys) == 0:
            logging.info(f"✅ All mandatory keys present: {present_keys}")
        else:
            logging.warning(f"⚠️ Missing mandatory keys: {missing_keys}. Present keys: {present_keys}")
        
        return len(missing_keys) == 0, missing_keys
    
    def dump_group_results(self, group_results: List[tuple], group_id: str) -> None:
        """
        Dump group results to CSV and update global DataFrame.
        
        Args:
            group_results: List of tuples (file_path, result_dict)
            group_id: Identifier for the group
        """
        logging.info(f"📊 Dumping results for group {group_id} to CSV...")
        
        # Process each file result in the group
        for file_path, result in group_results:
            self._process_single_result(file_path, result, group_id)
        
        # Save the updated global DataFrame to CSV
        self._save_to_csv()
        
        logging.info(f"✅ Group {group_id} results dumped to CSV. Total records: {len(self.global_df)}")
    
    def _process_single_result(self, file_path: str, result: Dict[str, Any], group_id: str) -> None:
        """
        Process a single file result and add it to the global DataFrame.
        
        Args:
            file_path: Path to the processed file
            result: Result dictionary containing extracted data
            group_id: Identifier for the group
        """
        # Check if result is None
        if result is None:
            logging.error(f"❌ CSV processing failed: Result is None for file {file_path}")
            return
        
        # Handle different result structures
        # New structure: result contains 'file_model_output', 'file_token_stats', 'file_info'
        # Legacy structure: result contains 'model_output', 'file_info', etc.
        
        # Extract basic information
        if 'file_model_output' in result:
            # New structure
            file_process_result = result.get('file_process_result', {})
            success = file_process_result.get('success', False)
            model_output = result.get('file_model_output', {})
            file_info = result.get('file_info', {})
            timestamp = file_process_result.get('proc_timestamp', '')
            retry_round = file_process_result.get('retry_round', None)
            
            # Create a row for the DataFrame
            row_data = {
                'file_path': file_path,
                'file_name': file_info.get('file_name', ''),
                'group_id': group_id,
                'success': success,
                'timestamp': timestamp,
                'retry_round': retry_round,
                'file_size_mb': file_info.get('file_size_mb', 0),
                'estimated_tokens': file_info.get('estimated_tokens', 0),
                'file_hash': file_info.get('file_hash', ''),
                'modification_time': file_info.get('modification_time', ''),
                # Token counts are only available at group/overall level, not file level
                'prompt_token_count': 0,
                'candidates_token_count': 0,
                'total_token_count': 0,
                'other_token_count': 0,
            }
            
            # Add model output fields
            if model_output and isinstance(model_output, dict):
                row_data.update({
                    'file_name_llm': model_output.get('file_name_llm', ''),
                    'DOC_TYPE': model_output.get('DOC_TYPE', ''),
                    'CNPJ_1': model_output.get('CNPJ_1', ''),
                    'CNPJ_2': model_output.get('CNPJ_2', ''),
                    'VALOR_TOTAL': model_output.get('VALOR_TOTAL', ''),
                    'Chassi': model_output.get('Chassi', ''),
                    'CLAIM_NUMBER': model_output.get('CLAIM_NUMBER', ''),
                    # Extract token fields from file_model_output
                    'prompt_token_count': model_output.get('prompt_token_count', 0),
                    'candidates_token_count': model_output.get('candidates_token_count', 0),
                    'total_token_count': model_output.get('total_token_count', 0),
                    'other_token_count': model_output.get('total_token_count', 0) - model_output.get('prompt_token_count', 0) - model_output.get('candidates_token_count', 0),
                })
            else:
                # Add empty values for model output fields if no model_output or if it's not a dict
                row_data.update({
                    'file_name_llm': '',
                    'DOC_TYPE': '',
                    'CNPJ_1': '',
                    'CNPJ_2': '',
                    'VALOR_TOTAL': '',
                    'Chassi': '',
                    'CLAIM_NUMBER': '',
                })
            
            # Add file group IDs if available
            if 'group_ids_incl_retries' in file_process_result:
                row_data['file_group_ids_incl_retries'] = ','.join(file_process_result['group_ids_incl_retries'])
            else:
                row_data['file_group_ids_incl_retries'] = group_id
                
        elif 'model_output' in result:
            # Legacy structure
            success = result.get('success', False)
            model_output = result.get('model_output', {})
            token_stats = {}  # No token stats in legacy structure
            file_info = result.get('file_info', {})
            timestamp = result.get('timestamp', '')
            retry_round = result.get('retry_round', None)
            
            # Create a row for the DataFrame
            row_data = {
                'file_path': file_path,
                'file_name': file_info.get('file_name', ''),
                'group_id': group_id,
                'success': success,
                'timestamp': timestamp,
                'retry_round': retry_round,
                'file_size_mb': file_info.get('file_size_mb', 0),
                'estimated_tokens': file_info.get('estimated_tokens', 0),
                'file_hash': file_info.get('file_hash', ''),
                'modification_time': file_info.get('modification_time', ''),
                'prompt_token_count': token_stats.get('prompt_tokens', 0),
                'candidates_token_count': token_stats.get('candidates_tokens', 0),
                'total_token_count': token_stats.get('actual_tokens', 0),
                'other_token_count': token_stats.get('other_tokens', 0),
            }
            
            # Add model output fields
            if model_output and isinstance(model_output, dict):
                row_data.update({
                    'file_name_llm': model_output.get('file_name_llm', ''),
                    'DOC_TYPE': model_output.get('DOC_TYPE', ''),
                    'CNPJ_1': model_output.get('CNPJ_1', ''),
                    'CNPJ_2': model_output.get('CNPJ_2', ''),
                    'VALOR_TOTAL': model_output.get('VALOR_TOTAL', ''),
                    'Chassi': model_output.get('Chassi', ''),
                    'CLAIM_NUMBER': model_output.get('CLAIM_NUMBER', ''),
                })
            else:
                # Add empty values for model output fields if no model_output or if it's not a dict
                row_data.update({
                    'file_name_llm': '',
                    'DOC_TYPE': '',
                    'CNPJ_1': '',
                    'CNPJ_2': '',
                    'VALOR_TOTAL': '',
                    'Chassi': '',
                    'CLAIM_NUMBER': '',
                })
            
            # Add file group IDs if available
            if 'file_group_ids_incl_retries' in file_info:
                row_data['file_group_ids_incl_retries'] = ','.join(file_info['file_group_ids_incl_retries'])
            else:
                row_data['file_group_ids_incl_retries'] = group_id
                
        else:
            # Current project structure - result contains extracted data directly
            # Check if result has error AND validate mandatory keys
            has_error = 'error' in result
            
            # Use the helper method to check mandatory keys
            has_all_keys, missing_keys = self._check_mandatory_keys(result)
            
            # Success means no error AND all mandatory keys are present
            success = not has_error and has_all_keys
            retry_round = None  # Extract retry info from group_id if available
            
            # Extract retry round from group_id if it contains retry information
            if '_retry_' in group_id:
                try:
                    retry_round = int(group_id.split('_retry_')[1].split('_')[0])
                    # Store the retry round for this file
                    self.file_retry_rounds[file_path] = retry_round
                except (IndexError, ValueError):
                    retry_round = None
            else:
                # Check if we have a stored retry round for this file
                retry_round = self.file_retry_rounds.get(file_path, None)
            
            # Get file information
            try:
                file_stat = os.stat(file_path)
                file_size_mb = round(file_stat.st_size / (1024 * 1024), 2)
                modification_time = datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                file_hash = self.get_file_hash(file_path)
            except (OSError, FileNotFoundError):
                file_size_mb = 0
                modification_time = ''
                file_hash = ''
            
            # Calculate estimated tokens based on file size and result
            estimated_tokens = 0
            if 'estimated_tokens' in result:
                estimated_tokens = result['estimated_tokens']
            else:
                # Simple estimation based on file size
                try:
                    file_size_mb_actual = os.path.getsize(file_path) / (1024 * 1024)
                    if file_size_mb_actual < 0.05:
                        estimated_tokens = 4500
                    elif file_size_mb_actual < 0.15:
                        estimated_tokens = 5000
                    else:
                        estimated_tokens = 5500
                except (OSError, FileNotFoundError):
                    estimated_tokens = 5000  # Default estimate
            
            # Create row data for current project structure
            row_data = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'group_id': group_id,
                'success': success,
                'timestamp': datetime.now().isoformat(),
                'retry_round': retry_round,
                'file_size_mb': file_size_mb,
                'estimated_tokens': estimated_tokens,
                'file_hash': file_hash,
                'modification_time': modification_time,
                'file_group_ids_incl_retries': group_id,
                'prompt_token_count': 0,  # Will be updated below if available
                'candidates_token_count': 0,  # Will be updated below if available
                'total_token_count': 0,  # Will be updated below if available
                'other_token_count': 0,  # Will be updated below if available
            }
            
            # Add extracted data fields directly from result
            if 'error' not in result:
                # Handle both new structure (file_model_output) and old structure
                if 'file_model_output' in result:
                    model_output = result['file_model_output']
                    row_data.update({
                        'file_name_llm': model_output.get('file_name_llm', ''),
                        'DOC_TYPE': model_output.get('DOC_TYPE', ''),
                        'CNPJ_1': model_output.get('CNPJ_1', ''),
                        'CNPJ_2': model_output.get('CNPJ_2', ''),
                        'VALOR_TOTAL': model_output.get('VALOR_TOTAL', ''),
                        'Chassi': model_output.get('Chassi', ''),
                        'CLAIM_NUMBER': model_output.get('CLAIM_NUMBER', ''),
                        # Extract token fields from file_model_output
                        'prompt_token_count': model_output.get('prompt_token_count', 0),
                        'candidates_token_count': model_output.get('candidates_token_count', 0),
                        'total_token_count': model_output.get('total_token_count', 0),
                        'other_token_count': model_output.get('total_token_count', 0) - model_output.get('prompt_token_count', 0) - model_output.get('candidates_token_count', 0),
                    })
                else:
                    # Old structure - extract directly from result
                    row_data.update({
                        'file_name_llm': result.get('file_name_llm', ''),
                        'DOC_TYPE': result.get('DOC_TYPE', ''),
                        'CNPJ_1': result.get('CNPJ_1', ''),
                        'CNPJ_2': result.get('CNPJ_2', ''),
                        'VALOR_TOTAL': result.get('VALOR_TOTAL', ''),
                        'Chassi': result.get('Chassi', ''),
                        'CLAIM_NUMBER': result.get('CLAIM_NUMBER', ''),
                    })
            else:
                # Add empty values for failed results
                row_data.update({
                    'file_name_llm': '',
                    'DOC_TYPE': '',
                    'CNPJ_1': '',
                    'CNPJ_2': '',
                    'VALOR_TOTAL': '',
                    'Chassi': '',
                    'CLAIM_NUMBER': '',
                })
        
        # Handle retries by removing previous entries for the same file
        if retry_round is not None and retry_round > 0:
            # Remove previous entries for this file from global DataFrame
            self.global_df = self.global_df[self.global_df['file_path'] != file_path]
            logging.info(f"🔄 Removed previous entry for retry file: {file_path} (retry round {retry_round})")
        elif '_retry_' in group_id:
            # For retry groups, always remove previous entries to ensure we get the latest result
            self.global_df = self.global_df[self.global_df['file_path'] != file_path]
            logging.info(f"🔄 Removed previous entry for retry file: {file_path} (retry group)")
        
        # Convert to DataFrame and append to global DataFrame
        row_df = pd.DataFrame([row_data])
        self.global_df = pd.concat([self.global_df, row_df], ignore_index=True)
    
    def _save_to_csv(self) -> None:
        """Save the global DataFrame to CSV file."""
        try:
            self.global_df.to_csv(self.output_file, index=False, encoding='utf-8')
            logging.info(f"💾 CSV file updated: {self.output_file}")
        except Exception as e:
            logging.error(f"❌ Error saving CSV file: {e}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from the global DataFrame."""
        if self.global_df.empty:
            return {
                'total_files': 0,
                'successful_files': 0,
                'failed_files': 0,
                'success_rate': 0.0,
                'unique_groups': 0,
                'files_with_retries': 0,
                'retry_files': 0
            }
        
        total_files = len(self.global_df)
        successful_files = len(self.global_df[self.global_df['success'] == True])
        failed_files = len(self.global_df[self.global_df['success'] == False])
        success_rate = (successful_files / total_files * 100) if total_files > 0 else 0
        unique_groups = self.global_df['group_id'].nunique()
        files_with_retries = len(self.global_df[self.global_df['retry_round'].notna()])
        
        # Count retry files (files with retry_round > 0)
        retry_files = len(self.global_df[(self.global_df['retry_round'].notna()) & (self.global_df['retry_round'] > 0)])
        
        return {
            'total_files': total_files,
            'successful_files': successful_files,
            'failed_files': failed_files,
            'success_rate': success_rate,
            'unique_groups': unique_groups,
            'files_with_retries': files_with_retries,
            'retry_files': retry_files
        }
    
    def print_summary(self) -> None:
        """Print a summary of the CSV data."""
        stats = self.get_summary_stats()
        
        print("\n" + "=" * 60)
        print("📊 CSV DUMP SUMMARY")
        print("=" * 60)
        print(f"📁 Output File: {self.output_file}")
        print(f"📄 Total Files: {stats['total_files']}")
        print(f"✅ Successful: {stats['successful_files']}")
        print(f"❌ Failed: {stats['failed_files']}")
        print(f"🔄 Retries: {stats['retry_files']}")
        print(f"📊 Success Rate: {stats['success_rate']:.1f}%")
        print(f"🔗 Unique Groups: {stats['unique_groups']}")
        print(f"🔄 Files with Retries: {stats['files_with_retries']}")
        print("=" * 60) 