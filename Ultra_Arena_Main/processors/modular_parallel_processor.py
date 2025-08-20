"""
Modular parallel processor supporting multiple processing strategies and modes.
"""

import copy
import json
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from llm_strategies.strategy_factory import ProcessingStrategyFactory
from processors.benchmark_tracker import BenchmarkTracker
from common.base_monitor import BasePerformanceMonitor
from common.csv_dumper import CSVResultDumper
from config.config_base import STRATEGY_DIRECT_FILE, STRATEGY_TEXT_FIRST, STRATEGY_HYBRID, MODE_PARALLEL, MODE_BATCH


class ModularParallelProcessor:
    """Modular parallel processor with support for multiple processing strategies."""
    
    def __init__(self, config: Dict[str, Any], strategy_type: str = "direct_file", 
                 mode: str = "parallel", max_workers: int = 5,
                 checkpoint_file: str = "modular_checkpoint.pkl", 
                 output_file: str = "modular_results.json",
                 real_time_save: bool = True, run_settings: Dict[str, str] = None,
                 csv_output_file: str = None, benchmark_comparator = None,
                 streaming: bool = False):
        """
        Initialize the modular parallel processor.
        
        Args:
            config: Configuration dictionary
            strategy_type: Processing strategy ("direct_file", "text_first", "hybrid")
            mode: Processing mode ("parallel" or "batch")
            max_workers: Maximum number of concurrent workers
            checkpoint_file: Checkpoint file path
            output_file: Output file path
            real_time_save: Whether to save results in real-time
            run_settings: Dictionary containing run settings (strategy, mode, llm_provider, llm_model)
            csv_output_file: CSV output file path (optional)
        """
        self.config = config
        self.strategy_type = strategy_type
        self.mode = mode
        self.max_workers = max_workers
        self.checkpoint_file = checkpoint_file
        self.output_file = output_file
        self.real_time_save = real_time_save
        self.run_settings = run_settings or {}
        self.benchmark_comparator = benchmark_comparator
        self.csv_output_file = csv_output_file  # Store the CSV output file path
        self.streaming = streaming  # Store the streaming flag
        
        # Initialize BenchmarkTracker instead of duplicating logic
        self.benchmark_tracker = BenchmarkTracker(benchmark_comparator, csv_output_file)
        
        # Initialize processing strategy
        self.strategy = ProcessingStrategyFactory.create_strategy(strategy_type, config, streaming=streaming)
        
        # Initialize components
        self.monitor = BasePerformanceMonitor("modular_parallel_processor")
        
        # Initialize CSV dumper with provided path or generate default
        if csv_output_file:
            # Use provided CSV output file path
            csv_output_dir = os.path.dirname(csv_output_file)
            csv_filename = os.path.basename(csv_output_file)
            self.csv_dumper = CSVResultDumper(output_dir=csv_output_dir, custom_filename=csv_filename)
        else:
            # Generate CSV filename based on run settings (fallback)
            csv_filename = f"{self.run_settings.get('strategy', strategy_type)}_{self.run_settings.get('mode', mode)}_{self.run_settings.get('llm_provider', 'unknown')}_{self.run_settings.get('llm_model', 'unknown')}_{datetime.now().strftime('%m-%d-%H-%M-%S')}.csv"
            csv_filename = csv_filename.replace(":", "_").replace("/", "_").replace("-", "_")
            csv_output_dir = "output/results/csv"
            self.csv_dumper = CSVResultDumper(output_dir=csv_output_dir, custom_filename=csv_filename)
        
        # Initialize structured output to match backup project exactly
        self.structured_output = {
            'run_settings': {
                'strategy': self.run_settings.get('strategy', strategy_type),
                'mode': self.run_settings.get('mode', mode),
                'llm_provider': self.run_settings.get('llm_provider', 'unknown'),
                'llm_model': self.run_settings.get('llm_model', 'unknown')
            },
            'file_stats': {},
            'group_stats': {},
            'retry_stats': {
                'num_files_may_need_retry': 0,
                'num_files_had_retry': 0,
                'percentage_files_had_retry': 0.0,
                'num_file_failed_after_max_retries': 0,
                'actual_tokens_for_retries': 0,
                'retry_prompt_tokens': 0,
                'retry_candidate_tokens': 0,
                'retry_other_tokens': 0,
                'retry_total_tokens': 0
            },
            'overall_stats': {
                'total_files': 0,
                'total_estimated_tokens': 0,
                'total_wall_time_in_sec': 0.0,
                'total_group_wall_time_in_sec': 0.0,
                'total_prompt_tokens': 0,
                'total_candidate_tokens': 0,
                'total_other_tokens': 0,
                'total_actual_tokens': 0,
                'average_prompt_tokens_per_file': 0.0,
                'average_candidate_tokens_per_file': 0.0,
                'average_other_tokens_per_file': 0.0,
                'average_total_tokens_per_file': 0.0
            },
            'benchmark_errors': {
                'total_unmatched_fields': 0,
                'total_unmatched_files': 0
            }
        }
        
        # Initialize tracking variables
        self.files_needed_retry = set()
        self.files_failed_after_max_retries = set()
        self.total_retry_tokens = 0
        self.processed_files = set()
        
        # Load checkpoint if exists
        self.load_checkpoint()
        
        logging.info(f"🔧 Initializing ModularParallelProcessor...")
        logging.info(f"   🎯 Strategy: {strategy_type}")
        logging.info(f"   🔄 Mode: {mode}")
        logging.info(f"   🤖 Max workers: {max_workers}")
        logging.info(f"   📁 Output file: {output_file}")
        logging.info(f"   💾 Checkpoint file: {checkpoint_file}")
        logging.info(f"🎉 ModularParallelProcessor initialization complete!")
    
    def load_checkpoint(self):
        """Load processing checkpoint if exists."""
        logging.info(f"💾 Loading checkpoint...")
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    self.processed_files = checkpoint_data.get('processed_files', set())
                    logging.info(f"✅ Loaded checkpoint with {len(self.processed_files)} processed files")
            except Exception as e:
                logging.warning(f"⚠️ Failed to load checkpoint: {e}")
    
    def save_checkpoint(self):
        """Save processing checkpoint."""
        try:
            checkpoint_data = {
                'processed_files': self.processed_files,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logging.info(f"💾 Checkpoint saved with {len(self.processed_files)} processed files")
        except Exception as e:
            logging.error(f"❌ Failed to save checkpoint: {e}")
    
    def process_files(self, *, pdf_files: List[str], system_prompt: Optional[str] = None, user_prompt: str) -> Dict[str, Any]:
        """
        Process PDF files using the configured strategy and mode.
        
        Args:
            pdf_files: List of PDF file paths
            system_prompt: Optional system prompt for LLM configuration
            user_prompt: User prompt for processing
            
        Returns:
            Dictionary containing processing results and statistics
        """
        start_time = time.time()
        logging.info(f"🚀 Starting file processing with {len(pdf_files)} files")
        logging.info(f"📋 Strategy: {self.strategy_type}, Mode: {self.mode}, Max Workers: {self.max_workers}")
        logging.info(f"📝 System prompt provided: {system_prompt is not None}")
        logging.info(f"📝 User prompt provided: {user_prompt is not None}")
        
        # Filter out already processed files
        unprocessed_files = [f for f in pdf_files if f not in self.processed_files]
        logging.info(f"📊 {len(unprocessed_files)} files need processing")
        
        if not unprocessed_files:
            logging.info("✅ All files already processed")
            return self.structured_output
        
        # Group files based on mode
        logging.info(f"📦 Grouping files for {self.mode} mode...")
        if self.mode == MODE_PARALLEL:
            file_groups = self._group_files_parallel(unprocessed_files)
        else:  # batch mode
            file_groups = self._group_files_batch(unprocessed_files)
        
        logging.info(f"📦 Created {len(file_groups)} file groups")
        for i, group in enumerate(file_groups):
            logging.info(f"   Group {i}: {len(group)} files")
        
        # Process groups
        logging.info("🔄 Starting group processing...")
        file_dict_for_retries = {}
        lot_timestamp_hash = str(int(time.time()))
        
        if self.mode == MODE_PARALLEL:
            logging.info("🔄 Processing groups in parallel mode...")
            results = self._process_groups_parallel(file_groups=file_groups, user_prompt=user_prompt, system_prompt=system_prompt, lot_timestamp_hash=lot_timestamp_hash, file_dict_for_retries=file_dict_for_retries)
        else:
            logging.info("🔄 Processing groups in batch mode...")
            results = self._process_groups_batch(file_groups=file_groups, user_prompt=user_prompt, system_prompt=system_prompt, lot_timestamp_hash=lot_timestamp_hash, file_dict_for_retries=file_dict_for_retries)
        
        logging.info(f"✅ Group processing completed, got {len(results)} results")
        
        # Process retries if needed
        if file_dict_for_retries:
            logging.info(f"🔄 Processing {len(file_dict_for_retries)} files that need retry...")
            self._process_retries(file_dict_for_retries=file_dict_for_retries, user_prompt=user_prompt, system_prompt=system_prompt, lot_timestamp_hash=lot_timestamp_hash)
        else:
            logging.info("✅ No files need retry")
        
        # Calculate final statistics
        logging.info("📊 Calculating final statistics...")
        self._calculate_final_statistics(start_time)
        
        # Check benchmark errors for all processed files after all processing is complete
        # This includes both successful and failed files
        if self.benchmark_comparator:
            logging.info("🔍 Checking benchmark errors for all processed files...")
            for file_path in self.structured_output.get('file_stats', {}):
                if file_path in self.structured_output['file_stats']:
                    result = self.structured_output['file_stats'][file_path]
                    self.check_file_benchmark_errors(file_path, result)
            
            # Update benchmark error statistics after comparison is complete
            error_stats = self.benchmark_tracker.get_error_stats()
            benchmark_errors = {
                'total_unmatched_fields': error_stats.get('total_unmatched_fields', 0),
                'total_unmatched_files': error_stats.get('total_unmatched_files', 0)
            }
            self.set_benchmark_errors(benchmark_errors)
        
        # Save results
        logging.info("💾 Saving results...")
        self.save_results()
        
        logging.info(f"🎉 Processing complete! Total time: {time.time() - start_time:.2f}s")
        return self.structured_output
    
    def _group_files_parallel(self, pdf_files: List[str]) -> List[List[str]]:
        """Group files for parallel processing."""
        max_files_per_group = self.config.get("max_num_files_per_request", 8)
        groups = []
        
        for i in range(0, len(pdf_files), max_files_per_group):
            group = pdf_files[i:i + max_files_per_group]
            groups.append(group)
        
        return groups
    
    def _group_files_batch(self, pdf_files: List[str]) -> List[List[str]]:
        """Group files for batch processing."""
        max_files_per_group = self.config.get("max_num_file_parts_per_batch", 100)
        groups = []
        
        for i in range(0, len(pdf_files), max_files_per_group):
            group = pdf_files[i:i + max_files_per_group]
            groups.append(group)
        
        return groups
    
    def _process_groups_parallel(self, *, file_groups: List[List[str]], user_prompt: str, system_prompt: Optional[str],
                                lot_timestamp_hash: str, file_dict_for_retries: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
        """Process groups in parallel to match backup project exactly."""
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all groups for processing
            future_to_group = {
                executor.submit(self._process_single_group, file_group=group, group_index=i, user_prompt=user_prompt, system_prompt=system_prompt, lot_timestamp_hash=lot_timestamp_hash): i
                for i, group in enumerate(file_groups)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_group):
                group_index = future_to_group[future]
                try:
                    group_results, group_stats, group_id = future.result()
                    all_results.extend(group_results)
                    
                    # Add group stats to group_stats (top level) to match backup project exactly
                    self.structured_output['group_stats'][group_id] = group_stats
                    
                    # Accumulate statistics incrementally for this group
                    self._calculate_accum_final_statistics(group_results, group_stats)
                    
                    # Check for files that need retry
                    self._check_group_for_retries(group_results, file_dict_for_retries)
                    
                    # Dump group results to CSV
                    self.csv_dumper.dump_group_results(group_results, group_id)
                    
                    # Save results incrementally
                    if self.real_time_save:
                        self._save_results_incrementally()
                    
                except Exception as e:
                    logging.error(f"❌ Error processing group {group_index}: {e}")
        
        return all_results
    
    def _process_groups_batch(self, *, file_groups: List[List[str]], user_prompt: str, system_prompt: Optional[str],
                             lot_timestamp_hash: str, file_dict_for_retries: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
        """Process groups sequentially (batch mode) to match backup project exactly."""
        all_results = []
        
        for i, group in enumerate(file_groups):
            try:
                group_results, group_stats, group_id = self._process_single_group(file_group=group, group_index=i, user_prompt=user_prompt, system_prompt=system_prompt, lot_timestamp_hash=lot_timestamp_hash)
                all_results.extend(group_results)
                
                # Add group stats to group_stats (top level) to match backup project exactly
                self.structured_output['group_stats'][group_id] = group_stats
                
                # Accumulate statistics incrementally for this group
                self._calculate_accum_final_statistics(group_results, group_stats)
                
                # Check for files that need retry
                self._check_group_for_retries(group_results, file_dict_for_retries)
                
                # Dump group results to CSV
                self.csv_dumper.dump_group_results(group_results, group_id)
                
                # Save results incrementally
                if self.real_time_save:
                    self._save_results_incrementally()
                
            except Exception as e:
                logging.error(f"❌ Error processing group {i}: {e}")
        
        return all_results
    
    def _process_single_group(self, *, file_group: List[str], group_index: int, 
                             user_prompt: str, system_prompt: Optional[str], lot_timestamp_hash: str) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Process a single group of files to match backup project structure exactly."""
        group_id = f"{lot_timestamp_hash}_group_{group_index}"
        logging.info(f"🔄 Processing group {group_index} ({group_id}) with {len(file_group)} files")
        logging.info(f"   Files: {file_group}")
        
        try:
            # Use the configured strategy to process the group
            logging.info(f"📋 Using strategy: {self.strategy_type}")
            results, stats, _ = self.strategy.process_file_group(file_group=file_group, group_index=group_index, user_prompt=user_prompt, system_prompt=system_prompt, group_id=group_id)
            logging.info(f"✅ Group {group_index} processed successfully, got {len(results)} results")
            
            # Process results to match backup project structure exactly
            processed_results = []
            for file_path, result in results:
                if "error" not in result:
                    # Extract LLM response attributes (including file_name_llm)
                    llm_response_attrs = {}
                    
                    for key, value in result.items():
                        # Include all fields including token fields in model output
                        llm_response_attrs[key] = value
                    
                    # Calculate other tokens for group/overall stats
                    total_tokens = result.get('total_token_count', 0)
                    prompt_tokens = result.get('prompt_token_count', 0)
                    candidates_tokens = result.get('candidates_token_count', 0)
                    other_tokens = total_tokens - prompt_tokens - candidates_tokens
                    
                    # Store actual tokens in result for group/overall aggregation
                    result['prompt_tokens'] = prompt_tokens
                    result['candidates_tokens'] = candidates_tokens
                    result['actual_tokens'] = total_tokens
                    result['other_tokens'] = other_tokens
                    
                    # Get actual file size
                    try:
                        file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
                    except (OSError, FileNotFoundError):
                        file_size_mb = 0
                    
                    # Create result structure with organized groups (file_process_result at top)
                    processed_result = {
                        'file_process_result': {
                            'success': True,
                            'retry_round': None,
                            'failure_reason': None,
                            'proc_timestamp': datetime.now().isoformat(),
                            'group_ids_incl_retries': []
                        },
                        'file_model_output': llm_response_attrs,
                        'file_info': {
                            'file_name': os.path.basename(file_path),
                            'file_size_mb': file_size_mb
                        }
                    }
                    processed_results.append((file_path, processed_result))
                    self.processed_files.add(file_path)
                else:
                    # Get actual file size
                    try:
                        file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
                    except (OSError, FileNotFoundError):
                        file_size_mb = 0
                    
                    # Create error result structure
                    processed_result = {
                        'file_process_result': {
                            'success': False,
                            'retry_round': None,
                            'failure_reason': result.get('error', 'Unknown error'),
                            'proc_timestamp': datetime.now().isoformat(),
                            'group_ids_incl_retries': []
                        },
                        'file_model_output': {},
                        'file_info': {
                            'file_name': os.path.basename(file_path),
                            'file_size_mb': file_size_mb
                        }
                    }
                    processed_results.append((file_path, processed_result))
            
            # Create group stats after token calculation
            group_stats = {
                "group_index": group_index,
                "submission_time": datetime.now().isoformat(),
                "file_count": len(file_group),
                "file_name_list": [os.path.basename(f) for f in file_group],
                "estimated_tokens": stats.get('estimated_tokens', 0),
                "actual_tokens": sum(result.get('actual_tokens', 0) for _, result in results if "error" not in result),
                "group_proc_time_in_sec": stats.get('processing_time', 0),
                "group_prompt_tokens": sum(result.get('prompt_tokens', 0) for _, result in results if "error" not in result),
                "group_candidate_tokens": sum(result.get('candidates_tokens', 0) for _, result in results if "error" not in result),
                "group_other_tokens": sum(result.get('other_tokens', 0) for _, result in results if "error" not in result),
                "group_total_tokens": sum(result.get('actual_tokens', 0) for _, result in results if "error" not in result)
            }
            
            # Save all processed results to file_stats
            for file_path, processed_result in processed_results:
                self.structured_output['file_stats'][file_path] = processed_result
            
            return processed_results, group_stats, group_id
            
        except Exception as e:
            logging.error(f"❌ Error processing group {group_index}: {e}", exc_info=True)
            # Return error results for all files in the group
            error_results = []
            for file_path in file_group:
                # Get actual file size
                try:
                    file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
                except (OSError, FileNotFoundError):
                    file_size_mb = 0
                
                error_result = {
                    'file_process_result': {
                        'success': False,
                        'retry_round': None,
                        'failure_reason': str(e),
                        'proc_timestamp': datetime.now().isoformat(),
                        'group_ids_incl_retries': []
                    },
                    'file_model_output': {},
                    'file_token_stats': {
                        'prompt_tokens': 0,
                        'candidates_tokens': 0,
                        'actual_tokens': 0,
                        'other_tokens': 0,
                        'estimated_tokens': 0
                    },
                    'file_info': {
                        'file_name': os.path.basename(file_path),
                        'file_size_mb': file_size_mb
                    }
                }
                error_results.append((file_path, error_result))
            
            error_stats = {
                "group_index": group_index,
                "submission_time": datetime.now().isoformat(),
                "file_count": len(file_group),
                "file_name_list": [os.path.basename(f) for f in file_group],
                "estimated_tokens": 0,
                "actual_tokens": 0,
                "group_proc_time_in_sec": 0,
                "error": str(e)
            }
            
            return error_results, error_stats, group_id
    
    def _check_group_for_retries(self, group_results: List[Tuple[str, Dict]], 
                                file_dict_for_retries: Dict[str, Dict]):
        """Check group results for files that need retry."""
        for file_path, result in group_results:
            if "error" in result:
                continue
            
            # Check if all mandatory keys are present
            model_output = result.get('file_model_output', result)  # Handle both structures
            has_all_keys, missing_keys = self.strategy.check_mandatory_keys(model_output, file_path, self.benchmark_comparator)
            
            if not has_all_keys:
                self.files_needed_retry.add(file_path)
                file_dict_for_retries[file_path] = {
                    'result': result,
                    'missing_keys': missing_keys,
                    'retry_count': 0,
                    'num_retries_left': self.config.get("num_retry_for_mandatory_keys", 10)
                }
    
    def _process_retries(self, *, file_dict_for_retries: Dict[str, Dict], user_prompt: str, system_prompt: Optional[str], lot_timestamp_hash: str):
        """Process files that need retry."""
        if not file_dict_for_retries:
            return
        
        logging.info(f"🔄 Processing {len(file_dict_for_retries)} files that need retry")
        
        max_retries = self.config.get("num_retry_for_mandatory_keys", 10)
        
        for retry_round in range(max_retries):
            if not file_dict_for_retries:
                break
            
            logging.info(f"🔄 Retry round {retry_round + 1}/{max_retries}")
            
            # Group files for retry
            retry_files = list(file_dict_for_retries.keys())
            if self.mode == MODE_PARALLEL:
                retry_groups = self._group_files_parallel(retry_files)
            else:
                retry_groups = self._group_files_batch(retry_files)
            
            # Process retry groups
            new_retry_dict = {}
            
            if self.mode == MODE_PARALLEL:
                self._process_retry_groups_parallel(retry_groups=retry_groups, user_prompt=user_prompt, system_prompt=system_prompt, lot_timestamp_hash=lot_timestamp_hash, 
                                                  file_dict_for_retries=file_dict_for_retries, new_retry_dict=new_retry_dict, retry_round=retry_round)
            else:
                self._process_retry_groups_batch(retry_groups=retry_groups, user_prompt=user_prompt, system_prompt=system_prompt, lot_timestamp_hash=lot_timestamp_hash, 
                                               file_dict_for_retries=file_dict_for_retries, new_retry_dict=new_retry_dict, retry_round=retry_round)
            
            # Update retry dictionary
            file_dict_for_retries = new_retry_dict
            
            # Update retry statistics (backup project doesn't track individual rounds)
        
        # Add remaining files to failed list (files that still need retry after max retries)
        for file_path in file_dict_for_retries:
            self.files_failed_after_max_retries.add(file_path)
            logging.info(f"❌ File {os.path.basename(file_path)} failed after max retries")
    
    def _process_retry_groups_parallel(self, *, retry_groups: List[List[str]], user_prompt: str, system_prompt: Optional[str],
                                      lot_timestamp_hash: str, file_dict_for_retries: Dict[str, Dict],
                                      new_retry_dict: Dict[str, Dict], retry_round: int):
        """Process retry groups in parallel to match backup project exactly."""
        
        def process_retry_group(file_group, group_index):
            # Fix retry group key format: retry_sequence_number starts from 1, not 0
            retry_sequence_number = retry_round + 1
            group_id = f"{lot_timestamp_hash}_retry_{retry_sequence_number}_group_{group_index}"
            results, stats, _ = self.strategy.process_file_group(file_group=file_group, group_index=group_index, user_prompt=user_prompt, system_prompt=system_prompt, group_id=group_id)
            
            # Create group stats to match backup project exactly
            group_stats = {
                "group_index": group_index,
                "submission_time": datetime.now().isoformat(),
                "file_count": len(file_group),
                "file_name_list": [os.path.basename(f) for f in file_group],
                "estimated_tokens": stats.get('estimated_tokens', 0),
                "actual_tokens": sum(result.get('total_token_count', 0) for _, result in results if "error" not in result),
                "group_proc_time_in_sec": stats.get('processing_time', 0)
            }
            
            return results, group_stats, group_id
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_group = {executor.submit(process_retry_group, group, idx): (group, idx) for idx, group in enumerate(retry_groups)}
            
            for future in as_completed(future_to_group):
                try:
                    group_results, group_stats, file_group_id = future.result()
                    
                    # Process retry results to match backup project exactly
                    for file_path, result in group_results:
                        # Check if result is None or invalid
                        if result is None:
                            logging.error(f"❌ Retry processing failed: Result is None for file {file_path}")
                            # Create a failed result entry
                            failed_result = {
                                'success': False,
                                'model_output': {},
                                'file_info': {'file_name': os.path.basename(file_path)},
                                'error': 'Retry processing failed: Result is None',
                                'retry_round': retry_round
                            }
                            self._save_single_file_result(file_path, failed_result)
                            continue
                        
                        # Get original retry entry
                        original_entry = file_dict_for_retries.get(file_path)
                        if not original_entry:
                            continue
                        
                        # Check if retry was successful
                        model_output = result.get('file_model_output', result)  # Handle both structures
                        is_valid, missing_keys = self.strategy.check_mandatory_keys(model_output, file_path, self.benchmark_comparator)
                        
                        if is_valid:
                            # Retry successful, add to file_stats
                            # Convert to enhanced structure if needed
                            if 'success' not in result:
                                # Extract LLM response attributes (including file_name_llm) and token stats
                                llm_response_attrs = {}
                                token_attrs = {}
                                
                                for key, value in result.items():
                                    if key in ['prompt_token_count', 'candidates_token_count', 'total_token_count', 'estimated_tokens']:
                                        token_attrs[key] = value
                                    else:
                                        llm_response_attrs[key] = value  # Include file_name_llm in model output
                                
                                # Calculate other tokens and rename fields
                                total_tokens = token_attrs.get('total_token_count', 0)
                                prompt_tokens = token_attrs.get('prompt_token_count', 0)
                                candidates_tokens = token_attrs.get('candidates_token_count', 0)
                                other_tokens = total_tokens - prompt_tokens - candidates_tokens
                                
                                token_attrs['prompt_tokens'] = token_attrs.pop('prompt_token_count', 0)
                                token_attrs['candidates_tokens'] = token_attrs.pop('candidates_token_count', 0)
                                token_attrs["actual_tokens"] = token_attrs.pop("total_token_count", 0)
                                token_attrs['other_tokens'] = other_tokens
                                
                                # Get actual file size
                                try:
                                    file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
                                except (OSError, FileNotFoundError):
                                    file_size_mb = 0
                                
                                final_result = {
                                    'file_process_result': {
                                        'success': True,
                                        'retry_round': retry_round,
                                        'failure_reason': None,
                                        'proc_timestamp': datetime.now().isoformat(),
                                        'group_ids_incl_retries': []
                                    },
                                    'file_model_output': llm_response_attrs,
                                    'file_token_stats': token_attrs,
                                    'file_info': {
                                        'file_name': os.path.basename(file_path),
                                        'file_size_mb': file_size_mb
                                    }
                                }
                            else:
                                # Result already has the new structure, just add retry_round
                                final_result = result.copy()
                                final_result['retry_round'] = retry_round
                                # Initialize file_info if not present
                                if 'file_info' not in final_result:
                                    final_result['file_info'] = {
                                        'file_name': os.path.basename(file_path),
                                        'file_size_mb': 0,
                                        'modification_time': '',
                                        'file_hash': '',
                                        'estimated_tokens': result.get('estimated_tokens', 0)
                                    }
                            
                            # Preserve existing group IDs and append the new one
                            if 'group_ids_incl_retries' in original_entry.get('file_info', {}):
                                final_result['file_process_result']['group_ids_incl_retries'] = original_entry['file_info']['group_ids_incl_retries'].copy()
                                final_result['file_process_result']['group_ids_incl_retries'].append(file_group_id)
                            else:
                                # Initialize with the initial group ID if not present
                                initial_group_id = f"{lot_timestamp_hash}_group_0"  # Assuming single group for now
                                final_result['file_process_result']['group_ids_incl_retries'] = [initial_group_id, file_group_id]
                            self._save_single_file_result(file_path, final_result)
                            self.monitor.log_progress(f"✅ {os.path.basename(file_path)} retry successful (round {retry_round})")
                        else:
                            # Check if we have retries left
                            num_retries_left = original_entry.get('num_retries_left', 0) - 1
                            if num_retries_left > 0:
                                # Still have retries, add to new retry dict
                                retry_entry = result.copy()
                                retry_entry['num_retries_left'] = num_retries_left
                                retry_entry['missing_keys'] = missing_keys
                                # Initialize file_info if not present
                                if 'file_info' not in retry_entry:
                                    retry_entry['file_info'] = {
                                        'file_name': os.path.basename(file_path),
                                        'file_size_mb': 0,
                                        'modification_time': '',
                                        'file_hash': '',
                                        'estimated_tokens': result.get('estimated_tokens', 0)
                                    }
                                # Preserve existing group IDs and append the new one
                                if 'group_ids_incl_retries' in original_entry.get('file_info', {}):
                                    retry_entry['group_ids_incl_retries'] = original_entry['file_info']['group_ids_incl_retries'].copy()
                                    retry_entry['group_ids_incl_retries'].append(file_group_id)
                                else:
                                    # Initialize with the initial group ID if not present
                                    initial_group_id = f"{lot_timestamp_hash}_group_0"  # Assuming single group for now
                                    retry_entry['group_ids_incl_retries'] = [initial_group_id, file_group_id]
                                new_retry_dict[file_path] = retry_entry
                                logging.warning(f"⚠️ {os.path.basename(file_path)} retry failed. Missing keys: {missing_keys}. Retries left: {num_retries_left}")
                            else:
                                # No retries left, mark as failed
                                # Extract LLM response attributes (including file_name_llm) and token stats if result doesn't have new structure
                                if 'file_model_output' not in result:
                                    llm_response_attrs = {}
                                    token_attrs = {}
                                    
                                    for key, value in result.items():
                                        if key in ['prompt_token_count', 'candidates_token_count', 'total_token_count', 'estimated_tokens']:
                                            token_attrs[key] = value
                                        else:
                                            llm_response_attrs[key] = value  # Include file_name_llm in model output
                                    
                                    # Calculate other tokens and rename fields
                                    total_tokens = token_attrs.get('total_token_count', 0)
                                    prompt_tokens = token_attrs.get('prompt_token_count', 0)
                                    candidates_tokens = token_attrs.get('candidates_token_count', 0)
                                    other_tokens = total_tokens - prompt_tokens - candidates_tokens
                                    
                                    token_attrs['prompt_tokens'] = token_attrs.pop('prompt_token_count', 0)
                                    token_attrs['candidates_tokens'] = token_attrs.pop('candidates_token_count', 0)
                                    token_attrs["actual_tokens"] = token_attrs.pop("total_token_count", 0)
                                    token_attrs['other_tokens'] = other_tokens
                                    
                                    # Get actual file size
                                    try:
                                        file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
                                    except (OSError, FileNotFoundError):
                                        file_size_mb = 0
                                    
                                    final_result = {
                                        'file_process_result': {
                                            'success': False,
                                            'retry_round': self.config.get("num_retry_for_mandatory_keys", 10),
                                            'failure_reason': f"Missing mandatory keys after {self.config.get('num_retry_for_mandatory_keys', 10)} retries: {missing_keys}",
                                            'proc_timestamp': datetime.now().isoformat(),
                                            'group_ids_incl_retries': []
                                        },
                                        'file_model_output': llm_response_attrs,
                                        'file_token_stats': token_attrs,
                                        'file_info': {
                                            'file_name': os.path.basename(file_path),
                                            'file_size_mb': file_size_mb
                                        }
                                    }
                                else:
                                    # Result already has new structure, just update success and add failure info
                                    final_result = result.copy()
                                    if 'file_process_result' not in final_result:
                                        final_result['file_process_result'] = {}
                                    final_result['file_process_result']['success'] = False
                                    final_result['file_process_result']['retry_round'] = self.config.get("num_retry_for_mandatory_keys", 10)
                                    final_result['file_process_result']['failure_reason'] = f"Missing mandatory keys after {self.config.get('num_retry_for_mandatory_keys', 10)} retries: {missing_keys}"
                                    final_result['file_process_result']['timestamp'] = datetime.now().isoformat()
                                    # Remove num_retries_left field
                                    if 'num_retries_left' in final_result:
                                        del final_result['num_retries_left']
                                    # Initialize file_info if not present
                                    if 'file_info' not in final_result:
                                        final_result['file_info'] = {
                                            'file_name': os.path.basename(file_path),
                                            'file_size_mb': 0
                                        }
                                # Preserve existing group IDs and append the new one
                                if 'group_ids_incl_retries' in original_entry.get('file_info', {}):
                                    final_result['file_process_result']['group_ids_incl_retries'] = original_entry['file_info']['group_ids_incl_retries'].copy()
                                    final_result['file_process_result']['group_ids_incl_retries'].append(file_group_id)
                                else:
                                    # Initialize with the initial group ID if not present
                                    initial_group_id = f"{lot_timestamp_hash}_group_0"  # Assuming single group for now
                                    final_result['file_process_result']['group_ids_incl_retries'] = [initial_group_id, file_group_id]
                                self._save_single_file_result(file_path, final_result)
                                
                                # Track files that failed after max retries (excluding 'Outros' documents)
                                model_output = final_result.get('model_output', {})
                                if model_output and model_output.get('DOC_TYPE') != 'Outros':
                                    self.files_failed_after_max_retries.add(file_path)
                                
                                self.monitor.log_progress(f"❌ {os.path.basename(file_path)} failed after {self.config.get('num_retry_for_mandatory_keys', 10)} retries", "ERROR")
                    
                    # Add group stats to group_stats (top level) to match backup project exactly
                    self.structured_output['group_stats'][file_group_id] = group_stats
                    
                    # Update overall stats and track retry tokens
                    self.structured_output['overall_stats']['total_estimated_tokens'] += group_stats['estimated_tokens']
                    self.structured_output['overall_stats']['total_group_wall_time_in_sec'] += group_stats['group_proc_time_in_sec']
                    
                    # Count successful and failed files in this retry group
                    successful_files = 0
                    failed_files = 0
                    for file_path, result in group_results:
                        model_output = result.get('model_output', result)  # Handle both structures
                        is_valid, _ = self.strategy.check_mandatory_keys(model_output, file_path, self.benchmark_comparator)
                        if is_valid:
                            successful_files += 1
                        else:
                            failed_files += 1
                    
                    # Update monitor with retry group processing results (separate from main totals)
                    self.monitor.update(
                        files_retried=len(group_results),  # Track retries separately
                        api_calls=1,  # One API call per retry group
                        total_tokens=group_stats['actual_tokens']
                    )
                    
                    # Create processed group results with retry_round information
                    processed_group_results = []
                    for file_path, result in group_results:
                        # Get the processed result from structured_output if it exists
                        if file_path in self.structured_output['file_stats']:
                            processed_result = self.structured_output['file_stats'][file_path]
                            processed_group_results.append((file_path, processed_result))
                        else:
                            # If not in structured_output, use the original result
                            processed_group_results.append((file_path, result))
                    
                    # Dump processed retry group results to CSV
                    self.csv_dumper.dump_group_results(processed_group_results, file_group_id)
                    
                    # Track retry tokens
                    self.total_retry_tokens += group_stats['actual_tokens']
                    
                    # Save results incrementally after each retry group
                    self._save_results_incrementally()
                    
                except Exception as e:
                    logging.error(f"💥 Error processing retry group: {e}")
    
    def _process_retry_groups_batch(self, *, retry_groups: List[List[str]], user_prompt: str, system_prompt: Optional[str],
                                   lot_timestamp_hash: str, file_dict_for_retries: Dict[str, Dict],
                                   new_retry_dict: Dict[str, Dict], retry_round: int):
        """Process retry groups sequentially to match backup project exactly."""
        for i, group in enumerate(retry_groups):
            try:
                # Fix retry group key format: retry_sequence_number starts from 1, not 0
                retry_sequence_number = retry_round + 1
                group_id = f"{lot_timestamp_hash}_retry_{retry_sequence_number}_group_{i}"
                group_results, stats, _ = self.strategy.process_file_group(file_group=group, group_index=i, user_prompt=user_prompt, system_prompt=system_prompt, group_id=group_id)
                
                # Create group stats to match backup project exactly
                group_stats = {
                    "group_index": i,
                    "submission_time": datetime.now().isoformat(),
                    "file_count": len(group),
                    "file_name_list": [os.path.basename(f) for f in group],
                    "estimated_tokens": stats.get('estimated_tokens', 0),
                    "actual_tokens": sum(result.get('total_token_count', 0) for _, result in group_results if "error" not in result),
                    "group_proc_time_in_sec": stats.get('processing_time', 0)
                }
                
                # Process retry results to match backup project exactly
                for file_path, result in group_results:
                    # Check if result is None or invalid
                    if result is None:
                        logging.error(f"❌ Retry processing failed: Result is None for file {file_path}")
                        # Get actual file size
                        try:
                            file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
                        except (OSError, FileNotFoundError):
                            file_size_mb = 0
                        
                        # Create a failed result entry
                        failed_result = {
                            'file_process_result': {
                                'success': False,
                                'retry_round': retry_round,
                                'failure_reason': 'Retry processing failed: Result is None',
                                'proc_timestamp': datetime.now().isoformat(),
                                'group_ids_incl_retries': []
                            },
                            'file_model_output': {},
                            'file_token_stats': {
                                'prompt_tokens': 0,
                                'candidates_tokens': 0,
                                'actual_tokens': 0,
                                'other_tokens': 0,
                                'estimated_tokens': 0
                            },
                            'file_info': {
                                'file_name': os.path.basename(file_path),
                                'file_size_mb': file_size_mb
                            }
                        }
                        self._save_single_file_result(file_path, failed_result)
                        continue
                    
                    # Get original retry entry
                    original_entry = file_dict_for_retries.get(file_path)
                    if not original_entry:
                        continue
                    
                    # Check if retry was successful
                    model_output = result.get('model_output', result)  # Handle both structures
                    is_valid, missing_keys = self.strategy.check_mandatory_keys(model_output, file_path, self.benchmark_comparator)
                    
                    if is_valid:
                        # Retry successful, add to file_stats
                        # Convert to enhanced structure if needed
                        if 'success' not in result:
                            # Extract LLM response attributes (including file_name_llm) and token stats
                            llm_response_attrs = {}
                            token_attrs = {}
                            
                            for key, value in result.items():
                                if key in ['prompt_token_count', 'candidates_token_count', 'total_token_count', 'estimated_tokens']:
                                    token_attrs[key] = value
                                else:
                                    llm_response_attrs[key] = value  # Include file_name_llm in model output
                            
                            # Calculate other tokens and rename fields
                            total_tokens = token_attrs.get('total_token_count', 0)
                            prompt_tokens = token_attrs.get('prompt_token_count', 0)
                            candidates_tokens = token_attrs.get('candidates_token_count', 0)
                            other_tokens = total_tokens - prompt_tokens - candidates_tokens
                            
                            token_attrs['prompt_tokens'] = token_attrs.pop('prompt_token_count', 0)
                            token_attrs['candidates_tokens'] = token_attrs.pop('candidates_token_count', 0)
                            token_attrs["actual_tokens"] = token_attrs.pop("total_token_count", 0)
                            token_attrs['other_tokens'] = other_tokens
                            
                            # Get actual file size
                            try:
                                file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
                            except (OSError, FileNotFoundError):
                                file_size_mb = 0
                            
                            final_result = {
                                'file_process_result': {
                                    'success': True,
                                    'retry_round': retry_round,
                                    'failure_reason': None,
                                    'proc_timestamp': datetime.now().isoformat(),
                                    'group_ids_incl_retries': []
                                },
                                'file_model_output': llm_response_attrs,
                                'file_token_stats': token_attrs,
                                'file_info': {
                                    'file_name': os.path.basename(file_path),
                                    'file_size_mb': file_size_mb
                                }
                            }
                        else:
                            final_result = result.copy()
                            final_result['retry_round'] = retry_round
                            # Initialize file_info if not present
                            if 'file_info' not in final_result:
                                final_result['file_info'] = {
                                    'file_name': os.path.basename(file_path),
                                    'file_size_mb': 0,
                                    'modification_time': '',
                                    'file_hash': '',
                                    'estimated_tokens': result.get('estimated_tokens', 0)
                                }
                        
                        # Preserve existing group IDs and append the new one
                        if 'group_ids_incl_retries' in original_entry.get('file_info', {}):
                            final_result['file_process_result']['group_ids_incl_retries'] = original_entry['file_info']['group_ids_incl_retries'].copy()
                            final_result['file_process_result']['group_ids_incl_retries'].append(group_id)
                        else:
                            # Initialize with the initial group ID if not present
                            initial_group_id = f"{lot_timestamp_hash}_group_0"  # Assuming single group for now
                            final_result['file_process_result']['group_ids_incl_retries'] = [initial_group_id, group_id]
                        self._save_single_file_result(file_path, final_result)
                        self.monitor.log_progress(f"✅ {os.path.basename(file_path)} retry successful (round {retry_round})")
                    else:
                        # Check if we have retries left
                        num_retries_left = original_entry.get('num_retries_left', 0) - 1
                        if num_retries_left > 0:
                            # Still have retries, add to new retry dict
                            retry_entry = result.copy()
                            retry_entry['num_retries_left'] = num_retries_left
                            retry_entry['missing_keys'] = missing_keys
                            # Initialize file_info if not present
                            if 'file_info' not in retry_entry:
                                retry_entry['file_info'] = {
                                    'file_name': os.path.basename(file_path),
                                    'file_size_mb': 0,
                                    'modification_time': '',
                                    'file_hash': '',
                                    'estimated_tokens': result.get('estimated_tokens', 0)
                                }
                            # Preserve existing group IDs and append the new one
                            if 'group_ids_incl_retries' in original_entry.get('file_info', {}):
                                retry_entry['group_ids_incl_retries'] = original_entry['file_info']['group_ids_incl_retries'].copy()
                                retry_entry['group_ids_incl_retries'].append(group_id)
                            else:
                                retry_entry['group_ids_incl_retries'] = [group_id]
                            new_retry_dict[file_path] = retry_entry
                            logging.warning(f"⚠️ {os.path.basename(file_path)} retry failed. Missing keys: {missing_keys}. Retries left: {num_retries_left}")
                        else:
                            # No retries left, mark as failed
                            # Extract LLM response attributes and token stats if result doesn't have new structure
                            if 'file_model_output' not in result:
                                llm_response_attrs = {}
                                token_attrs = {}
                                
                                for key, value in result.items():
                                    if key in ['prompt_token_count', 'candidates_token_count', 'total_token_count', 'estimated_tokens']:
                                        token_attrs[key] = value
                                    elif key not in ['file_name_llm']:  # Exclude file_name_llm from model output
                                        llm_response_attrs[key] = value
                                
                                final_result = {
                                    'success': False,
                                    'file_model_output': llm_response_attrs,
                                    'file_token_stats': token_attrs,
                                    'file_info': {
                                        'file_name': os.path.basename(file_path),
                                        'file_size_mb': 0,
                                        'modification_time': '',
                                        'file_hash': '',
                                        'estimated_tokens': result.get('estimated_tokens', 0)
                                    },
                                    'retry_round': self.config.get("num_retry_for_mandatory_keys", 10),
                                    'failure_reason': f"Missing mandatory keys after {self.config.get('num_retry_for_mandatory_keys', 10)} retries: {missing_keys}",
                                    'timestamp': datetime.now().isoformat()
                                }
                            else:
                                # Result already has new structure, just update success and add failure info
                                final_result = result.copy()
                                final_result['success'] = False
                                final_result['retry_round'] = self.config.get("num_retry_for_mandatory_keys", 10)
                                final_result['failure_reason'] = f"Missing mandatory keys after {self.config.get('num_retry_for_mandatory_keys', 10)} retries: {missing_keys}"
                                # Remove num_retries_left field
                                if 'num_retries_left' in final_result:
                                    del final_result['num_retries_left']
                                # Initialize file_info if not present
                                if 'file_info' not in final_result:
                                    final_result['file_info'] = {
                                        'file_name': os.path.basename(file_path),
                                        'file_size_mb': 0,
                                        'modification_time': '',
                                        'file_hash': '',
                                        'estimated_tokens': result.get('estimated_tokens', 0)
                                    }
                                                            # Preserve existing group IDs and append the new one
                                if 'group_ids_incl_retries' in original_entry.get('file_info', {}):
                                    final_result['file_process_result']['group_ids_incl_retries'] = original_entry['file_info']['group_ids_incl_retries'].copy()
                                    final_result['file_process_result']['group_ids_incl_retries'].append(group_id)
                                else:
                                    final_result['file_process_result']['group_ids_incl_retries'] = [group_id]
                                self._save_single_file_result(file_path, final_result)
                                
                                # Track files that failed after max retries (excluding 'Outros' documents)
                                model_output = final_result.get('model_output', {})
                                if model_output and model_output.get('DOC_TYPE') != 'Outros':
                                    self.files_failed_after_max_retries.add(file_path)
                                
                                self.monitor.log_progress(f"❌ {os.path.basename(file_path)} failed after {self.config.get('num_retry_for_mandatory_keys', 10)} retries", "ERROR")
                
                # Add group stats to overall_stats to match backup project exactly
                self.structured_output['overall_stats']['group_stats'][group_id] = group_stats
                
                # Update overall stats and track retry tokens
                self.structured_output['overall_stats']['total_estimated_tokens'] += group_stats['estimated_tokens']
                self.structured_output['overall_stats']['total_group_wall_time_in_sec'] += group_stats['group_proc_time_in_sec']
                
                # Count successful and failed files in this retry group
                successful_files = 0
                failed_files = 0
                for file_path, result in group_results:
                    model_output = result.get('model_output', result)  # Handle both structures
                    is_valid, _ = self.strategy.check_mandatory_keys(model_output, file_path, self.benchmark_comparator)
                    if is_valid:
                        successful_files += 1
                    else:
                        failed_files += 1
                
                # Update monitor with retry group processing results (separate from main totals)
                self.monitor.update(
                    files_retried=len(group_results),  # Track retries separately
                    api_calls=1,  # One API call per retry group
                    total_tokens=group_stats['actual_tokens']
                )
                
                # Create processed group results with retry_round information
                processed_group_results = []
                for file_path, result in group_results:
                    # Get the processed result from structured_output if it exists
                    if file_path in self.structured_output['file_stats']:
                        processed_result = self.structured_output['file_stats'][file_path]
                        processed_group_results.append((file_path, processed_result))
                    else:
                        # If not in structured_output, use the original result
                        processed_group_results.append((file_path, result))
                
                # Dump processed retry group results to CSV
                self.csv_dumper.dump_group_results(processed_group_results, group_id)
                
                # Track retry tokens
                self.total_retry_tokens += group_stats['actual_tokens']
                
                # Save results incrementally after each retry group
                self._save_results_incrementally()
                
            except Exception as e:
                logging.error(f"💥 Error processing retry group {i}: {e}")
    
    def _process_retry_group(self, *, file_group: List[str], group_index: int, user_prompt: str, system_prompt: Optional[str],
                            lot_timestamp_hash: str, file_dict_for_retries: Dict[str, Dict],
                            new_retry_dict: Dict[str, Dict], retry_round: int):
        """Process a single retry group with enhanced tracking and statistics."""
        group_id = f"{lot_timestamp_hash}_retry_{retry_round}_{group_index}"
        
        logging.info(f"🔄 Processing retry group {group_index} (round {retry_round}) with {len(file_group)} files")
        
        try:
            # Process the group
            results, stats, _ = self.strategy.process_file_group(file_group=file_group, group_index=group_index, user_prompt=user_prompt, system_prompt=system_prompt, group_id=group_id)
            
            # Enhanced group stats tracking
            enhanced_group_stats = {
                "group_index": group_index,
                "submission_time": datetime.now().isoformat(),
                "file_count": len(file_group),
                "file_name_list": [os.path.basename(f) for f in file_group],
                "estimated_tokens": stats.get('estimated_tokens', 0),
                "actual_tokens": stats.get('total_token_count', 0),
                "group_proc_time_in_sec": stats.get('processing_time', 0),
                "retry_round": retry_round
            }
            
            # Add to overall group stats
            self.structured_output['group_stats'][group_id] = enhanced_group_stats
            
            # Process retry results with enhanced tracking
            for file_path, result in results:
                # Check if result is None or invalid
                if result is None:
                    logging.error(f"❌ Retry processing failed: Result is None for file {file_path}")
                    # Get actual file size
                    try:
                        file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
                    except (OSError, FileNotFoundError):
                        file_size_mb = 0
                    
                    # Create a failed result entry with enhanced structure
                    failed_result = {
                        'file_process_result': {
                            'success': False,
                            'retry_round': retry_round,
                            'failure_reason': 'Retry processing failed: Result is None',
                            'proc_timestamp': datetime.now().isoformat(),
                            'group_ids_incl_retries': []
                        },
                        'file_model_output': {},
                        'file_token_stats': {
                            'prompt_tokens': 0,
                            'candidates_tokens': 0,
                            'actual_tokens': 0,
                            'other_tokens': 0,
                            'estimated_tokens': 0
                        },
                        'file_info': {
                            'file_name': os.path.basename(file_path),
                            'file_size_mb': file_size_mb
                        }
                    }
                    self.structured_output['file_stats'][file_path] = failed_result
                    continue
                
                # Get original entry from structured output to preserve group_ids_incl_retries
                original_entry = self.structured_output['file_stats'].get(file_path, {})
                
                # Check if retry was successful
                if "error" in result:
                    # File still failed, check retry count before adding to new retry dict
                    if file_path in file_dict_for_retries:
                        logging.info(f"🔍 Entered retry block for {file_path} at retry_round={retry_round}, group_id={group_id}")
                        file_dict_for_retries[file_path]['retry_count'] += 1
                        max_retries = self.config.get("num_retry_for_mandatory_keys", 10)
                        if file_dict_for_retries[file_path]['retry_count'] < max_retries:
                            # Still have retries, add to new retry dict
                            retry_entry = result.copy()
                            retry_entry['retry_count'] = file_dict_for_retries[file_path]['retry_count']
                            retry_entry['retry_round'] = retry_round
                            # Preserve existing group IDs and append the new one
                            if 'group_ids_incl_retries' in original_entry:
                                retry_entry['group_ids_incl_retries'] = original_entry['group_ids_incl_retries'].copy()
                                retry_entry['group_ids_incl_retries'].append(group_id)
                            else:
                                retry_entry['group_ids_incl_retries'] = [group_id]
                            new_retry_dict[file_path] = retry_entry
                            logging.warning(f"⚠️ {os.path.basename(file_path)} retry failed. Retries left: {max_retries - file_dict_for_retries[file_path]['retry_count']}")
                        else:
                            # Max retries reached, mark as failed
                            final_result = result.copy()
                            final_result['success'] = False
                            final_result['retry_round'] = max_retries
                            final_result['failure_reason'] = f"Processing error after {max_retries} retries"
                            final_result['timestamp'] = datetime.now().isoformat()
                            # Preserve existing group IDs and append the new one
                            if 'group_ids_incl_retries' in original_entry:
                                final_result['group_ids_incl_retries'] = original_entry['group_ids_incl_retries'].copy()
                                final_result['group_ids_incl_retries'].append(group_id)
                            else:
                                final_result['group_ids_incl_retries'] = [group_id]
                            self.structured_output['file_stats'][file_path] = final_result
                            self.files_failed_after_max_retries.add(file_path)
                            self.monitor.log_progress(f"❌ {os.path.basename(file_path)} failed after {max_retries} retries", "ERROR")
                else:
                    # Check if all mandatory keys are present
                    has_all_keys, missing_keys = self.strategy.check_mandatory_keys(result, file_path, self.benchmark_comparator)
                    
                    if has_all_keys:
                        # Retry successful, create enhanced result structure with new organization
                        # Extract LLM response attributes (including file_name_llm) and token stats
                        llm_response_attrs = {}
                        token_attrs = {}
                        
                        for key, value in result.items():
                            if key in ['prompt_token_count', 'candidates_token_count', 'total_token_count', 'estimated_tokens']:
                                token_attrs[key] = value
                            else:
                                llm_response_attrs[key] = value  # Include file_name_llm in model output
                        
                        # Calculate other tokens and rename fields
                        total_tokens = token_attrs.get('total_token_count', 0)
                        prompt_tokens = token_attrs.get('prompt_token_count', 0)
                        candidates_tokens = token_attrs.get('candidates_token_count', 0)
                        other_tokens = total_tokens - prompt_tokens - candidates_tokens
                        
                        token_attrs['prompt_tokens'] = token_attrs.pop('prompt_token_count', 0)
                        token_attrs['candidates_tokens'] = token_attrs.pop('candidates_token_count', 0)
                        token_attrs["actual_tokens"] = token_attrs.pop("total_token_count", 0)
                        token_attrs['other_tokens'] = other_tokens
                        
                        # Get actual file size
                        try:
                            file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
                        except (OSError, FileNotFoundError):
                            file_size_mb = 0
                        
                        final_result = {
                            'file_process_result': {
                                'success': True,
                                'retry_round': retry_round,
                                'failure_reason': None,
                                'proc_timestamp': datetime.now().isoformat(),
                                'group_ids_incl_retries': []
                            },
                            'file_model_output': llm_response_attrs,
                            'file_token_stats': token_attrs,
                            'file_info': {
                                'file_name': os.path.basename(file_path),
                                'file_size_mb': file_size_mb
                            }
                        }
                        # Preserve existing group IDs and append the new one
                        if 'group_ids_incl_retries' in original_entry:
                            final_result['file_process_result']['group_ids_incl_retries'] = original_entry['group_ids_incl_retries'].copy()
                            final_result['file_process_result']['group_ids_incl_retries'].append(group_id)
                        else:
                            final_result['file_process_result']['group_ids_incl_retries'] = [group_id]
                        
                        self.structured_output['file_stats'][file_path] = final_result
                        self.processed_files.add(file_path)
                        self.monitor.log_progress(f"✅ {os.path.basename(file_path)} retry successful (round {retry_round})")
                    else:
                        # Still missing keys, check retry count before adding to new retry dict
                        if file_path in file_dict_for_retries:
                            file_dict_for_retries[file_path]['retry_count'] += 1
                            file_dict_for_retries[file_path]['result'] = result
                            file_dict_for_retries[file_path]['missing_keys'] = missing_keys
                            max_retries = self.config.get("num_retry_for_mandatory_keys", 10)
                            logging.info(f"🔍 Debug: File {os.path.basename(file_path)} retry_count: {file_dict_for_retries[file_path]['retry_count']}, max_retries: {max_retries}")
                            if file_dict_for_retries[file_path]['retry_count'] < max_retries:
                                # Still have retries, add to new retry dict
                                retry_entry = result.copy()
                                retry_entry['retry_count'] = file_dict_for_retries[file_path]['retry_count']
                                retry_entry['missing_keys'] = missing_keys
                                retry_entry['retry_round'] = retry_round
                                # Preserve existing group IDs and append the new one
                                existing_group_ids = []
                                if 'file_process_result' in original_entry and 'group_ids_incl_retries' in original_entry['file_process_result']:
                                    existing_group_ids = original_entry['file_process_result']['group_ids_incl_retries'].copy()
                                elif 'group_ids_incl_retries' in original_entry:
                                    existing_group_ids = original_entry['group_ids_incl_retries'].copy()
                                
                                existing_group_ids.append(group_id)
                                retry_entry['group_ids_incl_retries'] = existing_group_ids
                                new_retry_dict[file_path] = retry_entry
                                # --- FIX: persist retry entry to structured_output for correct retry history ---
                                self.structured_output['file_stats'][file_path] = {
                                    "file_process_result": {
                                        "success": False,
                                        "retry_round": retry_round,
                                        "failure_reason": f"Missing mandatory keys after {retry_round} retries: {missing_keys}",
                                        "proc_timestamp": datetime.now().isoformat(),
                                        "group_ids_incl_retries": existing_group_ids
                                    },
                                    "file_model_output": {},
                                    "file_token_stats": {},
                                    "file_info": {
                                        "file_name": os.path.basename(file_path),
                                        "file_size_mb": 0
                                    }
                                }
                                logging.info(f"🔍 Debug: After retry, {file_path} group_ids_incl_retries: {existing_group_ids}")
                                logging.warning(f"⚠️ {os.path.basename(file_path)} retry failed. Missing keys: {missing_keys}. Retries left: {max_retries - file_dict_for_retries[file_path]['retry_count']}")
                            else:
                                # Max retries reached, mark as failed
                                # Extract LLM response attributes and token stats
                                llm_response_attrs = {}
                                token_attrs = {}
                                
                                for key, value in result.items():
                                    if key in ['prompt_token_count', 'candidates_token_count', 'total_token_count', 'estimated_tokens']:
                                        token_attrs[key] = value
                                    elif key not in ['file_name_llm']:  # Exclude file_name_llm from model output
                                        llm_response_attrs[key] = value
                                
                                final_result = {
                                    'file_process_result': {
                                        'success': False,
                                        'retry_round': max_retries,
                                        'failure_reason': f"Missing mandatory keys after {max_retries} retries: {missing_keys}",
                                        'proc_timestamp': datetime.now().isoformat(),
                                        'group_ids_incl_retries': []
                                    },
                                    'file_model_output': llm_response_attrs,
                                    'file_token_stats': token_attrs,
                                    'file_info': {
                                        'file_name': os.path.basename(file_path),
                                        'file_size_mb': 0
                                    }
                                }
                                # Preserve existing group IDs and append the new one
                                existing_group_ids = []
                                if 'file_process_result' in original_entry and 'group_ids_incl_retries' in original_entry['file_process_result']:
                                    existing_group_ids = original_entry['file_process_result']['group_ids_incl_retries'].copy()
                                    logging.info(f"🔍 Debug: Found group_ids_incl_retries in file_process_result: {existing_group_ids}")
                                elif 'group_ids_incl_retries' in original_entry:
                                    existing_group_ids = original_entry['group_ids_incl_retries'].copy()
                                    logging.info(f"🔍 Debug: Found group_ids_incl_retries at top level: {existing_group_ids}")
                                else:
                                    logging.info(f"🔍 Debug: No existing group_ids_incl_retries found in original_entry")
                                
                                existing_group_ids.append(group_id)
                                logging.info(f"🔍 Debug: Final group_ids_incl_retries: {existing_group_ids}")
                                final_result['file_process_result']['group_ids_incl_retries'] = existing_group_ids
                                
                                self.structured_output['file_stats'][file_path] = final_result
                                
                                # Track files that failed after max retries (excluding 'Outros' documents)
                                if result.get('DOC_TYPE') != 'Outros':
                                    self.files_failed_after_max_retries.add(file_path)
                                
                                self.monitor.log_progress(f"❌ {os.path.basename(file_path)} failed after {max_retries} retries", "ERROR")
            
            # Update overall stats and track retry tokens
            self.structured_output['overall_stats']['total_estimated_tokens'] += enhanced_group_stats['estimated_tokens']
            self.structured_output['overall_stats']['total_group_wall_time_in_sec'] += enhanced_group_stats['group_proc_time_in_sec']
            
            # Count successful and failed files in this retry group
            successful_files = 0
            failed_files = 0
            for file_path, result in results:
                if "error" in result:
                    failed_files += 1
                else:
                    has_all_keys, _ = self.strategy.check_mandatory_keys(result, file_path, self.benchmark_comparator)
                    if has_all_keys:
                        successful_files += 1
                    else:
                        failed_files += 1
            
            # Update monitor with retry group processing results
            self.monitor.update(
                files_retried=len(results),  # Track retries separately
                api_calls=1,  # One API call per retry group
                total_tokens=enhanced_group_stats['actual_tokens']
            )
            
            # Create processed group results with retry_round information
            processed_group_results = []
            for file_path, result in results:
                # Get the processed result from structured_output if it exists
                if file_path in self.structured_output['file_stats']:
                    processed_result = self.structured_output['file_stats'][file_path]
                    processed_group_results.append((file_path, processed_result))
                else:
                    # If not in structured_output, use the original result
                    processed_group_results.append((file_path, result))
            
            # Dump processed retry group results to CSV
            self.csv_dumper.dump_group_results(processed_group_results, group_id)
            
            # Track retry tokens
            self.total_retry_tokens += enhanced_group_stats['actual_tokens']
            
            # Save results incrementally after each retry group
            if self.real_time_save:
                self._save_results_incrementally(processed_group_results)
            
            logging.info(f"✅ Retry group {group_index} (round {retry_round}) completed: {successful_files} successful, {failed_files} failed")
            
        except Exception as e:
            logging.error(f"💥 Error processing retry group {group_index} (round {retry_round}): {e}")
            # Create failed results for all files in the group
            for file_path in file_group:
                failed_result = {
                    'success': False,
                    'model_output': {},
                    'file_info': {
                        'file_name': os.path.basename(file_path),
                        'file_size_mb': 0,
                        'modification_time': '',
                        'file_hash': '',
                        'actual_tokens': 0
                    },
                    'error': f'Retry group processing failed: {str(e)}',
                    'retry_round': retry_round,
                    'timestamp': datetime.now().isoformat()
                }
                self.structured_output['file_stats'][file_path] = failed_result
    
    def _calculate_accum_final_statistics(self, group_results: List[Tuple[str, Dict]], group_stats: Dict):
        """Accumulate statistics incrementally for a newly processed group."""
        # Update total_files count
        self.structured_output['overall_stats']['total_files'] = len(self.structured_output['file_stats'])
        
        # Accumulate token statistics from this group
        self.structured_output['overall_stats']['total_estimated_tokens'] += group_stats.get('estimated_tokens', 0)
        self.structured_output['overall_stats']['total_group_wall_time_in_sec'] += group_stats.get('group_proc_time_in_sec', 0)
        
        # Accumulate token breakdown from group_stats
        self.structured_output['overall_stats']['total_prompt_tokens'] += group_stats.get('group_prompt_tokens', 0)
        self.structured_output['overall_stats']['total_candidate_tokens'] += group_stats.get('group_candidate_tokens', 0)
        self.structured_output['overall_stats']['total_other_tokens'] += group_stats.get('group_other_tokens', 0)
        self.structured_output['overall_stats']['total_actual_tokens'] += group_stats.get('group_total_tokens', 0)
        
        # Accumulate wall time from group processing
        self.structured_output['overall_stats']['total_wall_time_in_sec'] += group_stats.get('group_proc_time_in_sec', 0)
        
        # Note: Retry statistics and cost calculations will be done at the end
        # to avoid complex incremental calculations

    def _calculate_final_statistics(self, start_time: float):
        """Calculate final processing statistics to match backup project exactly."""
        # Store current accumulated stats for verification
        current_accumulated_stats = {
            'total_files': self.structured_output['overall_stats'].get('total_files', 0),
            'total_estimated_tokens': self.structured_output['overall_stats'].get('total_estimated_tokens', 0),
            'total_group_wall_time_in_sec': self.structured_output['overall_stats'].get('total_group_wall_time_in_sec', 0),
            'total_prompt_tokens': self.structured_output['overall_stats'].get('total_prompt_tokens', 0),
            'total_candidate_tokens': self.structured_output['overall_stats'].get('total_candidate_tokens', 0),
            'total_other_tokens': self.structured_output['overall_stats'].get('total_other_tokens', 0),
            'total_actual_tokens': self.structured_output['overall_stats'].get('total_actual_tokens', 0),
            'total_wall_time_in_sec': self.structured_output['overall_stats'].get('total_wall_time_in_sec', 0)
        }
        
        # Calculate final statistics using the original method
        self.structured_output['overall_stats']['total_files'] = len(self.structured_output['file_stats'])
        # Update total_wall_time_in_sec incrementally (accumulated from group processing times)
        self.structured_output['overall_stats']['total_wall_time_in_sec'] = self.structured_output['overall_stats'].get('total_wall_time_in_sec', 0)
        
        # Calculate token statistics first (this populates group_stats token fields)
        self._calculate_token_statistics()
        
        # Calculate retry statistics (this needs the group_stats token fields)
        self._calculate_retry_statistics()
        
        # Calculate overall cost based on token usage and pricing
        self._calculate_overall_cost()
        
        # Verify accumulated stats match final calculation
        self._verify_accumulated_statistics(current_accumulated_stats)
        
        # Note: Benchmark error statistics will be updated after benchmark comparison is complete

    def _verify_accumulated_statistics(self, accumulated_stats: Dict):
        """Verify that accumulated statistics match final calculation."""
        import time
        start_time = time.time()
        
        logging.info("🔍 Starting verification of accumulated statistics...")
        
        # Check total_files
        final_total_files = len(self.structured_output['file_stats'])
        if accumulated_stats['total_files'] != final_total_files:
            logging.error(f"❌ CRITICAL ERROR: total_files mismatch! Accumulated: {accumulated_stats['total_files']}, Final: {final_total_files}")
        else:
            logging.info(f"✅ total_files verification passed: {final_total_files}")
        
        # Check total_estimated_tokens
        final_estimated_tokens = self.structured_output['overall_stats'].get('total_estimated_tokens', 0)
        if accumulated_stats['total_estimated_tokens'] != final_estimated_tokens:
            logging.error(f"❌ CRITICAL ERROR: total_estimated_tokens mismatch! Accumulated: {accumulated_stats['total_estimated_tokens']}, Final: {final_estimated_tokens}")
        else:
            logging.info(f"✅ total_estimated_tokens verification passed: {final_estimated_tokens}")
        
        # Check total_group_wall_time_in_sec
        final_group_wall_time = self.structured_output['overall_stats'].get('total_group_wall_time_in_sec', 0)
        if accumulated_stats['total_group_wall_time_in_sec'] != final_group_wall_time:
            logging.error(f"❌ CRITICAL ERROR: total_group_wall_time_in_sec mismatch! Accumulated: {accumulated_stats['total_group_wall_time_in_sec']}, Final: {final_group_wall_time}")
        else:
            logging.info(f"✅ total_group_wall_time_in_sec verification passed: {final_group_wall_time}")
        
        # Check total_actual_tokens
        final_actual_tokens = self.structured_output['overall_stats'].get('total_actual_tokens', 0)
        if accumulated_stats['total_actual_tokens'] != final_actual_tokens:
            logging.error(f"❌ CRITICAL ERROR: total_actual_tokens mismatch! Accumulated: {accumulated_stats['total_actual_tokens']}, Final: {final_actual_tokens}")
        else:
            logging.info(f"✅ total_actual_tokens verification passed: {final_actual_tokens}")
        
        # Check total_prompt_tokens
        final_prompt_tokens = self.structured_output['overall_stats'].get('total_prompt_tokens', 0)
        if accumulated_stats['total_prompt_tokens'] != final_prompt_tokens:
            logging.error(f"❌ CRITICAL ERROR: total_prompt_tokens mismatch! Accumulated: {accumulated_stats['total_prompt_tokens']}, Final: {final_prompt_tokens}")
        else:
            logging.info(f"✅ total_prompt_tokens verification passed: {final_prompt_tokens}")
        
        # Check total_candidate_tokens
        final_candidate_tokens = self.structured_output['overall_stats'].get('total_candidate_tokens', 0)
        if accumulated_stats['total_candidate_tokens'] != final_candidate_tokens:
            logging.error(f"❌ CRITICAL ERROR: total_candidate_tokens mismatch! Accumulated: {accumulated_stats['total_candidate_tokens']}, Final: {final_candidate_tokens}")
        else:
            logging.info(f"✅ total_candidate_tokens verification passed: {final_candidate_tokens}")
        
        # Check total_other_tokens
        final_other_tokens = self.structured_output['overall_stats'].get('total_other_tokens', 0)
        if accumulated_stats['total_other_tokens'] != final_other_tokens:
            logging.error(f"❌ CRITICAL ERROR: total_other_tokens mismatch! Accumulated: {accumulated_stats['total_other_tokens']}, Final: {final_other_tokens}")
        else:
            logging.info(f"✅ total_other_tokens verification passed: {final_other_tokens}")
        
        # Check total_wall_time_in_sec with 2-second tolerance
        final_wall_time = self.structured_output['overall_stats'].get('total_wall_time_in_sec', 0)
        time_diff = abs(accumulated_stats['total_wall_time_in_sec'] - final_wall_time)
        if time_diff >= 2.0:
            logging.error(f"❌ CRITICAL ERROR: total_wall_time_in_sec mismatch! Accumulated: {accumulated_stats['total_wall_time_in_sec']}, Final: {final_wall_time}, Difference: {time_diff:.2f}s")
        else:
            logging.info(f"✅ total_wall_time_in_sec verification passed: {final_wall_time} (difference: {time_diff:.2f}s)")
        
        verification_time = time.time() - start_time
        logging.info(f"⏱️ Statistics verification completed in {verification_time:.3f} seconds")
    
    def _calculate_retry_statistics(self):
        """Calculate retry statistics to match backup project exactly."""
        # Count files that needed retry
        files_needed_retry_count = len(self.files_needed_retry)
        self.structured_output['retry_stats']['num_files_may_need_retry'] = files_needed_retry_count
        self.structured_output['retry_stats']['num_files_had_retry'] = files_needed_retry_count
        
        # Count files that actually failed after max retries from file_stats
        failed_count = 0
        for file_path, result in self.structured_output['file_stats'].items():
            if isinstance(result, dict):
                file_process_result = result.get('file_process_result', {})
                retry_round = file_process_result.get('retry_round')
                if retry_round is None:
                    retry_round = 0
                if (file_process_result.get('success') == False and 
                    retry_round >= self.config.get('num_retry_for_mandatory_keys', 2)):
                    failed_count += 1
        
        self.structured_output['retry_stats']['num_file_failed_after_max_retries'] = failed_count
        
        # Calculate percentage of files that had retries
        # This should be based on the original number of files processed
        # processed_files contains all files that were processed (including those that needed retries)
        total_original_files = len(self.processed_files)
        if total_original_files > 0:
            self.structured_output['retry_stats']['percentage_files_had_retry'] = (
                files_needed_retry_count / total_original_files * 100
            )
        else:
            self.structured_output['retry_stats']['percentage_files_had_retry'] = 0.0
        
        # Calculate detailed token statistics for retries from group_stats
        retry_prompt_tokens = 0
        retry_candidate_tokens = 0
        retry_other_tokens = 0
        retry_total_tokens = 0
        
        # Sum up tokens from all retry groups
        if 'group_stats' in self.structured_output:
            for group_name, group_data in self.structured_output['group_stats'].items():
                # Check if this is a retry group (contains 'retry' in the name)
                if 'retry' in group_name:
                    retry_prompt_tokens += group_data.get('group_prompt_tokens', 0)
                    retry_candidate_tokens += group_data.get('group_candidate_tokens', 0)
                    retry_other_tokens += group_data.get('group_other_tokens', 0)
                    retry_total_tokens += group_data.get('group_total_tokens', 0)
        
        self.structured_output['retry_stats']['actual_tokens_for_retries'] = retry_total_tokens
        self.structured_output['retry_stats']['retry_prompt_tokens'] = retry_prompt_tokens
        self.structured_output['retry_stats']['retry_candidate_tokens'] = retry_candidate_tokens
        self.structured_output['retry_stats']['retry_other_tokens'] = retry_other_tokens
        self.structured_output['retry_stats']['retry_total_tokens'] = retry_total_tokens
    
    def _calculate_token_statistics(self):
        """Calculate token statistics for group_stats, retry_stats, and overall_stats."""
        # Calculate totals from all group_stats (including retry groups)
        total_estimated_tokens = 0
        total_group_wall_time_in_sec = 0
        total_actual_tokens = 0
        
        if 'group_stats' in self.structured_output:
            for group_name, group_data in self.structured_output['group_stats'].items():
                total_estimated_tokens += group_data.get('estimated_tokens', 0)
                total_group_wall_time_in_sec += group_data.get('group_proc_time_in_sec', 0)
                total_actual_tokens += group_data.get('actual_tokens', 0)
        
        # Calculate token breakdown from individual file statistics (for group calculations)
        total_prompt_tokens = 0
        total_candidate_tokens = 0
        total_other_tokens = 0
        total_actual_tokens = 0
        
        # Sum up tokens from all individual files
        # We need to get tokens from the original LLM response results, not from processed file_stats
        # The tokens are calculated in _process_single_group and stored in group_stats
        total_prompt_tokens = 0
        total_candidate_tokens = 0
        total_other_tokens = 0
        total_actual_tokens = 0
        
        # Aggregate tokens from group_stats
        if 'group_stats' in self.structured_output:
            for group_name, group_data in self.structured_output['group_stats'].items():
                total_prompt_tokens += group_data.get('group_prompt_tokens', 0)
                total_candidate_tokens += group_data.get('group_candidate_tokens', 0)
                total_other_tokens += group_data.get('group_other_tokens', 0)
                total_actual_tokens += group_data.get('group_total_tokens', 0)
        
        # Update group_stats with per-group token totals
        if 'group_stats' in self.structured_output:
            for group_name, group_data in self.structured_output['group_stats'].items():
                # Remove old total_tokens field
                if 'total_tokens' in group_data:
                    del group_data['total_tokens']
                
                # Use the actual_tokens from the group and distribute proportionally
                group_actual_tokens = group_data.get('actual_tokens', 0)
                
                # Calculate token breakdown for this group based on the group's actual token usage
                # We'll use the same proportions as the overall totals
                if total_actual_tokens > 0:
                    # Calculate the proportion of this group's tokens relative to total
                    group_proportion = group_actual_tokens / total_actual_tokens
                    
                    group_prompt_tokens = int(total_prompt_tokens * group_proportion)
                    group_candidate_tokens = int(total_candidate_tokens * group_proportion)
                    group_other_tokens = int(total_other_tokens * group_proportion)
                    group_total_tokens = group_actual_tokens  # Use the actual tokens for this group
                else:
                    # Fallback if no total tokens
                    group_prompt_tokens = 0
                    group_candidate_tokens = 0
                    group_other_tokens = 0
                    group_total_tokens = group_actual_tokens
                
                # Add new token fields with group_ prefix
                group_data.update({
                    'group_prompt_tokens': group_prompt_tokens,
                    'group_candidate_tokens': group_candidate_tokens,
                    'group_other_tokens': group_other_tokens,
                    'group_total_tokens': group_total_tokens
                })
        
        # Use the file-level token totals directly (already calculated above)
        # These are the actual totals from individual file processing
        overall_prompt_tokens = total_prompt_tokens
        overall_candidate_tokens = total_candidate_tokens
        overall_other_tokens = total_other_tokens
        overall_total_tokens = total_actual_tokens
        
        # Update overall_stats with proper totals from all groups
        if 'overall_stats' in self.structured_output:
            # Remove old total_actual_tokens field
            if 'total_actual_tokens' in self.structured_output['overall_stats']:
                del self.structured_output['overall_stats']['total_actual_tokens']
            
            # Calculate total processed files from overall_stats
            total_processed_files = self.structured_output['overall_stats'].get('total_files', 0)
            
            # Add actual token totals to overall_stats (sum of all groups)
            # Use OrderedDict to maintain field order
            from collections import OrderedDict
            overall_stats_update = OrderedDict([
                ('total_estimated_tokens', total_estimated_tokens),
                ('total_group_wall_time_in_sec', total_group_wall_time_in_sec),
                ('total_prompt_tokens', overall_prompt_tokens),
                ('total_candidate_tokens', overall_candidate_tokens),
                ('total_other_tokens', overall_other_tokens),
                ('total_actual_tokens', overall_total_tokens),
                ('average_prompt_tokens_per_file', int(overall_prompt_tokens / total_processed_files) if total_processed_files > 0 else 0),
                ('average_candidate_tokens_per_file', int(overall_candidate_tokens / total_processed_files) if total_processed_files > 0 else 0),
                ('average_other_tokens_per_file', int(overall_other_tokens / total_processed_files) if total_processed_files > 0 else 0),
                ('average_total_tokens_per_file', int(overall_total_tokens / total_processed_files) if total_processed_files > 0 else 0)
            ])
            self.structured_output['overall_stats'].update(overall_stats_update)
    
    def _estimate_prompt_tokens(self) -> int:
        """Estimate prompt tokens based on actual prompts."""
        # This would need to be calculated based on the actual prompts used
        # For now, using a simple estimation
        return len(self.processed_files) * 672  # Based on cost_calculator.py
    
    def _estimate_file_tokens(self) -> int:
        """Estimate file tokens based on file sizes."""
        total_file_tokens = 0
        for file_path in self.processed_files:
            try:
                file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                if file_size_mb < 0.05:
                    base_estimate = 4500
                elif file_size_mb < 0.15:
                    base_estimate = 5000
                else:
                    base_estimate = 5500
                total_file_tokens += int(base_estimate * 1.2)  # Add 20% buffer
            except:
                total_file_tokens += 6000  # Default fallback
        return total_file_tokens
    
    def _save_single_file_result(self, file_path: str, result: dict):
        """Save a single file result to structured output."""
        self.structured_output['file_stats'][file_path] = result
    
    def _save_results_incrementally(self):
        """Save results incrementally to file."""
        try:
            # Save current structured output to file
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.structured_output, f, indent=2, ensure_ascii=False)
            logging.debug(f"💾 Incremental save completed: {self.output_file}")
        except Exception as e:
            logging.error(f"❌ Failed to save results incrementally: {e}")
    
    def save_results(self):
        """Save final results to file."""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.structured_output, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 Results saved to {self.output_file}")
            
            # Generate error CSV file if there are benchmark errors
            self.generate_error_csv()
        except Exception as e:
            logging.error(f"❌ Failed to save results: {e}")
    
    def print_summary(self):
        """Print processing summary."""
        stats = self.structured_output['overall_stats']
        retry_stats = self.structured_output['retry_stats']
        
        print("\n" + "="*60)
        print("📊 PROCESSING SUMMARY")
        print("="*60)
        print(f"📁 Total files processed: {stats['total_files']}")
        print(f"⏱️  Total processing time: {stats['total_wall_time_in_sec']:.2f}s")
        print(f"🔄 Files that needed retry: {retry_stats['num_files_may_need_retry']}")
        print(f"❌ Files failed after max retries: {retry_stats['num_file_failed_after_max_retries']}")
        print(f"📈 Retry success rate: {retry_stats['percentage_files_had_retry']:.1f}%")
        print("="*60)
    
    def get_results(self) -> Dict[str, Any]:
        """Get processing results."""
        return self.structured_output
    
    def set_benchmark_errors(self, benchmark_errors: Dict[str, Any]):
        """Set benchmark comparison in the structured output.
        
        Args:
            benchmark_errors: Dictionary containing benchmark error statistics (deprecated, now uses BenchmarkTracker)
        """
        # Get benchmark error statistics from BenchmarkTracker
        error_stats = self.benchmark_tracker.get_error_stats()
        total_files = self.structured_output.get('overall_stats', {}).get('total_files', 0)
        total_unmatched_fields = error_stats.get('total_unmatched_fields', 0)
        total_unmatched_files = error_stats.get('total_unmatched_files', 0)
        
        # Calculate the new fields
        total_fields = total_files * 5  # 5 mandatory fields
        invalid_fields_percent = (total_unmatched_fields / total_fields * 100) if total_fields > 0 else 0
        invalid_files_percent = (total_unmatched_files / total_files * 100) if total_files > 0 else 0
        
        # Create the new benchmark_comparison structure
        benchmark_comparison = {
            "total_files": total_files,
            "total_fields": total_fields,
            "total_unmatched_fields": total_unmatched_fields,
            "total_unmatched_files": total_unmatched_files,
            "invalid_fields_percent": round(invalid_fields_percent, 2),
            "invalid_files_percent": round(invalid_files_percent, 2)
        }
        
        self.structured_output['benchmark_comparison'] = benchmark_comparison
    
    def track_benchmark_error(self, file_path: str, field_name: str, benchmark_value: str, extracted_value: str):
        """Track a benchmark error for a specific field."""
        self.benchmark_tracker.track_benchmark_error(file_path, field_name, benchmark_value, extracted_value)
    
    def track_file_benchmark_errors(self, file_path: str):
        """Track that a file has benchmark errors."""
        self.benchmark_tracker.track_file_benchmark_errors(file_path)
    
    def generate_error_csv(self):
        """Generate error CSV file with unmatched field details."""
        self.benchmark_tracker.generate_error_csv()
    
    def check_file_benchmark_errors(self, file_path: str, result: Dict[str, Any]):
        """Check for benchmark errors in a processed file and track them."""
        self.benchmark_tracker.check_file_benchmark_errors(file_path, result)
    
    def _calculate_overall_cost(self):
        """Calculate overall cost based on token usage and pricing data."""
        try:
            # Load pricing data from CSV
            import pandas as pd
            from pathlib import Path
            # Resolve path relative to Ultra_Arena_Main/config/pricing
            this_file = Path(__file__).resolve()
            pricing_path = this_file.parent.parent / 'config' / 'pricing' / 'llm_prices.csv'
            pricing_df = pd.read_csv(str(pricing_path))
            
            # Get provider and model from run_settings
            run_settings = self.structured_output.get('run_settings', {})
            llm_provider = run_settings.get('llm_provider', '')
            llm_model = run_settings.get('llm_model', '')
            
            # Find pricing data for this provider and model
            pricing_data = None
            for _, row in pricing_df.iterrows():
                if row['llm_provider'] == llm_provider and row['llm_model'] == llm_model:
                    pricing_data = row
                    break
            
            if pricing_data is None:
                logging.warning(f"⚠️ No pricing data found for {llm_provider}_{llm_model}")
                return
            
            # Get token counts from overall_stats
            overall_stats = self.structured_output.get('overall_stats', {})
            total_prompt_tokens = overall_stats.get('total_prompt_tokens', 0)
            total_candidate_tokens = overall_stats.get('total_candidate_tokens', 0)
            total_other_tokens = overall_stats.get('total_other_tokens', 0)
            
            # Get pricing per 1M tokens
            prompt_token_price_per_1M = pricing_data['prompt_token_price_per_1M']
            candidate_token_price_per_1M = pricing_data['candidate_token_price_per_1M']
            other_token_price_per_1M = prompt_token_price_per_1M  # Same as prompt token price
            
            # Calculate costs
            total_prompt_token_cost = (prompt_token_price_per_1M * total_prompt_tokens) / 1000000
            total_candidate_token_cost = (candidate_token_price_per_1M * total_candidate_tokens) / 1000000
            total_other_token_cost = (other_token_price_per_1M * total_other_tokens) / 1000000
            
            # Calculate total cost
            total_token_cost = total_prompt_token_cost + total_candidate_token_cost + total_other_token_cost
            
            # Create overall_cost dictionary
            overall_cost = {
                "price_obtained_date": pricing_data['price_obtained_date'],
                "prompt_token_price_per_1M": prompt_token_price_per_1M,
                "candidate_token_price_per_1M": candidate_token_price_per_1M,
                "other_token_price_per_1M": other_token_price_per_1M,
                "total_prompt_token_cost": round(total_prompt_token_cost, 6),
                "total_candidate_token_cost": round(total_candidate_token_cost, 6),
                "total_other_token_cost": round(total_other_token_cost, 6),
                "total_token_cost": round(total_token_cost, 6)
            }
            
            # Add overall_cost to structured output
            self.structured_output['overall_cost'] = overall_cost
            
            logging.info(f"💰 Calculated overall cost: ${total_token_cost:.6f} for {llm_provider}_{llm_model}")
            
        except Exception as e:
            logging.error(f"❌ Error calculating overall cost: {e}")
            # Don't fail the entire process if cost calculation fails
            pass