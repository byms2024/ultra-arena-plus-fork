"""
Direct file processing strategy.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .base_strategy import BaseProcessingStrategy
from llm_client.llm_client_factory import LLMClientFactory
from llm_metrics import TokenCounter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processors.file_mapping_utils import FileMappingFactory


class DirectFileProcessingStrategy(BaseProcessingStrategy):
    """Strategy for processing files directly by sending them to LLM."""
    
    def __init__(self, config: Dict[str, Any], streaming: bool = False, database_ops = None):
        super().__init__(config)
        self.streaming = streaming
        # Use llm_provider for consistency with configuration structure
        self.llm_provider = config.get("llm_provider", config.get("provider", "google"))
        self.provider_config = config.get("provider_configs", {}).get(self.llm_provider, {})
        
        # Initialize LLM client
        self.llm_client = LLMClientFactory.create_client(self.llm_provider, self.provider_config, streaming=self.streaming)
        
        # Initialize token counter for accurate estimation
        self.token_counter = TokenCounter(self.llm_client, provider=self.llm_provider)
        
        # Use hybrid retry limits if specified
        if "file_direct_max_retry" in config:
            self.max_retries = config["file_direct_max_retry"]
            logging.info(f"🌐 Direct file strategy using hybrid retry limit: {self.max_retries}")
    
    def process_file_group(self, *, config_manager, file_group: List[str], group_index: int, 
                          group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str, 
                          strategy_type: str = "direct_file", file_path_mapper=None) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Process files by sending them directly to LLM."""
        
        group_start_time = time.time()
        logging.info(f"🔄 Starting direct file processing for group {group_index} ({group_id}): {len(file_group)} files")
        
        results = []
        group_stats = {
            "total_files": len(file_group),
            "successful_files": 0,
            "failed_files": 0,
            "total_tokens": 0,
            "estimated_tokens": 0,
            "processing_time": 0
        }
        
        try:
            # Enhance the prompt to require file_name_llm field in each response (like backup project)
            enhanced_user_prompt = (
                f"{user_prompt}\n\n"
                "IMPORTANT: Use the FILE_PATH information provided to identify each document. "
                "Return a JSON array, one object per file. "
                "Each object MUST include a 'file_name_llm' field that contains the original filename from the FILE_PATH. "
                "For example, if FILE_PATH shows 'bbbb/aaaa/xxxx.pdf', "
                "then file_name_llm should be 'xxxx.pdf'. "
                "If you do not include the correct file_name_llm, your answer will be ignored."
            )
            
            # Process all files in the group together with enhanced prompt structure
            response = self._retry_with_backoff(
                self.llm_client.call_llm, files=file_group, system_prompt=system_prompt, user_prompt=enhanced_user_prompt, strategy_type=strategy_type
            )
            
            if "error" in response:
                logging.error(f"LLM API error for group {group_index}: {response['error']}")
                # Create failed results for all files
                for file_path in file_group:
                    results.append((file_path, {"error": response["error"]}))
                    group_stats["failed_files"] += 1
            else:
                # Parse response and match with files
                if isinstance(response, list):
                    # Response is already a list of results
                    file_results = response
                else:
                    # Single result, wrap in list
                    file_results = [response]
                
                # Map outputs to files using strategy-specific file mapping
                if strategy_type == "image_first" and file_path_mapper:
                    # Use the provided ImageFirstFilePathMapper with existing mappings
                    mapped_results = file_path_mapper.map_results_to_original_files(
                        [(file_path, result) for file_path, result in zip(file_group, file_results)], 
                        file_group
                    )
                elif strategy_type == "text_first" and file_path_mapper:
                    # Use the provided TextFirstFilePathMapper with existing mappings
                    try:
                        zipped_results = [(file_path, result) for file_path, result in zip(file_group, file_results)]
                        mapped_results = file_path_mapper.map_results_to_original_files(zipped_results, file_group)
                    except Exception as e:
                        logging.error(f"Error in text_first mapping: {e}")
                        raise
                else:
                    # Use provider-specific file mapping strategy for direct file strategy
                    file_mapping_strategy = FileMappingFactory.create_strategy(self.llm_provider)
                    mapped_results = file_mapping_strategy.map_outputs_to_files(file_results, file_group, group_index)
                
                # Process mapped results and add token estimates
                for file_path, file_result in mapped_results:
                    filename = Path(file_path).name
                    
                    if file_result is not None and "error" not in file_result:
                        # Calculate individual file token estimate using Google token counter
                        try:
                            file_token_estimate = self.token_counter.count_file_content_tokens(file_path)
                            file_result["estimated_tokens"] = file_token_estimate
                        except Exception as e:
                            logging.warning(f"Failed to estimate tokens for {filename}: {e}")
                            file_result["estimated_tokens"] = 0
                        
                        results.append((file_path, file_result))
                        group_stats["successful_files"] += 1
                        
                        # Add token usage if available
                        if "total_token_count" in file_result:
                            group_stats["total_tokens"] += file_result["total_token_count"]
                    else:
                        # Handle error cases
                        if "error" not in file_result:
                            file_result = {"error": "No result returned for this file"}
                        
                        # Calculate individual file token estimate even for errors
                        try:
                            file_token_estimate = self.token_counter.count_file_content_tokens(file_path)
                            file_result["estimated_tokens"] = file_token_estimate
                        except Exception as e:
                            logging.warning(f"Failed to estimate tokens for {filename}: {e}")
                            file_result["estimated_tokens"] = 0
                        
                        results.append((file_path, file_result))
                        group_stats["failed_files"] += 1
        
        except Exception as e:
            logging.error(f"Error processing group {group_index}: {e}")
            # Create failed results for all files
            for file_path in file_group:
                results.append((file_path, {"error": str(e)}))
                group_stats["failed_files"] += 1
        
        group_stats["processing_time"] = int(time.time() - group_start_time)
        
        # Calculate estimated tokens for the group using proper token counter
        try:
            estimation = self.token_counter.estimate_total_tokens_for_group(user_prompt, file_group)
            group_stats["estimated_tokens"] = estimation["total_estimated_tokens"]
        except Exception as e:
            logging.warning(f"Failed to calculate accurate token estimation: {e}")
            group_stats["estimated_tokens"] = 0
        
        logging.info(f"✅ Completed direct file processing for group {group_index}: {group_stats['successful_files']} successful, {group_stats['failed_files']} failed")
        
        return results, group_stats, group_id 