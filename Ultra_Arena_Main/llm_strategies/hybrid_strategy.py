"""
Hybrid processing strategy.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .base_strategy import BaseProcessingStrategy
from .direct_file_strategy import DirectFileProcessingStrategy
from .enhanced_text_first_strategy import EnhancedTextFirstProcessingStrategy


class HybridProcessingStrategy(BaseProcessingStrategy):
    """Strategy that combines direct file and text-first approaches."""
    
    def __init__(self, config: Dict[str, Any], streaming: bool = False):
        super().__init__(config)
        
        # Store streaming parameter
        self.streaming = streaming
        
        # Create both strategies
        self.direct_file_processor = DirectFileProcessingStrategy(config, streaming=streaming)
        self.text_first_processor = EnhancedTextFirstProcessingStrategy(config, streaming=streaming)
        
        # Use hybrid retry limits if specified
        if "file_direct_max_retry" in config:
            self.max_retries = config["file_direct_max_retry"]
            logging.info(f"🔄 Hybrid strategy using hybrid retry limit: {self.max_retries}")
    
    def process_file_group(self, *, config_manager=None, file_group: List[str], group_index: int, 
                          group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Process files using hybrid approach: try direct file first, fallback to text-first."""
        
        group_start_time = time.time()
        logging.info(f"🔄 Starting hybrid processing for group {group_index} ({group_id}): {len(file_group)} files")
        
        # Try direct file processing first
        try:
            direct_results, direct_stats, direct_group_id = self.direct_file_processor.process_file_group(
                config_manager=config_manager,
                file_group=file_group,
                group_index=group_index,
                group_id=group_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Check if direct processing was successful
            if direct_stats["failed_files"] == 0:
                logging.info(f"✅ Hybrid: Direct file processing successful for group {group_index}")
                return direct_results, direct_stats, direct_group_id
            
            # If direct processing failed, try text-first for failed files
            logging.info(f"⚠️ Hybrid: Direct file processing failed for {direct_stats['failed_files']} files, trying text-first")
            
            # Get list of failed files
            failed_files = []
            for file_path, result in direct_results:
                if "error" in result:
                    failed_files.append(file_path)
            
            if not failed_files:
                return direct_results, direct_stats, direct_group_id
            
            # Try text-first processing for failed files
            text_results, text_stats, text_group_id = self.text_first_processor.process_file_group(
                config_manager=config_manager,
                file_group=failed_files,
                group_index=group_index,
                group_id=f"{group_id}_text_fallback",
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Merge results
            merged_results = []
            merged_stats = {
                "total_files": len(file_group),
                "successful_files": 0,
                "failed_files": 0,
                "total_tokens": 0,
                "estimated_tokens": 0,
                "processing_time": 0
            }
            
            # Add successful direct results
            for file_path, result in direct_results:
                if "error" not in result:
                    merged_results.append((file_path, result))
                    merged_stats["successful_files"] += 1
                    if "total_token_count" in result:
                        merged_stats["total_tokens"] += result["total_token_count"]
                else:
                    # Find corresponding text result
                    text_result = None
                    for text_file_path, text_result_data in text_results:
                        if text_file_path == file_path:
                            text_result = text_result_data
                            break
                    
                    if text_result and "error" not in text_result:
                        merged_results.append((file_path, text_result))
                        merged_stats["successful_files"] += 1
                        if "total_token_count" in text_result:
                            merged_stats["total_tokens"] += text_result["total_token_count"]
                    else:
                        merged_results.append((file_path, result))
                        merged_stats["failed_files"] += 1
            
            merged_stats["processing_time"] = int(time.time() - group_start_time)
            merged_stats["estimated_tokens"] = direct_stats.get("estimated_tokens", 0) + text_stats.get("estimated_tokens", 0)
            
            logging.info(f"✅ Hybrid: Combined processing complete for group {group_index}: {merged_stats['successful_files']} successful, {merged_stats['failed_files']} failed")
            
            return merged_results, merged_stats, group_id
            
        except Exception as e:
            logging.error(f"Error in hybrid processing for group {group_index}: {e}")
            # Fallback to text-first processing
            logging.info(f"🔄 Hybrid: Falling back to text-first processing for group {group_index}")
            
            return self.text_first_processor.process_file_group(
                config_manager=config_manager,
                file_group=file_group,
                group_index=group_index,
                group_id=f"{group_id}_fallback",
                system_prompt=system_prompt,
                user_prompt=user_prompt
            ) 