"""
Advanced Chain processing strategy: execute chains of subchains with pre-processing, processing, and post-processing links.
"""

from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from pathlib import Path
from collections import Counter
import re

# from common.text_extractor import TextExtractor

from .base_strategy import BaseProcessingStrategy

from .strategy_factory import ProcessingLinkFactory
from .strategy_factory import PreProcessingLinkFactory
from .strategy_factory import PostProcessingLinkFactory



class ChainedProcessingStrategy(BaseProcessingStrategy):


    def __init__(self, config: Dict[str, Any], streaming: bool = False):
        super().__init__(config)
        self.streaming = streaming
        
        # Parse the new chain structure
        self.chain_config = config.get("chain_config", {})

        if not self.chain_config:
            raise ValueError("chains configuration must be provided for AdvancedChainedProcessingStrategy")
        
        self.chain_on_missing_keys = config.get("chain_on_missing_keys", False)
        
        # Validate chain configuration
        self._validate_chain_config()

    def _validate_chain_config(self):
        """Validate the chain configuration structure."""
        if not isinstance(self.chain_config, dict):
            raise ValueError("chains must be a dictionary")
        
        for _, subchains in self.chain_config.items():
            subchain_list = subchains['subchains']

            for subchain in subchain_list:
                if not isinstance(subchain, dict):
                    raise ValueError(f"Subchain must be a dictionary")
                 
                logging.info(f"⛓⛓⛓ Received subchain:{subchain['subchain_name']} ⛓⛓⛓")

                required_keys = [("pre-processing","pre-type"), ("processing","proc-type"), ("post-processing","post-type")]

                for stage, p_type in required_keys:
                    if stage not in subchain:
                        raise ValueError(f"Subchain '{subchain['subchain_name']}' missing required key: {stage}")
                    
                    if not isinstance(subchain[stage], dict):
                        raise ValueError(f"Subchain '{subchain['subchain_name']}' {stage} must be a dictionary")
                    
                    if p_type not in subchain[stage]:
                        raise ValueError(f"Subchain '{subchain['subchain_name']}' {stage} missing {p_type} field")
                    
                    logging.info(f'{stage}: {subchain[stage][p_type]}')

    def process_file_group(self, *, config_manager=None, file_group: List[str], group_index: int,
                           group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str
                           ) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        start_time = time.time()
        remaining_files = list(file_group)
        per_file_result: Dict[str, Dict] = {}

        agg_stats = {
            "total_files": len(file_group),
            "successful_files": 0,
            "failed_files": 0,
            "total_tokens": 0,
            "estimated_tokens": 0,
            "processing_time": 0
        }

        # Initialize passthrough with all files marked as Pending
        self.passthrough = {
            "files": [
                {"file_path": fp, "status": "Pending", "extracted_data": {}} for fp in file_group
            ]
        }
        

        # Execute each subchain in sequence, iterating through chain groups and their subchains
        for group_name, group_cfg in self.chain_config.items():
            subchain_list = group_cfg.get("subchains", [])
            for idx, subchain_cfg in enumerate(subchain_list):
                if not remaining_files:
                    break

                subchain_name = subchain_cfg.get("subchain_name", f"{group_name}_subchain_{idx}")
                logging.info(f"🔗 Executing subchain '{subchain_name}' on {len(remaining_files)} file(s)")
                
                # Process files through the three links of this subchain
                successful_results, failed_files = self._execute_subchain(
                    subchain_name,
                    subchain_cfg,
                    remaining_files,
                    config_manager,
                    group_index,
                    group_id,
                    system_prompt,
                    user_prompt,
                    agg_stats,
                    self.passthrough,
                )
                
                # Store successful results
                for file_path, result in successful_results.items():
                    if file_path not in per_file_result:
                        per_file_result[file_path] = result
                # Update remaining files for next subchain
                remaining_files = failed_files


        # Any file not finalized after all subchains => failure
        for file_path in file_group:
            file_name = Path(file_path).name
            found = False
            for k in per_file_result.keys():
                if file_name in k:
                    found = True
                    break
            if not found:
                per_file_result[file_name] = {"error": "All chained subchains exhausted without success"}
                logging.info(f"❌ Chain exhausted: {file_path}")

        merged_results = [(fp, per_file_result[Path(fp).name]) for fp in file_group]
        agg_stats["successful_files"] = sum(1 for _fp, res in merged_results if "error" not in res)
        agg_stats["failed_files"] = agg_stats["total_files"] - agg_stats["successful_files"]
        agg_stats["processing_time"] = max(agg_stats["processing_time"], int(time.time() - start_time))

        return merged_results, agg_stats, group_id

    def _execute_subchain(self, subchain_name: str, subchain_config: Dict[str, Any], 
                         file_group: List[str], config_manager, group_index: int, group_id: str,
                         system_prompt: Optional[str], user_prompt: str, agg_stats: Dict[str, Any],
                         passthrough: Dict[str, Any]) -> Tuple[Dict[str, Dict], List[str]]:
        """Execute a single subchain (pre-processing -> processing -> post-processing).
        
        Returns:
            Tuple of (successful_results_dict, failed_files_list)
        """
        
        current_files = file_group
        subchain_results = {}
        
        # Extract common subchain-level attributes as variables
        censor = subchain_config.get("censor", False)
        metadata_fields = subchain_config.get("metadata_fields", [])
        file_number_per_file = subchain_config.get("fileNumberPerFile", 1)
        
        # 1. Pre-processing link
        pre_config = subchain_config["pre-processing"]

        pre_type = pre_config.get("pre-type", pre_config.get("type"))
        
        logging.info(f"   📝 Pre-processing ({pre_type}) for subchain '{subchain_name}'")
        
        try:
            pre_strategy = PreProcessingLinkFactory.create_strategy(
                pre_type, 
                {**self.config, **pre_config}, 
                streaming=self.streaming
            )
            # share passthrough with pre-processing link
            _attach = getattr(pre_strategy, "attach_passthrough", None)
            if callable(_attach):
                _attach(passthrough)
            
            pre_results, pre_stats, _ = pre_strategy.process_file_group(
                config_manager=config_manager,
                file_group=current_files,
                group_index=group_index,
                group_id=f"{group_id}_{subchain_name}_pre",
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            agg_stats["estimated_tokens"] += pre_stats.get("estimated_tokens", 0)
            agg_stats["total_tokens"] += pre_stats.get("total_tokens", 0)
            agg_stats["processing_time"] += pre_stats.get("processing_time", 0)
            # pre_results is a list of PreprocessedData, so we need to extract per-file results
            # Each PreprocessedData contains: files (list[Path]), file_texts, file_classes, answers
            
            # for pre_data in pre_results:
            #     for file_path in pre_data.files:
            #         subchain_results[str(file_path.name)] = {
            #             "pre_processing": {
            #                 "file_text": pre_data.file_texts.get(file_path),
            #                 "file_class": pre_data.file_classes.get(file_path),
            #                 "answers": pre_data.answers,
            #             }
            #         }
                
        except Exception as e:
            logging.error(f"❌ Pre-processing failed for subchain '{subchain_name}': {e}")
            # On pre-processing failure, mark all files as failed for this subchain
            for file_path in current_files:
                subchain_results[file_path] = {"error": f"Pre-processing failed: {e}"}
            return {}, current_files  # Return empty successful results, all files as failed

        # 2. Processing link
        processing_config = subchain_config["processing"]
        # Accept new key name 'proc-type' with fallback to legacy 'type'
        processing_type = processing_config.get("proc-type", processing_config.get("type"))
        
        logging.info(f"   ⚙️ Processing ({processing_type}) for subchain '{subchain_name}'")
        
        successful_files = []
        failed_files = []
        
        try:
            processing_strategy = ProcessingLinkFactory.create_strategy(
                processing_type, 
                {**self.config, **processing_config}, 
                streaming=self.streaming
            )
            # share passthrough with processing link
            _attach = getattr(processing_strategy, "attach_passthrough", None)
            if callable(_attach):
                _attach(passthrough)
            
            processing_results, processing_stats, _ = processing_strategy.process_file_group(
                config_manager=config_manager,
                file_group=current_files,
                group_index=group_index,
                group_id=f"{group_id}_{subchain_name}_processing",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                pre_results = pre_results
            )

            agg_stats["estimated_tokens"] += processing_stats.get("estimated_tokens", 0)
            agg_stats["total_tokens"] += processing_stats.get("total_tokens", 0)
            agg_stats["processing_time"] += processing_stats.get("processing_time", 0)
            
            # Process results and check for success/failure
            for file_path, result in processing_results:
                
                # Ensure subchain_results entry exists for this file_path
                if file_path not in subchain_results:
                    subchain_results[file_path] = {}

                # Cache model output into passthrough for post-processing comparisons
                try:
                    # If result is a dict, use it directly; otherwise, wrap in dict
                    if isinstance(result, dict):
                        model_output = result.get("file_model_output", result)
                    else:
                        model_output = result
                    print(f"model_output (from result): {model_output}")
                except Exception as e:
                    print(f"Exception when getting model_output: {e}")
                    model_output = result

                try:
                    files_list = passthrough.setdefault("files", [])
                    matched_entry = None
                    for entry in files_list:
                        if file_path == entry.get("file_path"):
                            matched_entry = entry
                            break
                    if matched_entry is None:
                        matched_entry = {"file_path": file_path, "status": "Pending", "extracted_data": {}}
                        files_list.append(matched_entry)
                        print(f"No matched entry, created new: {matched_entry}")
                    matched_entry["processing_output"] = model_output
                except Exception as e:
                    print(f"Exception in passthrough file cache: {e}")
                    # Non-fatal; continue normal flow
                    pass


                # If result is not a dict, wrap it for consistency
                if not isinstance(result, dict):
                    result = {"file_model_output": result}

                if "error" in result:
                    subchain_results[file_path]["processing"] = result
                    subchain_results[file_path]["error"] = result["error"]
                    failed_files.append(file_path)
                    logging.info(f"➡️ Processing failed for {file_path}: {result.get('error')}")
                else:
                    # Check mandatory keys if enabled
                    if self.chain_on_missing_keys:
                        print(f"Checking mandatory keys for file_path {file_path}")
                        # Use the actual model output for key checking
                        ok, _missing = self.check_mandatory_keys(model_output, file_path, 
                                                                 getattr(self, "benchmark_comparator", None))
                        if not ok:
                            subchain_results[file_path]["processing"] = result
                            subchain_results[file_path]["error"] = "Missing mandatory keys after processing"
                            failed_files.append(file_path)
                            logging.info(f"➡️ Processing missing keys for {file_path}")
                            continue

                    # Success
                    subchain_results[file_path]["processing"] = result
                    successful_files.append(file_path)
                    logging.info(f"✅ Processing succeeded for {file_path}")
                    
        except Exception as e:
            logging.error(f"❌ Processing failed for subchain '{subchain_name}': {e}")
            # On processing failure, mark remaining files as failed
            for file_path in current_files:
                if file_path not in subchain_results:
                    subchain_results[file_path] = {"error": f"Processing failed: {e}"}
                else:
                    subchain_results[file_path]["processing"] = {"error": f"Processing failed: {e}"}
                    subchain_results[file_path]["error"] = f"Processing failed: {e}"
                failed_files.append(file_path)
        
        # 3. Post-processing link (only for successful files)
        post_config = subchain_config["post-processing"]
        # Accept new key name 'post-type' with fallback to legacy 'type'
        post_type = post_config.get("post-type", post_config.get("type"))
        # Extract retry configurations if present
        retries_cfg = post_config.get("retries", {})
        pre_retry_count = retries_cfg.get("pre_retry", {}).get("retry_count", 0)
        proc_retry_count = retries_cfg.get("proc_retry", {}).get("retry_count", 0)
        
        if successful_files and post_type != "none":
            logging.info(f"   📋 Post-processing ({post_type}) for subchain '{subchain_name}' on {len(successful_files)} successful file(s)")
            
            try:
                post_strategy = PostProcessingLinkFactory.create_strategy(
                    post_type, 
                    {**self.config, **post_config}, 
                    streaming=self.streaming
                )
                # share passthrough with post-processing link
                _attach = getattr(post_strategy, "attach_passthrough", None)
                if callable(_attach):
                    _attach(passthrough)
                
                post_results, post_stats, _ = post_strategy.process_file_group(
                    config_manager=config_manager,
                    file_group=successful_files,
                    group_index=group_index,
                    group_id=f"{group_id}_{subchain_name}_post",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
                
                agg_stats["estimated_tokens"] += post_stats.get("estimated_tokens", 0)
                agg_stats["total_tokens"] += post_stats.get("total_tokens", 0)
                agg_stats["processing_time"] += post_stats.get("processing_time", 0)
                
                # Store post-processing results
                for file_path, result in post_results:
                    if file_path in subchain_results:
                        subchain_results[file_path]["post_processing"] = result
                        
            except Exception as e:
                logging.error(f"❌ Post-processing failed for subchain '{subchain_name}': {e}")
                # Post-processing failure doesn't fail the whole subchain, just log it
        
        # Separate successful and failed results
        successful_results = {}
        print("---- DEBUG: subchain_results ----")
        for file_path, results in subchain_results.items():
            print(f"File: {file_path}")
            print(f"Results: {results}")
            if "error" not in results:
                print(f"File '{file_path}' is successful, adding to successful_results.")
                successful_results[file_path] = results
            else:
                print(f"File '{file_path}' has error: {results.get('error')}")
        print(f"---- DEBUG: successful_results ({len(successful_results)}) ----")
        for file_path in successful_results:
            print(f"Successful file: {file_path}")
        
        logging.info(f"🔁 Subchain '{subchain_name}' complete: successful={len(successful_results)}, failed={len(failed_files)}")

        return successful_results, failed_files




