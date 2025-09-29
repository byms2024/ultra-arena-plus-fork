"""
Advanced Chain processing strategy: execute chains of subchains with pre-processing, processing, and post-processing links.
"""

from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from pathlib import Path
from collections import Counter
import re
import json
import inspect

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

    # --- Logging helpers for passthrough introspection ---
    def _make_passthrough_summary(self, passthrough: Dict[str, Any]) -> List[Dict[str, Any]]:
        files_list = (passthrough or {}).get("files", [])
        summary: List[Dict[str, Any]] = []
        for entry in files_list:
            try:
                item: Dict[str, Any] = {
                    "file": entry.get("file_path"),
                    "status": entry.get("status"),
                }
                extracted = entry.get("extracted_data", {}) or {}
                if isinstance(extracted, dict):
                    for k in [
                        "type",
                        "claim_no",
                        "vin",
                        "cnpj",
                        "cnpj1",
                        "gross_credit_dms",
                        "labour_amount_dms",
                        "part_amount_dms",
                    ]:
                        if k in extracted:
                            item[f"dms.{k}"] = extracted.get(k)
                proc = entry.get("processing_output", {})
                if isinstance(proc, dict):
                    for k in [
                        "type",
                        "claim_no",
                        "vin",
                        "cnpj",
                        "cnpj2",
                        "parts_value",
                        "service_value",
                    ]:
                        if k in proc:
                            item[f"proc.{k}"] = proc.get(k)
                else:
                    if proc is not None:
                        item["proc.raw"] = proc
                proc_norm = entry.get("processing_extracted_data", {}) or {}
                if isinstance(proc_norm, dict):
                    for k in [
                        "type",
                        "claim_no",
                        "vin",
                        "cnpj",
                        "cnpj2",
                        "parts_value",
                        "service_value",
                    ]:
                        if k in proc_norm:
                            item[f"proc.{k}"] = proc_norm.get(k)
                if "unmatch_detail" in entry:
                    item["unmatch_detail"] = entry.get("unmatch_detail")
                summary.append(item)
            except Exception:
                try:
                    summary.append({"file": entry.get("file_path"), "status": entry.get("status"), "error": "summary_failed"})
                except Exception:
                    summary.append({"error": "summary_failed_unknown_entry"})
        return summary

    def _log_passthrough(self, where: str, passthrough: Dict[str, Any]) -> None:
        try:
            snapshot = self._make_passthrough_summary(passthrough)
            logging.info(f"🔎 Passthrough snapshot [{where}]: {json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)}")
        except Exception as e:
            logging.info(f"🔎 Passthrough snapshot [{where}] failed: {e}")

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
        self._log_passthrough("init", self.passthrough)
        

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
                {**self.config, **pre_config, 'censor' : censor}, 
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
            
            for pre_data in pre_results:
                files = getattr(pre_data, "files", None)
                file_texts = getattr(pre_data, "file_texts", None)
                file_classes = getattr(pre_data, "file_classes", None)
                answers = getattr(pre_data, "answers", None)
                if not files or file_texts is None or file_classes is None:
                    continue
                for file_path in files:
                    subchain_results[str(file_path)] = {
                        "pre_processing": {
                            "file_text": file_texts.get(file_path) if isinstance(file_texts, dict) else None,
                            "file_class": file_classes.get(file_path) if isinstance(file_classes, dict) else None,
                            "answers": answers,
                        }
                    }
            self._log_passthrough(f"after_pre:{subchain_name}", passthrough)
                
        except Exception as e:
            logging.error(f"❌ Pre-processing failed for subchain '{subchain_name}': {e}")
            # On pre-processing failure, mark all files as failed for this subchain
            for file_path in current_files:
                subchain_results[file_path] = {"error": f"Pre-processing failed: {e}"}
            return {}, current_files  # Return empty successful results, all files as failed)


        # 2. Processing link
        processing_config = subchain_config["processing"]
        # Accept new key name 'proc-type' with fallback to legacy 'type'
        processing_type = processing_config.get("proc-type", processing_config.get("type"))
        
        logging.info(f"   ⚙️ Processing ({processing_type}) for subchain '{subchain_name}'")

        from config import config_base

        user_prompt = config_base.SENSITIVE_USER_PROMPT if not censor else config_base.USER_PROMPT
        system_prompt = config_base.SYSTEM_PROMPT      

        successful_files = []
        failed_files = []
        
        try:
            processing_strategy = ProcessingLinkFactory.create_strategy(
                processing_type, 
                {**self.config, **processing_config, 'censor' : censor}, 
                streaming=self.streaming
            )
            # share passthrough with processing link
            _attach = getattr(processing_strategy, "attach_passthrough", None)
            if callable(_attach):
                _attach(passthrough)
            
            self._log_passthrough(f"before_processing:{subchain_name}", passthrough)
            extra_kwargs: Dict[str, Any] = {}
            try:
                sig = inspect.signature(processing_strategy.process_file_group)  # type: ignore[attr-defined]
                if "pre_results" in sig.parameters:
                    extra_kwargs["pre_results"] = pre_results
            except Exception:
                pass

            processing_results, processing_stats, _ = processing_strategy.process_file_group(
                config_manager=config_manager,
                file_group=current_files,
                group_index=group_index,
                group_id=f"{group_id}_{subchain_name}_processing",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                **extra_kwargs
            )

            agg_stats["estimated_tokens"] += processing_stats.get("estimated_tokens", 0)
            agg_stats["total_tokens"] += processing_stats.get("total_tokens", 0)
            agg_stats["processing_time"] += processing_stats.get("processing_time", 0)
            
            print('🚍'*100)
            # Process results and check for success/failure
            for file_path, result in processing_results:
                
                print(file_path)
                print(result)

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
                            print(f"Matched entry found (absolute match): {matched_entry}")
                            break
                        try:
                            if Path(str(file_path)).name == Path(str(entry.get("file_path"))).name:
                                matched_entry = entry
                                print(f"Matched entry found (basename match): {matched_entry}")
                                break
                        except Exception:
                            pass
                    if matched_entry is None:
                        matched_entry = {"file_path": file_path, "status": "Pending", "extracted_data": {}}
                        files_list.append(matched_entry)
                        print(f"No matched entry, created new: {matched_entry}")
                    matched_entry["processing_output"] = model_output
                    # Also set normalized processing extracted data for visibility and downstream usage
                    try:
                        normalized: Dict[str, Any] = {
                            "type": None,
                            "claim_no": None,
                            "vin": None,
                            "cnpj": None,
                            "cnpj2": None,
                            "parts_value": None,
                            "service_value": None,
                        }
                        if isinstance(model_output, dict):
                            normalized["type"] = model_output.get("class")
                            normalized["claim_no"] = model_output.get("collected_ClaimNO")
                            normalized["vin"] = model_output.get("collected_VIN")
                            normalized["cnpj"] = model_output.get("collected_CNPJ")
                            normalized["cnpj2"] = model_output.get("collected_CNPJ2")
                            normalized["parts_value"] = model_output.get("collected_parts_price")
                            normalized["service_value"] = model_output.get("collected_service_price")
                        matched_entry["processing_extracted_data"] = normalized
                    except Exception:
                        pass
                    print(f"Updated matched_entry with processing_output: {matched_entry}")
                    try:
                        short_name = None
                        try:
                            short_name = Path(file_path).name
                        except Exception:
                            short_name = str(file_path)
                        self._log_passthrough(f"after_processing_update:{subchain_name}:{short_name}", passthrough)
                    except Exception:
                        pass
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
        finally:
            self._log_passthrough(f"after_processing:{subchain_name}", passthrough)
        
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
                
                self._log_passthrough(f"before_post:{subchain_name}", passthrough)
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
                self._log_passthrough(f"after_post:{subchain_name}", passthrough)
                        
            except Exception as e:
                logging.error(f"❌ Post-processing failed for subchain '{subchain_name}': {e}")
                # Post-processing failure doesn't fail the whole subchain, just log it
        
        # Separate successful and failed results
        successful_results = {}
        #print("---- DEBUG: subchain_results ----")
        for file_path, results in subchain_results.items():
            #print(f"File: {file_path}")
            #print(f"Results: {results}")
            if "error" not in results:
                #print(f"File '{file_path}' is successful, adding to successful_results.")
                successful_results[file_path] = results
            else:
                print(f"File '{file_path}' has error: {results.get('error')}")
        #print(f"---- DEBUG: successful_results ({len(successful_results)}) ----")
        for file_path in successful_results:
            print(f"Successful file: {file_path}")
        
        logging.info(f"🔁 Subchain '{subchain_name}' complete: successful={len(successful_results)}, failed={len(failed_files)}")
        self._log_passthrough(f"end_subchain:{subchain_name}", passthrough)
        
        return successful_results, failed_files




