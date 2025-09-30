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

                # Add blacklist information if present
                blacklisted = entry.get("blacklisted", {})
                if blacklisted:
                    item["blacklisted"] = blacklisted
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
            "prompt_tokens": 0,
            "candidate_tokens": 0,
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
                
                # Store successful results and update status to Processed
                for file_path, result in successful_results.items():
                    if file_path not in per_file_result:
                        per_file_result[file_path] = result
                # Update remaining files for next subchain based on passthrough statuses
                # Rerun only files whose status is Pending, Unmatched, or Blacklisted.
                # Files marked as Matched should not be rerun.
                try:
                    # Support both the internal passthrough shape {"files": [...]} and
                    # a direct list (summary-style) where entries may use key "file".
                    root = (self.passthrough or {})
                    files_list = root.get("files", root if isinstance(root, list) else [])
                    rerun_statuses = {"pending", "unmatched", "failed"}
                    next_remaining: List[str] = []
                    seen: set = set()
                    for entry in files_list:
                        try:
                            fp = entry.get("file_path") or entry.get("file")
                            if not fp:
                                continue
                            status = str(entry.get("status", "Pending")).strip().lower()
                            if status in rerun_statuses and fp not in seen:
                                next_remaining.append(fp)
                                seen.add(fp)
                        except Exception:
                            continue
                    logging.info(f"🔁 Selecting {len(next_remaining)} file(s) to rerun for next subchain based on passthrough status")
                    remaining_files = next_remaining
                except Exception as e:
                    logging.info(f"⚠️ Passthrough scan failed; falling back to failed_files for next subchain: {e}")
                    remaining_files = failed_files



        # Any file not finalized after all subchains => failure
        for file_path in file_group:
            file_name = Path(file_path).name
            if file_path not in per_file_result:
                result = {"error": "All chained subchains exhausted without success"}
                per_file_result[file_path] = result
                # Files that never got processed remain Pending
                self._update_file_status(file_path, "Pending")
                logging.info(f"❌ Chain exhausted: {file_path}")
            else:
                # Files that completed processing should be marked as Completed
                if "error" not in per_file_result[file_path]:
                    self._update_file_status(file_path, "Completed")
                # Files with errors remain in their current status

        # Convert file_path keys to file_name keys for final output
        final_results = {}
        for file_path, result in per_file_result.items():
            file_name = Path(file_path).name
            final_results[file_name] = result

        merged_results = [(fp, self.passthrough.get("files", [])) for fp in file_group]
        agg_stats["successful_files"] = sum(1 for _fp, res in merged_results if "error" not in res)
        agg_stats["failed_files"] = agg_stats["total_files"] - agg_stats["successful_files"]
        agg_stats["processing_time"] = max(agg_stats["processing_time"], int(time.time() - start_time))

        # Transform self.passthrough to [(file_name, result)] format, including unmatch_detail if present
        passthrough_results = []
        passthrough_files = self.passthrough.get("files", [])
        for entry in passthrough_files:
            file_path = entry.get("file_path") or entry.get("file")
            if not file_path:
                continue
            file_name = Path(file_path).name
            result = {}
            # Add processing_output if present
            if "processing_output" in entry and entry["processing_output"] is not None:
                result["processing"] = entry["processing_output"]
            # Add post_processing if present (simulate post-processing result)
            if "postprocessing_output" in entry and entry["postprocessing_output"] is not None:
                result["post_processing"] = entry["postprocessing_output"]
            elif "extracted_data" in entry and entry["extracted_data"] is not None:
                result["post_processing"] = {
                    "postprocessed": True,
                    "postprocessing_type": "metadata",
                    "metadata": entry["extracted_data"]
                }
            # Add unmatch_detail if present
            if "unmatch_detail" in entry and entry["unmatch_detail"]:
                result["unmatch_detail"] = entry["unmatch_detail"]
            if "status" in entry and entry["status"]:
                result["status"] = entry["status"]
            passthrough_results.append((file_name, result))

        return passthrough_results, agg_stats, group_id

    def _update_file_status(self, file_path: str, status: str) -> None:
        """Update the status of a file in the passthrough."""
        if self.passthrough is None:
            return
        # Find the file entry in passthrough
        for entry in self.passthrough.get("files", []):
            if entry.get("file_path") == file_path:
                entry["status"] = status
                logging.debug(f"📊 Updated status for {file_path}: {status}")
                break

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

            processing_results, processing_stats, processing_stage_status = processing_strategy.process_file_group(
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
            agg_stats["prompt_tokens"] += processing_stats.get("prompt_tokens", 0)
            agg_stats["candidate_tokens"] += processing_stats.get("candidate_tokens", 0)
            
            # If processing was explicitly skipped due to missing metadata (e.g., regex),
            # do not write anything to passthrough and do not mark success/failure here.
            skip_passthrough_writes = (
                processing_type == "regex" and processing_stage_status == "skipped_no_metadata"
            )

            # Process results and check for success/failure
            for file_path, result in processing_results:

                if skip_passthrough_writes:
                    logging.info(f"⏭️ Skipping passthrough writes for {file_path} due to no metadata ({processing_type})")
                    # Ensure we don't mark success/failure; leave status as-is for next subchain
                    if file_path not in subchain_results:
                        subchain_results[file_path] = {}
                    continue

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
                    logging.info(f"model_output (from result): {model_output}")
                except Exception as e:
                    logging.info(f"Exception when getting model_output: {e}")
                    model_output = result

                try:
                    files_list = passthrough.setdefault("files", [])
                    matched_entry = None
                    for entry in files_list:
                        if file_path == entry.get("file_path"):
                            matched_entry = entry
                            logging.info(f"Matched entry found (absolute match): {matched_entry}")
                            break
                        try:
                            if Path(str(file_path)).name == Path(str(entry.get("file_path"))).name:
                                matched_entry = entry
                                logging.info(f"Matched entry found (basename match): {matched_entry}")
                                break
                        except Exception:
                            pass
                    if matched_entry is None:
                        matched_entry = {"file_path": file_path, "status": "Pending", "extracted_data": {}}
                        files_list.append(matched_entry)
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
                        if isinstance(model_output, dict) and model_output != {}:
                            # Helper to pick from multiple possible keys
                            def pick(obj, keys):
                                for k in keys:
                                    if k in obj:
                                        return obj[k]
                                    if k.upper() in obj:
                                        return obj[k.upper()]
                                    if k.lower() in obj:
                                        return obj[k.lower()]
                                return None

                            normalized["type"] = pick(model_output, ["class", "type", "document_type", "DOC_TYPE"])
                            normalized["claim_no"] = pick(model_output, ["collected_ClaimNO", "claim_no", "CLAIM_NO", "CLAIM_NUMBER"])
                            normalized["vin"] = pick(model_output, ["collected_VIN", "vin", "VIN", "Chassi"])
                            normalized["cnpj"] = pick(model_output, ["collected_CNPJ", "cnpj", "CNPJ", "CNPJ_1"])
                            normalized["cnpj2"] = pick(model_output, ["collected_CNPJ2", "cnpj2", "CNPJ2", "CNPJ_2"])
                            if normalized["type"] == "Serviço":
                                normalized["parts_value"] = None
                                normalized["service_value"] = pick(model_output, ["collected_service_price", "service_value", "SERVICE_VALUE", "labour_amount", "VALOR_TOTAL"])
                            if normalized["type"] == "Peças":
                                normalized["parts_value"] = pick(model_output, ["collected_parts_price", "parts_value", "PARTS_VALUE", "part_amount", "VALOR_TOTAL"])
                                normalized["service_value"] = None
                        matched_entry["processing_extracted_data"] = normalized
                    except Exception:
                        pass
                    logging.info(f"Updated matched_entry with processing_output: {matched_entry}")
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
                    logging.info(f"Exception in passthrough file cache: {e}")
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
                        logging.info(f"Checking mandatory keys for file_path {file_path}")
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
                agg_stats["prompt_tokens"] += post_stats.get("prompt_tokens", 0)
                agg_stats["candidate_tokens"] += post_stats.get("candidate_tokens", 0)
                
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

        for file_path, results in subchain_results.items():
            if "error" not in results:
                successful_results[file_path] = results
            else:
                logging.info(f"File '{file_path}' has error: {results.get('error')}")
        
        for file_path in successful_results:
            logging.info(f"Successful file: {file_path}")

        logging.info(f"🔁 Subchain '{subchain_name}' complete: successful={len(successful_results)}, failed={len(failed_files)}")
        self._log_passthrough(f"end_subchain:{subchain_name}", passthrough)
        
        return successful_results, failed_files




