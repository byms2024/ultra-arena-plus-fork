"""
Advanced Chain processing strategy: execute chains of subchains with pre-processing, processing, and post-processing links.
"""

from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from pathlib import Path
import re

from ..common.text_extractor import TextExtractor
from ..common.pdf_metadata import read_pdf_metadata_dict
from .base_strategy import BaseProcessingStrategy

from .strategy_factory import ProcessingLinkFactory
from .strategy_factory import PreProcessingLinkFactory
from .strategy_factory import PostProcessingLinkFactory


class PreProcessingStrategyFactory:
    """Factory for creating pre-processing strategies."""
    _passthrough: Optional[Dict[str, Any]] = None
    
    @classmethod
    def attach_passthrough(cls, passthrough: Dict[str, Any]) -> None:
        """Set a shared passthrough object that will be attached to created strategies."""
        cls._passthrough = passthrough

    @classmethod
    def create_strategy(cls, strategy_type: str, config: Dict[str, Any], streaming: bool = False) -> BaseProcessingStrategy:
        """Create a pre-processing strategy based on type and auto-attach passthrough if provided."""
        if strategy_type == "text":
            strategy = TextPreProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "image":
            strategy = ImagePreProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "file":
            strategy = FilePreProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "none":
            strategy = NoOpPreProcessingStrategy(config, streaming=streaming)
        else:
            raise ValueError(f"Unsupported pre-processing strategy type: {strategy_type}")

        if cls._passthrough is not None:
            _attach = getattr(strategy, "attach_passthrough", None)
            if callable(_attach):
                _attach(cls._passthrough)
        return strategy


class ProcessingStrategyFactory:
    """Factory for creating processing strategies."""
    _passthrough: Optional[Dict[str, Any]] = None
    
    @classmethod
    def attach_passthrough(cls, passthrough: Dict[str, Any]) -> None:
        """Set a shared passthrough object that will be attached to created strategies."""
        cls._passthrough = passthrough

    @classmethod
    def create_strategy(cls, strategy_type: str, config: Dict[str, Any], streaming: bool = False) -> BaseProcessingStrategy:
        """Create a processing strategy based on type and auto-attach passthrough if provided."""
        if strategy_type == "text_first":
            from .enhanced_text_first_strategy import EnhancedTextFirstProcessingStrategy
            strategy = EnhancedTextFirstProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "image_first":
            from .image_first_strategy import ImageFirstProcessingStrategy
            strategy = ImageFirstProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "file_first":
            from .direct_file_strategy import DirectFileProcessingStrategy
            strategy = DirectFileProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "none":
            # No processing
            strategy = NoOpProcessingStrategy(config, streaming=streaming)
        else:
            raise ValueError(f"Unsupported processing strategy type: {strategy_type}")

        if cls._passthrough is not None:
            _attach = getattr(strategy, "attach_passthrough", None)
            if callable(_attach):
                _attach(cls._passthrough)
        return strategy

class PostProcessingStrategyFactory:
    """Factory for creating post-processing strategies."""
    _passthrough: Optional[Dict[str, Any]] = None
    
    @classmethod
    def attach_passthrough(cls, passthrough: Dict[str, Any]) -> None:
        """Set a shared passthrough object that will be attached to created strategies."""
        cls._passthrough = passthrough
    
    @classmethod
    def create_strategy(cls, strategy_type: str, config: Dict[str, Any], streaming: bool = False) -> BaseProcessingStrategy:
        """Create a post-processing strategy based on type and auto-attach passthrough if provided."""
        if strategy_type == "metadata":
            strategy = MetadataPostProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "none":
            # No post-processing
            strategy = NoOpPostProcessingStrategy(config, streaming=streaming)
        else:
            raise ValueError(f"Unsupported post-processing strategy type: {strategy_type}")

        if cls._passthrough is not None:
            _attach = getattr(strategy, "attach_passthrough", None)
            if callable(_attach):
                _attach(cls._passthrough)
        return strategy


# Base class for all link strategies (pre-processing, processing, post-processing)
class LinkStrategy(BaseProcessingStrategy):
    """Base class for chain link strategies."""
    
    def __init__(self, config: Dict[str, Any], streaming: bool = False):
        super().__init__(config)
        self.streaming = streaming
        # Shared passthrough state across subchains/links
        self.passthrough: Optional[Dict[str, Any]] = None

    def attach_passthrough(self, passthrough: Dict[str, Any]) -> None:
        """Attach shared passthrough object so links can read/update file statuses and extracted data."""
        self.passthrough = passthrough

    # --- Extension hooks for future status management/blacklisting ---
    def _get_or_create_file_entry(self, file_path: str) -> Dict[str, Any]:
        if self.passthrough is None:
            return {"file_path": file_path}
        files_list = self.passthrough.setdefault("files", [])
        for entry in files_list:
            if entry.get("file_path") == file_path:
                return entry
        new_entry = {"file_path": file_path, "status": "Pending", "extracted_data": {}}
        files_list.append(new_entry)
        return new_entry

    def update_status(self, file_path: str, status: str, unmatch_detail: Optional[List[str]] = None) -> None:
        if self.passthrough is None:
            return
        entry = self._get_or_create_file_entry(file_path)
        entry["status"] = status
        if unmatch_detail is not None:
            entry["unmatch_detail"] = unmatch_detail

    def update_extracted_data(self, file_path: str, updates: Dict[str, Any]) -> None:
        if self.passthrough is None:
            return
        entry = self._get_or_create_file_entry(file_path)
        data = entry.setdefault("extracted_data", {})
        data.update(updates)


# Pre-processing strategies
class TextPreProcessingStrategy(LinkStrategy):
    """Pre-processing strategy for text-based operations."""

    def __init__(self, config: Dict[str, Any], streaming: bool = False):
        super().__init__(config, streaming)
        # Allow providing regex criteria via config; default to empty dict
        self.regex_criteria = config.get("text_first_regex_criteria", config.get("regex_criteria", {})) or {}

    def extraction_evaluation_regex(self, text_content):

        if not text_content:
            logging.warning(f"⚠️ Cannot evaluate empty text from extractor")
            return 0

        successful_matches = 0
        match_details = []
        
        for field_name, regex_pattern in self.regex_criteria.items():
            try:
                matches = re.findall(regex_pattern, text_content)
                if matches:
                    successful_matches += 1
                    match_details.append(f"{field_name}: {len(matches)} match(es)")

            except Exception as e:
                logging.error(f"❌ Regex evaluation failed for {field_name}: {e}")
        
        return successful_matches


    def extract_text(self, pdf_path):
        """Try to extract text with two different approach."""

        logging.info(f"🔄 Extracting text from PDF: {Path(pdf_path).name}")

        # Set extractors
        self.primary_extractor = TextExtractor('pymupdf')
        self.secundary_extractor = TextExtractor('pytesseract')

        # Extract with Primary Extractor
        primary_text = self.primary_extractor.extract_text(pdf_path, max_length=50000)
        primary_score = self.extraction_evaluation_regex(primary_text)
        chosen_text = primary_text

        # Decides whether should try the second option
        should_try_secondary = (
            ((not primary_text) or (len(primary_text) < 1000)) and 
            self.primary_extractor.extractor_lib != self.secundary_extractor.extractor_lib
        )

        # Tries second option and decides which text to use 
        if should_try_secondary:
            secondary_text = self.secundary_extractor.extract_text(pdf_path, max_length=50000)
            secondary_score = self.extraction_evaluation_regex(secondary_text)

            # Only selects the second if 2_score > 1_score or only the second has actually extracted text. 
            if (secondary_score, bool(secondary_text)) > (primary_score, bool(primary_text)):
                chosen_text = secondary_text
        
        return chosen_text

    def process_file_group(self, *, config_manager=None, file_group: List[str], group_index: int,
                           group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str = "") -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Apply text pre-processing to files."""
        start_time = time.time()
        results = []
        
        for file_path in file_group:
            # Optionally read and store PDF metadata (including DmsData)
            if self.config.get("enable_pdf_metadata", False):
                meta = read_pdf_metadata_dict(file_path)
                # Push parsed DMS data into passthrough extracted_data
                dms = meta.get("dms_data") or {}
                if dms:
                    # Map keys we care about directly
                    mapped = {
                        "claim_id": dms.get("claim_id"),
                        "claim_no": dms.get("claim_no"),
                        "vin": dms.get("vin"),
                        "dealer_code": dms.get("dealer_code"),
                        "dealer_name": dms.get("dealer_name"),
                        "cnpj1": dms.get("dealer_cnpj"),  # BYD CNPJ per example
                        "gross_credit_dms": dms.get("gross_credit"),
                        "labour_amount_dms": dms.get("labour_amount_dms"),
                        "part_amount_dms": dms.get("part_amount_dms"),
                        "dms_file_id": dms.get("file_id"),
                        "dms_embedded_at": dms.get("embedded_at"),
                    }
                    self.update_extracted_data(file_path, {k: v for k, v in mapped.items() if v is not None})
                # Optionally keep raw document info
                if self.config.get("store_raw_pdf_info", False):
                    self.update_extracted_data(file_path, {"pdf_document_info": meta.get("document_info", {})})

            text_content = self.extract_text(file_path)

            result = {"preprocessed": True, "preprocessing_type": "text"}
            results.append((file_path, result))
        
        agg_stats = {
            "total_files": len(file_group),
            "successful_files": len(file_group),
            "failed_files": 0,
            "total_tokens": 0,
            "estimated_tokens": 0,
            "processing_time": int(time.time() - start_time)
        }
        
        return results, agg_stats, group_id


class ImagePreProcessingStrategy(LinkStrategy):
    """Pre-processing strategy for image-based operations."""
    
    def process_file_group(self, *, config_manager=None, file_group: List[str], group_index: int,
                           group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str = "") -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Apply image pre-processing to files."""
        start_time = time.time()
        results = []
        
        for file_path in file_group:
            # Placeholder: Implement image preprocessing logic here
            # For now, just pass through the file unchanged
            result = {"preprocessed": True, "preprocessing_type": "image"}
            results.append((file_path, result))
        
        agg_stats = {
            "total_files": len(file_group),
            "successful_files": len(file_group),
            "failed_files": 0,
            "total_tokens": 0,
            "estimated_tokens": 0,
            "processing_time": int(time.time() - start_time)
        }
        
        return results, agg_stats, group_id


class FilePreProcessingStrategy(LinkStrategy):
    """Pre-processing strategy for file-based operations."""
    
    def process_file_group(self, *, config_manager=None, file_group: List[str], group_index: int,
                           group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str = "") -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Apply file pre-processing to files."""
        start_time = time.time()
        results = []
        
        for file_path in file_group:
            # Placeholder: Implement file preprocessing logic here
            # For now, just pass through the file unchanged
            result = {"preprocessed": True, "preprocessing_type": "file"}
            results.append((file_path, result))
        
        agg_stats = {
            "total_files": len(file_group),
            "successful_files": len(file_group),
            "failed_files": 0,
            "total_tokens": 0,
            "estimated_tokens": 0,
            "processing_time": int(time.time() - start_time)
        }
        
        return results, agg_stats, group_id


class NoOpPreProcessingStrategy(LinkStrategy):
    """No-operation pre-processing strategy."""
    
    def process_file_group(self, *, config_manager=None, file_group: List[str], group_index: int,
                           group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str = "") -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """No pre-processing - pass files through unchanged."""
        start_time = time.time()
        results = []
        
        for file_path in file_group:
            result = {"preprocessed": False, "preprocessing_type": "none"}
            results.append((file_path, result))
        
        agg_stats = {
            "total_files": len(file_group),
            "successful_files": len(file_group),
            "failed_files": 0,
            "total_tokens": 0,
            "estimated_tokens": 0,
            "processing_time": int(time.time() - start_time)
        }
        
        return results, agg_stats, group_id


# Processing strategies
class NoOpProcessingStrategy(LinkStrategy):
    """No-operation processing strategy."""
    
    def process_file_group(self, *, config_manager=None, file_group: List[str], group_index: int,
                           group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str = "") -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """No processing - pass files through unchanged."""
        start_time = time.time()
        results = []
        
        for file_path in file_group:
            result = {"processed": False, "processing_type": "none"}
            results.append((file_path, result))
        
        agg_stats = {
            "total_files": len(file_group),
            "successful_files": len(file_group),
            "failed_files": 0,
            "total_tokens": 0,
            "estimated_tokens": 0,
            "processing_time": int(time.time() - start_time)
        }
        
        return results, agg_stats, group_id


# Post-processing strategies
class MetadataPostProcessingStrategy(LinkStrategy):
    """Post-processing strategy for adding metadata."""
    
    def __init__(self, config: Dict[str, Any], streaming: bool = False):
        super().__init__(config, streaming)
        self.metadata_fields = config.get("metadata_fields", {})
        self.retry_processing = config.get("retry_processing", False)
        self.retry_pre_processing = config.get("retry_pre_processing", False)
        self.retry_count_processing = config.get("retry_count_processing", 3)
        self.retry_count_pre_processing = config.get("retry_count_pre_processing", 3)
        self.error_during_processing = config.get("error_during_processing", None)
        self.retry_count = config.get("retry_count", 3)
    
    def process_file_group(self, *, config_manager=None, file_group: List[str], group_index: int,
                           group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str = "") -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Apply metadata post-processing to files."""
        start_time = time.time()
        results = []
        
        for file_path in file_group:
            # Merge configured metadata with any pre-extracted passthrough metadata
            merged_metadata = self.metadata_fields.copy()
            if self.passthrough:
                # Find corresponding entry
                for entry in self.passthrough.get("files", []):
                    if entry.get("file_path") == file_path:
                        # Bring DMS mapped values into metadata namespace for downstream use
                        x = entry.get("extracted_data", {})
                        for k in [
                            "claim_id", "claim_no", "vin", "dealer_code", "dealer_name", "cnpj1",
                            "gross_credit_dms", "labour_amount_dms", "part_amount_dms", "dms_file_id", "dms_embedded_at",
                        ]:
                            if k in x and x[k] is not None:
                                merged_metadata[k] = x[k]
                        break

            result = {
                "postprocessed": True,
                "postprocessing_type": "metadata",
                "metadata": merged_metadata
            }
            results.append((file_path, result))

            # Status evaluation: compare processing output vs DMS metadata and mark as Matched if consistent
            try:
                if self.passthrough:
                    for entry in self.passthrough.get("files", []):
                        if entry.get("file_path") == file_path:
                            proc_raw = entry.get("processing_output", {}) or {}
                            dms = entry.get("extracted_data", {}) or {}

                            def pick(obj, keys):
                                for k in keys:
                                    if k in obj:
                                        return obj[k]
                                    if k.upper() in obj:
                                        return obj[k.upper()]
                                    if k.lower() in obj:
                                        return obj[k.lower()]
                                return None

                            # Normalize processing result into extracted_data schema
                            proc_fields = {
                                "type": pick(proc_raw, ["type", "DOC_TYPE", "document_type"]),
                                "cnpj": pick(proc_raw, ["cnpj", "CNPJ"]),
                                "cnpj2": pick(proc_raw, ["cnpj2", "CNPJ2"]),
                                "vin": pick(proc_raw, ["vin", "VIN"]),
                                "claim_no": pick(proc_raw, ["claim_no", "CLAIM_NO"]),
                                "parts_value": pick(proc_raw, ["parts_value", "PARTS_VALUE", "PART_AMOUNT", "part_amount"]),
                                "service_value": pick(proc_raw, ["service_value", "SERVICE_VALUE", "LABOUR_AMOUNT", "labour_amount"]),
                            }

                            # Compare with DMS metadata (from pre-processing)
                            issues = []
                            # Only compare keys that exist in DMS
                            if dms.get("claim_no") is not None:
                                if proc_fields.get("claim_no") is None or str(proc_fields["claim_no"]).strip() != str(dms.get("claim_no")).strip():
                                    issues.append("claim_no")
                            if dms.get("vin") is not None:
                                if proc_fields.get("vin") is None or str(proc_fields["vin"]).strip() != str(dms.get("vin")).strip():
                                    issues.append("vin")
                            if dms.get("cnpj1") is not None:
                                if proc_fields.get("cnpj") is None or str(proc_fields["cnpj"]).strip() != str(dms.get("cnpj1")).strip():
                                    issues.append("cnpj")

                            if issues:
                                # Unmatched: record details and null-out problematic fields in extracted_data
                                entry["status"] = "Unmatched"
                                entry["unmatch_detail"] = issues
                                final_fields = dict(proc_fields)
                                for k in issues:
                                    # issues use extracted_data key names (claim_no, vin, cnpj)
                                    final_fields[k] = None
                                entry["extracted_data"] = final_fields
                            else:
                                # Matched: extracted_data is the processing result
                                entry["status"] = "Matched"
                                entry["extracted_data"] = proc_fields
                            break
            except Exception:
                # Non-fatal; continue
                pass
        
        agg_stats = {
            "total_files": len(file_group),
            "successful_files": len(file_group),
            "failed_files": 0,
            "total_tokens": 0,
            "estimated_tokens": 0,
            "processing_time": int(time.time() - start_time)
        }
        
        return results, agg_stats, group_id


class NoOpPostProcessingStrategy(LinkStrategy):
    """No-operation post-processing strategy."""
    
    def process_file_group(self, *, config_manager=None, file_group: List[str], group_index: int,
                           group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str = "") -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """No post-processing - pass files through unchanged."""
        start_time = time.time()
        results = []
        
        for file_path in file_group:
            result = {"postprocessed": False, "postprocessing_type": "none"}
            results.append((file_path, result))
        
        agg_stats = {
            "total_files": len(file_group),
            "successful_files": len(file_group),
            "failed_files": 0,
            "total_tokens": 0,
            "estimated_tokens": 0,
            "processing_time": int(time.time() - start_time)
        }
        
        return results, agg_stats, group_id


class ChainedProcessingStrategy(BaseProcessingStrategy):


    def __init__(self, config: Dict[str, Any], streaming: bool = False):
        super().__init__(config)
        self.streaming = streaming
        
        # Parse the new chain structure
        self.chains_config = config.get("chains", {})
        if not self.chains_config:
            raise ValueError("chains configuration must be provided for AdvancedChainedProcessingStrategy")
        
        self.chain_on_missing_keys = config.get("chain_on_missing_keys", False)
        
        # Validate chain configuration
        self._validate_chain_config()

    def _validate_chain_config(self):
        """Validate the chain configuration structure."""
        if not isinstance(self.chains_config, dict):
            raise ValueError("chains must be a dictionary")
        
        for subchain_name, subchain_config in self.chains_config.items():
            if not isinstance(subchain_config, dict):
                raise ValueError(f"Subchain '{subchain_name}' must be a dictionary")
            
            required_keys = ["pre-processing", "processing", "post-processing"]
            for key in required_keys:
                if key not in subchain_config:
                    raise ValueError(f"Subchain '{subchain_name}' missing required key: {key}")
                
                if not isinstance(subchain_config[key], dict):
                    raise ValueError(f"Subchain '{subchain_name}' {key} must be a dictionary")
                
                if "type" not in subchain_config[key]:
                    raise ValueError(f"Subchain '{subchain_name}' {key} missing 'type' field")

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

        # Execute each subchain in sequence
        for subchain_name, subchain_config in self.chains_config.items():
            if not remaining_files:
                break

            logging.info(f"🔗 Executing subchain '{subchain_name}' on {len(remaining_files)} file(s)")
            
            # Process files through the three links of this subchain
            successful_results, failed_files = self._execute_subchain(subchain_name, subchain_config, remaining_files, 
                                                   config_manager, group_index, group_id, 
                                                   system_prompt, user_prompt, agg_stats, self.passthrough)
            
            # Store successful results
            for file_path, result in successful_results.items():
                if file_path not in per_file_result:
                    per_file_result[file_path] = result
            
            # Update remaining files for next subchain
            remaining_files = failed_files

        # Any file not finalized after all subchains => failure
        for file_path in file_group:
            if file_path not in per_file_result:
                per_file_result[file_path] = {"error": "All chained subchains exhausted without success"}
                logging.info(f"❌ Chain exhausted: {file_path}")

        merged_results = [(fp, per_file_result[fp]) for fp in file_group]
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
        
        # 1. Pre-processing link
        pre_config = subchain_config["pre-processing"]
        pre_type = pre_config["type"]
        
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
            
            # Store pre-processing results
            for file_path, result in pre_results:
                subchain_results[file_path] = {"pre_processing": result}
                
        except Exception as e:
            logging.error(f"❌ Pre-processing failed for subchain '{subchain_name}': {e}")
            # On pre-processing failure, mark all files as failed for this subchain
            for file_path in current_files:
                subchain_results[file_path] = {"error": f"Pre-processing failed: {e}"}
            return {}, current_files  # Return empty successful results, all files as failed
        
        # 2. Processing link
        processing_config = subchain_config["processing"]
        processing_type = processing_config["type"]
        
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
                user_prompt=user_prompt
            )
            
            agg_stats["estimated_tokens"] += processing_stats.get("estimated_tokens", 0)
            agg_stats["total_tokens"] += processing_stats.get("total_tokens", 0)
            agg_stats["processing_time"] += processing_stats.get("processing_time", 0)
            
            # Process results and check for success/failure
            for file_path, result in processing_results:
                # Cache model output into passthrough for post-processing comparisons
                try:
                    model_output = result.get("file_model_output", result)
                except AttributeError:
                    model_output = result
                try:
                    files_list = passthrough.setdefault("files", [])
                    matched_entry = None
                    for entry in files_list:
                        if entry.get("file_path") == file_path:
                            matched_entry = entry
                            break
                    if matched_entry is None:
                        matched_entry = {"file_path": file_path, "status": "Pending", "extracted_data": {}}
                        files_list.append(matched_entry)
                    matched_entry["processing_output"] = model_output
                except Exception:
                    # Non-fatal; continue normal flow
                    pass
                if "error" in result:
                    subchain_results[file_path]["processing"] = result
                    subchain_results[file_path]["error"] = result["error"]
                    failed_files.append(file_path)
                    logging.info(f"➡️ Processing failed for {file_path}: {result.get('error')}")
                else:
                    # Check mandatory keys if enabled
                    if self.chain_on_missing_keys:
                        model_output = result.get("file_model_output", result)
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
        post_type = post_config["type"]
        
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
        for file_path, results in subchain_results.items():
            if "error" not in results:
                successful_results[file_path] = results
        
        logging.info(f"🔁 Subchain '{subchain_name}' complete: successful={len(successful_results)}, failed={len(failed_files)}")
        
        return successful_results, failed_files




