"""
Strategy factory for creating processing strategies.
"""

from collections import Counter
import logging
from pathlib import Path
import re
import time
from typing import Dict, List, Any, Optional, Tuple

from .base_strategy import BaseProcessingStrategy


class ProcessingStrategyFactory:
    """Factory for creating processing strategies."""
    
    @staticmethod
    def create_strategy(strategy_type: str, config: Dict[str, Any], streaming: bool = False) -> BaseProcessingStrategy:
        """Create a processing strategy based on type."""
        print("================================CreatingStrategy================================", strategy_type)
        if strategy_type == "direct_file":
            from .direct_file_strategy import DirectFileProcessingStrategy
            return DirectFileProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "text_first":
            from .text_first_gemini import TextPreProcessingStrategy
            return TextPreProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "image_first":
            from .image_first_strategy import ImageFirstProcessingStrategy
            return ImageFirstProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "hybrid":
            from .hybrid_strategy import HybridProcessingStrategy
            return HybridProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "chain":
            from .chain_strategy import ChainedProcessingStrategy
            return ChainedProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "regex":
            from .regex_strategy import RegexProcessingStrategy
            return RegexProcessingStrategy(config, streaming=streaming)
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategy types."""
        return ["direct_file", "text_first", "image_first", "hybrid", "chain"]


class PreProcessingLinkFactory:
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
            from .text_first_gemini import TextPreProcessingStrategy
            strategy = TextPreProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "image":
            strategy = ImagePreProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "file":
            strategy = FilePreProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "regex":
            from .regex_strategy import RegexPreProcessingStrategy
            strategy = RegexPreProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "none":
            strategy = NoOpPreProcessingStrategy(config, streaming=streaming)
        else:
            raise ValueError(f"Unsupported pre-processing strategy type: {strategy_type}")

        if cls._passthrough is not None:
            _attach = getattr(strategy, "attach_passthrough", None)
            if callable(_attach):
                _attach(cls._passthrough)
        return strategy


class ProcessingLinkFactory:
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
            from .text_first_gemini import TextFirstProcessingStrategy
            strategy = TextFirstProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "image_first":
            from .image_first_strategy import ImageFirstProcessingStrategy
            strategy = ImageFirstProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "file_first":
            from .direct_file_strategy import DirectFileProcessingStrategy
            strategy = DirectFileProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "regex":
            from .regex_strategy import RegexProcessingStrategy
            strategy = RegexProcessingStrategy(config, streaming=streaming)
        else:
            raise ValueError(f"Unsupported processing strategy type: {strategy_type}")

        if cls._passthrough is not None:
            _attach = getattr(strategy, "attach_passthrough", None)
            if callable(_attach):
                _attach(cls._passthrough)
        return strategy

class PostProcessingLinkFactory:
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
        elif strategy_type == "regex":
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
