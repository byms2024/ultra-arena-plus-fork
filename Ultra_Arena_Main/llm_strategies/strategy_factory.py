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

# Pre-processing strategies
class TextPreProcessingStrategy(LinkStrategy):
    """Pre-processing strategy for text-based operations."""

    def __init__(self, config: Dict[str, Any], streaming: bool = False):
        super().__init__(config, streaming)
        # Allow providing regex criteria via config; default to empty dict
        self.regex_criteria = config.get("text_first_regex_criteria", config.get("regex_criteria", {})) or {}
    
    def desensitize_content(self, text_content: str, file_Name: str) -> str:
        from .data_sensitization import _collect_sensitive_values_from_text, _build_text_hash_maps, _hash_text_with_maps
        
        aggregate_values: dict[str, set[str]] = {
            "CNPJ": set(),
            "CPF": set(),
            "CEP": set(),
            "VIN": set(),
            "CLAIM": set(),
            "NAME": set(),
            "ADDRESS": set(),
            "PHONE": set(),
            "ORG": set(),
            "PLATE": set(),
        }

        file_text_cache: dict[str, str] = {}

        vals = _collect_sensitive_values_from_text(text_content)
        
        file_text_cache[file_Name] = text_content

        for k, s in vals.items():
            aggregate_values[k].update(s)

        per_label_maps, reverse_map = _build_text_hash_maps(aggregate_values)

        hashed_text = _hash_text_with_maps(text_content, per_label_maps)

        try:
            import csv
            rev_path = Path("reverse_map.csv")
            with rev_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["placeholder", "original"])
                for placeholder, original in reverse_map.items():
                    writer.writerow([placeholder, original])
        except Exception:
            pass
        
        return hashed_text

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
        from ..common.text_extractor import TextExtractor

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
        processed_texts = []
        original_filenames = []
        successful_files = []
        results = []
        
        for file_path in file_group:
            # Optionally read and store PDF metadata (including DmsData)
            from ..common.pdf_metadata import read_pdf_metadata_dict

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

            
            # Extract file content
            text_content = self.extract_text(file_path)

            # If extracted, checks if it desensitization is needed
            if text_content:
                if getattr(self, "desensitization_config", None):
                    text_to_add = self.desensitize_content(text_content, Path(file_path).name)
                else:
                    text_to_add = text_content
                
                # Store results
                processed_texts.append(text_to_add)
                original_filenames.append(Path(file_path).name)
                successful_files.append(file_path)
                result = {  "preprocessed": True,
                            "preprocessing_type": "text",
                            "preprocessing_result" : text_to_add}
                results.append((file_path, result))
            
            # If extraction failed, store results
            else:
                result = {  "preprocessed": False,
                            "preprocessing_type": "text",
                            "preprocessing_result":"No text content could be extracted from PDF using any available method (PyMuPDF, PyTesseract OCR). This may be an image-based PDF with no embedded text."}
                results.append((file_path, result))

        preprocessed_values = (result_dict.get('preprocessed') for _, result_dict in results)
        counts = Counter(preprocessed_values)

        agg_stats = {
            "total_files": len(file_group),
            "successful_files": counts.get(True, 0),
            "failed_files": counts.get(False, 0),
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
                # Find corresponding entry (absolute match, then basename match)
                for entry in self.passthrough.get("files", []):
                    matched = False
                    if entry.get("file_path") == file_path:
                        matched = True
                    else:
                        try:
                            from pathlib import Path as _P
                            matched = _P(str(entry.get("file_path"))).name == _P(str(file_path)).name
                        except Exception:
                            matched = False
                    if not matched:
                        continue
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
                            proc_norm = entry.get("processing_extracted_data", {}) or {}
                            matched = True
                        else:
                            matched = False
                        if not matched:
                            try:
                                from pathlib import Path as _P
                                if _P(str(entry.get("file_path"))).name == _P(str(file_path)).name:
                                    proc_raw = entry.get("processing_output", {}) or {}
                                    dms = entry.get("extracted_data", {}) or {}
                                    proc_norm = entry.get("processing_extracted_data", {}) or {}
                                    matched = True
                            except Exception:
                                matched = False
                        if not matched:
                            continue

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
                            "type": pick(proc_norm, ["type"]) or pick(proc_raw, ["type", "DOC_TYPE", "document_type", "class"]),
                            "cnpj": pick(proc_norm, ["cnpj"]) or pick(proc_raw, ["cnpj", "CNPJ", "collected_CNPJ"]),
                            "cnpj2": pick(proc_norm, ["cnpj2"]) or pick(proc_raw, ["cnpj2", "CNPJ2", "collected_CNPJ2"]),
                            "vin": pick(proc_norm, ["vin"]) or pick(proc_raw, ["vin", "VIN", "collected_VIN"]),
                            "claim_no": pick(proc_norm, ["claim_no"]) or pick(proc_raw, ["claim_no", "CLAIM_NO", "collected_ClaimNO"]),
                            "parts_value": pick(proc_norm, ["parts_value"]) or pick(proc_raw, ["parts_value", "PARTS_VALUE", "PART_AMOUNT", "part_amount", "collected_parts_price"]),
                            "service_value": pick(proc_norm, ["service_value"]) or pick(proc_raw, ["service_value", "SERVICE_VALUE", "LABOUR_AMOUNT", "labour_amount", "collected_service_price"]),
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
                        if dms.get("type") is "Peças":
                            if proc_fields.get("parts_value") is None or str(dms.get("part_amount_dms")).strip().replace(",", "").replace(".", "") not in str(proc_fields["parts_value"]).strip().replace(",", "").replace(".", ""):
                                issues.append("part_amount_dms")
                        if dms.get("type") is "Serviço":
                            if proc_fields.get("service_value") is None or str(dms.get("labour_amount_dms")).strip().replace(",", "").replace(".", "") not in str(proc_fields["service_value"]).strip().replace(",", "").replace(".", ""):
                                issues.append("labour_amount_dms")

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
