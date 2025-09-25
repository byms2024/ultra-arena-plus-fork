
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from pathlib import Path
from collections import Counter
import re

from common.text_extractor import TextExtractor
from .base_strategy import BaseProcessingStrategy
from .strategy_factory import ProcessingStrategyFactory
from .strategy_factory import PreProcessingStrategyFactory
from .strategy_factory import PostProcessingStrategyFactory
from llm_client.llm_client_factory import LLMClientFactory
from llm_metrics import TokenCounter
from .chain_strategy import LinkStrategy


class TextPreProcessingStrategy(LinkStrategy):
    """Pre-processing strategy for text-based operations."""

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

        file_text_cache: dict[Path, str] = {}

        vals = _collect_sensitive_values_from_text(text_content)
        
        file_text_cache[file_Name] = text_content

        for k, s in vals.items():
            aggregate_values[k].update(s)

        per_label_maps, reverse_map = _build_text_hash_maps(aggregate_values)

        hashed_text = _hash_text_with_maps(text_content, per_label_maps)

        try:
            import csv
            rev_path = "reverse_map.csv"
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
            
            # Extract file content
            text_content = self.extract_text(file_path)

            # If extracted, checks if it desensitization is needed
            if text_content:
                if self.desensitization_config:
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

class TextFirstProcessingStrategy(BaseProcessingStrategy):
    """Enhanced strategy for processing files with primary/secondary PDF extraction and regex validation."""
    
    def __init__(self, config: Dict[str, Any], streaming: bool = False, database_ops = None):
        super().__init__(config)
        self.streaming = streaming
        self.llm_provider = config.get("llm_provider", "ollama")
        self.llm_config = config.get("provider_configs", {}).get(self.llm_provider, {})
        self.llm_client = LLMClientFactory.create_client(self.llm_provider, self.llm_config, streaming=self.streaming)
        
        # Initialize token counter for accurate estimation
        self.token_counter = TokenCounter(self.llm_client, provider=self.llm_provider)
        
        # Text extraction configuration
        self.primary_extractor_lib = config.get("pdf_extractor_lib", "pymupdf")
        self.secondary_extractor_lib = config.get("secondary_pdf_extractor_lib", "pytesseract")
        self.regex_criteria = config.get("text_first_regex_criteria", {})
        
        # Initialize text extractors
        self.primary_extractor = TextExtractor(self.primary_extractor_lib)
        self.secondary_extractor = TextExtractor(self.secondary_extractor_lib)
        
        logging.info(f"🔧 Enhanced Text-First Strategy initialized:")
        logging.info(f"   Primary extractor: {self.primary_extractor_lib}")
        logging.info(f"   Secondary extractor: {self.secondary_extractor_lib}")
        logging.info(f"   Regex criteria keys: {list(self.regex_criteria.keys())}")
    
    def process_file_group(self, *, config_manager, file_group: List[str], group_index: int, 
                          group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str,
                          pre_results: List = None) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Process files with enhanced text-first approach."""
        
        text_to_process = [result['preprocessing_result']
                            for f_path, result in pre_results
                            if result['preprocessed']]
        
        successful_files = [f_path
                            for f_path, result in pre_results
                            if result['preprocessed']]
        
        original_filenames = [Path(f_path).name for f_path in successful_files]

        # Choose which prompt to use based on provider
        from config import config_base
        prompt_to_use = config_base.SIMPLIFIED_USER_PROMPT if self.llm_provider == "ollama" else user_prompt
        
        logging.info(f"📝 Using {'simplified' if self.llm_provider == 'ollama' else 'full'} prompt for {self.llm_provider} provider")
        
        # Process text contents directly using LLM with embedded content
        results, group_stats = self._process_text_contents_directly(
        text_contents=text_to_process,
            original_filenames=original_filenames,
            successful_files=successful_files,
            group_index=group_index,
            system_prompt=system_prompt,
            user_prompt=prompt_to_use
        )
        
        # Add error results for failed files
        failed_files = [f for f in file_group if f not in successful_files]
        for file_path in failed_files:
            results.append((file_path, {"error": "No text content could be extracted from PDF using any available method (PyMuPDF, PyTesseract OCR). This may be an image-based PDF with no embedded text."}))
            group_stats["failed_files"] += 1
        
        logging.info(f"✅ Completed text-first processing for group {group_index}: {group_stats['successful_files']} successful, {group_stats['failed_files']} failed")
        
        return results, group_stats, group_id
    
    def _process_text_contents_directly(self, *, text_contents: List[str], original_filenames: List[str], 
                                      successful_files: List[str], group_index: int, 
                                      system_prompt: Optional[str] = None, user_prompt: str) -> Tuple[List[Tuple[str, Dict]], Dict]:
        """Process text contents directly using LLM with embedded content structure."""
        
        # Create enhanced user prompt with embedded text content
        enhanced_user_prompt = self._create_enhanced_prompt_with_text_content(
            user_prompt=user_prompt,
            text_contents=text_contents,
            original_filenames=original_filenames
        )
        
        # Call LLM with enhanced prompt containing text content
        response = self._retry_with_backoff(
            self.llm_client.call_llm, files=None, system_prompt=system_prompt, user_prompt=enhanced_user_prompt
        )
        
        if "error" in response:
            logging.error(f"LLM API error for group {group_index}: {response['error']}")
            # Create failed results for all files
            results = [(file_path, {"error": response["error"]}) for file_path in successful_files]
            group_stats = {
                "total_files": len(successful_files),
                "successful_files": 0,
                "failed_files": len(successful_files),
                "total_tokens": 0,
                "estimated_tokens": 0
            }
            return results, group_stats
        
        # Parse response and match with files
        if isinstance(response, list):
            # Response is already a list of results
            file_results = response
        else:
            # Single result, wrap in list
            file_results = [response]
        
        # Map outputs to files using fuzzy matching
        results = []
        successful_count = 0
        failed_count = 0
        total_tokens = 0
        
        # Use the existing OpenAIFileMappingStrategy for fuzzy matching
        from processors.file_mapping_utils import FileMappingFactory
        
        # Create mapping strategy
        mapping_strategy = FileMappingFactory.create_strategy("openai")
        
        # Map results to files using fuzzy matching
        mapped_results = mapping_strategy.map_outputs_to_files(
            file_results=file_results,
            file_paths=successful_files,
            group_index=group_index
        )
        
        # Convert to the expected format
        for file_path, result in mapped_results:
            if "error" not in result:
                results.append((file_path, result))
                successful_count += 1
                if "total_token_count" in result:
                    total_tokens += result["total_token_count"]
            else:
                results.append((file_path, result))
                failed_count += 1
        
        group_stats = {
            "total_files": len(successful_files),
            "successful_files": successful_count,
            "failed_files": failed_count,
            "total_tokens": total_tokens,
            "estimated_tokens": 0
        }
        
        return results, group_stats
    
    def _create_enhanced_prompt_with_text_content(self, *, user_prompt: str, text_contents: List[str], 
                                                original_filenames: List[str]) -> str:
        """Create enhanced user prompt with embedded text content and filename markers."""
        
        from llm_client.client_utils import _create_text_first_prompt

        # Start with the enhanced prompt instructions
        enhanced_prompt = _create_text_first_prompt(user_prompt)
        
        # Add text contents with embedded filename markers
        for text_content, original_filename in zip(text_contents, original_filenames):
            enhanced_prompt += f"\n\n=== FILE: {original_filename} ===\n"
            enhanced_prompt += text_content
            enhanced_prompt += f"\n=== END FILE: {original_filename} ==="
        
        return enhanced_prompt
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff."""
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                delay = base_delay * (2 ** attempt)
                logging.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                time.sleep(delay) 
