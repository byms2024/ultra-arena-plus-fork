"""
Enhanced Text-first processing strategy with primary/secondary PDF extraction and regex validation.
"""

import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .base_strategy import BaseProcessingStrategy
from llm_client.llm_client_factory import LLMClientFactory
from llm_metrics import TokenCounter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processors.file_mapping_utils import FileMappingFactory, create_text_first_file_path_mapper, FilePathAwareLLMClient
from common.text_extractor import TextExtractor
from llm_client.client_utils import _create_text_first_prompt


class EnhancedTextFirstProcessingStrategy(BaseProcessingStrategy):
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
        
        if not pre_results:
            desensitization_config = config_manager._server_config.desensitization.desensitization_config
            sensitive_fields =config_manager._server_config.desensitization.sensitive_fields

            group_start_time = time.time()
            logging.info(f"📝 Starting enhanced text-first processing for group {group_index} ({group_id}): {len(file_group)} files")
            logging.info(f"Desestization Config: {desensitization_config}")
            

            # Extract text from PDFs with fallback and maintain mapping
            text_content_list = []
            text_content_sensitized_list = []
            original_filenames = []
            successful_files = []
            
            for file_path in file_group:
                text_content = self._extract_text_with_fallback(file_path)

                if text_content:

                    if desensitization_config:
                        text_content_sensitized = self._sensitize_text_content(text_content, Path(file_path).name)
                        text_content_sensitized_list.append(text_content_sensitized)
                        logging.info("🛡 Text Content Desensitezed 🛡")
                        logging.info(text_content_sensitized)

                    else:
                        text_content_list.append(text_content)

                    original_filenames.append(Path(file_path).name)
                    successful_files.append(file_path)
                    logging.info(f"✅ Successfully extracted text from: {Path(file_path).name} ({len(text_content)} characters)")
                else:
                    logging.error(f"❌ No text could be extracted from: {Path(file_path).name} using any available method")
            

            if (not text_content_list) and (not text_content_sensitized_list):
                logging.error(f"❌ No text could be extracted for group {group_index}")
                # Create error results for all files
                results = [(file_path, {"error": "No text content could be extracted from PDF using any available method (PyMuPDF, PyTesseract OCR). This may be an image-based PDF with no embedded text."}) for file_path in file_group]
                group_stats = {
                    "total_files": len(file_group),
                    "successful_files": 0,
                    "failed_files": len(file_group),
                    "total_tokens": 0,
                    "estimated_tokens": 0,
                    "processing_time": int(time.time() - group_start_time)
                }
                return results, group_stats, group_id
            
            if desensitization_config:
                text_to_process = text_content_sensitized_list
            else: 
                text_to_process = text_content_list

        else:
            # Get the already successfully preprocessed files
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
        
        group_stats["processing_time"] = int(time.time() - group_start_time)
        logging.info(f"✅ Completed enhanced text-first processing for group {group_index}: {group_stats['successful_files']} successful, {group_stats['failed_files']} failed")
        
        return results, group_stats, group_id
    
    def _extract_text_with_fallback(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF with primary/secondary extractor fallback and regex validation."""
        extraction_start_time = time.time()
        logging.info(f"🔄 Extracting text from PDF: {Path(pdf_path).name}")
        
        # Step 1: Try primary extractor
        primary_start_time = time.time()
        primary_text = self._extract_text_with_extractor(pdf_path, self.primary_extractor, "primary")
        primary_score = self._evaluate_text_with_regex(primary_text, "primary")
        primary_time = time.time() - primary_start_time
        
        logging.info(f"📊 Primary extractor ({self.primary_extractor_lib}) score: {primary_score}/{len(self.regex_criteria)} in {primary_time:.2f}s")
        
        # Step 2: Determine if we should try secondary extractor
        # Try secondary extractor if:
        # 1. Primary extractor returned no text, OR
        # 2. Primary extractor returned text but regex score is low (and we have regex criteria), OR
        # 3. Secondary extractor is different from primary
        total_keys = len(self.regex_criteria)
        secondary_time = 0.0
        # should_try_secondary = (
        #     (not primary_text or (total_keys > 0 and primary_score < total_keys)) and 
        #     self.secondary_extractor_lib != self.primary_extractor_lib
        # )
        should_try_secondary = (
            ((not primary_text) or (len(primary_text) < 1000)) and 
            self.secondary_extractor_lib != self.primary_extractor_lib
        )
        
        if should_try_secondary:
            secondary_start_time = time.time()
            secondary_text = self._extract_text_with_extractor(pdf_path, self.secondary_extractor, "secondary")
            secondary_score = self._evaluate_text_with_regex(secondary_text, "secondary")
            secondary_time = time.time() - secondary_start_time
            
            logging.info(f"📊 Secondary extractor ({self.secondary_extractor_lib}) score: {secondary_score}/{len(self.regex_criteria)} in {secondary_time:.2f}s")
            
            # Choose the extractor with the highest score, or the one that extracted text if scores are equal
            if secondary_score > primary_score:
                logging.info(f"✅ Secondary extractor chosen (score: {secondary_score} > {primary_score})")
                chosen_extractor = "secondary"
                chosen_text = secondary_text
            elif secondary_score == primary_score:
                # If scores are equal, prefer the one that extracted text
                if secondary_text and not primary_text:
                    logging.info(f"✅ Secondary extractor chosen (extracted text vs no text)")
                    chosen_extractor = "secondary"
                    chosen_text = secondary_text
                elif primary_text and not secondary_text:
                    logging.info(f"✅ Primary extractor chosen (extracted text vs no text)")
                    chosen_extractor = "primary"
                    chosen_text = primary_text
                else:
                    # Both extracted text or both didn't, prefer primary
                    logging.info(f"✅ Primary extractor chosen (equal scores, preferring primary)")
                    chosen_extractor = "primary"
                    chosen_text = primary_text
            else:
                logging.info(f"✅ Primary extractor chosen (score: {primary_score} > {secondary_score})")
                chosen_extractor = "primary"
                chosen_text = primary_text
        else:
            logging.info(f"✅ Using primary extractor only (no secondary needed)")
            chosen_extractor = "primary"
            chosen_text = primary_text
        
        # Calculate total extraction time
        total_extraction_time = time.time() - extraction_start_time
        
        # Log performance summary
        logging.info(f"⏱️ Text extraction performance summary:")
        logging.info(f"   Primary extraction: {primary_time:.2f}s")
        logging.info(f"   Secondary extraction: {secondary_time:.2f}s")
        logging.info(f"   Total extraction time: {total_extraction_time:.2f}s")
        logging.info(f"   Chosen extractor: {chosen_extractor}")
        logging.info(f"   Final text length: {len(chosen_text) if chosen_text else 0} characters")
        
        return chosen_text
    
    def _extract_text_with_extractor(self, pdf_path: str, extractor: TextExtractor, extractor_type: str) -> str:
        """Extract text using specified extractor."""
        extractor_start_time = time.time()
        try:
            logging.info(f"🔍 Extracting text with {extractor_type} extractor ({extractor.extractor_lib})")
            text_content = extractor.extract_text(pdf_path, max_length=50000)
            extractor_time = time.time() - extractor_start_time
            
            if text_content:
                logging.info(f"✅ {extractor_type.capitalize()} extraction successful: {len(text_content)} characters in {extractor_time:.2f}s")
                return text_content
            else:
                logging.warning(f"⚠️ {extractor_type.capitalize()} extraction returned empty text in {extractor_time:.2f}s")
                return ""
                
        except Exception as e:
            extractor_time = time.time() - extractor_start_time
            logging.error(f"❌ {extractor_type.capitalize()} extraction failed in {extractor_time:.2f}s: {e}")
            return ""
    
    def _evaluate_text_with_regex(self, text_content: str, extractor_type: str) -> int:
        """Evaluate text content using regex criteria and return number of successful matches."""
        regex_start_time = time.time()
        
        if not text_content:
            logging.warning(f"⚠️ Cannot evaluate empty text from {extractor_type} extractor")
            return 0
        
        successful_matches = 0
        match_details = []
        
        for field_name, regex_pattern in self.regex_criteria.items():
            try:
                matches = re.findall(regex_pattern, text_content)
                if matches:
                    successful_matches += 1
                    match_details.append(f"{field_name}: {len(matches)} match(es)")
                    logging.debug(f"🎯 {extractor_type.capitalize()} - {field_name}: found {len(matches)} match(es)")
                else:
                    logging.debug(f"❌ {extractor_type.capitalize()} - {field_name}: no matches")
            except Exception as e:
                logging.error(f"❌ Regex evaluation failed for {field_name}: {e}")
        
        regex_time = time.time() - regex_start_time
        
        if match_details:
            logging.info(f"📋 {extractor_type.capitalize()} extractor matches: {', '.join(match_details)} in {regex_time:.3f}s")
        
        return successful_matches
    
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
    
    def _sensitize_text_content(self, text_content: str, file_Name: str) -> str:
        from llm_strategies.data_sensitization import _collect_sensitive_values_from_text, _build_text_hash_maps, _hash_text_with_maps
        
        #SENSITIVE_INFO = [ "CNPJ", "CPF", "VIN","CEP","CLAIM","NAME","ADDRESS"]
        
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