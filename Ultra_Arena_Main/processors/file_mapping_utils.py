"""
File mapping utilities for different LLM providers.

This module provides provider-specific file mapping logic to handle different
ways that LLM providers return file_name_llm in their responses.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
from fuzzywuzzy import fuzz


def is_filename_substring_match(filename1: str, filename2: str) -> bool:
    """
    Check if one filename is a substring of another filename.
    
    Args:
        filename1: First filename (without extension)
        filename2: Second filename (without extension)
        
    Returns:
        True if one filename is a substring of the other, False otherwise
    """
    # Remove file extensions for comparison
    name1 = Path(filename1).stem.lower()
    name2 = Path(filename2).stem.lower()
    
    # Check if one is a substring of the other
    return name1 in name2 or name2 in name1


class FileMappingStrategy(ABC):
    """Abstract base class for file mapping strategies."""
    
    @abstractmethod
    def map_outputs_to_files(
        self, 
        file_results: List[Dict[str, Any]], 
        file_paths: List[str],
        group_index: int = 0
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Map LLM outputs to files using provider-specific logic.
        
        Args:
            file_results: List of results from LLM
            file_paths: List of original file paths
            group_index: Group index for logging
            
        Returns:
            List of (file_path, result) tuples
        """
        pass


class GenericFileMappingStrategy(FileMappingStrategy):
    """Generic file mapping strategy that works with most providers."""
    def map_outputs_to_files(
        self, 
        file_results: List[Dict[str, Any]], 
        file_paths: List[str],
        group_index: int = 0
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Map outputs to files using fuzzy matching logic.
        - Try exact match on filename
        - Try fuzzy match with 85% similarity threshold
        - Fallback to first available result if no match found
        """
        logging.info(f"🔗 Mapping outputs to files for group {group_index}...")
        file_name_llm_to_output = {}
        for i, file_result in enumerate(file_results):
            if isinstance(file_result, dict) and "file_name_llm" in file_result:
                file_name_llm_to_output[file_result["file_name_llm"]] = file_result
        
        results = []
        for file_path in file_paths:
            filename = Path(file_path).name
            file_result = file_name_llm_to_output.get(filename)
            
            if file_result is None:
                # Try substring matching as fallback
                best_match = None
                for llm_name, output in file_name_llm_to_output.items():
                    if is_filename_substring_match(filename, llm_name):
                        best_match = output
                        logging.info(f"🔍 Found substring match: '{filename}' matches '{llm_name}'")
                        break
                
                if best_match:
                    file_result = best_match
            
            if file_result is not None:
                file_result["file_name_llm"] = filename
                results.append((file_path, file_result))
            else:
                error_result = {"error": "No result returned for this file"}
                results.append((file_path, error_result))
        return results


class GoogleGeminiFileMappingStrategy(FileMappingStrategy):
    """File mapping strategy specifically for Google Gemini models."""
    
    def map_outputs_to_files(
        self, 
        file_results: List[Dict[str, Any]], 
        file_paths: List[str],
        group_index: int = 0
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Map outputs to files using Google Gemini-specific logic with fuzzy matching.
        
        Gemini models often return full paths in file_name_llm, so this strategy:
        1. Tries exact filename match
        2. Tries fuzzy matching with 85% similarity threshold
        3. Extracts filename from full path if llm_name contains path separators
        4. Falls back to first available result
        """
        logging.info(f"🔗 Mapping outputs to files for group {group_index} (Google Gemini)...")
        
        # Create mapping from file_name_llm to output
        file_name_llm_to_output = {}
        logging.info(f"🔍 Debug: Processing {len(file_results)} file results")
        for i, file_result in enumerate(file_results):
            logging.info(f"🔍 Debug: File result {i}: {type(file_result)} - {file_result}")
            if isinstance(file_result, dict) and "file_name_llm" in file_result:
                file_name_llm_to_output[file_result["file_name_llm"]] = file_result
                logging.info(f"🔍 Debug: Added mapping for {file_result['file_name_llm']}")
            else:
                logging.info(f"🔍 Debug: Skipped result {i} - not a dict or missing file_name_llm")
        
        logging.info(f"🔍 Debug: Mapped {len(file_name_llm_to_output)} outputs with file_name_llm")
        logging.info(f"🔍 Debug: Available file_name_llm keys: {list(file_name_llm_to_output.keys())}")
        
        results = []
        
        # Match each file to its corresponding output
        for file_path in file_paths:
            filename = Path(file_path).name
            logging.info(f"🔍 Debug: Looking for filename '{filename}' in mapped outputs")
            
            # Try exact match first
            file_result = file_name_llm_to_output.get(filename)
            
            if file_result is None:
                logging.warning(f"⚠️ No exact match found for filename: {filename}")
                # Try substring matching as fallback
                best_match = None
                logging.info(f"🔍 Attempting substring matching for '{filename}' against {len(file_name_llm_to_output)} available results")
                
                for llm_name, output in file_name_llm_to_output.items():
                    if is_filename_substring_match(filename, llm_name):
                        best_match = output
                        logging.info(f"🔍 Found substring match: '{filename}' matches '{llm_name}'")
                        break
                
                if best_match:
                    file_result = best_match
                else:
                    logging.warning(f"⚠️ No substring match found for '{filename}'")
            
            # Additional fallback: try matching by extracting filename from full path in llm_name
            if file_result is None:
                for llm_name, output in file_name_llm_to_output.items():
                    # Extract filename from full path if llm_name contains path separators
                    if '/' in llm_name or '\\' in llm_name:
                        llm_filename = Path(llm_name).name
                        if llm_filename == filename:
                            file_result = output
                            logging.info(f"✅ Found path-based match: '{llm_filename}' from '{llm_name}' matches '{filename}'")
                            break
            
            if file_result is not None:
                # Add file name to result for identification
                file_result["file_name_llm"] = filename
                results.append((file_path, file_result))
            else:
                logging.error(f"❌ No match found for file: {filename}")
                error_result = {"error": "No result returned for this file"}
                results.append((file_path, error_result))
        
        return results


class OpenAIFileMappingStrategy(FileMappingStrategy):
    """File mapping strategy specifically for OpenAI models."""
    
    def map_outputs_to_files(
        self, 
        file_results: List[Dict[str, Any]], 
        file_paths: List[str],
        group_index: int = 0
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Map outputs to files using OpenAI-specific logic with fuzzy matching.
        
        OpenAI models typically return just filenames in file_name_llm, so this strategy:
        1. Tries exact filename match
        2. Tries fuzzy matching with 85% similarity threshold
        3. Falls back to first available result
        """
        logging.info(f"🔗 Mapping outputs to files for group {group_index} (OpenAI)...")
        
        # Create mapping from file_name_llm to output
        file_name_llm_to_output = {}
        logging.info(f"🔍 Debug: Processing {len(file_results)} file results")
        for i, file_result in enumerate(file_results):
            logging.info(f"🔍 Debug: File result {i}: {type(file_result)} - {file_result}")
            if isinstance(file_result, dict) and "file_name_llm" in file_result:
                file_name_llm_to_output[file_result["file_name_llm"]] = file_result
                logging.info(f"🔍 Debug: Added mapping for {file_result['file_name_llm']}")
            else:
                logging.info(f"🔍 Debug: Skipped result {i} - not a dict or missing file_name_llm")
        
        logging.info(f"🔍 Debug: Mapped {len(file_name_llm_to_output)} outputs with file_name_llm")
        logging.info(f"🔍 Debug: Available file_name_llm keys: {list(file_name_llm_to_output.keys())}")
        
        results = []
        
        # Match each file to its corresponding output
        for file_path in file_paths:
            filename = Path(file_path).name
            logging.info(f"🔍 Debug: Looking for filename '{filename}' in mapped outputs")
            
            # Try exact match first
            file_result = file_name_llm_to_output.get(filename)
            
            if file_result is None:
                logging.warning(f"⚠️ No exact match found for filename: {filename}")
                # Try substring matching as fallback
                best_match = None
                logging.info(f"🔍 Attempting substring matching for '{filename}' against {len(file_name_llm_to_output)} available results")
                
                for llm_name, output in file_name_llm_to_output.items():
                    if is_filename_substring_match(filename, llm_name):
                        best_match = output
                        logging.info(f"🔍 Found substring match: '{filename}' matches '{llm_name}'")
                        break
                
                if best_match:
                    file_result = best_match
                else:
                    logging.warning(f"⚠️ No substring match found for '{filename}'")
            
            if file_result is not None:
                # Add file name to result for identification
                file_result["file_name_llm"] = filename
                results.append((file_path, file_result))
            else:
                logging.error(f"❌ No match found for file: {filename}")
                error_result = {"error": "No result returned for this file"}
                results.append((file_path, error_result))
        
        return results


class FileMappingFactory:
    """Factory for creating file mapping strategies based on LLM provider."""
    
    @staticmethod
    def create_strategy(provider: str) -> FileMappingStrategy:
        """
        Create a file mapping strategy based on the LLM provider.
        
        Args:
            provider: LLM provider name (e.g., 'google', 'openai', 'claude')
            
        Returns:
            FileMappingStrategy instance
        """
        provider_lower = provider.lower()
        
        if 'google' in provider_lower or 'gemini' in provider_lower:
            return GoogleGeminiFileMappingStrategy()
        elif 'openai' in provider_lower or 'gpt' in provider_lower:
            return OpenAIFileMappingStrategy()
        elif 'claude' in provider_lower:
            # Claude uses similar file mapping logic to OpenAI
            return OpenAIFileMappingStrategy()
        else:
            # Default to generic strategy for unknown providers
            return GenericFileMappingStrategy()
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategy names."""
        return ['generic', 'google_gemini', 'openai']


# Convenience functions for backward compatibility
def map_outputs_to_files_generic(
    file_results: List[Dict[str, Any]], 
    file_paths: List[str],
    group_index: int = 0
) -> List[Tuple[str, Dict[str, Any]]]:
    """Generic file mapping function for backward compatibility."""
    strategy = GenericFileMappingStrategy()
    return strategy.map_outputs_to_files(file_results, file_paths, group_index)


def map_outputs_to_files_by_provider(
    file_results: List[Dict[str, Any]], 
    file_paths: List[str],
    provider: str,
    group_index: int = 0
) -> List[Tuple[str, Dict[str, Any]]]:
    """Map outputs to files using provider-specific strategy."""
    strategy = FileMappingFactory.create_strategy(provider)
    return strategy.map_outputs_to_files(file_results, file_paths, group_index)


class GenericFilePathMapper:
    """Generic utility class for handling file path mapping in image_first and text_first strategies."""
    
    def __init__(self):
        # Clear and concise naming for the two-directional maps
        self.original_to_converted_file_path_map = {}  # original_path -> converted_path
        self.converted_to_original_file_path_map = {}  # converted_path -> original_path
    
    def add_mapping(self, original_path: str, converted_path: str):
        """Add a mapping between original and converted file paths."""
        self.original_to_converted_file_path_map[original_path] = converted_path
        self.converted_to_original_file_path_map[converted_path] = original_path
        logging.info(f"🔗 Added mapping: {Path(original_path).name} -> {Path(converted_path).name}")
        logging.info(f"🔗 Stored as: original_to_converted[{Path(original_path).name}] = {Path(converted_path).name}")
        logging.info(f"🔗 Stored as: converted_to_original[{Path(converted_path).name}] = {Path(original_path).name}")
    
    def get_original_path(self, converted_path: str) -> Optional[str]:
        """Get the original file path from a converted file path using fuzzy logic."""
        # First try exact match
        original_path = self.converted_to_original_file_path_map.get(converted_path)
        if original_path:
            return original_path
        
        # Try fuzzy matching if exact match fails
        best_match = None
        best_score = 0
        converted_filename = Path(converted_path).name
        
        for mapped_converted_path, mapped_original_path in self.converted_to_original_file_path_map.items():
            mapped_converted_filename = Path(mapped_converted_path).name
            score = fuzz.ratio(converted_filename.lower(), mapped_converted_filename.lower())
            if score > best_score and score >= 85:  # 85% similarity threshold
                best_score = score
                best_match = mapped_original_path
                logging.info(f"✅ Found fuzzy match: '{converted_filename}' matches '{mapped_converted_filename}' with score {score}")
        
        if best_match:
            return best_match
        
        return None
    
    def get_converted_path(self, original_path: str) -> Optional[str]:
        """Get the converted file path from an original file path."""
        return self.original_to_converted_file_path_map.get(original_path)
    
    def get_converted_file_group(self, original_file_group: List[str]) -> List[str]:
        """Convert a group of original file paths to converted file paths."""
        converted_group = []
        for original_path in original_file_group:
            converted_path = self.get_converted_path(original_path)
            if converted_path:
                converted_group.append(converted_path)
            else:
                logging.warning(f"⚠️ No converted path found for: {original_path}")
        return converted_group
    
    def map_results_to_original_files(self, results: List[Tuple[str, Dict]], 
                                    original_file_group: List[str]) -> List[Tuple[str, Dict]]:
        """
        Map results from converted files back to original files using fuzzy logic.
        
        Args:
            results: List of (converted_file_path, result) tuples from LLM
            original_file_group: List of original file paths
            
        Returns:
            List of (original_file_path, result) tuples
        """
        mapped_results = []
        
        # Create a mapping from original filename to original path
        original_filename_to_path = {Path(path).name: path for path in original_file_group}
        
        # Process each result
        for converted_path, result in results:
            if isinstance(result, dict) and "file_name_llm" in result:
                # The LLM returned the original PDF filename
                original_filename = result["file_name_llm"]
                original_path = original_filename_to_path.get(original_filename)
                
                if original_path:
                    mapped_results.append((original_path, result))
                    logging.debug(f"✅ Mapped result: {original_filename} -> {Path(original_path).name}")
                else:
                    logging.error(f"❌ No original path found for filename: {original_filename}")
            else:
                # Fallback: try to map using converted path with fuzzy logic
                original_path = self.get_original_path(converted_path)
                if original_path:
                    # Update file_name_llm to use original filename
                    if isinstance(result, dict) and "file_name_llm" in result:
                        result["file_name_llm"] = Path(original_path).name
                    mapped_results.append((original_path, result))
                    logging.debug(f"✅ Mapped result: {Path(converted_path).name} -> {Path(original_path).name}")
                else:
                    logging.error(f"❌ No original path found for converted file: {converted_path}")
        
        return mapped_results


class ImageFirstFilePathMapper(GenericFilePathMapper):
    """Specialized file path mapper for image first strategy."""
    
    def add_mapping(self, pdf_path: str, image_path: str):
        """Add a mapping from PDF path to image path for image first strategy."""
        super().add_mapping(pdf_path, image_path)
    
    def map_results_to_original_files(self, results: List[Tuple[str, Dict]], 
                                    original_file_group: List[str]) -> List[Tuple[str, Dict]]:
        """
        Map results from image files back to original PDF files using fuzzy logic.
        For image first strategy, the LLM returns the image filename in file_name_llm.
        We need to map from image filename to PDF path using our mapping.
        """
        logging.info(f"🔍 ImageFirstFilePathMapper.map_results_to_original_files called with {len(results)} results")
        logging.info(f"🔍 Original file group: {original_file_group}")
        logging.info(f"🔍 Available mappings (converted_to_original): {list(self.converted_to_original_file_path_map.keys())}")
        logging.info(f"🔍 Available mappings (original_to_converted): {list(self.original_to_converted_file_path_map.keys())}")
        mapped_results = []
        
        # Process each result
        for image_path, result in results:
            if isinstance(result, dict) and "file_name_llm" in result:
                # The LLM returned the image filename
                image_filename = result["file_name_llm"]
                
                # Map from image path to PDF path using our mapping with fuzzy logic
                logging.info(f"🔍 Looking for mapping: {image_path}")
                logging.info(f"🔍 Available mappings: {list(self.converted_to_original_file_path_map.keys())}")
                
                # Try to find the mapping using the image filename from LLM response
                pdf_path = self.get_original_path(image_filename)
                logging.info(f"🔍 Looking for image filename: {image_filename}")
                logging.info(f"🔍 Found PDF path: {pdf_path}")
                
                # If not found by filename, try with full path
                if not pdf_path:
                    pdf_path = self.get_original_path(image_path)
                    logging.info(f"🔍 Trying full path, found PDF path: {pdf_path}")
                
                if pdf_path:
                    mapped_results.append((pdf_path, result))
                    logging.debug(f"✅ Mapped image result: {image_filename} -> {Path(pdf_path).name}")
                else:
                    logging.error(f"❌ No PDF path found for image: {image_filename}")
                    mapped_results.append((image_path, result))
            else:
                # Fallback: try to map using image path with fuzzy logic
                pdf_path = self.get_original_path(image_path)
                if pdf_path:
                    mapped_results.append((pdf_path, result))
                    logging.debug(f"✅ Fallback mapping: {Path(image_path).name} -> {Path(pdf_path).name}")
                else:
                    mapped_results.append((image_path, result))
        
        return mapped_results


class TextFirstFilePathMapper(GenericFilePathMapper):
    """Specialized file path mapper for text first strategy."""
    
    def add_mapping(self, pdf_path: str, text_path: str):
        """Add a mapping from PDF path to text path for text first strategy."""
        super().add_mapping(pdf_path, text_path)
    
    def map_results_to_original_files(self, results: List[Tuple[str, Dict]], 
                                    original_file_group: List[str]) -> List[Tuple[str, Dict]]:
        """
        Map results from text files back to original PDF files using fuzzy logic.
        For text first strategy, the LLM returns the text filename in file_name_llm.
        We need to map from text filename to PDF path using our mapping.
        """
        mapped_results = []
        
        # Process each result
        for text_path, result in results:
            if isinstance(result, dict) and "file_name_llm" in result:
                # The LLM returned the text filename
                text_filename = result["file_name_llm"]
                
                # Map from text path to PDF path using our mapping with fuzzy logic
                pdf_path = self.get_original_path(text_path)
                
                if pdf_path:
                    mapped_results.append((pdf_path, result))
                    logging.debug(f"✅ Mapped text result: {text_filename} -> {Path(pdf_path).name}")
                else:
                    logging.error(f"❌ No PDF path found for text file: {text_filename}")
                    mapped_results.append((text_path, result))
            else:
                # Fallback: try to map using text path with fuzzy logic
                pdf_path = self.get_original_path(text_path)
                if pdf_path:
                    mapped_results.append((pdf_path, result))
                    logging.debug(f"✅ Fallback mapping: {Path(text_path).name} -> {Path(pdf_path).name}")
                else:
                    mapped_results.append((text_path, result))
        
        return mapped_results


def create_file_path_mapper() -> GenericFilePathMapper:
    """Create a new GenericFilePathMapper instance."""
    return GenericFilePathMapper()


def create_image_first_file_path_mapper() -> ImageFirstFilePathMapper:
    """Create a new ImageFirstFilePathMapper instance."""
    return ImageFirstFilePathMapper()


def create_text_first_file_path_mapper() -> TextFirstFilePathMapper:
    """Create a new TextFirstFilePathMapper instance."""
    return TextFirstFilePathMapper()


class FilePathAwareLLMClient():
    """Wrapper for LLM clients that handles FILE_PATH mapping for image_first and text_first strategies."""
    
    def __init__(self, llm_client, file_path_mapper: GenericFilePathMapper):
        self.llm_client = llm_client
        self.file_path_mapper = file_path_mapper
    
    def call_llm(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str, 
                 strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Call LLM with FILE_PATH mapping for processed files.
        
        This method intercepts the file paths and replaces them with the original PDF paths
        in the FILE_PATH information sent to the LLM, while still sending the processed files.
        """
        if not files:
            return self.llm_client.call_llm(files=files, system_prompt=system_prompt, user_prompt=user_prompt, strategy_type=strategy_type)
        
        # For Google GenAI client, we need to modify the FILE_PATH information
        if hasattr(self.llm_client, '_modify_file_paths_for_llm'):
            return self.llm_client._modify_file_paths_for_llm(
                files=files, 
                system_prompt=system_prompt, 
                user_prompt=user_prompt,
                file_path_mapper=self.file_path_mapper,
                strategy_type=strategy_type
            )
        else:
            # For other clients (like Claude), we need to modify the user prompt
            # to include the original PDF paths in FILE_PATH information
            modified_user_prompt = user_prompt
            
            # Add FILE_PATH mapping information to the user prompt
            file_path_mappings = []
            for file_path in files:
                original_pdf_path = self.file_path_mapper.get_original_path(file_path)
                if original_pdf_path:
                    file_path_mappings.append(f"FILE_PATH: {original_pdf_path}")
                    logging.debug(f"🔗 Added FILE_PATH mapping: {os.path.basename(file_path)} -> {os.path.basename(original_pdf_path)}")
                else:
                    file_path_mappings.append(f"FILE_PATH: {file_path}")
            
            # Add the FILE_PATH information to the user prompt
            if file_path_mappings:
                file_path_info = "\n".join(file_path_mappings)
                modified_user_prompt = f"{user_prompt}\n\n{file_path_info}"
            
            # Call the LLM with the modified user prompt
            result = self.llm_client.call_llm(files=files, system_prompt=system_prompt, user_prompt=modified_user_prompt, strategy_type=strategy_type)
            
            return result
    
    async def call_llm_async(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str,
                           strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Async version of call_llm."""
        if not files:
            return await self.llm_client.call_llm_async(files=files, system_prompt=system_prompt, user_prompt=user_prompt, strategy_type=strategy_type)
        
        # For Google GenAI client, we need to modify the FILE_PATH information
        if hasattr(self.llm_client, '_modify_file_paths_for_llm_async'):
            return await self.llm_client._modify_file_paths_for_llm_async(
                files=files, 
                system_prompt=system_prompt, 
                user_prompt=user_prompt,
                file_path_mapper=self.file_path_mapper,
                strategy_type=strategy_type
            )
        else:
            # For other clients (like Claude), we need to modify the user prompt
            # to include the original PDF paths in FILE_PATH information
            modified_user_prompt = user_prompt
            
            # Add FILE_PATH mapping information to the user prompt
            # to include the original PDF paths in FILE_PATH information
            file_path_mappings = []
            for file_path in files:
                original_pdf_path = self.file_path_mapper.get_original_path(file_path)
                if original_pdf_path:
                    file_path_mappings.append(f"FILE_PATH: {original_pdf_path}")
                    logging.debug(f"🔗 Added FILE_PATH mapping: {os.path.basename(file_path)} -> {os.path.basename(original_pdf_path)}")
                else:
                    file_path_mappings.append(f"FILE_PATH: {file_path}")
            
            # Add the FILE_PATH information to the user prompt
            if file_path_mappings:
                file_path_info = "\n".join(file_path_mappings)
                modified_user_prompt = f"{user_prompt}\n\n{file_path_info}"
            
            # Call the LLM with the modified user prompt
            result = await self.llm_client.call_llm_async(files=files, system_prompt=system_prompt, user_prompt=modified_user_prompt, strategy_type=strategy_type)
            
            return result 