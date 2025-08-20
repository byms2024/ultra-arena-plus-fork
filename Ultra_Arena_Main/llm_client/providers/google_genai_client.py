"""
Google GenAI client implementation.
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import google.genai as genai
from google.genai import types

from ..llm_client_base import BaseLLMClient
from ..client_utils import create_content_parts_with_embedded_names


class GoogleGenAIClientBase(BaseLLMClient):
    """Base class for Google GenAI clients with common functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize Google GenAI client
        api_key = config.get("api_key") or os.environ.get("GCP_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        try:
            masked = (api_key[:4] + "..." + api_key[-4:]) if api_key and len(api_key) > 8 else "(empty)"
            logging.info(f"🔐 [GoogleGenAI] Using API key: {masked}")
        except Exception:
            logging.info("🔐 [GoogleGenAI] Using API key: (masked)")
        self.client = genai.Client(api_key=api_key)
        
        # Model configuration
        self.model_id = config.get("model_id", "gemini-2.5-flash")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 1000000)
        
        # File upload tracking
        self.uploaded_files = []
        
    def _upload_files(self, files: List[str]) -> Tuple[List[Any], List[str]]:
        """Upload files to Google GenAI and return uploaded files and original filenames."""
        uploaded_files = []
        original_filenames = []
        
        for i, file_path in enumerate(files):
            try:
                logging.info(f"📤 Uploading file {i+1}/{len(files)}: {os.path.basename(file_path)} ({os.path.getsize(file_path)} bytes)")
                
                # Upload file
                uploaded_file = self.client.files.upload(file=file_path)
                self.uploaded_files.append(uploaded_file)
                
                # Wait for file to be active
                logging.info(f"⏳ Waiting for file to be active: {os.path.basename(file_path)}")
                while uploaded_file.state.name != "ACTIVE":
                    time.sleep(0.1)
                    uploaded_file = self.client.files.get(name=uploaded_file.name)
                
                logging.info(f"✅ File {uploaded_file.name} is now active")
                
                # Store uploaded file and original filename
                uploaded_files.append(uploaded_file)
                original_filenames.append(os.path.basename(file_path))
                
                logging.info(f"✅ File uploaded and active: {os.path.basename(file_path)}")
                
            except Exception as e:
                logging.error(f"❌ Error uploading file {file_path}: {str(e)}")
                raise
        
        return uploaded_files, original_filenames

    def _cleanup_files(self):
        """Clean up uploaded files."""
        for uploaded_file in self.uploaded_files:
            try:
                # Check file state before deletion
                file_info = self.client.files.get(name=uploaded_file.name)
                logging.debug(f"🔍 File {uploaded_file.name} state: {file_info.state.name}")
                
                # Delete file
                self.client.files.delete(name=uploaded_file.name)
                logging.debug(f"🗑️ Cleaned up file: {uploaded_file.name}")
                
            except Exception as e:
                if "403" in str(e):
                    logging.warning(f"⚠️ Failed to cleanup file {uploaded_file.name} - 403 PERMISSION_DENIED (file may have expired)")
                else:
                    logging.warning(f"⚠️ Failed to cleanup file {uploaded_file.name}: {str(e)}")
        
        self.uploaded_files = []
    
    def _parse_google_response(self, response, user_prompt: str = ""):
        """Parse Google GenAI response and extract structured data"""
        try:
            # Handle 'text' objects in response
            if hasattr(response, 'text'):
                response_text = response.text
            else:
                # This is a regular Google GenAI response
                if not hasattr(response, 'candidates') or not response.candidates:
                    logging.warning("🔍 No candidates in Google GenAI response")
                    return []
                
                candidate = response.candidates[0]
                if not hasattr(candidate, 'content') or not candidate.content:
                    logging.warning("🔍 No content in Google GenAI candidate")
                    return []
                
                # Extract text from content parts
                response_text = ""
                for part in candidate.content.parts:
                    if hasattr(part, 'text'):
                        response_text += part.text
            
            # Log the raw response text
            logging.debug(f"🔍 Raw response text from Google (length: {len(response_text)}): {response_text[:500]}...")
            
            # Try to parse as JSON
            try:
                # Remove markdown code blocks if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]  # Remove "```json"
                if response_text.endswith("```"):
                    response_text = response_text[:-3]  # Remove "```"
                
                # Parse JSON
                data = json.loads(response_text.strip())
                
                # Handle both array and single object responses
                if isinstance(data, list):
                    # Add file_name_llm field to each result if not present
                    for item in data:
                        if isinstance(item, dict) and "file_name_llm" not in item:
                            # Extract filename from the prompt if available
                            filename = self._extract_filename_from_prompt(user_prompt)
                            if filename:
                                item["file_name_llm"] = filename
                    return data
                else:
                    # Single object - wrap in list and add file_name_llm
                    if isinstance(data, dict) and "file_name_llm" not in data:
                        filename = self._extract_filename_from_prompt(user_prompt)
                        if filename:
                            data["file_name_llm"] = filename
                    return [data]
                    
            except json.JSONDecodeError as e:
                logging.warning(f"🔍 Response is not valid JSON, treating as plain text: {e}")
                # Return as plain text
                return [{"text": response_text}]
                
        except Exception as e:
            logging.error(f"🔍 Error parsing Google GenAI response: {e}")
            return []
    
    def _extract_filename_from_prompt(self, user_prompt: str) -> Optional[str]:
        """Extract original filename from the user prompt."""
        try:
            # Look for FILE_PATH information in the prompt
            if "FILE_PATH:" in user_prompt:
                lines = user_prompt.split('\n')
                for line in lines:
                    if line.startswith("FILE_PATH:"):
                        file_path = line.replace("FILE_PATH:", "").strip()
                        return os.path.basename(file_path)
        except Exception as e:
            logging.debug(f"🔍 Error extracting filename from prompt: {e}")
        
        return None


class GoogleGenAIClient(GoogleGenAIClientBase):
    """Google GenAI client for non-streaming requests."""
    
    def call_llm(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str, 
                 strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call Google GenAI with non-streaming API."""
        try:
            # Handle image_first strategy differently
            if strategy_type == "image_first" and files:
                return self._call_llm_image_first(files=files, system_prompt=system_prompt, user_prompt=user_prompt)
            
            # Standard direct_file processing
            return self._call_llm_file_first(files=files, system_prompt=system_prompt, user_prompt=user_prompt)
            
        except Exception as e:
            logging.error(f"Google GenAI API error: {e}")
            logging.debug(f"Request details - Model: {self.model_id}, Files: {files}, Prompt length: {len(user_prompt)}")
            raise
        finally:
            # Clean up uploaded files
            self._cleanup_files()

    def _call_llm_file_first(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str) -> Dict[str, Any]:
        """Handle file_first strategy for Google GenAI."""
        try:
            # Upload files
            uploaded_files, original_filenames = self._upload_files(files or [])
            
            # Prepare content parts
            parts = create_content_parts_with_embedded_names(
                files=uploaded_files,
                original_filenames=original_filenames,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                file_uri_getter=lambda file_obj: file_obj.uri
            )
            
            # Log request details
            logging.debug(f"Making request to {self.model_id} with {len(parts)} content parts")
            if files:
                logging.debug(f"Files included: {[os.path.basename(f) for f in files]}")
            
            # Make API call
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=parts
            )
            
            # Debug: Log the raw response using centralized logging utility
            from llm_client.llm_response_logging import log_llm_response
            log_llm_response("Google GenAI", response)
            
            # Parse Google GenAI specific response
            result = self._parse_google_response(response, user_prompt)
            
            # Add token usage info to the result(s)
            if hasattr(response, 'usage_metadata'):
                total_prompt_tokens = response.usage_metadata.prompt_token_count
                total_candidates_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count
                
                # Use common token population function
                from llm_client.token_utils import populate_requestgroup_actual_tokens
                populate_requestgroup_actual_tokens(
                    result=result,
                    prompt_tokens=total_prompt_tokens,
                    candidate_tokens=total_candidates_tokens,
                    total_tokens=total_tokens,
                    provider_name="Google GenAI"
                )
            
            return result
            
        except Exception as e:
            logging.error(f"Google GenAI File First API error: {e}")
            logging.debug(f"Request details - Model: {self.model_id}, Files: {files}, Prompt length: {len(user_prompt)}")
            raise

    def _call_llm_image_first(self, *, files: List[str], system_prompt: Optional[str] = None, user_prompt: str) -> Dict[str, Any]:
        """Handle image_first strategy for Google GenAI."""
        try:
            logging.info(f"🖼️ Using Google GenAI image_first strategy with {len(files)} image files")
            
            # For image_first strategy, files are already image files (PNG, JPG, etc.)
            # We don't need to upload them since we'll use inline_data
            original_filenames = [os.path.basename(f) for f in files]
            
            # Prepare content parts for images
            parts = create_content_parts_with_embedded_names(
                files=files,
                original_filenames=original_filenames,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                is_image_mode=True,
                file_uri_getter=lambda file_obj: file_obj.uri
            )
            
            # Log request details
            logging.debug(f"Making image_first request to {self.model_id} with {len(parts)} content parts")
            logging.debug(f"Image files included: {[os.path.basename(f) for f in files]}")
            
            # Make API call
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=parts
            )
            
            # Debug: Log the raw response using centralized logging utility
            from llm_client.llm_response_logging import log_llm_response
            log_llm_response("Google GenAI (Image First)", response)
            
            # Parse Google GenAI specific response
            result = self._parse_google_response(response, user_prompt)
            
            # Add token usage info to the result(s)
            if hasattr(response, 'usage_metadata'):
                total_prompt_tokens = response.usage_metadata.prompt_token_count
                total_candidates_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count
                
                # Use common token population function
                from llm_client.token_utils import populate_requestgroup_actual_tokens
                populate_requestgroup_actual_tokens(
                    result=result,
                    prompt_tokens=total_prompt_tokens,
                    candidate_tokens=total_candidates_tokens,
                    total_tokens=total_tokens,
                    provider_name="Google GenAI (Image First)"
                )
            
            return result
            
        except Exception as e:
            logging.error(f"Google GenAI Image First API error: {e}")
            logging.debug(f"Request details - Model: {self.model_id}, Image files: {files}, Prompt length: {len(user_prompt)}")
            raise
    
    async def call_llm_async(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str,
                           strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call Google GenAI asynchronously (runs sync method in thread pool)."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.call_llm, files=files, system_prompt=system_prompt, user_prompt=user_prompt)


 