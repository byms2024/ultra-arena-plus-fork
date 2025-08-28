"""
Base client mixin for OpenAI-style clients.

This mixin provides common functionality for clients that follow the OpenAI API pattern,
including OpenAI, TogetherAI, and future providers like Grok.
"""

import json
import logging
import base64
import os
from typing import Dict, List, Any, Optional

from ..client_utils import _create_filename_embedded_prompt


class BaseClientMixin:
    """Common functionality for OpenAI-style clients (OpenAI, TogetherAI, Grok, etc.)"""
    
    def _validate_strategy(self, strategy_type: str) -> Optional[Dict[str, Any]]:
        """Validate strategy type for OpenAI-style APIs."""
        if strategy_type in ["direct_file", "text_first"]:
            error_msg = f"{self.provider_name} does not support {strategy_type} strategy. Use image_first strategy instead."
            logging.error(f"❌ {error_msg}")
            return {"error": error_msg}
        return None
    
    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type from file extension."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Image types
        if file_ext == '.png':
            return 'image/png'
        elif file_ext == '.jpg' or file_ext == '.jpeg':
            return 'image/jpeg'
        elif file_ext == '.gif':
            return 'image/gif'
        elif file_ext == '.webp':
            return 'image/webp'
        # Document types
        elif file_ext == '.pdf':
            return 'application/pdf'
        else:
            return 'application/octet-stream'  # default
    
    def _encode_file_to_base64(self, file_path: str) -> str:
        """Encode file to base64."""
        with open(file_path, 'rb') as f:
            file_data = f.read()
            return base64.b64encode(file_data).decode('utf-8')
    
    def _build_image_message_content(self, user_prompt: str, files: List[str]) -> List[Dict]:
        """Build message content with images for image_first strategy."""
        content = [{"type": "text", "text": user_prompt}]
        
        for file_path in files:
            # Add filename marker
            original_filename = os.path.basename(file_path)
            content.append({
                "type": "text", 
                "text": f"=== FILE: {original_filename} ==="
            })
            
            # Encode image to base64
            base64_image = self._encode_file_to_base64(file_path)
            mime_type = self._get_mime_type(file_path)
            
            # Add image to content
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            })
            
            logging.debug(f"📎 Added base64 image: {file_path} ({mime_type})")
            
            # Add end filename marker
            content.append({
                "type": "text", 
                "text": f"=== END FILE: {original_filename} ==="
            })
        
        return content
    
    def _build_standard_message_content(self, user_prompt: str, files: Optional[List[str]] = None) -> Any:
        """Build message content for standard processing."""
        if not files:
            return user_prompt
        
        # For files, use base64 encoding
        content = [{"type": "text", "text": user_prompt}]
        
        for file_path in files:
            base64_data = self._encode_file_to_base64(file_path)
            mime_type = self._get_mime_type(file_path)
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_data}"
                }
            })
            
            logging.debug(f"📎 Added base64 file: {file_path} ({mime_type})")
        
        return content
    
    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse LLM response into standardized format."""
        try:
            if isinstance(response, str):
                # Handle empty or whitespace-only responses
                if not response or response.strip() == "":
                    logging.warning(f"Received empty response from {self.provider_name}")
                    return {"error": f"Empty response from {self.provider_name}"}
                parsed = json.loads(response)
            elif hasattr(response, 'text'):
                # Handle empty or whitespace-only responses
                if not response.text or response.text.strip() == "":
                    logging.warning(f"Received empty response from {self.provider_name}")
                    return {"error": f"Empty response from {self.provider_name}"}
                parsed = json.loads(response.text)
            elif isinstance(response, dict):
                parsed = response
            else:
                logging.error(f"Unexpected response type: {type(response)}")
                return {"error": f"Unexpected response type: {type(response)}"}
            
            # Handle batch response format with "results" or "result" key
            if isinstance(parsed, dict) and "results" in parsed and isinstance(parsed["results"], list):
                logging.info(f"🔍 Detected {self.provider_name} batch response with {len(parsed['results'])} results")
                return parsed["results"]
            elif isinstance(parsed, dict) and "result" in parsed and isinstance(parsed["result"], list):
                logging.info(f"🔍 Detected {self.provider_name} batch response with {len(parsed['result'])} results")
                return parsed["result"]
            
            return parsed
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            return {"error": f"JSON parsing failed: {e}"}
    
    def _add_token_usage_to_result(self, result: Dict[str, Any], response: Any) -> None:
        """Add token usage information to result."""
        if hasattr(response, 'usage') and response.usage:
            # Use common token population function if available
            try:
                from ..token_utils import populate_requestgroup_actual_tokens
                populate_requestgroup_actual_tokens(
                    result=result,
                    prompt_tokens=response.usage.prompt_tokens,
                    candidate_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    provider_name=self.provider_name
                )
            except ImportError:
                # Fallback to direct assignment
                result['prompt_token_count'] = response.usage.prompt_tokens
                result['candidates_token_count'] = response.usage.completion_tokens
                result['total_token_count'] = response.usage.total_tokens
    
    def _log_llm_response(self, response: Any) -> None:
        """Log LLM response using centralized logging utility."""
        try:
            from ..llm_response_logging import log_llm_response
            log_llm_response(self.provider_name, response)
        except ImportError:
            logging.debug(f"Response logging not available for {self.provider_name}")
    
    def _create_messages(self, system_prompt: Optional[str], user_content: Any) -> List[Dict]:
        """Create messages array for API call."""
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": user_content})
        
        return messages 