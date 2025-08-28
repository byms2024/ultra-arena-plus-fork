"""
OpenAI styled streaming client implementation.

This module contains the streaming-specific implementation for OpenAI,
separated from the main client to improve code organization.
"""

import json
import logging
import asyncio
import base64
import os
from typing import Dict, List, Any, Optional

# Import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Install with: pip install openai")

from ..llm_client_base import BaseLLMClient
from ..client_utils import _create_filename_embedded_prompt


class OpenAIStyledStreamingClient(BaseLLMClient):
    """OpenAI client for streaming requests."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available")
        
        self.client = OpenAI(api_key=config["api_key"])
        self.model_id = config["model"]
    
    def call_llm(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str, 
                 strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call OpenAI with streaming API."""
        
        # OpenAI only supports image_first strategy - other strategies don't work with OpenAI's API limitations
        if strategy_type in ["direct_file", "text_first"]:
            error_msg = f"OpenAI does not support {strategy_type} strategy. Use image_first strategy instead."
            logging.error(f"❌ {error_msg}")
            return {"error": error_msg}
        
        # Check if this is image first strategy with OpenAI - apply special treatment
        if strategy_type == "image_first" and files:
            return self._call_llm_image_first_streaming(files=files, system_prompt=system_prompt, user_prompt=user_prompt)
        
        # Standard OpenAI processing for other strategies
        return self._call_llm_standard_streaming(files=files, system_prompt=system_prompt, user_prompt=user_prompt)
    
    def _call_llm_image_first_streaming(self, *, files: List[str], system_prompt: Optional[str] = None, user_prompt: str) -> Dict[str, Any]:
        """
        Special treatment for OpenAI with image first strategy using streaming.
        """
        try:
            logging.info(f"🔄 Using OpenAI streaming for image first strategy with {len(files)} files")
            
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message with special image first handling
            # Use the utility function to create filename-embedded prompt
            enhanced_prompt = _create_filename_embedded_prompt(user_prompt, file_type="image", example_filename="image1.png")
            user_message = {"role": "user", "content": enhanced_prompt}
            
            # For image first strategy with OpenAI, we need to handle images differently
            # Use base64 encoding for images instead of file uploads
            
            # Add file references to the user message
            user_message["content"] = [
                {"type": "text", "text": user_prompt}
            ]
            
            for file_path in files:
                # Add filename marker
                original_filename = os.path.basename(file_path)
                user_message["content"].append({
                    "type": "text", 
                    "text": f"=== FILE: {original_filename} ==="
                })
                
                # Read image file and encode as base64
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    
                    # Determine image type from file extension
                    file_ext = os.path.splitext(file_path)[1].lower()
                    if file_ext == '.png':
                        mime_type = 'image/png'
                    elif file_ext == '.jpg' or file_ext == '.jpeg':
                        mime_type = 'image/jpeg'
                    elif file_ext == '.gif':
                        mime_type = 'image/gif'
                    elif file_ext == '.webp':
                        mime_type = 'image/webp'
                    else:
                        mime_type = 'image/png'  # default
                    
                    # Add image to message content
                    user_message["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    })
                    
                    logging.debug(f"📎 Added base64 image: {file_path} ({mime_type})")
                
                # Add end filename marker
                user_message["content"].append({
                    "type": "text", 
                    "text": f"=== END FILE: {original_filename} ==="
                })
            
            messages.append(user_message)
            
            # Use streaming API
            logging.info(f"🔄 Using streaming API for large response handling")
            response_stream = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                stream=True
            )
            
            # Collect all chunks
            full_text = ""
            for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    full_text += chunk.choices[0].delta.content
            
            # Create a mock response object for consistency
            class StreamingResponse:
                def __init__(self, text):
                    self.text = text
                    self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': text})()})()]
                
                def __str__(self):
                    return f"StreamingResponse(text='{self.text[:100]}...')"
            
            response = StreamingResponse(full_text)
            
            # Debug: Log the raw response using centralized logging utility
            from llm_client.llm_response_logging import log_llm_response
            log_llm_response("OpenAI (Streaming)", response)
            
            result = self._parse_response(full_text)
            
            # For streaming, we need to estimate tokens since usage_metadata is not available
            # Estimate prompt tokens based on the full prompt
            full_prompt = ""
            if system_prompt:
                full_prompt += f"{system_prompt}\n\n"
            full_prompt += user_prompt
            
            # Estimate tokens using tiktoken (rough approximation)
            try:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")  # Use the same encoding as GPT models
                
                # Estimate prompt tokens
                prompt_tokens = len(encoding.encode(full_prompt))
                
                # Estimate response tokens
                response_tokens = len(encoding.encode(full_text))
                
                # Total tokens
                total_tokens = prompt_tokens + response_tokens
                
                # Use common token population function
                from llm_client.token_utils import populate_requestgroup_actual_tokens
                populate_requestgroup_actual_tokens(
                    result=result,
                    prompt_tokens=prompt_tokens,
                    candidate_tokens=response_tokens,
                    total_tokens=total_tokens,
                    provider_name="OpenAI (Streaming)"
                )
                
            except ImportError:
                logging.warning("tiktoken not available for token estimation")
                # Fallback: use simple character count estimation
                prompt_tokens = len(full_prompt) // 4  # Rough estimate
                response_tokens = len(full_text) // 4   # Rough estimate
                total_tokens = prompt_tokens + response_tokens
                
                from llm_client.token_utils import populate_requestgroup_actual_tokens
                populate_requestgroup_actual_tokens(
                    result=result,
                    prompt_tokens=prompt_tokens,
                    candidate_tokens=response_tokens,
                    total_tokens=total_tokens,
                    provider_name="OpenAI (Streaming)"
                )
            
            return result
            
        except Exception as e:
            logging.error(f"OpenAI streaming image first special treatment error: {e}")
            return {"error": str(e)}
    
    def _call_llm_standard_streaming(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str) -> Dict[str, Any]:
        """Standard OpenAI processing with streaming for non-image-first strategies."""
        try:
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message
            user_message = {"role": "user", "content": user_prompt}
            
            if files:
                # For OpenAI, we need to handle files using base64 encoding
                
                # Add file references to the user message
                user_message["content"] = [
                    {"type": "text", "text": user_prompt}
                ]
                
                for file_path in files:
                    # Read file and encode as base64
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                        base64_data = base64.b64encode(file_data).decode('utf-8')
                        
                        # Determine file type from file extension
                        file_ext = os.path.splitext(file_path)[1].lower()
                        if file_ext == '.pdf':
                            mime_type = 'application/pdf'
                        elif file_ext == '.png':
                            mime_type = 'image/png'
                        elif file_ext == '.jpg' or file_ext == '.jpeg':
                            mime_type = 'image/jpeg'
                        elif file_ext == '.gif':
                            mime_type = 'image/gif'
                        elif file_ext == '.webp':
                            mime_type = 'image/webp'
                        else:
                            mime_type = 'application/octet-stream'  # default
                        
                        # Add file to message content
                        user_message["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_data}"
                            }
                        })
                        
                        logging.debug(f"📎 Added base64 file: {file_path} ({mime_type})")
            
            messages.append(user_message)
            
            # Use streaming API
            logging.info(f"🔄 Using streaming API for large response handling")
            response_stream = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                stream=True
            )
            
            # Collect all chunks
            full_text = ""
            for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    full_text += chunk.choices[0].delta.content
            
            # Create a mock response object for consistency
            class StreamingResponse:
                def __init__(self, text):
                    self.text = text
                    self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': text})()})()]
                
                def __str__(self):
                    return f"StreamingResponse(text='{self.text[:100]}...')"
            
            response = StreamingResponse(full_text)
            
            # Debug: Log the raw response using centralized logging utility
            from llm_client.llm_response_logging import log_llm_response
            log_llm_response("OpenAI (Streaming)", response)
            
            result = self._parse_response(full_text)
            
            # For streaming, estimate tokens
            full_prompt = ""
            if system_prompt:
                full_prompt += f"{system_prompt}\n\n"
            full_prompt += user_prompt
            
            try:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")
                
                prompt_tokens = len(encoding.encode(full_prompt))
                response_tokens = len(encoding.encode(full_text))
                total_tokens = prompt_tokens + response_tokens
                
                from llm_client.token_utils import populate_requestgroup_actual_tokens
                populate_requestgroup_actual_tokens(
                    result=result,
                    prompt_tokens=prompt_tokens,
                    candidate_tokens=response_tokens,
                    total_tokens=total_tokens,
                    provider_name="OpenAI (Streaming)"
                )
                
            except ImportError:
                logging.warning("tiktoken not available for token estimation")
                prompt_tokens = len(full_prompt) // 4
                response_tokens = len(full_text) // 4
                total_tokens = prompt_tokens + response_tokens
                
                from llm_client.token_utils import populate_requestgroup_actual_tokens
                populate_requestgroup_actual_tokens(
                    result=result,
                    prompt_tokens=prompt_tokens,
                    candidate_tokens=response_tokens,
                    total_tokens=total_tokens,
                    provider_name="OpenAI (Streaming)"
                )
            
            return result
            
        except Exception as e:
            logging.error(f"OpenAI streaming API error: {e}")
            return {"error": str(e)}
    
    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse OpenAI LLM response into standardized format."""
        try:
            if isinstance(response, str):
                # Handle empty or whitespace-only responses
                if not response or response.strip() == "":
                    logging.warning("Received empty response from OpenAI")
                    return {"error": "Empty response from OpenAI"}
                parsed = json.loads(response)
            elif hasattr(response, 'text'):
                # Handle empty or whitespace-only responses
                if not response.text or response.text.strip() == "":
                    logging.warning("Received empty response from OpenAI")
                    return {"error": "Empty response from OpenAI"}
                parsed = json.loads(response.text)
            elif isinstance(response, dict):
                parsed = response
            else:
                logging.error(f"Unexpected response type: {type(response)}")
                return {"error": f"Unexpected response type: {type(response)}"}
            
            # Handle OpenAI's response for multi-parts request format with "results" or "result" key
            if isinstance(parsed, dict) and "results" in parsed and isinstance(parsed["results"], list):
                logging.info(f"🔍 Detected OpenAI batch response with {len(parsed['results'])} results")
                return parsed["results"]
            elif isinstance(parsed, dict) and "result" in parsed and isinstance(parsed["result"], list):
                logging.info(f"🔍 Detected OpenAI batch response with {len(parsed['result'])} results")
                return parsed["result"]
            
            return parsed
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            return {"error": f"JSON parsing failed: {e}"}
    
    async def call_llm_async(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str,
                           strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call OpenAI asynchronously with streaming."""
        # Run in thread pool since OpenAI doesn't have async API in this context
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.call_llm, files=files, system_prompt=system_prompt, user_prompt=user_prompt, strategy_type=strategy_type) 