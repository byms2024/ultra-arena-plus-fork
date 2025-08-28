"""
Claude API client implementation.
"""

import json
import logging
import base64
import mimetypes
import os
import asyncio
from typing import Dict, List, Any, Optional

# Import Anthropic Claude
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("Anthropic not available. Install with: pip install anthropic")

from ..llm_client_base import BaseLLMClient
from ..client_utils import create_content_parts_with_embedded_names


class ClaudeClient(BaseLLMClient):
    """Claude API client for direct file processing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic not available")
        
        self.client = anthropic.Anthropic(api_key=config["api_key"])
        self.model_id = config.get("model", "claude-sonnet-4-20250514")
    
    def call_llm(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str, 
                 strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call Claude API with user prompt, optional system prompt, and files."""
        try:
            # Build messages
            messages = []
            content_parts = []
            
            # Add user prompt with filename embedding if files are provided
            if files:
                # Use the evolved filename embedding method for the prompt
                from ..client_utils import _create_filename_embedded_prompt
                enhanced_prompt = _create_filename_embedded_prompt(
                    user_prompt=user_prompt,
                    file_type="file",
                    example_filename=os.path.basename(files[0])
                )
                content_parts.append({
                    "type": "text",
                    "text": enhanced_prompt
                })
                
                # Add files with filename markers
                for i, file_path in enumerate(files):
                    logging.info(f"📤 Processing file {i+1}/{len(files)}: {os.path.basename(file_path)}")
                    
                    # Add filename marker
                    content_parts.append({
                        "type": "text",
                        "text": f"=== FILE: {os.path.basename(file_path)} ==="
                    })
                    
                    # Process file based on type
                    file_content = self._process_file(file_path)
                    if file_content:
                        content_parts.append(file_content)
                        logging.info(f"✅ File processed: {os.path.basename(file_path)}")
                    else:
                        logging.warning(f"⚠️ Could not process file: {os.path.basename(file_path)}")
                    
                    # Add end filename marker
                    content_parts.append({
                        "type": "text",
                        "text": f"=== END FILE: {os.path.basename(file_path)} ==="
                    })
            else:
                # No files, just add the original user prompt
                content_parts.append({
                    "type": "text",
                    "text": user_prompt
                })
            
            messages.append({
                "role": "user",
                "content": content_parts
            })
            
            # Prepare request parameters
            request_params = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            # Add system prompt if provided
            if system_prompt:
                request_params["system"] = system_prompt
            
            # Log the request being sent to Claude
            logging.info("🔍 CLAUDE REQUEST:")
            logging.debug(f"   System Prompt: {system_prompt}")
            logging.debug(f"   User Prompt: {user_prompt}")
            if files:
                logging.info(f"   Files: {[os.path.basename(f) for f in files]}")
            logging.info(f"   Model: {self.model_id}")
            logging.info(f"   Temperature: {self.temperature}")
            logging.info(f"   Max Tokens: {self.max_tokens}")
            
            response = self.client.messages.create(**request_params)
            
            # Debug: Log the raw response using centralized logging utility
            from llm_client.llm_response_logging import log_llm_response
            log_llm_response("Claude", response)
            
            # Log the raw response from Claude
            logging.info("🔍 CLAUDE RESPONSE:")
            if hasattr(response, 'content') and response.content:
                response_text = ""
                for content_block in response.content:
                    if hasattr(content_block, 'text'):
                        response_text += content_block.text
                # Truncate content to prevent large log files
                truncated_response = response_text[:500] + "..." if len(response_text) > 500 else response_text
                logging.info(f"   Raw Response (truncated): {truncated_response}")
            if hasattr(response, 'usage'):
                usage = response.usage
                logging.info(f"   Token Usage: {usage.input_tokens} input, {usage.output_tokens} output")
                logging.info(f"   Complete Usage Field: {usage}")
            
            # Parse Claude specific response
            result = self._parse_claude_response(response)
            
            return result
            
        except Exception as e:
            logging.error(f"Claude API error: {e}")
            return {"error": str(e)}
    
    async def call_llm_async(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str,
                           strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call Claude API asynchronously."""
        try:
            # Build messages
            messages = []
            content_parts = []
            
            # Add user prompt
            content_parts.append({
                "type": "text",
                "text": user_prompt
            })
            
            # Add files if provided
            if files:
                for i, file_path in enumerate(files):
                    logging.info(f"📤 Processing file {i+1}/{len(files)}: {os.path.basename(file_path)}")
                    
                    # Add FILE_PATH information
                    content_parts.append({
                        "type": "text",
                        "text": f"FILE_PATH: {file_path}"
                    })
                    
                    # Process file based on type
                    file_content = self._process_file(file_path)
                    if file_content:
                        content_parts.append(file_content)
                        logging.info(f"✅ File processed: {os.path.basename(file_path)}")
                    else:
                        logging.warning(f"⚠️ Could not process file: {os.path.basename(file_path)}")
            
            messages.append({
                "role": "user",
                "content": content_parts
            })
            
            # Prepare request parameters
            request_params = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            # Add system prompt if provided
            if system_prompt:
                request_params["system"] = system_prompt
            
            # Log the request being sent to Claude (async)
            logging.info("🔍 CLAUDE REQUEST (ASYNC):")
            logging.debug(f"   System Prompt: {system_prompt}")
            logging.debug(f"   User Prompt: {user_prompt}")
            if files:
                logging.info(f"   Files: {[os.path.basename(f) for f in files]}")
            logging.info(f"   Model: {self.model_id}")
            logging.info(f"   Temperature: {self.temperature}")
            logging.info(f"   Max Tokens: {self.max_tokens}")
            
            # Use async client
            async_client = anthropic.AsyncAnthropic(api_key=self.client.api_key)
            response = await async_client.messages.create(**request_params)
            
            # Log the raw response from Claude (async)
            logging.info("🔍 CLAUDE RESPONSE (ASYNC):")
            if hasattr(response, 'content') and response.content:
                response_text = ""
                for content_block in response.content:
                    if hasattr(content_block, 'text'):
                        response_text += content_block.text
                # Truncate content to prevent large log files
                truncated_response = response_text[:500] + "..." if len(response_text) > 500 else response_text
                logging.info(f"   Raw Response (truncated): {truncated_response}")
            if hasattr(response, 'usage'):
                usage = response.usage
                logging.info(f"   Token Usage: {usage.input_tokens} input, {usage.output_tokens} output")
                logging.info(f"   Complete Usage Field: {usage}")
            
            # Parse Claude specific response
            result = self._parse_claude_response(response)
            
            return result
            
        except Exception as e:
            logging.error(f"Claude API error: {e}")
            return {"error": str(e)}
    
    def _process_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Process file based on its type for Claude API."""
        try:
            # Get file mime type
            mime_type, _ = mimetypes.guess_type(file_path)
            
            # Check if file is an image that Claude supports
            supported_image_types = [
                "image/jpeg", "image/png", "image/gif", "image/webp"
            ]
            
            if mime_type in supported_image_types:
                # Process as image
                with open(file_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": image_data
                    }
                }
            elif mime_type == "application/pdf":
                # Convert PDF to image for Claude (consistent with image-first strategy)
                import fitz  # PyMuPDF
                import io
                from PIL import Image
                
                # Open PDF and convert first page to image
                doc = fitz.open(file_path)
                page = doc.load_page(0)  # First page
                
                # Convert to image with higher resolution
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Convert to base64
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                
                doc.close()
                
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_data
                    }
                }
            else:
                # Process as text file
                try:
                    with open(file_path, 'r', encoding='utf-8') as text_file:
                        text_content = text_file.read()
                    
                    return {
                        "type": "text",
                        "text": f"File content of {os.path.basename(file_path)}:\n\n{text_content}"
                    }
                except UnicodeDecodeError:
                    # Try with different encoding or treat as binary
                    logging.warning(f"Could not decode {file_path} as UTF-8")
                    return {
                        "type": "text",
                        "text": f"[Binary file: {os.path.basename(file_path)} - content not displayable as text]"
                    }
                    
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            return None
    
    def count_tokens(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str) -> Dict[str, int]:
        """Count tokens for a Claude request without sending it."""
        try:
            # Build messages (same logic as call_llm)
            messages = []
            content_parts = []
            
            # Add user prompt
            content_parts.append({
                "type": "text",
                "text": user_prompt
            })
            
            # Add files if provided
            if files:
                for i, file_path in enumerate(files):
                    # Add FILE_PATH information
                    content_parts.append({
                        "type": "text",
                        "text": f"FILE_PATH: {file_path}"
                    })
                    
                    # Process file based on type
                    file_content = self._process_file(file_path)
                    if file_content:
                        content_parts.append(file_content)
            
            messages.append({
                "role": "user",
                "content": content_parts
            })
            
            # Prepare request parameters for token counting
            request_params = {
                "model": self.model_id,
                "messages": messages
            }
            
            # Add system prompt if provided
            if system_prompt:
                request_params["system"] = system_prompt
            
            # Count tokens using Claude's API
            token_count = self.client.messages.count_tokens(**request_params)
            
            return {
                'input_tokens': token_count.input_tokens,
                'output_tokens': 0,  # We can't predict output tokens
                'total_tokens': token_count.input_tokens
            }
            
        except Exception as e:
            logging.error(f"Error counting Claude tokens: {e}")
            return {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}

    async def count_tokens_async(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str) -> Dict[str, int]:
        """Count tokens for a Claude request asynchronously."""
        try:
            # Build messages (same logic as call_llm_async)
            messages = []
            content_parts = []
            
            # Add user prompt
            content_parts.append({
                "type": "text",
                "text": user_prompt
            })
            
            # Add files if provided
            if files:
                for i, file_path in enumerate(files):
                    # Add FILE_PATH information
                    content_parts.append({
                        "type": "text",
                        "text": f"FILE_PATH: {file_path}"
                    })
                    
                    # Process file based on type
                    file_content = self._process_file(file_path)
                    if file_content:
                        content_parts.append(file_content)
                    else:
                        # If file processing failed, add a placeholder text instead of empty content
                        content_parts.append({
                            "type": "text",
                            "text": f"[Error processing file: {os.path.basename(file_path)}]"
                        })
            
            messages.append({
                "role": "user",
                "content": content_parts
            })
            
            # Prepare request parameters for token counting
            request_params = {
                "model": self.model_id,
                "messages": messages
            }
            
            # Add system prompt if provided
            if system_prompt:
                request_params["system"] = system_prompt
            
            # Use async client for token counting
            async_client = anthropic.AsyncAnthropic(api_key=self.client.api_key)
            token_count = await async_client.messages.count_tokens(**request_params)
            
            return {
                'input_tokens': token_count.input_tokens,
                'output_tokens': 0,  # We can't predict output tokens
                'total_tokens': token_count.input_tokens
            }
            
        except Exception as e:
            logging.error(f"Error counting Claude tokens (async): {e}")
            return {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}

    def _parse_claude_response(self, response) -> Dict[str, Any]:
        """Parse Claude specific response format."""
        try:
            # Extract text from response
            text_content = ""
            if hasattr(response, 'content') and response.content:
                for content_block in response.content:
                    if hasattr(content_block, 'text'):
                        text_content += content_block.text
            
            # Try to parse as JSON first
            try:
                result = json.loads(text_content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                import re
                # First try to match object JSON (most common)
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', text_content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        # If object JSON fails, try array JSON
                        json_match = re.search(r'```json\s*(\[.*?\])\s*```', text_content, re.DOTALL)
                        if json_match:
                            try:
                                result = json.loads(json_match.group(1))
                            except json.JSONDecodeError:
                                result = {"text": text_content}
                        else:
                            result = {"text": text_content}
                else:
                    # If no object JSON found, try array JSON
                    json_match = re.search(r'```json\s*(\[.*?\])\s*```', text_content, re.DOTALL)
                    if json_match:
                        try:
                            result = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            result = {"text": text_content}
                    else:
                        # If not JSON, return as plain text
                        result = {"text": text_content}
            
            # Add token usage info if available
            if hasattr(response, 'usage'):
                usage = response.usage
                # Use common token population function
                from llm_client.token_utils import populate_requestgroup_actual_tokens
                populate_requestgroup_actual_tokens(
                    result=result,
                    prompt_tokens=usage.input_tokens,
                    candidate_tokens=usage.output_tokens,
                    total_tokens=usage.input_tokens + usage.output_tokens,
                    provider_name="Claude"
                )
            
            return result
            
        except Exception as e:
            logging.error(f"Error parsing Claude response: {e}")
            return {"error": str(e)} 