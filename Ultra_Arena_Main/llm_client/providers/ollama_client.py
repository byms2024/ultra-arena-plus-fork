"""
Ollama client implementation.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
import os # Added for os.path.basename

# Import Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available. Install with: pip install ollama")

from ..llm_client_base import BaseLLMClient
from ..client_utils import create_content_parts_with_embedded_names


class OllamaClient(BaseLLMClient):
    """Ollama client for local LLM processing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama not available")
        
        self.model_name = config["model"]
    
    def call_llm(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str, 
                 strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call Ollama with user prompt, optional system prompt (text-only for now)."""
        try:
            # Ollama doesn't support file uploads in the same way
            # For text-first processing, we pass the extracted text as part of the prompt
            messages = []
            
            # Add system prompt if provided, otherwise use default
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append({"role": "system", "content": "You are a helpful assistant that extracts information from documents."})
            
            # Add user prompt with filename embedding if files are provided
            if files and len(files) > 0:
                # Use the evolved filename embedding method
                content_parts = create_content_parts_with_embedded_names(
                    files=files,
                    original_filenames=[os.path.basename(f) for f in files],
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    is_image_mode=False
                )
                
                # Extract text content from parts for Ollama
                enhanced_prompt = ""
                for part in content_parts:
                    if part.get("text"):
                        enhanced_prompt += part["text"] + "\n"
                
                messages.append({"role": "user", "content": enhanced_prompt.strip()})
            else:
                messages.append({"role": "user", "content": user_prompt})
            
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            
            # Debug: Log the raw response using centralized logging utility
            from llm_client.llm_response_logging import log_llm_response
            log_llm_response("Ollama", response)
            
            result = self._parse_ollama_response(response.message.content)
            
            # Add file_name_llm mapping for proper file tracking
            if files and len(files) > 0:
                # With the evolved filename embedding method, the LLM should return file_name_llm
                # But we'll add fallback logic in case it doesn't
                if isinstance(result, list):
                    # Multi-file response - add file_name_llm to each result if missing
                    for i, item in enumerate(result):
                        if isinstance(item, dict) and i < len(files):
                            if "file_name_llm" not in item:
                                item["file_name_llm"] = os.path.basename(files[i])
                elif isinstance(result, dict):
                    # Single file response - add file_name_llm if missing
                    if "file_name_llm" not in result:
                        result["file_name_llm"] = os.path.basename(files[0])
            
            return result
            
        except Exception as e:
            # Enhanced error logging for Ollama with clear visual indicators
            error_str = str(e).lower()
            
            if "failed to connect" in error_str or "connection refused" in error_str:
                logging.error(f"🔌❌ CRITICAL: Ollama service unavailable - {e}")
                logging.error(f"💡 Please start Ollama service: https://ollama.com/download")
            elif "not found" in error_str and "model" in error_str:
                logging.error(f"🤖❌ MODEL ERROR: Ollama model '{self.model_name}' not found - {e}")
                logging.error(f"💡 Pull model with: ollama pull {self.model_name}")
            elif "timeout" in error_str:
                logging.error(f"⏰❌ TIMEOUT: Ollama request timed out - {e}")
            elif "refused" in error_str or "unavailable" in error_str:
                logging.error(f"🚨 CONNECTION ERROR: {e}")
            else:
                logging.error(f"❌ Ollama API error: {e}")
            
            return {"error": str(e)}
    
    def _parse_ollama_response(self, content: str) -> Dict[str, Any]:
        """Parse Ollama response, handling thinking tags and JSON extraction."""
        try:
            # Remove thinking tags if present
            if "<think>" in content and "</think>" in content:
                # Extract content after </think>
                parts = content.split("</think>")
                if len(parts) > 1:
                    content = parts[1].strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content.split("```json", 1)[1]
            elif content.startswith("```"):
                content = content.split("```", 1)[1]
            
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            
            content = content.strip()
            
            # Handle empty content
            if not content:
                logging.warning("Received empty response from Ollama")
                return {"error": "Empty response from Ollama"}
            
            # Try to parse as JSON
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Ollama JSON response: {e}")
            # Truncate content to prevent large log files
            truncated_content = content[:500] + "..." if len(content) > 500 else content
            logging.error(f"Content (truncated): {truncated_content}")
            return {"error": f"JSON parsing failed: {e}"}
        except Exception as e:
            logging.error(f"Error parsing Ollama response: {e}")
            return {"error": f"Response parsing failed: {e}"}
    
    async def call_llm_async(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str,
                           strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call Ollama asynchronously."""
        # Run in thread pool since Ollama doesn't have async API
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.call_llm, files=files, system_prompt=system_prompt, user_prompt=user_prompt) 