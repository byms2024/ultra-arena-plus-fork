"""
LLM client factory for creating provider-specific clients.
"""

import logging
from typing import Dict, List, Any

from .llm_client_base import BaseLLMClient

# Import provider availability checks
try:
    import google.genai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    logging.warning("Google GenAI not available. Install with: pip install google-genai")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Install with: pip install openai")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available. Install with: pip install ollama")

try:
    from openai import OpenAI
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    logging.warning("DeepSeek not available. Install with: pip install openai")

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    logging.warning("Claude not available. Install with: pip install anthropic")

try:
    from openai import OpenAI
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logging.warning("HuggingFace not available. Install with: pip install openai")

try:
    from together import Together
    TOGETHERAI_AVAILABLE = True
except ImportError:
    TOGETHERAI_AVAILABLE = False
    logging.warning("TogetherAI not available. Install with: pip install together")

try:
    from openai import OpenAI
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False
    logging.warning("Grok not available. Install with: pip install openai")


class LLMClientFactory:
    """Factory for creating LLM clients."""
    
    @staticmethod
    def create_client(provider: str, config: Dict[str, Any], streaming: bool = False) -> BaseLLMClient:
        """Create an LLM client based on provider and streaming preference."""
        # Log masked API key just before creating the concrete client
        try:
            key = config.get("api_key", "")
            masked = (key[:4] + "..." + key[-4:]) if key and len(key) > 8 else "(empty)"
            logging.info(f"🔐 [Factory] Provider={provider} Using API key: {masked}")
        except Exception:
            logging.info(f"🔐 [Factory] Provider={provider} Using API key: (masked)")
        if provider == "google":
            if streaming:
                from .providers.google_genai_streaming_client import GoogleGenAIStreamingClient
                return GoogleGenAIStreamingClient(config)
            else:
                from .providers.google_genai_client import GoogleGenAIClient
                return GoogleGenAIClient(config)
        elif provider == "openai":
            if streaming:
                from .providers.openai_styled_streaming_client import OpenAIStyledStreamingClient
                return OpenAIStyledStreamingClient(config)
            else:
                from .providers.openai_styled_client import OpenAIClient
                return OpenAIClient(config)
        elif provider == "ollama":
            from .providers.ollama_client import OllamaClient
            return OllamaClient(config)
        elif provider == "deepseek":
            from .providers.deepseek_client import DeepSeekClient
            return DeepSeekClient(config)
        elif provider == "claude":
            from .providers.claude_client import ClaudeClient
            return ClaudeClient(config)
        elif provider == "huggingface":
            from .providers.huggingface_client import HuggingFaceClient
            return HuggingFaceClient(config)
        elif provider == "togetherai":
            if streaming:
                from .providers.togetherai_streaming_client import TogetherAIStyledStreamingClient
                return TogetherAIStyledStreamingClient(config)
            else:
                from .providers.togetherai_client import TogetherAIClient
                return TogetherAIClient(config)
        elif provider == "grok":
            if GROK_AVAILABLE:
                from .providers.grok_client import GrokClient
                return GrokClient(config)
            else:
                raise ImportError("Grok not available. Install with: pip install openai")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available providers."""
        providers = []
        if GOOGLE_AVAILABLE:
            providers.append("google")
        if OPENAI_AVAILABLE:
            providers.append("openai")
        if OLLAMA_AVAILABLE:
            providers.append("ollama")
        if DEEPSEEK_AVAILABLE:
            providers.append("deepseek")
        if CLAUDE_AVAILABLE:
            providers.append("claude")
        if HUGGINGFACE_AVAILABLE:
            providers.append("huggingface")
        if TOGETHERAI_AVAILABLE:
            providers.append("togetherai")
        if GROK_AVAILABLE:
            providers.append("grok")
        return providers 