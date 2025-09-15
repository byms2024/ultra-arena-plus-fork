"""
Rest Server Configuration Assembler

This module provides the RestServerConfigAssembler for assembling static server configuration
once at startup, which is then cached and reused for all requests.
"""

import time
import logging
from pathlib import Path
from typing import Optional

from .base_config_assembler import BaseConfigAssembler
from .config_models import ServerConfig

logger = logging.getLogger(__name__)


class RestServerConfigAssembler(BaseConfigAssembler):
    """Assembles static server configuration once at startup."""
    
    def __init__(self, run_profile: str):
        """
        Initialize the REST server config assembler.
        
        Args:
            run_profile (str): The profile name to use for configuration
        """
        super().__init__(run_profile)
        self._cached_config: Optional[ServerConfig] = None
        self._assembly_time_ms: float = 0.0
    
    def _get_profile_dir(self) -> Path:
        """Get the REST profile directory path."""
        current_dir = Path(__file__).parent.parent.parent  # server_utils/config_assemblers -> server_utils -> Ultra_Arena_Main_Restful
        return current_dir / "run_profiles" / self.run_profile
    
    def assemble_config(self) -> ServerConfig:
        """
        Assemble static server configuration.
        
        This method loads all static configuration once and caches it.
        Subsequent calls return the cached configuration for performance.
        
        Returns:
            ServerConfig: Assembled server configuration
        """
        # Return cached config if available
        if self._cached_config is not None:
            logger.debug(f"🚀 Using cached server config (assembly time: {self._assembly_time_ms:.3f}ms)")
            return self._cached_config
        
        # Assemble fresh configuration
        logger.info(f"🔧 Assembling fresh server configuration for profile: {self.run_profile}")
        assembly_start = time.time()
        
        try:
            # Load all configuration components
            desensitization = self._load_desensitization_config()
            prompts = self._load_prompt_config(desensitization.desensitization_config)
            api_keys = self._load_api_key_config()
            processing = self._load_processing_config()
            default_combo_name = self._get_default_combo_name()
            available_combos = self._get_available_combos()
            
            # Create server configuration
            server_config = ServerConfig(
                run_profile=self.run_profile,
                prompts=prompts,
                api_keys=api_keys,
                processing=processing,
                default_combo_name=default_combo_name,
                available_combos=available_combos,
                desensitization=desensitization
            )

            logger.info('='*50)
            logger.info('PROMPTS:')
            logger.info(prompts)
            logger.info('='*50)
            
            # Cache the configuration
            self._cached_config = server_config
            self._assembly_time_ms = (time.time() - assembly_start) * 1000
            
            logger.info(f"✅ Server configuration assembled successfully in {self._assembly_time_ms:.3f}ms")
            logger.info(f"📊 Configuration summary:")
            logger.info(f"   - Profile: {self.run_profile}")
            logger.info(f"   - Prompts loaded: {len(prompts.source_info)}")
            logger.info(f"   - API keys loaded: {len([k for k, v in api_keys.dict().items() if v])}")
            logger.info(f"   - Default combo: {default_combo_name}")
            logger.info(f"   - Available combos: {len(available_combos)}")
            
            return server_config
            
        except Exception as e:
            logger.error(f"❌ Failed to assemble server configuration: {e}")
            raise
    
    def get_assembly_time_ms(self) -> float:
        """
        Get the time taken to assemble the configuration.
        
        Returns:
            float: Assembly time in milliseconds
        """
        return self._assembly_time_ms
    
    def is_cached(self) -> bool:
        """
        Check if configuration is cached.
        
        Returns:
            bool: True if configuration is cached
        """
        return self._cached_config is not None
    
    def clear_cache(self) -> None:
        """Clear the cached configuration."""
        self._cached_config = None
        self._assembly_time_ms = 0.0
        logger.info("🗑️ Server configuration cache cleared")
    
    def inject_into_config_base(self) -> None:
        """
        Inject the assembled configuration into config_base for backward compatibility.
        
        This method is used to maintain compatibility with the existing main library
        that expects configuration to be injected into config_base.
        """
        if self._cached_config is None:
            logger.warning("⚠️ No cached configuration to inject")
            return
        
        try:
            import config.config_base as config_base
            
            # Inject prompts
            config_base.SYSTEM_PROMPT = self._cached_config.prompts.system_prompt
            config_base.USER_PROMPT = self._cached_config.prompts.user_prompt
            config_base.SIMPLIFIED_USER_PROMPT = self._cached_config.prompts.simplified_user_prompt
            config_base.JSON_FORMAT_INSTRUCTIONS = self._cached_config.prompts.json_format_instructions
            config_base.MANDATORY_KEYS = self._cached_config.prompts.mandatory_keys
            config_base.TEXT_FIRST_REGEX_CRITERIA = self._cached_config.prompts.text_first_regex_criteria
            
            # Inject API keys
            config_base.GCP_API_KEY = self._cached_config.api_keys.gcp_api_key
            config_base.OPENAI_API_KEY = self._cached_config.api_keys.openai_api_key
            config_base.DEEPSEEK_API_KEY = self._cached_config.api_keys.deepseek_api_key
            config_base.CLAUDE_API_KEY = self._cached_config.api_keys.claude_api_key
            config_base.HUGGINGFACE_TOKEN = self._cached_config.api_keys.huggingface_token
            config_base.TOGETHERAI_API_KEY = self._cached_config.api_keys.togetherai_api_key
            config_base.XAI_API_KEY = self._cached_config.api_keys.xai_api_key
            
            # Inject processing config
            config_base.DEFAULT_STREAMING = self._cached_config.processing.streaming
            config_base.DEFAULT_MAX_CC_STRATEGIES = self._cached_config.processing.max_cc_strategies
            config_base.DEFAULT_MAX_CC_FILEGROUPS = self._cached_config.processing.max_cc_filegroups
            config_base.DEFAULT_MAX_FILES_PER_REQUEST = self._cached_config.processing.max_files_per_request
            
            logger.info("✅ Configuration injected into config_base successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to inject configuration into config_base: {e}")
            raise
