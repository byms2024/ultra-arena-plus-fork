"""
Base Configuration Assembler

This module provides the base class for configuration assembly with common functionality.
"""

import os
import logging
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from .config_models import ConfigSource, PromptConfig, ApiKeyConfig, ProcessingConfig, DesensitizationConfig

logger = logging.getLogger(__name__)


class BaseConfigAssembler(ABC):
    """Base class for configuration assembly with common functionality."""
    
    def __init__(self, run_profile: str):
        """
        Initialize the base config assembler.
        
        Args:
            run_profile (str): The profile name to use for configuration
        """
        self.run_profile = run_profile
        self._profile_dir = self._get_profile_dir()
    
    def _get_profile_dir(self) -> Path:
        """Get the profile directory path."""
        # This will be overridden by subclasses to point to the correct location
        raise NotImplementedError("Subclasses must implement _get_profile_dir")
    
    def _load_module_from_path(self, module_name: str, file_path: Path) -> Optional[Any]:
        """
        Load a Python module from a file path.
        
        Args:
            module_name (str): Name for the module
            file_path (Path): Path to the module file
            
        Returns:
            Optional[Any]: Loaded module or None if failed
        """
        try:
            if not file_path.exists():
                logger.warning(f"⚠️ Module file not found: {file_path}")
                return None
            
            spec = importlib.util.spec_from_file_location(module_name, str(file_path))
            if spec is None or spec.loader is None:
                logger.warning(f"⚠️ Could not create spec for module: {file_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
            
        except Exception as e:
            logger.error(f"❌ Failed to load module {module_name} from {file_path}: {e}")
            return None
    
    def _load_prompt_config(self, desensitization : bool) -> PromptConfig:
        """
        Load prompt configuration from profile.
        
        Returns:
            PromptConfig: Loaded prompt configuration
        """
        prompt_config_path = self._profile_dir / "profile_prompts_config.py"
        module = self._load_module_from_path("profile_prompts_config", prompt_config_path)
        
        if module is None:
            logger.warning(f"⚠️ Using empty prompt config - could not load from {prompt_config_path}")
            return PromptConfig()
        
        system_prompt_attr = 'SYSTEM_PROMPT'
        user_prompt_attr = 'USER_PROMPT'
        json_format_attr = 'JSON_FORMAT_INSTRUCTIONS'

        if not desensitization:
            user_prompt_attr = 'SENSITIVE_USER_PROMPT'
            json_format_attr = 'SENSITIVE_JSON_FORMAT_INSTRUCTIONS'
        
        # Extract prompt values with source tracking
        prompts = {}
        source_info = {}
        
        prompt_fields = {
            'system_prompt': system_prompt_attr,
            'user_prompt': user_prompt_attr, 
            'json_format_instructions': json_format_attr,
            'simplified_user_prompt': 'SIMPLIFIED_USER_PROMPT',
            'json_format_instructions': 'JSON_FORMAT_INSTRUCTIONS',
            'mandatory_keys': 'MANDATORY_KEYS',
            'text_first_regex_criteria': 'TEXT_FIRST_REGEX_CRITERIA'
        }
        
        for field, attr in prompt_fields.items():
            if hasattr(module, attr):
                value = getattr(module, attr)
                prompts[field] = value
                source_info[field] = {
                    "value": value,
                    "source": ConfigSource.PROFILE_DEFAULT
                }
                logger.debug(f"📝 Loaded {field} from profile")
            else:
                # Use system defaults
                system_default = self._get_system_default_prompt(field)
                prompts[field] = system_default
                source_info[field] = {
                    "value": system_default,
                    "source": ConfigSource.SYSTEM_DEFAULT
                }
                logger.debug(f"📝 Using system default for {field}")
        
        # Remove source_info from prompts to avoid duplicate
        prompts.pop('source_info', None)
        return PromptConfig(**prompts, source_info=source_info)
    
    def _load_api_key_config(self) -> ApiKeyConfig:
        """
        Load API key configuration from profile.
        
        Returns:
            ApiKeyConfig: Loaded API key configuration
        """
        api_keys_path = self._profile_dir / "config_api_keys.py"
        module = self._load_module_from_path("config_api_keys", api_keys_path)
        
        if module is None:
            logger.warning(f"⚠️ Using empty API key config - could not load from {api_keys_path}")
            return ApiKeyConfig()
        
        # Extract API keys
        api_keys = {}
        api_key_fields = {
            'gcp_api_key': 'GCP_API_KEY',
            'openai_api_key': 'OPENAI_API_KEY',
            'deepseek_api_key': 'DEEPSEEK_API_KEY',
            'claude_api_key': 'CLAUDE_API_KEY',
            'huggingface_token': 'HUGGINGFACE_TOKEN',
            'togetherai_api_key': 'TOGETHERAI_API_KEY',
            'xai_api_key': 'XAI_API_KEY'
        }
        
        for field, attr in api_key_fields.items():
            if hasattr(module, attr):
                value = getattr(module, attr) or ""
                api_keys[field] = value
                logger.debug(f"🔐 Loaded {field} from profile")
            else:
                api_keys[field] = ""
                logger.debug(f"🔐 No {field} found in profile")
        
        # Log masked keys for debugging
        config = ApiKeyConfig(**api_keys)
        masked_keys = config.get_masked_keys()
        for key_name, masked_value in masked_keys.items():
            logger.info(f"🔐 {key_name}: {masked_value}")
        
        return config
    
    def _load_processing_config(self) -> ProcessingConfig:
        """
        Load processing configuration from profile.
        
        Returns:
            ProcessingConfig: Loaded processing configuration
        """
        profile_config_path = self._profile_dir / "profile_config.py"
        module = self._load_module_from_path("profile_config", profile_config_path)
        
        if module is None:
            logger.warning(f"⚠️ Using default processing config - could not load from {profile_config_path}")
            return ProcessingConfig()
        
        # Extract processing values
        processing = {}
        processing_fields = {
            'streaming': 'DEFAULT_STREAMING',
            'max_cc_strategies': 'DEFAULT_MAX_CC_STRATEGIES',
            'max_cc_filegroups': 'DEFAULT_MAX_CC_FILEGROUPS',
            'max_files_per_request': 'DEFAULT_MAX_FILES_PER_REQUEST'
        }
        
        for field, attr in processing_fields.items():
            if hasattr(module, attr):
                value = getattr(module, attr)
                processing[field] = value
                logger.debug(f"⚙️ Loaded {field} from profile: {value}")
            else:
                logger.debug(f"⚙️ Using default for {field}")
        
        return ProcessingConfig(**processing)
    
    def _get_system_default_prompt(self, field: str) -> Any:
        """
        Get system default prompt value.
        
        Args:
            field (str): Prompt field name
            
        Returns:
            Any: System default value
        """
        system_defaults = {
            'system_prompt': "You are a specialized AI assistant for data extraction and analysis.",
            'user_prompt': "Please extract the requested information from the provided documents.",
            'json_format_instructions': "Return the extracted data in valid JSON format.",
            'mandatory_keys': ["INVOICE_NO", "TOTAL_AMOUNT", "INVOICE_ISSUE_DATE"],
            'text_first_regex_criteria': {}
        }
        
        return system_defaults.get(field, "")
    
    def _get_default_combo_name(self) -> str:
        """
        Get default combo name from profile configuration.
        
        Returns:
            str: Default combo name
        """
        profile_config_path = self._profile_dir / "profile_config.py"
        module = self._load_module_from_path("profile_config", profile_config_path)
        
        if module is None:
            logger.warning(f"⚠️ Could not load profile config for default combo name")
            return ""
        
        if hasattr(module, 'DEFAULT_STRATEGY_PARAM_GRP'):
            combo_name = module.DEFAULT_STRATEGY_PARAM_GRP
            logger.info(f"🎯 Default combo name from profile: {combo_name}")
            return combo_name
        else:
            logger.warning(f"⚠️ DEFAULT_STRATEGY_PARAM_GRP not found in profile config")
            return ""
    
    def _get_available_combos(self) -> list:
        """
        Get available combos from centralized configuration.
        
        Returns:
            list: List of available combo names
        """
        try:
            # Import the centralized combo configuration
            # Fix the path to point to the correct location
            current_dir = Path(__file__).parent.parent.parent  # server_utils/config_assemblers -> server_utils -> Ultra_Arena_Main_Restful
            ultra_arena_main_path = current_dir.parent / "Ultra_Arena_Main"
            
            logger.info(f"🔍 Looking for combo config in: {ultra_arena_main_path}")
            
            # Add Ultra_Arena_Main to Python path to handle imports
            import sys
            if str(ultra_arena_main_path) not in sys.path:
                sys.path.insert(0, str(ultra_arena_main_path))
            
            # Try to import the combo config directly as a module
            try:
                from config.config_combo_run import combo_config
                combos = list(combo_config.keys())
                logger.info(f"📋 Loaded {len(combos)} available combos via direct import: {combos[:5]}...")
                
                # Store the combo_config globally for later use
                import config.config_combo_run
                config.config_combo_run.combo_config = combo_config
                
                return combos
            except ImportError as e:
                logger.warning(f"⚠️ Direct import failed: {e}")
                
                # Fallback: try to load the file manually
                central_combo_config_path = ultra_arena_main_path / "config" / "config_combo_run.py"
                
                if not central_combo_config_path.exists():
                    logger.error(f"❌ Combo config file not found: {central_combo_config_path}")
                    return []
                
                # Load the file content and extract combo_config
                import ast
                with open(central_combo_config_path, 'r') as f:
                    content = f.read()
                
                # Try a more robust approach using exec to load the actual configuration
                try:
                    # Create a namespace to execute the file
                    namespace = {}
                    exec(content, namespace)
                    
                    if 'combo_config' in namespace:
                        combo_config = namespace['combo_config']
                        combos = list(combo_config.keys())
                        logger.info(f"📋 Loaded {len(combos)} available combos via exec: {combos[:5]}...")
                        
                        # Store the combo_config globally for later use
                        import config.config_combo_run
                        config.config_combo_run.combo_config = combo_config
                        
                        return combos
                    else:
                        logger.warning(f"⚠️ combo_config not found in namespace")
                        
                except Exception as exec_error:
                    logger.warning(f"⚠️ Exec approach failed: {exec_error}")
                
                # Final fallback: try regex for combo names only (this is the problematic fallback)
                import re
                combo_pattern = r'"([^"]+)"\s*:\s*{'
                matches = re.findall(combo_pattern, content)
                if matches:
                    combos = matches
                    logger.warning(f"⚠️ Using regex fallback - only combo names loaded, not strategy groups: {combos[:5]}...")
                    logger.warning(f"⚠️ This may cause issues with combo processing!")
                    return combos
                
                logger.error(f"❌ Could not extract combo configuration from file")
                return []
            
        except Exception as e:
            logger.error(f"❌ Failed to load combo configuration: {e}")
            return []
    
    def _load_desensitization_config(self) -> DesensitizationConfig:
        """
        Load desensitization configuration from profile.
        
        Returns:
            ProcessingConfig: Loaded processing configuration
        """
    
        desensitize_config_path = self._profile_dir / "profile_config.py"
        module = self._load_module_from_path("profile_config", desensitize_config_path)
        
        if module is None:
            logger.warning(f"⚠️ Using empty desensitization config - could not load from {desensitize_config_path}")
            return DesensitizationConfig()
        
        # Extract prompt values with source tracking
        desensitization = getattr(module, 'DESENSITIZATION')
        field_list = getattr(module, 'SENSITIVE_INFO')
        
        return DesensitizationConfig(desensitization_config=desensitization, sensitive_fields=field_list)




    @abstractmethod
    def assemble_config(self) -> Any:
        """
        Assemble configuration. Must be implemented by subclasses.
        
        Returns:
            Any: Assembled configuration object
        """
        pass
