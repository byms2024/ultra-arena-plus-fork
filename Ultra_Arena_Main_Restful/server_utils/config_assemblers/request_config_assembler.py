"""
Request Configuration Assembler

This module provides the RequestConfigAssembler for assembling dynamic request configuration
per request, which merges static server configuration with request-specific overrides.
"""

import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

from .config_models import (
    ServerConfig, RequestConfig, PromptConfig, ProcessingConfig, ConfigSource, ConfigAssemblyResult
)

logger = logging.getLogger(__name__)


class RequestConfigAssembler:
    """Assembles dynamic request configuration per request."""
    
    def __init__(self, server_config: ServerConfig):
        """
        Initialize the request config assembler.
        
        Args:
            server_config (ServerConfig): The static server configuration
        """
        self.server_config = server_config
    
    def assemble_request_config(self, request_data: Dict[str, Any], use_default_combo: bool = False) -> ConfigAssemblyResult:
        """
        Assemble request configuration from request data.
        
        Args:
            request_data (Dict[str, Any]): Raw request data from the endpoint
            use_default_combo (bool): If True, use default combo and force max_cc_strategies=1
            
        Returns:
            ConfigAssemblyResult: Assembled configuration with metadata
        """
        assembly_start = time.time()
        
        try:
            logger.debug(f"🔧 Assembling request configuration")
            
            # Extract request-specific overrides
            request_overrides = self._extract_request_overrides(request_data, use_default_combo)
            
            # Merge configurations
            final_prompts = self._merge_prompt_configs(request_overrides.get('prompt_overrides'))
            final_processing = self._merge_processing_configs(request_overrides.get('processing_overrides'))
            
            # Create request configuration
            request_config = RequestConfig(
                combo_name=request_overrides.get('combo_name'),
                chain_name=request_overrides.get('chain_name'),
                input_pdf_dir_path=request_overrides.get('input_pdf_dir_path'),
                output_dir=request_overrides.get('output_dir'),
                benchmark_file_path=request_overrides.get('benchmark_file_path'),
                prompt_overrides=request_overrides.get('prompt_overrides'),
                processing_overrides=request_overrides.get('processing_overrides'),
                final_prompts=final_prompts,
                final_processing=final_processing
            )
            
            # Calculate assembly time
            assembly_time_ms = (time.time() - assembly_start) * 1000
            
            # Create result
            result = ConfigAssemblyResult(
                server_config=self.server_config,
                request_config=request_config,
                assembly_time_ms=assembly_time_ms,
                cache_hit=True  # Server config is always cached
            )
            
            logger.debug(f"✅ Request configuration assembled in {assembly_time_ms:.3f}ms")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to assemble request configuration: {e}")
            raise
    
    def _extract_request_overrides(self, request_data: Dict[str, Any], use_default_combo: bool) -> Dict[str, Any]:
        """
        Extract request-specific overrides from request data.
        
        Args:
            request_data (Dict[str, Any]): Raw request data
            use_default_combo (bool): If True, use default combo
            
        Returns:
            Dict[str, Any]: Extracted request overrides
        """
        overrides = {}
        
        # Handle chain first (takes precedence over combo if provided)
        if 'chain_name' in request_data and request_data['chain_name']:
            overrides['chain_name'] = request_data['chain_name']
            logger.debug(f"🔗 Using request chain: {overrides['chain_name']}")

        # Handle combo name
        if use_default_combo:
            overrides['combo_name'] = self.server_config.default_combo_name
            logger.debug(f"🎯 Using default combo: {overrides['combo_name']}")
        else:
            overrides['combo_name'] = request_data.get('combo_name')
            if overrides['combo_name']:
                logger.debug(f"🎯 Using request combo: {overrides['combo_name']}")
            else:
                overrides['combo_name'] = self.server_config.default_combo_name
                logger.debug(f"🎯 No combo specified, using default: {overrides['combo_name']}")
        
        # Handle paths
        if 'input_pdf_dir_path' in request_data:
            overrides['input_pdf_dir_path'] = Path(request_data['input_pdf_dir_path'])
        if 'output_dir' in request_data:
            overrides['output_dir'] = Path(request_data['output_dir'])
        if 'benchmark_file_path' in request_data:
            overrides['benchmark_file_path'] = Path(request_data['benchmark_file_path'])
        
        # Handle prompt overrides
        if 'prompt' in request_data and isinstance(request_data['prompt'], dict):
            overrides['prompt_overrides'] = self._extract_prompt_overrides(request_data['prompt'])
        
        # Handle processing overrides
        processing_overrides = {}
        processing_fields = ['streaming', 'max_cc_strategies', 'max_cc_filegroups', 'max_files_per_request']
        
        for field in processing_fields:
            if field in request_data:
                processing_overrides[field] = request_data[field]
        
        # Special handling for use_default_combo
        if use_default_combo:
            processing_overrides['max_cc_strategies'] = 1
            logger.debug("🎯 Simple processing: forcing max_cc_strategies=1")
        
        if processing_overrides:
            overrides['processing_overrides'] = ProcessingConfig(**processing_overrides)
        
        return overrides
    
    def _extract_prompt_overrides(self, prompt_data: Dict[str, Any]) -> PromptConfig:
        """
        Extract prompt overrides from request data.
        
        Args:
            prompt_data (Dict[str, Any]): Prompt data from request
            
        Returns:
            PromptConfig: Extracted prompt overrides with source tracking
        """
        prompts = {}
        source_info = {}
        
        prompt_fields = {
            'system_prompt': 'system_prompt',
            'user_prompt': 'user_prompt',
            'json_format_instructions': 'json_format_instructions',
            'mandatory_keys': 'mandatory_keys',
            'text_first_regex_criteria': 'text_first_regex_criteria'
        }
        
        for field, key in prompt_fields.items():
            if key in prompt_data:
                value = prompt_data[key]
                prompts[field] = value
                source_info[field] = {
                    "value": value,
                    "source": ConfigSource.REQUEST_OVERRIDE
                }
                logger.debug(f"📝 Request override for {field}")
        
        # Remove source_info from prompts to avoid duplicate
        prompts.pop('source_info', None)
        return PromptConfig(**prompts, source_info=source_info)
    
    def _merge_prompt_configs(self, request_prompts: Optional[PromptConfig]) -> PromptConfig:
        """
        Merge server prompts with request overrides.
        
        Args:
            request_prompts (Optional[PromptConfig]): Request-specific prompt overrides
            
        Returns:
            PromptConfig: Merged prompt configuration
        """
        # Start with server prompts
        merged_prompts = self.server_config.prompts.model_dump()
        merged_source_info = self.server_config.prompts.source_info.copy()
        
        # Apply request overrides if provided
        if request_prompts:
            for field, value in request_prompts.model_dump().items():
                if field != 'source_info' and value is not None:
                    merged_prompts[field] = value
                    merged_source_info[field] = {
                        "value": value,
                        "source": ConfigSource.REQUEST_OVERRIDE
                    }
                    logger.debug(f"📝 Applied request override for {field}")
        
        # Remove source_info from merged_prompts to avoid duplicate
        merged_prompts.pop('source_info', None)
        return PromptConfig(**merged_prompts, source_info=merged_source_info)
    
    def _merge_processing_configs(self, request_processing: Optional[ProcessingConfig]) -> ProcessingConfig:
        """
        Merge server processing config with request overrides.
        
        Args:
            request_processing (Optional[ProcessingConfig]): Request-specific processing overrides
            
        Returns:
            ProcessingConfig: Merged processing configuration
        """
        # Start with server processing config
        merged_processing = self.server_config.processing.dict()
        
        # Apply request overrides if provided
        if request_processing:
            for field, value in request_processing.dict().items():
                if value is not None:
                    merged_processing[field] = value
                    logger.debug(f"⚙️ Applied request override for {field}: {value}")
        
        return ProcessingConfig(**merged_processing)
    
    def log_configuration_summary(self, result: ConfigAssemblyResult) -> None:
        """
        Log a summary of the assembled configuration.
        
        Args:
            result (ConfigAssemblyResult): The assembled configuration result
        """
        request_config = result.request_config
        
        logger.info(f"📊 REQUEST CONFIGURATION SUMMARY:")
        logger.info(f"   {'='*60}")
        logger.info(f"   Combo Name: {request_config.combo_name}")
        logger.info(f"   Input Path: {request_config.input_pdf_dir_path}")
        logger.info(f"   Output Dir: {request_config.output_dir}")
        logger.info(f"   Benchmark: {request_config.benchmark_file_path}")
        logger.info(f"   Streaming: {request_config.final_processing.streaming}")
        logger.info(f"   Max CC Strategies: {request_config.final_processing.max_cc_strategies}")
        logger.info(f"   Max CC Filegroups: {request_config.final_processing.max_cc_filegroups}")
        logger.info(f"   Max Files Per Request: {request_config.final_processing.max_files_per_request}")
        
        # Log prompt sources
        if request_config.final_prompts.source_info:
            logger.info(f"   📝 PROMPT SOURCES:")
            for field, info in request_config.final_prompts.source_info.items():
                source = info.get('source', 'unknown')
                logger.info(f"     {field}: {source}")
        
        logger.info(f"   Assembly Time: {result.assembly_time_ms:.3f}ms")
        logger.info(f"   {'='*60}")
    
    def inject_final_config_into_base(self, result: ConfigAssemblyResult) -> None:
        """
        Inject the final configuration into config_base for backward compatibility.
        
        Args:
            result (ConfigAssemblyResult): The assembled configuration result
        """
        try:
            import importlib
            from typing import Any, cast
            config_base = importlib.import_module('config.config_base')
            cb = cast(Any, config_base)
            
            final_prompts = result.request_config.final_prompts
            final_processing = result.request_config.final_processing
            
            # Inject final prompts
            cb.SYSTEM_PROMPT = final_prompts.system_prompt
            cb.USER_PROMPT = final_prompts.user_prompt
            cb.JSON_FORMAT_INSTRUCTIONS = final_prompts.json_format_instructions
            cb.MANDATORY_KEYS = final_prompts.mandatory_keys
            cb.TEXT_FIRST_REGEX_CRITERIA = final_prompts.text_first_regex_criteria
            
            # Inject final processing config
            cb.DEFAULT_STREAMING = final_processing.streaming
            cb.DEFAULT_MAX_CC_STRATEGIES = final_processing.max_cc_strategies
            cb.DEFAULT_MAX_CC_FILEGROUPS = final_processing.max_cc_filegroups
            cb.DEFAULT_MAX_FILES_PER_REQUEST = final_processing.max_files_per_request
            
            logger.debug("✅ Final configuration injected into config_base")
            
        except Exception as e:
            logger.error(f"❌ Failed to inject final configuration into config_base: {e}")
            raise
