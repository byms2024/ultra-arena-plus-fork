"""
Configuration Models for Ultra Arena RESTful API

This module defines Pydantic models for configuration validation and type safety.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, validator
from enum import Enum


class ConfigSource(str, Enum):
    """Enumeration of configuration sources for tracking."""
    SYSTEM_DEFAULT = "system_default"
    PROFILE_DEFAULT = "profile_default"
    REQUEST_OVERRIDE = "request_override"
    TEST_FIXTURE = "test_fixture"
    CLI_OVERRIDE = "cli_override"
    DIRECT_TEST_OVERRIDE = "direct_test_override"


class DesensitizationConfig(BaseModel):
    """Static sensitive data configuration"""

    desensitization_config: bool = Field(description="Whether the information will be desensitized", default=False)
    sensitive_fields: list = Field(description="List of sensitive informatons")

    class Config:
        frozen = True


class PromptConfig(BaseModel):
    """Configuration for LLM prompts with source tracking."""
    system_prompt: str = Field(default="", description="System prompt for LLM")
    user_prompt: str = Field(default="", description="User prompt for LLM")
    simplified_user_prompt: str = Field(default="", description="Simplified user prompt for local models")
    json_format_instructions: str = Field(default="", description="JSON formatting instructions")
    mandatory_keys: List[str] = Field(default_factory=list, description="Mandatory keys for extraction")
    text_first_regex_criteria: Dict[str, Any] = Field(default_factory=dict, description="Text-first regex criteria")
    
    # Source tracking (without leading underscore for Pydantic compatibility)
    source_info: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Source tracking information")
    
    class Config:
        validate_by_name = True


class ApiKeyConfig(BaseModel):
    """Configuration for API keys."""
    gcp_api_key: str = Field(default="", description="Google Cloud Platform API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    deepseek_api_key: str = Field(default="", description="DeepSeek API key")
    claude_api_key: str = Field(default="", description="Claude API key")
    huggingface_token: str = Field(default="", description="HuggingFace token")
    togetherai_api_key: str = Field(default="", description="TogetherAI API key")
    xai_api_key: str = Field(default="", description="XAI API key")
    
    def get_masked_keys(self) -> Dict[str, str]:
        """Get API keys with masking for logging."""
        masked = {}
        for field, value in self.dict().items():
            if value and len(value) > 8:
                masked[field] = f"{value[:4]}...{value[-4:]}"
            else:
                masked[field] = "(empty)" if not value else "(masked)"
        return masked


class ProcessingConfig(BaseModel):
    """Configuration for processing parameters."""
    streaming: bool = Field(default=False, description="Enable streaming mode")
    max_cc_strategies: int = Field(default=1, description="Max concurrent strategies")
    max_cc_filegroups: int = Field(default=5, description="Max concurrent file groups")
    max_files_per_request: int = Field(default=10, description="Max files per request")
    benchmark_eval_mode: bool = Field(default=False, description="Enable benchmark evaluation")
    
    @validator('max_cc_strategies')
    def validate_max_cc_strategies(cls, v):
        if v < 1:
            raise ValueError('max_cc_strategies must be at least 1')
        return v
    
    @validator('max_cc_filegroups')
    def validate_max_cc_filegroups(cls, v):
        if v < 1:
            raise ValueError('max_cc_filegroups must be at least 1')
        return v


class ServerConfig(BaseModel):
    """Static server configuration assembled once at startup."""
    run_profile: str = Field(description="Profile name for this server instance")
    desensitization: DesensitizationConfig = Field(description="Set how to handle sensitive information")
    prompts: PromptConfig = Field(default_factory=PromptConfig, description="Profile prompt configuration")
    api_keys: ApiKeyConfig = Field(default_factory=ApiKeyConfig, description="Profile API key configuration")
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig, description="Profile processing configuration")
    default_combo_name: str = Field(default="", description="Default combo name from profile")
    available_combos: List[str] = Field(default_factory=list, description="Available combo names")
    
    class Config:
        frozen = True  # Immutable for thread safety


class RequestConfig(BaseModel):
    """Dynamic request configuration assembled per request."""
    # Request-specific overrides
    combo_name: Optional[str] = Field(default=None, description="Combo name for this request")
    input_pdf_dir_path: Optional[Path] = Field(default=None, description="Input PDF directory path")
    output_dir: Optional[Path] = Field(default=None, description="Output directory path")
    benchmark_file_path: Optional[Path] = Field(default=None, description="Benchmark file path")
    
    # Prompt overrides
    prompt_overrides: Optional[PromptConfig] = Field(default=None, description="Request-specific prompt overrides")
    
    # Processing overrides
    processing_overrides: Optional[ProcessingConfig] = Field(default=None, description="Request-specific processing overrides")
    
    # Merged final configuration (computed)
    final_prompts: PromptConfig = Field(description="Final prompt configuration after merging")
    final_processing: ProcessingConfig = Field(description="Final processing configuration after merging")
    
    class Config:
        frozen = True  # Immutable for thread safety


class ConfigAssemblyResult(BaseModel):
    """Result of configuration assembly with metadata."""
    server_config: ServerConfig = Field(description="Static server configuration")
    request_config: RequestConfig = Field(description="Dynamic request configuration")
    assembly_time_ms: float = Field(description="Time taken to assemble configuration in milliseconds")
    cache_hit: bool = Field(description="Whether static config was loaded from cache")
    
    class Config:
        frozen = True

