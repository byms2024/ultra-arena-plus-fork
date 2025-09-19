#!/usr/bin/env python3
"""
Main script for the modular PDF processing system.

This script demonstrates how to use the new modular processing system
with different strategies and configurations.
"""

import argparse
import copy
import logging
import os
import sys
import time
import json
import warnings
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Suppress the specific warning about BATCH_STATE_RUNNING from Google GenAI SDK
warnings.filterwarnings("ignore", message="BATCH_STATE_RUNNING is not a valid JobState")

# Import configurations
from config.config_base import (
    # NOTE: SYSTEM_PROMPT and USER_PROMPT are accessed dynamically via config_base.SYSTEM_PROMPT
    # to ensure we get the current injected values, not stale imports from module load time
    DEFAULT_STRATEGY_TYPE, DEFAULT_MODE, DEFAULT_MAX_CC_FILEGROUPS,
    DEFAULT_OUTPUT_FILE, DEFAULT_CHECKPOINT_FILE,
    STRATEGY_DIRECT_FILE, STRATEGY_TEXT_FIRST, STRATEGY_IMAGE_FIRST, STRATEGY_HYBRID, STRATEGY_CHAIN,
    MODE_PARALLEL, MODE_BATCH,
    # Direct file strategy configurations
    DIRECT_FILE_PROVIDER_CONFIGS, DIRECT_FILE_MAX_RETRIES, DIRECT_FILE_RETRY_DELAY_SECONDS,
    # Text first strategy configurations
    LOCAL_LLM_PROVIDER, PDF_EXTRACTOR_LIB, SECONDARY_PDF_EXTRACTOR_LIB, 
    TEXT_FIRST_REGEX_CRITERIA, TEXT_PROVIDER_CONFIGS,
    # Image first strategy configurations
    PDF_TO_IMAGE_DPI, PDF_TO_IMAGE_FORMAT, PDF_TO_IMAGE_QUALITY, IMAGE_PROVIDER_CONFIGS,
    # HuggingFace strategy configurations
    HUGGINGFACE_PROVIDER_CONFIG, HUGGINGFACE_MODEL_CONFIGS
)
import config.config_base as config_base

# Import the modular processor
from processors.modular_parallel_processor import ModularParallelProcessor

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    from logging_utils import setup_logging as setup_enhanced_logging
    setup_enhanced_logging(
        level='DEBUG' if verbose else 'INFO',
        log_file='modular_processing.log',
        use_thread_function_format=True,
        verbose=verbose,
        suppress_http_logs=True
    )


def get_max_available_threads() -> int:
    """Get the maximum number of available CPU threads from the OS."""
    try:
        # Get the number of CPU cores/threads available
        max_threads = multiprocessing.cpu_count()
        logging.info(f"🖥️ Detected {max_threads} CPU threads available")
        return max_threads
    except Exception as e:
        logging.warning(f"⚠️ Could not detect CPU count, using fallback value: {e}")
        return 8  # Fallback to a reasonable default


def parse_max_cc_strategies(value: str) -> int:
    """Parse the max concurrent strategies argument, handling 'max' keyword."""
    if value.lower() == 'max':
        return get_max_available_threads()
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value '{value}'. Use 'max' or an integer.")

def get_pdf_files(input_path: str) -> List[str]:
    """Get list of PDF files from input path (file or directory)."""
    input_path = Path(input_path)
    
    if input_path.is_file():
        if input_path.suffix.lower() == '.pdf':
            return [str(input_path)]
        else:
            raise ValueError(f"Input file {input_path} is not a PDF")
    
    elif input_path.is_dir():
        pdf_files = []
        for file_path in input_path.rglob("*.pdf"):
            pdf_files.append(str(file_path))
        return sorted(pdf_files)
    
    else:
        raise ValueError(f"Input path {input_path} does not exist")


def generate_timestamped_filename(strategy: str, mode: str, llm_provider: str, llm_model: str, extension: str = "json") -> str:
    """Generate a timestamped filename with the specified format."""
    timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
    # Clean the model name for filename (remove special characters)
    clean_model = llm_model.replace(":", "_").replace("/", "_").replace("-", "_")
    return f"{strategy}_{mode}_{llm_provider}_{clean_model}_{timestamp}.{extension}"


def get_config_for_strategy(strategy_type: str, llm_provider: str = None, llm_model: str = None, streaming: bool = False, output_dir: str = None) -> Dict[str, Any]:
    """
    Get configuration for a specific strategy type.
    
    Args:
        strategy_type (str): The strategy type
        llm_provider (str, optional): Override LLM provider
        llm_model (str, optional): Override LLM model
        streaming (bool): Whether to use streaming mode
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    if strategy_type == STRATEGY_DIRECT_FILE:
        config = {
            "llm_provider": llm_provider or config_base.DEFAULT_LLM_PROVIDER,
            "provider_configs": DIRECT_FILE_PROVIDER_CONFIGS,
            "mandatory_keys": config_base.MANDATORY_KEYS,
            "num_retry_for_mandatory_keys": config_base.NUM_RETRY_FOR_MANDATORY_KEYS,
            "max_num_files_per_request": config_base.MAX_NUM_FILES_PER_REQUEST,
            "max_num_file_parts_per_batch": config_base.MAX_NUM_FILE_PARTS_PER_BATCH,
            "max_retries": config_base.API_INFRA_MAX_RETRIES,
            "retry_delay_seconds": config_base.API_INFRA_RETRY_DELAY_SECONDS
        }
        # Refresh provider API keys from current config_base values (profile-injected)
        try:
            if "google" in config["provider_configs"]:
                config["provider_configs"]["google"]["api_key"] = getattr(config_base, "GCP_API_KEY", "")
            if "openai" in config["provider_configs"]:
                config["provider_configs"]["openai"]["api_key"] = getattr(config_base, "OPENAI_API_KEY", "")
            if "deepseek" in config["provider_configs"]:
                config["provider_configs"]["deepseek"]["api_key"] = getattr(config_base, "DEEPSEEK_API_KEY", "")
            if "claude" in config["provider_configs"]:
                config["provider_configs"]["claude"]["api_key"] = getattr(config_base, "CLAUDE_API_KEY", "")
            if "huggingface" in config["provider_configs"]:
                config["provider_configs"]["huggingface"]["api_key"] = getattr(config_base, "HUGGINGFACE_TOKEN", "")
            if "togetherai" in config["provider_configs"]:
                config["provider_configs"]["togetherai"]["api_key"] = getattr(config_base, "TOGETHERAI_API_KEY", "")
            if "grok" in config["provider_configs"]:
                config["provider_configs"]["grok"]["api_key"] = getattr(config_base, "XAI_API_KEY", "")
        except Exception as _e:
            logging.debug(f"Key refresh skipped: {_e}")
        # Override provider and model if specified
        if llm_provider:
            config["llm_provider"] = llm_provider
        if llm_model and llm_provider in config["provider_configs"]:
            # Create a deep copy to avoid race condition with shared provider_configs
            config["provider_configs"] = copy.deepcopy(config["provider_configs"])
            config["provider_configs"][llm_provider]["model"] = llm_model
        
        # Add streaming configuration to provider configs
        if streaming and llm_provider in config["provider_configs"]:
            config["provider_configs"][llm_provider]["streaming"] = streaming
        
        return config
    elif strategy_type == STRATEGY_TEXT_FIRST:
        config = {
            "llm_provider": llm_provider or LOCAL_LLM_PROVIDER,
            "provider_configs": TEXT_PROVIDER_CONFIGS,
            "pdf_extractor_lib": PDF_EXTRACTOR_LIB,
            "secondary_pdf_extractor_lib": SECONDARY_PDF_EXTRACTOR_LIB,
            "text_first_regex_criteria": TEXT_FIRST_REGEX_CRITERIA,
            "max_text_length": config_base.MAX_TEXT_LENGTH,
            "mandatory_keys": config_base.MANDATORY_KEYS,
            "num_retry_for_mandatory_keys": config_base.NUM_RETRY_FOR_MANDATORY_KEYS,
            "max_num_files_per_request": config_base.MAX_NUM_FILES_PER_REQUEST,
            "max_num_file_parts_per_batch": config_base.MAX_NUM_FILE_PARTS_PER_BATCH,
            "max_retries": config_base.API_INFRA_MAX_RETRIES,
            "retry_delay_seconds": config_base.API_INFRA_RETRY_DELAY_SECONDS
        }
        # Refresh provider API keys for text providers
        try:
            if "google" in config["provider_configs"]:
                config["provider_configs"]["google"]["api_key"] = getattr(config_base, "GCP_API_KEY", "")
            if "openai" in config["provider_configs"]:
                config["provider_configs"]["openai"]["api_key"] = getattr(config_base, "OPENAI_API_KEY", "")
            if "deepseek" in config["provider_configs"]:
                config["provider_configs"]["deepseek"]["api_key"] = getattr(config_base, "DEEPSEEK_API_KEY", "")
            if "claude" in config["provider_configs"]:
                config["provider_configs"]["claude"]["api_key"] = getattr(config_base, "CLAUDE_API_KEY", "")
        except Exception as _e:
            logging.debug(f"Key refresh skipped: {_e}")
        # Override provider and model if specified
        if llm_provider:
            config["llm_provider"] = llm_provider
        if llm_model and llm_provider in config["provider_configs"]:
            # Create a deep copy to avoid race condition with shared provider_configs
            config["provider_configs"] = copy.deepcopy(config["provider_configs"])
            config["provider_configs"][llm_provider]["model"] = llm_model
        return config
    elif strategy_type == STRATEGY_IMAGE_FIRST:
        config = {
            "llm_provider": llm_provider or config_base.DEFAULT_LLM_PROVIDER,
            "provider_configs": IMAGE_PROVIDER_CONFIGS,
            "pdf_to_image_dpi": PDF_TO_IMAGE_DPI,
            "pdf_to_image_format": PDF_TO_IMAGE_FORMAT,
            "pdf_to_image_quality": PDF_TO_IMAGE_QUALITY,
            "mandatory_keys": config_base.MANDATORY_KEYS,
            "num_retry_for_mandatory_keys": config_base.NUM_RETRY_FOR_MANDATORY_KEYS,
            "max_num_files_per_request": config_base.MAX_NUM_FILES_PER_REQUEST,
            "max_num_file_parts_per_batch": config_base.MAX_NUM_FILE_PARTS_PER_BATCH,
            "max_retries": config_base.API_INFRA_MAX_RETRIES,
            "retry_delay_seconds": config_base.API_INFRA_RETRY_DELAY_SECONDS,
            "output_dir": output_dir,
            "max_file_size_mb": config_base.MAX_FILE_SIZE_MB
        }
        # Refresh provider API keys for image providers
        try:
            if "google" in config["provider_configs"]:
                config["provider_configs"]["google"]["api_key"] = getattr(config_base, "GCP_API_KEY", "")
            if "openai" in config["provider_configs"]:
                config["provider_configs"]["openai"]["api_key"] = getattr(config_base, "OPENAI_API_KEY", "")
            if "deepseek" in config["provider_configs"]:
                config["provider_configs"]["deepseek"]["api_key"] = getattr(config_base, "DEEPSEEK_API_KEY", "")
            if "claude" in config["provider_configs"]:
                config["provider_configs"]["claude"]["api_key"] = getattr(config_base, "CLAUDE_API_KEY", "")
            if "huggingface" in config["provider_configs"]:
                config["provider_configs"]["huggingface"]["api_key"] = getattr(config_base, "HUGGINGFACE_TOKEN", "")
            if "togetherai" in config["provider_configs"]:
                config["provider_configs"]["togetherai"]["api_key"] = getattr(config_base, "TOGETHERAI_API_KEY", "")
            if "grok" in config["provider_configs"]:
                config["provider_configs"]["grok"]["api_key"] = getattr(config_base, "XAI_API_KEY", "")
        except Exception as _e:
            logging.debug(f"Key refresh skipped: {_e}")
        # Override provider and model if specified
        if llm_provider:
            config["llm_provider"] = llm_provider
        if llm_model and llm_provider in config["provider_configs"]:
            # Create a deep copy to avoid race condition with shared provider_configs
            config["provider_configs"] = copy.deepcopy(config["provider_configs"])
            config["provider_configs"][llm_provider]["model"] = llm_model
        return config
    elif strategy_type == STRATEGY_HYBRID:
        # Combine both configurations for hybrid strategy
        direct_config = get_config_for_strategy(STRATEGY_DIRECT_FILE, streaming=streaming)
        text_config = get_config_for_strategy(STRATEGY_TEXT_FIRST, streaming=streaming)
        # Merge configurations, preferring direct_file settings for conflicts
        hybrid_config = {**text_config, **direct_config}
        return hybrid_config
    elif strategy_type == STRATEGY_CHAIN:
        # Pass-through for chain: caller must provide chain_steps and flags in higher-level param group
        config = {
            "llm_provider": llm_provider or config_base.DEFAULT_LLM_PROVIDER,
            "provider_configs": IMAGE_PROVIDER_CONFIGS | TEXT_PROVIDER_CONFIGS | DIRECT_FILE_PROVIDER_CONFIGS,
            "mandatory_keys": config_base.MANDATORY_KEYS,
            "num_retry_for_mandatory_keys": config_base.NUM_RETRY_FOR_MANDATORY_KEYS,
            "max_num_files_per_request": config_base.MAX_NUM_FILES_PER_REQUEST,
            "max_num_file_parts_per_batch": config_base.MAX_NUM_FILE_PARTS_PER_BATCH,
            "max_retries": config_base.API_INFRA_MAX_RETRIES,
            "retry_delay_seconds": config_base.API_INFRA_RETRY_DELAY_SECONDS,
            # Chain-specific fields expected to be injected by param groups or API layer
            "chain_steps": getattr(config_base, "CHAIN_STEPS", []),
            "chain_on_missing_keys": getattr(config_base, "CHAIN_ON_MISSING_KEYS", False),
        }
        return config
    else:
        raise ValueError(f"Unsupported strategy type: {strategy_type}")


def run_file_processing_full(*,config_manager, input_pdf_dir_path: Path, pdf_file_paths: List[Path] = [], 
                       strategy_type: str = STRATEGY_DIRECT_FILE, mode: str = MODE_PARALLEL,
                       system_prompt: Optional[str] = None, user_prompt: Optional[str] = None,
                       max_workers: int = 5, output_file: str = "modular_results.json",
                       checkpoint_file: str = "modular_checkpoint.pkl", 
                       llm_provider: str = None, llm_model: str = None,
                       csv_output_file: str = None, benchmark_eval_mode: bool = False,
                       streaming: bool = False, max_files_per_request: int = None,
                       benchmark_file_path: Path = None) -> Dict[str, Any]:
    """
    Full-featured synchronous main entry point for batch processing using the modular system.
    
    Args:
        input_pdf_dir_path (Path): Directory containing PDF files to process (used only if pdf_file_paths is empty)
        pdf_file_paths (List[Path]): Optional list of specific PDF file paths to process
        strategy_type (str): Processing strategy - STRATEGY_DIRECT_FILE, STRATEGY_TEXT_FIRST, or STRATEGY_HYBRID
        mode (str): Processing mode - MODE_PARALLEL or MODE_BATCH
        system_prompt (Optional[str]): Optional system prompt for LLM configuration
        user_prompt (Optional[str]): User prompt for processing (if None, uses strategy default)
        max_workers (int): Maximum number of concurrent workers
        output_file (str): Output file path for results
        checkpoint_file (str): Checkpoint file path
    
    Returns:
        Dict[str, Any]: Structured output containing processing results and statistics
    """
    overall_start_time = time.time()
    
    # Log the function parameters
    logging.info(f"🚀 Starting modular batch processing with parameters:")
    logging.info(f"   - Input Directory: {input_pdf_dir_path}")
    logging.info(f"   - Strategy Type: {strategy_type}")
    logging.info(f"   - Mode: {mode}")
    logging.info(f"   - PDF File Paths provided: {len(pdf_file_paths)}")
    logging.info(f"   - Max Workers: {max_workers}")
    logging.info(f"   - Output File: {output_file}")
    logging.info(f"   - System Prompt provided: {system_prompt}")
    logging.info(f"   - User Prompt provided: {user_prompt}")
    
    # Setup logging if not already configured
    from logging_utils import setup_logging as setup_enhanced_logging
    setup_enhanced_logging(
        level='INFO',
        use_thread_function_format=True,
        suppress_http_logs=True
    )
    
    # Determine which files to process
    logging.info("🔍 Determining files to process...")
    if not pdf_file_paths:
        input_path_obj = Path(input_pdf_dir_path) if isinstance(input_pdf_dir_path, str) else input_pdf_dir_path
        if not input_path_obj.exists():
            raise FileNotFoundError(f"The specified path '{input_pdf_dir_path}' does not exist.")
        # Get all PDFs in the directory and its subdirectories
        pdf_file_paths = list(input_path_obj.rglob("*.pdf"))
        logging.info(f"Found {len(pdf_file_paths)} PDF files in directory and subdirectories")
    
    # Convert to string paths for processing
    pdf_files = [str(path) for path in pdf_file_paths if str(path).endswith('.pdf')]
    logging.info(f"Processing {len(pdf_files)} PDF files using {strategy_type} strategy in {mode} mode")
    
    if not pdf_files:
        logging.error(f"No PDF files found to process")
        return {}
    
    try:
        # Get configuration
        logging.info(f"📋 Getting configuration for strategy: {strategy_type}")
        # Create temp_images directory as sibling to json/csv folders
        # Use the base output directory (parent of json folder) for temp_images
        temp_images_dir = str(Path(output_file).parent.parent / "temp_images") if output_file else None
        config = get_config_for_strategy(strategy_type, llm_provider=llm_provider, llm_model=llm_model, streaming=streaming, output_dir=temp_images_dir)
        logging.info(f"✅ Configuration loaded successfully")
        
        # Override max_files_per_request if specified
        if max_files_per_request is not None:
            config["max_num_files_per_request"] = max_files_per_request
            logging.info(f"📊 Overriding max_files_per_request to: {max_files_per_request}")
        
        # Determine actual LLM provider and model being used
        if strategy_type == STRATEGY_DIRECT_FILE:
            actual_llm_provider = config.get("llm_provider", "unknown")
            actual_llm_model = config.get("provider_configs", {}).get(actual_llm_provider, {}).get("model", "unknown")
        else:
            actual_llm_provider = config.get("llm_provider", "unknown")
            actual_llm_model = config.get("provider_configs", {}).get(actual_llm_provider, {}).get("model", "unknown")
        
        # Create run settings dictionary
        run_settings = {
            'strategy': strategy_type,
            'mode': mode,
            'llm_provider': actual_llm_provider,
            'llm_model': actual_llm_model
        }
        
        # Initialize benchmark comparator if benchmark_eval_mode is enabled
        benchmark_comparator = None
        if benchmark_eval_mode:
            try:
                from common.benchmark_comparator import BenchmarkComparator
                logging.info(f"🔍 DEBUG: benchmark_file_path = {benchmark_file_path} (type: {type(benchmark_file_path)})")
                if benchmark_file_path:
                    benchmark_comparator = BenchmarkComparator(str(benchmark_file_path))
                    logging.info(f"🔍 Benchmark comparison enabled with file: {benchmark_file_path}")
                else:
                    logging.warning(f"⚠️ Benchmark evaluation mode enabled but no benchmark file path provided")
                    benchmark_comparator = None
            except Exception as e:
                logging.error(f"❌ Failed to initialize benchmark comparator: {e}")
                benchmark_comparator = None
        
        # Create processor
        logging.info(f"🔧 Creating ModularParallelProcessor...")
        processor = ModularParallelProcessor(
            config_manager = config_manager,
            config=config,
            strategy_type=strategy_type,
            mode=mode,
            max_workers=max_workers,
            checkpoint_file=checkpoint_file,
            output_file=output_file,
            real_time_save=True,
            run_settings=run_settings,
            csv_output_file=csv_output_file,
            benchmark_comparator=benchmark_comparator,
            streaming=streaming
        )
        logging.info(f"✅ Processor created successfully")
        
        # Process files
        logging.info(f"🚀 Starting processing with strategy: {strategy_type}, mode: {mode}")
        logging.info(f"📁 Files to process: {pdf_files}")
        results = processor.process_files(pdf_files=pdf_files, system_prompt=system_prompt, user_prompt=user_prompt)
        logging.info(f"✅ Processing completed, got structured output with keys: {list(results.keys())}")
        
        # Note: Benchmark comparison is already handled within the processor
        # The processor.process_files() method includes benchmark comparison for all processed files
        logging.info(f"📊 Benchmark comparison completed within processor")
        
        # Print summary
        # Summary is already printed by the processor
        
        logging.info(f"✅ Processing complete! Results saved to: {output_file}")
        return results
        
    except Exception as e:
        logging.error(f"❌ Processing failed: {e}", exc_info=True)
        raise


def run_file_processing(*,config_manager, input_pdf_dir_path: Path, pdf_file_paths: List[Path] = [],
                      strategy_type: str = DEFAULT_STRATEGY_TYPE, mode: str = DEFAULT_MODE,
                      system_prompt: Optional[str] = None, user_prompt: Optional[str] = None,
                      max_workers: int = DEFAULT_MAX_CC_FILEGROUPS, output_file: str = DEFAULT_OUTPUT_FILE,
                      checkpoint_file: str = DEFAULT_CHECKPOINT_FILE,
                      llm_provider: str = None, llm_model: str = None,
                      csv_output_file: str = None, benchmark_eval_mode: bool = False,
                      streaming: bool = False, max_files_per_request: int = None,
                      benchmark_file_path: Path = None) -> Dict[str, Any]:
    """
    Simplified wrapper function for file processing with default values from config_base.py.
    
    This function provides a simpler interface to run_file_processing_full() by using
    default values for unspecified parameters from config_base.py.
    
    Args:
        input_pdf_dir_path (Path): Directory containing PDF files to process (used only if pdf_file_paths is empty)
        pdf_file_paths (List[Path]): Optional list of specific PDF file paths to process
        strategy_type (str): Processing strategy (default: DEFAULT_STRATEGY_TYPE)
        mode (str): Processing mode (default: DEFAULT_MODE)
        system_prompt (Optional[str]): Optional system prompt for LLM configuration
        user_prompt (Optional[str]): User prompt for processing
        max_workers (int): Maximum number of concurrent workers
        output_file (str): Output file path
        checkpoint_file (str): Checkpoint file path
    
    Returns:
        Dict[str, Any]: Structured output containing processing results and statistics
    """
    return run_file_processing_full(
        input_pdf_dir_path=input_pdf_dir_path,
        pdf_file_paths=pdf_file_paths,
        strategy_type=strategy_type,
        mode=mode,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_workers=max_workers,
        output_file=output_file,
        checkpoint_file=checkpoint_file,
        llm_provider=llm_provider,
        llm_model=llm_model,
        csv_output_file=csv_output_file,
        benchmark_eval_mode=benchmark_eval_mode,
        streaming=streaming,
        max_files_per_request=max_files_per_request,
        benchmark_file_path=benchmark_file_path,
        config_manager=config_manager
    )





def resolve_combo_processing_params(combo_name: str = None,
                                  input_pdf_dir_path: Path = None,
                                  pdf_file_paths: List[Path] = [],
                                  output_dir: str = None,
                                  benchmark_file_path: Path = None,
                                  benchmark_eval_mode: bool = False):
    """
    Resolve parameters with intelligent defaults and type consistency.
    
    Args:
        combo_name: Name of the combo to run (optional, uses DEFAULT_STRATEGY_PARAM_GRP if None)
        input_pdf_dir_path: Input directory path (required)
        pdf_file_paths: List of PDF file paths (optional, will scan if empty)
        output_dir: Output directory (optional, uses default if not provided)
        benchmark_file_path: Benchmark file path (optional, required for evaluation runs)
    
    Returns:
        Dictionary with resolved parameters
    
    Raises:
        ValueError: If validation fails
    """
    from pathlib import Path
    import config.config_base as config_base
    
    # Handle default combo case
    if combo_name is None:
        # Use default strategy group as single-strategy combo
        combo_name = "default_single_strategy"
        strategy_groups = [config_base.DEFAULT_STRATEGY_PARAM_GRP]
        logging.info(f"🎯 Using default strategy group: {config_base.DEFAULT_STRATEGY_PARAM_GRP}")
    else:
        # Use centralized combo configuration (prefer injected config from server)
        try:
            # Try to use injected configuration from server first
            from config.config_combo_run import combo_config
            if hasattr(combo_config, '__dict__') and 'combo_config' in combo_config.__dict__:
                # Use injected configuration
                injected_combo_config = combo_config.combo_config
                logging.info(f"🎯 Using injected combo configuration from server")
            else:
                # Fall back to direct import
                injected_combo_config = combo_config
                logging.info(f"🎯 Using direct combo configuration import")
        except ImportError:
            # Fall back to direct import
            from config.config_combo_run import combo_config
            injected_combo_config = combo_config
            logging.info(f"🎯 Using fallback combo configuration import")
        
        if combo_name not in injected_combo_config:
            raise ValueError(f"Combo '{combo_name}' not found in configuration")
        
        # Get combo strategy groups from centralized config
        strategy_groups = injected_combo_config[combo_name]["strategy_groups"]
    
    # Handle both scenarios with proper priority
    if pdf_file_paths:
        # Scenario 1: pdf_file_paths provided (takes precedence)
        logging.info(f"🔄 Using provided pdf_file_paths: {len(pdf_file_paths)} files")
        
        # Validate each file exists
        for pdf_path in pdf_file_paths:
            pdf_path_obj = Path(pdf_path) if isinstance(pdf_path, str) else pdf_path
            if not pdf_path_obj.exists():
                raise ValueError(f"PDF file does not exist: {pdf_path}")
            if not pdf_path_obj.is_file():
                raise ValueError(f"Path is not a file: {pdf_path}")
        
        # Extract input_pdf_dir_path from first file for compatibility
        if pdf_file_paths:
            first_file_path = Path(pdf_file_paths[0]) if isinstance(pdf_file_paths[0], str) else pdf_file_paths[0]
            input_pdf_dir_path = first_file_path.parent
            logging.info(f"📁 Extracted input_pdf_dir_path from pdf_file_paths: {input_pdf_dir_path}")
        
    elif input_pdf_dir_path is not None:
        # Scenario 2: input_pdf_dir_path provided, convert to pdf_file_paths
        logging.info(f"📁 Using input_pdf_dir_path: {input_pdf_dir_path}")
        
        input_path_obj = Path(input_pdf_dir_path) if isinstance(input_pdf_dir_path, str) else input_pdf_dir_path
        if not input_path_obj.exists():
            raise ValueError(f"Input directory does not exist: {input_pdf_dir_path}")
        if not input_path_obj.is_dir():
            raise ValueError(f"Input path is not a directory: {input_pdf_dir_path}")
        
        pdf_files = get_pdf_files(str(input_pdf_dir_path))
        pdf_file_paths = [Path(f) for f in pdf_files]
        logging.info(f"📄 Scanned {len(pdf_file_paths)} PDF files from directory")
        
    else:
        # Scenario 3: Neither provided (should be caught by validator)
        raise ValueError("Either input_pdf_dir_path or pdf_file_paths is required")
    
    # Resolve output_dir (use server profile default if not provided)
    if output_dir is None:
        output_dir = getattr(config_base, 'OUTPUT_BASE_DIR', './output')
    
    # Validate benchmark_file_path for evaluation runs
    if benchmark_eval_mode and benchmark_file_path is None:
        raise ValueError("benchmark_file_path is required for evaluation runs")
    
    if benchmark_file_path:
        benchmark_path_obj = Path(benchmark_file_path) if isinstance(benchmark_file_path, str) else benchmark_file_path
        if not benchmark_path_obj.exists():
            raise ValueError(f"Benchmark file does not exist: {benchmark_file_path}")
    
    return {
        "strategy_groups": strategy_groups,
        "input_pdf_dir_path": input_pdf_dir_path,
        "pdf_file_paths": pdf_file_paths,
        "output_dir": output_dir,
        "benchmark_file_path": benchmark_file_path
    }





def _run_combo_processing_parallel(config_manager, benchmark_eval_mode: bool = False, combo_name: str = None, 
                                streaming: bool = False, max_cc_strategies: int = 3, max_cc_filegroups: int = 5,
                                max_files_per_request: int = None,
                                input_pdf_dir_path: Path = None,
                                pdf_file_paths: List[Path] = [],
                                output_dir: str = None,
                                benchmark_file_path: Path = None) -> int:
    """Run combination processing with strategy-level parallelization."""
    
    try:
        # Import the centralized combo configuration (prefer injected config from server)
        try:
            # Try to use injected configuration from server first
            from config.config_combo_run import combo_config
            if hasattr(combo_config, '__dict__') and 'combo_config' in combo_config.__dict__:
                # Use injected configuration
                injected_combo_config = combo_config.combo_config
                logging.info(f"🎯 Using injected combo configuration from server")
            else:
                # Fall back to direct import
                injected_combo_config = combo_config
                logging.info(f"🎯 Using direct combo configuration import")
        except ImportError:
            # Fall back to direct import
            from config.config_combo_run import combo_config
            injected_combo_config = combo_config
            logging.info(f"🎯 Using fallback combo configuration import")
        
        logging.info(f"📋 Loaded centralized combo configuration")
        logging.info(f"📊 Found {len(injected_combo_config)} combo(s)")
        
        # Filter combos if specific combo_name is provided
        if combo_name:
            if combo_name not in injected_combo_config:
                logging.error(f"❌ Combo '{combo_name}' not found in configuration. Available combos: {list(injected_combo_config.keys())}")
                return 1
            combo_config = {combo_name: injected_combo_config[combo_name]}
            logging.info(f"🎯 Running specific combo: {combo_name}")
        else:
            logging.info(f"🔄 Running all {len(injected_combo_config)} combos")
        
        # Resolve parameters for the specific combo
        params = resolve_combo_processing_params(
            combo_name=combo_name,
            input_pdf_dir_path=input_pdf_dir_path,
            pdf_file_paths=pdf_file_paths,
            output_dir=output_dir,
            benchmark_file_path=benchmark_file_path,
            benchmark_eval_mode=benchmark_eval_mode
        )
        
        # Use resolved parameters
        strategy_groups = params["strategy_groups"]
        input_pdf_dir_path = params["input_pdf_dir_path"]
        pdf_file_paths = params["pdf_file_paths"]
        output_base_dir = params["output_dir"]
        benchmark_file_path = params["benchmark_file_path"]
        
        # Handle default combo case
        if combo_name is None:
            combo_name = "default_single_strategy"
        
        logging.info(f"🚀 Starting combo: {combo_name}")
        logging.info(f"📋 Resolved parameters:")
        logging.info(f"   Input: {input_pdf_dir_path}")
        logging.info(f"   Output: {output_base_dir}")
        logging.info(f"   PDF files: {len(pdf_file_paths)}")
        logging.info(f"   Strategy groups: {len(strategy_groups)}")
        
        # Generate request metadata for new directory structure
        from common.request_id_generator import RequestIDGenerator
        from common.combo_meta_manager import ComboMetaManager
        
        request_metadata = RequestIDGenerator.create_request_metadata()
        request_id = request_metadata["request_id"]
        
        # Create new timestamped results directory
        results_dir = ComboMetaManager.create_results_directory(output_base_dir, request_id)
        combo_csv_dir, combo_json_dir = ComboMetaManager.create_combo_directories(results_dir)
        
        # Create combo_meta.json file
        ComboMetaManager.create_combo_meta_file(
            results_dir, request_metadata, combo_name, strategy_groups, len(pdf_file_paths)
        )
        
        # Import parameter groups definition
        from config.config_param_grps import param_grps
        
        # Initialize strategy files with detailed structure
        strategy_filename_mapping = ComboMetaManager.initialize_strategy_files(combo_json_dir, combo_csv_dir, strategy_groups, param_grps)
        
        logging.info(f"📁 Created new results directory: {results_dir}")
        logging.info(f"📂 Processing input files from: {input_pdf_dir_path}")
        logging.info(f"📊 Found {len(pdf_file_paths)} PDF files")
        
        if not pdf_file_paths:
            logging.warning(f"⚠️ No PDF files found in {input_pdf_dir_path}")
            return 1
        
        # Process each strategy group
        strategy_group_names = strategy_groups
        logging.info(f"🔄 Processing {len(strategy_group_names)} strategy group(s) with parallelization")
        
        # Group strategies by provider to avoid rate limiting
        provider_groups = _group_strategies_by_provider(strategy_group_names, param_grps)
        
        # Process provider groups sequentially, but strategies within each provider in parallel
        for provider_name, provider_strategies in provider_groups.items():
                logging.info(f"🔄 Processing provider group: {provider_name} with {len(provider_strategies)} strategies")
                logging.info(f"🔧🔧🔧 THREAD POOL: Creating executor with {max_cc_strategies} worker threads 🔧🔧🔧")
                
                with ThreadPoolExecutor(max_workers=max_cc_strategies) as executor:
                    # Submit strategies for this provider group
                    futures = []
                    for group_name in provider_strategies:
                        if group_name not in param_grps:
                            logging.error(f"❌ Parameter group '{group_name}' not found in param_grps definition")
                            continue
                            
                        group_params = param_grps[group_name]
                        logging.info(f"⚙️ Submitting parameter group: {group_name}")
                        
                        # Get the pre-generated filenames for this strategy
                        filename_mapping = strategy_filename_mapping.get(group_name)
                        if not filename_mapping:
                            logging.error(f"❌ No filename mapping found for strategy group: {group_name}")
                            continue
                        
                        json_filename = filename_mapping["json"]
                        csv_filename = filename_mapping["csv"]
                        
                        # Create a future for this strategy
                        future = executor.submit(
                            _process_single_strategy,
                            group_name=group_name,
                            group_params=group_params,
                            combo_name=combo_name,
                            combo_json_dir=combo_json_dir,
                            combo_csv_dir=combo_csv_dir,
                            input_files_path=str(input_pdf_dir_path),
                            pdf_file_paths=pdf_file_paths,
                            benchmark_eval_mode=benchmark_eval_mode,  # This should be the function parameter
                            streaming=streaming,
                            max_cc_filegroups=max_cc_filegroups,
                            max_files_per_request=max_files_per_request,
                            json_filename=json_filename,
                            csv_filename=csv_filename,
                            benchmark_file_path=benchmark_file_path,
                            config_manager = config_manager
                        )
                        futures.append((group_name, future))
                    
                    # Wait for all strategies in this provider group to complete
                    for group_name, future in futures:
                        try:
                            result = future.result()
                            logging.info(f"✅ Successfully processed combo {combo_name}, group {group_name}")
                        except Exception as e:
                            logging.error(f"❌ Error processing combo {combo_name}, group {group_name}: {str(e)}")
                            continue
        
        logging.info(f"✅ Completed combo: {combo_name}")
        
        return 0
        
    except Exception as e:
        logging.error(f"❌ Error in parallel combo processing: {str(e)}")
        return 1


def _group_strategies_by_provider(parameter_group_names: List[str], param_grps: Dict) -> Dict[str, List[str]]:
    """Group strategies by provider to avoid rate limiting."""
    provider_groups = {}
    
    for group_name in parameter_group_names:
        if group_name not in param_grps:
            continue
            
        group_params = param_grps[group_name]
        provider = group_params.get("provider", "unknown")
        
        if provider not in provider_groups:
            provider_groups[provider] = []
        
        provider_groups[provider].append(group_name)
    
    logging.info(f"📊 Grouped strategies by provider: {provider_groups}")
    return provider_groups


def _process_single_strategy(config_manager, group_name: str, group_params: Dict, combo_name: str, 
                           combo_json_dir: Path, combo_csv_dir: Path, input_files_path: str,
                           pdf_file_paths: List[Path], benchmark_eval_mode: bool, streaming: bool, 
                           max_cc_filegroups: int = 5, max_files_per_request: int = None,
                           json_filename: str = None, csv_filename: str = None,
                           benchmark_file_path: Path = None) -> Dict[str, Any]:
    """Process a single strategy within a combo run."""
    logging.info(f"⚙️ Processing parameter group: {group_name}")
    logging.info(f"📋 Parameters: {group_params}")

    # Extract parameters
    strategy = group_params.get("strategy", STRATEGY_DIRECT_FILE)
    mode = group_params.get("mode", MODE_PARALLEL)
    provider = group_params.get("provider", "google")
    model = group_params.get("model", "gemini-2.5-flash")
    temperature = group_params.get("temperature", 0.1)
    
    # Generate timestamped filenames for this group
    # Use the pre-generated filenames if provided, otherwise generate new ones
    if json_filename is None or csv_filename is None:
        # Fallback to generating new filenames if not provided
        json_filename = generate_timestamped_filename(strategy, mode, provider, model, "json")
        csv_filename = generate_timestamped_filename(strategy, mode, provider, model, "csv")
    
    output_file = str(combo_json_dir / json_filename)
    csv_output_file = str(combo_csv_dir / csv_filename)
    
    logging.info(f"📄 Output files: {output_file}, {csv_output_file}")
    
    # Run the processing
    results = run_file_processing(
        input_pdf_dir_path=Path(input_files_path),
        pdf_file_paths=pdf_file_paths,
        strategy_type=strategy,
        mode=mode,
        system_prompt=config_base.SYSTEM_PROMPT,
        user_prompt=config_base.USER_PROMPT,
        max_workers=max_cc_filegroups,  # Maximum Concurrency Level for File Groups, for a Single Strategy
        output_file=output_file,
        checkpoint_file=f"modular_checkpoint_{combo_name}_{group_name}.pkl",
        llm_provider=provider,
        llm_model=model,
        csv_output_file=csv_output_file,
        benchmark_eval_mode=benchmark_eval_mode,
        streaming=streaming,
        max_files_per_request=max_files_per_request,
        benchmark_file_path=benchmark_file_path,
        config_manager=config_manager
    )
    
    return results


def run_combo_processing(config_manager, benchmark_eval_mode: bool = False, combo_name: str = None, 
                        streaming: bool = False, max_cc_strategies: int = 3, max_cc_filegroups: int = 5,
                        max_files_per_request: int = None,
                        input_pdf_dir_path: Path = None,
                        pdf_file_paths: List[Path] = [],
                        output_dir: str = None,
                        benchmark_file_path: Path = None,) -> int:
    """Run combination processing using centralized configuration."""
    
    # Validate parameters using the new validator
    from processors.parameter_validator import ParameterValidator
    
    # Get config defaults from config_base
    config_defaults = {
        'DEFAULT_MAX_CC_STRATEGIES': getattr(config_base, 'DEFAULT_MAX_CC_STRATEGIES', 3),
        'DEFAULT_MAX_CC_FILEGROUPS': getattr(config_base, 'DEFAULT_MAX_CC_FILEGROUPS', 5),
        'DEFAULT_MAX_FILES_PER_REQUEST': getattr(config_base, 'DEFAULT_MAX_FILES_PER_REQUEST', 10),
        'OUTPUT_BASE_DIR': getattr(config_base, 'OUTPUT_BASE_DIR', './output')
    }
    
    validator = ParameterValidator(config_defaults)
    validation_result = validator.validate_run_combo_processing(
        benchmark_eval_mode=benchmark_eval_mode,
        combo_name=combo_name,
        streaming=streaming,
        max_cc_strategies=max_cc_strategies,
        max_cc_filegroups=max_cc_filegroups,
        max_files_per_request=max_files_per_request,
        input_pdf_dir_path=input_pdf_dir_path,
        pdf_file_paths=pdf_file_paths,
        output_dir=output_dir,
        benchmark_file_path=benchmark_file_path
    )
    
    if not validation_result.is_valid:
        error_msg = "; ".join(validation_result.errors)
        logging.error(f"❌ Parameter validation failed: {error_msg}")
        raise ValueError(f"Parameter validation failed: {error_msg}")
    
    # Use validated parameters
    validated_params = validation_result.validated_params
    
    # Always use _run_combo_processing_parallel - it handles both sequential and parallel
    # Sequential processing is just a special case when max_cc_strategies = 1
    if validated_params['max_cc_strategies'] > 1:
        logging.info(f"⚡⚡⚡ PARALLEL MODE ENABLED: Running combo '{validated_params['combo_name'] or 'default'}' with up to {validated_params['max_cc_strategies']} strategies concurrently ⚡⚡⚡")
    else:
        logging.info(f"🔄 SEQUENTIAL MODE: Running combo '{validated_params['combo_name'] or 'default'}' with up to {validated_params['max_cc_strategies']} strategy sequentially 🔄")
    
    return _run_combo_processing_parallel(
        benchmark_eval_mode=validated_params['benchmark_eval_mode'],
        combo_name=validated_params['combo_name'],
        streaming=validated_params['streaming'],
        max_cc_strategies=validated_params['max_cc_strategies'],
        max_cc_filegroups=validated_params['max_cc_filegroups'],
        max_files_per_request=validated_params['max_files_per_request'],
        input_pdf_dir_path=validated_params['input_pdf_dir_path'],
        pdf_file_paths=validated_params['pdf_file_paths'],
        output_dir=validated_params['output_dir'],
        benchmark_file_path=validated_params['benchmark_file_path'],
        config_manager = config_manager
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Modular PDF Processing System')
    
    # Combo configuration arguments
    parser.add_argument('--combo_name', type=str, required=False,
                       help='Combo name to run from the combo configuration file (optional - uses default parameter group if not specified)')
    
    # Input/output arguments
    parser.add_argument('input', nargs='?', help='Input PDF file or directory containing PDF files (ignored if --combo_name is specified)')
    parser.add_argument('--output', '-o', default='modular_results.json', 
                       help='Output file path (default: modular_results.json)')

    

    
    # Processing arguments
    parser.add_argument('--max-cc-filegroups', '-w', type=int, default=5,
                       help='Maximum Concurrency Level for File Groups, for a Single Strategy (default: 5, use 1 for sequential)')
    parser.add_argument('--max-files-per-request', type=int, default=None,
                       help='Maximum number of files per request (default: uses strategy default)')

    
    # Prompt arguments
    parser.add_argument('--system-prompt', type=str, default=None,
                       help='Optional system prompt for LLM configuration')
    parser.add_argument('--user-prompt', type=str, default=None,
                       help='User prompt for processing (default: uses strategy default)')
    
    # Logging arguments
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    
    # Benchmark testing arguments
    parser.add_argument('--benchmark-eval-mode', action='store_true',
                       help='Enable benchmark evaluation mode (requires benchmark file)')
    parser.add_argument('--benchmark-file', type=str, default=None,
                       help='Path to benchmark file for evaluation mode')
    
    # Streaming arguments
    parser.add_argument('--streaming', action='store_true',
                       help='Enable streaming mode for LLM responses (currently only supported for Google GenAI)')
    
    # Strategy-level parallelization arguments
    # Strategy 1 ──┐
    # Strategy 2 ──┼── Concurrent Strategy Execution
    # Strategy 3 ──┘
    #     ├── File Groups (Parallel) ──┐
    #     ├── File Groups (Parallel) ──┼── Still parallel within each strategy
    #     └── File Groups (Parallel) ──┘
    #         └── Individual Files (Parallel) ── Still parallel within each file group
    parser.add_argument('--max-cc-strategies', type=parse_max_cc_strategies, default=3,
                       help='Maximum Concurrency Level for Strategies. Use "max" to use all available CPU threads, or specify a number (default: 3)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Check if combo configuration is specified
        if args.combo_name:
            logging.info(f"🔄 Running combo mode with combo: {args.combo_name}")
            return run_combo_processing(
                benchmark_eval_mode=args.benchmark_eval_mode, 
                combo_name=args.combo_name, 
                streaming=args.streaming,
                max_cc_strategies=args.max_cc_strategies,
                max_cc_filegroups=args.max_cc_filegroups,
                max_files_per_request=args.max_files_per_request,
                benchmark_file_path=Path(args.benchmark_file) if args.benchmark_file else None
            )
        else:
            # Use unified processing with default parameter group
            logging.info(f"🔄 Running unified processing using default parameter group")
            return run_combo_processing(
                benchmark_eval_mode=args.benchmark_eval_mode, 
                combo_name=None,  # Will use DEFAULT_STRATEGY_PARAM_GRP
                streaming=args.streaming,
                max_cc_strategies=args.max_cc_strategies,
                max_cc_filegroups=args.max_cc_filegroups,
                max_files_per_request=args.max_files_per_request,
                benchmark_file_path=Path(args.benchmark_file) if args.benchmark_file else None
            )
        
    except Exception as e:
        logging.error(f"❌ Processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 