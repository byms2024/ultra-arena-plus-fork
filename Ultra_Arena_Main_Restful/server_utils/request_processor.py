#!/usr/bin/env python3
"""
Request Processing Module for Ultra Arena RESTful API

This module handles all request processing operations using the optimized configuration system.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple
from flask import jsonify

from .config_manager import ConfigManager
from .request_validator import RequestValidator
from .config_assemblers import ConfigAssemblyResult

logger = logging.getLogger(__name__)


class RequestProcessor:
    """Processes requests for the Ultra Arena RESTful API using optimized configuration."""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the request processor.
        
        Args:
            config_manager: The configuration manager instance
        """
        self.config_manager = config_manager
    
    def validate_combo_request(self, data: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Validate combo processing request before configuration assembly.
        
        Note: This method is kept for backward compatibility but validation is now handled
        by the RequestValidator in the server endpoints.
        
        Args:
            data: The request data to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Validation is now handled by RequestValidator in server endpoints
        # This method is kept for backward compatibility
        return True, None
    
    def create_unified_request_config(self, data: Dict[str, Any], use_default_combo: bool = False) -> Dict[str, Any]:
        """
        Create unified configuration for both simple and combo processing requests.
        
        Args:
            data: Request data from the endpoint
            use_default_combo: If True, use DEFAULT_STRATEGY_PARAM_GRP from profile config and force max_cc_strategies=1
            
        Returns:
            Dict[str, Any]: Unified configuration
            
        Raises:
            ValueError: If configuration creation failed
        """
        try:
            # Validate request before processing (only for combo requests)
            if not use_default_combo:
                is_valid, error_msg = self.validate_combo_request(data)
                if not is_valid:
                    raise ValueError(error_msg)
            
            # Generate request metadata for tracking
            try:
                import sys
                import os
                # Add the parent directory to the path to import Ultra_Arena_Main
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
                from Ultra_Arena_Main.common.request_id_generator import RequestIDGenerator
                request_metadata = RequestIDGenerator.create_request_metadata()
                logger.info(f"🔑 Generated request metadata: {request_metadata['request_id']}")
            except Exception as e:
                logger.warning(f"⚠️ Could not generate request metadata: {e}")
                request_metadata = {
                    "request_id": "unknown",
                    "request_mechanism": "rest",
                    "request_start_time": "",
                    "utc_timezone": "UTC"
                }
            
            # Use the optimized configuration assembler
            config_result = self.config_manager.assemble_request_config(data, use_default_combo)
            
            # Log configuration summary
            assembler = self.config_manager.get_request_assembler()
            if assembler:
                assembler.log_configuration_summary(config_result)
                # Inject final configuration into config_base for backward compatibility
                assembler.inject_final_config_into_base(config_result)
            
            # Convert to the expected format for backward compatibility
            request_config = config_result.request_config
            
            return {
                "combo_name": request_config.combo_name,
                "chain_name": request_config.chain_name,
                "chain_config" : request_config.chain_config,
                "input_pdf_dir_path": request_config.input_pdf_dir_path,
                "benchmark_eval_mode": request_config.final_processing.benchmark_eval_mode,
                "streaming": request_config.final_processing.streaming,
                "max_cc_strategies": request_config.final_processing.max_cc_strategies,
                "max_cc_filegroups": request_config.final_processing.max_cc_filegroups,
                "max_files_per_request": request_config.final_processing.max_files_per_request,
                "output_dir": request_config.output_dir,
                "benchmark_file_path": request_config.benchmark_file_path,
                "request_metadata": request_metadata,  # Add request metadata
                "prompt_config": {
                    "system_prompt": request_config.final_prompts.system_prompt,
                    "user_prompt": request_config.final_prompts.user_prompt,
                    "json_format_instructions": request_config.final_prompts.json_format_instructions,
                    "mandatory_keys": request_config.final_prompts.mandatory_keys,
                    "text_first_regex_criteria": request_config.final_prompts.text_first_regex_criteria,
                    "_source_info": request_config.final_prompts.source_info
                },
                "run_config": {
                    "input_pdf_dir_path": str(request_config.input_pdf_dir_path) if request_config.input_pdf_dir_path else None,
                    "output_dir": str(request_config.output_dir) if request_config.output_dir else None,
                    "benchmark_file_path": str(request_config.benchmark_file_path) if request_config.benchmark_file_path else None,
                    "benchmark_eval_mode": request_config.final_processing.benchmark_eval_mode,
                    "pdf_file_paths": []  # Will be populated during processing
                },
                "_config_result": config_result  # Store the full result for response formatting
            }
            
        except Exception as e:
            logger.error(f"❌ Error in create_unified_request_config: {e}")
            raise ValueError(f"Configuration creation failed: {str(e)}")
    
    def execute_processing(self, config: Dict[str, Any]) -> int:
        """
        Execute processing with the given configuration.
        
        Args:
            config: The processing configuration
            
        Returns:
            int: Result code from processing
        """
        processing_start_time = time.time()
        
        # Time profile configuration injection (now very fast due to caching)
        logger.info(f"⏱️  START: Profile Configuration Injection")
        profile_injection_start = time.time()
        
        # Configuration is already injected by the assembler, just verify
        self.config_manager.inject_profile_config()
        
        profile_injection_time = time.time() - profile_injection_start
        logger.info(f"⏱️  END: Profile Configuration Injection - Duration: {profile_injection_time:.3f}s")
        
        # Time module import
        import_start_time = time.time()
        # Dynamically import to avoid static import resolution issues
        import importlib, os, sys
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Ultra_Arena_Main'))
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)
        run_combo_processing = importlib.import_module('main_modular').run_combo_processing
        # If a chain_name is provided, we will ignore combo_name and run a single chain strategy
        import_time = time.time() - import_start_time
        logger.info(f"⏱️  Module Import Time: {import_time:.3f}s")
        
        # Time the main processing call (this is the core bottleneck)
        main_processing_start = time.time()
        logger.info(f"⏱️  START: Main Library Processing Call")
        
        chain_name = config.get("chain_name")
        if chain_name:
            # Build a synthetic combo run with a single parameter group that uses chain strategy
            # Use centralized chain definitions
            try:
                from Ultra_Arena_Main.config.config_chain_defs import chain_definitions
            except Exception as e:
                logger.error(f"❌ Chain requested but chain definitions could not be imported: {e}")
                raise

            if chain_name not in chain_definitions:
                raise ValueError(f"Chain '{chain_name}' not found in chain_definitions")

            from Ultra_Arena_Main.llm_strategies.strategy_factory import ProcessingStrategyFactory
            from Ultra_Arena_Main.config import config_base as ua_config_base
            from Ultra_Arena_Main.main_modular import get_config_for_strategy
            from Ultra_Arena_Main.processors.modular_parallel_processor import ModularParallelProcessor

            # Resolve files
            input_path = config["input_pdf_dir_path"]
            if input_path is None:
                raise ValueError("input_pdf_dir_path is required when using chain_name")
            from Ultra_Arena_Main.main_modular import get_pdf_files
            pdf_files = get_pdf_files(str(input_path))

            # Prepare output filenames using same utilities
            from Ultra_Arena_Main.main_modular import generate_timestamped_filename
            provider = ua_config_base.DEFAULT_LLM_PROVIDER
            model = ua_config_base.GOOGLE_DEFAULT_MODEL_ID if provider == "google" else "model"
            strategy = ua_config_base.STRATEGY_CHAIN
            mode = ua_config_base.MODE_PARALLEL
            json_filename = generate_timestamped_filename(strategy, mode, provider, model, "json")
            csv_filename = generate_timestamped_filename(strategy, mode, provider, model, "csv")
            from Ultra_Arena_Main.common.combo_meta_manager import ComboMetaManager
            from Ultra_Arena_Main.common.request_id_generator import RequestIDGenerator
            request_metadata = RequestIDGenerator.create_request_metadata()
            results_dir = ComboMetaManager.create_results_directory(config["output_dir"], request_metadata["request_id"]) 
            combo_csv_dir, combo_json_dir = ComboMetaManager.create_combo_directories(results_dir)
            output_file = str((combo_json_dir / json_filename))
            csv_output_file = str((combo_csv_dir / csv_filename))

            # Build chain config by merging defaults with chain steps
            base_chain_config = get_config_for_strategy(ua_config_base.STRATEGY_CHAIN, streaming=config["streaming"])            
            chain_cfg = {**base_chain_config, **chain_definitions[chain_name]}

            # Create processor for chain
            processor = ModularParallelProcessor(
                config_manager=self.config_manager,
                config=chain_cfg,
                strategy_type=ua_config_base.STRATEGY_CHAIN,
                mode=mode,
                max_workers=config["max_cc_filegroups"],
                checkpoint_file=f"modular_checkpoint_chain_{chain_name}.pkl",
                output_file=output_file,
                real_time_save=True,
                run_settings={"strategy": strategy, "mode": mode, "llm_provider": provider, "llm_model": model},
                csv_output_file=csv_output_file,
                benchmark_comparator=None,
                streaming=config["streaming"],
            )

            # Process using the unified processor API
            import importlib
            cb = importlib.import_module('config.config_base')
            results = processor.process_files(pdf_files=pdf_files, system_prompt=getattr(cb, 'SYSTEM_PROMPT', ''), user_prompt=getattr(cb, 'USER_PROMPT', ''))
            # Return a success marker consistent with run_combo_processing
            result_code = 0 if results else 1
        else:
            result_code = run_combo_processing(
                combo_name=config["combo_name"],
                benchmark_eval_mode=config["benchmark_eval_mode"],
                streaming=config["streaming"],
                max_cc_strategies=config["max_cc_strategies"],
                max_cc_filegroups=config["max_cc_filegroups"],
                max_files_per_request=config["max_files_per_request"],
                input_pdf_dir_path=config["input_pdf_dir_path"],
                pdf_file_paths=[],
                output_dir=config["output_dir"],
                benchmark_file_path=config["benchmark_file_path"]
            )
        
        main_processing_time = time.time() - main_processing_start
        logger.info(f"⏱️  END: Main Library Processing Call - Duration: {main_processing_time:.3f}s")
        
        # Log detailed breakdown
        total_processing_time = time.time() - processing_start_time
        logger.info(f"📊 MAIN PROCESSING BREAKDOWN:")
        logger.info(f"  Profile Injection: {profile_injection_time:.3f}s")
        logger.info(f"  Module Import: {import_time:.3f}s")
        logger.info(f"  Main Library Call: {main_processing_time:.3f}s")
        logger.info(f"  Total Processing: {total_processing_time:.3f}s")
        
        return result_code
    
    def format_combo_response(self, result_code: int, config: Dict[str, Any]):
        """
        Format response for combo processing endpoint.
        
        Args:
            result_code: The processing result code
            config: The processing configuration
            
        Returns:
            Flask response: Formatted JSON response
        """
        run_config = config.get("run_config", {})
        
        # Determine combo_name for response
        combo_name = config.get("combo_name")
        if combo_name is None:
            combo_name = "default_single_combo_name"
        
        # Include prompt configuration that was used with source information
        prompt_info = {}
        prompt_config = config.get("prompt_config", {})
        
        # Get source information if available
        source_info = prompt_config.get("_source_info", {})
        
        if source_info:
            # Create detailed prompt info with sources
            prompt_details = {}
            for field, info in source_info.items():
                if field != "_source_info":
                    prompt_details[field] = {
                        "value": info.get("value"),
                        "source": info.get("source")
                    }
            
            prompt_info = {
                "prompt_configuration": prompt_details,
                "summary": {
                    "total_prompts": len(prompt_details),
                    "sources_used": list(set(info.get("source") for info in source_info.values() if info.get("source")))
                }
            }
            logger.info(f"📤 RESPONSE: Including {len(prompt_details)} prompt configurations with sources in response")
        elif prompt_config and len(prompt_config) > 0:
            # Fallback for legacy format
            prompt_info = {
                "prompt_overrides": list(prompt_config.keys()),
                "note": "Legacy format - source information not available"
            }
            logger.info(f"📤 RESPONSE: Including {len(prompt_config)} prompt overrides in response (legacy)")
        else:
            prompt_info = {
                "prompt_overrides": [],
                "note": "Using profile default prompts"
            }
            logger.info(f"📤 RESPONSE: Using profile default prompts in response")
        
        # Include configuration assembly performance metrics if available
        config_result = config.get("_config_result")
        performance_info = {}
        if config_result:
            performance_info = {
                "configuration_assembly_time_ms": config_result.assembly_time_ms,
                "server_config_cached": config_result.cache_hit
            }
        
        # Get strategy groups information
        strategy_groups = []
        try:
            import importlib
            combo_module = importlib.import_module('config.config_combo_run')
            combo_config = getattr(combo_module, 'combo_config', {})
            if combo_name and combo_name in combo_config:
                strategy_groups = combo_config[combo_name].get("strategy_groups", [])
        except Exception as e:
            logger.warning(f"⚠️ Could not get strategy groups: {e}")
        
        # Get request metadata if available
        request_metadata = config.get("request_metadata", {})
        
        response_data = {
            "status": "success",
            "combo_name": combo_name,
            "benchmark_eval_mode": run_config.get("benchmark_eval_mode", False),
            "input_pdf_dir_path": str(run_config.get('input_pdf_dir_path', '')),
            "output_dir": str(run_config.get('output_dir', '')),
            "benchmark_file_path": str(run_config.get('benchmark_file_path')) if run_config.get('benchmark_file_path') else None,
            "prompt": prompt_info,
            "performance": performance_info,
            "results": result_code,
            # New fields as requested
            "request_id": request_metadata.get("request_id", "unknown"),
            "request_mechanism": request_metadata.get("request_mechanism", "rest"),
            "request_start_time": request_metadata.get("request_start_time", ""),
            "utc_timezone": request_metadata.get("utc_timezone", "UTC"),
            "num_files_to_process": len(run_config.get('pdf_file_paths', [])),
            "num_strategies": len(strategy_groups),
            "strategy_groups": strategy_groups
        }
        
        return jsonify(response_data)
    
    def format_async_combo_response(self, request_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format response for async combo processing endpoint (202 Accepted).
        
        Args:
            request_id: The request ID for tracking
            config: The processing configuration
            
        Returns:
            Dict[str, Any]: Formatted JSON response for async processing
        """
        run_config = config.get("run_config", {})
        
        # Determine combo_name for response
        combo_name = config.get("combo_name")
        if combo_name is None:
            combo_name = "default_single_combo_name"
        
        # Include prompt configuration that was used with source information
        prompt_info = {}
        prompt_config = config.get("prompt_config", {})
        
        # Get source information if available
        source_info = prompt_config.get("_source_info", {})
        
        if source_info:
            # Create detailed prompt info with sources
            prompt_details = {}
            for field, info in source_info.items():
                if field != "_source_info":
                    prompt_details[field] = {
                        "value": info.get("value"),
                        "source": info.get("source")
                    }
            
            prompt_info = {
                "prompt_configuration": prompt_details,
                "summary": {
                    "total_prompts": len(prompt_details),
                    "sources_used": list(set(info.get("source") for info in source_info.values() if info.get("source")))
                }
            }
        elif prompt_config and len(prompt_config) > 0:
            # Fallback for legacy format
            prompt_info = {
                "prompt_overrides": list(prompt_config.keys()),
                "note": "Legacy format - source information not available"
            }
        else:
            prompt_info = {
                "prompt_overrides": [],
                "note": "Using profile default prompts"
            }
        
        # Get strategy groups information
        strategy_groups = []
        try:
            import importlib
            combo_module = importlib.import_module('config.config_combo_run')
            combo_config = getattr(combo_module, 'combo_config', {})
            if combo_name and combo_name in combo_config:
                strategy_groups = combo_config[combo_name].get("strategy_groups", [])
        except Exception as e:
            logger.warning(f"⚠️ Could not get strategy groups: {e}")
        
        # Get request metadata if available
        request_metadata = config.get("request_metadata", {})
        
        response_data = {
            "status": "accepted",
            "request_id": request_id,
            "combo_name": combo_name,
            "benchmark_eval_mode": run_config.get("benchmark_eval_mode", False),
            "input_pdf_dir_path": str(run_config.get('input_pdf_dir_path', '')),
            "output_dir": str(run_config.get('output_dir', '')),
            "benchmark_file_path": str(run_config.get('benchmark_file_path')) if run_config.get('benchmark_file_path') else None,
            "prompt": prompt_info,
            # Note: performance and results are not included in async response
            "request_id": request_metadata.get("request_id", "unknown"),
            "request_mechanism": request_metadata.get("request_mechanism", "rest"),
            "request_start_time": request_metadata.get("request_start_time", ""),
            "utc_timezone": request_metadata.get("utc_timezone", "UTC"),
            "num_files_to_process": len(run_config.get('pdf_file_paths', [])),
            "num_strategies": len(strategy_groups),
            "strategy_groups": strategy_groups,
            "message": "Request accepted for processing. Use GET /api/requests/{request_id} to check status."
        }
        
        return response_data
    
    def log_processing_info(self, config: Dict[str, Any], endpoint_name: str):
        """
        Log processing information for both endpoints.
        
        Args:
            config: The processing configuration
            endpoint_name: The name of the endpoint being processed
        """
        combo_name = config.get("combo_name") or "default_single_strategy"
        input_path = config.get("input_pdf_dir_path")
        output_dir = config.get("output_dir")
        benchmark_eval_mode = config.get("benchmark_eval_mode", False)
        benchmark_file_path = config.get("benchmark_file_path")
        
        logger.info(f"Processing {endpoint_name}: {combo_name}")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Benchmark eval mode: {benchmark_eval_mode}")
        if benchmark_file_path:
            logger.info(f"Benchmark file: {benchmark_file_path}")
