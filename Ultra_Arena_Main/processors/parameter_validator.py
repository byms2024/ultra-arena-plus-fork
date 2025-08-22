#!/usr/bin/env python3
"""
Parameter Validator for Ultra Arena Main Library

This module provides validation logic for main library functions and logs parameter sources.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ParameterSource(Enum):
    """Enum for tracking parameter sources."""
    FUNCTION_ARG = "function_arg"
    CONFIG_DEFAULT = "config_default"
    SYSTEM_DEFAULT = "system_default"


@dataclass
class ValidationResult:
    """Result of parameter validation."""
    is_valid: bool
    errors: List[str]
    validated_params: Dict[str, Any]
    param_sources: Dict[str, ParameterSource]


class ParameterValidator:
    """Validates main library function parameters and tracks parameter sources."""
    
    def __init__(self, config_defaults: Dict[str, Any]):
        """
        Initialize the parameter validator.
        
        Args:
            config_defaults: Default configuration values from config_base
        """
        self.config_defaults = config_defaults
    
    def validate_run_combo_processing(self, 
                                    benchmark_eval_mode: bool = False, 
                                    combo_name: str = None,
                                    streaming: bool = False, 
                                    max_cc_strategies: int = 3, 
                                    max_cc_filegroups: int = 5,
                                    max_files_per_request: int = None,
                                    input_pdf_dir_path: Path = None,
                                    pdf_file_paths: List[Path] = [],
                                    output_dir: str = None,
                                    benchmark_file_path: Path = None) -> ValidationResult:
        """
        Validate parameters for run_combo_processing function.
        
        Args:
            benchmark_eval_mode: Whether to run in benchmark evaluation mode
            combo_name: Name of the combo to run
            streaming: Whether to use streaming mode
            max_cc_strategies: Maximum concurrent strategies
            max_cc_filegroups: Maximum concurrent file groups
            max_files_per_request: Maximum files per request
            input_pdf_dir_path: Input PDF directory path
            pdf_file_paths: List of PDF file paths
            output_dir: Output directory
            benchmark_file_path: Benchmark file path
            
        Returns:
            ValidationResult: Validation result with errors and validated parameters
        """
        errors = []
        validated_params = {}
        param_sources = {}
        
        # Validate and track benchmark_eval_mode
        validated_params['benchmark_eval_mode'] = bool(benchmark_eval_mode)
        param_sources['benchmark_eval_mode'] = ParameterSource.FUNCTION_ARG
        
        # Validate and track combo_name
        if combo_name is not None:
            validated_params['combo_name'] = combo_name
            param_sources['combo_name'] = ParameterSource.FUNCTION_ARG
        else:
            validated_params['combo_name'] = None
            param_sources['combo_name'] = ParameterSource.SYSTEM_DEFAULT
        
        # Validate and track streaming
        validated_params['streaming'] = bool(streaming)
        param_sources['streaming'] = ParameterSource.FUNCTION_ARG
        
        # Validate and track max_cc_strategies
        if max_cc_strategies is not None:
            if max_cc_strategies < 1:
                errors.append("max_cc_strategies must be at least 1")
            validated_params['max_cc_strategies'] = max_cc_strategies
            param_sources['max_cc_strategies'] = ParameterSource.FUNCTION_ARG
        else:
            validated_params['max_cc_strategies'] = self.config_defaults.get('DEFAULT_MAX_CC_STRATEGIES', 3)
            param_sources['max_cc_strategies'] = ParameterSource.CONFIG_DEFAULT
        
        # Validate and track max_cc_filegroups
        if max_cc_filegroups is not None:
            if max_cc_filegroups < 1:
                errors.append("max_cc_filegroups must be at least 1")
            validated_params['max_cc_filegroups'] = max_cc_filegroups
            param_sources['max_cc_filegroups'] = ParameterSource.FUNCTION_ARG
        else:
            validated_params['max_cc_filegroups'] = self.config_defaults.get('DEFAULT_MAX_CC_FILEGROUPS', 5)
            param_sources['max_cc_filegroups'] = ParameterSource.CONFIG_DEFAULT
        
        # Validate and track max_files_per_request
        if max_files_per_request is not None:
            if max_files_per_request < 1:
                errors.append("max_files_per_request must be at least 1")
            validated_params['max_files_per_request'] = max_files_per_request
            param_sources['max_files_per_request'] = ParameterSource.FUNCTION_ARG
        else:
            validated_params['max_files_per_request'] = self.config_defaults.get('DEFAULT_MAX_FILES_PER_REQUEST', 10)
            param_sources['max_files_per_request'] = ParameterSource.CONFIG_DEFAULT
        
        # Handle input_pdf_dir_path vs pdf_file_paths mutual exclusivity with priority
        if input_pdf_dir_path is not None and pdf_file_paths:
            # Both provided - pdf_file_paths takes precedence
            logger.info("🔄 Both input_pdf_dir_path and pdf_file_paths provided - using pdf_file_paths")
            validated_params['input_pdf_dir_path'] = None
            validated_params['pdf_file_paths'] = pdf_file_paths
            param_sources['input_pdf_dir_path'] = ParameterSource.SYSTEM_DEFAULT
            param_sources['pdf_file_paths'] = ParameterSource.FUNCTION_ARG
        elif input_pdf_dir_path is not None:
            # Only input_pdf_dir_path provided
            validated_params['input_pdf_dir_path'] = input_pdf_dir_path
            validated_params['pdf_file_paths'] = []
            param_sources['input_pdf_dir_path'] = ParameterSource.FUNCTION_ARG
            param_sources['pdf_file_paths'] = ParameterSource.SYSTEM_DEFAULT
        elif pdf_file_paths:
            # Only pdf_file_paths provided
            validated_params['input_pdf_dir_path'] = None
            validated_params['pdf_file_paths'] = pdf_file_paths
            param_sources['input_pdf_dir_path'] = ParameterSource.SYSTEM_DEFAULT
            param_sources['pdf_file_paths'] = ParameterSource.FUNCTION_ARG
        else:
            # Neither provided
            errors.append("Either input_pdf_dir_path or pdf_file_paths is required")
        
        # Validate and track output_dir
        if output_dir is not None:
            validated_params['output_dir'] = output_dir
            param_sources['output_dir'] = ParameterSource.FUNCTION_ARG
        else:
            validated_params['output_dir'] = self.config_defaults.get('OUTPUT_BASE_DIR', './output')
            param_sources['output_dir'] = ParameterSource.CONFIG_DEFAULT
        
        # Validate and track benchmark_file_path
        if benchmark_file_path is not None:
            validated_params['benchmark_file_path'] = benchmark_file_path
            param_sources['benchmark_file_path'] = ParameterSource.FUNCTION_ARG
        else:
            validated_params['benchmark_file_path'] = None
            param_sources['benchmark_file_path'] = ParameterSource.SYSTEM_DEFAULT
        
        # Validate paths
        validated_params, param_sources, errors = self._validate_paths(validated_params, param_sources, errors)
        
        # Validate benchmark requirements
        validated_params, param_sources, errors = self._validate_benchmark_requirements(validated_params, param_sources, errors)
        
        # Log parameter summary
        self._log_parameter_summary(validated_params, param_sources)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            validated_params=validated_params,
            param_sources=param_sources
        )
    
    def validate_run_file_processing(self,
                                   input_pdf_dir_path: Path,
                                   pdf_file_paths: List[Path] = [],
                                   strategy_type: str = None,
                                   mode: str = None,
                                   system_prompt: Optional[str] = None,
                                   user_prompt: Optional[str] = None,
                                   max_workers: int = None,
                                   output_file: str = None,
                                   checkpoint_file: str = None,
                                   llm_provider: str = None,
                                   llm_model: str = None,
                                   csv_output_file: str = None,
                                   benchmark_eval_mode: bool = False,
                                   streaming: bool = False,
                                   max_files_per_request: int = None) -> ValidationResult:
        """
        Validate parameters for run_file_processing function.
        
        Args:
            input_pdf_dir_path: Input PDF directory path
            pdf_file_paths: List of PDF file paths
            strategy_type: Processing strategy type
            mode: Processing mode
            system_prompt: System prompt
            user_prompt: User prompt
            max_workers: Maximum workers
            output_file: Output file path
            checkpoint_file: Checkpoint file path
            llm_provider: LLM provider
            llm_model: LLM model
            csv_output_file: CSV output file path
            benchmark_eval_mode: Whether to run in benchmark evaluation mode
            streaming: Whether to use streaming mode
            max_files_per_request: Maximum files per request
            
        Returns:
            ValidationResult: Validation result with errors and validated parameters
        """
        errors = []
        validated_params = {}
        param_sources = {}
        
        # Validate and track input_pdf_dir_path
        if input_pdf_dir_path is not None:
            validated_params['input_pdf_dir_path'] = input_pdf_dir_path
            param_sources['input_pdf_dir_path'] = ParameterSource.FUNCTION_ARG
        else:
            errors.append("input_pdf_dir_path is required")
        
        # Validate and track pdf_file_paths
        validated_params['pdf_file_paths'] = pdf_file_paths
        param_sources['pdf_file_paths'] = ParameterSource.FUNCTION_ARG
        
        # Validate and track strategy_type
        if strategy_type is not None:
            validated_params['strategy_type'] = strategy_type
            param_sources['strategy_type'] = ParameterSource.FUNCTION_ARG
        else:
            validated_params['strategy_type'] = self.config_defaults.get('DEFAULT_STRATEGY_TYPE', 'direct_file')
            param_sources['strategy_type'] = ParameterSource.CONFIG_DEFAULT
        
        # Validate and track mode
        if mode is not None:
            validated_params['mode'] = mode
            param_sources['mode'] = ParameterSource.FUNCTION_ARG
        else:
            validated_params['mode'] = self.config_defaults.get('DEFAULT_MODE', 'parallel')
            param_sources['mode'] = ParameterSource.CONFIG_DEFAULT
        
        # Validate and track system_prompt
        if system_prompt is not None:
            validated_params['system_prompt'] = system_prompt
            param_sources['system_prompt'] = ParameterSource.FUNCTION_ARG
        else:
            validated_params['system_prompt'] = None
            param_sources['system_prompt'] = ParameterSource.SYSTEM_DEFAULT
        
        # Validate and track user_prompt
        if user_prompt is not None:
            validated_params['user_prompt'] = user_prompt
            param_sources['user_prompt'] = ParameterSource.FUNCTION_ARG
        else:
            validated_params['user_prompt'] = None
            param_sources['user_prompt'] = ParameterSource.SYSTEM_DEFAULT
        
        # Validate and track max_workers
        if max_workers is not None:
            if max_workers < 1:
                errors.append("max_workers must be at least 1")
            validated_params['max_workers'] = max_workers
            param_sources['max_workers'] = ParameterSource.FUNCTION_ARG
        else:
            validated_params['max_workers'] = self.config_defaults.get('DEFAULT_MAX_CC_FILEGROUPS', 5)
            param_sources['max_workers'] = ParameterSource.CONFIG_DEFAULT
        
        # Validate and track output_file
        if output_file is not None:
            validated_params['output_file'] = output_file
            param_sources['output_file'] = ParameterSource.FUNCTION_ARG
        else:
            validated_params['output_file'] = self.config_defaults.get('DEFAULT_OUTPUT_FILE', 'modular_results.json')
            param_sources['output_file'] = ParameterSource.CONFIG_DEFAULT
        
        # Validate and track checkpoint_file
        if checkpoint_file is not None:
            validated_params['checkpoint_file'] = checkpoint_file
            param_sources['checkpoint_file'] = ParameterSource.FUNCTION_ARG
        else:
            validated_params['checkpoint_file'] = self.config_defaults.get('DEFAULT_CHECKPOINT_FILE', 'modular_checkpoint.pkl')
            param_sources['checkpoint_file'] = ParameterSource.CONFIG_DEFAULT
        
        # Validate and track llm_provider
        if llm_provider is not None:
            validated_params['llm_provider'] = llm_provider
            param_sources['llm_provider'] = ParameterSource.FUNCTION_ARG
        else:
            validated_params['llm_provider'] = None
            param_sources['llm_provider'] = ParameterSource.SYSTEM_DEFAULT
        
        # Validate and track llm_model
        if llm_model is not None:
            validated_params['llm_model'] = llm_model
            param_sources['llm_model'] = ParameterSource.FUNCTION_ARG
        else:
            validated_params['llm_model'] = None
            param_sources['llm_model'] = ParameterSource.SYSTEM_DEFAULT
        
        # Validate and track csv_output_file
        if csv_output_file is not None:
            validated_params['csv_output_file'] = csv_output_file
            param_sources['csv_output_file'] = ParameterSource.FUNCTION_ARG
        else:
            validated_params['csv_output_file'] = None
            param_sources['csv_output_file'] = ParameterSource.SYSTEM_DEFAULT
        
        # Validate and track benchmark_eval_mode
        validated_params['benchmark_eval_mode'] = bool(benchmark_eval_mode)
        param_sources['benchmark_eval_mode'] = ParameterSource.FUNCTION_ARG
        
        # Validate and track streaming
        validated_params['streaming'] = bool(streaming)
        param_sources['streaming'] = ParameterSource.FUNCTION_ARG
        
        # Validate and track max_files_per_request
        if max_files_per_request is not None:
            if max_files_per_request < 1:
                errors.append("max_files_per_request must be at least 1")
            validated_params['max_files_per_request'] = max_files_per_request
            param_sources['max_files_per_request'] = ParameterSource.FUNCTION_ARG
        else:
            validated_params['max_files_per_request'] = None
            param_sources['max_files_per_request'] = ParameterSource.SYSTEM_DEFAULT
        
        # Validate paths
        validated_params, param_sources, errors = self._validate_file_processing_paths(validated_params, param_sources, errors)
        
        # Log parameter summary
        self._log_parameter_summary(validated_params, param_sources)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            validated_params=validated_params,
            param_sources=param_sources
        )
    
    def _validate_paths(self, validated_params: Dict[str, Any], 
                       param_sources: Dict[str, ParameterSource], 
                       errors: List[str]) -> tuple:
        """Validate file paths for combo processing."""
        
        # Validate based on what's actually being used
        input_path = validated_params.get('input_pdf_dir_path')
        pdf_paths = validated_params.get('pdf_file_paths', [])
        
        if input_path:
            # Validate input_pdf_dir_path exists
            input_path_obj = Path(input_path) if isinstance(input_path, str) else input_path
            if not input_path_obj.exists():
                errors.append(f"Input directory does not exist: {input_path}")
            elif not input_path_obj.is_dir():
                errors.append(f"Input path is not a directory: {input_path}")
        
        if pdf_paths:
            # Validate each PDF file exists
            for pdf_path in pdf_paths:
                pdf_path_obj = Path(pdf_path) if isinstance(pdf_path, str) else pdf_path
                if not pdf_path_obj.exists():
                    errors.append(f"PDF file does not exist: {pdf_path}")
                elif not pdf_path_obj.is_file():
                    errors.append(f"Path is not a file: {pdf_path}")
        
        # Validate benchmark_file_path exists if provided
        benchmark_file_path = validated_params.get('benchmark_file_path')
        if benchmark_file_path:
            benchmark_path_obj = Path(benchmark_file_path) if isinstance(benchmark_file_path, str) else benchmark_file_path
            if not benchmark_path_obj.exists():
                errors.append(f"Benchmark file does not exist: {benchmark_file_path}")
            elif not benchmark_path_obj.is_file():
                errors.append(f"Benchmark path is not a file: {benchmark_file_path}")
        
        # Validate output_dir can be created
        output_dir = validated_params.get('output_dir')
        if output_dir:
            path = Path(output_dir)
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create output directory {output_dir}: {e}")
        
        return validated_params, param_sources, errors
    
    def _validate_file_processing_paths(self, validated_params: Dict[str, Any], 
                                      param_sources: Dict[str, ParameterSource], 
                                      errors: List[str]) -> tuple:
        """Validate file paths for file processing."""
        
        # Validate input_pdf_dir_path exists
        input_path = validated_params.get('input_pdf_dir_path')
        if input_path:
            input_path_obj = Path(input_path) if isinstance(input_path, str) else input_path
            if not input_path_obj.exists():
                errors.append(f"Input directory does not exist: {input_path}")
            elif not input_path_obj.is_dir():
                errors.append(f"Input path is not a directory: {input_path}")
        
        # Validate output_file directory can be created
        output_file = validated_params.get('output_file')
        if output_file:
            output_path = Path(output_file)
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create output file directory {output_path.parent}: {e}")
        
        # Validate checkpoint_file directory can be created
        checkpoint_file = validated_params.get('checkpoint_file')
        if checkpoint_file:
            checkpoint_path = Path(checkpoint_file)
            try:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create checkpoint file directory {checkpoint_path.parent}: {e}")
        
        return validated_params, param_sources, errors
    
    def _validate_benchmark_requirements(self, validated_params: Dict[str, Any], 
                                       param_sources: Dict[str, ParameterSource], 
                                       errors: List[str]) -> tuple:
        """Validate benchmark evaluation requirements."""
        
        benchmark_eval_mode = validated_params.get('benchmark_eval_mode', False)
        benchmark_file_path = validated_params.get('benchmark_file_path')
        
        if benchmark_eval_mode and not benchmark_file_path:
            errors.append("benchmark_file_path is required when benchmark_eval_mode=True")
        
        return validated_params, param_sources, errors
    
    def _log_parameter_summary(self, validated_params: Dict[str, Any], param_sources: Dict[str, ParameterSource]) -> None:
        """Log a compact summary of all parameters with their sources."""
        
        # Group parameters by source
        function_params = []
        config_params = []
        system_params = []
        
        for param, value in validated_params.items():
            source = param_sources.get(param, ParameterSource.SYSTEM_DEFAULT)
            
            # Format the parameter value for display
            if isinstance(value, (str, Path)) and len(str(value)) > 50:
                display_value = f"{str(value)[:30]}...{str(value)[-20:]}"
            else:
                display_value = str(value)
            
            param_display = f"{param}={display_value}"
            
            if source == ParameterSource.FUNCTION_ARG:
                function_params.append(param_display)
            elif source == ParameterSource.CONFIG_DEFAULT:
                config_params.append(param_display)
            else:  # SYSTEM_DEFAULT
                system_params.append(param_display)
        
        # Log the summary
        logger.info("🔍 PARAMETER VALIDATION SUMMARY:")
        
        if function_params:
            logger.info(f"   📤 FUNCTION: {', '.join(function_params)}")
        
        if config_params:
            logger.info(f"   ⚙️  CONFIG: {', '.join(config_params)}")
        
        if system_params:
            logger.info(f"   🔧 SYSTEM: {', '.join(system_params)}")
        
        logger.info("🔍 END PARAMETER SUMMARY")
