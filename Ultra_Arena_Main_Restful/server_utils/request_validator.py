#!/usr/bin/env python3
"""
Request Validator for Ultra Arena RESTful API

This module provides validation logic for REST API requests and logs parameter sources.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ParameterSource(Enum):
    """Enum for tracking parameter sources."""
    REQUEST = "request"
    PROFILE_DEFAULT = "profile_default"
    BASE_DEFAULT = "base_default"


@dataclass
class ValidationResult:
    """Result of parameter validation."""
    is_valid: bool
    errors: List[str]
    validated_params: Dict[str, Any]
    param_sources: Dict[str, ParameterSource]


class RequestValidator:
    """Validates REST API requests and tracks parameter sources."""
    
    def __init__(self, config_defaults: Dict[str, Any]):
        """
        Initialize the request validator.
        
        Args:
            config_defaults: Default configuration values from profile
        """
        self.config_defaults = config_defaults
    
    # Static methods for backward compatibility
    @staticmethod
    def validate_json_request(data: Dict[str, Any]) -> tuple:
        """Validate that JSON data is provided (backward compatibility)."""
        if not data:
            return False, "No JSON data provided"
        return True, None
    
    @staticmethod
    def validate_combo_name(combo_name: str, available_combos: list) -> tuple:
        """Validate that the combo name exists (backward compatibility)."""
        if not combo_name:
            return False, "combo_name is required"
        
        if combo_name not in available_combos:
            return False, f"Invalid combo_name: '{combo_name}'. Available combos: {available_combos}"
        
        return True, None
    
    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: list) -> tuple:
        """Validate that all required fields are present (backward compatibility)."""
        missing_fields = []
        empty_fields = []
        
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
            elif data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
                empty_fields.append(field)
        
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"
        
        if empty_fields:
            return False, f"Required fields cannot be empty: {empty_fields}"
        
        return True, None
    
    @staticmethod
    def validate_file_paths(data: Dict[str, Any]) -> tuple:
        """Validate that file paths exist (backward compatibility)."""
        # Validate input directory
        if 'input_pdf_dir_path' in data:
            input_path = Path(data['input_pdf_dir_path'])
            if not input_path.exists():
                return False, f"Input directory does not exist: {input_path}"
            if not input_path.is_dir():
                return False, f"Input path is not a directory: {input_path}"
        
        # Validate benchmark file (if provided)
        if 'benchmark_file_path' in data and data['benchmark_file_path']:
            benchmark_path = Path(data['benchmark_file_path'])
            if not benchmark_path.exists():
                return False, f"Benchmark file does not exist: {benchmark_path}"
            if not benchmark_path.is_file():
                return False, f"Benchmark path is not a file: {benchmark_path}"
        
        return True, None
    
    def validate_combo_request(self, request_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate a combo processing request.
        
        Args:
            request_data: Raw request data from the API
            
        Returns:
            ValidationResult: Validation result with errors and validated parameters
        """
        errors = []
        validated_params = {}
        param_sources = {}
        
        # Validate required parameters
        validated_params, param_sources, errors = self._validate_required_params(request_data, errors, validated_params, param_sources)
        
        # Validate optional parameters with defaults
        validated_params, param_sources = self._validate_optional_params(request_data, validated_params, param_sources)
        
        # Validate paths
        validated_params, param_sources, errors = self._validate_paths(request_data, errors, validated_params, param_sources)
        
        # Log parameter summary
        self._log_parameter_summary(validated_params, param_sources)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            validated_params=validated_params,
            param_sources=param_sources
        )
    
    def _validate_required_params(self, request_data: Dict[str, Any], errors: List[str], 
                                validated_params: Dict[str, Any], param_sources: Dict[str, ParameterSource]) -> tuple:
        """Validate required parameters."""
        
        # combo_name is required
        combo_name = request_data.get('combo_name')
        if not combo_name:
            errors.append("Missing required parameter: combo_name")
        else:
            validated_params['combo_name'] = combo_name
            param_sources['combo_name'] = ParameterSource.REQUEST
        
        # Handle input_pdf_dir_path vs pdf_file_paths mutual exclusivity
        input_path = request_data.get('input_pdf_dir_path')
        pdf_paths = request_data.get('pdf_file_paths', [])
        
        if not input_path and not pdf_paths:
            errors.append("Either input_pdf_dir_path or pdf_file_paths is required")
        elif input_path and pdf_paths:
            # Both provided - pdf_file_paths takes precedence
            logger.info("🔄 Both input_pdf_dir_path and pdf_file_paths provided - using pdf_file_paths")
            validated_params['input_pdf_dir_path'] = None
            validated_params['pdf_file_paths'] = pdf_paths
            param_sources['input_pdf_dir_path'] = ParameterSource.SYSTEM_DEFAULT
            param_sources['pdf_file_paths'] = ParameterSource.REQUEST
        else:
            # Only one provided
            if input_path:
                validated_params['input_pdf_dir_path'] = input_path
                validated_params['pdf_file_paths'] = []
                param_sources['input_pdf_dir_path'] = ParameterSource.REQUEST
                param_sources['pdf_file_paths'] = ParameterSource.SYSTEM_DEFAULT
            else:
                validated_params['input_pdf_dir_path'] = None
                validated_params['pdf_file_paths'] = pdf_paths
                param_sources['input_pdf_dir_path'] = ParameterSource.SYSTEM_DEFAULT
                param_sources['pdf_file_paths'] = ParameterSource.REQUEST
        
        # output_dir is required
        output_dir = request_data.get('output_dir')
        if not output_dir:
            errors.append("Missing required parameter: output_dir")
        else:
            validated_params['output_dir'] = output_dir
            param_sources['output_dir'] = ParameterSource.REQUEST
        
        return validated_params, param_sources, errors
    
    def _validate_optional_params(self, request_data: Dict[str, Any], 
                                validated_params: Dict[str, Any], 
                                param_sources: Dict[str, ParameterSource]) -> tuple:
        """Validate optional parameters with defaults."""
        
        # streaming
        streaming = request_data.get('streaming')
        if streaming is not None:
            validated_params['streaming'] = bool(streaming)
            param_sources['streaming'] = ParameterSource.REQUEST
        else:
            validated_params['streaming'] = self.config_defaults.get('streaming', False)
            param_sources['streaming'] = ParameterSource.PROFILE_DEFAULT
        
        # max_cc_strategies
        max_cc_strategies = request_data.get('max_cc_strategies')
        if max_cc_strategies is not None:
            try:
                validated_params['max_cc_strategies'] = int(max_cc_strategies)
                param_sources['max_cc_strategies'] = ParameterSource.REQUEST
            except (ValueError, TypeError):
                validated_params['max_cc_strategies'] = self.config_defaults.get('max_cc_strategies', 3)
                param_sources['max_cc_strategies'] = ParameterSource.PROFILE_DEFAULT
        else:
            validated_params['max_cc_strategies'] = self.config_defaults.get('max_cc_strategies', 3)
            param_sources['max_cc_strategies'] = ParameterSource.PROFILE_DEFAULT
        
        # max_cc_filegroups
        max_cc_filegroups = request_data.get('max_cc_filegroups')
        if max_cc_filegroups is not None:
            try:
                validated_params['max_cc_filegroups'] = int(max_cc_filegroups)
                param_sources['max_cc_filegroups'] = ParameterSource.REQUEST
            except (ValueError, TypeError):
                validated_params['max_cc_filegroups'] = self.config_defaults.get('max_cc_filegroups', 5)
                param_sources['max_cc_filegroups'] = ParameterSource.PROFILE_DEFAULT
        else:
            validated_params['max_cc_filegroups'] = self.config_defaults.get('max_cc_filegroups', 5)
            param_sources['max_cc_filegroups'] = ParameterSource.PROFILE_DEFAULT
        
        # max_files_per_request
        max_files_per_request = request_data.get('max_files_per_request')
        if max_files_per_request is not None:
            try:
                validated_params['max_files_per_request'] = int(max_files_per_request)
                param_sources['max_files_per_request'] = ParameterSource.REQUEST
            except (ValueError, TypeError):
                validated_params['max_files_per_request'] = self.config_defaults.get('max_files_per_request', 10)
                param_sources['max_files_per_request'] = ParameterSource.PROFILE_DEFAULT
        else:
            validated_params['max_files_per_request'] = self.config_defaults.get('max_files_per_request', 10)
            param_sources['max_files_per_request'] = ParameterSource.PROFILE_DEFAULT
        
        # run_type
        run_type = request_data.get('run_type')
        if run_type:
            validated_params['run_type'] = run_type
            param_sources['run_type'] = ParameterSource.REQUEST
        else:
            validated_params['run_type'] = 'processing'  # Default
            param_sources['run_type'] = ParameterSource.BASE_DEFAULT
        
        # benchmark_file_path (optional for evaluation mode)
        benchmark_file_path = request_data.get('benchmark_file_path')
        if benchmark_file_path:
            validated_params['benchmark_file_path'] = benchmark_file_path
            param_sources['benchmark_file_path'] = ParameterSource.REQUEST
        
        return validated_params, param_sources
    
    def _validate_paths(self, request_data: Dict[str, Any], errors: List[str], 
                       validated_params: Dict[str, Any], param_sources: Dict[str, ParameterSource]) -> tuple:
        """Validate file paths."""
        
        # Validate based on what's actually being used
        input_path = validated_params.get('input_pdf_dir_path')
        pdf_paths = validated_params.get('pdf_file_paths', [])
        
        if input_path:
            # Validate input_pdf_dir_path exists
            path = Path(input_path)
            if not path.exists():
                errors.append(f"Input directory does not exist: {input_path}")
            elif not path.is_dir():
                errors.append(f"Input path is not a directory: {input_path}")
        
        if pdf_paths:
            # Validate each PDF file exists
            for pdf_path in pdf_paths:
                path = Path(pdf_path)
                if not path.exists():
                    errors.append(f"PDF file does not exist: {pdf_path}")
                elif not path.is_file():
                    errors.append(f"Path is not a file: {pdf_path}")
        
        # Validate benchmark_file_path exists if provided
        benchmark_file_path = validated_params.get('benchmark_file_path')
        if benchmark_file_path:
            path = Path(benchmark_file_path)
            if not path.exists():
                errors.append(f"Benchmark file does not exist: {benchmark_file_path}")
            elif not path.is_file():
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
    
    def _log_parameter_summary(self, validated_params: Dict[str, Any], param_sources: Dict[str, ParameterSource]) -> None:
        """Log a compact summary of all parameters with their sources."""
        
        # Group parameters by source
        request_params = []
        profile_params = []
        base_params = []
        
        for param, value in validated_params.items():
            source = param_sources.get(param, ParameterSource.BASE_DEFAULT)
            
            # Format the parameter value for display
            if isinstance(value, (str, Path)) and len(str(value)) > 50:
                display_value = f"{str(value)[:30]}...{str(value)[-20:]}"
            else:
                display_value = str(value)
            
            param_display = f"{param}={display_value}"
            
            if source == ParameterSource.REQUEST:
                request_params.append(param_display)
            elif source == ParameterSource.PROFILE_DEFAULT:
                profile_params.append(param_display)
            else:  # BASE_DEFAULT
                base_params.append(param_display)
        
        # Log the summary
        logger.info("🔍 PARAMETER VALIDATION SUMMARY:")
        
        if request_params:
            logger.info(f"   📤 REQUEST: {', '.join(request_params)}")
        
        if profile_params:
            logger.info(f"   ⚙️  PROFILE: {', '.join(profile_params)}")
        
        if base_params:
            logger.info(f"   🔧 BASE: {', '.join(base_params)}")
        
        logger.info("🔍 END PARAMETER SUMMARY")
