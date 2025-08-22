#!/usr/bin/env python3
"""
RESTful API Server for Ultra Arena Main

This module provides a REST API interface to the Ultra_Arena_Main library.
It exposes endpoints for file processing and combo operations.

The server is organized into modular components:
- ConfigManager: Handles configuration management
- RequestValidator: Handles request validation
- RequestProcessor: Handles request processing
"""

import sys
import os
import logging
import time
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps

# Add the Ultra_Arena_Main to the Python path
ultra_arena_main_path = Path(__file__).parent.parent / "Ultra_Arena_Main"
sys.path.insert(0, str(ultra_arena_main_path))

# Import the main functions from Ultra_Arena_Main
from main_modular import setup_logging

# Import our modular components
from server_utils.config_manager import ConfigManager
from server_utils.request_validator import RequestValidator
from server_utils.request_processor import RequestProcessor
from server_utils.async_task_manager import task_manager

# Setup Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
setup_logging(verbose=True)
logger = logging.getLogger(__name__)

# Get server configuration from environment
RUN_PROFILE = os.environ.get('RUN_PROFILE', 'default_profile_restful')
logger.info(f"Server configured to use profile: {RUN_PROFILE}")

# Initialize modular components
config_manager = ConfigManager(RUN_PROFILE)
request_processor = RequestProcessor(config_manager)


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

# Import the new performance monitoring system
from performance_measure import (
    start_monitoring,
    end_monitoring,
    time_operation,
    mark_point,
    track_memory
)


# =============================================================================
# HEALTH CHECK ENDPOINT
# =============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Ultra Arena Main RESTful API",
        "version": "1.0.0"
    })


# =============================================================================
# PROCESSING ENDPOINTS
# =============================================================================


@app.route('/api/process/combo', methods=['POST'])
def process_combo():
    """
    Enhanced combo processing with consistent parameter naming and strict validation.
    
    Expected JSON payload (flattened structure):
    {
        "combo_name": "combo_test_10_strategies",  // Required
        "input_pdf_dir_path": "/path/to/input",  // Required
        "output_dir": "/path/to/output",  // Required
        "run_type": "normal",  // optional, defaults to "normal"
        "streaming": false,  // optional, defaults to config DEFAULT_STREAMING
        "max_cc_strategies": 3,  // optional, defaults to config DEFAULT_MAX_CC_STRATEGIES
        "max_cc_filegroups": 5,  // optional, defaults to config DEFAULT_MAX_CC_FILEGROUPS
        "max_files_per_request": 10  // optional, defaults to config DEFAULT_MAX_FILES_PER_REQUEST
    }
    
    For evaluation runs:
    {
        "combo_name": "combo_test_10_strategies",  // Required
        "input_pdf_dir_path": "/path/to/input",  // Required
        "output_dir": "/path/to/output",  // Required
        "run_type": "evaluation",  // optional, defaults to "normal"
        "benchmark_file_path": "/path/to/benchmark.xlsx"  // Required for evaluation mode
    }
    
    Note: Defaults can be overridden in profile_config.py
    """
    # Start performance monitoring
    monitor = start_monitoring("REST_API_Process_Combo")
    track_memory("request_start")
    
    try:
        logger.info(f"🚀 STARTING /api/process/combo request")
        
        with time_operation("http_request_parsing", "http_request_parsing"):
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            # Validate request using new validator
            config_defaults = config_manager.get_config_defaults()
            validator = RequestValidator(config_defaults)
            validation_result = validator.validate_combo_request(data)
            
            if not validation_result.is_valid:
                error_msg = "; ".join(validation_result.errors)
                return jsonify({"error": f"Validation failed: {error_msg}"}), 400
            
            # Use validated parameters
            data = validation_result.validated_params
        
        track_memory("after_request_parsing")
        
        with time_operation("configuration_setup", "configuration_setup"):
            try:
                config = request_processor.create_unified_request_config(data, use_default_combo=False)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
        
        track_memory("after_config_setup")
        
        with time_operation("main_library_processing", "main_library_processing"):
            result_code = request_processor.execute_processing(config)
        
        track_memory("after_main_processing")
        
        with time_operation("response_formatting", "response_formatting"):
            response = request_processor.format_combo_response(result_code, config)
        
        track_memory("request_end")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ ERROR in /api/process/combo endpoint: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        end_monitoring()


@app.route('/api/process/combo/async', methods=['POST'])
def process_combo_async():
    """
    Asynchronous combo processing endpoint that returns immediately with task ID.
    
    Expected JSON payload (same as /api/process/combo):
    {
        "combo_name": "combo_test_10_strategies",  // Required
        "input_pdf_dir_path": "/path/to/input",  // Required
        "output_dir": "/path/to/output",  // Required
        "run_type": "normal",  // optional, defaults to "normal"
        "streaming": false,  // optional, defaults to config DEFAULT_STREAMING
        "max_cc_strategies": 3,  // optional, defaults to config DEFAULT_MAX_CC_STRATEGIES
        "max_cc_filegroups": 5,  // optional, defaults to config DEFAULT_MAX_CC_FILEGROUPS
        "max_files_per_request": 10  // optional, defaults to config DEFAULT_MAX_FILES_PER_REQUEST
    }
    
    Returns:
        HTTP 202 Accepted with task ID for tracking
    """
    try:
        logger.info(f"🚀 STARTING /api/process/combo/async request")
        
        # Parse request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate request using new validator
        config_defaults = config_manager.get_config_defaults()
        validator = RequestValidator(config_defaults)
        validation_result = validator.validate_combo_request(data)
        
        if not validation_result.is_valid:
            error_msg = "; ".join(validation_result.errors)
            return jsonify({"error": f"Validation failed: {error_msg}"}), 400
        
        # Use validated parameters
        validated_data = validation_result.validated_params
        
        # Create configuration
        try:
            config = request_processor.create_unified_request_config(validated_data, use_default_combo=False)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        
        # Get request_id from config
        request_id = config.get("request_metadata", {}).get("request_id", "unknown")
        
        # Create async task with validated data
        task_manager.create_task(validated_data, config_manager, request_id)
        
        # Format response
        response_data = request_processor.format_async_combo_response(request_id, config)
        
        return jsonify(response_data), 202
        
    except Exception as e:
        import traceback
        logger.error(f"❌ ERROR in /api/process/combo/async endpoint: {e}")
        logger.error(f"❌ FULL STACK TRACE: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/requests/<request_id>', methods=['GET'])
def get_request_status(request_id):
    """
    Get the status of an async request.
    
    Args:
        request_id: The request ID from the async processing request
        
    Returns:
        JSON with request status and results (if completed)
    """
    try:
        logger.info(f"🔍 Getting status for request: {request_id}")
        
        request_info = task_manager.get_request_status(request_id)
        if not request_info:
            return jsonify({"error": "Request not found"}), 404
        
        # Format response based on request status
        if request_info["status"] == "complete":
            # Include performance and results for completed requests
            response_data = {
                "status": "complete",
                "request_id": request_id,
                "created_at": request_info["created_at"],
                "completed_at": request_info["completed_at"],
                "progress": request_info["progress"],
                "total_files_of_all_strategies_to_process": request_info.get("total_files_of_all_strategies_to_process", 0),
                # "total_files_of_all_strategies_processed": request_info.get("total_files_of_all_strategies_processed", 0),
                "performance": {
                    "configuration_assembly_time_ms": 45.2,  # This would come from actual processing
                    "server_config_cached": True
                },
                "results": request_info["result"]
            }
        elif request_info["status"] == "incomplete":
            # Include progress for incomplete requests
            response_data = {
                "status": "incomplete",
                "request_id": request_id,
                "created_at": request_info["created_at"],
                "progress": request_info["progress"],
                "total_files_of_all_strategies_to_process": request_info.get("total_files_of_all_strategies_to_process", 0),
                # "total_files_of_all_strategies_processed": request_info.get("total_files_of_all_strategies_processed", 0)
            }
        elif request_info["status"] == "failed":
            # Include error for failed requests
            response_data = {
                "status": "failed",
                "request_id": request_id,
                "created_at": request_info["created_at"],
                "failed_at": request_info["failed_at"],
                "progress": request_info["progress"],
                "error": request_info["error"]
            }
        else:
            # For queued/processing requests
            response_data = {
                "status": request_info["status"],
                "request_id": request_id,
                "created_at": request_info["created_at"],
                "progress": request_info["progress"],
                "total_files_of_all_strategies_to_process": request_info.get("total_files_of_all_strategies_to_process", 0),
                # "total_files_of_all_strategies_processed": request_info.get("total_files_of_all_strategies_processed", 0)
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ ERROR getting request status for {request_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/requests', methods=['GET'])
def get_all_requests():
    """
    Get all async requests.
    
    Returns:
        JSON with all requests and their statuses
    """
    try:
        logger.info("🔍 Getting all requests")
        
        requests = task_manager.get_all_tasks()  # Still using the same method, just renamed for clarity
        
        # Clean up old requests periodically
        task_manager.cleanup_completed_tasks()
        
        return jsonify({
            "status": "success",
            "requests": requests,
            "count": len(requests)
        })
        
    except Exception as e:
        logger.error(f"❌ ERROR getting all requests: {e}")
        return jsonify({"error": str(e)}), 500


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@app.route('/api/combos', methods=['GET'])
def get_combos():
    """
    Endpoint 3.1.3: GET API that exposes available combos
    """
    # Start performance monitoring
    monitor = start_monitoring("REST_API_Get_Combos")
    track_memory("request_start")
    
    try:
        logger.info(f"🚀 STARTING /api/combos request")
        
        with time_operation("combo_retrieval", "combo_retrieval"):
            logger.info(f"Getting available combos for profile: {RUN_PROFILE}")
            combos = config_manager.get_available_combos()
        
        track_memory("after_combo_retrieval")
        
        with time_operation("response_formatting", "response_formatting"):
            response = jsonify({
                "status": "success",
                "combos": combos,
                "count": len(combos)
            })
        
        track_memory("request_end")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ ERROR in /api/combos endpoint: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        end_monitoring()


# =============================================================================
# SERVER STARTUP
# =============================================================================

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5002))
    
    logger.info(f"Starting Ultra Arena Main RESTful API server on port {port}")
    logger.info(f"Using profile: {RUN_PROFILE}")
    logger.info("Available endpoints:")
    logger.info("  GET  /health - Health check")
    logger.info("  POST /api/process/combo - Process combo (synchronous)")
    logger.info("  POST /api/process/combo/async - Process combo (asynchronous)")
    logger.info("  GET  /api/requests/<request_id> - Get request status")
    logger.info("  GET  /api/requests - Get all requests")
    logger.info("  GET  /api/combos - Get available combos")
    
    # Configure Flask for production use (no debug mode to prevent timeouts)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True) 