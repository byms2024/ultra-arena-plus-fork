#!/usr/bin/env python3
"""
Async Task Manager for Ultra Arena RESTful API

This module handles asynchronous processing of combo requests using threading.
"""

import threading
import time
import uuid
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class AsyncTaskManager:
    """Manages asynchronous processing tasks for the REST API."""
    
    def __init__(self):
        """Initialize the async task manager."""
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.task_lock = threading.Lock()
    
    def create_task(self, request_data: Dict[str, Any], config_manager, request_id: str = None) -> str:
        """
        Create a new async task and start processing in background.
        
        Args:
            request_data: The request data from the API
            config_manager: The configuration manager instance
            request_id: The request ID (if None, will generate one)
            
        Returns:
            str: Request ID for tracking
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # Calculate total work units (files * strategies)
        num_files = len(request_data.get('pdf_file_paths', [])) if 'pdf_file_paths' in request_data else 0
        num_strategies = len(request_data.get('strategy_groups', [])) if 'strategy_groups' in request_data else 0
        
        # If we don't have the data yet, we'll calculate it during processing
        logger.info(f"🔍 DEBUG: Initial calculation - num_files: {num_files}, num_strategies: {num_strategies}")
        total_files_of_all_strategies_to_process = num_files * num_strategies if num_files > 0 and num_strategies > 0 else 0
        logger.info(f"🔍 DEBUG: Initial total_files_of_all_strategies_to_process: {total_files_of_all_strategies_to_process}")
        
        # Create task record
        task_info = {
            "request_id": request_id,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "request_data": request_data,
            "progress": 0,
            "total_files_of_all_strategies_to_process": total_files_of_all_strategies_to_process,
            # "total_files_of_all_strategies_processed": 0,
            "result": None,
            "error": None
        }
        
        with self.task_lock:
            self.tasks[request_id] = task_info

        # Start processing in background thread
        thread = threading.Thread(
            target=self._process_task,
            args=(request_id, request_data, config_manager),
            daemon=True
        )
        thread.start()
        
        logger.info(f"🚀 Created async task {request_id} for combo: {request_data.get('combo_name')}")
        return request_id
    
    def _process_task(self, request_id: str, request_data: Dict[str, Any], config_manager):
        """
        Process a task in the background.
        
        Args:
            request_id: The request ID
            request_data: The request data
            config_manager: The configuration manager instance
        """
        # Initialize the variable in this scope
        total_files_of_all_strategies_to_process = 0
        
        try:
            # Update status to processing
            with self.task_lock:
                self.tasks[request_id]["status"] = "processing"
                self.tasks[request_id]["progress"] = 0
            
            logger.info(f"🔄 Starting processing for request {request_id}")
            
            # Import the main processing module
            import sys
            import os
            # Add the parent directory to the path to import Ultra_Arena_Main
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
            from Ultra_Arena_Main.main_modular import run_combo_processing

            # Get the actual configuration to calculate total work units
            # TODO: Implement proper progress tracking
            try:
                # We need to get the actual file count and strategy count
                from Ultra_Arena_Main.config.config_combo_run import combo_config

                if request_data.get('chain_name'):
                    combo_name = request_data.get('chain_name')
                else:
                    combo_name = request_data.get('combo_name')
                

                if combo_name and combo_name in combo_config:
                    strategy_groups = combo_config[combo_name].get("strategy_groups", [])
                    num_strategies = len(strategy_groups)
                    
                    # Get file count from the input directory
                    input_path = Path(request_data.get('input_pdf_dir_path', ''))
                    if input_path.exists():
                        pdf_files = list(input_path.glob("*.pdf"))
                        num_files = len(pdf_files)
                        
                        # Update total work units
                        logger.info(f"🔍 DEBUG: Before recalculation - total_files_of_all_strategies_to_process: {total_files_of_all_strategies_to_process}")
                        total_files_of_all_strategies_to_process = num_files * num_strategies
                        logger.info(f"🔍 DEBUG: After recalculation - total_files_of_all_strategies_to_process: {total_files_of_all_strategies_to_process}")
                        with self.task_lock:
                            self.tasks[request_id]["total_files_of_all_strategies_to_process"] = total_files_of_all_strategies_to_process
                        
                        logger.info(f"📊 Calculated total files of all strategies to process: {num_files} files × {num_strategies} strategies = {total_files_of_all_strategies_to_process}")
                    else:
                        logger.warning(f"⚠️ Input path does not exist: {input_path}")
                        logger.info(f"🔍 DEBUG: Setting total_files_of_all_strategies_to_process to 0 (path not found)")
                        total_files_of_all_strategies_to_process = 0
                else:
                    logger.warning(f"⚠️ Combo not found: {combo_name}")
                    logger.info(f"🔍 DEBUG: Setting total_files_of_all_strategies_to_process to 0 (combo not found)")
                    total_files_of_all_strategies_to_process = 0
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not calculate total files of all strategies to process: {e}")
                logger.info(f"🔍 DEBUG: Setting total_files_of_all_strategies_to_process to 0 (exception)")
                total_files_of_all_strategies_to_process = 0
            
            # Execute the processing
            logger.info(f"🔍 DEBUG: About to call run_combo_processing")
            logger.info(f"🔍 DEBUG: total_files_of_all_strategies_to_process before call: {total_files_of_all_strategies_to_process}")
            
            # Get configuration defaults from config_manager
            config_defaults = config_manager.get_config_defaults()
        
            try:
                if request_data.get('chain_name'):
                    combo_name = request_data.get('chain_name')
                    chain_config = request_data.get('chain_config')
                else:
                    combo_name = request_data.get('combo_name')
                    chain_config = None

                print('⚙️'*100)
                print(chain_config)
                print('⚙️'*100)

                result_code = run_combo_processing(
                    benchmark_eval_mode=request_data.get('run_type') == 'evaluation',
                    combo_name=combo_name,
                    streaming=request_data.get('streaming', config_defaults.get('streaming', False)),
                    max_cc_strategies=request_data.get('max_cc_strategies', config_defaults.get('max_cc_strategies', 3)),
                    max_cc_filegroups=request_data.get('max_cc_filegroups', config_defaults.get('max_cc_filegroups', 5)),
                    max_files_per_request=request_data.get('max_files_per_request', config_defaults.get('max_files_per_request', 10)),
                    input_pdf_dir_path=request_data.get('input_pdf_dir_path'),
                    pdf_file_paths=request_data.get('pdf_file_paths', []),
                    output_dir=request_data.get('output_dir'),
                    benchmark_file_path=Path(request_data.get('benchmark_file_path')) if request_data.get('benchmark_file_path') else None,
                    config_manager = config_manager,
                    chain_config= chain_config
                )
                logger.info(f"🔍 DEBUG: run_combo_processing completed with result_code: {result_code}")
            except Exception as e:
                import traceback
                full_traceback = traceback.format_exc()
                logger.error(f"❌ ERROR in run_combo_processing: {e}")
                logger.error(f"❌ ERROR type: {type(e)}")
                logger.error(f"❌ FULL STACK TRACE:\n{full_traceback}")
                raise
            
            # Calculate final progress based on actual completion
            # TODO: Implement real progress tracking during processing
            # For now, we'll assume completion, but in a real implementation,
            # we'd track actual progress during processing
            # total_files_of_all_strategies_processed = total_files_of_all_strategies_to_process if total_files_of_all_strategies_to_process > 0 else 1
            
            # Add verbose logging to debug the error
            logger.info(f"🔍 DEBUG: total_files_of_all_strategies_to_process = {total_files_of_all_strategies_to_process} (type: {type(total_files_of_all_strategies_to_process)})")
            logger.info(f"🔍 DEBUG: About to compare total_files_of_all_strategies_to_process > 0")
            
            # Check if the variable is None and handle it
            if total_files_of_all_strategies_to_process is None:
                logger.error(f"❌ ERROR: total_files_of_all_strategies_to_process is None!")
                progress = 100  # Fallback value
            else:
                try:
                    progress = 100 if total_files_of_all_strategies_to_process > 0 else 100
                    logger.info(f"🔍 DEBUG: Progress calculation successful: {progress}")
                except Exception as e:
                    logger.error(f"❌ ERROR in progress calculation: {e}")
                    logger.error(f"❌ total_files_of_all_strategies_to_process value: {total_files_of_all_strategies_to_process}")
                    logger.error(f"❌ total_files_of_all_strategies_to_process type: {type(total_files_of_all_strategies_to_process)}")
                    progress = 100  # Fallback value
            
            # Update task with results
            with self.task_lock:
                self.tasks[request_id]["status"] = "complete" if result_code == 0 else "failed"
                self.tasks[request_id]["progress"] = progress
                # self.tasks[request_id]["total_files_of_all_strategies_processed"] = total_files_of_all_strategies_processed
                # result_code is an integer, so we'll store it as a simple result
                self.tasks[request_id]["result"] = {"return_code": result_code, "status": "success" if result_code == 0 else "error"}
                self.tasks[request_id]["completed_at"] = datetime.utcnow().isoformat() + "Z"
            
            # Log completion status based on actual result
            if result_code == 0:
                logger.info(f"✅ Request {request_id} completed successfully")
            else:
                logger.error(f"🚨 Request {request_id} completed with errors (exit code: {result_code})")
            
        except Exception as e:
            logger.error(f"❌ Request {request_id} failed: {e}")
            
            # Update task with error
            with self.task_lock:
                self.tasks[request_id]["status"] = "failed"
                self.tasks[request_id]["progress"] = 0
                self.tasks[request_id]["error"] = str(e)
                self.tasks[request_id]["failed_at"] = datetime.utcnow().isoformat() + "Z"
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a request.
        
        Args:
            request_id: The request ID
            
        Returns:
            Dict[str, Any]: Request status information or None if not found
        """
        with self.task_lock:
            return self.tasks.get(request_id)
    
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all tasks.
        
        Returns:
            Dict[str, Dict[str, Any]]: All tasks
        """
        with self.task_lock:
            return self.tasks.copy()
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """
        Clean up completed tasks older than specified hours.
        
        Args:
            max_age_hours: Maximum age in hours for completed tasks
        """
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        
        with self.task_lock:
            tasks_to_remove = []
            for task_id, task_info in self.tasks.items():
                if task_info["status"] in ["completed", "failed"]:
                    # Parse the completion time
                    completion_time_str = task_info.get("completed_at") or task_info.get("failed_at")
                    if completion_time_str:
                        try:
                            completion_time = datetime.fromisoformat(completion_time_str.replace("Z", "+00:00"))
                            if completion_time.timestamp() < cutoff_time:
                                tasks_to_remove.append(task_id)
                        except ValueError:
                            # If we can't parse the time, remove the task
                            tasks_to_remove.append(task_id)
            
            # Remove old tasks
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
                logger.info(f"🧹 Cleaned up old task {task_id}")


# Global instance
task_manager = AsyncTaskManager()
