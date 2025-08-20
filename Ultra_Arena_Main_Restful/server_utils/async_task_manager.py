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
        total_files_of_all_strategies_to_process = num_files * num_strategies if num_files > 0 and num_strategies > 0 else 0
        
        # Create task record
        task_info = {
            "request_id": request_id,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "request_data": request_data,
            "progress": 0,
            "total_files_of_all_strategies_to_process": total_files_of_all_strategies_to_process,
            "total_files_of_all_strategies_processed": 0,
            "result": None,
            "error": None
        }
        
        with self.task_lock:
            self.tasks[task_id] = task_info
        
        # Start processing in background thread
        thread = threading.Thread(
            target=self._process_task,
            args=(task_id, request_data, config_manager),
            daemon=True
        )
        thread.start()
        
        logger.info(f"🚀 Created async task {task_id} for combo: {request_data.get('combo_name')}")
        return task_id
    
    def _process_task(self, request_id: str, request_data: Dict[str, Any], config_manager):
        """
        Process a task in the background.
        
        Args:
            request_id: The request ID
            request_data: The request data
            config_manager: The configuration manager instance
        """
        try:
            # Update status to processing
            with self.task_lock:
                self.tasks[request_id]["status"] = "processing"
                self.tasks[request_id]["progress"] = 0
            
            logger.info(f"🔄 Starting processing for request {request_id}")
            
            # Import the main processing module
            from Ultra_Arena_Main.main_modular import main_modular_processing
            
            # Get the actual configuration to calculate total work units
            try:
                # We need to get the actual file count and strategy count
                from Ultra_Arena_Main.config.config_combo_run import combo_config
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
                        total_files_of_all_strategies_to_process = num_files * num_strategies
                        with self.task_lock:
                            self.tasks[request_id]["total_files_of_all_strategies_to_process"] = total_files_of_all_strategies_to_process
                        
                        logger.info(f"📊 Calculated total files of all strategies to process: {num_files} files × {num_strategies} strategies = {total_files_of_all_strategies_to_process}")
                    else:
                        logger.warning(f"⚠️ Input path does not exist: {input_path}")
                        total_files_of_all_strategies_to_process = 0
                else:
                    logger.warning(f"⚠️ Combo not found: {combo_name}")
                    total_files_of_all_strategies_to_process = 0
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not calculate total files of all strategies to process: {e}")
                total_files_of_all_strategies_to_process = 0
            
            # Execute the processing
            result_code = main_modular_processing(
                combo_name=request_data.get('combo_name'),
                input_pdf_dir_path=request_data.get('input_pdf_dir_path'),
                output_dir=request_data.get('output_dir'),
                run_type=request_data.get('run_type', 'normal'),
                streaming=request_data.get('streaming', False),
                max_cc_strategies=request_data.get('max_cc_strategies'),
                max_cc_filegroups=request_data.get('max_cc_filegroups'),
                max_files_per_request=request_data.get('max_files_per_request'),
                benchmark_file_path=request_data.get('benchmark_file_path')
            )
            
            # Calculate final progress based on actual completion
            # For now, we'll assume completion, but in a real implementation,
            # we'd track actual progress during processing
            total_files_of_all_strategies_processed = total_files_of_all_strategies_to_process if total_files_of_all_strategies_to_process > 0 else 1
            progress = 100 if total_files_of_all_strategies_to_process > 0 else 100
            
            # Update task with results
            with self.task_lock:
                self.tasks[request_id]["status"] = "complete" if progress >= 100 else "incomplete"
                self.tasks[request_id]["progress"] = progress
                self.tasks[request_id]["total_files_of_all_strategies_processed"] = total_files_of_all_strategies_processed
                self.tasks[request_id]["result"] = result_code
                self.tasks[request_id]["completed_at"] = datetime.utcnow().isoformat() + "Z"
            
            logger.info(f"✅ Request {request_id} completed successfully")
            
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
