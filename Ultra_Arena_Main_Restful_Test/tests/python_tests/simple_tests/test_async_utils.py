#!/usr/bin/env python3
"""
Async Test Utilities for Ultra Arena Main RESTful Tests

This module provides common utilities for running asynchronous combo processing tests
against the REST API. It eliminates code duplication across test files.
"""

import requests
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum


class TestStatus(Enum):
    """Enumeration for test status values."""
    COMPLETE = "complete"
    COMPLETED = "completed"
    FAILED = "failed"
    RUNNING = "running"
    PROCESSING = "processing"
    QUEUED = "queued"


@dataclass
class TestConfig:
    """Configuration for async test runs."""
    combo_name: str
    file_name: str
    max_wait_time: int = 3000  # 50 minutes default
    poll_interval: int = 10    # 10 seconds default
    max_cc_strategies: int = 5
    max_cc_filegroups: int = 5
    max_files_per_request: int = 10
    base_url: str = "http://localhost:5002"
    run_type: str = "evaluation"
    streaming: bool = False
    timeout: int = 30


class AsyncTestRunner:
    """Main class for running async tests with common functionality."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.api_endpoint = f"{config.base_url}/api/process/combo/async"
        self.script_dir = Path(__file__).parent.parent.parent.parent
        self.test_fixtures_dir = self.script_dir / "test_fixtures" / "br_fixture"
        self.input_pdf_dir_path = self.test_fixtures_dir / "input_files" / config.file_name
        
    def validate_input_directory(self) -> bool:
        """Validate that the input directory exists."""
        if not self.input_pdf_dir_path.exists():
            print(f"❌ Error: Input directory not found: {self.input_pdf_dir_path}")
            return False
        return True
    
    def build_payload(self) -> Dict[str, Any]:
        """Build the request payload for the API call."""
        return {
            'combo_name': self.config.combo_name,
            'input_pdf_dir_path': str(self.input_pdf_dir_path),
            'run_type': self.config.run_type,
            'output_dir': str(self.test_fixtures_dir / "output_files"),
            'benchmark_file_path': str(self.test_fixtures_dir / "benchmark_files" / "benchmark_252.csv"),
            'streaming': self.config.streaming,
            'max_cc_strategies': self.config.max_cc_strategies,
            'max_cc_filegroups': self.config.max_cc_filegroups,
            'max_files_per_request': self.config.max_files_per_request
        }
    
    def print_test_info(self):
        """Print test configuration information."""
        print(f"🚀 Ultra Arena Main - Async Test")
        print(f"Combo: {self.config.combo_name}")
        print(f"Files: {self.config.file_name}")
        print("=" * 80)
        print(f"📁 Input Directory: {self.input_pdf_dir_path}")
        print(f"📊 Benchmark File: {self.test_fixtures_dir / 'benchmark_files' / 'benchmark_252.csv'}")
        print(f"📤 Output Directory: {self.test_fixtures_dir / 'output_files'}")
        print()
    
    def submit_task(self, payload: Dict[str, Any]) -> Optional[str]:
        """Submit the async task and return the request/task ID."""
        try:
            print("🔄 Submitting async task...")
            response = requests.post(self.api_endpoint, json=payload, timeout=self.config.timeout)
            
            if response.status_code == 202:
                print("✅ Task submitted successfully!")
                response_data = response.json()
                
                # Handle both request_id and task_id formats
                request_id = response_data.get('request_id') or response_data.get('task_id')
                if request_id:
                    print(f"📋 Request ID: {request_id}")
                    return request_id
                else:
                    print("❌ No request/task ID found in response")
                    print(f"📋 Full Response: {response_data}")
                    return None
            else:
                print(f"❌ Task submission failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    def get_status_endpoint(self, request_id: str) -> str:
        """Get the appropriate status endpoint based on the request ID format."""
        # Try to determine if it's a request_id or task_id based on endpoint availability
        # For now, try the request endpoint first, then fallback to task endpoint
        return f"{self.config.base_url}/api/requests/{request_id}"
    
    def is_completion_status(self, status: str) -> bool:
        """Check if the status indicates completion."""
        return status in [TestStatus.COMPLETE.value, TestStatus.COMPLETED.value]
    
    def is_failure_status(self, status: str) -> bool:
        """Check if the status indicates failure."""
        return status == TestStatus.FAILED.value
    
    def is_processing_status(self, status: str) -> bool:
        """Check if the status indicates the task is still processing."""
        return status in [TestStatus.RUNNING.value, TestStatus.PROCESSING.value, TestStatus.QUEUED.value]
    
    def poll_for_completion(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Poll for task completion and return the final result."""
        status_endpoint = self.get_status_endpoint(request_id)
        elapsed_time = 0
        
        print("\n🔄 Polling for task completion...")
        while elapsed_time < self.config.max_wait_time:
            try:
                status_response = requests.get(status_endpoint, timeout=10)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get('status')
                    progress = status_data.get('progress', 0)
                    
                    print(f"⏱️  Elapsed: {elapsed_time}s | Status: {status} | Progress: {progress}%")
                    
                    if self.is_completion_status(status):
                        print("✅ Task completed successfully!")
                        result = status_data.get('result', {})
                        print(f"📊 Results: {json.dumps(result, indent=2)}")
                        return result
                    elif self.is_failure_status(status):
                        print("❌ Task failed!")
                        error = status_data.get('error', 'Unknown error')
                        print(f"🚨 Error: {error}")
                        return None
                    elif self.is_processing_status(status):
                        # Continue polling
                        pass
                    else:
                        print(f"⚠️  Unknown status: {status}")
                        return None
                else:
                    print(f"⚠️  Status check failed: {status_response.status_code}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Status check error: {e}")
                return None
            
            time.sleep(self.config.poll_interval)
            elapsed_time += self.config.poll_interval
        
        print("⏰ Timeout: Task did not complete within the expected time")
        return None
    
    def run_test(self) -> Optional[Dict[str, Any]]:
        """Run the complete async test."""
        # Validate input directory
        if not self.validate_input_directory():
            return None
        
        # Print test information
        self.print_test_info()
        
        # Build payload
        payload = self.build_payload()
        
        # Submit task
        request_id = self.submit_task(payload)
        if not request_id:
            return None
        
        # Poll for completion
        return self.poll_for_completion(request_id)


def run_async_test(config: TestConfig) -> Optional[Dict[str, Any]]:
    """Convenience function to run an async test with the given configuration."""
    runner = AsyncTestRunner(config)
    return runner.run_test()
