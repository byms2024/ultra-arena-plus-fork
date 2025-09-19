"""
Centralized Chain definitions for REST payloads.

Map a chain_name to its steps and options so REST requests can specify
"chain_name" without editing param groups.
"""

from typing import Dict, Any, List

chain_definitions: Dict[str, Dict[str, Any]] = {
    # Basic 2-step chain: try text_first first, fallback to direct_file
    "test_1": {
        "chain_steps": [
            # Prefer local model for text_first to avoid remote API key requirements in basic tests
            {"type": "text_first", "overrides": {"llm_provider": "google"}},
            {"type": "direct_file", "overrides": {"llm_provider": "google"}},
        ],
        "chain_on_missing_keys": False,
    },
}


