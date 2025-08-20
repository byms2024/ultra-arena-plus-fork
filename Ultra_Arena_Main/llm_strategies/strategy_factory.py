"""
Strategy factory for creating processing strategies.
"""

import logging
from typing import Dict, List, Any

from .base_strategy import BaseProcessingStrategy


class ProcessingStrategyFactory:
    """Factory for creating processing strategies."""
    
    @staticmethod
    def create_strategy(strategy_type: str, config: Dict[str, Any], streaming: bool = False) -> BaseProcessingStrategy:
        """Create a processing strategy based on type."""
        if strategy_type == "direct_file":
            from .direct_file_strategy import DirectFileProcessingStrategy
            return DirectFileProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "text_first":
            from .enhanced_text_first_strategy import EnhancedTextFirstProcessingStrategy
            return EnhancedTextFirstProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "image_first":
            from .image_first_strategy import ImageFirstProcessingStrategy
            return ImageFirstProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "hybrid":
            from .hybrid_strategy import HybridProcessingStrategy
            return HybridProcessingStrategy(config, streaming=streaming)
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategy types."""
        return ["direct_file", "text_first", "image_first", "hybrid"] 