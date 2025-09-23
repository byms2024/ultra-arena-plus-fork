"""
Strategy factory for creating processing strategies.
"""

import logging
from typing import Dict, List, Any

from .base_strategy import BaseProcessingStrategy


class ProcessingStrategyFactory:
    """Factory for creating processing strategies."""
    
    @staticmethod
    def create_strategy(strategy_type: str, config: Dict[str, Any], streaming: bool = False, database_ops = None) -> BaseProcessingStrategy:
        """Create a processing strategy based on type."""
        if strategy_type == "direct_file":
            from .direct_file_strategy import DirectFileProcessingStrategy
            return DirectFileProcessingStrategy(config, streaming=streaming, database_ops=database_ops)
        elif strategy_type == "text_first":
            from .enhanced_text_first_strategy import EnhancedTextFirstProcessingStrategy
            return EnhancedTextFirstProcessingStrategy(config, streaming=streaming, database_ops=database_ops)
        elif strategy_type == "image_first":
            from .image_first_strategy import ImageFirstProcessingStrategy
            return ImageFirstProcessingStrategy(config, streaming=streaming, database_ops=database_ops)
        elif strategy_type == "hybrid":
            from .hybrid_strategy import HybridProcessingStrategy
            return HybridProcessingStrategy(config, streaming=streaming, database_ops=database_ops)
        elif strategy_type == "chain":
            from .chain_strategy import ChainedProcessingStrategy
            return ChainedProcessingStrategy(config, streaming=streaming, database_ops=database_ops)
        elif strategy_type == "regex":
            from .regex_strategy import RegexProcessingStrategy
            return RegexProcessingStrategy(config, streaming=streaming)
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategy types."""
        return ["direct_file", "text_first", "image_first", "hybrid", "chain"]


class PreProcessingStrategyFactory:
    """Factory for creating pre-processing strategies."""
    
    @staticmethod
    def create_strategy(strategy_type: str, config: Dict[str, Any], streaming: bool = False, database_ops = None) -> BaseProcessingStrategy:
        """Create a pre-processing strategy based on type."""
        if strategy_type == "text":
            return TextPreProcessingStrategy(config, streaming=streaming, database_ops=database_ops)
        elif strategy_type == "image":
            return ImagePreProcessingStrategy(config, streaming=streaming, database_ops=database_ops)
        elif strategy_type == "file":
            return FilePreProcessingStrategy(config, streaming=streaming, database_ops=database_ops)
        else:
            raise ValueError(f"Unsupported pre-processing strategy type: {strategy_type}")


class ProcessingStrategyFactory:
    """Factory for creating processing strategies."""
    
    @staticmethod
    def create_strategy(strategy_type: str, config: Dict[str, Any], streaming: bool = False, database_ops = None) -> BaseProcessingStrategy:
        """Create a processing strategy based on type."""
        if strategy_type == "text_first":
            from .enhanced_text_first_strategy import EnhancedTextFirstProcessingStrategy
            return EnhancedTextFirstProcessingStrategy(config, streaming=streaming, database_ops=database_ops)
        elif strategy_type == "image_first":
            from .image_first_strategy import ImageFirstProcessingStrategy
            return ImageFirstProcessingStrategy(config, streaming=streaming, database_ops=database_ops)
        elif strategy_type == "file_first":
            from .direct_file_strategy import DirectFileProcessingStrategy
            return DirectFileProcessingStrategy(config, streaming=streaming, database_ops=database_ops)
        else:
            raise ValueError(f"Unsupported processing strategy type: {strategy_type}")


class PostProcessingStrategyFactory:
    """Factory for creating post-processing strategies."""
    
    @staticmethod
    def create_strategy(strategy_type: str, config: Dict[str, Any], streaming: bool = False, database_ops = None) -> BaseProcessingStrategy:
        """Create a post-processing strategy based on type."""
        if strategy_type == "metadata":
            return MetadataPostProcessingStrategy(config, streaming=streaming, database_ops=database_ops)
        else:
            raise ValueError(f"Unsupported post-processing strategy type: {strategy_type}")
