"""
Chained processing strategy: run multiple strategies in order as fallbacks.
"""

from typing import Dict, List, Tuple, Optional, Any
import logging
import time

from .base_strategy import BaseProcessingStrategy
from .strategy_factory import ProcessingStrategyFactory


class ChainedProcessingStrategy(BaseProcessingStrategy):
    """Execute an ordered list of strategies as a fallback chain."""

    def __init__(self, config: Dict[str, Any], streaming: bool = False, database_ops = None):
        super().__init__(config)
        self.streaming = streaming
        self.steps = config.get("chain_steps", [])
        if not self.steps:
            raise ValueError("chain_steps must be provided for ChainedProcessingStrategy")
        self.chain_on_missing_keys = config.get("chain_on_missing_keys", False)
        self.database_ops = database_ops

    def process_file_group(self, *, config_manager=None, file_group: List[str], group_index: int,
                           group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str
                           ) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        start_time = time.time()
        remaining_files = list(file_group)
        per_file_result: Dict[str, Dict] = {}

        agg_stats = {
            "total_files": len(file_group),
            "successful_files": 0,
            "failed_files": 0,
            "total_tokens": 0,
            "estimated_tokens": 0,
            "processing_time": 0
        }

        for step_idx, step in enumerate(self.steps):
            if not remaining_files:
                break

            strategy_type = step.get("type")
            overrides = step.get("overrides", {})
            step_config = {**self.config, **overrides}
            strategy = ProcessingStrategyFactory.create_strategy(strategy_type, step_config, streaming=self.streaming, database_ops=self.database_ops)

            logging.info(f"🔗 Chain step {step_idx + 1}/{len(self.steps)}: {strategy_type} on {len(remaining_files)} file(s)")

            # Call underlying strategy (be tolerant to different signatures)
            try:
                results, stats, _ = strategy.process_file_group(
                    config_manager=config_manager,
                    file_group=remaining_files,
                    group_index=group_index,
                    group_id=f"{group_id}_chain_{step_idx + 1}",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
            except TypeError:
                results, stats, _ = strategy.process_file_group(
                    file_group=remaining_files,
                    group_index=group_index,
                    group_id=f"{group_id}_chain_{step_idx + 1}",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )

            agg_stats["estimated_tokens"] += stats.get("estimated_tokens", 0)
            agg_stats["total_tokens"] += stats.get("total_tokens", 0)
            agg_stats["processing_time"] += stats.get("processing_time", 0)

            next_remaining: List[str] = []
            for file_path, result in results:
                if "error" in result:
                    per_file_result[file_path] = result
                    next_remaining.append(file_path)
                    continue

                if self.chain_on_missing_keys:
                    model_output = result.get("file_model_output", result)
                    ok, _missing = self.check_mandatory_keys(model_output, file_path, getattr(self, "benchmark_comparator", None), self.database_ops)
                    if not ok:
                        per_file_result[file_path] = result
                        next_remaining.append(file_path)
                        continue

                # success for this file
                if file_path not in per_file_result:
                    per_file_result[file_path] = result

            remaining_files = next_remaining

        # Any file not finalized after all steps => failure
        for file_path in file_group:
            if file_path not in per_file_result:
                per_file_result[file_path] = {"error": "All chained strategies exhausted without success"}

        merged_results = [(fp, per_file_result[fp]) for fp in file_group]
        agg_stats["successful_files"] = sum(1 for _fp, res in merged_results if "error" not in res)
        agg_stats["failed_files"] = agg_stats["total_files"] - agg_stats["successful_files"]
        agg_stats["processing_time"] = max(agg_stats["processing_time"], int(time.time() - start_time))

        return merged_results, agg_stats, group_id


