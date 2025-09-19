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

            # Build per-step config using canonical generator to ensure proper provider configs and API keys
            try:
                from Ultra_Arena_Main.main_modular import get_config_for_strategy as _get_conf
            except Exception:
                import importlib
                _get_conf = importlib.import_module('Ultra_Arena_Main.main_modular').get_config_for_strategy

            provider_override = overrides.get("llm_provider") or self.config.get("llm_provider")
            model_override = overrides.get("llm_model") or self.config.get("model")
            
            step_config = {**self.config, **overrides}
            base_step_config = _get_conf(strategy_type, llm_provider=provider_override, llm_model=model_override, streaming=self.streaming, database_ops=self.database_ops)

            # Carry forward chain-level generic limits without clobbering provider configs
            for key in [
                "mandatory_keys",
                "num_retry_for_mandatory_keys",
                "max_num_files_per_request",
                "max_num_file_parts_per_batch",
            ]:
                if key in self.config and key not in overrides:
                    base_step_config[key] = self.config[key]

            # Apply per-step overrides last
            step_config = {**base_step_config, **overrides}

            strategy = ProcessingStrategyFactory.create_strategy(strategy_type, step_config, streaming=self.streaming)

            logging.info(f"🔗 Chain step {step_idx + 1}/{len(self.steps)}: {strategy_type} on {len(remaining_files)} file(s)")
            logging.debug(f"🔧 Step overrides: {overrides}")

            # Call underlying strategy (be tolerant to different signatures)
            call_kwargs = {
                "file_group": remaining_files,
                "group_index": group_index,
                "group_id": f"{group_id}_chain_{step_idx + 1}",
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
            try:
                results, stats, _ = strategy.process_file_group(**{**call_kwargs, "config_manager": config_manager})
            except TypeError:
                results, stats, _ = strategy.process_file_group(**call_kwargs)

            agg_stats["estimated_tokens"] += stats.get("estimated_tokens", 0)
            agg_stats["total_tokens"] += stats.get("total_tokens", 0)
            agg_stats["processing_time"] += stats.get("processing_time", 0)

            next_remaining: List[str] = []
            forwarded_count = 0
            finalized_count = 0
            for file_path, result in results:
                if "error" in result:
                    per_file_result[file_path] = result
                    next_remaining.append(file_path)
                    forwarded_count += 1
                    logging.info(f"➡️  Step {step_idx + 1}: forwarding due to error → {file_path}: {result.get('error')}")
                    continue

                if self.chain_on_missing_keys:
                    model_output = result.get("file_model_output", result)
                    ok, _missing = self.check_mandatory_keys(model_output, file_path, getattr(self, "benchmark_comparator", None), self.database_ops)
                    if not ok:
                        per_file_result[file_path] = result
                        next_remaining.append(file_path)
                        forwarded_count += 1
                        logging.info(f"➡️  Step {step_idx + 1}: forwarding due to missing mandatory keys → {file_path}")
                        continue

                # success for this file
                if file_path not in per_file_result:
                    per_file_result[file_path] = result
                    finalized_count += 1
                    logging.info(f"✅ Step {step_idx + 1}: finalized → {file_path}")

            logging.info(f"🔁 Step {step_idx + 1} complete: finalized={finalized_count}, forwarded={forwarded_count}")
            if next_remaining:
                logging.info(f"➡️  Forwarding {len(next_remaining)} file(s) to step {step_idx + 2}")
            remaining_files = next_remaining

        # Any file not finalized after all steps => failure
        for file_path in file_group:
            if file_path not in per_file_result:
                per_file_result[file_path] = {"error": "All chained strategies exhausted without success"}
                logging.info(f"❌ Chain exhausted: {file_path}")

        merged_results = [(fp, per_file_result[fp]) for fp in file_group]
        agg_stats["successful_files"] = sum(1 for _fp, res in merged_results if "error" not in res)
        agg_stats["failed_files"] = agg_stats["total_files"] - agg_stats["successful_files"]
        agg_stats["processing_time"] = max(agg_stats["processing_time"], int(time.time() - start_time))

        return merged_results, agg_stats, group_id


