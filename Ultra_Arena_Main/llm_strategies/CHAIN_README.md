# Chain Strategy

A Chain is an ordered list of existing strategies executed as a fallback pipeline. For each file group, the first step runs on all files; files that fail move to the next step, and so on, until a step succeeds for that file or the chain is exhausted.

## Goals
- Increase robustness across varied PDFs without overfitting a single approach
- Reuse existing strategies (`direct_file`, `text_first`, `image_first`, `hybrid`)
- Keep UX consistent: select `chain` like any other strategy or include it in combos

## How it works
1. You choose `strategy: "chain"` in a parameter group.
2. Provide `chain_steps`, an ordered list of steps, each with a `type` (one of existing strategies) and optional per-step `overrides`.
3. For a file group, step 1 runs over all files. Files with errors are forwarded to step 2, then step 3, etc.
4. Optional: treat “missing mandatory keys” as failure to trigger fallback (`chain_on_missing_keys: true`).
5. Returns the same result/stat shapes as other strategies; existing processor, CSV, and checkpoint logic continue to work.

## Data flow (file-group level)
- Input: `[file_a.pdf, file_b.pdf, file_c.pdf]`
- Step 1 (`direct_file`): `file_a` succeeds, `file_b` fails, `file_c` fails → forward `b, c`
- Step 2 (`text_first`): `file_b` succeeds, `file_c` fails → forward `c`
- Step 3 (`image_first`): `file_c` succeeds (or remains failed)
- Output: one result per original file; group stats aggregated across steps

## Configuration
Add a param group in `config/config_param_grps.py`:
```python
param_grps["chain_df_tf_if_google"] = {
    "strategy": "chain",
    "mode": "parallel",
    "provider": "google",
    "model": "gemini-2.5-flash",
    "chain_steps": [
        {"type": "direct_file"},
        {"type": "text_first"},
        {"type": "image_first"}
    ],
    # Optional behavior: only fallback on hard errors by default
    "chain_on_missing_keys": False,
    # Avoid duplicate work if you chain on missing keys
    "num_retry_for_mandatory_keys": 0,
}
```

Per-step overrides (optional):
```python
"chain_steps": [
    {"type": "direct_file", "overrides": {"llm_provider": "google"}},
    {"type": "text_first", "overrides": {"llm_provider": "ollama", "pdf_extractor_lib": "pymupdf"}},
]
```

## Selecting Chain (standalone or in combos)
- In combos, `chain` is treated like any other strategy; it will be scheduled by provider groups and run its internal steps sequentially for each file group.
- Standalone runs simply reference the param group that sets `strategy: "chain"`.

## Behavior options
- `chain_on_missing_keys: false` (default): fallback only on hard errors. Mandatory key retries remain managed by the processor.
- `chain_on_missing_keys: true`: also fallback when mandatory keys are missing. In this mode, set `num_retry_for_mandatory_keys: 0` to disable processor retries and avoid duplicate work.

## Implementation status
Already implemented:
- `Ultra_Arena_Main/llm_strategies/chain_strategy.py` (skeleton implementation)
- Factory wiring in `Ultra_Arena_Main/llm_strategies/strategy_factory.py` (`strategy_type == "chain"`)

To be built/refined:
- Add `STRATEGY_CHAIN = "chain"` to `config/config_base.py`
- Update `get_config_for_strategy` (in `Ultra_Arena_Main/main_modular.py`) to pass through `chain_steps` and flags
- Add an example entry in `config/config_param_grps.py` (see above)
- Optional: harmonize `process_file_group` signatures so all strategies accept `config_manager=None, **kwargs`

## Stats and accounting
- Group stats aggregate estimated and actual tokens and processing time across steps.
- Per-file, only the successful step’s tokens are considered in the final result. If no step succeeds, the file returns an error.

## Error handling
- If all steps fail for a file, the result contains `{ "error": "All chained strategies exhausted without success" }`.
- Chain tolerates varying strategy method signatures by catching `TypeError` and retrying without `config_manager`.

## Testing guidelines
- Unit: two-step chain where step 1 errors, step 2 succeeds → ensure per-file results and aggregated stats look correct.
- Unit: `chain_on_missing_keys=True` where step 1 lacks keys, step 2 provides them → ensure no processor retry loops are triggered if retries are disabled.
- Integration: include chain param group in a combo; verify provider grouping and CSV/checkpoint outputs are produced.

## FAQ
- Q: Can steps run in parallel?
  - A: The chain runs steps sequentially to preserve fallback semantics. Within each step, the inner strategy can parallelize its own file group processing.
- Q: Will this break existing CSV/checkpoints?
  - A: No. Chain returns the same structure used by existing strategies.
