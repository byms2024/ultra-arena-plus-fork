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
- `Ultra_Arena_Main/llm_strategies/chain_strategy.py` (full implementation)
- Factory wiring in `Ultra_Arena_Main/llm_strategies/strategy_factory.py` (`strategy_type == "chain"`)

New wiring added:
- `STRATEGY_CHAIN` added to `config/config_base.py`.
- `get_config_for_strategy` updated to return a pass-through config for chain; chain steps are merged at call time.
- `Ultra_Arena_Main_Restful/server_utils` path updated so REST requests can specify `chain_name` and bypass `combo_name`.
- Central map `config/config_chain_defs.py` defines named chains.

REST flow (chain_name overrides combo_name):
1) Client POSTs to `/api/process/combo` with payload containing `chain_name` plus the usual I/O fields. When `chain_name` is present, `combo_name` is ignored.
2) `server_utils/request_validator.py` accepts `chain_name` (optional); existing validation for I/O remains.
3) `server_utils/config_assemblers/request_config_assembler.py` extracts `chain_name` and builds a unified request config.
4) `server_utils/request_processor.py` sees `chain_name`, loads `config/config_chain_defs.py`, builds a single-chain config via `get_config_for_strategy(STRATEGY_CHAIN)` and merges in that chain’s `chain_steps` and options.
5) It creates a `ModularParallelProcessor` with `strategy_type="chain"` and runs processing over the resolved input files. Results and CSV/JSON outputs are produced in the same directory structure as combos.
6) The synchronous endpoint returns `status: success` and results summary consistent with combo runs.

How to define and use chains:
- Add or edit `Ultra_Arena_Main/config/config_chain_defs.py`:
```python
chain_definitions = {
    "test_1": {
        "chain_steps": [
            {"type": "text_first"},
            {"type": "direct_file"},
        ],
        "chain_on_missing_keys": False,
    },
}
```
- Call the REST endpoint with a `chain_name`:
```json
{
  "chain_name": "test_1",
  "input_pdf_dir_path": "C:\\path\\to\\pdfs",
  "output_dir": "C:\\path\\to\\output",
  "streaming": false,
  "max_cc_strategies": 1,
  "max_cc_filegroups": 1,
  "max_files_per_request": 10
}
```

Behavior details:
- If `chain_name` is present, the server ignores any `combo_name` and runs the requested chain.
- If no `chain_name` is provided, normal combo processing path is used.
- Per-step overrides inside `chain_steps[*].overrides` are supported and merged onto the base chain config.
- `chain_on_missing_keys` controls whether to fallback when mandatory keys are missing, in addition to hard errors.

Testing
- Use `Ultra_Arena_Main_Restful_Test/tests/python_tests/simple_tests/test_chain_rest_basic.py` which posts `chain_name` and prints the response.


## Stats and accounting
- Group stats aggregate estimated and actual tokens and processing time across steps.
- Per-file, only the successful step’s tokens are considered in the final result. If no step succeeds, the file returns an error.

## Error handling
- If all steps fail for a file, the result contains `{ "error": "All chained strategies exhausted without success" }`.
- Chain tolerates varying strategy method signatures by catching `TypeError` and retrying without `config_manager`.

## Logging (fallback visibility)
- At each step, logs include:
  - Which step is running, how many files it’s attempting, and the step overrides.
  - For each file:
    - `➡️ forwarding due to error` with the error reason.
    - `➡️ forwarding due to missing mandatory keys` when `chain_on_missing_keys=True` and required keys are absent.
    - `✅ finalized` when the file is completed in the current step.
  - A step summary: `finalized` and `forwarded` counts, and notice of how many are forwarded to the next step.
  - When the chain is exhausted for a file: `❌ Chain exhausted: <file>`.

## Testing guidelines
- Unit: two-step chain where step 1 errors, step 2 succeeds → ensure per-file results and aggregated stats look correct.
- Unit: `chain_on_missing_keys=True` where step 1 lacks keys, step 2 provides them → ensure no processor retry loops are triggered if retries are disabled.
- Integration: include chain param group in a combo; verify provider grouping and CSV/checkpoint outputs are produced.

## FAQ
- Q: Can steps run in parallel?
  - A: The chain runs steps sequentially to preserve fallback semantics. Within each step, the inner strategy can parallelize its own file group processing.
- Q: Will this break existing CSV/checkpoints?
  - A: No. Chain returns the same structure used by existing strategies.
