# Filename Pattern Validation

## Overview

The filename pattern validation feature ensures that PDF files follow a specific naming convention before they are processed. Files that don't match the expected pattern are marked with a validation status and excluded from further processing in the chain.

## Expected Pattern

Files must follow this naming format:
```
{DEALER_CODE}_NF{NUMBER}_nota_(peca|servico)
```

### Examples of Valid Filenames:
- `ABC123_NF12345_nota_peca.pdf`
- `XYZ789_NF98765_nota_servico.pdf`
- `DEALER001_NF555_nota_peca.pdf`

### Examples of Invalid Filenames:
- `ABC123_12345_nota_peca.pdf` (missing "NF" prefix)
- `ABC123_NF12345_invoice_peca.pdf` (using "invoice" instead of "nota")
- `ABC123_NF12345_nota_parts.pdf` (using "parts" instead of "peca" or "servico")
- `WRONG_NF12345_nota_peca.pdf` (dealer code doesn't match metadata)

## Configuration

### Enable Metadata Reading

Filename validation requires PDF metadata to be enabled. In your configuration:

```python
config = {
    "enable_pdf_metadata": True,  # Required for validation
    "filename_pattern_case_sensitive": True,  # Optional, default is True
}
```

### Case Sensitivity

By default, pattern matching is **case-sensitive**. To enable case-insensitive matching:

```python
config = {
    "enable_pdf_metadata": True,
    "filename_pattern_case_sensitive": False,  # Allow "NF", "nf", "Nf", etc.
}
```

## Validation Statuses

Files are assigned one of three validation statuses:

### 1. VALID
- The filename matches the expected pattern
- File proceeds normally through the processing chain
- Status: `"VALID"`

### 2. PATTERN_MISMATCH
- The filename doesn't match the expected pattern
- File is excluded from further processing
- Status: `"PatternMismatch"`
- Reason is logged with details

### 3. MISSING_METADATA
- Required metadata (`remote_file_name` or `dealer_code`) is missing
- File is excluded from further processing
- Status: `"MissingMetadata"`
- Reason is logged with details

## How It Works

1. **Pre-processing Stage**: During pre-processing, metadata is extracted from PDFs
2. **Pattern Validation**: The `remote_file_name` is validated against the expected pattern using the `dealer_code` from metadata
3. **Status Assignment**: Files are marked with validation status
4. **Chain Processing**: Files with `PATTERN_MISMATCH` or `MISSING_METADATA` are excluded from subsequent processing chains

## Logging

The validation process provides clear logging:

### Successful Validation:
```
✅ Filename validation PASSED: 'ABC123_NF12345_nota_peca.pdf' matches pattern for dealer 'ABC123'
✅ Pattern validation PASSED for /path/to/file.pdf
```

### Failed Validation:
```
❌ Filename validation FAILED: 'WRONG_NF12345_nota_peca.pdf' does not match expected pattern 'ABC123_NF{NUMBER}_nota_(peca|servico)'
⚠️ Pattern validation FAILED for /path/to/file.pdf: Filename does not match expected pattern
```

### Missing Metadata:
```
⚠️ Pattern validation SKIPPED for /path/to/file.pdf: Missing remote_file_name in metadata
```

## Affected Pre-processing Strategies

Filename validation is implemented in the following pre-processing strategies:

1. **TextPreProcessingStrategy** (in `strategy_factory.py`)
2. **TextPreProcessingStrategy** (in `text_first_gemini.py`)
3. **RegexPreProcessingStrategy** (in `regex_strategy.py`)
4. **ImagePreProcessingStrategy** (in `strategy_factory.py`)
5. **FilePreProcessingStrategy** (in `strategy_factory.py`)

## Passthrough Integration

Validation results are stored in the passthrough object for each file:

```python
{
    "file_path": "/path/to/file.pdf",
    "status": "PatternMismatch",  # or "MissingMetadata" or "Pending"
    "pattern_validation_status": "PATTERN_MISMATCH",  # or "MISSING_METADATA" or "VALID"
    "pattern_validation_reason": "Filename does not match expected pattern: ABC123_NF{NUMBER}_nota_(peca|servico)",
    "extracted_data": {
        "dealer_code": "ABC123",
        "remote_file_name": "WRONG_NF12345_nota_peca.pdf"
    }
}
```

## Chain Processing Behavior

Files with validation failures are automatically excluded from chain processing:

- Files with status `PatternMismatch` or `MissingMetadata` are **not rerun** in subsequent subchains
- These files are logged with a skip message:
  ```
  ⏭️ Skipping /path/to/file.pdf due to validation status: patternmismatch
  ```

## API and Usage

### Direct Validation

You can validate filenames directly using the `FilenameValidator` class:

```python
from Ultra_Arena_Main.common.filename_validator import FilenameValidator

# Create validator (case-sensitive by default)
validator = FilenameValidator(case_sensitive=True)

# Validate a filename
status, reason = validator.validate_filename(
    remote_file_name="ABC123_NF12345_nota_peca.pdf",
    dealer_code="ABC123"
)

if status == FilenameValidator.VALID:
    print("Filename is valid!")
elif status == FilenameValidator.PATTERN_MISMATCH:
    print(f"Filename is invalid: {reason}")
elif status == FilenameValidator.MISSING_METADATA:
    print(f"Metadata missing: {reason}")
```

### In Pre-processing Strategies

The validation is automatically performed when `enable_pdf_metadata` is set to `True`:

```python
# In your strategy's process_file_group method
from ..common.filename_validator import FilenameValidator

# After extracting metadata
validation_status = self.validate_filename_pattern(file_path, case_sensitive=True)

# Check if file should be skipped
if FilenameValidator.should_skip_processing(validation_status):
    # File has validation failure, skip processing
    continue
```

## Benefits

1. **Early Validation**: Files with incorrect naming are identified immediately in pre-processing
2. **Resource Efficiency**: Invalid files don't consume processing resources
3. **Clear Reporting**: Validation status and reasons are tracked and logged
4. **Flexible Configuration**: Case sensitivity can be configured per deployment
5. **Consistent Behavior**: Same validation logic across all pre-processing strategies

