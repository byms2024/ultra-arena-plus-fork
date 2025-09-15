# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ultra Arena is a comprehensive multi-LLM processing and evaluation platform that compares multiple LLM providers (Claude, GPT, DeepSeek, Gemini, etc.) across different processing strategies for document analysis tasks. The platform supports modular architecture with CLI, REST API, and direct testing interfaces.

## Architecture

The codebase is organized into modular components:

- **Ultra_Arena_Main**: Core processing engine with LLM clients, strategies, and configuration
- **Ultra_Arena_Main_CLI**: Command-line interface with performance monitoring
- **Ultra_Arena_Main_Restful**: Flask-based REST API server
- **Ultra_Arena_Monitor**: Real-time monitoring dashboard (HTML/JS frontend, Python backend)
- **Ultra_Arena_Main_*_Test**: Comprehensive testing frameworks for each interface

### Key Processing Concepts

- **Strategies**: Different document processing approaches (direct_file, image_first, text_first, hybrid)
- **Providers**: LLM providers (claude, openai, google, deepseek, ollama, huggingface, togetherai, grok)
- **Combo Processing**: Running multiple strategies in parallel for comparison
- **Dual-Level Parallelization**: Strategy-level and file-level concurrency

## Common Commands

### Setup
```bash
# Install dependencies for each component
cd Ultra_Arena_Main && pip install -r requirements.txt
cd ../Ultra_Arena_Main_CLI && pip install -r requirements.txt
cd ../Ultra_Arena_Main_Restful && pip install -r requirements.txt
cd ../Ultra_Arena_Monitor && pip install -r requirements.txt
```

### Running Components

#### CLI Processing
```bash
cd Ultra_Arena_Main_CLI
python main.py --help
```

#### REST API Server
```bash
cd Ultra_Arena_Main_Restful
python server.py
```

#### Monitoring Dashboard
```bash
cd Ultra_Arena_Monitor/frontend
python -m http.server 3000
```

### Testing

#### Run All Tests
```bash
# Run tests for each component
pytest Ultra_Arena_Main_CLI_Test/
pytest Ultra_Arena_Main_Restful_Test/
pytest Ultra_Arena_Main_Direct_Test/
```

#### Performance Testing
```bash
# CLI Performance Tests
cd Ultra_Arena_Main_CLI_Test/tests/python_tests/performance_tests/
python comprehensive_performance_test.py

# REST API Performance Tests
cd Ultra_Arena_Main_Restful_Test/tests/python_tests/performance_tests/
python comprehensive_performance_test.py
```

#### Running Single Test
```bash
# Run specific async test
cd Ultra_Arena_Main_Restful_Test/tests/python_tests/simple_tests/
python test_async_1s_252f_tfOr1_refactored.py
```

### Development Commands

#### Linting and Formatting (if available)
```bash
# Run from project root if these tools are configured
black .
flake8 .
```

## Configuration System

### Main Configuration Files
- `Ultra_Arena_Main/config/config_base.py`: Base configuration with provider settings, file limits, and processing parameters
- `Ultra_Arena_Main/config/config_combo_run.py`: Combo processing configurations
- `Ultra_Arena_Main/config/config_param_grps.py`: Parameter group definitions

### Profile-Based Configuration
Each interface has its own profile directory:
- `Ultra_Arena_Main_CLI/run_profiles/default_profile_cli/`
- `Ultra_Arena_Main_Restful/run_profiles/default_profile_restful/`

Profiles contain API keys, prompt configurations, and interface-specific settings that are injected at runtime.

### Key Configuration Constants
- `DEFAULT_STRATEGY_TYPE`: Default processing strategy
- `MAX_NUM_FILES_PER_REQUEST`: File batch limits (default: 4 for Google GenAI)
- `DEFAULT_MAX_CC_STRATEGIES`: Concurrent strategy limit
- `PROVIDER_*`: Provider name constants (PROVIDER_CLAUDE, PROVIDER_OPENAI, etc.)

## Core Processing Flow

1. **Initialization**: Load configuration and checkpoint data
2. **Strategy Selection**: Choose processing approach (direct_file, image_first, text_first, hybrid)
3. **Provider Selection**: Select LLM provider and model
4. **Processing**: Execute with dual-level parallelization
5. **Results**: Generate performance metrics and output

## Checkpoint System

The platform includes a checkpoint system for resuming interrupted processing:
- `Ultra_Arena_Main/processors/checkpoint_manager.py`: Generic checkpoint manager
- `Ultra_Arena_Main/processors/modular_parallel_processor.py`: Processor with checkpoint integration
- Default checkpoint file: `modular_checkpoint.pkl`

**Note**: The checkpoint system currently loads checkpoints but doesn't save them during processing.

## Testing Architecture

### Test Structure
- **Simple Tests**: Basic functionality with small file sets
- **Performance Tests**: Large-scale processing with timing metrics
- **Health Tests**: API endpoint availability and basic functionality
- **Functional Tests**: End-to-end processing workflows

### Test Naming Convention
- `test_async_<strategies>s_<files>f_<provider>_refactored.py`
- Example: `test_async_1s_252f_tfOr1_refactored.py` (1 strategy, 252 files, text-first with Ollama)

### Test Configuration
- `pytest.ini`: Configures pytest with cache disabled and custom paths
- `test_async_utils.py`: Shared utilities for async test execution

## Key Dependencies

### Core LLM Providers
- `anthropic>=0.7.0`: Claude API client
- `openai>=1.0.0`: GPT API client
- `google-genai>=0.3.0`: Gemini API client
- `ollama>=0.1.0`: Local LLM support

### Document Processing
- `PyPDF2>=3.0.0`: PDF text extraction
- `PyMuPDF>=1.23.0`: Advanced PDF processing (fitz module)
- `pytesseract>=0.3.10`: OCR support
- `pdf2image>=1.16.0`: PDF to image conversion

### Web Framework
- `flask>=2.0.0`: REST API server
- `flask-cors`: CORS support for web clients

## Important Notes

### File Processing Limits
- Google GenAI models have strict limits: use `MAX_NUM_FILES_PER_REQUEST = 4` to avoid truncation
- Use `gemini-2.5-flash` for production; experimental models may cause 400 INVALID_ARGUMENT errors

### API Key Management
- API keys are stored in profile configurations, not in code
- Each interface (CLI, REST) has separate profile directories for key management

### Performance Optimization
- Supports configurable concurrency levels for both strategies and files
- Automatic streaming for large file batches to prevent provider API limits

### Error Handling
- Comprehensive retry mechanisms with configurable delays
- Provider-specific error handling and fallback strategies
- Detailed logging and monitoring integration