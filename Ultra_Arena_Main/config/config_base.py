"""
Base configuration settings for the PDF processing system.

This file serves as the base config that other config files import from,
minimizing code duplication. It also contains the only SYSTEM_PROMPT and USER_PROMPT
for use as examples or tests with the top-level run_batch_processing() function.
"""

# =============================================================================
# BASE CONFIGURATION CONSTANTS (imported by other config files)
# =============================================================================
DEFAULT_STRATEGY_PARAM_GRP = "grp_directF_google_gemini25_para"

# File Processing Configuration
MAX_NUM_FILES_PER_REQUEST = 4  # Reduced from 6 to 4 to prevent Google GenAI truncation
MAX_NUM_FILE_PARTS_PER_BATCH = 100    # For Batch/Slow/High Throughput Mode

# API Default Configuration (can be overridden by profiles)
DEFAULT_STREAMING = False
DEFAULT_MAX_CC_STRATEGIES = 1
DEFAULT_MAX_CC_FILEGROUPS = 1
DEFAULT_MAX_FILES_PER_REQUEST = 10

# Provider Configuration
PROVIDER_OPENAI = "openai"
PROVIDER_GOOGLE = "google"
PROVIDER_DEEPSEEK = "deepseek"
PROVIDER_CLAUDE = "claude"
PROVIDER_OLLAMA = "ollama"
PROVIDER_HUGGINGFACE = "huggingface"
PROVIDER_TOGETHERAI = "togetherai"
PROVIDER_GROK = "grok"

PROVIDER = PROVIDER_GOOGLE  # "google" or "openai"

# Mandatory keys are now provided by profiles (see PROFILE_DIR/profile_config.py)
# NUM_RETRY_FOR_MANDATORY_KEYS remains here unless overridden by profile
NUM_RETRY_FOR_MANDATORY_KEYS = 2

# Default mandatory keys (fallback for backward compatibility)
MANDATORY_KEYS = []

# Default benchmark file path (fallback for backward compatibility)
BENCHMARK_FILE_PATH = ""

# Google GenAI Configuration
# IMPORTANT: Use gemini-2.5-flash for production and testing to avoid 400 INVALID_ARGUMENT errors
# Experimental models like gemini-2.0-flash-exp may not support all API parameters and can cause errors
# NOTE: For large file batches (>6 files), the client will automatically use streaming to avoid truncation
GCP_API_KEY = ""
GOOGLE_MODEL_ID_GEMINI_25_FLASH = "gemini-2.5-flash"
GOOGLE_DEFAULT_MODEL_ID = GOOGLE_MODEL_ID_GEMINI_25_FLASH
GOOGLE_MODEL_TEMPERATURE = 0.1
GOOGLE_MODEL_MAX_TOKENS = 1000000  # Increased from 32000 to 1000000 to handle large responses

# OpenAI Configuration
# https://chatgpt.com/share/688aced8-1b0c-800e-a96a-b201969f21b4 
# - open API is almost impossible to upload pdfs, and more expensive to read images
# - after a lot of testing,cursor concluded: "So the image_first strategy is the only way to use GPT-4.1 with PDF documents - we convert them to images first, then let GPT-4.1 process the images. This is why our recent test was successful!"
# Performance Improvements with GPT-4.1 (over GPT-4o-mini) with 2 pdfs that were missing CLAIM_NUMBER:
# Token Usage: Much more efficient (~2,450 tokens vs ~38,000 tokens with GPT-4o-mini)
# Processing Time: Faster (29.33s vs previous longer times)
# Data Quality: Better extraction - found chassis numbers and second CNPJ that were missing before
# Success Rate: 100% for API calls, much better field extraction
OPENAI_API_KEY = ""
OPENAI_MODEL_GPT_41 = "gpt-4.1"
OPENAI_MODEL_GPT_41_MINI = "gpt-4o-mini"
# https://openai.com/api/pricing/
# OPENAI_MODEL = "gpt-4.1-nano"  #https://platform.openai.com/docs/models/gpt-4.1-nano -- not working 
OPENAI_DEFAULT_MODEL_ID = OPENAI_MODEL_GPT_41_MINI  #https://platform.openai.com/docs/models/gpt-4o-mini -- working with image first strategy
# OPENAI_DEFAULT_MODEL_ID = OPENAI_MODEL_GPT_41 #https://platform.openai.com/docs/models/gpt-4.1 -- working with image first strategy
OPENAI_MODEL_TEMPERATURE = 0.1

# DeepSeek Configuration
DEEPSEEK_API_KEY = ""
DEEPSEEK_MODEL_DCHAT = "deepseek-chat"
DEEPSEEK_DEFAULT_MODEL_ID = DEEPSEEK_MODEL_DCHAT
DEEPSEEK_MODEL_TEMPERATURE = 0.1

# Claude Configuration
CLAUDE_API_KEY = ""
CLAUDE_MODEL_CLAUDE_4_SONNET = "claude-sonnet-4-20250514"
CLAUDE_DEFAULT_MODEL_ID = CLAUDE_MODEL_CLAUDE_4_SONNET
CLAUDE_MODEL_TEMPERATURE = 0.1
CLAUDE_MODEL_MAX_TOKENS = 10000

# HuggingFace Configuration
HUGGINGFACE_TOKEN = "" # read token
# huggingface biggest and most popular vision models:
# huggingface host open source smaller models, but they are not as good as the models hosted by openai and google
HUGGINGFACE_MODEL_ID_QWEN2_VL_72B = "Qwen/Qwen2.5-VL-72B-Instruct"
HUGGINGFACE_MODEL_ID_LLAMA_VISION_90B = "meta-llama/Llama-3.2-90B-Vision-Instruct"
HUGGINGFACE_MODEL_ID_DOTS_OCR = "rednote-hilab/dots.ocr"
# HUGGINGFACE_MODEL_INTERNLVL2_76B = "internvl2-76b" # https://huggingface.co/OpenGVLab/InternVL2_5-78B not yet integrated in hugging face api apparently
HUGGINGFACE_DEFAULT_MODEL_ID = HUGGINGFACE_MODEL_ID_QWEN2_VL_72B
HUGGINGFACE_MODEL_TEMPERATURE = 0.1

# TogetherAI Configuration  # https://together.ai/docs/api-reference/introduction
TOGETHERAI_API_KEY = ""  # Will be set after imports
TOGETHERAI_MODEL_ID_LLAMA_VISION_90B = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"
# TOGETHERAI_MODEL_ID_QWEN2_VL_72B = "Qwen/Qwen2.5-VL-72B-Instruct" # too slow - 196s for 1 file with no accuracy
TOGETHERAI_MODEL_ID_LLAMA_4_17b = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
TOGETHERAI_MODEL_ID_LLAMA_3_70b = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

TOGETHERAI_MODEL_TEMPERATURE = 0.1

# Grok Configuration  # https://docs.x.ai/docs/models 
XAI_API_KEY = ""  # Will be set after imports
# GROK_MODEL_ID_GROK_4 = "grok-4-0709" # very expensive!! # too slow - 196s for 1 file with no accuracy
GROK_MODEL_ID_GROK_2 = "grok-2-vision-latest" 
GROK_MODEL_TEMPERATURE = 0.1

# Ollama Configuration (for local LLMs)
LOCAL_OLLAMA_MODEL_GPT_OSS_20B = "gpt-oss:20b"
LOCAL_OLLAMA_MODEL_DEEP_R1 = "deepseek-r1:latest"
LOCAL_OLLAMA_MODEL_MISTRAL = "mistral:latest"
LOCAL_OLLAMA_MODEL_LLAMA_3_2_11B = "llama3.2-vision:11b"
LOCAL_OLLAMA_MODEL_LLAVA_7B = "llava:7b"
LOCAL_OLLAMA_TEMPERATURE = 0.1
LOCAL_OLLAMA_MAX_TOKENS = 4000
LOCAL_OLLAMA_TIMEOUT = 120

# Processing Modes
MODE_BATCH_OPTIMIZED = "batch_optimized"
MODE_BATCH_PARALLEL = "parallel"

# File Limits
MAX_TOKENS_PER_REQUEST = 50000
MAX_FILE_SIZE_MB = 10  # Maximum file size in MB for direct upload

# Retry Configuration 
# - infrastructure: Retries for API/network failures (timeouts, connection errors, etc.)
API_INFRA_MAX_RETRIES = 3
API_INFRA_RETRY_DELAY_SECONDS = 1
API_INFRA_BACKOFF_MULTIPLIER = 2

# Hybrid approach retry limits
TEXT_FIRST_MAX_RETRY = 3  # Maximum retries for text-first processing in hybrid mode
FILE_DIRECT_MAX_RETRY = 2  # Maximum retries for direct file processing in hybrid mode

# Text processing limits
MAX_TEXT_LENGTH = 10000  # Maximum text length for processing
MAX_TEXT_CHUNK_SIZE = 10000  # Maximum text chunk size for processing

# Output directory structure (unified combo approach)
OUTPUT_BASE_DIR = "output"
OUTPUT_RESULTS_DIR = f"{OUTPUT_BASE_DIR}/results"
OUTPUT_COMBO_DIR = f"{OUTPUT_RESULTS_DIR}/combo"
OUTPUT_COMBO_BACKUP_DIR = f"{OUTPUT_COMBO_DIR}/backup"
OUTPUT_COMBO_CSV_DIR = "csv"
OUTPUT_COMBO_JSON_DIR = "json"

# Other output directories
OUTPUT_CHECKPOINTS_DIR = f"{OUTPUT_BASE_DIR}/checkpoints"
OUTPUT_LOGS_DIR = f"{OUTPUT_BASE_DIR}/logs"
OUTPUT_NOTE_GEN_DIR = f"{OUTPUT_BASE_DIR}/note_gen"

# =============================================================================
# STRATEGY AND MODE CONSTANTS
# =============================================================================

STRATEGY_DIRECT_FILE = "direct_file"
STRATEGY_TEXT_FIRST = "text_first"
STRATEGY_IMAGE_FIRST = "image_first"
STRATEGY_HYBRID = "hybrid"

MODE_PARALLEL = "parallel"
MODE_BATCH = "batch"

# =============================================================================
# DEFAULT VALUES FOR run_file_processing() WRAPPER
# =============================================================================

# Default processing settings
DEFAULT_STRATEGY_TYPE = STRATEGY_DIRECT_FILE
DEFAULT_MODE = MODE_PARALLEL
DEFAULT_MAX_CC_FILEGROUPS = 5  # Maximum Concurrency Level for File Groups, for a Single Strategy
DEFAULT_OUTPUT_FILE = "modular_results.json"
DEFAULT_CHECKPOINT_FILE = "modular_checkpoint.pkl"
DEFAULT_LLM_PROVIDER = "google"  # Default provider for all strategies

# Simplified configuration - no longer profile-driven
# API keys are now managed at component level for security

# Defaults (components provide their own overrides)
BENCHMARK_FILE_PATH = ""
SYSTEM_PROMPT = ""
JSON_FORMAT_INSTRUCTIONS = ""
USER_PROMPT = ""
PROFILE_INPUT_DIR = ""
INPUT_DIR_FOR_COMBO = ""

# Prompt configuration defaults (will be overridden by profile-specific configs)
MANDATORY_KEYS = []

# API keys are now managed at component level for security
# Components will inject their own API keys when calling library functions
# ============================================================================
# DIRECT FILE STRATEGY CONFIGURATIONS
# ============================================================================

# Retry Configuration for direct file strategy
DIRECT_FILE_MAX_RETRIES = 2  # Maximum retry rounds for failed files
DIRECT_FILE_RETRY_DELAY_SECONDS = 1  # Delay between retry rounds

# Provider-specific settings for direct file processing
# NOTE: For Google GenAI, use gemini-2.5-flash to avoid 400 INVALID_ARGUMENT errors
# Experimental models may not support all API parameters like response_mime_type
DIRECT_FILE_PROVIDER_CONFIGS = {
    "google": {
        "api_key": GCP_API_KEY,
        "model": GOOGLE_DEFAULT_MODEL_ID,
        "temperature": GOOGLE_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    },
    "openai": {
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_DEFAULT_MODEL_ID,
        "temperature": OPENAI_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    },
    "deepseek": {
        "api_key": DEEPSEEK_API_KEY,
        "model": DEEPSEEK_DEFAULT_MODEL_ID,
        "temperature": DEEPSEEK_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    },
    "claude": {
        "api_key": CLAUDE_API_KEY,
        "model": CLAUDE_DEFAULT_MODEL_ID,
        "temperature": CLAUDE_MODEL_TEMPERATURE,
        "max_tokens": 10000,
        "timeout": 60
    }
}

# ============================================================================
# TEXT FIRST STRATEGY CONFIGURATIONS
# ============================================================================

# LLM Provider Configuration
LOCAL_LLM_PROVIDER = "ollama"  # "ollama", "google", "openai", "deepseek"

# PDF Text Extraction Configuration
PDF_EXTRACTOR_LIB = "pymupdf"  # "pymupdf" or "pytesseract"
SECONDARY_PDF_EXTRACTOR_LIB = "pytesseract"  # Secondary extractor for fallback

# Text First Strategy Regex Criteria for validation
TEXT_FIRST_REGEX_CRITERIA = {}

# Provider-specific settings for text processing
TEXT_PROVIDER_CONFIGS = {
    "ollama": {
        "model": LOCAL_OLLAMA_MODEL_DEEP_R1,
        "temperature": LOCAL_OLLAMA_TEMPERATURE,
        "max_tokens": LOCAL_OLLAMA_MAX_TOKENS,
        "timeout": LOCAL_OLLAMA_TIMEOUT
    },
    "google": {
        "api_key": GCP_API_KEY,
        "model": GOOGLE_DEFAULT_MODEL_ID,
        "temperature": GOOGLE_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    },
    "openai": {
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_DEFAULT_MODEL_ID,
        "temperature": OPENAI_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    },
    "deepseek": {
        "api_key": DEEPSEEK_API_KEY,
        "model": DEEPSEEK_DEFAULT_MODEL_ID,
        "temperature": DEEPSEEK_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    },
    "claude": {
        "api_key": CLAUDE_API_KEY,
        "model": CLAUDE_DEFAULT_MODEL_ID,
        "temperature": CLAUDE_MODEL_TEMPERATURE,
        "max_tokens": 10000,
        "timeout": 60
    },
    "grok": {
        "api_key": XAI_API_KEY,
        "model": GROK_MODEL_ID_GROK_2,
        "temperature": GROK_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    },
    "togetherai": {
        "api_key": TOGETHERAI_API_KEY,
        "model": TOGETHERAI_MODEL_ID_LLAMA_4_17b,
        "temperature": TOGETHERAI_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    },
}

# ============================================================================
# IMAGE FIRST STRATEGY CONFIGURATIONS
# ============================================================================

# PDF to Image Conversion Configuration
PDF_TO_IMAGE_DPI = 300  # DPI for image conversion
PDF_TO_IMAGE_FORMAT = "PNG"  # Output image format
PDF_TO_IMAGE_QUALITY = 95  # Image quality (for JPEG)

# Provider-specific settings for image processing
IMAGE_PROVIDER_CONFIGS = {
    "ollama": {
        "model": LOCAL_OLLAMA_MODEL_DEEP_R1,
        "temperature": LOCAL_OLLAMA_TEMPERATURE,
        "max_tokens": LOCAL_OLLAMA_MAX_TOKENS,
        "timeout": LOCAL_OLLAMA_TIMEOUT
    },
    "google": {
        "api_key": GCP_API_KEY,
        "model": GOOGLE_DEFAULT_MODEL_ID,
        "temperature": GOOGLE_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    },
    "openai": {
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_DEFAULT_MODEL_ID,
        "temperature": OPENAI_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    },
    "deepseek": {
        "api_key": DEEPSEEK_API_KEY,
        "model": DEEPSEEK_DEFAULT_MODEL_ID,
        "temperature": DEEPSEEK_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    },
    "claude": {
        "api_key": CLAUDE_API_KEY,
        "model": CLAUDE_DEFAULT_MODEL_ID,
        "temperature": CLAUDE_MODEL_TEMPERATURE,
        "max_tokens": 10000,
        "timeout": 60
    },
    "huggingface": {
        "api_key": HUGGINGFACE_TOKEN,
        "model": HUGGINGFACE_DEFAULT_MODEL_ID,
        "temperature": HUGGINGFACE_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    },
    "togetherai": {
        "api_key": TOGETHERAI_API_KEY,
        "model": TOGETHERAI_MODEL_ID_LLAMA_VISION_90B,
        "temperature": TOGETHERAI_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    },
    "grok": {
        "api_key": XAI_API_KEY,
        "model": GROK_MODEL_ID_GROK_2,
        "temperature": GROK_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    }
}

# ============================================================================
# HUGGINGFACE STRATEGY CONFIGURATIONS
# ============================================================================

# HuggingFace Provider Configuration
HUGGINGFACE_PROVIDER_CONFIG = {
    "api_key": HUGGINGFACE_TOKEN,
    "model": HUGGINGFACE_DEFAULT_MODEL_ID,
    "temperature": HUGGINGFACE_MODEL_TEMPERATURE,
    "max_tokens": 24000,
    "timeout": 60
}

# Model-specific configurations for HuggingFace
HUGGINGFACE_MODEL_CONFIGS = {
    HUGGINGFACE_MODEL_ID_QWEN2_VL_72B: {
        "api_key": HUGGINGFACE_TOKEN,
        "model": HUGGINGFACE_MODEL_ID_QWEN2_VL_72B,
        "temperature": HUGGINGFACE_MODEL_TEMPERATURE,
        "max_tokens": 24000,
        "timeout": 60
    },
    HUGGINGFACE_MODEL_ID_LLAMA_VISION_90B: {
        "api_key": HUGGINGFACE_TOKEN,
        "model": HUGGINGFACE_MODEL_ID_LLAMA_VISION_90B,
        "temperature": HUGGINGFACE_MODEL_TEMPERATURE,
        "max_tokens": 24000,
        "timeout": 60
    }
}
