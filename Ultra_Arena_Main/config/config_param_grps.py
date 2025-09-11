from config import config_base

param_grps = {
    "grp_textF_ollama_gptOss20b_para" : {
        "strategy": config_base.STRATEGY_TEXT_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_OLLAMA,
        "model": config_base.LOCAL_OLLAMA_MODEL_GPT_OSS_20B,
        "temperature": config_base.LOCAL_OLLAMA_TEMPERATURE
    },  
    "grp_textF_ollama_deepR1_para" : {
        "strategy": config_base.STRATEGY_TEXT_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_OLLAMA,
        "model": config_base.LOCAL_OLLAMA_MODEL_DEEP_R1,
        "temperature": config_base.LOCAL_OLLAMA_TEMPERATURE
    },    
    "grp_textF_ollama_mistral_para" : {
        "strategy": config_base.STRATEGY_TEXT_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_OLLAMA,
        "model": config_base.LOCAL_OLLAMA_MODEL_MISTRAL,
        "temperature": config_base.LOCAL_OLLAMA_TEMPERATURE
    },    
    # Deepseek, Currently, the DeepSeek API (as of the latest version) does not support direct file uploads (PDF, DOCX, images, etc.) or processing via URLs. You can only send plain text as input in the API request.
    # The directF strategy for deepseek has been replaced with textF strategy since it actually goes through PyPDF2 text extraction 
    "grp_imageF_dSeek_dChat_para" : {
        "strategy": config_base.STRATEGY_IMAGE_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_DEEPSEEK,
        "model": config_base.DEEPSEEK_MODEL_DCHAT,
        "temperature": config_base.DEEPSEEK_MODEL_TEMPERATURE
    },
    "grp_textF_dSeek_dChat_para" : {
        "strategy": config_base.STRATEGY_TEXT_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_DEEPSEEK,
        "model": config_base.DEEPSEEK_MODEL_DCHAT,
        "temperature": config_base.DEEPSEEK_MODEL_TEMPERATURE
    },
    "grp_directF_google_gemini25_para" : {
        "strategy": config_base.STRATEGY_DIRECT_FILE,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_GOOGLE,
        "model": config_base.GOOGLE_MODEL_ID_GEMINI_25_FLASH,
        "temperature": config_base.GOOGLE_MODEL_TEMPERATURE,
        "max_tokens": config_base.GOOGLE_MODEL_MAX_TOKENS
    },
    "grp_imageF_google_gemini25_para" : {
        "strategy": config_base.STRATEGY_IMAGE_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_GOOGLE,
        "model": config_base.GOOGLE_MODEL_ID_GEMINI_25_FLASH,
        "temperature": config_base.GOOGLE_MODEL_TEMPERATURE,
        "max_tokens": config_base.GOOGLE_MODEL_MAX_TOKENS
    },
    "grp_textF_google_gemini25_para" : {
        "strategy": config_base.STRATEGY_TEXT_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_GOOGLE,
        "model": config_base.GOOGLE_MODEL_ID_GEMINI_25_FLASH,
        "temperature": config_base.GOOGLE_MODEL_TEMPERATURE,
        "max_tokens": config_base.GOOGLE_MODEL_MAX_TOKENS
    },
    "grp_test_imageF_openai_para" : {
        "strategy": config_base.STRATEGY_IMAGE_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_OPENAI,
        "model": config_base.OPENAI_MODEL_GPT_41,
        "temperature": config_base.OPENAI_MODEL_TEMPERATURE
    },
    "grp_test_textF_openai_para" : {
        "strategy": config_base.STRATEGY_TEXT_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_OPENAI,
        "model": config_base.OPENAI_MODEL_GPT_41,
        "temperature": config_base.OPENAI_MODEL_TEMPERATURE
    },
    "grp_test_imageF_claude_para" : {
        "strategy": config_base.STRATEGY_IMAGE_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_CLAUDE,
        "model": config_base.CLAUDE_MODEL_CLAUDE_4_SONNET,
        "temperature": config_base.CLAUDE_MODEL_TEMPERATURE
    },
    "grp_test_textF_claude_para" : {
        "strategy": config_base.STRATEGY_TEXT_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_CLAUDE,
        "model": config_base.CLAUDE_MODEL_CLAUDE_4_SONNET,
        "temperature": config_base.CLAUDE_MODEL_TEMPERATURE
    },
    "grp_textFirst_openai_gpt4_para" : {
        "strategy": config_base.STRATEGY_TEXT_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_OPENAI,
        "model": config_base.OPENAI_DEFAULT_MODEL_ID,
        "temperature": config_base.OPENAI_MODEL_TEMPERATURE
    },
    # "grp_imageF_huggingface_qwen_para" : { # huggingface is not working - File too large error
    #     "strategy": config_base.STRATEGY_IMAGE_FIRST,
    #     "mode": config_base.MODE_BATCH_PARALLEL,
    #     "provider": config_base.PROVIDER_HUGGINGFACE,
    #     "model": config_base.HUGGINGFACE_MODEL_ID_QWEN2_VL_72B,
    #     "temperature": config_base.HUGGINGFACE_MODEL_TEMPERATURE
    # },
    # "grp_imageF_huggingface_llama_para" : {  # huggingface is not working - File too large error
    #     "strategy": config_base.STRATEGY_IMAGE_FIRST,
    #     "mode": config_base.MODE_BATCH_PARALLEL,
    #     "provider": config_base.PROVIDER_HUGGINGFACE,
    #     "model": config_base.HUGGINGFACE_MODEL_ID_LLAMA_VISION_90B,
    #     "temperature": config_base.HUGGINGFACE_MODEL_TEMPERATURE
    # },
    # "grp_imageF_huggingface_dotocr_para" : { # huggingface is not working - File too large error
    #     "strategy": config_base.STRATEGY_IMAGE_FIRST,
    #     "mode": config_base.MODE_BATCH_PARALLEL,
    #     "provider": config_base.PROVIDER_HUGGINGFACE,
    #     "model": config_base.HUGGINGFACE_MODEL_ID_DOTS_OCR,
    #     "temperature": config_base.HUGGINGFACE_MODEL_TEMPERATURE
    # },
    "grp_imageF_togetherai_llama_vision_90b_para" : {
        "strategy": config_base.STRATEGY_IMAGE_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_TOGETHERAI,
        "model": config_base.TOGETHERAI_MODEL_ID_LLAMA_VISION_90B,
        "temperature": config_base.TOGETHERAI_MODEL_TEMPERATURE
    },
    "grp_textF_togetherai_llama_vision_90b_para" : {
        "strategy": config_base.STRATEGY_TEXT_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_TOGETHERAI,
        "model": config_base.TOGETHERAI_MODEL_ID_LLAMA_VISION_90B,
        "temperature": config_base.TOGETHERAI_MODEL_TEMPERATURE
    },
    # "grp_imageF_togetherai_qwen_vl_72b_para" : {  # Model too slow - 196s for 1 file with no accuracy
    #     "strategy": config_base.STRATEGY_IMAGE_FIRST,
    #     "mode": config_base.MODE_BATCH_PARALLEL,
    #     "provider": config_base.PROVIDER_TOGETHERAI,
    #     "model": config_base.TOGETHERAI_MODEL_ID_QWEN2_VL_72B,
    #     "temperature": config_base.TOGETHERAI_MODEL_TEMPERATURE
    # },
    # "grp_textF_togetherai_qwen_vl_72b_para" : {  # Model too slow - 196s for 1 file with no accuracy
    #     "strategy": config_base.STRATEGY_TEXT_FIRST,
    #     "mode": config_base.MODE_BATCH_PARALLEL,
    #     "provider": config_base.PROVIDER_TOGETHERAI,
    #     "model": config_base.TOGETHERAI_MODEL_ID_QWEN2_VL_72B,
    #     "temperature": config_base.TOGETHERAI_MODEL_TEMPERATURE
    # },    
    "grp_imageF_togetherai_llama_4_17b_para" : {
        "strategy": config_base.STRATEGY_IMAGE_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_TOGETHERAI,
        "model": config_base.TOGETHERAI_MODEL_ID_LLAMA_4_17b,
        "temperature": config_base.TOGETHERAI_MODEL_TEMPERATURE
    },
    "grp_textF_togetherai_llama_4_17b_para" : {
        "strategy": config_base.STRATEGY_TEXT_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_TOGETHERAI,
        "model": config_base.TOGETHERAI_MODEL_ID_LLAMA_4_17b,
        "temperature": config_base.TOGETHERAI_MODEL_TEMPERATURE
    },
    # "grp_imageF_grok_4_para" : {  # Model too expensive and slow
    #     "strategy": config_base.STRATEGY_IMAGE_FIRST,
    #     "mode": config_base.MODE_BATCH_PARALLEL,
    #     "provider": config_base.PROVIDER_GROK,
    #     "model": config_base.GROK_MODEL_ID_GROK_4,
    #     "temperature": config_base.GROK_MODEL_TEMPERATURE
    # },
    # "grp_textF_grok_4_para" : {  # Model too expensive and slow
    #     "strategy": config_base.STRATEGY_TEXT_FIRST,
    #     "mode": config_base.MODE_BATCH_PARALLEL,
    #     "provider": config_base.PROVIDER_GROK,
    #     "model": config_base.GROK_MODEL_ID_GROK_4,
    #     "temperature": config_base.GROK_MODEL_TEMPERATURE
    # },
    "grp_imageF_grok_2_para" : {
        "strategy": config_base.STRATEGY_IMAGE_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_GROK,
        "model": config_base.GROK_MODEL_ID_GROK_2,
        "temperature": config_base.GROK_MODEL_TEMPERATURE
    },
    "grp_textF_grok_2_para" : {
        "strategy": config_base.STRATEGY_TEXT_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_GROK,
        "model": config_base.GROK_MODEL_ID_GROK_2,
        "temperature": config_base.GROK_MODEL_TEMPERATURE
    }

}