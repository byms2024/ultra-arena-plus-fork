from .config_param_grps import param_grps

# Centralized combo configurations
# All combo configurations consolidated from profile-specific configs
combo_config = {
    "combo1" : {
        "strategy_groups" : [
            "grp_textF_dSeek_dChat_para",
            "grp_directF_google_gemini25_para"
        ]
    },
    "combo2" : {
        "strategy_groups" : [
            "grp_textF_dSeek_dChat_para",
            "grp_textFirst_openai_gpt4_para"
        ]
    },
    "combo_2_text_first_strategies" : {
        "strategy_groups" : [
            "grp_textF_dSeek_dChat_para",
            "grp_textF_google_gemini25_para"
        ]
    },
    "combo_test_3_strategies" : {
        "strategy_groups" : [
            "grp_directF_google_gemini25_para",
            "grp_textF_dSeek_dChat_para",
            "grp_textFirst_openai_gpt4_para"
        ]
    }, 
    "combo_potential_top_4_strategies" : {
        "strategy_groups" : [
            "grp_textF_dSeek_dChat_para",
            "grp_textF_google_gemini25_para",            
            "grp_directF_google_gemini25_para",
            "grp_imageF_google_gemini25_para",
        ]
    },    
     "combo_test_top_4_strategies_2" : {
        "strategy_groups" : [
            "grp_directF_google_gemini25_para",
            "grp_textF_google_gemini25_para",
            "grp_textF_dSeek_dChat_para",
            "grp_test_textF_claude_para",
        ]
    },
    "combo_test_top_3_strategies_2" : {
        "strategy_groups" : [
            "grp_directF_google_gemini25_para",
            "grp_textF_google_gemini25_para",
            # "grp_textF_dSeek_dChat_para",
            "grp_test_textF_claude_para",
        ]
    },
    "combo_test_top_2_strategies_2" : {
        "strategy_groups" : [
            "grp_directF_google_gemini25_para",
            "grp_textF_google_gemini25_para",
            # "grp_textF_dSeek_dChat_para",
            # "grp_test_textF_claude_para",
        ]
    },    
    "combo_test_5_sub_strategies" : {
        "strategy_groups" : [
            "grp_test_imageF_claude_para",
            "grp_test_textF_claude_para",
            "grp_imageF_togetherai_llama_4_17b_para",
            "grp_imageF_togetherai_llama_vision_90b_para",
            "grp_imageF_grok_2_para"
        ]
    },  
    "combo_test_10_strategies" : {
        "strategy_groups" : [
            "grp_directF_google_gemini25_para",
            "grp_imageF_google_gemini25_para",
            "grp_textF_google_gemini25_para",
            # "grp_directF_dSeek_dChat_para", # WARNING - no support for direct pdf file
            # "grp_imageF_dSeek_dChat_para", #    WARNING - "Unsupported file type: temp_images/image_5eb95081.png"
            "grp_textF_dSeek_dChat_para",  
            # "grp_test_textF_openai_para", 
            # "error": "Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}"
            # 2025-08-28 00:00:15,202 - [ThreadPoolExecutor-21_3][request] - DEBUG - Encountered httpx.HTTPStatusError
            # 
            # Traceback (most recent call last):
            # File "/Users/jameswang9311/anaconda3/lib/python3.11/site-packages/openai/_base_client.py", line 1024, in request
            #     response.raise_for_status()
            # File "/Users/jameswang9311/anaconda3/lib/python3.11/site-packages/httpx/_models.py", line 829, in raise_for_status
            #     raise HTTPStatusError(message, request=request, response=self)
            # httpx.HTTPStatusError: Client error '429 Too Many Requests' for url 'https://api.openai.com/v1/chat/completions'
            "grp_test_imageF_openai_para",
            "grp_test_imageF_claude_para",
            "grp_test_textF_claude_para",
            # "grp_imageF_togetherai_llama_4_17b_para", # PROBLEM: togetherai error: "Only one image_url per content array is supported" 
            # "grp_imageF_togetherai_llama_vision_90b_para", # PROBLEM: togetherai error: "Only one image_url per content array is supported" 
            # "grp_imageF_grok_2_para" # no point, obviously over-expensive
        ]
    },
    "combo_test_7_strategies" : {
        "strategy_groups" : [
            "grp_directF_google_gemini25_para",
            "grp_imageF_google_gemini25_para",
            "grp_textF_google_gemini25_para",
            # "grp_directF_dSeek_dChat_para", # WARNING - no support for direct pdf file
            # "grp_imageF_dSeek_dChat_para", #    WARNING - Unsupported file type: temp_images/image_5eb95081.png
            "grp_textF_dSeek_dChat_para",
            "grp_test_imageF_openai_para",
            "grp_test_imageF_claude_para",
            "grp_test_textF_claude_para"  
        ]
    },    
    "combo_test_textF_ollama_strategies" : {
        "strategy_groups" : [
            "grp_textF_ollama_deepR1_para",      # text_first + Ollama DeepR1 local
        ]
    },
    "combo_test_textF_strategies" : {
        "strategy_groups" : [
            "grp_textF_google_gemini25_para",      # text_first + Google
            "grp_textF_dSeek_dChat_para",        # text_first + DeepSeek  
            "grp_test_textF_claude_para",         # image_first + Claude
            "grp_textF_togetherai_llama_4_17b_para",
            # "grp_textF_togetherai_llama_vision_90b_para",
            # "grp_textF_togetherai_qwen_vl_72b_para",
            # "grp_textF_grok_4_para", # not configured in the TEXT_PROVIDER in the base config
            "grp_textF_grok_2_para",
        ]
    },
    "combo_test_fileF_strategies" : {
        "strategy_groups" : [
            "grp_directF_google_gemini25_para",
            # "grp_directF_dSeek_dChat_para", # WARNING - no support for direct pdf file
            # "grp_textF_dSeek_dChat_para", # no longer bad, after fixing ""list indices must be integers or slices, not str""
        ]
    },
    "combo_test_imageF_strategies" : {
        "strategy_groups" : [
            "grp_test_imageF_openai_para", # only one that was successful
            "grp_test_imageF_claude_para",
            # "grp_imageF_google_gemini25_para",
            # "grp_textF_dSeek_dChat_para", # no longer bad, after fixing ""list indices must be integers or slices, not str""
        ]
    },
    "combo_test_imageF_openai_strategies" : {
        "strategy_groups" : [
            "grp_test_imageF_openai_para",
        ]
    },    

    "single_google_imageF_strategy" : {
        "strategy_groups" : [
            "grp_imageF_google_gemini25_para"
        ]
    }, 
    "combo_test_bad_strategies" : {
        "strategy_groups" : [
            "grp_test_imageF_claude_para",
            "grp_imageF_google_gemini25_para",
            # "grp_textF_dSeek_dChat_para", # no longer bad, after fixing ""list indices must be integers or slices, not str""
        ]
    },
    "combo_test_google_strategies" : {
        "strategy_groups" : [
            "grp_textF_google_gemini25_para",
            "grp_directF_google_gemini25_para",
            "grp_imageF_google_gemini25_para", 
        ]
    },  
    "combo_test_google_file_strategies" : {
        "strategy_groups" : [
            "grp_directF_google_gemini25_para",
            "grp_imageF_google_gemini25_para", 
        ]
    }, 
    "combo_test_google_directF_strategies" : {
        "strategy_groups" : [
            "grp_directF_google_gemini25_para",
            "grp_textF_dSeek_dChat_para"
        ]
    },
    "combo_test_claude_strategies" : {
        "strategy_groups" : [
            "grp_test_imageF_claude_para",
            "grp_test_textF_claude_para"
        ]
    },  
    "combo_test_deepseek_strategies" : {
        "strategy_groups" : [
            # "grp_imageF_dSeek_dChat_para", #  TODO: investigateno more this error but still failed:  WARNING - Unsupported file type: temp_images/image_5eb95081.png
            # "grp_directF_dSeek_dChat_para", # WARNING - no support for direct pdf file
            "grp_textF_dSeek_dChat_para", 
        ]
    },    
    "combo_test_3_textF_strategies" : {
        "strategy_groups" : [
            "grp_textF_dSeek_dChat_para",
            "grp_textF_google_gemini25_para",
            "grp_test_textF_claude_para"
        ]
    },
    "test_directF_dSeek_dChat_only" : {
        "strategy_groups" : [
            # "grp_directF_dSeek_dChat_para", # WARNING - no support for direct pdf file
        ]
    },
  
    # "test_imageF_dSeek_dChat_only" : {
    #     "input_files" : DIR_1_FILES,
    #     "strategy_groups" : [
    #         "grp_imageF_dSeek_dChat_para"
    #     ]
    # },        

    "test_textF_openai_only" : {
        "strategy_groups" : [
            "grp_test_textF_openai_para"
        ]
    },

    "test_both_strategies_openai" : {
        "strategy_groups" : [
            "grp_test_imageF_openai_para",
            "grp_test_textF_openai_para"
        ]
    },
    "test_huggingface_models" : {
        "strategy_groups" : [
            "grp_imageF_huggingface_qwen_para",
            "grp_imageF_huggingface_llama_para"
        ]
    },
    "test_huggingface_qwen_only" : {
        "strategy_groups" : [
            "grp_imageF_huggingface_qwen_para"
        ]
    },
    "test_huggingface_llama_only" : {
        "strategy_groups" : [
            "grp_imageF_huggingface_llama_para"
        ]
    },
    "test_huggingface_dotocr_only" : {
        "strategy_groups" : [
            "grp_imageF_huggingface_dotocr_para"
        ]
    },
    "single_test_textF_openai" : {
        "strategy_groups" : [
            "grp_test_textF_openai_para"
        ]
    },    
    "single_test_textF_ollama_gptOss20b" : {
        "strategy_groups" : [
            "grp_textF_ollama_gptOss20b_para"
        ]
    },
    "single_test_textF_ollama_llama3" : {
        "strategy_groups" : [
            "grp_textF_ollama_llama3_para"
        ]
    },
    "single_test_textF_ollama_gemma3" : {
        "strategy_groups" : [
            "grp_textF_ollama_gemma3_para"
        ]
    },        
    "single_test_textF_ollama_deepR1" : {
        "strategy_groups" : [
            "grp_textF_ollama_deepR1_para"
        ]
    },
    "single_test_textF_ollama_mistral" : {
        "strategy_groups" : [
            "grp_textF_ollama_mistral_para"
        ]
    },
    "combo_test_textF_ollama_2s" : {
        "strategy_groups" : [
            "grp_textF_ollama_deepR1_para",
            "grp_textF_ollama_mistral_para"
        ]
    },
    "combo_test_textF_ollama_3s" : {
        "strategy_groups" : [
            "grp_textF_ollama_gptOss20b_para",
            "grp_textF_ollama_deepR1_para",
            "grp_textF_ollama_mistral_para"
        ]
    },
    "combo_test_textF_ollama_3s_noDp" : {
        "strategy_groups" : [
            "grp_textF_ollama_gemma3_para",
            "grp_textF_ollama_llama3_para",
            "grp_textF_ollama_mistral_para"
        ]
    },    
    
    # # qwen_vl_72b take 300s to run, not practical
    # "test_togetherai_qwen_vl_72b_only" : {
    #     "input_files" : DIR_1_FILES,
    #     "strategy_groups" : [
    #         "grp_imageF_togetherai_qwen_vl_72b_para"
    #     ]
    # },
    "test_togetherai_llama_vision_90b_only" : {
        "strategy_groups" : [
            "grp_imageF_togetherai_llama_vision_90b_para"
        ]
    },
    "test_togetherai_llama_4_17b_only" : {
        "strategy_groups" : [
            "grp_imageF_togetherai_llama_4_17b_para"
        ]
    },
    # # too slow - 196s for 1 file with no accuracy
    # "test_grok_4_only" : {
    #     "input_files" : DIR_1_FILES,
    #     "strategy_groups" : [
    #         "grp_imageF_grok_4_para"
    #     ]
    # },
    "test_grok_2_only" : {
        "strategy_groups" : [
            "grp_imageF_grok_2_para"
        ]
    },
    
    # Additional single-strategy combos for common use cases

    # "single_strategy_direct_file_deepseek" : {
    #     "strategy_groups" : [
    #         # "grp_directF_dSeek_dChat_para", # WARNING - no support for direct pdf file
    #     ]
    # },
    "single_strategy_text_first_deepseek" : {
        "strategy_groups" : [
            "grp_textF_dSeek_dChat_para"
        ]
    },
    "single_strategy_text_first_google" : {
        "strategy_groups" : [
            "grp_textF_google_gemini25_para"
        ]
    },
    "single_strategy_direct_file_google" : {
        "strategy_groups" : [
            "grp_directF_google_gemini25_para"
        ]
    },
    "single_strategy_text_first_claude" : {
        "strategy_groups" : [
            "grp_test_textF_claude_para"
        ]
    },    
    "single_strategy_image_first_claude" : {
        "strategy_groups" : [
            "grp_test_imageF_claude_para"
        ]
    },
    'chain_strategy' : {
        "strategy_groups" : [
            "grp_chain_df_tf_if_google_para"
        ]
    }


}
