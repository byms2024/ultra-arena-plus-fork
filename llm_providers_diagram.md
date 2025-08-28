graph TD
    %% Define houses with different colors
    subgraph "🏠 OpenAI House"
        A[👤 OpenAI<br/>GPT-4.1, GPT-4o-mini]
    end
    
    subgraph "🏠 Claude House"
        B[👤 Claude<br/>Sonnet 4]
    end
    
    subgraph "🏠 Google House"
        C[👤 Google<br/>Gemini 2.5 Flash]
    end
    
    subgraph "🏠 Grok House"
        D[👤 Grok<br/>Grok 2 Vision]
    end
    
    subgraph "🏠 DeepSeek House" 
        E[👤 DeepSeek<br/>DeepSeek Chat]
        style E fill:#ff9999
    end
    
    subgraph "🏠 TogetherAI House"
        F[👤 TogetherAI<br/>Llama 4 Maverick<br/>Llama 3.2 90B Vision]
        style F fill:#99ccff
    end
    
    %% Arrange houses in a circle
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> A
    
    %% Style the houses
    style A fill:#f9f9f9
    style B fill:#f9f9f9
    style C fill:#f9f9f9
    style D fill:#f9f9f9
    style E fill:#ffcccc
    style F fill:#cce5ff
    
    %% Add title
    classDef titleClass fill:#e6f3ff,stroke:#333,stroke-width:2px
    class A,B,C,D,E,F titleClass

## LLM Providers Architecture

This diagram shows the different LLM providers as stick figures living in houses, arranged in a circular pattern:

- **Standard Houses** (Light Gray): OpenAI, Claude, Google, Grok
- **DeepSeek House** (Light Red): DeepSeek with DeepSeek Chat model
- **TogetherAI House** (Light Blue): TogetherAI with Llama 4 Maverick and Llama 3.2 90B Vision models

Each provider represents a different AI service that can be used for document processing and text extraction tasks.

