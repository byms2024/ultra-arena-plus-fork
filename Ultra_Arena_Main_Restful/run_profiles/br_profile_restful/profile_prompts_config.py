from pathlib import Path
# =============================================================================
# PROMPT CONFIGURATION SECTION
# =============================================================================
# This section contains domain-specific prompts and extraction rules for processing
# BYD car post-sale documents in Brazilian Portuguese.

# System Prompt
SYSTEM_PROMPT = """
    Você é um especialista em extrair informações importantes de documentos de pós-venda de carros BYD no Brasil.
"""

# JSON Formatting Instructions (concatenated to main prompt)
SENSITIVE_JSON_FORMAT_INSTRUCTIONS = """
    **⚠️ REGRA CRÍTICA DE FORMATAÇÃO JSON:**
    * **NUNCA** responda em texto livre ou narrativo
    * **SEMPRE** responda com JSON válido e bem formatado
    * Se houver múltiplas arquivos, retorne um **ARRAY JSON** com um objeto para cada imagem
    * Se houver uma única arquivo, retorne um **ARRAY JSON** com um **OBJETO JSON** único
    * **EXEMPLO PARA MÚLTIPLAS IMAGENS:**
      ```json
      [
        {
          "DOC_TYPE": "Peças",
          "CNPJ_1": "46.621.491/0002-70",
          "CNPJ_2": null,
          "VALOR_TOTAL": "2.465,73",
          "Chassi": "LGXCE4CC7S0023860",
          "CLAIM_NUMBER": "BYDAMEBR0015WCN241200032_01"
        },
        {
          "DOC_TYPE": "Serviço",
          "CNPJ_1": "46.621.491/0002-70",
          "CNPJ_2": "17.140.820/0007-77",
          "VALOR_TOTAL": "1.023,40",
          "Chassi": null,
          "CLAIM_NUMBER": "BYDAMEBR0015WCN250100042_01"
        }
      ]
      ```
"""

# JSON Formatting Instructions (concatenated to main prompt)
JSON_FORMAT_INSTRUCTIONS = """
    **⚠️ REGRA CRÍTICA DE FORMATAÇÃO JSON:**
    * **NUNCA** responda em texto livre ou narrativo
    * **SEMPRE** responda com JSON válido e bem formatado
    * Se houver múltiplas arquivos, retorne um **ARRAY JSON** com um objeto para cada imagem
    * Se houver uma única arquivo, retorne um **ARRAY JSON** com um **OBJETO JSON** único
    * **EXEMPLO PARA MÚLTIPLAS IMAGENS:**
      ```json
      [
        {
          "DOC_TYPE": "Peças",
          "CNPJ_1": "CNPJ_3AD8FA19805EC192",
          "CNPJ_2": null,
          "VALOR_TOTAL": "2.465,73",
          "Chassi": "VIN_FB15C42CAE04A151",
          "CLAIM_NUMBER": "CLAIM_6047809DCD016B4D"
        },
        {
          "DOC_TYPE": "Serviço",
          "CNPJ_1": "CNPJ_3AA2GA45310DH021",
          "CNPJ_2": "CNPJ_4AR9GD83025MA531",
          "VALOR_TOTAL": "1.023,40",
          "Chassi": null,
          "CLAIM_NUMBER": "CLAIM_6047809DCD016B4D"
        }
      ]
      ```
"""


# Simplified JSON Formatting Instructions for Ollama models
SIMPLIFIED_JSON_FORMAT_INSTRUCTIONS = """
**⚠️ REGRA CRÍTICA DE FORMATAÇÃO JSON:**
* **NUNCA** responda com texto explicativo antes ou depois do JSON
* **NUNCA** use markdown code blocks (```json ou ```)
* **SEMPRE** responda APENAS com JSON válido, sem texto adicional
* **NUNCA** adicione notas, explicações ou comentários
* Se houver múltiplas arquivos, retorne um **ARRAY JSON** com um objeto para cada arquivo
* Se houver uma única arquivo, retorne um **ARRAY JSON** com um **OBJETO JSON** único
* **EXEMPLO PARA ARQUIVO ÚNICO:**
[
  {
    "DOC_TYPE": "Peças",
    "CNPJ_1": "46.621.491/0002-70",
    "VALOR_TOTAL": "2.465,73",
    "Chassi": "LGXCE4CC7S0023860",
    "CLAIM_NUMBER": "BYDAMEBR0015WCN241200032_01"
  }
]
* **EXEMPLO PARA MÚLTIPLAS ARQUIVOS:**
[
  {
    "DOC_TYPE": "Peças",
    "CNPJ_1": "46.621.491/0002-70",
    "VALOR_TOTAL": "2.465,73",
    "Chassi": "LGXCE4CC7S0023860",
    "CLAIM_NUMBER": "BYDAMEBR0015WCN241200032_01"
  },
  {
    "DOC_TYPE": "Serviço",
    "CNPJ_1": "46.621.491/0002-70",
    "VALOR_TOTAL": "1.023,40",
    "Chassi": null,
    "CLAIM_NUMBER": "BYDAMEBR0015WCN250100042_01"
  }
]
* **EXEMPLO INCORRETO:**
"Here is the extracted information in JSON format: [ ... ] Note: ..."
"""

# Mandatory Keys Configuration
MANDATORY_KEYS = ['DOC_TYPE', 'CNPJ_1', 'VALOR_TOTAL', 'Chassi', 'CLAIM_NUMBER']

# Reasoning Suppression Instruction (for reasoning models like DeepSeek R1)
REASONING_SUPPRESSION = "CRÍTICO: Responda APENAS com JSON válido. NÃO mostre raciocínio, NÃO mostre pensamentos, NÃO explique o processo. Forneça SOMENTE a resposta JSON final."

# Text First Strategy Criteria for No Need to Switch to 
# Secondary Text Extractor before LLM
TEXT_FIRST_REGEX_CRITERIA = {
    'CNPJ_1' : r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b',
    'Chassi' : r'\bL[A-Z0-9]{16}\b',
    'CLAIM_NUMBER' : r'\bBYDAMEBR[A-Z0-9]{16,18}_\d{2}\b'
}

# User prompt when sensitive informations will not be desensitized
SENSITIVE_USER_PROMPT = """
    Esta deve ser um arquivo de um recibo de serviços ou peças vendidas para pós-venda de carros BYD no Brasil. Você é um especialista em extrair informações importantes dele.

    **⚠️ REGRA CRÍTICA: Extraia APENAS informações que estão REALMENTE presentes no arquivo. NÃO invente, copie ou alucine valores baseados em exemplos ou padrões.**

    Precisamos **extrair informações cruciais** e formatá-las em um objeto JSON.

    **As seguintes chaves DEVEM estar no NÍVEL MAIS ALTO do objeto JSON de saída:**
    **['DOC_TYPE', 'CNPJ_1', 'CNPJ_2', 'VALOR_TOTAL', 'Chassi', 'CLAIM_NUMBER']**
    Para cada uma dessas chaves, extraia o valor único correspondente mas não invente o valor se ele não puder ser encontrado.

    ### **Instruções Específicas para Extração:**
        
    * **'DOC_TYPE'**: O valor para a chave **'DOC_TYPE'** deve ser **OBRIGATORIAMENTE** um dos seguintes: **'Serviço'**, **'Peças'**, ou **'Outros'**. A derivação deve seguir estas regras:
        1.  **'Serviço'**: Atribua este valor se o documento contiver **QUALQUER UM DOS SEGUINTES indicadores** em **posição de destaque (título, subtítulo ou cabeçalho superior)**:
            * A marca "NFS-e" ou "NFSe" (sem distinção de maiúsculas e minúsculas).
            * O "NOTA FISCAL ELETRÔNICA DE SERVIÇOS" (sem distinção de maiúsculas e minúsculas) ou uma variação muito próxima.
            * O "TOMADOR DE SERVIÇOS" (sem distinção de maiúsculas e minúsculas) ou uma variação muito semelhante.
        2.  **'Peças'**: Atribua este valor **SOMENTE** se as condições para 'Serviço' **NÃO** forem atendidas e o documento contiver a marca "NF-e" (case-insensitive) **em posição de destaque (título, subtítulo ou cabeçalho superior)**.
        3.  **'Outros'**: Atribua este valor se nenhuma das condições para 'Serviço' ou 'Peças' for atendida. 
    
    * se 'DOC_TYPE' for 'Outros', você deve ignorar o resto do prompt e atribuir valores nulos aos outros campos do json e encerrar o processamento.
     * **'CNPJ_1'**: O valor único para 'CNPJ_1' - primeira ocorrência que corresponde aos critérios: **DEVE TER EXATAMENTE 18 CARACTERES** e seguir o formato "XX.XXX.XXX/XXXX-XX". **IMPORTANTE**: Extraia APENAS o valor que está REALMENTE presente no documento. NÃO invente ou copie valores de exemplos.
    * se 'DOC_TYPE' for 'Serviço', **'CNPJ_2'**: O valor único para 'CNPJ_2' - segunda ocorrência que corresponde ao critério: **DEVE TER EXATAMENTE 18 CARACTERES** e seguir o formato "XX.XXX.XXX/XXXX-XX". **IMPORTANTE**: Extraia APENAS o valor que está REALMENTE presente no documento. NÃO invente ou copie valores de exemplos.
    
    * **'VALOR_TOTAL'**: O valor único para 'VALOR_TOTAL' é o valor associado a o campo 'VALOR_TOTAL'(sem distinção de maiúsculas e minúsculas) ou 'VALOR_TOTAL DO NOTA' (sem distinção de maiúsculas e minúsculas) e deve ser um número científico brasileiro (ex: 1.234,56). Se o valor original usar ponto como separador decimal (ex: 1234.56 ou 1,234.56), **converta-o para o formato brasileiro com vírgula como separador decimal.**
     * **'Chassi'**: O valor único para 'Chassi' **SEMPRE** começa com 'L' e **DEVE TER EXATAMENTE 17 CARACTERES** (número VIN padrão). **IMPORTANTE**: Extraia APENAS o valor que está REALMENTE presente no documento. NÃO invente ou copie valores de exemplos.
    * **'CLAIM_NUMBER'**: O valor único para 'CLAIM_NUMBER' está localizado no final do arquivo, após a seção 'INFORMAÇÕES COMPLEMENTARES'. Este valor **SEMPRE** começa com 'BYDAMEBR' e **DEVE TER 28-30 CARACTERES**, no formato "BYDAMEBRXXXXXXXXXXXXXXXXX_XX". 
    
    **⚠️ CRÍTICO PARA CLAIM_NUMBER:**
    * **SEMPRE** inclua o sufixo "_XX" (ex: "_01", "_02", etc.) no final do CLAIM_NUMBER
    * **NUNCA** trunque ou remova o sufixo "_XX" 
    * O formato completo deve ser: "BYDAMEBR" + 16-18 caracteres + "_" + 2 dígitos
    * **EXEMPLOS CORRETOS**: "BYDAMEBR0020WCN241200011_01", "BYDAMEBR0015WCN241200032_01"
    * **EXEMPLOS INCORRETOS**: "BYDAMEBR0020WCN241200011" (faltando "_01")
    
    **IMPORTANTE**: Extraia APENAS o valor que está REALMENTE presente no documento. NÃO invente ou copie valores de exemplos.

    **Formato de Saída:**
    * A saída **DEVE SER um objeto JSON VÁLIDO**.
    * O idioma da saída JSON (chaves e valores extraídos) **DEVE ser o Português do Brasil**.

    **REGRAS CRÍTICAS PARA AS CHAVES PRINCIPAIS:**
    As chaves especificadas (['DOC_TYPE', 'CNPJ_1', 'CNPJ_2', 'VALOR_TOTAL', 'Chassi', 'CLAIM_NUMBER']) SÃO OBRIGATÓRIAS e DEVEM SER COLOCADAS DIRETAMENTE NO NÍVEL RAIZ/TOPO DO JSON. Elas não devem estar aninhadas.
    
    **VALORES NULOS:**
    Se uma informação não puder ser encontrada no documento, use `null` para esse campo. É melhor retornar `null` do que inventar um valor.  

""" + SENSITIVE_JSON_FORMAT_INSTRUCTIONS

# User prompt when sensitive informations are desensitized
# Simplified User Prompt for Ollama (enhanced with specific extraction rules)
SIMPLIFIED_USER_PROMPT = """
Esta deve ser um arquivo de um recibo de serviços ou peças vendidas para pós-venda de carros BYD no Brasil. 
Extraia APENAS informações que estão REALMENTE presentes no arquivo. NÃO invente valores.

Extraia estas 5 informações e formate em JSON:

{
  "DOC_TYPE": "Serviço" ou "Peças" ou "Outros",
  "CNPJ_1": "primeiro CNPJ no formato XX.XXX.XXX/XXXX-XX",
  "VALOR_TOTAL": "valor total no formato brasileiro com vírgula",
  "Chassi": "código VIN de 17 caracteres começando com L",
  "CLAIM_NUMBER": "código completo começando com BYDAMEBR"
}

**REGRAS ESPECÍFICAS:**

**DOC_TYPE**: Deve ser OBRIGATORIAMENTE um dos seguintes:
- "Serviço": Se o documento contiver "NFS-e", "NFSe", "NOTA FISCAL ELETRÔNICA DE SERVIÇOS" ou "TOMADOR DE SERVIÇOS" em posição de destaque
- "Peças": Se não for serviço e contiver "NF-e" em posição de destaque
- "Outros": Se nenhuma das condições anteriores for atendida

**CNPJ_1**: Primeira ocorrência de CNPJ com EXATAMENTE 18 caracteres no formato XX.XXX.XXX/XXXX-XX

**VALOR_TOTAL**: Valor associado ao campo "VALOR_TOTAL" ou "VALOR_TOTAL DO NOTA". 
Deve estar no formato brasileiro (ex: 1.234,56). Se o valor original usar ponto como separador decimal, converta para vírgula.

**Chassi**: Código VIN que SEMPRE começa com 'L' e tem EXATAMENTE 17 caracteres

**CLAIM_NUMBER**: Localizado após "INFORMAÇÕES COMPLEMENTARES", SEMPRE começa com 'BYDAMEBR' 
e tem formato "BYDAMEBR" + 16-18 caracteres + "_" + 2 dígitos (ex: "BYDAMEBR0015WCN241200032_01")
**CRÍTICO**: SEMPRE inclua o sufixo "_XX" completo, nunca remova ou trunque

**FORMATO DE SAÍDA:**
- """ + REASONING_SUPPRESSION + """
- Responda APENAS com JSON válido
- Use null se não encontrar o valor
- Não adicione explicações ou texto extra
- Se DOC_TYPE for "Outros", defina outros campos como null
""" + SIMPLIFIED_JSON_FORMAT_INSTRUCTIONS


USER_PROMPT = """
     Esta deve ser um arquivo de um recibo de serviços ou peças vendidas para pós-venda de carros BYD no Brasil. Você é um especialista em extrair informações importantes dele.

    **⚠️ REGRA CRÍTICA: Extraia APENAS informações que estão REALMENTE presentes no arquivo. NÃO invente, copie ou alucine valores baseados em exemplos ou padrões.**

    Precisamos **extrair informações cruciais** e formatá-las em um objeto JSON.

    **As seguintes chaves DEVEM estar no NÍVEL MAIS ALTO do objeto JSON de saída:**
    **['DOC_TYPE', 'CNPJ_1', 'CNPJ_2', 'VALOR_TOTAL', 'Chassi', 'CLAIM_NUMBER']**
    Para cada uma dessas chaves, extraia o valor único correspondente mas não invente o valor se ele não puder ser encontrado.

    ### **Instruções Específicas para Extração:**
        
    * **'DOC_TYPE'**: O valor para a chave **'DOC_TYPE'** deve ser **OBRIGATORIAMENTE** um dos seguintes: **'Serviço'**, **'Peças'**, ou **'Outros'**. A derivação deve seguir estas regras:
        1.  **'Serviço'**: Atribua este valor se o documento contiver **QUALQUER UM DOS SEGUINTES indicadores** em **posição de destaque (título, subtítulo ou cabeçalho superior)**:
            * A marca "NFS-e" ou "NFSe" (sem distinção de maiúsculas e minúsculas).
            * O "NOTA FISCAL ELETRÔNICA DE SERVIÇOS" (sem distinção de maiúsculas e minúsculas) ou uma variação muito próxima.
            * O "TOMADOR DE SERVIÇOS" (sem distinção de maiúsculas e minúsculas) ou uma variação muito semelhante.
        2.  **'Peças'**: Atribua este valor **SOMENTE** se as condições para 'Serviço' **NÃO** forem atendidas e o documento contiver a marca "NF-e" (case-insensitive) **em posição de destaque (título, subtítulo ou cabeçalho superior)**.
        3.  **'Outros'**: Atribua este valor se nenhuma das condições para 'Serviço' ou 'Peças' for atendida. 
    
    * se 'DOC_TYPE' for 'Outros', você deve ignorar o resto do prompt e atribuir valores nulos aos outros campos do json e encerrar o processamento.
    * **'CNPJ_1'**: O valor único para 'CNPJ_1' - primeira ocorrência que corresponde aos critérios: **DEVE TER EXATAMENTE 21 CARACTERES** e seguir o formato "CNPJ_XXXXXXXXXXXXXXXX". **SOMENTE CASO NÃO ENCONTRE ESSE VALOR** considere a primeira correspondência que **DEVE TER EXATAMENTE 18 CARACTERES** e seguir o formato "XX.XXX.XXX/XXXX-XX". **IMPORTANTE**: Extraia APENAS o valor que está REALMENTE presente no documento. NÃO invente ou copie valores de exemplos.
    * se 'DOC_TYPE' for 'Serviço', **'CNPJ_2'**: O valor único para 'CNPJ_2' - segunda ocorrência que corresponde ao critério: **DEVE TER EXATAMENTE 21 CARACTERES** e seguir o formato "CNPJ_XXXXXXXXXXXXXXXX". **SOMENTE CASO NÃO ENCONTRE ESSE VALOR** considere a primeira correspondência que **DEVE TER EXATAMENTE 18 CARACTERES** e seguir o formato "XX.XXX.XXX/XXXX-XX". **IMPORTANTE**: Extraia APENAS o valor que está REALMENTE presente no documento. NÃO invente ou copie valores de exemplos.
    
    * **'VALOR_TOTAL'**: O valor único para 'VALOR_TOTAL' é o valor associado a o campo 'VALOR_TOTAL'(sem distinção de maiúsculas e minúsculas) ou 'VALOR_TOTAL DO NOTA' (sem distinção de maiúsculas e minúsculas) e deve ser um número científico brasileiro (ex: 1.234,56). Se o valor original usar ponto como separador decimal (ex: 1234.56 ou 1,234.56), **converta-o para o formato brasileiro com vírgula como separador decimal.**
    * **'Chassi'**: O valor único para 'Chassi' que corresponde aos critérios: **DEVE TER EXATAMENTE 20 CARACTERES** e seguir o formato "VIN_XXXXXXXXXXXXXXXX". **IMPORTANTE**: Extraia APENAS o valor que está REALMENTE presente no documento. NÃO invente ou copie valores de exemplos.
    * **'CLAIM_NUMBER'**: O valor único para 'CLAIM_NUMBER' está localizado no final do arquivo, após a seção 'INFORMAÇÕES COMPLEMENTARES'. Este valor **DEVE TER EXATAMENTE 22 CARACTERES** e seguir o formato "CLAIM_XXXXXXXXXXXXXXXX". 
    
    **IMPORTANTE**: Extraia APENAS o valor que está REALMENTE presente no documento. NÃO invente ou copie valores de exemplos.

    **Formato de Saída:**
    * A saída **DEVE SER um objeto JSON VÁLIDO**.
    * O idioma da saída JSON (chaves e valores extraídos) **DEVE ser o Português do Brasil**.

    **REGRAS CRÍTICAS PARA AS CHAVES PRINCIPAIS:**
    As chaves especificadas (['DOC_TYPE', 'CNPJ_1', 'CNPJ_2', 'VALOR_TOTAL', 'Chassi', 'CLAIM_NUMBER']) SÃO OBRIGATÓRIAS e DEVEM SER COLOCADAS DIRETAMENTE NO NÍVEL RAIZ/TOPO DO JSON. Elas não devem estar aninhadas.
    
    **VALORES NULOS:**
    Se uma informação não puder ser encontrada no documento, use `null` para esse campo. É melhor retornar `null` do que inventar um valor.  

""" + JSON_FORMAT_INSTRUCTIONS