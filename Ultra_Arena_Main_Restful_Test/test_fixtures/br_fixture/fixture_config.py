from pathlib import Path

INPUT_PDF_DIR_PATH = Path("input_files") / "1_file"
# BENCHMARK_FILE_PATH = Path("benchmark_files") / "benchmark_252_files.xlsx"
BENCHMARK_FILE_PATH = Path("benchmark_files") / "benchmark_252.csv"
OUTPUT_DIR = Path("output_files")

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
          "CNPJ_1": "46.621.491/0002-70",
          "CNPJ_2": null,
          "VALOR_TOTAL": "2.465,73",
          "Chassi": "LGXCE4CC7S0023860",
          "CLAIM_NUMBER": "BYDAMEBR0015WCN241200032_01",
          "INVOICE_NO" : "1859",
          "INVOICE_ISSUE_DATE" : "2025-01-01"
        },
        {
          "DOC_TYPE": "Serviço",
          "CNPJ_1": "46.621.491/0002-70",
          "CNPJ_2": "17.140.820/0007-77",
          "VALOR_TOTAL": "1.023,40",
          "Chassi": null,
          "CLAIM_NUMBER": "BYDAMEBR0015WCN250100042_01",
          "INVOICE_NO" : "1860",
          "INVOICE_ISSUE_DATE" : "2025-01-02"
        }
      ]
      ```
"""

# Mandatory Keys Configuration
MANDATORY_KEYS = ['DOC_TYPE', 'CNPJ_1', 'VALOR_TOTAL', 'Chassi', 'CLAIM_NUMBER', "INVOICE_NO", "INVOICE_ISSUE_DATE"]

# Text First Strategy Criteria for No Need to Switch to 
# Secondary Text Extractor before LLM
TEXT_FIRST_REGEX_CRITERIA = {
    'CNPJ_1' : r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b',
    'Chassi' : r'\bL[A-Z0-9]{16}\b',
    'CLAIM_NUMBER' : r'\bBYDAMEBR[A-Z0-9]{16,18}_\d{2}\b'
}

# User Prompt
USER_PROMPT = """
    Esta deve ser um arquivo de um recibo de serviços ou peças vendidas para pós-venda de carros BYD no Brasil. Você é um especialista em extrair informações importantes dele.

    **⚠️ REGRA CRÍTICA: Extraia APENAS informações que estão REALMENTE presentes no arquivo. NÃO invente, copie ou alucine valores baseados em exemplos ou padrões.**

    Precisamos **extrair informações cruciais** e formatá-las em um objeto JSON.

    **As seguintes chaves DEVEM estar no NÍVEL MAIS ALTO do objeto JSON de saída:**
    **['DOC_TYPE', 'CNPJ_1', 'CNPJ_2', 'VALOR_TOTAL', 'Chassi', 'CLAIM_NUMBER', "INVOICE_NO", "INVOICE_ISSUE_DATE"]**
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
    * **'Chassi'**: O valor único para 'Chassi' **SEMPRE** começa com 'L' e **DEVE TER EXATAMENTE 17 CARACTERES** (número VIN padrão). **IMPORTANTE**: Extraia APENAS o valor que está REALMENTE presente no documento. **NÃO** invente ou copie valores de exemplos.
    * **'CLAIM_NUMBER'**: O valor único para 'CLAIM_NUMBER' está localizado no final do arquivo, após a seção 'INFORMAÇÕES COMPLEMENTARES'. Este valor **SEMPRE** começa com 'BYDAMEBR' e **DEVE TER 28-30 CARACTERES**, no formato "BYDAMEBRXXXXXXXXXXXXXXXXX_XX". 
    * **'INVOICE_NO'**: O valor único para 'INVOICE_NO' está localizado no início do arquivo, normalmente após o texto 'Nº da NF', 'Nº da NFS'. Este valor **NUNCA** vem depois do 'Número RPS', pois se trata de outro número que **NÃO** desejamos coletar. O número da nota fiscal, que queremos coletar é um número de 0 a 1.000.000, normalmente na casa de 1.000 a 10.000. (ex.: 1980)
    * **'INVOICE_ISSUE_DATE'**: O valor único para 'INVOICE_ISSUE_DATE' está localizado no início do arquivo, normalmente após o texto 'Emitida em' ou semelhante. Este valor **SEMPRE** possui formato de data brasileira, com ''dia'/'mês'/'ano'', no formato "DD/MM/AAAA, onde DD, MM, AAAA representam o dia, mês e ano, respectivamente". **IMPORTANTE**: Extraia APENAS o valor que está REALMENTE presente no documento. NÃO invente ou copie valores de exemplos.

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
    As chaves especificadas (['DOC_TYPE', 'CNPJ_1', 'CNPJ_2', 'VALOR_TOTAL', 'Chassi', 'CLAIM_NUMBER', "INVOICE_NO", "INVOICE_ISSUE_DATE"]) SÃO OBRIGATÓRIAS e DEVEM SER COLOCADAS DIRETAMENTE NO NÍVEL RAIZ/TOPO DO JSON. Elas não devem estar aninhadas.
    
    **VALORES NULOS:**
    Se uma informação não puder ser encontrada no documento, use `null` para esse campo. É melhor retornar `null` do que inventar um valor.  

""" + JSON_FORMAT_INSTRUCTIONS