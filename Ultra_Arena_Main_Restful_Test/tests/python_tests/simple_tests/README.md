# Simple Tests

This directory contains simple test scripts for the Ultra Arena Main RESTful API.

## Available Tests

### `test_text_first_deepseek_4_files.py`

A simple test script that demonstrates the text first strategy with DeepSeek using 4 files from the br_fixture.

**Features:**
- Uses the `br_fixture` configuration (Brazilian Portuguese prompts for BYD car documents)
- Overrides `INPUT_PDF_DIR_PATH` to use the `4_files` directory instead of the default `1_file`
- Tests the `combo_test_deepseek_strategies` combo
- Runs in evaluation mode with benchmark comparison
- Processes 4 PDF files from the test fixture

**Files processed:**
- `single/NFP 615 BYDAMEBR0049WCN240700001_01.pdf`
- `dahruj/DAHRUJ - CEASA/NFS 2499 BYDAMEBR0020WCN241200011_01.pdf`
- `dahruj/DAHRUJ - CEASA/NFP 16693 BYDAMEBR0020WCN250100005_01.pdf`
- `dahruj/DAHRUJ - CAMPINAS/NFS 4651 BYDAMEBR0015WCN250100042_01.pdf`
- `dahruj/DAHRUJ - CAMPINAS/NFP 25943 BYDAMEBR0015WCN241200032_01.pdf`

## Usage

### Prerequisites

1. Make sure the REST server is running with the br_profile_restful:
   ```bash
   cd Ultra_Arena_Main_Restful
   export RUN_PROFILE=br_profile_restful
   nohup python server.py > server.log 2>&1 &
   ```

2. Verify the server is running:
   ```bash
   curl http://localhost:5002/health
   ```

### Running the Test

```bash
cd Ultra_Arena_Main_Restful_Test
python tests/python_tests/simple_tests/test_text_first_deepseek_4_files.py
```

## Configuration

The test uses the `br_fixture` configuration from `test_fixtures/br_fixture/fixture_config.py` with the following overrides:

- **INPUT_PDF_DIR_PATH**: Changed from `input_files/1_file` to `input_files/4_files`
- **All other settings**: Use the default br_fixture configuration (Brazilian Portuguese prompts, BYD document processing)

## Expected Output

The test should return a successful response with:
- Status: `success`
- Combo used: `combo_test_deepseek_strategies`
- Strategy groups: `grp_textF_dSeek_dChat_para`
- Brazilian Portuguese prompts for BYD car document processing
