import os
import time
import json
import requests
from pathlib import Path

BASE_URL = os.environ.get("UA_REST_BASE_URL", "http://localhost:5002")
API = f"{BASE_URL}/api/process/combo"

# This test assumes:
# - Server is running with a profile that includes the chain param group we added
# - Input dir contains at least two PDFs; one expected to fail the first strategy (direct_file)
#   but succeed on the second (text_first)

def _resolve_under_rest_root(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    # Ultra_Arena_Main_Restful_Test root (three levels up from this file)
    rest_root = Path(__file__).resolve().parents[3]
    parts = p.parts
    # If the relative starts with the root dir name, drop it to avoid duplication
    if parts and parts[0] == rest_root.name:
        p = Path(*parts[1:])
    return str((rest_root / p).resolve())


def test_chain_rest_basic():
    input_dir_env = os.environ.get("UA_TEST_INPUT_DIR", "Ultra_Arena_Main_Restful_Test/test_fixtures/br_fixture/input_files/3_files_chain_test/")
    output_dir_env = os.environ.get("UA_TEST_OUTPUT_DIR", "Ultra_Arena_Main_Restful_Test/test_fixtures/br_fixture/output_files")

    # Use absolute paths directly
    input_dir = "C:\\Users\\pcampos\\Desktop\\projetos\\byd\\ultra-arena-plus-fork\\Ultra_Arena_Main_Restful_Test\\test_fixtures\\br_fixture\\input_files\\3_files_chain_test"
    output_dir = "C:\\Users\\pcampos\\Desktop\\projetos\\byd\\ultra-arena-plus-fork\\Ultra_Arena_Main_Restful_Test\\test_fixtures\\br_fixture\\output_files"

    # Use a minimal combo that includes the chain param group
    # You can add this group to any combo and pass that combo here.
    combo_name = os.environ.get("UA_TEST_COMBO", "single_google_imageF_strategy")
    chain_name = os.environ.get("UA_TEST_CHAIN", "test_1")

    payload = {
        # When chain_name is present, combo_name is ignored by the server
        "chain_name": chain_name,
        "combo_name": combo_name,
        "input_pdf_dir_path": input_dir,
        "output_dir": output_dir,
        # Keep defaults for performance
        "streaming": False,
        "max_cc_strategies": 1,
        "max_cc_filegroups": 1,
        "max_files_per_request": 10
    }

    print("\n===== Chain REST Basic Test =====")
    print(f"Endpoint: {API}")
    print(f"Input Dir (env): {input_dir_env}")
    print(f"Input Dir (abs): {input_dir}")
    print(f"Output Dir (env): {output_dir_env}")
    print(f"Output Dir (abs): {output_dir}")
    print(f"Combo Name: {combo_name}")
    print(f"Chain Name: {chain_name}")
    print("Payload:")
    print(json.dumps(payload, indent=2))

    print(f"\n🔄 POST {API} ...")
    t0 = time.time()
    resp = requests.post(API, json=payload, timeout=300)
    dt = time.time() - t0
    print(f"HTTP {resp.status_code} in {dt:.2f}s")

    if resp.status_code != 200:
        print("Response Text:")
        print(resp.text)
        assert False, f"Unexpected status {resp.status_code}: {resp.text}"

    try:
        data = resp.json()
    except Exception:
        print("Non-JSON Response:")
        print(resp.text)
        assert False, "Response is not valid JSON"

    print("Response JSON:")
    print(json.dumps(data, indent=2))

    assert data.get("status") == "success", f"Unexpected response: {data}"
    print("✅ Chain REST basic test executed. Inspect server logs for per-file fallback messages.")

if __name__ == "__main__":
    test_chain_rest_basic()
