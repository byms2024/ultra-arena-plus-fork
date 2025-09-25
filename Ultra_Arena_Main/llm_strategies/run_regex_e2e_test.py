# save as: run_regex_e2e_test.py
import sys
from pathlib import Path
import pandas as pd

# Point this to your repo root (so imports work)
REPO_ROOT = Path(r"C:\Users\alexandre.carrer\Desktop\ultra-arena-plus-fork")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Ultra_Arena_Main.llm_strategies.regex_strategy import (
    RegexPreProcessingStrategy,
    RegexProcessingStrategy,
    Answers,
)

def main():
    input_dir = Path(r"C:\Users\alexandre.carrer\Desktop\ultra-arena-plus-fork\Ultra_Arena_Main_Restful_Test\test_fixtures\br_fixture\input_files\1_file")
    file_paths = sorted(str(p) for p in input_dir.glob("*.pdf"))
    if not file_paths:
        print(f"No PDFs found in: {input_dir}")
        return

    print("Testing with files:")
    for p in file_paths:
        print(f" - {p}")


    # Preprocess (extract text, classify, build Answers per file)
    preprocessor = RegexPreProcessingStrategy({})
    # Create a list of Answers, one per file, with different or same values as needed
    manual_answers_list = [
        Answers(
            claim_no="BYDAMEBR0125WCN250800026_01",
            vin="LGXCE4CC6S0053853",
            service_price="30,10",
            parts_price=None,
            cnpj="54168855000155"
        ),
        Answers(
            claim_no="BYDAMEBR0015WCN241200032_01",
            vin="LGXCE4CC7S0023860",
            service_price=None,
            parts_price="2.465,73",
            cnpj="46621491000270"
        )
    ]
    # Preprocess each file individually, collecting PreprocessedData for each
    pre_list = [
        preprocessor.preprocess_filepaths([file_path], manual_answers=manual_answers)
        for file_path, manual_answers in zip(file_paths, manual_answers_list)
    ]

    # Process (compare extracted text against targets; avoid LLM init)
    all_results = []
    for i in range(len(pre_list)):
        processor = RegexProcessingStrategy(config={"fallback_llm": False}, streaming=False, answers=pre_list[i].answers)
        results, agg_stats, status = processor.process_file_group(
            file_group=[file_paths[i]],
            group_index=i,
            user_prompt="",
        )
        # Collect results for each file
        all_results.extend([r[1] for r in results])

    df = pd.DataFrame(all_results)

    print("\nResults:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()