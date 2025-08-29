
CHART_TITLE = "chart_title"
DECIMAL = "decimal_point"

# Configuration for chart layout
CHARTS_PER_ROW = 4  # Number of charts to display per row on desktop
VAL_LEGEND_RIGHT_BIAS = 10  # Pixel offset to move value legends to the right
MIN_BAR_HEIGHT = 3  # Minimum bar height in pixels for better visibility of low values
MAX_BAR_HEIGHT = 300  # Maximum bar height in pixels before scaling down all bars in the chart

# Import monitoring configuration
try:
    from .monitoring_config import (
        REAL_TIME_MONITORING, 
        UPDATE_FREQUENCY_SECONDS, 
        FILE_WATCH_ENABLED
    )
except ImportError:
    # Fallback values if monitoring config is not available
    REAL_TIME_MONITORING = True
    UPDATE_FREQUENCY_SECONDS = 1
    FILE_WATCH_ENABLED = True

# Configuration for data sources
# Directory containing JSON data files. Point this at where your JSON summaries live,
# relative to the repository root (resolved from the monitor package root).
# For the current repo, we have a consolidated JSON at Ultra_Arena_Test_Example/modular_results.json
# JSON_DATA_DIR = "../Ultra_Arena_Main_Restful_Test/test_fixtures/default_fixture/output_files/results/results_250819_152301_8ce01d31-f39c-42ea-b1da-53b4dbdcf323/json"
JSON_DATA_DIR = "../Ultra_Arena_Main_Restful_Test/test_fixtures/br_fixture/output_files/results_250829_010752_0303c87f-2b6a-4633-ab61-1d934db7e6de/json"
PARENT_JSON_DATA_DIR = "../Ultra_Arena_Main_Restful_Test/test_fixtures/br_fixture/output_files"
# JSON_DATA_DIR = "/Users/jameswang9311/Documents/My_Ultra_Arena_Fork/ultra-arena-frk_bk_250815_200f/Ultra_Arena_Main_Restful/run_profiles/default_profile_restful/output/results/combo/combo_test_8_strategies_13f/json"
# JSON_DATA_DIR = "../Ultra_Arena_Main/run_profiles/default_profile/output/results/combo/combo_test_8_strategies_13f/json"
# Track Latest Configuration
# When enabled, automatically tracks the most recently modified results directory
# When disabled, tracks the specific directory defined in JSON_DATA_DIR
TRACK_LATEST_ENABLED = True  # Default to ON as requested
CURRENT_JSON_DATA_DIR = None  # Dynamic directory path (set at runtime)

# NEW DIRECTORY STRUCTURE (as of latest update):
# Results are now stored in timestamped directories: results_YYMMDD_HHmmss_[request_id]/
# Example: results_241215_143022_550e8400-e29b-41d4-a716-446655440000/
# Each directory contains:
#   - combo_meta.json (request metadata and strategy info)
#   - csv/ (CSV files)
#   - json/ (JSON files for each strategy)
# To monitor the latest results, point to a specific results directory:
# JSON_DATA_DIR = "../Ultra_Arena_Main_Restful_Test/test_fixtures/default_fixture/output_files/results_241215_143022_550e8400-e29b-41d4-a716-446655440000/json"

chart_config_all = {
    "chart_config_1": {
        "comparing_fields" : {
            "retry_stats": {
                "percentage_files_had_retry" : {CHART_TITLE : "Percent of Files Retried", DECIMAL: 2, "ORDER": 10}
            },
            "overall_stats": {
                "total_files": {CHART_TITLE : "Total Files Processed", "ORDER": 1},
                "total_wall_time_in_sec": {CHART_TITLE : "Total Processing Time In Seconds", "ORDER": 4},
                "total_actual_tokens": {CHART_TITLE : "All Tokens Spent", "ORDER": 5}
            },
            "overall_cost": {
                # "total_prompt_token_cost":  {CHART_TITLE : "Total Cost for Prompt Tokens ($USD)"},
                # "total_candidate_token_cost":  {CHART_TITLE : "Total Cost for Candidate Tokens ($USD)"},
                # "total_other_token_cost":  {CHART_TITLE : "Total Cost for Other Token ($USD)"},
                "total_token_cost":  {CHART_TITLE : "Total Cost for All Tokens ($USD)", DECIMAL: 6, "ORDER": 6}
            },
            "benchmark_comparison": {
                # "total_unmatched_fields": {CHART_TITLE : "Total Incorrect Data Fields Count"},
                # "total_unmatched_files": {CHART_TITLE : "Total Incorrect Extract File Count"},
                "invalid_fields_percent": {CHART_TITLE : "Percentage of Incorrectly Extracted Fields", DECIMAL: 2, "ORDER": 2},
                "invalid_files_percent": {CHART_TITLE : "Percentage of Incorrectly Processed Files", DECIMAL: 2, "ORDER": 3}
            }
        }
    }

}