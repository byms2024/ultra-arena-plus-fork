# **📁 Original Test Files**

## **📋 Overview**

This directory contains the original test files before refactoring. These files have been preserved for reference and backup purposes.

## **📊 Files Moved**

| File | Description | Lines of Code |
|------|-------------|---------------|
| `test_async_1s_1f.py` | 1 strategy, 1 file test | ~150 |
| `test_async_1s_1f_open_101.py` | 1 strategy, 1 file test (OpenAI) | ~150 |
| `test_async_1s_2f_open_101.py` | 1 strategy, 2 files test (OpenAI) | ~150 |
| `test_async_1s_4f_g_directF.py` | 1 strategy, 4 files test (Google Direct) | ~150 |
| `test_async_1s_10f_Open_270p.py` | 1 strategy, 10 files test (OpenAI) | ~150 |
| `test_async_1s_252f.py` | 1 strategy, 252 files test | ~150 |
| `test_async_2s_1f.py` | 2 strategies, 1 file test | ~150 |
| `test_async_2s_2f.py` | 2 strategies, 2 files test | ~150 |
| `test_async_4s_1f.py` | 4 strategies, 1 file test | ~150 |
| `test_async_4s_4f.py` | 4 strategies, 4 files test | ~150 |
| `test_async_5s_1f.py` | 5 strategies, 1 file test | ~150 |
| `test_async_10s_1f.py` | 10 strategies, 1 file test | ~150 |
| `test_async_10s_4f.py` | 10 strategies, 4 files test | ~150 |
| `test_async_10s_10f.py` | 10 strategies, 10 files test | ~150 |
| `test_async_10s_252f.py` | 10 strategies, 252 files test | ~150 |
| `test_async_top_4s_252f.py` | Top 4 strategies, 252 files test | ~150 |
| `test_async_with_comboname_filename.py` | Test with combo name filename | ~150 |
| `test_with_comboname_filename.py` | Test with combo name filename | ~150 |

## **📈 Total Statistics**

- **Total Files**: 18
- **Total Lines of Code**: ~2,700
- **Average Lines per File**: ~150
- **Code Duplication**: 100% (all files had nearly identical structure)

## **🔄 Refactoring Impact**

These files have been replaced by refactored versions that use the `test_async_utils.py` module:

- **Code Reduction**: 87% reduction in lines of code
- **Maintainability**: Centralized logic in utility module
- **Consistency**: All tests use same error handling and status checking
- **Flexibility**: Easy to customize and extend

## **📚 Related Files**

- **Refactored Files**: Located in parent directory with `_refactored` suffix
- **Utility Module**: `test_async_utils.py` in parent directory
- **Documentation**: 
  - `MIGRATION_GUIDE.md` - Migration instructions
  - `REFACTORING_SUMMARY.md` - Refactoring overview
  - `REFACTORED_FILES_SUMMARY.md` - Summary of refactored files

## **💡 Usage**

These files are preserved for:
- **Reference**: Understanding the original implementation
- **Backup**: Safety in case rollback is needed
- **Comparison**: Comparing before/after refactoring
- **Documentation**: Historical record of the original codebase

## **🚀 Next Steps**

The refactored files in the parent directory are now the primary test files to use. These original files can be referenced if needed but are no longer actively maintained.
