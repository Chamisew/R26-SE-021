"""
CSV Decimal Proportion Analyzer
Commit 1: feat: file validation and CSV loading scaffold

Adds:
  - CSV_PATH constant
  - File existence guard (sys.exit on missing file)
  - Open CSV with DictReader
  - Collect fieldnames and all rows into memory
  - Print basic file stats (row count, column count)
"""

import csv
import os
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_PATH = r"C:\Users\wimal\Desktop\c2\c2\Queue-Aware CPU Spike Analyzer\research_framework\final_research_dataset.csv"


# =============================================================================
# FILE VALIDATION
# =============================================================================

if not os.path.isfile(CSV_PATH):
    print(f"CSV file not found at {CSV_PATH}")
    sys.exit(1)


# =============================================================================
# CSV LOADING
# =============================================================================

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    fieldnames = reader.fieldnames       # list[str] — column headers
    rows = list(reader)                  # load all rows for reuse

total_rows = len(rows)

print(f"Loaded CSV: {total_rows} rows, {len(fieldnames)} columns")
print(f"Columns: {fieldnames}")

