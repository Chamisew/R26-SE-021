"""
CSV Decimal Proportion Analyzer
Commit 2: feat: per-column decimal counting logic

Adds on top of commit 1:
  - decimal_counts dict  {field: int}
  - Row-by-row, cell-by-cell scan
  - Skip rules:
      • val is None or empty string  → skip
      • val in {"0","1","0.0","1.0"} → skip (binary/flag columns)
      • not parseable as float       → skip
      • float value is whole number  → skip
  - Increments decimal_counts[field] for every genuinely decimal cell
  - Prints per-column raw counts as a debug summary
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
    fieldnames = reader.fieldnames
    rows = list(reader)

total_rows = len(rows)


# =============================================================================
# DECIMAL COUNTING  ← NEW
# =============================================================================

# Initialise a counter for every column.
decimal_counts = {field: 0 for field in fieldnames}

for row in rows:
    for field, val in row.items():

        # ── Skip None or empty ──────────────────────────────────────
        if val is None:
            continue
        v = val.strip()
        if not v:
            continue

        # ── Skip explicit binary / flag values ──────────────────────
        # 0 and 1 (and their float reprs) appear in categorical/boolean
        # columns and must not inflate decimal counts.
        if v in ("0", "1", "0.0", "1.0"):
            continue

        # ── Skip non-numeric cells ──────────────────────────────────
        try:
            fv = float(v)
        except ValueError:
            continue

        # ── Count only genuinely fractional values ──────────────────
        if not fv.is_integer():
            decimal_counts[field] += 1


# Debug: show raw counts for every column that has at least one hit.
print(f"\nDecimal cell counts (out of {total_rows} rows):")
for field in fieldnames:
    count = decimal_counts[field]
    if count > 0:
        print(f"  {field}: {count}")

