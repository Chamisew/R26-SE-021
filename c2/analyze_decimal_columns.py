import csv, sys, os

csv_path = r"C:\\Users\\wimal\\Desktop\\c2\\c2\\Queue-Aware CPU Spike Analyzer\\research_framework\\final_research_dataset.csv"

if not os.path.isfile(csv_path):
    print(f"CSV file not found at {csv_path}")
    sys.exit(1)

with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    total_rows = 0
    decimal_counts = {field: 0 for field in fieldnames}
    for row in reader:
        total_rows += 1
        for field, val in row.items():
            if val is None:
                continue
            v = val.strip()
            if not v:
                continue
            # Exclude explicit 0 or 1 (including float representations)
            if v in ("0", "1", "0.0", "1.0"):
                continue
            try:
                fv = float(v)
            except ValueError:
                continue
            # Count if value has a fractional part
            if not fv.is_integer():
                decimal_counts[field] += 1

# Compute proportion of rows with decimal values for each column
proportions = []
for field in fieldnames:
    if total_rows == 0:
        prop = 0.0
    else:
        prop = decimal_counts[field] / total_rows
    proportions.append((field, prop))

# Sort descending by proportion
proportions.sort(key=lambda x: x[1], reverse=True)

print("Column decimal proportion (excluding 0/1):")
for field, prop in proportions:
    if prop > 0:
        print(f"{field}: {prop:.2%}")
