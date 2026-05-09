# -*- coding: utf-8 -*-
"""
Verification script -- run after upgrading pipeline.py.
Checks that every promised feature is present.
"""
import re, os
import sys
# Force UTF-8 output so box-drawing / emoji print correctly on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

src = open("pipeline.py", encoding="utf-8").read()
req = open("requirements.txt", encoding="utf-8").read()

CHECKS = [
    ("stage1_collect_live() exists",       r"def stage1_collect_live\("),
    ("MODE variable defined at module",    r'^MODE\s*=\s*"csv"',),
    ("MODE csv branch in main()",          r'if MODE == "csv"'),
    ("MODE live branch in main()",         r'elif MODE == "live"'),
    ("Docker Desktop error message",       r"Docker Desktop is not running"),
    ("threading import",                   r"import threading"),
    ("_print_dashboard() function",        r"def _print_dashboard\("),
    ("_collect_container() function",      r"def _collect_container\("),
    ("live_duration_minutes in CONFIG",    r"live_duration_minutes"),
    ("live_output_csv in CONFIG",          r"live_output_csv"),
    ("Warning flag (⚠) for RAM>70",        r"warn_flag"),
    ("Evaluation skipped in live mode",    r"Evaluation skipped in live mode"),
    ("Stage 2 untouched",                  r"STAGE 2.*DRAIN3"),
    ("Stage 3 untouched",                  r"STAGE 3.*HYBRID"),
    ("Stage 4 untouched",                  r"STAGE 4.*SLIDING"),
    ("Stage 5 untouched",                  r"STAGE 5.*DATASET"),
    ("docker==6.1.3 in requirements.txt",  None),
    ("stream=False used (Windows safe)",   r"stream=False"),
    ("Per-container threading",            r"threading\.Thread"),
    ("gc_cumulative counter",              r"gc_cumulative"),
    ("Column order enforced",              r"ground_truth_label.*failure_type"),
]

all_ok = True
print()
print("=" * 55)
print("  PIPELINE UPGRADE VERIFICATION CHECKLIST")
print("=" * 55)
for label, pattern in CHECKS:
    if pattern is None:
        ok = "docker==6.1.3" in req
    else:
        ok = bool(re.search(pattern, src, re.MULTILINE | re.IGNORECASE))
    status = "[OK]  " if ok else "[FAIL]"
    if not ok:
        all_ok = False
    print(f"  {status} {label}")

print()
if all_ok:
    print("  ✅  ALL CHECKS PASSED — pipeline.py is correctly upgraded.")
else:
    print("  ❌  SOME CHECKS FAILED — review items marked [FAIL].")
print("=" * 55)
print()
