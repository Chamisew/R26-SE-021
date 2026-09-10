"""
=============================================================================
pipeline.py
Stack-Agnostic Log Parsing and Hybrid Feature Extraction Pipeline
for Memory Failure Detection

Core Responsibilities:
1. Data Ingestion & Normalization: Ingest telemetry and logs from sidecars or
   CSV inputs; handle missing data, scheme validation, and duplicate removal.
2. Log Parsing: Process logs via a single global Drain3 miner to extract templates
   and cluster IDs regardless of service stack.
3. Hybrid Classification: Execute a three-layer detection engine (Layer 1: Regex
   Keyword Matching, Layer 2: TF-IDF / Embedding Semantic Similarity,
   Layer 3: Metric Fusion) with debouncing and confidence scoring.
4. Feature Engineering: Perform per-service multi-scale sliding-window aggregation
   to capture RAM, heap, CPU, and GC trends.
5. Ground-Truth Alignment & Export: Map ground-truth failure windows, label
   pre-failure/failure incident phases, and export ML-ready datasets.
=============================================================================
"""