"""
Test script verifying the 4 classifier changes in pipeline.py:
1. Regex word-boundary keyword matching & specific failure_type per pattern.
2. Semantic layer with NORMAL_REFERENCE_CORPUS, margin filtering, and tune_semantic_params.
3. Sentence-embedding backend switch (sentence-transformers) & ablation study comparison.
4. Failure type validation: no FAILURE row has failure_type 'none'.
"""
import sys
import pandas as pd
import numpy as np
import pipeline

def run_tests():
    print("=" * 60)
    print("RUNNING CLASSIFIER IMPROVEMENTS VALIDATION")
    print("=" * 60)

    # -------------------------------------------------------------
    # Test 1: Layer 1 keyword & word-boundary patterns
    # -------------------------------------------------------------
    print("\n[TEST 1] Keyword word-boundary & multi-language memory errors")
    
    # 1.1 "ROOM service started" must NOT trigger OOM keyword (word boundary)
    lbl, lyr, ft = pipeline._layer1_keyword("ROOM service started")
    assert lbl is None, f"Expected None for 'ROOM service started', got {lbl}"
    print("  [PASS] 'ROOM service started' does not match OOM keyword (word-boundary verified)")

    # 1.2 "java.lang.OutOfMemoryError: Java heap space"
    lbl, lyr, ft = pipeline._layer1_keyword("java.lang.OutOfMemoryError: Java heap space")
    assert lbl == "FAILURE" and ft == "oom", f"Expected FAILURE/oom, got {lbl}/{ft}"
    print(f"  [PASS] 'java.lang.OutOfMemoryError: Java heap space' -> label={lbl}, failure_type={ft}")

    # 1.3 Node.js
    lbl, lyr, ft = pipeline._layer1_keyword("FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory")
    assert lbl == "FAILURE" and ft == "oom", f"Expected FAILURE/oom for Node, got {lbl}/{ft}"
    print(f"  [PASS] Node heap limit -> label={lbl}, failure_type={ft}")

    # 1.4 Python
    lbl, lyr, ft = pipeline._layer1_keyword("MemoryError: unable to allocate array with shape")
    assert lbl == "FAILURE" and ft == "oom", f"Expected FAILURE/oom for Python, got {lbl}/{ft}"
    print(f"  [PASS] Python MemoryError -> label={lbl}, failure_type={ft}")

    # 1.5 Go
    lbl, lyr, ft = pipeline._layer1_keyword("fatal error: runtime: out of memory")
    assert lbl == "FAILURE" and ft == "oom", f"Expected FAILURE/oom for Go, got {lbl}/{ft}"
    print(f"  [PASS] Go runtime out of memory -> label={lbl}, failure_type={ft}")

    # 1.6 PHP
    lbl, lyr, ft = pipeline._layer1_keyword("PHP Fatal error: Allowed memory size of 134217728 bytes exhausted")
    assert lbl == "FAILURE" and ft == "oom", f"Expected FAILURE/oom for PHP, got {lbl}/{ft}"
    print(f"  [PASS] PHP Allowed memory size exhausted -> label={lbl}, failure_type={ft}")

    # 1.7 Ruby
    lbl, lyr, ft = pipeline._layer1_keyword("NoMemoryError: failed to allocate memory")
    assert lbl == "FAILURE" and ft == "oom", f"Expected FAILURE/oom for Ruby, got {lbl}/{ft}"
    print(f"  [PASS] Ruby NoMemoryError -> label={lbl}, failure_type={ft}")

    # 1.8 Containers
    for container_msg in ["Container OOMKilled by Linux kernel", "Process terminated with exit code 137", "Killed process 4123 (python)"]:
        lbl, lyr, ft = pipeline._layer1_keyword(container_msg)
        assert lbl == "FAILURE" and ft == "oom", f"Expected FAILURE/oom for '{container_msg}', got {lbl}/{ft}"
    print("  [PASS] Container patterns (OOMKilled, exit code 137, Killed process) -> failure_type=oom")

    # 1.9 GC overhead vs stable
    lbl, lyr, ft = pipeline._layer1_keyword("java.lang.OutOfMemoryError: GC overhead limit exceeded")
    assert lbl == "FAILURE" and ft == "gc_overhead", f"Expected FAILURE/gc_overhead, got {lbl}/{ft}"
    print(f"  [PASS] GC overhead exceeded -> label={lbl}, failure_type={ft}")

    lbl, lyr, ft = pipeline._layer1_keyword("INFO: garbage collection overhead limit stable")
    assert lbl is None, f"Expected None in Layer 1 for stable GC line, got {lbl}"
    print("  [PASS] 'INFO: garbage collection overhead limit stable' does not match error keywords in Layer 1")

    # -------------------------------------------------------------
    # Test 2: Semantic layer dual-corpus margin & tune_semantic_params
    # -------------------------------------------------------------
    print("\n[TEST 2] Semantic dual-corpus margin check & parameter tuning")

    # 2.1 "INFO: garbage collection overhead limit stable" in Layer 2 TF-IDF
    lbl, lyr, ft = pipeline._layer2_tfidf("INFO: garbage collection overhead limit stable", threshold=0.12, margin=0.05)
    assert lbl == "NORMAL", f"Expected NORMAL for 'INFO: garbage collection overhead limit stable', got {lbl}"
    print(f"  [PASS] 'INFO: garbage collection overhead limit stable' in Layer 2 TF-IDF -> label={lbl} (margin suppressed)")

    # 2.2 "ROOM service started" in Layer 2 TF-IDF
    lbl, lyr, ft = pipeline._layer2_tfidf("ROOM service started", threshold=0.12, margin=0.05)
    assert lbl == "NORMAL", f"Expected NORMAL for 'ROOM service started', got {lbl}"
    print(f"  [PASS] 'ROOM service started' in Layer 2 TF-IDF -> label={lbl}")

    # 2.3 tune_semantic_params
    tune_df = pd.DataFrame({
        "log_template": [
            "health check passed service status ok",
            "routine garbage collection overhead limit stable",
            "heap memory consumption rising steadily without release",
            "out of memory error oom condition triggered",
            "request completed successfully 200 ok",
        ] * 4,
        "ground_truth_label": [
            "NORMAL",
            "NORMAL",
            "FAILURE",
            "FAILURE",
            "NORMAL",
        ] * 4,
    })
    best_thresh, best_margin = pipeline.tune_semantic_params(tune_df)
    assert best_thresh > 0 and best_margin >= 0, "tune_semantic_params should return positive threshold and margin"
    print(f"  [PASS] tune_semantic_params returned: threshold={best_thresh:.4f}, margin={best_margin:.4f}")

    # -------------------------------------------------------------
    # Test 3: Embedding Backend & Ablation Study
    # -------------------------------------------------------------
    print("\n[TEST 3] Sentence-embedding backend & ablation study")
    
    # Test Layer 2 with embedding
    lbl_emb, lyr_emb, ft_emb = pipeline._layer2_embedding("INFO: garbage collection overhead limit stable", threshold=0.35, margin=0.05)
    assert lbl_emb == "NORMAL", f"Expected NORMAL with embedding for stable GC, got {lbl_emb}"
    print(f"  [PASS] 'INFO: garbage collection overhead limit stable' in Layer 2 Embedding -> label={lbl_emb}")

    lbl_emb_fail, lyr_emb_fail, ft_emb_fail = pipeline._layer2_embedding("process terminated after exceeding memory limit heap exhausted", threshold=0.35, margin=0.05)
    assert lbl_emb_fail == "FAILURE", f"Expected FAILURE with embedding for memory exhaustion, got {lbl_emb_fail}"
    print(f"  [PASS] Semantic memory exhaustion in Layer 2 Embedding -> label={lbl_emb_fail}, failure_type={ft_emb_fail}")

    # Test ablation study comparison
    ablation_df = pd.DataFrame({
        "log_template": [
            "java.lang.OutOfMemoryError: Java heap space",
            "routine garbage collection overhead limit stable",
            "ROOM service started",
            "memory leak heap allocation increasing",
            "http request completed successfully status 200",
        ],
        "ground_truth_label": ["FAILURE", "NORMAL", "NORMAL", "FAILURE", "NORMAL"],
        "hybrid_label": ["FAILURE", "NORMAL", "NORMAL", "FAILURE", "NORMAL"],
        "detection_layer": ["keyword", "normal", "normal", "tfidf_semantic", "normal"],
    })
    pipeline.run_ablation_study(ablation_df)
    print("  [PASS] run_ablation_study executed successfully with both TF-IDF and Embedding")

    # -------------------------------------------------------------
    # Test 4: End-to-end stage3_hybrid_classifier & failure_type guarantees
    # -------------------------------------------------------------
    print("\n[TEST 4] End-to-end stage3_hybrid_classifier & failure_type check")
    
    test_rows = [
        # 1. Stable GC log (must be NORMAL)
        {"service_name": "svc1", "timestamp": "2026-01-01 00:00:00", "log_template": "INFO: garbage collection overhead limit stable", "ram_percent": 50.0, "cpu_percent": 20.0, "heap_mb_used": 1000.0, "gc_count": 1},
        # 2. ROOM service started (must be NORMAL)
        {"service_name": "svc1", "timestamp": "2026-01-01 00:00:05", "log_template": "ROOM service started", "ram_percent": 50.0, "cpu_percent": 20.0, "heap_mb_used": 1000.0, "gc_count": 1},
        # 3. Java OOM error (must be FAILURE, oom)
        {"service_name": "svc1", "timestamp": "2026-01-01 00:00:10", "log_template": "java.lang.OutOfMemoryError: Java heap space", "ram_percent": 50.0, "cpu_percent": 20.0, "heap_mb_used": 1000.0, "gc_count": 1},
        # 4. Semantic failure (must be FAILURE, inferred from reference corpus)
        {"service_name": "svc1", "timestamp": "2026-01-01 00:00:15", "log_template": "memory leak heap allocation increasing rapidly", "ram_percent": 50.0, "cpu_percent": 20.0, "heap_mb_used": 1000.0, "gc_count": 1},
        # 5. Metric-only failure (high RAM & CPU, normal log)
        {"service_name": "svc1", "timestamp": "2026-01-01 00:00:20", "log_template": "periodic health check passed ok", "ram_percent": 95.0, "cpu_percent": 90.0, "heap_mb_used": 3500.0, "gc_count": 10},
    ]
    df_in = pd.DataFrame(test_rows)
    cfg = dict(pipeline.CONFIG)
    cfg["metric_score_threshold"] = 2
    df_out = pipeline.stage3_hybrid_classifier(df_in, cfg)

    row_gc = df_out.iloc[0]
    row_room = df_out.iloc[1]
    row_java = df_out.iloc[2]
    row_semantic = df_out.iloc[3]
    row_metric = df_out.iloc[4]

    # Verification: "INFO: garbage collection overhead limit stable" is NORMAL
    assert row_gc["raw_hybrid_label"] == "NORMAL", f"Expected NORMAL for GC stable, got {row_gc['raw_hybrid_label']}"
    print(f"  [PASS] 'INFO: garbage collection overhead limit stable' raw_hybrid_label={row_gc['raw_hybrid_label']}")

    # Verification: "ROOM service started" is NORMAL
    assert row_room["raw_hybrid_label"] == "NORMAL", f"Expected NORMAL for ROOM service, got {row_room['raw_hybrid_label']}"
    print(f"  [PASS] 'ROOM service started' raw_hybrid_label={row_room['raw_hybrid_label']}")

    # Verification: "java.lang.OutOfMemoryError: Java heap space" is FAILURE with failure_type oom
    assert row_java["raw_hybrid_label"] == "FAILURE" and row_java["failure_type"] == "oom", \
        f"Expected FAILURE/oom for Java, got {row_java['raw_hybrid_label']}/{row_java['failure_type']}"
    print(f"  [PASS] 'java.lang.OutOfMemoryError: Java heap space' raw_hybrid_label={row_java['raw_hybrid_label']}, failure_type={row_java['failure_type']}")

    # Verification: Semantic failure has inferred failure_type (not none)
    assert row_semantic["raw_hybrid_label"] == "FAILURE" and row_semantic["failure_type"] != "none", \
        f"Expected FAILURE/non-none for semantic, got {row_semantic['raw_hybrid_label']}/{row_semantic['failure_type']}"
    print(f"  [PASS] Semantic failure raw_hybrid_label={row_semantic['raw_hybrid_label']}, failure_type={row_semantic['failure_type']}")

    # Verification: Metric-only failure has failure_type metric_only
    assert row_metric["raw_hybrid_label"] == "FAILURE" and row_metric["failure_type"] == "metric_only", \
        f"Expected FAILURE/metric_only for metric spike, got {row_metric['raw_hybrid_label']}/{row_metric['failure_type']}"
    print(f"  [PASS] Metric-only failure raw_hybrid_label={row_metric['raw_hybrid_label']}, failure_type={row_metric['failure_type']}")

    # Verification: NO FAILURE row has failure_type "none"
    failures_raw = df_out[df_out["raw_hybrid_label"] == "FAILURE"]
    assert not (failures_raw["failure_type"] == "none").any(), "Found raw FAILURE row with failure_type 'none'"
    assert not (failures_raw["failure_type"] == "").any(), "Found raw FAILURE row with empty failure_type"
    print("  [PASS] Confirmed: No raw FAILURE row has failure_type 'none' or empty!")

    # -------------------------------------------------------------
    # Test 5: Component 3 dataset schema verification (Exact 20 columns)
    # -------------------------------------------------------------
    print("\n[TEST 5] Component 3 ML dataset schema preservation (Exact 20 columns)")
    c3_req_cols = [
        "timestamp", "project_id", "service_name", "window_id", "label", "failure_type",
        "incident_phase_1", "incident_phase_2", "ram_mean", "ram_max", "ram_std", "ram_std_trend",
        "ram_percent", "memory_change_5min", "memory_change_10min", "heap_mb_used", "heap_max",
        "heap_rate", "gc_count", "gc_spike_count"
    ]
    assert pipeline.FINAL_ML_COLS == c3_req_cols, f"FINAL_ML_COLS mismatch: expected {c3_req_cols}, got {pipeline.FINAL_ML_COLS}"
    print(f"  [PASS] FINAL_ML_COLS matches the exact 20 Component 3 columns: {len(pipeline.FINAL_ML_COLS)} columns")

    # Check on-disk csv
    df_csv = pd.read_csv("output/ml_ready_dataset.csv")
    assert list(df_csv.columns) == c3_req_cols, f"On-disk CSV mismatch: {list(df_csv.columns)}"
    print(f"  [PASS] output/ml_ready_dataset.csv contains exactly the 20 required columns: {len(df_csv.columns)} columns")

    # -------------------------------------------------------------
    # Test 6: Missing log_message skips keyword & semantic, uses metric layer only
    # -------------------------------------------------------------
    print("\n[TEST 6] Real logs only: missing log_message skips keyword & semantic classification")
    row_no_log_normal = pd.DataFrame([{
        "service_name": "s1", "timestamp": "2026-01-01 00:00:00",
        "log_message": "", "log_template": "<no_log>",
        "ram_percent": 50.0, "cpu_percent": 20.0, "heap_mb_used": 1000.0, "gc_count": 1
    }])
    out_no_log = pipeline.stage3_hybrid_classifier(row_no_log_normal, cfg)
    assert out_no_log.iloc[0]["raw_hybrid_label"] == "NORMAL" and out_no_log.iloc[0]["detection_layer"] == "none"
    print("  [PASS] Row with empty log_message and normal metrics -> label=NORMAL, layer=none")

    row_no_log_fail = pd.DataFrame([{
        "service_name": "s1", "timestamp": "2026-01-01 00:00:00",
        "log_message": "", "log_template": "<no_log>",
        "ram_percent": 95.0, "cpu_percent": 90.0, "heap_mb_used": 3500.0, "gc_count": 10
    }])
    out_no_log_fail = pipeline.stage3_hybrid_classifier(row_no_log_fail, cfg)
    assert out_no_log_fail.iloc[0]["raw_hybrid_label"] == "FAILURE" and out_no_log_fail.iloc[0]["detection_layer"] == "metric_fusion"
    assert out_no_log_fail.iloc[0]["failure_type"] == "metric_only"
    print("  [PASS] Row with empty log_message and high metrics -> classified by metric layer only (metric_fusion / metric_only)")

    # -------------------------------------------------------------
    # Test 7: services_config.json dynamic updates without hardcoding
    # -------------------------------------------------------------
    print("\n[TEST 7] Dynamic services_config.json update and discovery")
    import app as fastapi_app
    test_svc = fastapi_app.AzureServiceInput(
        name="test-dynamic-svc",
        base_url="https://test-dynamic-svc.canadacentral.azurecontainerapps.io",
        enabled=True
    )
    add_res = fastapi_app.api_add_azure_service(test_svc)
    assert add_res["status"] == "saved"
    print(f"  [PASS] Added service dynamically via API: {add_res['service']['name']}")

    discovered = pipeline.load_azure_services()
    assert "test-dynamic-svc" in discovered
    print(f"  [PASS] pipeline discovered new service from services_config.json: {discovered['test-dynamic-svc']}")

    del_res = fastapi_app.api_delete_azure_service("test-dynamic-svc")
    assert del_res["status"] == "deleted"
    print("  [PASS] Deleted service dynamically via API without hardcoding")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
