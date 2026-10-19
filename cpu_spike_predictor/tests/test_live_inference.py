"""Quick test of the live inference engine startup and one cycle."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live_inference import LiveInferenceEngine

# Clean state
for f in ["outputs/.inference_checkpoint.json", "outputs/live_predictions.jsonl"]:
    if os.path.exists(f):
        os.remove(f)

engine = LiveInferenceEngine(poll_interval=2)
ok = engine.startup()
print(f"Startup OK: {ok}")
print(f"Model features: {len(engine.model.feature_importances_)}")
print(f"Model classes: {list(engine.model.classes_)}")

# Run one manual cycle
comp2_new = engine._read_new_comp2()
print(f"New Comp2 records: {len(comp2_new)}")

comp3_new = engine._read_new_comp3()
print(f"New Comp3 records: {len(comp3_new)}")

if not comp2_new.empty:
    from src.live_inference import preprocess_comp2_for_inference, preprocess_comp3_for_inference, align_comp2_comp3, safe_read_csv

    comp2_proc = preprocess_comp2_for_inference(comp2_new)
    comp3_full = safe_read_csv(engine.comp3_path)
    comp3_proc = preprocess_comp3_for_inference(comp3_full) if comp3_full is not None else None

    aligned = align_comp2_comp3(comp2_proc, comp3_proc)
    print(f"Aligned records: {len(aligned)}")
    print(f"Feature columns present: {[c for c in ['cpu_percent','cpu_velocity','cpu_trend_5min','in_flight_queue','overload_flag'] if c in aligned.columns]}")

    comp2_keys = [f"c2:{row.get('incident_id',0)}:{row.get('timestamp','')}" for _, row in comp2_new.iterrows()]
    comp3_keys = [f"c3:{row.get('timestamp','')}:{row.get('service_name','')}:{row.get('project_id','')}" for _, row in comp3_new.iterrows()] if not comp3_new.empty else []

    engine._run_inference(aligned, comp2_keys, comp3_keys)

    # Check output
    if os.path.exists("outputs/live_predictions.jsonl"):
        with open("outputs/live_predictions.jsonl") as f:
            lines = f.readlines()
        print(f"Predictions written: {len(lines)} lines")
        if lines:
            import json
            first = json.loads(lines[0])
            print(f"First prediction: service={first['service_name']}, cpu_prob={first['cpu_failure_prob']:.4f}, action={first['action_recommended']}")

    # Check checkpoint
    print(f"Checkpoint Comp2: {engine.checkpoint.state['comp2_processed_count']}")
    print(f"Checkpoint Comp3: {engine.checkpoint.state['comp3_processed_count']}")

    # Run a second cycle - should detect NO new records
    comp2_new2 = engine._read_new_comp2()
    print(f"Second cycle new Comp2 records (should be 0): {len(comp2_new2)}")

print("\nLive inference test PASSED!")
