import argparse
import logging
import sys
import os
import uvicorn
from src.ingestion import (
    load_cpu_features,
    load_memory_predictions,
    merge_datasets,
    DEFAULT_COMP2_CSV,
    DEFAULT_COMP3_CSV,
)
from src.validation import run_lopo_cross_validation, save_validation_results
from src.mtta import analyze_mtta_strategies
from src.visualization import create_mtta_visualizations
from src.model import train_model, save_model, FEATURE_COLS, TARGET_COL

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("cpu_predictor_cli")

def run_pipeline(cpu_path: str, mem_path: str, lead_time: int, output_dir: str):
    """
    Runs ingestion -> LOPO validation -> MTTA computation -> Visualization.
    Data flow:
      - Component 2 (CPU/queue) loaded separately via load_cpu_features
      - Component 3 (Memory) loaded separately via load_memory_predictions
      - Integration layer (merge_datasets) combines them for decision analysis only
      - Model is trained on Comp2 CPU features only; Comp3 memory is for downstream MTTA strategies
    """
    logger.info("Starting CPU Spike Prediction validation pipeline...")
    logger.info(f"  Comp2 (CPU)   path: {cpu_path}")
    logger.info(f"  Comp3 (Memory) path: {mem_path}")

    logger.info("Step 1: Separate Component Ingestion & Preprocessing...")
    cpu_df = load_cpu_features(cpu_path, lead_time_minutes=lead_time)
    logger.info(f"  Comp2 loaded: {len(cpu_df)} rows, CPU+queue+system+incident columns only")
    mem_df = load_memory_predictions(mem_path)
    logger.info(f"  Comp3 loaded: {len(mem_df)} rows, memory_prob+alert+pred_label columns only")

    logger.info("Step 1b: Integration Layer (merge for downstream analysis)...")
    merged_df = merge_datasets(cpu_df, mem_df)

    from src.ingestion import save_aligned_telemetry
    save_aligned_telemetry(merged_df, "data/processed")

    logger.info("Step 2: Leave-One-Project-Out Validation (Comp2 CPU features only)...")
    lopo_results, all_preds = run_lopo_cross_validation(merged_df)
    save_validation_results(lopo_results, all_preds, output_dir)

    logger.info("Step 3: MTTA Calculation (evaluates joint CPU+Memory strategies)...")
    predictions_path = os.path.join(output_dir, "cpu_predictions.csv")
    analyze_mtta_strategies(predictions_path, output_dir)

    logger.info("Step 4: Generating plots...")
    mtta_results_path = os.path.join(output_dir, "cpu_mtta_results.csv")
    chart_path = os.path.join(output_dir, "cpu_mtta_chart.png")
    create_mtta_visualizations(mtta_results_path, chart_path)

    logger.info("Pipeline validation complete! All output files are generated in the outputs/ directory.")


def train_production_model(cpu_path: str, mem_path: str, lead_time: int, model_output_path: str):
    """
    Trains a production model. The model trains on Component 2 CPU/queue features ONLY.
    Component 3 memory data is NOT a training feature (avoids leakage from Comp3 predictions).
    Comp3 memory signals are used only at inference time for decision augmentation.
    """
    logger.info("Training production-ready model (Comp2 CPU features only)...")
    cpu_df = load_cpu_features(cpu_path, lead_time_minutes=lead_time)
    mem_df = load_memory_predictions(mem_path)
    merged_df = merge_datasets(cpu_df, mem_df)

    from src.ingestion import save_aligned_telemetry
    save_aligned_telemetry(merged_df, "data/processed")

    from src.model import get_features_and_target
    X, y = get_features_and_target(merged_df, TARGET_COL)

    model = train_model(X, y)
    save_model(model, model_output_path)
    logger.info(f"Production model saved to {model_output_path} "
                f"(trained on {X.shape[1]} Comp2 CPU features, {X.shape[0]} samples)")

def start_server(host: str, port: int):
    """
    Launches the FastAPI server.
    """
    logger.info(f"Launching FastAPI Server on {host}:{port}...")
    uvicorn.run("src.api:app", host=host, port=port, reload=False)

def start_live_inference(
    comp2_path: str, comp3_path: str, model_path: str,
    poll_interval: float, output_path: str, status_path: str
):
    """
    Starts the real-time CSV consumer & inference engine.
    Reads live-updating CSVs from Component 2 and Component 3,
    applies the pre-trained model, and records predictions.
    Does NOT retrain the model. Does NOT generate any data.
    """
    from src.live_inference import LiveInferenceEngine
    engine = LiveInferenceEngine(
        comp2_path=comp2_path,
        comp3_path=comp3_path,
        model_path=model_path,
        output_path=output_path,
        status_path=status_path,
        poll_interval=poll_interval,
    )
    if not engine.startup():
        logger.critical("Live inference startup failed. Exiting.")
        sys.exit(1)
    engine.run()


def main():
    parser = argparse.ArgumentParser(
        description="CPU Spike Prediction Engine (Component 4) Command Line Interface."
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # 1. Pipeline parser
    pipeline_parser = subparsers.add_parser("validate", help="Run full cross-validation and MTTA pipeline")
    pipeline_parser.add_argument("--cpu-data", default=DEFAULT_COMP2_CSV, help="Path to Component 2 CPU features CSV (final_research_dataset.csv)")
    pipeline_parser.add_argument("--mem-data", default=DEFAULT_COMP3_CSV, help="Path to Component 3 memory predictions CSV")
    pipeline_parser.add_argument("--lead-time", type=int, default=5, help="Failure prediction lead-time window in minutes")
    pipeline_parser.add_argument("--output-dir", default="outputs", help="Directory to save artifacts")

    # 2. Train production parser
    train_parser = subparsers.add_parser("train", help="Train a final model on all data for production deployment")
    train_parser.add_argument("--cpu-data", default=DEFAULT_COMP2_CSV, help="Path to Component 2 CPU features CSV")
    train_parser.add_argument("--mem-data", default=DEFAULT_COMP3_CSV, help="Path to Component 3 memory predictions CSV")
    train_parser.add_argument("--lead-time", type=int, default=5, help="Failure prediction lead-time window in minutes")
    train_parser.add_argument("--model-out", default="outputs/cpu_rf_model.joblib", help="Path to write the model file")
    
    # 3. Serve parser
    serve_parser = subparsers.add_parser("serve", help="Start the real-time REST API server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    # 4. Live inference parser (CSV consumer mode)
    live_parser = subparsers.add_parser(
        "live",
        help="Start real-time CSV consumer & inference engine (reads live CSVs from Comp2/Comp3)"
    )
    live_parser.add_argument("--comp2-csv", default=DEFAULT_COMP2_CSV,
                             help="Path to Component 2 live CSV (final_research_dataset.csv)")
    live_parser.add_argument("--comp3-csv", default=DEFAULT_COMP3_CSV,
                             help="Path to Component 3 live CSV (memory_predictions.csv)")
    live_parser.add_argument("--model", default="outputs/cpu_rf_model.joblib",
                             help="Path to trained model file")
    live_parser.add_argument("--poll-interval", type=float, default=5.0,
                             help="Seconds between each CSV poll cycle (default: 5)")
    live_parser.add_argument("--output", default="outputs/live_predictions.jsonl",
                             help="Path to write prediction records (JSONL)")
    live_parser.add_argument("--status", default="outputs/live_status.json",
                             help="Path to write live status for dashboard (JSON)")
    
    args = parser.parse_args()
    
    if args.command == "validate":
        run_pipeline(args.cpu_data, args.mem_data, args.lead_time, args.output_dir)
    elif args.command == "train":
        train_production_model(args.cpu_data, args.mem_data, args.lead_time, args.model_out)
    elif args.command == "serve":
        start_server(args.host, args.port)
    elif args.command == "live":
        start_live_inference(
            comp2_path=args.comp2_csv,
            comp3_path=args.comp3_csv,
            model_path=args.model,
            poll_interval=args.poll_interval,
            output_path=args.output,
            status_path=args.status,
        )
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
