"""
Queue-Aware CPU Spike Analyzer — Combined Research Pipeline Launcher
---------------------------------------------------------------------
Executes both:
1. Live Telemetry Streaming Engine (realtime_experiment.py)
2. Offline Analyzer in Watch Mode (queue_analyzer.py --watch)

Usage:
  python run_pipeline.py
"""

import subprocess
import sys
import os
import time
import signal

def main():
    print("=" * 70)
    print("    STARTING QUEUE-AWARE TELEMETRY & RESEARCH PIPELINE")
    print("=" * 70)
    print("[1/2] Starting Live Experiment Engine (realtime_experiment.py)...")
    print("[2/2] Starting Queue Analyzer Watcher (queue_analyzer.py --watch)...")
    print("-" * 70)
    print("Press Ctrl+C at any time to stop both processes cleanly.\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_script = os.path.join(script_dir, "realtime_experiment.py")
    analyzer_script = os.path.join(script_dir, "queue_analyzer.py")

    # Start live experiment engine
    p_exp = subprocess.Popen([sys.executable, exp_script])

    # Allow realtime_experiment 2 seconds to initialize CSV headers if missing
    time.sleep(2)

    # Start queue analyzer in watch mode
    p_analyzer = subprocess.Popen([sys.executable, analyzer_script, "--watch"])

    try:
        while True:
            # Check if any process terminated unexpectedly
            if p_exp.poll() is not None:
                print("[!] Live Experiment Engine stopped unexpectedly.")
                break
            if p_analyzer.poll() is not None:
                print("[!] Queue Analyzer stopped unexpectedly.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Shutting down research pipeline processes...")
    finally:
        for proc, name in [(p_exp, "realtime_experiment"), (p_analyzer, "queue_analyzer")]:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                print(f"[+] Stopped {name}.")
        print("[+] Combined research pipeline shutdown complete.")

if __name__ == "__main__":
    main()
