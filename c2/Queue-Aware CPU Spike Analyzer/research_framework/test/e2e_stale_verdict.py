"""End-to-end: stale-queue service with healthy traffic must NOT read FAILED.

Spins up a stub cloud agent (stale queue=1 placeholder, cumulative counters,
calm CPU) plus a backend pointed at an ISOLATED cwd (temp dir) so the user's
real services_config.json and dataset are untouched. Asserts the verdict once
the queue is flagged stale in PROVISIONAL tier.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

STUB_PORT = 18091
BACKEND_PORT = 8091
T0 = int(time.time())

STUB_CODE = f"""
import json, math, time
from http.server import BaseHTTPRequestHandler, HTTPServer

class H(BaseHTTPRequestHandler):
    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        t = time.time() - {T0}
        if self.path.startswith("/metrics"):
            self._send({{"cpu_usage_percent": round(20 + 2 * math.sin(t / 10), 2),
                        "incoming_requests": int(t * 3) + 1000,
                        "processing_requests": int(t * 3) + 999,
                        "queue_length": 1, "latency_ms": 1.5}})
        elif self.path.startswith("/health"):
            self._send({{"target_reachable": True}})
        elif self.path.startswith("/ping"):
            self._send({{"status": "ok"}})
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a):
        pass

HTTPServer(("127.0.0.1", {STUB_PORT}), H).serve_forever()
"""

workdir = tempfile.mkdtemp(prefix="qa_e2e_")
with open(os.path.join(workdir, "services_config.json"), "w") as f:
    json.dump({"_comment": "e2e", "services": [
        {"id": "e2e-stub", "name": "e2e-stub",
         "base_url": f"http://127.0.0.1:{STUB_PORT}",
         "platform": "other", "enabled": True, "status": "unknown",
         "last_checked": None}]}, f)

repo_fw = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
stub = subprocess.Popen([sys.executable, "-c", STUB_CODE])
env = dict(os.environ, PORT=str(BACKEND_PORT))
backend = subprocess.Popen([sys.executable,
                             os.path.join(repo_fw, "realtime_experiment.py")],
                            cwd=workdir, env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
try:
    api = f"http://127.0.0.1:{BACKEND_PORT}/api/data"

    def snap():
        with urllib.request.urlopen(api, timeout=15) as r:
            return json.load(r)

    row = None
    post_calib_states = []
    for i in range(46):  # ~230s budget
        time.sleep(5)
        try:
            d = snap()
        except Exception as e:  # noqa: BLE001
            print(f"[{i}] backend not up yet ({e})", flush=True)
            continue
        services = d.get("services") or {}
        row = services.get("e2e-stub")
        if not row:
            print(f"[{i}] service not discovered yet", flush=True)
            continue
        print(f"[{i}] t+{(i+1)*5:3d}s tier={row.get('baseline_tier')} "
              f"stale={row.get('queue_stale')} state={row.get('predicted_state')} "
              f"cpu={row.get('cpu_percent')} qref={row.get('qref')} qpi={row.get('qpi')} "
              f"qpiz={row.get('qpi_zscore')} lam={row.get('incoming_rate')}",
              flush=True)
        if row.get("baseline_tier") != "CALIBRATING":
            post_calib_states.append(row.get("predicted_state"))
        if row.get("baseline_tier") == "PROVISIONAL" and row.get("queue_stale") == 1:
            break
    assert row, "stub service never appeared in /api/data"
    assert row.get("queue_stale") == 1, "queue never flagged stale"
    assert row.get("baseline_tier") in ("PROVISIONAL", "STATISTICAL"), row.get("baseline_tier")
    # A calm, reachable service must never read FAILED — not during the
    # pre-stale window (Qref must not freeze from placeholder history) and
    # not after (queue terms gated).
    assert "FAILED" not in post_calib_states, \
        f"false FAILED observed post-calibration: {post_calib_states}"
    assert row.get("predicted_state") in ("HEALTHY", "DEGRADED", "RECOVERING"), \
        f"stale-queue service misclassified: {row.get('predicted_state')}"
    # Qref must NOT have frozen from placeholder history (stays default 10.0:
    # zero-variance queue history is not a distribution). Queue baselines may
    # hold pre-flag placeholder samples, but they are inert: zero variance ->
    # zscore None, and all queue terms are gated once stale flags.
    assert row.get("qref") == 10.0, f"Qref froze from placeholder: {row.get('qref')}"
    assert (row.get("cpu_mean") or 0) > 10, "cpu baseline missing"
    print("E2E STALE-QUEUE VERDICT CHECKS PASSED")
finally:
    backend.terminate()
    stub.terminate()
    time.sleep(2)
    shutil.rmtree(workdir, ignore_errors=True)
