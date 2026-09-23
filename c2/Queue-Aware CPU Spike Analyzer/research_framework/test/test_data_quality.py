"""Regression tests for counter-semantics-aware rates and queue-staleness flags."""
import os
import sys
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(FRAMEWORK_DIR)
from realtime_experiment import (classify_counter, long_window_rate,
                                 rate_windows, update_queue_staleness,
                                 queue_stale, queue_signals_ok,
                                 QUEUE_STALE_WINDOW, RATE_AVG_WINDOW)

# 1. cumulative counter: steady growth, one restart dip
hist = deque([100 + i for i in range(29)] + [5], maxlen=60)
assert classify_counter(hist) == "cumulative", "cumulative misclassified"

# 2. idle cumulative (no traffic, /ping excluded): constant
hist = deque([3457] * 30, maxlen=60)
assert classify_counter(hist) == "cumulative"

# 3. windowed per-second counts (sidecar.py style): jitter up/down
hist = deque([3, 1, 4, 2, 5, 1, 3, 1, 4, 2, 5, 1, 3, 4, 0, 2, 5, 3, 1, 4, 2,
              0, 3, 5, 1, 2, 4, 3, 0, 1], maxlen=60)
assert classify_counter(hist) == "windowed", "windowed misclassified"

# 4. too few samples
assert classify_counter(deque([1, 2, 3], maxlen=60)) == "unknown"

# 5. long-window average: steady 3/s observed every tick -> ~3.0, never 0/spike
name = "test-svc-steady"
reported = []
for t in range(120):
    r = long_window_rate(name, "inc", 1000 + 3 * t, float(t))
    if t >= 10:
        reported.append(r)
assert all(2.5 <= r <= 3.5 for r in reported), f"steady rate unstable: {reported[:5]}"

# 6. BURST-SWEEP REPLAY (live incident 2026-10-09): true 3/s, but the backend
# only observes the counters every 9s (+27 jumps). Old per-tick delta math
# read 0.0 ~93% of ticks with 16x-51x spikes; the window average must hold ~3.
name = "test-svc-burst"
reported = []
for t in range(150):
    observed = 5000 + 27 * (t // 9)  # frozen between 9s sweeps, +27 per sweep
    r = long_window_rate(name, "inc", observed, float(t))
    if t >= 15:
        reported.append(r)
assert reported, "no rates reported after warmup"
assert all(r > 0 for r in reported), "burst pattern must never read 0.0 once warm"
assert all(r < 6.0 for r in reported), f"spike leaked through: {max(reported)}"
assert 2.0 <= sum(reported) / len(reported) <= 4.0, "mean must track true 3/s"

# 7. counter reset mid-stream: rebase to 0, then recover to true rate
name = "test-svc-reset"
for t in range(30):
    long_window_rate(name, "inc", 2000 + 3 * t, float(t))
r = long_window_rate(name, "inc", 4, 30.0)  # redeploy: counters restart near 0
assert r == 0.0, "reset tick must rebase to 0"
for t in range(31, 60):
    r = long_window_rate(name, "inc", 4 + 3 * (t - 30), float(t))
assert 2.0 <= r <= 4.0, f"rate must recover after reset, got {r}"

# 8. idle service: constant counters -> 0.0, never NaN/negative
name = "test-svc-idle"
for t in range(40):
    r = long_window_rate(name, "inc", 3457, float(t))
assert r == 0.0
assert RATE_AVG_WINDOW == 90.0 and len(rate_windows[name]["inc"]) > 0

# 5. queue staleness: frozen 1.0 flags after window; 0 never flags; unstick re-arms
name = "test-svc-stale"
for _ in range(QUEUE_STALE_WINDOW - 1):
    assert update_queue_staleness(name, 1.0) is False
assert update_queue_staleness(name, 1.0) is True, "frozen queue must flag"
assert queue_stale[name] is True
assert update_queue_staleness(name, 1.0) is True, "stays flagged while stuck"
assert update_queue_staleness(name, 2.0) is False, "unstick clears"
name0 = "test-svc-idle"
for _ in range(QUEUE_STALE_WINDOW):
    assert update_queue_staleness(name0, 0.0) is False, "idle zero queue is legit"

# 9. queue-signal trust gate: stale placeholders never drive verdicts/baselines
assert queue_signals_ok(False) is True, "live queue must be trusted"
assert queue_signals_ok(True) is False, "stale queue must be distrusted"
assert queue_signals_ok(0) is True
assert queue_signals_ok(1) is False

print("ALL DATA-QUALITY UNIT CHECKS PASSED")
