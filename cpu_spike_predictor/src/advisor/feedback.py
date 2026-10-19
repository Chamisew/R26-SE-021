"""
advisor.feedback — Append-only operator decision log
====================================================

Every operator interaction with the Decision Advisor (accept / ignore / override,
the action finally taken, and the later outcome) is recorded to an append-only
JSONL file using the same atomic write discipline as
:class:`src.live_inference.CheckpointManager`: lines are written to a temp file
and ``os.replace``-d, so a crash never leaves a half-written record and the log is
never corrupted mid-append.

This log is the raw material for offline policy evaluation (see
:func:`src.advisor.policy.evaluate_offline_policy`) and for retrain suggestions.
"""

from __future__ import annotations

import os
import json
import time
import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger("advisor.feedback")

DEFAULT_FEEDBACK_PATH = "outputs/advisor_feedback.jsonl"
VALID_DECISIONS = ("accepted", "ignored", "overridden")


class FeedbackLogger:
    """Thread-safe, atomic, append-only JSONL decision log."""

    def __init__(self, path: str = DEFAULT_FEEDBACK_PATH):
        self.path = path
        self._lock = threading.Lock()

    def log(self, service_name: str, decision: str, recommended_action: str,
            chosen_action: str | None = None, outcome: str | None = None,
            confidence: float | None = None, context: dict | None = None,
            actor: str = "operator") -> dict:
        """Append one decision record. Raises ValueError on an unknown decision."""
        if decision not in VALID_DECISIONS:
            raise ValueError(f"decision must be one of {VALID_DECISIONS}, got '{decision}'")
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "service_name": service_name,
            "decision": decision,
            "recommended_action": recommended_action,
            "chosen_action": chosen_action if decision == "overridden" else (
                recommended_action if decision == "accepted" else None),
            "outcome": outcome,
            "confidence": confidence,
            "actor": actor,
            "context": context or {},
        }
        with self._lock:
            self._append_atomic(record)
        return record

    def _append_atomic(self, record: dict) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        line = json.dumps(record, separators=(",", ":")) + "\n"
        # Read existing, append, write to temp, atomic replace. Keeps writes
        # crash-safe (mirrors CheckpointManager.save) at the cost of O(n) file
        # size — bounded by the trim below.
        existing = ""
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                existing = f.read()
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            f.write(existing + line)
        os.replace(tmp, self.path)
        self._maybe_trim()

    def _maybe_trim(self, max_records: int = 50000) -> None:
        try:
            with open(self.path) as f:
                lines = f.readlines()
        except Exception:
            return
        if len(lines) > max_records:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                f.writelines(lines[-max_records:])
            os.replace(tmp, self.path)

    def read_all(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    logger.warning("Skipping corrupt feedback line")
        return out

    def override_rate(self) -> dict:
        recs = self.read_all()
        n = len(recs)
        if n == 0:
            return {"n": 0, "accepted": 0.0, "ignored": 0.0, "overridden": 0.0}
        counts = {d: sum(1 for r in recs if r.get("decision") == d) for d in VALID_DECISIONS}
        return {"n": n, **{k: round(v / n, 4) for k, v in counts.items()}}


# Module-level default instance used by the API layer.
_default_logger: FeedbackLogger | None = None


def get_default_logger(path: str = DEFAULT_FEEDBACK_PATH) -> FeedbackLogger:
    global _default_logger
    if _default_logger is None or _default_logger.path != path:
        _default_logger = FeedbackLogger(path)
    return _default_logger


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    fl = FeedbackLogger("outputs/_feedback_demo.jsonl")
    fl.log("checkout", "accepted", "PROACTIVE_RESTART", confidence=0.62)
    fl.log("cart", "overridden", "LOAD_SHEDDING", chosen_action="SCALE_OUT")
    fl.log("auth", "ignored", "NO_ACTION")
    print("override stats:", fl.override_rate())
    os.remove("outputs/_feedback_demo.jsonl")
