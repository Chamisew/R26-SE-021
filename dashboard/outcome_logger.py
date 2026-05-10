"""
OUTCOME LOGGER  (Step 4 — Feedback Loop)
=========================================
Every alarm event and its outcome (did a real failure happen?)
is recorded to SQLite so you can later:
  - Measure real-world precision / recall
  - Detect model drift (accuracy dropping over time)
  - Retrain the model with real labelled events

The logger runs inside the Flask app and is accessed via:
    GET /api/outcomes          — list all alarm events + outcomes
    POST /api/outcomes/<id>/confirm   — mark an alarm as "true positive"
    POST /api/outcomes/<id>/dismiss   — mark as "false positive"
    GET /api/drift             — model drift summary

Schema
------
alarm_events:
  id           INTEGER PRIMARY KEY
  service      TEXT
  fired_at     REAL (unix epoch)
  prob         REAL
  cpu_percent  REAL
  outcome      TEXT  NULL | 'true_positive' | 'false_positive'
  resolved_at  REAL  NULL
"""

import sqlite3
import os
import time
import threading
import logging
from datetime import datetime

logger = logging.getLogger("outcome_logger")

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "logs", "outcomes.db")

_lock = threading.Lock()


def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _lock:
        conn = _get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alarm_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                service     TEXT    NOT NULL,
                fired_at    REAL    NOT NULL,
                prob        REAL    NOT NULL,
                cpu_percent REAL    NOT NULL,
                outcome     TEXT    DEFAULT NULL,
                resolved_at REAL    DEFAULT NULL,
                notified    INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_service ON alarm_events(service)
        """)
        conn.commit()
        conn.close()
    logger.info(f"Outcome DB ready at {DB_PATH}")


def record_alarm(service: str, prob: float, cpu_percent: float) -> int:
    """Insert a new alarm event. Returns the new row id."""
    with _lock:
        conn = _get_conn()
        cur  = conn.execute(
            "INSERT INTO alarm_events (service, fired_at, prob, cpu_percent) VALUES (?,?,?,?)",
            (service, time.time(), prob, cpu_percent)
        )
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
    logger.info(f"Alarm recorded: id={row_id} service={service} prob={prob:.3f}")
    return row_id


def resolve_alarm(alarm_id: int, outcome: str):
    """outcome must be 'true_positive' or 'false_positive'."""
    if outcome not in ("true_positive", "false_positive"):
        raise ValueError("outcome must be 'true_positive' or 'false_positive'")
    with _lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE alarm_events SET outcome=?, resolved_at=? WHERE id=?",
            (outcome, time.time(), alarm_id)
        )
        conn.commit()
        conn.close()
    logger.info(f"Alarm {alarm_id} resolved as {outcome}")


def list_alarms(service: str = None, limit: int = 200) -> list[dict]:
    with _lock:
        conn = _get_conn()
        if service:
            rows = conn.execute(
                "SELECT * FROM alarm_events WHERE service=? ORDER BY fired_at DESC LIMIT ?",
                (service, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM alarm_events ORDER BY fired_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def drift_summary() -> dict:
    """
    Return model performance metrics from logged outcomes.
    Used to detect model drift over time.
    """
    with _lock:
        conn  = _get_conn()
        total = conn.execute("SELECT COUNT(*) FROM alarm_events").fetchone()[0]
        resolved = conn.execute(
            "SELECT COUNT(*) FROM alarm_events WHERE outcome IS NOT NULL"
        ).fetchone()[0]
        tp = conn.execute(
            "SELECT COUNT(*) FROM alarm_events WHERE outcome='true_positive'"
        ).fetchone()[0]
        fp = conn.execute(
            "SELECT COUNT(*) FROM alarm_events WHERE outcome='false_positive'"
        ).fetchone()[0]
        # Weekly trend
        week_ago = time.time() - 7 * 86400
        week_tp  = conn.execute(
            "SELECT COUNT(*) FROM alarm_events WHERE outcome='true_positive' AND fired_at>?",
            (week_ago,)
        ).fetchone()[0]
        week_fp  = conn.execute(
            "SELECT COUNT(*) FROM alarm_events WHERE outcome='false_positive' AND fired_at>?",
            (week_ago,)
        ).fetchone()[0]
        conn.close()

    precision    = tp / max(resolved, 1)
    week_prec    = week_tp / max(week_tp + week_fp, 1)
    drift_flag   = resolved >= 20 and week_prec < 0.70

    return {
        "total_alarms":    total,
        "resolved":        resolved,
        "true_positives":  tp,
        "false_positives": fp,
        "precision":       round(precision, 3),
        "week_precision":  round(week_prec, 3),
        "drift_detected":  drift_flag,
        "drift_message":   "⚠️  Precision dropped below 70% this week — consider retraining"
                           if drift_flag else "Model performing normally",
    }
