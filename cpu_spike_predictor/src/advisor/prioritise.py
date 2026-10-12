"""
advisor.prioritise — Smart fleet prioritisation + cascade grouping
==================================================================

Two jobs:

1. Rank the fleet so the operator sees the *right service first*. Urgency is a
   weighted blend of:
     * composite failure risk (CPU + memory channels),
     * time pressure (an inverse function of forecast time-to-failure — less
       runway is more urgent),
     * service criticality (business weight 1..5),
     * dependency fan-out (how many downstream services would be hit).

2. Collapse correlated alerts across dependent services into a single cascade
   group (union-find over dependency edges + simultaneous activation), so one
   upstream spike firing twenty downstream alarms reads as ONE incident with a
   named patient-zero instead of twenty competing pages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger("advisor.prioritise")

# Weights of the urgency blend (tunable; sum ~1).
W_RISK = 0.40
W_TIME = 0.30
W_CRIT = 0.20
W_FANOUT = 0.10
MAX_FANOUT = 10.0


@dataclass
class ServiceSignal:
    service_name: str
    cpu_prob: float
    mem_prob: float | None = None
    ttf_p50: float | None = None
    criticality: float = 3.0                 # 1 (low) .. 5 (business-critical)
    fan_out: float = 1.0                     # dependent-service count
    action: str = "NO_ACTION"
    depends_on: list[str] = field(default_factory=list)
    is_alert: bool = False
    incident_id: str | None = None


def _risk(signal: ServiceSignal) -> float:
    p_cpu = float(np.clip(signal.cpu_prob, 0.0, 1.0))
    p_mem = float(np.clip(signal.mem_prob, 0.0, 1.0)) if signal.mem_prob is not None else 0.0
    return 1.0 - (1.0 - p_cpu) * (1.0 - p_mem)


def _time_pressure(ttf_p50: float | None) -> float:
    """Map minutes-to-failure onto [0,1] urgency (short runway -> ~1)."""
    if ttf_p50 is None:
        return 0.0
    if ttf_p50 <= 0:
        return 1.0
    return float(np.clip(1.0 / (1.0 + ttf_p50 / 5.0), 0.0, 1.0))


def urgency_score(signal: ServiceSignal) -> float:
    risk = _risk(signal)
    time_p = _time_pressure(signal.ttf_p50)
    crit = float(np.clip(signal.criticality, 1, 5)) / 5.0
    fan = float(np.clip(signal.fan_out / MAX_FANOUT, 0.0, 1.0))
    # Risk gates everything: a healthy service is never urgent regardless of crit.
    raw = W_RISK * risk + W_TIME * time_p * (0.3 + 0.7 * risk) + W_CRIT * crit * risk + W_FANOUT * fan * risk
    return round(float(np.clip(raw, 0.0, 1.0)), 4)


def prioritise_fleet(signals: list[ServiceSignal]) -> list[dict]:
    """Return services ordered by descending urgency with the score breakdown."""
    rows = []
    for s in signals:
        rows.append({
            "service_name": s.service_name,
            "urgency": urgency_score(s),
            "risk": round(_risk(s), 4),
            "time_pressure": round(_time_pressure(s.ttf_p50), 4),
            "ttf_p50_min": s.ttf_p50,
            "criticality": s.criticality,
            "fan_out": s.fan_out,
            "action": s.action,
            "is_alert": s.is_alert or _risk(s) >= 0.5,
        })
    rows.sort(key=lambda r: r["urgency"], reverse=True)
    for i, r in enumerate(rows, start=1):
        r["priority_rank"] = i
    return rows


class _UnionFind:
    def __init__(self, items):
        self.parent = {it: it for it in items}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def group_cascades(signals: list[ServiceSignal]) -> list[dict]:
    """Collapse correlated alarms into cascade groups via union-find.

    Two services join the same cascade when (a) one ``depends_on`` the other, or
    (b) they share an ``incident_id``. Each group is summarised by its
    patient-zero (highest-urgency member) with the members sorted by urgency so
    the operator triages the root first and the noise is deduplicated.
    """
    by_name = {s.service_name: s for s in signals}
    names = list(by_name.keys())
    uf = _UnionFind(names)

    for s in signals:
        for dep in s.depends_on:
            if dep in by_name:
                uf.union(s.service_name, dep)
        if s.incident_id is not None:
            for other in signals:
                if other.incident_id is not None and other.incident_id == s.incident_id:
                    uf.union(s.service_name, other.service_name)

    groups: dict[str, list[ServiceSignal]] = {}
    for n in names:
        groups.setdefault(uf.find(n), []).append(by_name[n])

    out = []
    for root, members in groups.items():
        scored = sorted(members, key=urgency_score, reverse=True)
        lead = scored[0]
        alert_members = [m for m in scored if m.is_alert or _risk(m) >= 0.5]
        out.append({
            "cascade_id": root,
            "patient_zero": lead.service_name,
            "lead_urgency": urgency_score(lead),
            "lead_action": lead.action,
            "size": len(scored),
            "alert_count": len(alert_members),
            "deduped_alerts": len(alert_members) - 1 if len(alert_members) > 1 else 0,
            "members": [m.service_name for m in scored],
        })
    out.sort(key=lambda g: g["lead_urgency"], reverse=True)
    return out


def prioritise_from_dataframe(df: pd.DataFrame,
                              criticality_map: dict[str, float] | None = None,
                              fanout_map: dict[str, float] | None = None) -> list[dict]:
    """Convenience adapter from a predictions frame to ranked fleet rows."""
    criticality_map = criticality_map or {}
    fanout_map = fanout_map or {}
    signals = []
    for _, r in df.iterrows():
        signals.append(ServiceSignal(
            service_name=str(r.get("service_name", "unknown")),
            cpu_prob=float(r.get("cpu_failure_prob", 0.0) or 0.0),
            mem_prob=float(r["memory_prob"]) if "memory_prob" in r and pd.notna(r.get("memory_prob")) else None,
            ttf_p50=float(r["ttf_p50"]) if "ttf_p50" in r and pd.notna(r.get("ttf_p50")) else None,
            criticality=float(criticality_map.get(r.get("service_name"), 3.0)),
            fan_out=float(fanout_map.get(r.get("service_name"), 1.0)),
        ))
    return prioritise_fleet(signals)


if __name__ == "__main__":  # pragma: no cover
    demo = [
        ServiceSignal("checkout", 0.9, 0.2, ttf_p50=1.5, criticality=5, fan_out=8,
                      depends_on=["cart", "auth"]),
        ServiceSignal("cart", 0.7, 0.1, ttf_p50=3.0, criticality=4, fan_out=3,
                      depends_on=["auth"]),
        ServiceSignal("auth", 0.3, 0.1, ttf_p50=None, criticality=5, fan_out=12),
        ServiceSignal("batch-report", 0.85, 0.1, ttf_p50=2.0, criticality=1, fan_out=1),
    ]
    for row in prioritise_fleet(demo):
        print(f"#{row['priority_rank']} {row['service_name']:12s} urgency={row['urgency']}")
    print("--- cascades ---")
    for g in group_cascades(demo):
        print(g["cascade_id"], "PZ=", g["patient_zero"], "members=", g["members"],
              "deduped=", g["deduped_alerts"])
