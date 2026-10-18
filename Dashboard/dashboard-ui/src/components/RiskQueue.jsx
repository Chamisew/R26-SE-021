import React, { useState, useMemo } from "react";
import {
  RISK_ORDER, ACTION_SEVERITY,
  fmtPct, fmtNum, pillClass, barClass,
  buildPath,
} from "../utils.js";
import TrendChart from "./TrendChart.jsx";
import ServiceDetail from "./ServiceDetail.jsx";

export default function RiskQueue({ services, policy, countdowns, setCountdowns }) {
  const [sortKey, setSortKey] = useState("risk");
  const [filterText, setFilterText] = useState("");
  const [groupBy, setGroupBy] = useState("none");
  const [collapsedGroups, setCollapsedGroups] = useState(() => new Set());
  const [expandedRow, setExpandedRow] = useState(null);

  const sortedServices = useMemo(() => {
    let arr = [...services];
    if (filterText.trim()) {
      const q = filterText.trim().toLowerCase();
      arr = arr.filter(s =>
        s.service.toLowerCase().includes(q) ||
        (s.project || "").toLowerCase().includes(q) ||
        (s.runtime || "").toLowerCase().includes(q) ||
        (s.action || "").toLowerCase().includes(q) ||
        s.risk.toLowerCase().includes(q)
      );
    }
    arr.sort((a, b) => {
      if (sortKey === "risk") return RISK_ORDER[a.risk] - RISK_ORDER[b.risk] || (b.cpu_probability || 0) - (a.cpu_probability || 0);
      if (sortKey === "time") {
        const av = a.historical_warning_minutes ?? 999; const bv = b.historical_warning_minutes ?? 999;
        return av - bv;
      }
      if (sortKey === "prob") return (b.cpu_probability || 0) - (a.cpu_probability || 0);
      if (sortKey === "service") return a.service.localeCompare(b.service);
      if (sortKey === "env") return (a.environment || "").localeCompare(b.environment || "");
      return 0;
    });
    return arr;
  }, [services, sortKey, filterText]);

  const sortedSorts = [["risk", "Severity"], ["time", "Time-to-failure"], ["prob", "Probability"], ["service", "Service"]];

  const groupedHtml = () => {
    if (sortedServices.length === 0) {
      return <div className="no-results">No services match &quot;{filterText}&quot;. Clear the filter to see the full queue.</div>;
    }
    if (groupBy === "none") {
      return sortedServices.map(s => (
        <RiskRow
          key={s.id}
          service={s}
          expanded={expandedRow === s.id}
          onToggle={() => setExpandedRow(expandedRow === s.id ? null : s.id)}
          countdown={countdowns[s.service]}
          setCountdowns={setCountdowns}
        />
      ));
    }
    const groups = {};
    sortedServices.forEach(s => {
      const key = s[groupBy] || "Ungrouped";
      (groups[key] = groups[key] || []).push(s);
    });
    return Object.entries(groups).map(([key, list]) => {
      const collapsed = collapsedGroups.has(key);
      const criticalIn = list.filter(s => s.risk === "CRITICAL").length;
      return (
        <React.Fragment key={key}>
          <div
            className={`group-header ${collapsed ? "collapsed" : ""}`}
            onClick={() => {
              setCollapsedGroups(prev => {
                const next = new Set(prev);
                if (next.has(key)) next.delete(key);
                else next.add(key);
                return next;
              });
            }}
          >
            <span className="chev">▾</span>
            <span>{key}</span>
            <span className="count">{list.length} service{list.length > 1 ? "s" : ""}{criticalIn ? ` · ${criticalIn} critical` : ""}</span>
          </div>
          {!collapsed && list.map(s => (
            <RiskRow
              key={s.id}
              service={s}
              expanded={expandedRow === s.id}
              onToggle={() => setExpandedRow(expandedRow === s.id ? null : s.id)}
              countdown={countdowns[s.service]}
              setCountdowns={setCountdowns}
            />
          ))}
        </React.Fragment>
      );
    });
  };

  const jumpToQueue = () => {
    const el = document.querySelector(".queue-toolbar");
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  // Expose jump function for overview cards
  if (typeof window !== "undefined") window.__jumpToQueue = jumpToQueue;

  return (
    <div className="panel">
      <div className="queue-toolbar">
        <div>
          <div className="title">Active Risk Queue</div>
          <div className="hint">Predicted incidents in priority order — click a row for the full decision breakdown</div>
        </div>
        <div className="sort-row">
          {sortedSorts.map(([k, l]) => (
            <button
              key={k}
              className={`sort-btn ${sortKey === k ? "active" : ""}`}
              onClick={() => setSortKey(k)}
            >{l}</button>
          ))}
        </div>
      </div>
      <div className="filter-bar">
        <input
          type="text"
          placeholder="Filter by service, project, runtime, action…"
          value={filterText}
          onChange={(e) => { setFilterText(e.target.value); setExpandedRow(null); }}
        />
        <div className="group-toggle">
          <button
            className={groupBy === "none" ? "active" : ""}
            onClick={() => { setGroupBy("none"); setCollapsedGroups(new Set()); }}
          >No grouping</button>
          <button
            className={groupBy === "project" ? "active" : ""}
            onClick={() => { setGroupBy("project"); setCollapsedGroups(new Set()); }}
          >Group by project</button>
        </div>
      </div>
      <div className="risk-head">
        <div></div><div>Service</div><div>Risk</div><div>CPU prob.</div><div>Memory prob.</div><div>Warning time</div><div>Failure-prob. trend</div><div>Recommended action</div>
      </div>
      {groupedHtml()}
    </div>
  );
}

function RiskRow({ service: s, expanded, onToggle, countdown, setCountdowns }) {
  const probHistory = (s.history || []).map(h => h.probability);
  const sparkColor = s.risk === "CRITICAL" ? "var(--critical)" : s.risk === "WARNING" ? "var(--warning)" : "var(--healthy)";
  const actSev = ACTION_SEVERITY[s.action] || "none";
  const actDisplay = s.action ? s.action.replaceAll("_", " ") : "No decision";

  // Dynamic countdown for warning time
  let warnTimeHtml;
  if (s.risk === "CRITICAL" && countdown) {
    if (countdown.remainingSeconds > 0 && !countdown.turnedOff) {
      const mins = Math.floor(countdown.remainingSeconds / 60);
      const secs = countdown.remainingSeconds % 60;
      const timeStr = `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
      warnTimeHtml = (
        <span className="warn-time" style={{ color: "var(--critical)", fontWeight: "bold" }}>
          🚨 {timeStr}
          <span className="lab" style={{ color: "var(--critical)" }}>
            Countdown Active
          </span>
        </span>
      );
    } else {
      warnTimeHtml = (
        <span className="warn-time" style={{ color: "var(--healthy)", fontWeight: "bold" }}>
          ✔ 00:00
          <span className="lab" style={{ color: "var(--healthy)" }}>
            Reminder Off
          </span>
        </span>
      );
    }
  } else {
    warnTimeHtml = s.historical_warning_minutes != null
      ? (<span className="warn-time">{s.historical_warning_minutes} min<span className="lab">Historical estimate</span></span>)
      : (<span className="warn-time" style={{ color: "var(--text-faint)" }}>Unknown<span className="lab">Insufficient evidence</span></span>);
  }

  return (
    <React.Fragment>
      <div
        className={`risk-row ${expanded ? "expanded" : ""}`}
        onClick={onToggle}
      >
        <div>{s.risk === "CRITICAL" ? <span className="pulse-dot critical"></span> : null}</div>
        <div>
          <div className="svc-name">{s.service}</div>
          <div className="svc-meta">{s.runtime || "Unknown"} · production · {s.project || "Unknown"}</div>
        </div>
        <div><span className={`pill ${pillClass(s.risk)}`}>{s.risk}</span></div>
        <div className="prob-cell">
          <div className="num">{fmtPct(s.cpu_probability)}</div>
          <div className="bar-track"><div className={`bar-fill ${barClass(s.risk)}`} style={{ width: `${(s.cpu_probability || 0) * 100}%` }}></div></div>
        </div>
        <div className="prob-cell">
          <div className="num">{fmtPct(s.memory_probability)}</div>
          <div className="bar-track"><div className={`bar-fill ${s.memory_alarm ? "c-crit" : "c-heal"}`} style={{ width: `${(s.memory_probability || 0) * 100}%` }}></div></div>
        </div>
        <div>{warnTimeHtml}</div>
        <div><MiniSparkline values={probHistory} color={sparkColor} /></div>
        <div><span className={`action-txt act-${actSev}`}>{actDisplay}</span></div>
      </div>
      {expanded && <ServiceDetail service={s} />}
    </React.Fragment>
  );
}

function MiniSparkline({ values, color }) {
  const w = 120, h = 32;
  if (values.length < 2) {
    return <span className="mono" style={{ color: "var(--text-faint)", fontSize: "10px" }}>insufficient history</span>;
  }
  const { line } = buildPath(values, w, h, 3, true);
  return (
    <svg className="trend-mini" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <path className="line" d={line} stroke={color} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
