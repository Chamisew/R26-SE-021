import { useState } from "react";

export default function Topbar({ page, setPage, freshText, dashboard }) {
  return (
    <div className="topbar">
      <div className="topbar-row1">
        <div className="brand">
          <div className="brand-mark" title="Sentinel AIOps Engine">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L3 6V11C3 16.55 6.84 21.74 12 23C17.16 21.74 21 16.55 21 11V6L12 2Z" fill="rgba(255,255,255,0.18)" stroke="white" strokeWidth="1.8" strokeLinejoin="round"/>
              <path d="M7 12H9.5L11 8.5L13.5 15.5L15 12H17" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="12" cy="5.5" r="1.2" fill="white" />
            </svg>
          </div>
          <div>
            <div className="brand-text" style={{ fontSize: "16px", fontWeight: 800, letterSpacing: "0.5px" }}>
              SENTINEL AIOPS
            </div>
            <div className="brand-sub" style={{ fontSize: "11px", opacity: 0.85 }}>
              SRE Command Center — AIOps Decision &amp; Reliability Console
            </div>
          </div>
        </div>
        <nav className="tabs">
          <button
            className={`tab-btn ${page === "command" ? "active" : ""}`}
            onClick={() => setPage("command")}
          >Command center</button>
          <button
            className={`tab-btn ${page === "reliability" ? "active" : ""}`}
            onClick={() => setPage("reliability")}
          >Reliability &amp; model health</button>
        </nav>
        <div className="brand-sub">{freshText}</div>
      </div>
      <StatusStrip dashboard={dashboard} />
    </div>
  );
}

function StatusStrip({ dashboard: dash }) {
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savedMsg, setSavedMsg] = useState("");
  const [cpuInput, setCpuInput] = useState("");
  const [memInput, setMemInput] = useState("");

  if (!dash || !dash.summary) {
    return (
      <div className="status-strip">
        <div className="status-seg hero">
          <div className="lab">Predicted-risk situation</div>
          <div className="val">Loading…</div>
        </div>
      </div>
    );
  }

  const x = dash.summary;
  const pol = dash.policy || {};
  const cpuThr = pol.cpu_threshold ?? 0.6;
  const memThr = pol.memory_threshold ?? 0.7;
  const lastUpdated = pol.last_updated ? new Date(pol.last_updated).toLocaleTimeString() : null;

  const headline = x.critical
    ? `${x.critical} critical prediction${x.critical > 1 ? "s" : ""}`
    : x.warning
      ? `${x.warning} warning prediction${x.warning > 1 ? "s" : ""}`
      : "No active alarm";
  const hlClass = x.critical ? "critical" : x.warning ? "warning" : "healthy";

  const openEdit = () => {
    setCpuInput(Math.round(cpuThr * 100));
    setMemInput(Math.round(memThr * 100));
    setSavedMsg("");
    setEditing(true);
  };

  const applyPolicy = async () => {
    const cpu = parseFloat(cpuInput) / 100;
    const mem = parseFloat(memInput) / 100;
    if (isNaN(cpu) || isNaN(mem) || cpu <= 0 || cpu > 1 || mem <= 0 || mem > 1) {
      setSavedMsg("⚠ Enter valid values (1–100)");
      return;
    }
    setSaving(true);
    try {
      await fetch("/api/policy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cpu_threshold: cpu, memory_threshold: mem }),
      });
      setSavedMsg("✓ Policy updated live!");
      setTimeout(() => { setEditing(false); setSavedMsg(""); }, 1200);
    } catch {
      setSavedMsg("✗ Update failed");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="status-strip">
      <div className="status-seg hero">
        <div className="lab">Predicted-risk situation</div>
        <div className={`val ${hlClass}`}>
          {x.critical ? <span className="pulse-dot critical"></span>
            : x.healthy ? <span className="pulse-dot healthy"></span>
              : null}
          {headline}
        </div>
      </div>
      <div className="status-seg">
        <div className="lab">Monitored</div>
        <div className="val">{x.services}</div>
      </div>
      <div className="status-seg">
        <div className="lab">At risk</div>
        <div className="val" style={{ color: "var(--warning)" }}>{x.at_risk}</div>
      </div>
      <div className="status-seg">
        <div className="lab">Critical</div>
        <div className="val" style={{ color: "var(--critical)" }}>{x.critical}</div>
      </div>
      <div className="status-seg">
        <div className="lab">Healthy</div>
        <div className="val" style={{ color: "var(--healthy)" }}>{x.healthy}</div>
      </div>

      {/* Live-editable Threshold segment */}
      <div className="status-seg policy-seg" style={{ position: "relative" }}>
        <div className="lab">
          Thresholds{" "}
          <span style={{ fontSize: "0.65rem", opacity: 0.6 }}>
            {lastUpdated ? `· updated ${lastUpdated}` : "· default"}
          </span>
        </div>
        <div
          className="val policy-val"
          onClick={openEdit}
          title="Click to adjust SRE alarm thresholds"
          style={{ cursor: "pointer", display: "flex", alignItems: "center", gap: "0.4rem" }}
        >
          <span style={{ color: "var(--warning)" }}>CPU {Math.round(cpuThr * 100)}%</span>
          <span style={{ opacity: 0.4 }}>/</span>
          <span style={{ color: "#a78bfa" }}>Mem {Math.round(memThr * 100)}%</span>
          <span style={{ fontSize: "0.7rem", opacity: 0.5 }}>✎</span>
        </div>

        {editing && (
          <div style={{
            position: "absolute",
            bottom: "calc(100% + 8px)",
            right: 0,
            background: "var(--surface, #1a1a2e)",
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: "10px",
            padding: "1rem",
            minWidth: "240px",
            boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
            zIndex: 999,
          }}>
            <div style={{ fontWeight: 600, marginBottom: "0.75rem", fontSize: "0.85rem" }}>
              ⚙ SRE Alarm Policy
            </div>
            <label style={{ fontSize: "0.75rem", opacity: 0.7, display: "block", marginBottom: "0.25rem" }}>
              CPU failure threshold (%)
            </label>
            <input
              type="number"
              min="1"
              max="100"
              value={cpuInput}
              onChange={e => setCpuInput(e.target.value)}
              style={{
                width: "100%",
                padding: "0.4rem 0.6rem",
                borderRadius: "6px",
                border: "1px solid rgba(255,255,255,0.15)",
                background: "rgba(255,255,255,0.06)",
                color: "inherit",
                fontSize: "0.9rem",
                marginBottom: "0.6rem",
                boxSizing: "border-box",
              }}
            />
            <label style={{ fontSize: "0.75rem", opacity: 0.7, display: "block", marginBottom: "0.25rem" }}>
              Memory failure threshold (%)
            </label>
            <input
              type="number"
              min="1"
              max="100"
              value={memInput}
              onChange={e => setMemInput(e.target.value)}
              style={{
                width: "100%",
                padding: "0.4rem 0.6rem",
                borderRadius: "6px",
                border: "1px solid rgba(255,255,255,0.15)",
                background: "rgba(255,255,255,0.06)",
                color: "inherit",
                fontSize: "0.9rem",
                marginBottom: "0.75rem",
                boxSizing: "border-box",
              }}
            />
            {savedMsg && (
              <div style={{
                fontSize: "0.75rem",
                marginBottom: "0.5rem",
                color: savedMsg.startsWith("✓") ? "#4ade80" : "#f87171",
              }}>
                {savedMsg}
              </div>
            )}
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <button
                onClick={applyPolicy}
                disabled={saving}
                style={{
                  flex: 1,
                  padding: "0.4rem 0.75rem",
                  borderRadius: "6px",
                  border: "none",
                  background: saving ? "#444" : "#f59e0b",
                  color: "#000",
                  fontWeight: 600,
                  fontSize: "0.8rem",
                  cursor: saving ? "not-allowed" : "pointer",
                }}
              >
                {saving ? "Saving…" : "Apply live"}
              </button>
              <button
                onClick={() => setEditing(false)}
                style={{
                  padding: "0.4rem 0.75rem",
                  borderRadius: "6px",
                  border: "1px solid rgba(255,255,255,0.15)",
                  background: "transparent",
                  color: "inherit",
                  fontSize: "0.8rem",
                  cursor: "pointer",
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
