import { useState } from "react";

export default function OverviewCards({ summary, policy, services, onJumpToQueue, onExport, onGotoReliability }) {
  const [showRunbooks, setShowRunbooks] = useState(false);
  const [showModelConfig, setShowModelConfig] = useState(false);
  const [selectedRunbookSvc, setSelectedRunbookSvc] = useState(null);

  const hasCritical = summary.critical > 0;
  const totalRisk = summary.at_risk;
  const overallHealthPct = summary.services > 0 ? Math.round((summary.healthy / summary.services) * 100) : 100;
  const gaugeColor = overallHealthPct >= 80 ? "var(--healthy)" : overallHealthPct >= 50 ? "var(--warning)" : "var(--critical)";
  const circ = 2 * Math.PI * 52;
  const dash = (overallHealthPct / 100) * circ;
  const spotClass = hasCritical ? "" : "healthy";
  const spotTitle = hasCritical
    ? `${summary.critical} service${summary.critical > 1 ? "s" : ""} need${summary.critical > 1 ? "" : "s"} immediate attention`
    : "All services within normal operating range";
  const spotSub = hasCritical
    ? `${summary.warning} additional warning${summary.warning === 1 ? "" : "s"} in queue. Prioritize critical row first.`
    : `No active alarms — ${summary.healthy} service${summary.healthy === 1 ? "" : "s"} reporting nominal telemetry.`;
  const spotEyebrow = hasCritical ? "Active incident response" : "Nominal operating state";

  // Identify at-risk or active services for the runbook list
  const runbookServices = (services || []).filter(s => s.sre_runbook_steps && s.sre_runbook_steps.length > 0);

  const handleOpenRunbooks = () => {
    if (runbookServices.length > 0) {
      // Prioritize selecting an at-risk service first
      const atRiskSvc = runbookServices.find(s => s.risk !== "HEALTHY");
      setSelectedRunbookSvc(atRiskSvc ? atRiskSvc.id : runbookServices[0].id);
    }
    setShowRunbooks(true);
  };

  const activeRunbookSvc = runbookServices.find(s => s.id === selectedRunbookSvc);

  return (
    <>
      <div className="overview-wrap">
        <div className="ov-row row1">
          <div className="ov-card">
            <div className="ov-card-title">System-wide reliability overview</div>
            <div className="ov-card-sub">Live snapshot of the monitored fleet, updated every 1s</div>
            <div className="info-card">
              <svg className="info-illustration" viewBox="0 0 88 88" fill="none">
                <rect x="4" y="4" width="80" height="80" rx="16" fill="var(--accent-dim)" />
                <path d="M22 52 L34 40 L46 48 L66 26" stroke="var(--accent)" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
                <circle cx="66" cy="26" r="5" fill="var(--accent)" />
              </svg>
              <div className="info-rows">
                <div className="info-row"><span className="k">Services monitored</span><span className="v">{summary.services}</span></div>
                <div className="info-row"><span className="k">Currently at risk</span><span className="v" style={{ color: "var(--warning)" }}>{summary.at_risk}</span></div>
                <div className="info-row"><span className="k">Critical predictions</span><span className="v" style={{ color: "var(--critical)" }}>{summary.critical}</span></div>
                <div className="info-row"><span className="k">Healthy services</span><span className="v" style={{ color: "var(--healthy)" }}>{summary.healthy}</span></div>
              </div>
            </div>
          </div>
          <div className="ov-card">
            <div className="ov-card-title">Overall fleet health</div>
            <div className="ov-card-sub">Fraction of services not in alarm state</div>
            <div className="gauge-card">
              <div className="gauge-wrap">
                <svg viewBox="0 0 120 120">
                  <circle cx="60" cy="60" r="52" stroke="var(--border)" strokeWidth="10" fill="none" />
                  <circle cx="60" cy="60" r="52" stroke={gaugeColor} strokeWidth="10" fill="none"
                    strokeDasharray={`${dash} ${circ}`} strokeLinecap="round" />
                </svg>
                <div className="gauge-center">
                  <div className="pct" style={{ color: gaugeColor }}>{overallHealthPct}%</div>
                  <div className="mono" style={{ fontSize: "10px", color: "var(--text-faint)" }}>HEALTHY</div>
                </div>
              </div>
              <div className="gauge-foot">Health score based on live predictions</div>
            </div>
          </div>
          <div className="ov-card">
            <div className="ov-card-title">Risk situation</div>
            <div className="ov-card-sub">Operational status right now</div>
            <div className="badge-card">
              <div className={`badge-ring ${hasCritical ? "crit" : "ok"}`}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  {hasCritical ? (
                    <>
                      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                      <line x1="12" y1="9" x2="12" y2="13" />
                      <line x1="12" y1="17" x2="12.01" y2="17" />
                    </>
                  ) : (
                    <>
                      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                      <polyline points="22 4 12 14.01 9 11.01" />
                    </>
                  )}
                </svg>
              </div>
              <div className="badge-status-txt" style={{ color: hasCritical ? "var(--critical)" : "var(--healthy)" }}>
                {hasCritical ? "Active incident" : "All clear"}
              </div>
              <div className="badge-status-sub">
                {summary.warning} warning{summary.warning === 1 ? "" : "s"} · {summary.healthy} healthy · <a onClick={onJumpToQueue}>view queue</a>
              </div>
            </div>
          </div>
        </div>

        <div className="ov-row row2">
          <div className={`spotlight-card ${spotClass}`}>
            <div>
              <div className="spotlight-eyebrow">{spotEyebrow}</div>
              <div className="spotlight-title">{spotTitle}</div>
              <div className="spotlight-meta">{spotSub}</div>
            </div>
            <button className="spotlight-link" onClick={onJumpToQueue}>
              Open risk queue
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="5" y1="12" x2="19" y2="12" />
                <polyline points="12 5 19 12 12 19" />
              </svg>
            </button>
          </div>
          <div className="ov-card">
            <div className="ov-card-title">Quick actions</div>
            <div className="ov-card-sub">Common SRE operations</div>
            <div className="qa-grid">
              <button className="qa-btn" onClick={onExport} title="Download live telemetry report in JSON format">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                Export report
              </button>
              <button className="qa-btn" onClick={onGotoReliability} title="View model validation & LOPO scores tab">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M4 19V5m0 14h16M8 19v-6m4 6V9m4 10v-9m4 9V6" />
                </svg>
                Reliability
              </button>
              <button className="qa-btn" onClick={handleOpenRunbooks} title="View generated SRE mitigation runbooks">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="16" y1="13" x2="8" y2="13" />
                  <line x1="16" y1="17" x2="8" y2="17" />
                </svg>
                Runbooks
              </button>
              <button className="qa-btn" onClick={() => setShowModelConfig(true)} title="View AIOps model features and thresholds config">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="3" />
                  <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
                </svg>
                Model config
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Runbooks Modal */}
      {showRunbooks && (
        <div className="modal-overlay" onClick={() => setShowRunbooks(false)}>
          <div className="modal-container" onClick={e => e.stopPropagation()} style={{ maxWidth: "750px" }}>
            <div className="modal-header">
              <h3>📖 SRE Mitigation Runbooks</h3>
              <div className="modal-close" onClick={() => setShowRunbooks(false)}>×</div>
            </div>
            <div className="modal-body" style={{ display: "grid", gridTemplateColumns: "220px 1fr", gap: "20px", height: "450px" }}>
              <div style={{ borderRight: "1px solid var(--border-soft)", paddingRight: "15px", display: "flex", flexDirection: "column", gap: "8px", overflowY: "auto" }}>
                <div style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", color: "var(--text-faint)", marginBottom: "4px" }}>Monitored Fleet</div>
                {runbookServices.map(s => {
                  const isSelected = s.id === selectedRunbookSvc;
                  const isAtRisk = s.risk !== "HEALTHY";
                  return (
                    <button
                      key={s.id}
                      onClick={() => setSelectedRunbookSvc(s.id)}
                      style={{
                        textAlign: "left",
                        padding: "8px 12px",
                        borderRadius: "8px",
                        fontSize: "12px",
                        fontWeight: "600",
                        display: "flex",
                        flexDirection: "column",
                        gap: "2px",
                        background: isSelected ? "var(--accent-dim)" : "transparent",
                        color: isSelected ? "var(--accent)" : "var(--text-dim)",
                        border: isSelected ? "1px solid var(--accent-dim)" : "1px solid transparent",
                      }}
                    >
                      <span>{s.service}</span>
                      <span style={{ fontSize: "10px", fontWeight: "700", color: isAtRisk ? (s.risk === "CRITICAL" ? "var(--critical)" : "var(--warning)") : "var(--healthy)" }}>
                        {s.risk}
                      </span>
                    </button>
                  );
                })}
              </div>
              <div style={{ overflowY: "auto" }}>
                {activeRunbookSvc ? (
                  <div>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
                      <h4 style={{ margin: 0, fontSize: "16px", fontWeight: 800 }}>{activeRunbookSvc.service}</h4>
                      <span className={`pill ${activeRunbookSvc.risk.toLowerCase()}`}>{activeRunbookSvc.risk}</span>
                    </div>
                    {activeRunbookSvc.warning_message && (
                      <div className="narrative" style={{ marginBottom: "14px", borderLeft: "3px solid var(--accent)" }}>
                        <strong>Diagnostic:</strong> {activeRunbookSvc.warning_message}
                      </div>
                    )}
                    <div style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", color: "var(--text-faint)", marginBottom: "8px" }}>Recommended Playbook Steps</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                      {(activeRunbookSvc.sre_runbook_steps || []).map((step, idx) => (
                        <div key={idx} style={{ background: "var(--panel2)", border: "1px solid var(--border-soft)", borderRadius: "8px", padding: "10px 12px", fontSize: "12px" }}>
                          <div style={{ fontWeight: "700", color: "var(--accent)", marginBottom: "2px", fontSize: "10px", textTransform: "uppercase" }}>Step {idx + 1}</div>
                          <div style={{ fontFamily: "var(--mono)", color: "var(--text)" }}>{step}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="empty-note">Select a service to view its mitigation playbook.</div>
                )}
              </div>
            </div>
            <div className="modal-footer">
              <button className="modal-btn secondary" onClick={() => setShowRunbooks(false)}>Close</button>
            </div>
          </div>
        </div>
      )}

      {/* Model Config Modal */}
      {showModelConfig && (
        <div className="modal-overlay" onClick={() => setShowModelConfig(false)}>
          <div className="modal-container" onClick={e => e.stopPropagation()} style={{ maxWidth: "550px" }}>
            <div className="modal-header">
              <h3>⚙ AIOps Model &amp; Decision Configurations</h3>
              <div className="modal-close" onClick={() => setShowModelConfig(false)}>×</div>
            </div>
            <div className="modal-body" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px" }}>
                <div style={{ background: "var(--panel2)", border: "1px solid var(--border-soft)", borderRadius: "8px", padding: "12px" }}>
                  <div style={{ fontSize: "10px", color: "var(--text-faint)", fontWeight: "700", textTransform: "uppercase" }}>CPU Alarm Policy</div>
                  <div style={{ fontSize: "20px", fontWeight: "800", color: "var(--warning)", marginTop: "4px" }}>
                    {Math.round((policy?.cpu_threshold ?? 0.6) * 100)}%
                  </div>
                  <div style={{ fontSize: "11px", color: "var(--text-dim)", marginTop: "2px" }}>Spike Predictor probability trigger</div>
                </div>
                <div style={{ background: "var(--panel2)", border: "1px solid var(--border-soft)", borderRadius: "8px", padding: "12px" }}>
                  <div style={{ fontSize: "10px", color: "var(--text-faint)", fontWeight: "700", textTransform: "uppercase" }}>Memory Alarm Policy</div>
                  <div style={{ fontSize: "20px", fontWeight: "800", color: "#a78bfa", marginTop: "4px" }}>
                    {Math.round((policy?.memory_threshold ?? 0.7) * 100)}%
                  </div>
                  <div style={{ fontSize: "11px", color: "var(--text-dim)", marginTop: "2px" }}>MUP memory leak trigger</div>
                </div>
              </div>

              <div>
                <div style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", color: "var(--text-faint)", marginBottom: "6px" }}>Model Details</div>
                <table className="data-table" style={{ width: "100%", border: "1px solid var(--border-soft)", borderRadius: "8px" }}>
                  <tbody>
                    <tr>
                      <td style={{ fontWeight: "600", color: "var(--text-dim)" }}>Engine Classifier</td>
                      <td>Random Forest Classifier (scikit-learn 1.4.2)</td>
                    </tr>
                    <tr>
                      <td style={{ fontWeight: "600", color: "var(--text-dim)" }}>Features Attributed</td>
                      <td>10 Telemetry Signals (SHAPley Explainer)</td>
                    </tr>
                    <tr>
                      <td style={{ fontWeight: "600", color: "var(--text-dim)" }}>Inference Interval</td>
                      <td>1.0 Second Real-time Streaming Batch</td>
                    </tr>
                    <tr>
                      <td style={{ fontWeight: "600", color: "var(--text-dim)" }}>Training Dataset</td>
                      <td>final_research_dataset.csv (Component 2)</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div>
                <div style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", color: "var(--text-faint)", marginBottom: "6px" }}>Mitigation Escalation Tiers</div>
                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <div style={{ display: "flex", alignItems: "baseline", gap: "8px", fontSize: "12px" }}>
                    <span style={{ fontWeight: "700", color: "var(--healthy)" }}>TIER 0:</span>
                    <span style={{ color: "var(--text-dim)" }}>Nominal. Continuous observation, no action.</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "baseline", gap: "8px", fontSize: "12px" }}>
                    <span style={{ fontWeight: "700", color: "var(--warning)" }}>TIER 1:</span>
                    <span style={{ color: "var(--text-dim)" }}>Single alarm. Trigger load shedding or proactive pod restart.</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "baseline", gap: "8px", fontSize: "12px" }}>
                    <span style={{ fontWeight: "700", color: "var(--critical)" }}>TIER 2:</span>
                    <span style={{ color: "var(--text-dim)" }}>Joint alarm. Execute restart &amp; traffic reroute immediately.</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="modal-btn primary" onClick={() => setShowModelConfig(false)}>OK</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
