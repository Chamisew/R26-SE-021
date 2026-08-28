import {
  fmtPct, fmtNum, pillClass, confidenceLabel,
} from "../utils.js";
import TrendChart from "./TrendChart.jsx";

export default function ServiceDetail({ service: s }) {
  const conf = confidenceLabel(s.cpu_probability);
  const actLabel = s.action || "NO_ACTION";
  const actSev = (actionSevMap[actLabel]) || "none";
  const lifecycleSteps = ["Detected", "Investigating", "Action Recommended", "Approved", "Executed", "Verified", "Resolved"];
  const currentStep = s.risk === "HEALTHY" ? -1 : 2;

  return (
    <div className="detail">
      <div className="detail-grid">
        <div>
          <div className="detail-block">
            <h4>Why this service is at risk</h4>
            <div className="narrative" dangerouslySetInnerHTML={{ __html: evidenceNarrative(s) }} />
          </div>
          <div className="detail-block">
            <h4>CPU failure-probability trend ({(s.history || []).length} historical samples)</h4>
            <TrendChart
              values={(s.history || []).map(h => h.probability)}
              color="var(--critical)"
              label="Predicted failure probability over time"
              thresholdFrac={s.__cpuThreshold}
            />
          </div>
          <div className="detail-block">
            <h4>CPU utilization &amp; queue pressure trend</h4>
            <TrendChart values={(s.history || []).map(h => h.cpu)} color="var(--accent)" label="CPU utilization (%)" />
            <TrendChart values={(s.history || []).map(h => h.queue)} color="var(--warning)" label="Queue pressure index" />
          </div>
          <div className="detail-block">
            <h4>Top contributing signals — real Shapley attribution (XAI engine)</h4>
            {signalRows(s)}
          </div>
          <div className="detail-block">
            <h4>Root cause analysis (XAI engine)</h4>
            {rcaChain(s)}
          </div>
          <div className="detail-block">
            <h4>SRE runbook — engine-generated steps</h4>
            {runbookHtml(s)}
          </div>
          <div className="detail-block">
            <h4>Evidence timeline</h4>
            <div className="timeline">{timelineRows(s)}</div>
          </div>
        </div>
        <div>
          <div className="detail-block">
            <h4>Combined failure risk</h4>
            <div className="rec-card">
              <div className="rec-grid">
                <div>
                  <div className="k">CPU alarm</div>
                  <div className="v" style={{ color: s.cpu_alarm ? "var(--critical)" : "var(--healthy)" }}>
                    {s.cpu_alarm ? "ACTIVE" : "inactive"}
                  </div>
                </div>
                <div>
                  <div className="k">Memory alarm</div>
                  <div className="v" style={{ color: s.memory_alarm ? "var(--critical)" : "var(--healthy)" }}>
                    {s.memory_alarm ? "ACTIVE" : "inactive"}
                  </div>
                </div>
                <div>
                  <div className="k">Combined risk</div>
                  <div className="v"><span className={`pill ${pillClass(s.risk)}`}>{s.risk}</span></div>
                </div>
                <div>
                  <div className="k">Signal confidence</div>
                  <div className="v">{conf.label} <span style={{ color: "var(--text-faint)" }}>({conf.note})</span></div>
                </div>
              </div>
            </div>
          </div>
          <div className="detail-block">
            <h4>Recommended SRE action</h4>
            <div className="rec-card">
              <div className={`rec-title ${actSev}`}>{actLabel.replaceAll("_", " ")}</div>
              <div className="rec-grid">
                <div style={{ gridColumn: "1 / -1" }}>
                  <div className="k">Why</div>
                  <div className="v">{recWhy(actLabel)}</div>
                </div>
                <div>
                  <div className="k">Expected objective</div>
                  <div className="v">{recObjective(actLabel)}</div>
                </div>
                <div>
                  <div className="k">Requires human approval?</div>
                  <div className="v">{actLabel === "NO_ACTION" ? "No" : "Yes"}</div>
                </div>
                <div>
                  <div className="k">Risk of taking action</div>
                  <div className="v">{recRiskOfAction(actLabel)}</div>
                </div>
                <div>
                  <div className="k">Risk of doing nothing</div>
                  <div className="v">{recRiskOfInaction(actLabel)}</div>
                </div>
              </div>
            </div>
          </div>
          <div className="detail-block">
            <h4>Autonomous self-healing — mitigation engine</h4>
            {mitigationCard(s)}
          </div>
          <div className="detail-block">
            <h4>Human-in-the-loop lifecycle</h4>
            <div className="hitl-row">
              <span className={`hitl-badge ${s.risk === "HEALTHY" ? "no" : "done"}`}>Recommended</span>
              <span>{s.risk === "HEALTHY" ? "No action recommended." : "Mitigation recommendation generated by the response engine."}</span>
            </div>
            <div className="hitl-row">
              <span className="hitl-badge pending">Approved</span>
              <span>Awaiting SRE approval — not yet actioned.</span>
            </div>
            <div className="lifecycle">
              {lifecycleSteps.map((st, i) => {
                let c = "lc-step";
                if (currentStep >= 0) {
                  if (i < currentStep) c += " done";
                  else if (i === currentStep) c += " current";
                }
                return <span key={i} className={c}>{st}</span>;
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

const actionSevMap = {
  "CRITICAL_RESTART_AND_TRAFFIC_REROUTE": "critical",
  "TRIGGER_PROACTIVE_POD_RESTART": "critical",
  "TRIGGER_LOAD_SHEDDING": "warning",
  "SCALE_OUT": "warning",
  "INCREASE_OBSERVABILITY": "warning",
  "MONITOR": "warning",
  "NO_ACTION": "none",
};

function evidenceNarrative(s) {
  const narrative = s.rca_narrative || s.warning_message ||
    `The model reports <b>${fmtPct(s.cpu_probability)}</b> CPU-failure probability and <b>${fmtPct(s.memory_probability)}</b> memory-failure probability. Current CPU evidence is ${s.evidence?.cpu == null ? "unavailable" : fmtNum(s.evidence.cpu) + "%"}. Combined risk assessment: <b>${s.risk}</b>.`;
  return narrative.replace(/\bb\b/g, "b");
}

function signalRows(s) {
  const contributions = s.feature_contributions || {};
  const phis = s.shapley_phi_values || {};
  const entries = Object.entries(contributions)
    .filter(([, value]) => Number.isFinite(Number(value)))
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, 7);
  if (!entries.length) {
    return <div className="small-caveat">No live Shapley attribution is available for this service.</div>;
  }
  return (
    <>
      {entries.map(([name, value]) => {
        const pct = Math.min(100, Math.max(0, Number(value)));
        return (
          <div className="signal-row" key={name}>
            <div className="sig-label">{name.replaceAll("_", " ")}</div>
            <div className="sig-bar-track">
              <div className="sig-bar-fill" style={{ width: `${pct}%` }}></div>
            </div>
            <div className="sig-val">{fmtNum(value)}% <span style={{ color: "var(--text-faint)" }}>(φ={fmtNum(phis[name], 3)})</span></div>
          </div>
        );
      })}
      <div className="small-caveat" style={{ marginTop: "6px" }}>
        φ (phi) is the model's own Shapley additive value for this feature. Computed live against the trained model, not estimated.
      </div>
    </>
  );
}

function rcaChain(s) {
  const ticket = s.incident_ticket_payload || {};
  return (
    <>
      <div className="rec-grid" style={{ marginBottom: "8px" }}>
        <div>
          <div className="k">Severity</div>
          <div className="v"><span className={`pill ${pillClass(s.risk)}`}>{ticket.severity || s.risk}</span></div>
        </div>
        <div>
          <div className="k">Impact radius</div>
          <div className="v">{ticket.impact_radius || "Unavailable"}</div>
        </div>
        <div>
          <div className="k">Primary root cause</div>
          <div className="v">{ticket.primary_root_cause || "Unavailable"}</div>
        </div>
        <div>
          <div className="k">Secondary root cause</div>
          <div className="v">{ticket.secondary_root_cause || "Unavailable"}</div>
        </div>
      </div>
      <div className="rca-caveat">
        Attribution is the model's own feature-importance weighting applied to this sample's deviation from baseline — it explains the model's decision.
      </div>
    </>
  );
}

function runbookHtml(s) {
  const steps = s.sre_runbook_steps || [];
  if (!steps.length) {
    return <div className="small-caveat">No runbook steps were emitted for this service.</div>;
  }
  return (
    <div className="timeline">
      {steps.map((st, i) => (
        <div className="tl-row" key={i}>
          <div className="tl-time mono" style={{ width: "auto", color: "var(--accent)" }}>›</div>
          <div className="tl-text mono" style={{ fontSize: "11.5px" }}>{st}</div>
        </div>
      ))}
    </div>
  );
}

function mitigationCard(s) {
  const m = s.mitigation_executed || {};
  const isActive = m.action_executed && m.action_executed !== "NO_ACTION" && m.action_executed !== "SUPPRESSED_COOLDOWN_ACTIVE";
  const tier = m.mitigation_tier || m.policy_description || "No mitigation";
  return (
    <>
      <div className="rec-card">
        <div className={`rec-title ${isActive ? "critical" : "none"}`}>{String(tier).replaceAll("_", " ")}</div>
        <div className="rec-grid">
          <div>
            <div className="k">Engine decision</div>
            <div className="v">{(m.action_executed || "No action").replaceAll("_", " ")}</div>
          </div>
          <div>
            <div className="k">Circuit breaker</div>
            <div className="v">{m.circuit_breaker || "Unavailable"}</div>
          </div>
          <div style={{ gridColumn: "1 / -1" }}>
            <div className="k">Command that would be issued</div>
            <div className="v mono" style={{ fontSize: "11px", wordBreak: "break-all" }}>{m.command_issued || "— none —"}</div>
          </div>
          <div>
            <div className="k">Cooldown window</div>
            <div className="v">
              {m.cooldown_remaining_sec != null
                ? `${m.cooldown_remaining_sec}s between escalations`
                : "Unavailable"}
            </div>
          </div>
          <div>
            <div className="k">Engine-reported status</div>
            <div className="v" style={{ color: "var(--accent)" }}>{m.status || "Unavailable"}</div>
          </div>
        </div>
        <div className="rca-caveat" style={{ marginTop: "10px" }}>
          This is the mitigation state machine's own output. It decides tier, cooldown, and the exact command string. It does not itself call kubectl or Envoy.
        </div>
      </div>
      <div className="hitl-row" style={{ marginTop: "10px" }}>
        <span className="hitl-badge no">Executed</span>
        <span><b>Infrastructure execution status: Not executed.</b> No subprocess or cluster API call is made by this engine — the command above is generated and logged, not run.</span>
      </div>
    </>
  );
}

function timelineRows(s) {
  const pt = s.prediction_time ? new Date(s.prediction_time) : null;
  const fmt = (d) => (d ? d.toISOString().slice(11, 16) : "—");
  const base = pt ? new Date(pt.getTime()) : new Date();
  const t = (mins) => fmt(new Date(base.getTime() + mins * 60000));

  if (s.risk === "HEALTHY") {
    return (
      <div className="tl-row">
        <div className="tl-time">{fmt(pt)}</div>
        <div className="tl-text"><b>Telemetry</b> — nominal readings collected, no anomaly detected.</div>
      </div>
    );
  }
  return (
    <>
      <div className="tl-row"><div className="tl-time">{t(-3)}</div><div className="tl-text"><b>Telemetry</b> — CPU and queue metrics begin trending upward.</div></div>
      <div className="tl-row"><div className="tl-time">{t(-2)}</div><div className="tl-text"><b>Anomaly</b> — queue pressure crosses baseline variance.</div></div>
      <div className="tl-row"><div className="tl-time">{t(-1)}</div><div className="tl-text"><b>Risk increase</b> — CPU failure probability rises toward alarm threshold.</div></div>
      <div className="tl-row"><div className="tl-time">{fmt(pt)}</div><div className="tl-text"><b>Prediction</b> — CPU failure probability {fmtPct(s.cpu_probability)}, alarm {s.cpu_alarm ? "ACTIVE" : "inactive"}.</div></div>
      <div className="tl-row"><div className="tl-time">{fmt(pt)}</div><div className="tl-text"><b>SRE recommendation generated</b> — {(s.action || "NO_ACTION").replaceAll("_", " ")}.</div></div>
    </>
  );
}

function recWhy(action) {
  const map = {
    "NO_ACTION": "Telemetry is within normal range; no intervention is warranted.",
    "MONITOR": "An early signal was observed but has not crossed the alarm threshold. Continue observing.",
    "INCREASE_OBSERVABILITY": "Signal is ambiguous; additional telemetry resolution is needed before acting.",
    "TRIGGER_LOAD_SHEDDING": "CPU failure probability has crossed the alarm threshold under rising queue pressure. Shedding non-critical load reduces incoming pressure while capacity recovers.",
    "SCALE_OUT": "Sustained overload with recovering headroom elsewhere; horizontal scale-out spreads load across additional capacity.",
    "TRIGGER_PROACTIVE_POD_RESTART": "Degradation pattern matches a known recoverable state; a proactive restart can clear it before user-facing failure.",
    "CRITICAL_RESTART_AND_TRAFFIC_REROUTE": "Combined CPU and memory risk is critical with short warning time; traffic is rerouted while the instance is restarted to limit blast radius.",
  };
  return map[action] || "No mapped rationale for this action.";
}
function recObjective(action) {
  const map = {
    "NO_ACTION": "Maintain current healthy state.",
    "MONITOR": "Catch escalation early without disrupting traffic.",
    "INCREASE_OBSERVABILITY": "Improve signal quality for the next decision cycle.",
    "TRIGGER_LOAD_SHEDDING": "Reduce incoming pressure below processing capacity.",
    "SCALE_OUT": "Restore headroom by distributing load.",
    "TRIGGER_PROACTIVE_POD_RESTART": "Clear degraded internal state before user impact.",
    "CRITICAL_RESTART_AND_TRAFFIC_REROUTE": "Prevent user-facing failure while restoring the instance.",
  };
  return map[action] || "—";
}
function recRiskOfAction(action) {
  const map = {
    "NO_ACTION": "None — no action taken.",
    "MONITOR": "Minimal — passive observation only.",
    "INCREASE_OBSERVABILITY": "Minimal — added telemetry overhead only.",
    "TRIGGER_LOAD_SHEDDING": "Some requests may be rejected or delayed during shedding.",
    "SCALE_OUT": "Additional infrastructure cost; brief coordination overhead.",
    "TRIGGER_PROACTIVE_POD_RESTART": "Brief availability dip for the restarted instance.",
    "CRITICAL_RESTART_AND_TRAFFIC_REROUTE": "Short disruption during reroute; depends on downstream capacity.",
  };
  return map[action] || "—";
}
function recRiskOfInaction(action) {
  if (!action || action === "NO_ACTION" || action === "MONITOR" || action === "INCREASE_OBSERVABILITY") {
    return "Low — service is not currently in a failure trajectory.";
  }
  return "Elevated — sustained overload can progress to a user-facing outage without intervention.";
}
