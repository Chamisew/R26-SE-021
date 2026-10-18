import { useCallback, useEffect, useState } from "react";
import { api, toast } from "../api.js";
import StackIcon, { stackOf, STACK_NAMES } from "../components/StackIcon.jsx";
import ConnectionForm from "../components/ConnectionForm.jsx";

/* pickup: null | "waiting" | "ready" | "stale" — set by the Services add flow */
export default function Configuration({ highlight, pickup, pickupName }) {
  const [config, setConfig] = useState(null);
  const [error, setError] = useState(null);

  const loadConfig = useCallback(async () => {
    try {
      const cfg = await api("/api/config");
      setConfig(cfg);
      setError(null);
    } catch (e) {
      setError("Could not load configuration: " + e.message);
    }
  }, []);

  useEffect(() => {
    loadConfig();
  }, [loadConfig, highlight, pickup]);

  const downloadConfig = () => {
    const blob = new Blob([JSON.stringify(config ? config.config : {}, null, 2)], {
      type: "application/json",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "services_config.json";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const copyConfig = () => {
    navigator.clipboard
      .writeText(JSON.stringify(config ? config.config : {}, null, 2))
      .then(() => toast("Configuration copied to clipboard.", "ok"))
      .catch((e) => toast("Copy failed: " + e.message, "err"));
  };

  const raw = JSON.stringify(config?.config || {}, null, 2);
  const lines = raw.split("\n");
  const svcCount = (config?.config?.services || []).length;
  const discoveredCount = (config?.discovered || []).length;
  const reportingCount = (config?.running || []).length;

  return (
    <section className="page active" id="page-configuration">
      <div className="page-head">
        <h2>Configuration</h2>
        <div className="spacer" />
        <button className="btn small" onClick={loadConfig}>
          Reload
        </button>
        <button className="btn small" onClick={copyConfig}>
          Copy JSON
        </button>
        <button className="btn small" onClick={downloadConfig}>
          Download
        </button>
      </div>
      <p className="muted" style={{ marginTop: -8 }}>
        This is the exact content of <span className="mono">services_config.json</span> — every service added on the
        Services tab is written here automatically.
      </p>
      <div className="stat-tiles">
        <div className="kpi k1 g-blue">
          <span className="chip">▨</span>
          <div>
            <small>Configured</small>
            <b>{config ? svcCount : "—"}</b>
          </div>
        </div>
        <div className="kpi k2">
          <span className="chip">◉</span>
          <div>
            <small>Discovered</small>
            <b>{config ? discoveredCount : "—"}</b>
          </div>
        </div>
        <div className="kpi k3 g-teal">
          <span className="chip">≋</span>
          <div>
            <small>Reporting</small>
            <b>{config ? reportingCount : "—"}</b>
          </div>
        </div>
      </div>
      <div>
        {pickup === "waiting" && (
          <div className="alert info">
            Waiting for backend discovery to pick up <b>{pickupName}</b> (polls services_config.json every ~10s)…
          </div>
        )}
        {pickup === "ready" && (
          <div className="alert ok">
            Backend is now monitoring <b>{pickupName}</b> — no restart was needed. See it live on the Dashboard.
          </div>
        )}
        {pickup === "stale" && (
          <div className="alert warn">
            Still not discovered after ~90s. Check that the service is <b>enabled</b> and the link is reachable.
          </div>
        )}
      </div>
      <div className="card chart-card">
        <div className="chart-head">
          <h3>Discovery state</h3>
          <span className="live-dot" />
        </div>
        <div>
          {config ? (
            (config.discovered || []).length ? (
              <div className="disc-list">
                {(config.discovered || []).map((n) => {
                  const reporting = (config.running || []).includes(n);
                  const svc = (config.config?.services || []).find((s) => s.name === n);
                  // build stack first (link keyword, then live-detected runtime);
                  // deployed platform only as the last-resort icon.
                  const kw = stackOf(svc);
                  const live = (config.stacks || {})[n];
                  const stack = kw !== "other" ? kw : live || "other";
                  const plat = String(svc?.platform || "other").toLowerCase();
                  const sub = stack !== "other" ? STACK_NAMES[stack] || stack : plat;
                  return (
                    <div className="disc-row" key={n}>
                      <StackIcon stack={stack} platform={plat} />
                      <div className="disc-main">
                        <div className="disc-name">{n}</div>
                        <div className="disc-sub">
                          {sub} · {reporting ? "reporting telemetry" : "discovered, waiting for data"}
                        </div>
                      </div>
                      <span
                        className={`conn-dot${reporting ? " on" : ""}`}
                        title={reporting ? "reporting" : "not reporting"}
                      />
                    </div>
                  );
                })}
              </div>
            ) : (
              <span className="muted">none</span>
            )
          ) : (
            <span className="muted">Loading…</span>
          )}
        </div>
        <p className="muted" style={{ marginBottom: 0 }}>
          {config && (
            <>
              File: <span className="mono">{config.file || "services_config.json"}</span> · {svcCount} service(s) ·
              discovered: <b>{(config.discovered || []).length}</b> · reporting:{" "}
              <b>{(config.running || []).length}</b>
              <br />
              <span className="muted">{config.auto_update_note || ""}</span>
            </>
          )}
        </p>
      </div>
      <div className="card chart-card">
        <div className="chart-head">
          <h3>services_config.json (live)</h3>
          <span className="live-dot" />
        </div>
        <pre className="json" style={{ marginTop: 10 }}>
          {error
            ? error
            : lines.map((ln, i) => (
                <span key={i} className={highlight && ln.includes(highlight) ? "hl" : undefined}>
                  {ln}
                  {"\n"}
                </span>
              ))}
        </pre>
      </div>
      <div className="card chart-card">
        <div className="chart-head">
          <h3>Connection</h3>
          <span className="live-dot" />
        </div>
        <div style={{ marginTop: 10 }}>
          <ConnectionForm />
        </div>
      </div>
    </section>
  );
}
