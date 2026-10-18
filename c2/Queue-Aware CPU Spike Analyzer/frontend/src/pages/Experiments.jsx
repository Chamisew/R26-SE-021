import { useCallback, useEffect, useState } from "react";
import { api, toast } from "../api.js";

export default function Experiments() {
  const [status, setStatus] = useState("Loading…");
  const [level, setLevel] = useState("medium");
  const [wlDuration, setWlDuration] = useState(120);
  const [wlRps, setWlRps] = useState("");
  const [spDuration, setSpDuration] = useState(60);
  const [spService, setSpService] = useState("");
  const [spAll, setSpAll] = useState(true);
  const [targets, setTargets] = useState([]);

  const loadStatus = useCallback(async () => {
    try {
      const st = await api("/api/experiments/status");
      setStatus({ ...st });
    } catch (e) {
      setStatus({ error: `Status unavailable: ${e.message}` });
    }
  }, []);

  const loadTargets = useCallback(async () => {
    try {
      const res = await api("/api/services");
      setTargets((res.services || []).filter((s) => s.enabled).map((s) => s.name));
    } catch {
      /* non-fatal */
    }
  }, []);

  useEffect(() => {
    loadStatus();
    loadTargets();
    const t = setInterval(loadStatus, 5000);
    return () => clearInterval(t);
  }, [loadStatus, loadTargets]);

  const startWorkload = async (ev) => {
    ev.preventDefault();
    const body = { level, duration: parseFloat(wlDuration) };
    if (String(wlRps).trim()) body.rps = parseFloat(wlRps);
    try {
      const r = await api("/api/experiments/workload", { method: "POST", body: JSON.stringify(body) });
      toast(`Workload started (pid ${r.pid}).`, "ok");
      loadStatus();
    } catch (e) {
      toast("Workload failed: " + e.message, "err");
    }
  };

  const startSpike = async (ev) => {
    ev.preventDefault();
    try {
      const r = await api("/api/experiments/spike", {
        method: "POST",
        body: JSON.stringify({ duration: parseFloat(spDuration), all: spAll, service: spService }),
      });
      toast(`Spike injected (pid ${r.pid}).`, "ok");
      loadStatus();
    } catch (e) {
      toast("Spike failed: " + e.message, "err");
    }
  };

  const fmt = (x) =>
    x && x.running ? (
      <span className="pill p-calibrating">running (pid {x.pid})</span>
    ) : (
      <span className="pill p-unknown">idle</span>
    );

  const running = typeof status === "object" && !status.error;
  const wl = running ? status.workload : null;
  const sp = running ? status.spike : null;
  const auto = running ? (status.spike_active ? status.incident_service : null) : null;

  return (
    <section className="page active" id="page-experiments">
      <div className="page-head">
        <h2>Experiments</h2>
        <div className="spacer" />
        <button className="btn small" onClick={loadStatus}>
          Refresh status
        </button>
      </div>
      {typeof status === "string" || status.error ? (
        <div className="card">
          <h3>Status</h3>
          <p style={{ margin: 0 }}>
            <span className="muted">{typeof status === "string" ? status : status.error}</span>
          </p>
        </div>
      ) : (
        <div className="stat-tiles">
          <div className="kpi k1 g-blue">
            <span className="chip">◉</span>
            <div>
              <small>Workload</small>
              <div style={{ marginTop: 2 }}>{fmt(wl)}</div>
            </div>
          </div>
          <div className="kpi k2">
            <span className="chip">⚡︎</span>
            <div>
              <small>Spike</small>
              <div style={{ marginTop: 2 }}>{fmt(sp)}</div>
            </div>
          </div>
          <div className="kpi k3 g-teal">
            <span className="chip">▨</span>
            <div>
              <small>Auto trigger</small>
              <div style={{ marginTop: 2 }}>
                {auto ? <b>{auto}</b> : <span className="pill p-unknown">idle</span>}
              </div>
            </div>
          </div>
        </div>
      )}
      <div className="grid two">
        <div className="card chart-card">
          <div className="chart-head">
            <h3>Workload generator (request traffic)</h3>
            <span className="live-dot" />
          </div>
          <form onSubmit={startWorkload}>
            <div className="form-grid">
              <label className="field">
                Level
                <select value={level} onChange={(e) => setLevel(e.target.value)}>
                  <option>low</option>
                  <option>medium</option>
                  <option>high</option>
                  <option>very_high</option>
                </select>
              </label>
              <label className="field">
                Duration (s)
                <input type="number" value={wlDuration} min="5" max="3600" onChange={(e) => setWlDuration(e.target.value)} />
              </label>
              <label className="field">
                Total RPS <span className="hint">(blank = auto from level)</span>
                <input
                  type="number"
                  placeholder="auto"
                  min="0"
                  max="1000"
                  value={wlRps}
                  onChange={(e) => setWlRps(e.target.value)}
                />
              </label>
            </div>
            <div className="row" style={{ marginTop: 12 }}>
              <button className="btn primary" type="submit">
                Start workload
              </button>
            </div>
          </form>
        </div>
        <div className="card chart-card">
          <div className="chart-head">
            <h3>CPU spike injector</h3>
            <span className="live-dot" />
          </div>
          <form onSubmit={startSpike}>
            <div className="form-grid">
              <label className="field">
                Duration (s)
                <input type="number" value={spDuration} min="5" max="3600" onChange={(e) => setSpDuration(e.target.value)} />
              </label>
              <label className="field">
                Target service
                <select value={spService} onChange={(e) => setSpService(e.target.value)}>
                  {targets.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                Scope
                <span className="checkline">
                  <input type="checkbox" checked={spAll} onChange={(e) => setSpAll(e.target.checked)} /> All enabled
                  services
                </span>
              </label>
            </div>
            <div className="row" style={{ marginTop: 12 }}>
              <button className="btn danger" type="submit">
                Inject CPU spike
              </button>
            </div>
          </form>
        </div>
      </div>
      <div className="card">
        <h3>Notes</h3>
        <p className="muted" style={{ margin: 0 }}>
          Experiments run server-side in the background — you can keep browsing while they execute. The automatic
          3-minute spike trigger (single / cascade / slow modes) continues independently once calibration (60 samples)
          completes.
        </p>
      </div>
    </section>
  );
}
