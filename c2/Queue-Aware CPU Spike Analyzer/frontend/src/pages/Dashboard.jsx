import { useCallback, useEffect, useRef, useState } from "react";
import { api, store, pillClass, stateColor, toast } from "../api.js";
import { drawStackedBars, hitTestStacked, stackedOrder, STACK_COLORS } from "../charts.js";
import heroImg from "../assets/cpu-dash.jpg";
import QuickAdd from "../components/QuickAdd.jsx";
import LineChart from "../components/LineChart.jsx";
import { drawAreaChart } from "../charts.js";

/* Future: paste the Azure main-web URL here to make "Back to main" navigate there. */
const MAIN_PORTAL_URL = "";

const PINNED_WINDOW = 30;

function pushHist(bucket, name, val, max = 120) {
  if (!bucket[name]) bucket[name] = [];
  bucket[name].push(val);
  if (bucket[name].length > max) bucket[name].shift();
}

export default function Dashboard({ query, refreshSignal, onNav, onStatus, onNavigateConfig }) {
  const [data, setData] = useState(null);
  const [kpis, setKpis] = useState({ services: "—", cpu: "—", cpuSub: "—", qpi: "—", err: "—" });
  const [rows, setRows] = useState([]);
  const [error, setError] = useState(null);
  const [redraw, setRedraw] = useState(0);
  const [showTelemetry, setShowTelemetry] = useState(false);
  const [stackedTip, setStackedTip] = useState(null);
  const [showStackedSvcs, setShowStackedSvcs] = useState(false);

  const cpuHist = useRef({});
  const qHist = useRef({});
  const stacked = useRef([]);
  const timer = useRef(null);
  const stopped = useRef(false);
  const onStatusRef = useRef(onStatus);
  onStatusRef.current = onStatus;

  const stackedRef = useRef(null);

  const poll = useCallback(async () => {
    try {
      const d = await api("/api/data");
      onStatusRef.current(true);
      setError(null);

      const services = d.services || {};
      const names = Object.keys(services);

      let peakCpu = null, peakSvc = "—", sumQpi = 0, nQpi = 0, errs = 0;
      let fallbackCpu = 0, fallbackSvc = "—", pinnedCount = 0;
      names.forEach((n) => {
        const r = services[n] || {};
        const cpu = parseFloat(r.cpu_percent || 0);
        if (cpu > fallbackCpu) {
          fallbackCpu = cpu;
          fallbackSvc = n;
        }
        if (r.qpi != null && !r.queue_stale) {
          sumQpi += parseFloat(r.qpi);
          nQpi++;
        }
        errs += parseFloat(r.http_errors_per_sec || 0);
        pushHist(cpuHist.current, n, cpu);
        pushHist(qHist.current, n, parseFloat(r.in_flight_queue || 0));
        // A sensor bit-identical at exactly 100.0 for PINNED_WINDOW straight
        // ticks is stuck, not measuring — exclude it from the headline peak.
        const recent = (cpuHist.current[n] || []).slice(-PINNED_WINDOW);
        if (recent.length >= PINNED_WINDOW && recent.every((v) => v === 100)) {
          pinnedCount++;
          return;
        }
        if (peakCpu === null || cpu > peakCpu) {
          peakCpu = cpu;
          peakSvc = n;
        }
      });
      if (peakCpu === null) {
        peakCpu = fallbackCpu;
        peakSvc = names.length ? `${fallbackSvc} (stuck?)` : "—";
      }
      setKpis({
        services: names.length,
        cpu: `${peakCpu.toFixed(1)}%`,
        cpuSub: pinnedCount ? `${peakSvc} · ${pinnedCount} stuck @100% ignored` : peakSvc,
        qpi: nQpi ? (sumQpi / nQpi).toFixed(3) : "—",
        err: `${errs.toFixed(1)}/s`,
      });
      setRows(names.map((n) => ({ name: n, ...(services[n] || {}) })));

      stacked.current.push({
        t: d.time || "",
        vals: Object.fromEntries(
          names.map((n) => [n, parseFloat((services[n] || {}).cpu_percent || 0)])
        ),
      });
      if (stacked.current.length > 60) stacked.current.shift();

      setData(d);
    } catch (e) {
      onStatusRef.current(false);
      setError(e.message);
    } finally {
      if (!stopped.current) {
        timer.current = setTimeout(poll, store.refreshMs);
      }
    }
  }, []);

  // mount: start poll loop; unmount: stop it
  useEffect(() => {
    stopped.current = false;
    poll();
    return () => {
      stopped.current = true;
      clearTimeout(timer.current);
    };
  }, [poll]);

  // external "refresh now" signal from the topbar
  const firstSignal = useRef(true);
  useEffect(() => {
    if (firstSignal.current) {
      firstSignal.current = false;
      return;
    }
    poll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshSignal]);

  // redraw the stacked bars on new data + window resize
  // (line charts redraw themselves via LineChart)
  useEffect(() => {
    if (stackedRef.current) drawStackedBars(stackedRef.current, stacked.current);
  }, [data, redraw]);

  useEffect(() => {
    const onResize = () => setRedraw((v) => v + 1);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // close the telemetry popup on Escape + lock background scroll while open
  useEffect(() => {
    if (!showTelemetry) return;
    const onKey = (e) => {
      if (e.key === "Escape") setShowTelemetry(false);
    };
    window.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [showTelemetry]);

  const onStackedMove = (e) => {
    const canvas = stackedRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const hit = hitTestStacked(canvas, e.clientX - rect.left, e.clientY - rect.top);
    if (!hit) {
      setStackedTip(null);
      return;
    }
    setStackedTip({
      name: hit.name,
      value: `${Math.round(hit.value * 100) / 100}`,
      color: hit.color,
      left: Math.min(Math.max(hit.x, 70), Math.max(rect.width - 70, 70)),
      top: Math.max(hit.yTop, 30),
    });
  };

  // services + colors for the stacked popup (latest tick, sorted by CPU)
  const stackedSvcs = (() => {
    const order = stackedOrder(stacked.current);
    const last = stacked.current[stacked.current.length - 1];
    if (!last) return [];
    return order
      .map((name, oi) => ({
        name,
        value: last.vals[name] || 0,
        color: STACK_COLORS[oi % STACK_COLORS.length],
      }))
      .sort((a, b) => b.value - a.value);
  })();

  const q = (query || "").toLowerCase();
  const visible = !q ? rows : rows.filter((r) => `${r.name} ${r.cloud_provider || ""}`.toLowerCase().includes(q));

  const stateOrder = ["FAILED", "DANGEROUS", "DEGRADED", "RECOVERING", "HEALTHY", "CALIBRATING"];
  const stateCounts = {};
  rows.forEach((r) => {
    const s = r.predicted_state || "unknown";
    stateCounts[s] = (stateCounts[s] || 0) + 1;
  });
  const shownStates = stateOrder.filter((s) => stateCounts[s] > 0);
  Object.keys(stateCounts).forEach((s) => {
    if (!shownStates.includes(s)) shownStates.push(s);
  });

  const state = data?.system_state || "—";
  const sUp = state.toUpperCase();
  const statusClass = !data
    ? "st-idle"
    : /FAILED|CRITICAL/.test(sUp)
      ? "st-failed"
      : /DANGEROUS/.test(sUp)
        ? "st-dangerous"
        : /DEGRADED|QUEUE_BOUND|CPU_BOUND/.test(sUp)
          ? "st-degraded"
          : /HEALTHY/.test(sUp)
            ? "st-healthy"
            : "st-calibrating";

  return (
    <section className="page active" id="page-dashboard">
      <div className="page-head">
        <h2>Queue-Aware CPU Analyzer</h2>
        <div className="spacer" />
        <button
          className="btn small"
          onClick={() => {
            if (MAIN_PORTAL_URL) window.open(MAIN_PORTAL_URL, "_blank", "noopener");
            else toast("Main portal link is not configured yet.", "err");
          }}
        >
          Back to main
        </button>
      </div>
      <div className="dash-grid">
        <div className="dash-col">
          <div className={`card status-card ${statusClass}`}>
            <h3>System status</h3>
            <div className="dash-state">
              {error ? (
                <>
                  <span style={{ color: "#fff" }}>Backend unreachable: {error}</span>
                  <small>
                    Check Settings → Backend URL. The backend must be running (python realtime_experiment.py).
                  </small>
                </>
              ) : !data ? (
                <span className="muted" style={{ color: "rgba(255,255,255,.85)" }}>Connecting to backend…</span>
              ) : (
                <>
                  System: <span className="status-pill">{state}</span>
                  &nbsp;&nbsp;|&nbsp;&nbsp;Workload: {data.workload_state || "—"}
                  <small>
                    {data.time || ""} — {rows.length} service(s) reporting · auto-refresh {store.refreshMs} ms
                  </small>
                </>
              )}
            </div>
          </div>
          <div className="card">
            <h3>Key metrics</h3>
            <div className="kpi-grid">
              <div className="kpi k1 g-blue">
                <span className="chip">▨</span>
                <div>
                  <small>Services</small>
                  <b>{kpis.services}</b>
                </div>
              </div>
              <div className="kpi k2">
                <span className="chip">◉</span>
                <div>
                  <small>Peak CPU</small>
                  <b>{kpis.cpu}</b>
                  <small className="sub">{kpis.cpuSub}</small>
                </div>
              </div>
              <div className="kpi k3 g-teal">
                <span className="chip">≋</span>
                <div>
                  <small>Avg QPI</small>
                  <b>{kpis.qpi}</b>
                </div>
              </div>
              <div className="kpi k4">
                <span className="chip">⚠</span>
                <div>
                  <small>HTTP errors</small>
                  <b>{kpis.err}</b>
                </div>
              </div>
            </div>
          </div>
          <div className="card">
            <h3>Service health</h3>
            {rows.length ? (
              shownStates.map((s) => (
                <div className="health-row" key={s}>
                  <span className={pillClass(s)}>{s}</span>
                  <span className="health-bar">
                    <i style={{ width: `${(100 * stateCounts[s]) / rows.length}%`, background: stateColor(s) }} />
                  </span>
                  <b className="health-count">{stateCounts[s]}</b>
                </div>
              ))
            ) : (
              <span className="muted">Waiting for telemetry…</span>
            )}
          </div>
        </div>

        <div className="dash-col dash-main">
          <div className="card hero-card">
            <img src={heroImg} alt="Cloud infrastructure telemetry" />
            <div className="hero-cap">
              <div>
                <h3>Live cloud telemetry</h3>
                <p className="muted">
                  {data ? `${rows.length} service(s) reporting · ${data.time || ""} · ${state}` : "Connecting to backend…"}
                </p>
              </div>
              <div className="row">
                <button className="btn small primary" id="liveTelemetryBtn" onClick={() => setShowTelemetry(true)}>
                  Live telemetry
                </button>
              </div>
            </div>
          </div>

          <div className="card chart-card">
            <div className="chart-head">
              <h3>Consumed CPU by service</h3>
              <button className="btn small" id="stackedServicesBtn" onClick={() => setShowStackedSvcs(true)}>
                View services
              </button>
            </div>
            <p className="chart-sub">
              Each bar is one poll tick; segments are per-service CPU and bar height is total consumed CPU.
            </p>
            <div className="chart-wrap">
              <canvas
                className="chart"
                id="chartStacked"
                ref={stackedRef}
                onMouseMove={onStackedMove}
                onMouseLeave={() => setStackedTip(null)}
              />
              {stackedTip && (
                <div className="chart-tip" style={{ left: stackedTip.left, top: stackedTip.top }}>
                  <span className="dot" style={{ background: stackedTip.color }} />
                  <b>{stackedTip.name}</b>
                  <span>{stackedTip.value}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="dash-col dash-side">
          <QuickAdd onNavigateConfig={onNavigateConfig} />
          <div className="card">
            <h3>Backend</h3>
            <BackendRow />
            <div className="row" style={{ marginTop: 10 }}>
              <button className="btn small ghost" onClick={() => onNav("configuration")}>
                Connection settings
              </button>
            </div>
          </div>
        </div>

        <div className="grid two dash-full">
          <div className="card chart-card">
            <div className="chart-head">
              <h3>Live CPU utilization by service</h3>
              <span className="live-dot" />
            </div>
            <p className="chart-sub">Percent of allocated compute, per poll tick</p>
            <LineChart bucket={cpuHist.current} opts={{ max: 100, unit: "%", colorSet: 0 }} tick={data} />
          </div>
          <div className="card chart-card">
            <div className="chart-head">
              <h3>In-flight queue length</h3>
              <span className="live-dot" />
            </div>
            <p className="chart-sub">Queued requests waiting for a worker</p>
            <LineChart bucket={qHist.current} opts={{ unit: "", colorSet: 1 }} tick={data} draw={drawAreaChart} />
          </div>
        </div>
      </div>

      {showStackedSvcs && (
        <div className="modal-backdrop" onClick={() => setShowStackedSvcs(false)}>
          <div
            className="modal modal-narrow"
            role="dialog"
            aria-modal="true"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-head">
              <h3>Services in this chart</h3>
              <button className="icon-btn modal-close" onClick={() => setShowStackedSvcs(false)} title="Close">
                ✕
              </button>
            </div>
            <p className="muted" style={{ margin: "0 0 12px" }}>
              {stackedSvcs.length} service(s) · colors match the stacked bars · values from the latest tick
            </p>
            <div className="svc-list">
              {stackedSvcs.length ? (
                stackedSvcs.map((s) => (
                  <div className="svc-list-row" key={s.name}>
                    <i className="dot" style={{ background: s.color }} />
                    <span className="mono">{s.name}</span>
                    <b>{s.value.toFixed(1)}%</b>
                  </div>
                ))
              ) : (
                <span className="muted">Waiting for telemetry…</span>
              )}
            </div>
          </div>
        </div>
      )}

      {showTelemetry && (
        <div className="modal-backdrop" onClick={() => setShowTelemetry(false)}>
          <div className="modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h3>Services — live telemetry</h3>
              <button className="icon-btn modal-close" onClick={() => setShowTelemetry(false)} title="Close">
                ✕
              </button>
            </div>
            <p className="muted" style={{ margin: "0 0 12px" }}>
              {rows.length} service(s) reporting · updates every poll tick
              {q ? ` · filtered by “${query}”` : ""}
            </p>
            <div className="table-wrap modal-table">
              <table>
                <thead>
                  <tr>
                    <th>Service</th>
                    <th>Cloud</th>
                    <th>CPU %</th>
                    <th>Queue</th>
                    <th>In/s (λ)</th>
                    <th>Proc/s (μ)</th>
                    <th>Latency ms</th>
                    <th>QPI</th>
                    <th>State</th>
                  </tr>
                </thead>
                <tbody id="svcTableBody">
                  {visible.length ? (
                    visible.map((r) => (
                      <tr key={r.name}>
                        <td className="mono">{r.name}</td>
                        <td>{r.cloud_provider || ""}</td>
                        <td>{r.cpu_percent ?? ""}</td>
                        <td>{r.in_flight_queue ?? ""}</td>
                        <td>{r.incoming_rate ?? ""}</td>
                        <td>{r.processing_rate ?? ""}</td>
                        <td>{r.latency_ms ?? ""}</td>
                        <td>{r.qpi ?? ""}</td>
                        <td>
                          <span className={pillClass(r.predicted_state)}>{r.predicted_state || "unknown"}</span>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="9" className="muted">
                        {rows.length
                          ? "No services match the search."
                          : "No services reporting yet. Add a microservice link on the Services page."}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function BackendRow() {
  const [online, setOnline] = useState(null);
  useEffect(() => {
    let dead = false;
    api("/api/status")
      .then(() => !dead && setOnline(true))
      .catch(() => !dead && setOnline(false));
    return () => {
      dead = true;
    };
  }, []);
  return (
    <div className="stat-row">
      <span>Status</span>
      <span>
        <span className={`conn-dot${online ? " on" : online === false ? " off" : ""}`} />
        <span>{online ? "online" : online === false ? "unreachable" : "connecting…"}</span>
      </span>
    </div>
  );
}
