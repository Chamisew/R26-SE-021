import { useCallback, useEffect, useRef, useState } from "react";
import { api, store } from "../api.js";
import { drawMultiLine } from "../charts.js";

export default function Benchmark() {
  const [status, setStatus] = useState("Loading cached results…");
  const [results, setResults] = useState([]);
  const [log, setLog] = useState("—");
  const [chartKind, setChartKind] = useState(null);
  const [png, setPng] = useState(null);
  const idleChartStates = () => ({
    empirical: { state: "idle", detail: "Not rendered yet." },
    ablation: { state: "idle", detail: "Not rendered yet." },
    live: { state: "idle", detail: "Not rendered yet." },
  });
  const [chartStates, setChartStates] = useState(idleChartStates);
  const setChartState = (which, patch) =>
    setChartStates((prev) => ({ ...prev, [which]: { ...prev[which], ...patch } }));
  const chartStatePill = (s) =>
    ({ idle: "p-unknown", loading: "p-calibrating", ready: "p-healthy", error: "p-failed" }[s] || "p-unknown");
  const chartLabels = { empirical: "Empirical validation", ablation: "QPI ablation", live: "Live validation" };
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const wasRunning = useRef(false);
  const researchRef = useRef(null);

  const best = results.reduce(
    (acc, m) => {
      const f1 = parseFloat(m.f1 ?? m.f1_score ?? NaN);
      const lead = parseFloat(m.mean_lead_s ?? m.lead_time ?? NaN);
      return {
        f1: !isNaN(f1) && (acc.f1 === null || f1 > acc.f1) ? f1 : acc.f1,
        lead: !isNaN(lead) && (acc.lead === null || lead > acc.lead) ? lead : acc.lead,
      };
    },
    { f1: null, lead: null }
  );
  const fmt4 = (v) => (typeof v === "number" && !isNaN(v) ? v.toFixed(4) : v ?? "");

  const renderBenchmark = (r) => {
    setResults(r.results || []);
    setLog(((r.raw_output || [r.empirical_output, r.feature_sets_output].filter(Boolean).join("\n\n")) || "").slice(0, 12000) || "—");
  };

  const runBenchmark = async () => {
    setStatus("Running live benchmark (empirical + feature-set models)… this can take a few minutes.");
    setBusy(true);
    try {
      const r = await api("/api/benchmark/run", { method: "POST", body: JSON.stringify({}) });
      renderBenchmark(r);
      setStatus(`Done at ${new Date((r.timestamp || Date.now() / 1000) * 1000).toLocaleTimeString()}.`);
    } catch (e) {
      setStatus(
        /already in progress/i.test(e.message)
          ? "A benchmark run is already in progress — results will load automatically when it finishes."
          : `Benchmark failed: ${e.message}`
      );
    } finally {
      setBusy(false);
    }
  };

  const loadCache = useCallback(async () => {
    try {
      const r = await api("/api/benchmark/results");
      renderBenchmark(r);
      setStatus("Showing cached results.");
    } catch (e) {
      setStatus(e.message);
    }
  }, []);

  // auto-load cache on open + track live runs across navigation
  useEffect(() => {
    const pollStatus = async () => {
      try {
        const s = await api("/api/benchmark/status");
        if (wasRunning.current && !s.running) loadCache();
        wasRunning.current = s.running;
        setRunning(s.running);
        setElapsed(s.elapsed || 0);
      } catch {
        /* backend unreachable — leave last state */
      }
    };
    loadCache();
    pollStatus();
    const t = setInterval(pollStatus, 5000);
    return () => clearInterval(t);
  }, [loadCache]);

  const loadChart = async (which) => {
    setChartKind(which);
    setPng(null);
    setChartState(which, { state: "loading", detail: `Fetching ${which} data from the backend…` });
    try {
      const r = await api(`/api/charts/data/${which}`);
      const data = r.data || [];
      if (!data.length) throw new Error("empty");
      const series = {};
      const numKeys = Object.keys(data[0]).filter(
        (k) => typeof data[0][k] === "number" && !/time|timestamp|errors|failure/i.test(k)
      );
      numKeys.slice(0, 4).forEach((k) => {
        series[k] = data.map((d) => d[k]);
      });
      // draw after paint
      requestAnimationFrame(() => {
        if (researchRef.current) drawMultiLine(researchRef.current, series, {});
      });
      setChartState(which, {
        state: "ready",
        detail: `Interactive · ${data.length} points · ${numKeys.slice(0, 4).join(", ")}${
          r.q_ref ? ` · Qref=${r.q_ref}` : ""
        }`,
      });
    } catch (e) {
      setPng(store.backend.replace(/\/$/, "") + `/api/charts/${which}?t=${Date.now()}`);
      setChartState(which, {
        state: "ready",
        detail: `Static PNG fallback — interactive data unavailable (${e.message}).`,
      });
    }
  };

  const generateCharts = async () => {
    (Object.keys(chartLabels) || []).forEach((k) =>
      setChartState(k, { state: "loading", detail: "Regenerating via empirical_validation.py… (takes minutes)" })
    );
    try {
      const r = await api("/api/charts/generate", { method: "POST", body: JSON.stringify({}) });
      const done = { state: "idle", detail: `${r.message || "Charts regenerated."} Click a chart to render it.` };
      setChartStates({ empirical: { ...done }, ablation: { ...done }, live: { ...done } });
    } catch (e) {
      const failed = { state: "error", detail: `Generation failed: ${e.message}` };
      setChartStates({ empirical: { ...failed }, ablation: { ...failed }, live: { ...failed } });
    }
  };

  return (
    <section className="page active" id="page-benchmark">
      <div className="page-head">
        <h2>Benchmark &amp; Research Charts</h2>
        <div className="spacer" />
        <button className="btn small" onClick={loadCache}>
          Load cached results
        </button>
        <button className="btn primary small" disabled={busy || running} onClick={runBenchmark}>
          Run live benchmark
        </button>
      </div>
      <p className="muted">
        {running
          ? `Live benchmark running (${elapsed}s elapsed)… you can browse other tabs; results load automatically when it finishes.`
          : status}
      </p>
      <div className="stat-tiles">
        <div className="kpi k1 g-blue">
          <span className="chip">▨</span>
          <div>
            <small>Models evaluated</small>
            <b>{results.length || "—"}</b>
          </div>
        </div>
        <div className="kpi k2">
          <span className="chip">◉</span>
          <div>
            <small>Best F1</small>
            <b>{best.f1 !== null ? best.f1.toFixed(3) : "—"}</b>
          </div>
        </div>
        <div className="kpi k3 g-teal">
          <span className="chip">≋</span>
          <div>
            <small>Best mean lead (s)</small>
            <b>{best.lead !== null ? best.lead.toFixed(1) : "—"}</b>
          </div>
        </div>
      </div>
      <div className="card chart-card">
        <div className="chart-head">
          <h3>Model comparison (A–E feature representations)</h3>
          <span className="live-dot" />
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Threshold</th>
              </tr>
            </thead>
            <tbody>
              {results.length ? (
                results.map((m, i) => (
                  <tr key={i}>
                    <td>{m.model || m.model_name || ""}</td>
                    <td>{fmt4(m.precision)}</td>
                    <td>{fmt4(m.recall)}</td>
                    <td>{fmt4(m.f1_score ?? m.f1)}</td>
                    <td>{fmt4(m.threshold)}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="5" className="muted">
                    Run the benchmark to populate.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
      <div className="card chart-card">
        <div className="chart-head">
          <h3>Terminal output</h3>
          <span className="live-dot" />
        </div>
        <div className="log" style={{ marginTop: 10 }}>{log}</div>
      </div>
      <div className="card chart-card">
        <div className="chart-head">
          <h3>Research charts</h3>
          <span className="live-dot" />
        </div>
        <div className="row" style={{ marginBottom: 10 }}>
          <button className="btn small" onClick={() => loadChart("empirical")}>
            Empirical validation
          </button>
          <button className="btn small" onClick={() => loadChart("ablation")}>
            QPI ablation
          </button>
          <button className="btn small" onClick={() => loadChart("live")}>
            Live validation
          </button>
          <button className="btn small" onClick={generateCharts}>
            Regenerate charts
          </button>
        </div>
        <div className="chart-status">
          {Object.keys(chartLabels).map((k) => (
            <div className="chart-status-row" key={k}>
              <span className={`pill ${chartStatePill(chartStates[k].state)}`}>{chartStates[k].state}</span>
              <div className="chart-status-main">
                <b>{chartLabels[k]}</b>
                <small>{chartStates[k].detail}</small>
              </div>
            </div>
          ))}
        </div>
        {chartKind && !png && <canvas className="chart" ref={researchRef} />}
        {png && (
          <img
            src={png}
            alt="research chart"
            style={{ maxWidth: "100%", border: "1px solid #e2e8f0", borderRadius: 10, marginTop: 10 }}
          />
        )}
      </div>
    </section>
  );
}
