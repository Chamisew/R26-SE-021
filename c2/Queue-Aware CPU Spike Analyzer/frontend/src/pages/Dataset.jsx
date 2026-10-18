import { useCallback, useEffect, useState } from "react";
import { store } from "../api.js";

const PICK = [
  "time",
  "service",
  "cpu_percent",
  "in_flight_queue",
  "incoming_rate",
  "processing_rate",
  "qpi",
  "predicted_state",
  "actual_failure",
  "incident_phase",
];

export default function Dataset() {
  const [info, setInfo] = useState("Loading…");
  const [head, setHead] = useState([]);
  const [rows, setRows] = useState([]);
  const [failed, setFailed] = useState(null);
  const [stats, setStats] = useState({ total: "—", cols: "—", services: "—" });

  const load = useCallback(async () => {
    setFailed(null);
    setInfo("Loading dataset…");
    try {
      const res = await fetch(store.backend.replace(/\/$/, "") + "/api/csv");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      const lines = text.trim().split("\n");
      if (!lines.length) throw new Error("Empty dataset");
      const header = lines[0].split(",");
      const dataRows = lines.slice(-200).map((l) => l.split(","));
      const idx = PICK.map((c) => header.indexOf(c)).filter((i) => i >= 0);
      const showIdx = idx.length ? idx : header.map((_, i) => i).slice(0, 10);
      setHead(showIdx.map((i) => header[i]));
      setRows(dataRows.map((r) => showIdx.map((i) => r[i] ?? "")));
      const svcIdx = header.indexOf("service");
      const svcCount =
        svcIdx >= 0 ? new Set(dataRows.map((r) => r[svcIdx]).filter(Boolean)).size : "—";
      setStats({ total: lines.length - 1, cols: header.length, services: svcCount });
      setInfo(`showing last ${dataRows.length} rows (key columns) ·`);
    } catch (e) {
      setFailed(`Dataset not available yet: ${e.message}. Run the backend to start collecting telemetry.`);
      setHead([]);
      setRows([]);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const downloadCsv = () => {
    const a = document.createElement("a");
    a.href = store.backend.replace(/\/$/, "") + "/api/csv";
    a.download = "final_research_dataset.csv";
    a.click();
  };

  return (
    <section className="page active" id="page-dataset">
      <div className="page-head">
        <h2>Your Services data</h2>
        <div className="spacer" />
        <button className="btn small" onClick={load}>
          Reload preview
        </button>
      </div>
      <div className="stat-tiles">
        <div className="kpi k1 g-blue">
          <span className="chip">▨</span>
          <div>
            <small>Total rows</small>
            <b>{stats.total}</b>
          </div>
        </div>
        <div className="kpi k2">
          <span className="chip">◉</span>
          <div>
            <small>Columns</small>
            <b>{stats.cols}</b>
          </div>
        </div>
        <div className="kpi k3 g-teal">
          <span className="chip">≋</span>
          <div>
            <small>Services in view</small>
            <b>{stats.services}</b>
          </div>
        </div>
      </div>
      <div className="card chart-card">
        <div className="chart-head">
          <h3>Latest telemetry rows</h3>
          <span className="live-dot" />
        </div>
        <p className="muted">
          {failed ? (
            <span className="muted">{failed}</span>
          ) : (
            <>
              {info} <button className="btn small" onClick={downloadCsv}>Download full CSV</button>
            </>
          )}
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {head.map((c) => (
                  <th key={c}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  {r.map((v, j) => (
                    <td key={j}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
