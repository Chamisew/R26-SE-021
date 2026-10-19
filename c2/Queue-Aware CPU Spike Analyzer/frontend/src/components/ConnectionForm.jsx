import { useState } from "react";
import { api, store } from "../api.js";

/* Backend connection form (lives on the Configuration page). */
export default function ConnectionForm() {
  const [backend, setBackend] = useState(store.backend);
  const [refresh, setRefresh] = useState(store.refreshMs);
  const [msgs, setMsgs] = useState([]);

  const save = async (ev) => {
    ev.preventDefault();
    const b = backend.trim().replace(/\/$/, "") || window.location.origin;
    const r = Math.min(Math.max(parseInt(refresh, 10) || 1000, 500), 10000);
    store.backend = b;
    store.refreshMs = r;
    const out = [`Settings saved. Polling every ${r} ms against ${b}.`];
    try {
      const st = await api("/api/status");
      out.push(`Backend says: ${st.status || "online"} (port ${st.port || ""})`);
    } catch (e) {
      out.push(`Backend not reachable at ${b}: ${e.message}`);
    }
    setMsgs(out);
  };

  return (
    <>
      <form onSubmit={save}>
        <div className="form-grid">
          <label className="field">
            Backend URL <span className="hint">(same origin by default — the server that served this page)</span>
            <input
              type="text"
              placeholder="http://localhost:8081"
              value={backend}
              onChange={(e) => setBackend(e.target.value)}
            />
          </label>
          <label className="field">
            Refresh interval (ms)
            <input
              type="number"
              min="500"
              max="10000"
              step="100"
              value={refresh}
              onChange={(e) => setRefresh(e.target.value)}
            />
          </label>
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          <button className="btn primary" type="submit">
            Apply
          </button>
        </div>
      </form>
      <div style={{ marginTop: 12 }}>
        {msgs.map((m, i) => (
          <div key={i} className={`alert ${/not reachable/i.test(m) ? "err" : /Backend says/i.test(m) ? "info" : "ok"}`}>
            {m}
          </div>
        ))}
      </div>
    </>
  );
}
