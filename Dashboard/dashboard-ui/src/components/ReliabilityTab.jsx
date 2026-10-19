import { fmtPct } from "../utils.js";

export default function ReliabilityTab({ reliability }) {
  if (!reliability) {
    return (
      <div className="panel empty-note" style={{ padding: "40px" }}>
        No reliability data is available from the server.
      </div>
    );
  }
  const folds = reliability.cpu_lopo || [];
  const memFolds = reliability.memory_lopo || [];
  const avg = (key) => folds.length ? (folds.reduce((a, f) => a + (f[key] || 0), 0) / folds.length) : 0;

  return (
    <>
      <div className="section-title"><span>CPU model — leave-one-project-out (LOPO) validation</span><span className="line"></span></div>
      <div className="metric-strip">
        <div className="metric-box"><div className="v">{fmtPct(avg("precision"))}</div><div className="l">Mean precision</div></div>
        <div className="metric-box"><div className="v">{fmtPct(avg("recall"))}</div><div className="l">Mean recall</div></div>
        <div className="metric-box"><div className="v">{fmtPct(avg("f1"))}</div><div className="l">Mean F1</div></div>
        <div className="metric-box"><div className="v">{fmtPct(avg("far"))}</div><div className="l">Mean false-alarm rate</div></div>
      </div>
      <div className="panel" style={{ padding: "6px 0", overflowX: "auto" }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Held-out project</th><th>Precision</th><th>Recall</th><th>F1</th><th>False-alarm rate</th>
            </tr>
          </thead>
          <tbody>
            {folds.length ? (
              folds.map((f, i) => (
                <tr key={i}>
                  <td>{f.project}</td>
                  <td>{fmtPct(f.precision)}</td>
                  <td>{fmtPct(f.recall)}</td>
                  <td>{fmtPct(f.f1)}</td>
                  <td>{fmtPct(f.far)}</td>
                </tr>
              ))
            ) : (
              <tr><td colSpan="5" className="empty-note">No CPU LOPO output is available.</td></tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="small-caveat">
        LOPO = model retrained with one project's traffic pattern fully held out, then evaluated on it — a proxy for how the model generalizes to a service it has never seen.
      </div>

      <div className="section-title"><span>Memory model (MUP)</span><span className="line"></span></div>
      <div className="panel" style={{ padding: "6px 0", overflowX: "auto" }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Held-out project</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC AUC</th>
            </tr>
          </thead>
          <tbody>
            {memFolds.length ? (
              memFolds.map((f, i) => (
                <tr key={i}>
                  <td>{f.project}</td>
                  <td>{fmtPct(f.precision)}</td>
                  <td>{fmtPct(f.recall)}</td>
                  <td>{fmtPct(f.f1)}</td>
                  <td>{fmtPct(f.auc)}</td>
                </tr>
              ))
            ) : (
              <tr><td colSpan="5" className="empty-note">No memory LOPO output is available.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </>
  );
}
