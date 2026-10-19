import { useCallback, useEffect, useRef, useState } from "react";
import { api, toast, pillClass } from "../api.js";

function platformGuess(name, url) {
  const blob = `${name} ${url}`.toLowerCase();
  if (blob.includes("azurecontainerapps.io") || blob.includes("azure")) return "azure";
  if (blob.includes("onrender.com") || blob.includes("render")) return "render";
  if (blob.includes("railway")) return "railway";
  if (/^https?:\/\/\d+\.\d+\.\d+\.\d+/.test(url.trim())) return "oracle";
  return "other";
}

export default function Services({ onNavigateConfig }) {
  const [services, setServices] = useState([]);
  const [baseUrl, setBaseUrl] = useState("");
  const [name, setName] = useState("");
  const [platform, setPlatform] = useState("other");
  const [platformTouched, setPlatformTouched] = useState(false);
  const [enabled, setEnabled] = useState(true);
  const [check, setCheck] = useState(null); // {tone, html} or {probing:true}
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState(1);
  const checkTimer = useRef(null);
  const discoveryTimer = useRef(null);

  const loadServices = useCallback(async (silent = false) => {
    try {
      const res = await api("/api/services");
      setServices(res.services || []);
    } catch (e) {
      if (!silent) toast("Could not load services: " + e.message, "err");
    }
  }, []);

  useEffect(() => {
    loadServices();
    return () => {
      clearTimeout(checkTimer.current);
      clearInterval(discoveryTimer.current);
    };
  }, [loadServices]);

  const probe = useCallback(
    async (url, nm) => {
      if (!url.trim()) return;
      setCheck({ probing: true });
      try {
        const r = await api("/api/services/check", {
          method: "POST",
          body: JSON.stringify({ base_url: url.trim(), name: nm.trim() }),
        });
        if (!r.valid) {
          setCheck({ tone: "err", text: `Invalid link: ${r.detail}` });
          return;
        }
        if (!nm.trim() && r.suggested_name) setName(r.suggested_name);
        setPlatform((p) => {
          if (!platformTouched && r.suggested_platform) return r.suggested_platform;
          return p;
        });
        const tone = /healthy|reachable/.test(r.status) ? "ok" : "warn";
        setCheck({ tone, text: `Link valid · live probe: ${r.status} — ${r.detail}`, bold: r.status });
      } catch (e) {
        setCheck({ tone: "err", text: `Check failed: ${e.message}` });
      }
    },
    [platformTouched]
  );

  const onLinkInput = (url) => {
    setBaseUrl(url);
    if (!platformTouched && url.trim()) setPlatform(platformGuess(name, url));
    clearTimeout(checkTimer.current);
    setCheck(null);
    if (url.trim().length < 8) return;
    checkTimer.current = setTimeout(() => probe(url, name), 700);
  };

  const suggestedPlaceholder = (() => {
    if (name.trim() || !baseUrl.trim()) return "e.g. payment-service";
    try {
      const host = new URL(baseUrl.trim()).hostname.split(".")[0].replace(/[^a-zA-Z0-9_-]/g, "-");
      if (host) return `Suggested name: ${host}`;
    } catch {
      /* invalid (yet) */
    }
    return "e.g. payment-service";
  })();

  const watchDiscovery = useCallback(
    (svcName) => {
      clearInterval(discoveryTimer.current);
      let tries = 0;
      onNavigateConfig(svcName, "waiting");
      discoveryTimer.current = setInterval(async () => {
        tries++;
        try {
          const cfg = await api("/api/config");
          const found = (cfg.discovered || []).includes(svcName) || (cfg.running || []).includes(svcName);
          if (found) {
            clearInterval(discoveryTimer.current);
            onNavigateConfig(svcName, "ready");
            setStep(3);
            toast(`${svcName} is now monitored.`, "ok");
          } else if (tries >= 9) {
            clearInterval(discoveryTimer.current);
            onNavigateConfig(svcName, "stale");
          }
        } catch {
          /* keep waiting */
        }
      }, 10000);
    },
    [onNavigateConfig]
  );

  const addService = async (ev) => {
    ev.preventDefault();
    if (!baseUrl.trim()) {
      toast("Paste a microservice link first.", "err");
      return;
    }
    setBusy(true);
    try {
      const r = await api("/api/services", {
        method: "POST",
        body: JSON.stringify({
          name: name.trim() || undefined,
          base_url: baseUrl.trim(),
          platform,
          enabled,
        }),
      });
      const svc = r.service || {};
      toast(`Registered ${svc.name} (${svc.status}). Opening configuration…`, "ok");
      setBaseUrl("");
      setName("");
      setPlatform("other");
      setPlatformTouched(false);
      setEnabled(true);
      setCheck(null);
      await loadServices(true);
      setStep(2);
      watchDiscovery(svc.name);
    } catch (e) {
      toast("Registration failed: " + e.message, "err");
    } finally {
      setBusy(false);
    }
  };

  const toggleService = async (svc) => {
    try {
      await api(`/api/services/${encodeURIComponent(svc.id)}`, {
        method: "PUT",
        body: JSON.stringify({ enabled: !svc.enabled }),
      });
      toast(`${svc.name} ${svc.enabled ? "disabled" : "enabled"}.`, "ok");
      loadServices(true);
    } catch (e) {
      toast("Update failed: " + e.message, "err");
    }
  };

  const editService = async (svc) => {
    const url = prompt(`New base URL for ${svc.name}:`, svc.base_url || "");
    if (url === null) return;
    try {
      await api(`/api/services/${encodeURIComponent(svc.id)}`, {
        method: "PUT",
        body: JSON.stringify({ base_url: url.trim() }),
      });
      toast(`${svc.name} URL updated.`, "ok");
      loadServices(true);
    } catch (e) {
      toast("Update failed: " + e.message, "err");
    }
  };

  const deleteService = async (svc) => {
    if (!confirm(`Delete ${svc.name}? Telemetry collection stops within ~10s.`)) return;
    try {
      await api(`/api/services/${encodeURIComponent(svc.id)}`, { method: "DELETE" });
      toast(`${svc.name} deleted.`, "ok");
      loadServices(true);
    } catch (e) {
      toast("Delete failed: " + e.message, "err");
    }
  };

  const enabledCount = services.filter((s) => s.enabled).length;
  const healthyCount = services.filter((s) => /healthy|reachable|ok/i.test(s.status || "")).length;

  return (
    <section className="page active" id="page-services">
      <div className="page-head">
        <h2>Microservices</h2>
        <div className="spacer" />
      </div>
      <p className="muted" style={{ marginTop: -8 }}>
        Paste a microservice link below — it is validated, probed live, written to{" "}
        <span className="mono">services_config.json</span>, and picked up by the backend automatically (no restart).
      </p>
      <div className="stat-tiles">
        <div className="kpi k1 g-blue">
          <span className="chip">▨</span>
          <div>
            <small>Registered</small>
            <b>{services.length}</b>
          </div>
        </div>
        <div className="kpi k2">
          <span className="chip">◉</span>
          <div>
            <small>Enabled</small>
            <b>{enabledCount}</b>
          </div>
        </div>
        <div className="kpi k3 g-teal">
          <span className="chip">≋</span>
          <div>
            <small>Probing healthy</small>
            <b>{healthyCount}</b>
          </div>
        </div>
      </div>
      <div className="card chart-card">
        <div className="chart-head">
          <h3>Add a microservice (3 steps)</h3>
          <span className="live-dot" />
        </div>
        <ol className="steps">
          <li className={step === 1 ? "active" : step > 1 ? "done" : ""}>
            <b>Paste link &amp; register</b> — fill the form and click “Add microservice”.
          </li>
          <li className={step === 2 ? "active" : step > 2 ? "done" : ""}>
            <b>Review configuration</b> — you are redirected to the Configuration tab; your service is highlighted.
          </li>
          <li className={step === 3 ? "active" : ""}>
            <b>Auto-update confirmed</b> — backend discovery picks it up within ~10s and telemetry appears on the
            Dashboard.
          </li>
        </ol>
        <form
          onSubmit={addService}
          onReset={() => {
            setBaseUrl("");
            setName("");
            setPlatform("other");
            setPlatformTouched(false);
            setEnabled(true);
            setCheck(null);
          }}
        >
          <div className="form-grid">
            <label className="field">
              Service link (base URL) *
              <input
                type="url"
                placeholder="https://my-service.onrender.com  or  http://140.245.2.114:8082"
                value={baseUrl}
                onChange={(e) => onLinkInput(e.target.value)}
                required
              />
              <span className="hint">Sidecar root URL exposing /metrics, /health, /ping, /spike.</span>
            </label>
            <label className="field">
              Name <span className="hint">(optional — auto-suggested from link)</span>
              <input type="text" placeholder={suggestedPlaceholder} value={name} onChange={(e) => setName(e.target.value)} />
            </label>
            <label className="field">
              Platform <span className="hint">(auto-detected, overridable)</span>
              <select
                value={platform}
                onChange={(e) => {
                  setPlatform(e.target.value);
                  setPlatformTouched(true);
                }}
              >
                <option value="azure">azure</option>
                <option value="render">render</option>
                <option value="oracle">oracle</option>
                <option value="railway">railway</option>
                <option value="other">other</option>
              </select>
            </label>
            <label className="field">
              State
              <span className="checkline">
                <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} /> Enabled
                (monitor immediately)
              </span>
            </label>
          </div>
          <div className="row" style={{ marginTop: 12 }}>
            <button type="button" className="btn" onClick={() => probe(baseUrl, name)}>
              Check link
            </button>
            <button type="submit" className="btn primary" disabled={busy}>
              {busy ? "Registering…" : "Add microservice"}
            </button>
          </div>
        </form>
        <div style={{ marginTop: 12 }}>
          {check?.probing && <span className="muted">Probing link…</span>}
          {check?.tone && <div className={`alert ${check.tone}`}>{check.text}</div>}
        </div>
      </div>
      <div className="card chart-card">
        <div className="chart-head">
          <h3>Registered services</h3>
          <span className="live-dot" />
        </div>
        <div className="row" style={{ marginBottom: 10, marginTop: 10 }}>
          <button className="btn small" onClick={() => loadServices()}>
            Refresh list
          </button>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>ID</th>
                <th>Platform</th>
                <th>Base URL</th>
                <th>Enabled</th>
                <th>Status</th>
                <th>Last checked</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {services.length ? (
                services.map((s) => (
                  <tr key={s.id}>
                    <td className="mono">{s.name}</td>
                    <td className="mono muted">{s.id || ""}</td>
                    <td>{s.platform || ""}</td>
                    <td className="mono" style={{ maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis" }}>
                      {s.base_url || ""}
                    </td>
                    <td>
                      {s.enabled ? (
                        <span className="pill p-enabled">enabled</span>
                      ) : (
                        <span className="pill p-disabled">disabled</span>
                      )}
                    </td>
                    <td>
                      <span className={pillClass(s.status)}>{s.status || "unknown"}</span>
                    </td>
                    <td className="mono muted" style={{ fontSize: 11 }}>
                      {s.last_checked || ""}
                    </td>
                    <td>
                      <div className="row" style={{ gap: 6 }}>
                        <button
                          className={`icon-btn sm${s.enabled ? " on" : ""}`}
                          title={s.enabled ? "Disable" : "Enable"}
                          onClick={() => toggleService(s)}
                        >
                          ⏻
                        </button>
                        <button className="icon-btn sm" title="Edit URL" onClick={() => editService(s)}>
                          ✎
                        </button>
                        <button
                          className="icon-btn sm danger"
                          title="Delete"
                          onClick={() => deleteService(s)}
                        >
                          ✕
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="8" className="muted">
                    No microservices registered yet — paste your first service link in the form above.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
