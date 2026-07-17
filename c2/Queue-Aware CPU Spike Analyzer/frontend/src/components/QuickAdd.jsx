import { useEffect, useRef, useState } from "react";
import { api, toast } from "../api.js";
import formImg from "../assets/form.jpg";

/* Newsletter-style quick-add card (right rail): paste a service link,
   register it, and hand off to the Configuration flow for pickup status. */
export default function QuickAdd({ onNavigateConfig }) {
  const [url, setUrl] = useState("");
  const [name, setName] = useState("");
  const [platform, setPlatform] = useState("other");
  const [enabled, setEnabled] = useState(true);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState(null);
  const watchTimer = useRef(null);

  useEffect(() => () => clearInterval(watchTimer.current), []);

  const watchDiscovery = (svcName) => {
    clearInterval(watchTimer.current);
    let tries = 0;
    onNavigateConfig(svcName, "waiting");
    watchTimer.current = setInterval(async () => {
      tries++;
      try {
        const cfg = await api("/api/config");
        const found = (cfg.discovered || []).includes(svcName) || (cfg.running || []).includes(svcName);
        if (found) {
          clearInterval(watchTimer.current);
          onNavigateConfig(svcName, "ready");
          toast(`${svcName} is now monitored.`, "ok");
        } else if (tries >= 9) {
          clearInterval(watchTimer.current);
          onNavigateConfig(svcName, "stale");
        }
      } catch {
        /* keep waiting */
      }
    }, 10000);
  };

  const submit = async (ev) => {
    ev.preventDefault();
    if (!url.trim()) {
      toast("Paste a microservice link first.", "err");
      return;
    }
    setBusy(true);
    setNote(null);
    try {
      const r = await api("/api/services", {
        method: "POST",
        body: JSON.stringify({
          base_url: url.trim(),
          name: name.trim() || undefined,
          platform,
          enabled,
        }),
      });
      const svc = r.service || {};
      setNote({ tone: "ok", text: `Registered ${svc.name} (${svc.status}). Opening configuration…` });
      toast(`Registered ${svc.name} (${svc.status}). Opening configuration…`, "ok");
      setUrl("");
      setName("");
      setPlatform("other");
      setEnabled(true);
      watchDiscovery(svc.name);
    } catch (e) {
      setNote({ tone: "err", text: e.message });
      toast("Registration failed: " + e.message, "err");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card signup-card">
      <img className="signup-img" src={formImg} alt="Monitor a new microservice" />
      <h3 className="signup-title">Monitoring a new service?</h3>
      <p className="signup-text">
        Paste its link below — it is probed live and picked up automatically. No restart needed.
      </p>
      <form onSubmit={submit}>
        <input
          className="signup-input"
          type="url"
          placeholder="https://my-service.onrender.com"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          required
        />
        <input
          className="signup-input"
          type="text"
          placeholder="Name (optional — auto-suggested)"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <div className="signup-row">
          <select className="signup-input signup-half" value={platform} onChange={(e) => setPlatform(e.target.value)}>
            <option value="azure">azure</option>
            <option value="render">render</option>
            <option value="oracle">oracle</option>
            <option value="railway">railway</option>
            <option value="other">other</option>
          </select>
          <label className="signup-check">
            <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} /> Enabled
          </label>
        </div>
        <button className="btn primary block signup-btn" type="submit" disabled={busy}>
          {busy ? "Adding…" : "Add service"}
        </button>
      </form>
      {note && <div className={`alert ${note.tone}`} style={{ marginTop: 10, marginBottom: 0 }}>{note.text}</div>}
    </div>
  );
}
