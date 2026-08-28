import { useEffect, useRef, useState } from "react";
import Topbar from "./components/Topbar.jsx";
import OverviewCards from "./components/OverviewCards.jsx";
import RiskQueue from "./components/RiskQueue.jsx";
import ReliabilityTab from "./components/ReliabilityTab.jsx";
import { fetchDashboard, time } from "./utils.js";
import "./App.css";

export default function App() {
  const [page, setPage] = useState("command");
  const [dashboard, setDashboard] = useState(null);
  const [error, setError] = useState(null);
  const [freshText, setFreshText] = useState("Loading live outputs…");
  const loadedOnceRef = useRef(false);

  // Countdown state for CRITICAL alarms (used in RiskQueue table)
  const [countdowns, setCountdowns] = useState({});

  const load = async () => {
    try {
      const d = await fetchDashboard();
      window.__dashboardSnapshot = d;
      if (d.services && d.policy) {
        d.services.forEach(s => { s.__cpuThreshold = d.policy.cpu_threshold; });
      }
      setDashboard(d);
      setError(null);
      setFreshText(
        <>
          <span className="pulse-dot healthy"></span>Outputs refreshed {time(d.generated_at)}
        </>
      );
      loadedOnceRef.current = true;
    } catch (err) {
      setError(err.message || String(err));
      setDashboard(null);
      if (!loadedOnceRef.current) setFreshText("Could not load dashboard data.");
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 1000);
    return () => clearInterval(id);
  }, []);

  // ── Sync CRITICAL countdowns for RiskQueue ──────────────────────────────
  useEffect(() => {
    if (!dashboard?.services) return;

    setCountdowns(prev => {
      const next = { ...prev };
      let changed = false;
      const activeCritNames = new Set();

      dashboard.services.forEach(s => {
        if (s.risk === "CRITICAL") {
          activeCritNames.add(s.service);
          if (!next[s.service]) {
            const CRITICAL_COUNTDOWN_SECONDS = 5 * 60; // Fixed 5-minute countdown
            next[s.service] = {
              service: s.service,
              remainingSeconds: CRITICAL_COUNTDOWN_SECONDS,
              initialWarningMinutes: 5,
              turnedOff: false,
              action: s.action || "NO_ACTION"
            };
            changed = true;
          }
        }
      });

      for (const name in next) {
        if (!activeCritNames.has(name)) {
          delete next[name];
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [dashboard]);

  // ── Tick CRITICAL countdown every second ─────────────────────────────────
  useEffect(() => {
    const timer = setInterval(() => {
      setCountdowns(prev => {
        let changed = false;
        const next = {};
        for (const name in prev) {
          const item = prev[name];
          let newSecs = item.remainingSeconds;
          let newTurnedOff = item.turnedOff;
          if (item.remainingSeconds > 0) {
            newSecs = item.remainingSeconds - 1;
            changed = true;
            if (newSecs === 0) newTurnedOff = true;
          }
          next[name] = { ...item, remainingSeconds: newSecs, turnedOff: newTurnedOff };
        }
        return changed ? next : prev;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const onJumpToQueue = () => {
    const f = typeof window.__jumpToQueue === "function" ? window.__jumpToQueue : null;
    if (f) { f(); return; }
    const el = document.querySelector(".queue-toolbar");
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const onExport = () => {
    try {
      if (!dashboard) return;
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(dashboard, null, 2));
      const a = document.createElement("a");
      a.setAttribute("href", dataStr);
      const ts = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "_");
      a.setAttribute("download", `sre_command_center_snapshot_${ts}.json`);
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      alert("Failed to export report: " + err.message);
    }
  };

  if (error && !dashboard) {
    return (
      <>
        <Topbar page={page} setPage={setPage} freshText={freshText} dashboard={dashboard} />
        <div className="page">
          <div className="panel empty-note" style={{ padding: "40px" }}>
            Dashboard data gateway is unavailable. Run{" "}
            <span className="mono">python Dashboard/server.py</span>, then restart the React dev server.
            <div style={{ marginTop: "12px", color: "var(--text-dim)", fontSize: "12px" }}>Error: {error}</div>
          </div>
        </div>
      </>
    );
  }

  if (!dashboard) {
    return (
      <>
        <Topbar page={page} setPage={setPage} freshText={freshText} dashboard={dashboard} />
        <div className="page">
          <div className="panel empty-note" style={{ padding: "40px" }}>Loading dashboard from server…</div>
        </div>
      </>
    );
  }

  return (
    <>
      <Topbar page={page} setPage={setPage} freshText={freshText} dashboard={dashboard} />
      <div className="page">
        <OverviewCards
          summary={dashboard.summary}
          policy={dashboard.policy}
          services={dashboard.services}
          onJumpToQueue={onJumpToQueue}
          onExport={onExport}
          onGotoReliability={() => setPage("reliability")}
        />
        <div className="ov-section-gap"></div>
        {page === "command" ? (
          <RiskQueue
            services={dashboard.services}
            policy={dashboard.policy}
            countdowns={countdowns}
            setCountdowns={setCountdowns}
          />
        ) : (
          <>
            <div className="section-title"><span>Reliability report</span><span className="line"></span></div>
            <ReliabilityTab reliability={dashboard.reliability} />
          </>
        )}
      </div>
    </>
  );

}
