import { useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import Topbar from "./components/Topbar.jsx";
import Toasts from "./components/Toasts.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Services from "./pages/Services.jsx";
import Configuration from "./pages/Configuration.jsx";
import Dataset from "./pages/Dataset.jsx";
import Experiments from "./pages/Experiments.jsx";

export default function App() {
  const [page, setPage] = useState("dashboard");
  const [query, setQuery] = useState("");
  const [refreshSignal, setRefreshSignal] = useState(0);
  const [connected, setConnected] = useState(null);
  const [highlight, setHighlight] = useState(null);
  const [pickup, setPickup] = useState(null);
  const [pickupName, setPickupName] = useState("");

  const nav = (p) => {
    setPage(p);
    if (window.innerWidth < 860) window.scrollTo({ top: 0 });
  };

  /* Services add flow: highlight the new row on the Configuration page and
     surface the backend discovery pickup state there. */
  const handleNavigateConfig = (name, state) => {
    setHighlight(name);
    setPickupName(name);
    setPickup(state);
    setPage("configuration");
  };

  return (
    <div id="app">
      <Sidebar page={page} onNav={nav} connected={connected} />
      <main className="main">
        <Topbar
          query={query}
          onQuery={setQuery}
          onRefresh={() => setRefreshSignal((v) => v + 1)}
          onSettings={() => nav("configuration")}
        />
        {page === "dashboard" && (
          <Dashboard
            query={query}
            refreshSignal={refreshSignal}
            onNav={nav}
            onStatus={setConnected}
            onNavigateConfig={handleNavigateConfig}
          />
        )}
        {page === "services" && <Services onNavigateConfig={handleNavigateConfig} />}
        {page === "configuration" && <Configuration highlight={highlight} pickup={pickup} pickupName={pickupName} />}
        {page === "dataset" && <Dataset />}
        {page === "experiments" && <Experiments />}
        <footer className="page-foot">Queue-Aware CPU Spike Analyzer · live telemetry console</footer>
      </main>
      <Toasts />
    </div>
  );
}
