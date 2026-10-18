/* ==========================================================================
   Memory Failure Detection Dashboard — Application Logic
   ========================================================================== */

const DEFAULT_API_BASE = (window.location.origin && window.location.origin.startsWith("http"))
  ? window.location.origin
  : "http://localhost:8001";
let API_BASE = localStorage.getItem("c1_backend_url") || DEFAULT_API_BASE;
let currentTab = "overview";
let isSimulating = false;
let socket = null;
let pollTimer = null;
let configPollTimer = null;
let configRefreshMs = Number(localStorage.getItem("c1_config_refresh_ms")) || 1000;
let configFileSnapshot = {};

// Global State
let pipelineData = {
  status: "running",
  total_rows: 0,
  active_services: [],
  raw_metrics: [],
  templates: [],
  windows: [],
  stats: {},
  feed: [],
  memory_alerts: [],
  explanations: []
};

// Chart instances
let chartHeroMemory = null;
let chartMemoryTrend = null;
let chartHeapMemory = null;
let chartTrend = null;

// Init Application on Load
document.addEventListener("DOMContentLoaded", () => {
  reorderSidebarNavigation();
  initCharts();
  initConfigurationPanel();
  connectBackend();
  pollTimer = setInterval(fetchBackendState, 3000);
});

function reorderSidebarNavigation() {
  const nav = document.querySelector(".sidebar-nav");
  if (!nav) return;

  const itemsByTab = {};
  Array.from(nav.querySelectorAll(".nav-item")).forEach(item => {
    const match = item.getAttribute("onclick")?.match(/switchTab\('([^']+)'\)/);
    if (match) itemsByTab[match[1]] = item;
  });

  ["overview", "configuration", "memory", "windows"].forEach(tabId => {
    if (itemsByTab[tabId]) nav.appendChild(itemsByTab[tabId]);
  });
}

// Connect to FastAPI backend REST & WebSocket
function connectBackend() {
  fetchBackendState();
  connectWebSocket();
}

function openAzureSettings() {
  const modal = document.getElementById("azureSettingsModal");
  if (!modal) return;
  modal.hidden = false;
  loadAzureServices();
}

function closeAzureSettings() {
  const modal = document.getElementById("azureSettingsModal");
  if (modal) modal.hidden = true;
}

async function loadAzureServices() {
  const list = document.getElementById("azureServicesList");
  if (!list) return;
  try {
    const response = await fetch(`${API_BASE}/api/azure-services`);
    if (!response.ok) throw new Error("Unable to load services");
    const services = await response.json();
    list.innerHTML = services.length
      ? services.map(service => `
          <div class="azure-service-row" style="display:flex; justify-content:space-between; align-items:center; padding:0.6rem; margin-bottom:0.4rem; background:rgba(255,255,255,0.03); border-radius:6px; border:1px solid rgba(255,255,255,0.08);">
            <div>
              <strong style="color:#e2e8f0;">${escapeHtml(service.name)}</strong>
              <div style="font-size:0.8rem; color:#94a3b8; word-break:break-all;">${escapeHtml(service.base_url)}</div>
              <em style="font-size:0.75rem; color:${service.enabled ? '#4ade80' : '#f87171'}">${service.enabled ? "Active" : "Disabled"}</em>
            </div>
            <button class="action-btn" type="button" style="padding:0.3rem 0.6rem; font-size:0.75rem; color:#f87171; border:1px solid rgba(248,113,113,0.3); background:rgba(248,113,113,0.1); border-radius:4px; cursor:pointer;" onclick="deleteAzureService('${escapeHtml(service.name)}')">Delete</button>
          </div>
        `).join("")
      : "<p style='color:#94a3b8; font-size:0.85rem;'>No Azure services configured.</p>";
  } catch (error) {
    list.textContent = error.message;
  }
}

async function deleteAzureService(name) {
  if (!confirm(`Are you sure you want to remove '${name}' from services_config.json?`)) return;
  try {
    const response = await fetch(`${API_BASE}/api/azure-services/${encodeURIComponent(name)}`, {
      method: "DELETE"
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Unable to delete service");
    loadAzureServices();
    await fetchBackendState();
  } catch (error) {
    alert(error.message);
  }
}

async function saveAzureService(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const message = document.getElementById("azureSettingsMessage");
  const payload = {
    name: form.elements.name.value.trim(),
    base_url: form.elements.base_url.value.trim(),
    enabled: form.elements.enabled.checked
  };
  message.className = "settings-message";
  message.textContent = "Verifying /ping and /metrics...";
  try {
    const response = await fetch(`${API_BASE}/api/azure-services`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Unable to add Azure service");
    message.className = "settings-message success";
    message.textContent = result.message;
    form.reset();
    document.getElementById("azureServiceEnabled").checked = true;
    loadAzureServices();
    await fetchBackendState();
  } catch (error) {
    message.className = "settings-message error";
    message.textContent = error.message;
  }
}

function connectWebSocket() {
  let wsUrl;
  try {
    const loc = window.location;
    if (loc && loc.protocol && loc.protocol.startsWith("http")) {
      const proto = loc.protocol === "https:" ? "wss:" : "ws:";
      wsUrl = `${proto}//${loc.host}/ws/live`;
    } else {
      const host = API_BASE.replace(/^https?:\/\//, "");
      const proto = API_BASE.startsWith("https") ? "wss:" : "ws:";
      wsUrl = `${proto}//${host}/ws/live`;
    }
  } catch (e) {
    wsUrl = "ws://127.0.0.1:8001/ws/live";
  }

  try {
    socket = new WebSocket(wsUrl);
    socket.onopen = () => {
      console.log("WebSocket connected to", wsUrl);
      const dot = document.getElementById("statusDot");
      const txt = document.getElementById("statusText");
      if (dot) dot.className = "status-dot";
      if (txt) txt.innerText = "Live Ingestion";
    };
    socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "metrics_update" && Array.isArray(msg.data)) {
          if (msg.data.length > 0) {
            pipelineData.raw_metrics = msg.data;
            if (Array.isArray(msg.active_services) && msg.active_services.length > 0) {
              pipelineData.active_services = msg.active_services;
            }
            processMemoryData();
            updateOverview();
            updateServiceFilterDropdown();
            renderStageData();
          }
        } else if (msg.type === "failure_alert") {
          processMemoryAlert(msg);
        }
      } catch (e) {
        console.error("Error parsing WS message:", e);
      }
    };
    socket.onclose = () => {
      const dot = document.getElementById("statusDot");
      const txt = document.getElementById("statusText");
      if (dot) dot.className = "status-dot offline";
      if (txt) txt.innerText = "Reconnecting...";
      setTimeout(connectWebSocket, 5000);
    };
    socket.onerror = () => {
      if (socket) socket.close();
    };
  } catch (e) {
    console.error("WebSocket setup failed:", e);
  }
}

// Switch active tab and nav highlight
function switchTab(tabId) {
  currentTab = tabId;

  // Update sidebar nav active state
  const tabIds = ["overview", "configuration", "memory", "windows"];
  document.querySelectorAll(".nav-item").forEach((nav, idx) => {
    nav.classList.remove("active");
    if (idx < tabIds.length && tabIds[idx] === tabId) {
      nav.classList.add("active");
    }
  });

  // Update tab pane
  document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));
  const paneMap = {
    "overview": "paneOverview",
    "memory": "paneMemory",
    "windows": "paneWindows",
    "configuration": "paneConfiguration"
  };
  if (paneMap[tabId]) {
    document.getElementById(paneMap[tabId])?.classList.add("active");
  }

  document.querySelector(".content-scroll")?.classList.toggle("config-scroll", tabId === "configuration");
  document.querySelector(".top-header")?.classList.toggle("config-hide-header", tabId === "configuration");

  if (tabId === "configuration") {
    reloadConfiguration();
    startConfigPolling();
  } else {
    stopConfigPolling();
  }

  renderStageData();
}

function initConfigurationPanel() {
  const urlInput = document.getElementById("configBackendUrl");
  const msInput = document.getElementById("configRefreshMs");
  if (urlInput) urlInput.value = API_BASE;
  if (msInput) msInput.value = String(configRefreshMs);
}

function startConfigPolling() {
  stopConfigPolling();
  configPollTimer = setInterval(() => {
    if (currentTab === "configuration") reloadConfiguration();
  }, configRefreshMs);
}

function stopConfigPolling() {
  if (configPollTimer) {
    clearInterval(configPollTimer);
    configPollTimer = null;
  }
}

function applyConnectionSettings() {
  const urlInput = document.getElementById("configBackendUrl");
  const msInput = document.getElementById("configRefreshMs");
  const nextUrl = (urlInput?.value || DEFAULT_API_BASE).trim().replace(/\/$/, "");
  const nextMs = Math.max(500, Number(msInput?.value) || 1000);
  API_BASE = nextUrl || DEFAULT_API_BASE;
  configRefreshMs = nextMs;
  localStorage.setItem("c1_backend_url", API_BASE);
  localStorage.setItem("c1_config_refresh_ms", String(configRefreshMs));
  if (currentTab === "configuration") startConfigPolling();
  reloadConfiguration();
  fetchBackendState();
}

async function reloadConfiguration() {
  const connDot = document.getElementById("configConnectionDot");
  try {
    const [configRes, statusRes, metricsRes] = await Promise.all([
      fetch(`${API_BASE}/api/services-config`),
      fetch(`${API_BASE}/api/status`),
      fetch(`${API_BASE}/api/live-metrics?limit=500`)
    ]);
    if (!configRes.ok) throw new Error("Unable to load services_config.json");
    const fileConfig = await configRes.json();
    configFileSnapshot = fileConfig;

    const statusData = statusRes.ok ? await statusRes.json() : {};
    const metrics = metricsRes.ok ? await metricsRes.json() : [];

    renderConfigurationView(fileConfig, statusData, metrics);
    if (connDot) connDot.classList.remove("offline");
  } catch (err) {
    if (connDot) connDot.classList.add("offline");
    const grid = document.getElementById("configDiscoveryGrid");
    if (grid) grid.innerHTML = `<p class="config-empty">${escapeHtml(err.message)}</p>`;
    const pre = document.getElementById("configJsonLive");
    if (pre) pre.textContent = "{}";
  }
}

function inferServicePlatform(service) {
  const name = String(service.name || service.id || "").toLowerCase();
  const url = String(service.base_url || "").toLowerCase();
  if (name.includes("python") || url.includes("python")) return "python";
  if (name.includes("node") || url.includes("node")) return "node";
  if (name.includes("ruby") || url.includes("ruby")) return "ruby";
  if (name.includes("go") || url.includes("-go")) return "go";
  if (name.includes("php") || url.includes("php")) return "php";
  if (url.includes("azurecontainerapps.io") || name.includes("azure")) return "azure";
  if (url.includes("onrender.com") || name.includes("render")) return "render";
  if (url.includes("railway.app")) return "railway";
  return "service";
}

const DEVICON_BASE = "https://cdn.jsdelivr.net/gh/devicons/devicon@master/icons";

function platformMeta(platform) {
  const map = {
    python: { label: "python", className: "platform-python", glyph: "Py", icon: `${DEVICON_BASE}/python/python-original.svg` },
    node: { label: "node.js", className: "platform-node", glyph: "JS", icon: `${DEVICON_BASE}/nodejs/nodejs-original.svg` },
    ruby: { label: "ruby", className: "platform-ruby", glyph: "Rb", icon: `${DEVICON_BASE}/ruby/ruby-original.svg` },
    go: { label: "go", className: "platform-go", glyph: "Go", icon: `${DEVICON_BASE}/go/go-original.svg` },
    php: { label: "php", className: "platform-php", glyph: "PHP", icon: `${DEVICON_BASE}/php/php-original.svg` },
    azure: { label: "azure", className: "platform-azure", glyph: "Az", icon: `${DEVICON_BASE}/azure/azure-original.svg` },
    render: { label: "render", className: "platform-render", glyph: "R", icon: null },
    railway: { label: "railway", className: "platform-railway", glyph: "Rw", icon: `${DEVICON_BASE}/railway/railway-original.svg` },
    service: { label: "service", className: "platform-generic", glyph: "S", icon: null }
  };
  return map[platform] || map.service;
}

function platformIconMarkup(meta) {
  if (meta.icon) {
    return `<img class="config-platform-img" src="${meta.icon}" alt="" width="24" height="24" loading="lazy" onerror="this.style.display='none';if(this.nextElementSibling)this.nextElementSibling.style.display='flex';"><span class="config-platform-fallback ${meta.className}" style="display:none">${meta.glyph}</span>`;
  }
  return `<span class="config-platform-fallback ${meta.className}">${meta.glyph}</span>`;
}

function renderConfigurationView(fileConfig, statusData, metrics) {
  const services = Array.isArray(fileConfig.services) ? fileConfig.services : [];
  const enabledServices = services.filter(s => s && s.enabled !== false && s.name);
  const configuredNames = enabledServices.map(s => s.name);

  const activeSet = new Set(statusData.active_containers || []);
  const configuredSet = new Set(statusData.configured_services || configuredNames);
  const reportingSet = new Set(
    (metrics || []).map(m => m.service_name).filter(Boolean)
  );

  const discoveredNames = configuredNames.filter(n => activeSet.has(n) || configuredSet.has(n));
  const reportingNames = configuredNames.filter(n => reportingSet.has(n));
  const pendingNames = configuredNames.filter(n => !activeSet.has(n) && !reportingSet.has(n));

  const discoveredCount = discoveredNames.length || (statusData.status === "running" ? configuredNames.length : 0);
  const reportingCount = reportingNames.length;

  setTextContent("configStatConfigured", String(configuredNames.length));
  setTextContent("configStatDiscovered", String(discoveredCount));
  setTextContent("configStatReporting", String(reportingCount));

  const banner = document.getElementById("configWaitBanner");
  if (banner) {
    if (pendingNames.length > 0) {
      banner.hidden = false;
      banner.innerHTML = `Waiting for backend discovery to pick up <strong>${escapeHtml(pendingNames.join(", "))}</strong> (polls <code>services_config.json</code> every ~10s)…`;
    } else {
      banner.hidden = true;
      banner.textContent = "";
    }
  }

  const liveJson = buildLiveConfigPreview(fileConfig, enabledServices, activeSet, reportingSet);
  const pre = document.getElementById("configJsonLive");
  if (pre) pre.textContent = JSON.stringify(liveJson, null, 2);

  const grid = document.getElementById("configDiscoveryGrid");
  if (grid) {
    if (!enabledServices.length) {
      grid.innerHTML = '<p class="config-empty">No services in services_config.json yet. Use the gear icon in the header to add an Azure microservice.</p>';
    } else {
      grid.innerHTML = enabledServices.map(service => {
        const name = service.name;
        const platform = inferServicePlatform(service);
        const meta = platformMeta(platform);
        const isReporting = reportingSet.has(name);
        const isDiscovered = activeSet.has(name) || configuredSet.has(name);
        let statusLine = `${meta.label} · pending discovery`;
        if (isReporting) statusLine = `${meta.label} · reporting telemetry`;
        else if (isDiscovered) statusLine = `${meta.label} · discovered`;
        const dotClass = isReporting || isDiscovered ? "ok" : "wait";
        return `
          <article class="config-service-card">
            <div class="config-service-icon">${platformIconMarkup(meta)}</div>
            <div class="config-service-body">
              <div class="config-service-name">${escapeHtml(name)}</div>
              <div class="config-service-status">${escapeHtml(statusLine)}</div>
            </div>
            <span class="config-service-dot ${dotClass}" title="${escapeHtml(statusLine)}"></span>
          </article>
        `;
      }).join("");
    }
  }

  const footer = document.getElementById("configDiscoveryFooter");
  if (footer) {
    footer.innerHTML = `File: services_config.json &middot; ${configuredNames.length} service(s) &mdash; discovered: ${discoveredCount} &mdash; reporting: ${reportingCount}<br>Backend discovery_task re-reads this file every 10s; no restart needed.`;
  }
}

function buildLiveConfigPreview(fileConfig, enabledServices, activeSet, reportingSet) {
  const comment = fileConfig._comment;
  const services = enabledServices.map(service => {
    const platform = inferServicePlatform(service);
    const reporting = reportingSet.has(service.name);
    const discovered = activeSet.has(service.name);
    return {
      id: service.name,
      name: service.name,
      base_url: service.base_url,
      platform,
      enabled: service.enabled !== false,
      status: reporting ? "healthy" : discovered ? "discovered" : "pending",
      last_checked: null
    };
  });
  const out = { services };
  if (comment) out._comment = comment;
  return out;
}

function copyConfigJson() {
  const text = JSON.stringify(configFileSnapshot && Object.keys(configFileSnapshot).length
    ? configFileSnapshot
    : {}, null, 2);
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector(".config-header-actions .config-btn:nth-child(2)");
    if (btn) {
      const prev = btn.textContent;
      btn.textContent = "Copied!";
      setTimeout(() => { btn.textContent = prev; }, 1500);
    }
  }).catch(() => alert("Copy failed — use Download instead."));
}

function downloadConfigJson() {
  const text = JSON.stringify(configFileSnapshot && Object.keys(configFileSnapshot).length
    ? configFileSnapshot
    : { services: [] }, null, 2);
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "services_config.json";
  a.click();
  URL.revokeObjectURL(url);
}

function setTextContent(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

// Connect to FastAPI backend
async function fetchBackendState() {
  try {
    const resStatus = await fetch(`${API_BASE}/api/status`);
    if (!resStatus.ok) throw new Error("Backend offline");

    const statusData = await resStatus.json();
    const dot = document.getElementById("statusDot");
    const txt = document.getElementById("statusText");
    if (dot) dot.className = "status-dot";
    if (txt) {
      txt.innerText = statusData.status === "stopped" ? "Stopped" : "Live Ingestion";
    }

    const sysStatus = document.getElementById("systemStatus");
    if (sysStatus) sysStatus.innerText = statusData.status || "N/A";
    const lastUp = document.getElementById("lastUpdate");
    if (lastUp) lastUp.innerText = new Date().toLocaleTimeString();

    const heroUptime = document.getElementById("heroUptime");
    if (heroUptime && statusData.uptime_seconds != null) {
      const m = Math.floor(statusData.uptime_seconds / 60);
      const s = statusData.uptime_seconds % 60;
      heroUptime.innerText = `${m}m ${s}s`;
    }

    isSimulating = false;
    const simState = document.getElementById("simState");
    if (simState) simState.innerText = "OFF";

    const [resMetrics, resTemplates, resWindows, resStats, resFeed] = await Promise.all([
      fetch(`${API_BASE}/api/live-metrics`).then(r => r.json()).catch(() => []),
      fetch(`${API_BASE}/api/templates`).then(r => r.json()).catch(() => []),
      fetch(`${API_BASE}/api/windows`).then(r => r.json()).catch(() => []),
      fetch(`${API_BASE}/api/stats`).then(r => r.json()).catch(() => ({})),
      fetch(`${API_BASE}/api/component3/feed`).then(r => r.json()).catch(() => ({ windows: [] }))
    ]);

    pipelineData.raw_metrics = resMetrics || [];
    pipelineData.templates = resTemplates || [];
    pipelineData.windows = resWindows || [];
    pipelineData.stats = resStats || {};
    pipelineData.feed = (resFeed && resFeed.windows) ? resFeed.windows : [];
    pipelineData.total_rows = statusData.total_rows_collected || pipelineData.raw_metrics.length;
    pipelineData.active_services = statusData.active_containers || [];

    processMemoryData();
    updateOverview();
    updateServiceFilterDropdown();
    renderStageData();
  } catch (err) {
    console.warn("fetchBackendState warning:", err.message);
    if (isSimulating) {
      updateMockStep();
    } else {
      const dot = document.getElementById("statusDot");
      const txt = document.getElementById("statusText");
      if (dot) dot.className = "status-dot offline";
      if (txt) txt.innerText = "Reconnecting...";
    }
  }
}

// Toggle Live Simulation
function toggleSimulation() {
  isSimulating = !isSimulating;
  document.getElementById("simState").innerText = isSimulating ? "ACTIVE" : "OFF";
  if (isSimulating) {
    generateMockTelemetry();
  }
}

// Export current pipeline data as JSON file
function exportData() {
  const dataStr = JSON.stringify({
    timestamp: new Date().toISOString(),
    memory_data: pipelineData
  }, null, 2);
  const blob = new Blob([dataStr], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `memory-export-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// Generate mock telemetry for demo purposes
function generateMockTelemetry() {
  const services = [
    { name: "node-azure", stack: "Node.js/Express" },
    { name: "python-azure", stack: "Python/Flask" },
    { name: "java-azure", stack: "Java/SpringBoot" },
    { name: "go-azure", stack: "Go/net_http" }
  ];

  const memoryLogs = [
    { msg: "memory usage within normal limits", level: "INFO", failure: null },
    { msg: "memory pressure warning: heap allocation high", level: "WARNING", failure: "MEMORY_LEAK", layer: "keyword" },
    { msg: "FATAL java.lang.OutOfMemoryError: Java heap space", level: "CRITICAL", failure: "OOM", layer: "keyword" },
    { msg: "garbage collection overhead limit exceeded", level: "ERROR", failure: "GC_OVERHEAD", layer: "tfidf" },
    { msg: "heap growing rapidly - possible memory leak detected", level: "WARNING", failure: "MEMORY_LEAK", layer: "tfidf" },
    { msg: "GC pause time exceeding threshold", level: "WARNING", failure: "GC_STALL", layer: "tfidf" }
  ];

  pipelineData.raw_metrics = [];
  const now = new Date();

  services.forEach((s, idx) => {
    const isFail = idx === 0 || idx === 2;
    const logItem = isFail ? memoryLogs[idx % memoryLogs.length] : memoryLogs[0];
    const ram = isFail ? 88 + (idx * 2) : 42 + (idx * 3);
    const heap = isFail ? 2450 + (idx * 300) : 512 + (idx * 50);

    pipelineData.raw_metrics.push({
      timestamp: new Date(now - idx * 10000).toISOString(),
      service_name: s.name,
      stack: s.stack,
      ram_percent: parseFloat(ram.toFixed(1)),
      heap_mb_used: parseFloat(heap.toFixed(1)),
      gc_count: isFail ? 12 + idx : 2,
      log_level: logItem.level,
      log_message: logItem.msg,
      hybrid_label: isFail ? "FAILURE" : "NORMAL",
      failure_type: isFail ? (logItem.failure || "MEMORY_LEAK") : "none",
      detection_layer: isFail ? (logItem.layer || "metric_fusion") : "none"
    });
  });

  pipelineData.total_rows = 1250;
  pipelineData.active_services = services.map(s => s.name);

  pipelineData.templates = [
    { template: "memory usage within normal limits", count: 620, failure_related: false },
    { template: "memory pressure warning: heap allocation high", count: 120, failure_related: true },
    { template: "FATAL java.lang.OutOfMemoryError: *", count: 45, failure_related: true },
    { template: "garbage collection overhead limit exceeded", count: 32, failure_related: true },
    { template: "heap growing rapidly - possible memory leak detected", count: 85, failure_related: true }
  ];

  pipelineData.windows = services.map((s, i) => ({
    window_id: 200 + i,
    service_name: s.name,
    stack: s.stack,
    trend_slope_5m: (i % 2 === 0) ? 0.45 : -0.05,
    trend_variance_5m: (i % 2 === 0) ? 1.82 : 0.12,
    failure_ratio: (i % 2 === 0) ? 0.8 : 0.0,
    hybrid_label: (i % 2 === 0) ? "FAILURE" : "NORMAL",
    ground_truth_label: (i % 2 === 0) ? "FAILURE" : "NORMAL"
  }));

  pipelineData.stats = {
    total_windows: pipelineData.windows.length,
    failure_windows: 2,
    normal_windows: 2,
    failure_percent: 50.0,
    overall_f1: 1.0,
    per_service_f1: {
      "node-azure": 1.0,
      "python-azure": 1.0,
      "java-azure": 1.0,
      "go-azure": 1.0
    },
    trend_5m: [
      { time: "17:20", failures: 1 },
      { time: "17:21", failures: 1 },
      { time: "17:22", failures: 2 },
      { time: "17:23", failures: 2 },
      { time: "17:24", failures: 2 }
    ]
  };

  pipelineData.feed = pipelineData.windows;

  processMemoryData();
  updateOverview();
  updateServiceFilterDropdown();
  renderStageData();
}

function updateMockStep() {
  pipelineData.raw_metrics.forEach(m => {
    m.ram_percent = Math.min(99, Math.max(10, m.ram_percent + (Math.random() * 2 - 1)));
    if (m.heap_mb_used) {
      m.heap_mb_used = Math.min(10000, Math.max(100, m.heap_mb_used + (Math.random() * 50 - 25)));
    }
  });
  pipelineData.total_rows += 5;
  processMemoryData();
  updateOverview();
  renderStageData();
}

// Process raw metrics & windows into memory-specific alerts and explanations
function processMemoryData() {
  const metrics = pipelineData.raw_metrics || [];
  const windows = pipelineData.windows || [];

  const alertsMap = new Map();

  // 1. Process from raw metrics (strictly memory failure telemetry — NO CPU alerts)
  metrics.forEach(m => {
    // Exclude CPU-only signals
    if (m.failure_type && m.failure_type.toLowerCase().includes("cpu")) return;

    const isFail = m.hybrid_label === "FAILURE" || (m.ram_percent != null && m.ram_percent >= 85);
    if (isFail && m.service_name) {
      const key = `${m.service_name}-${m.timestamp || Math.random()}`;
      alertsMap.set(key, {
        service: m.service_name,
        type: m.failure_type || (m.ram_percent >= 90 ? "oom" : "memory_leak"),
        ram_percent: m.ram_percent,
        heap_mb: m.heap_mb_used,
        gc_count: m.gc_count,
        log_message: m.log_message,
        timestamp: m.timestamp || new Date().toISOString(),
        layer: m.detection_layer || "hybrid"
      });
    }
  });

  // 2. Also incorporate failure windows if any (strictly memory failures)
  windows.forEach(w => {
    if (w.failure_type && w.failure_type.toLowerCase().includes("cpu")) return;
    if ((w.hybrid_label === "FAILURE" || w.label === "FAILURE") && w.service_name) {
      const key = `${w.service_name}-win-${w.window_id}`;
      if (!alertsMap.has(key)) {
        alertsMap.set(key, {
          service: w.service_name,
          type: w.failure_type || "memory_leak",
          ram_percent: w.ram_max != null ? w.ram_max : w.ram_percent,
          heap_mb: w.heap_max != null ? w.heap_max : w.heap_mb_used,
          gc_count: w.gc_spike_count != null ? w.gc_spike_count : w.gc_count,
          log_message: w.log_template || "Sliding window memory threshold exceeded",
          timestamp: w.timestamp || new Date().toISOString(),
          layer: w.detection_layer || "sliding_window"
        });
      }
    }
  });

  pipelineData.memory_alerts = Array.from(alertsMap.values()).slice(0, 30);

  pipelineData.explanations = pipelineData.memory_alerts.map(a => {
    const t = (a.type || "").toLowerCase();
    let explanation = "";
    if (t.includes("oom") || t.includes("out of memory")) {
      explanation = `${a.service} encountered critical OutOfMemory / Heap Exhaustion. Memory reached ${a.ram_percent != null ? a.ram_percent + '%' : '96%'} and heap reached ${a.heap_mb != null ? a.heap_mb + ' MB' : 'exhaustion'}. Severe allocation pressure detected.`;
    } else if (t.includes("leak") || t.includes("memory")) {
      explanation = `${a.service} exhibits persistent memory pressure and heap growth. RAM is at ${a.ram_percent != null ? a.ram_percent + '%' : 'high'} (Heap: ${a.heap_mb != null ? a.heap_mb + ' MB' : 'elevated'}, GC: ${a.gc_count || 0}). Retained objects are accumulating.`;
    } else if (t.includes("gc") || t.includes("overhead")) {
      explanation = `${a.service} is experiencing garbage collection overhead limits (GC Count: ${a.gc_count != null ? a.gc_count : 'abnormal'}). GC pauses are impacting throughput.`;
    } else {
      explanation = `${a.service} flagged by ${a.layer || "hybrid"} detection layer for anomalous memory resource usage.`;
    }
    return {
      service: a.service,
      type: a.type,
      explanation: explanation,
      timestamp: a.timestamp
    };
  });
}

// Process WebSocket alert (memory-only)
function processMemoryAlert(alert) {
  if (alert.type === "failure_alert") {
    if (alert.failure_type && alert.failure_type.toLowerCase().includes("cpu")) return;
    const memoryAlert = {
      service: alert.service_name || "unknown",
      type: alert.failure_type || "memory_leak",
      ram_percent: alert.ram_percent || 0,
      heap_mb: null,
      gc_count: null,
      log_message: alert.log_message || "",
      timestamp: alert.timestamp || new Date().toISOString(),
      layer: "websocket"
    };
    pipelineData.memory_alerts.unshift(memoryAlert);
    if (pipelineData.memory_alerts.length > 30) {
      pipelineData.memory_alerts = pipelineData.memory_alerts.slice(0, 30);
    }
    updateOverview();
    renderStageData();
  }
}

// Update Overview Tab
function updateOverview() {
  const metrics = pipelineData.raw_metrics;

  if (metrics && metrics.length > 0) {
    const avgRam = metrics.reduce((sum, m) => sum + (m.ram_percent || 0), 0) / metrics.length;
    document.getElementById("statAvgMemory").innerText = avgRam.toFixed(1) + "%";

    const heapValues = metrics.filter(m => m.heap_mb_used != null).map(m => m.heap_mb_used);
    if (heapValues.length > 0) {
      const avgHeap = heapValues.reduce((a, b) => a + b, 0) / heapValues.length;
      document.getElementById("statHeapMemory").innerText = avgHeap.toFixed(0) + " MB";
    } else {
      document.getElementById("statHeapMemory").innerText = "N/A";
    }

    const gcValues = metrics.filter(m => m.gc_count != null).map(m => m.gc_count);
    if (gcValues.length > 0) {
      const totalGC = gcValues.reduce((a, b) => a + b, 0);
      document.getElementById("statGCCount").innerText = totalGC;
    } else {
      document.getElementById("statGCCount").innerText = "N/A";
    }
  } else {
    document.getElementById("statAvgMemory").innerText = "N/A";
    document.getElementById("statHeapMemory").innerText = "N/A";
    document.getElementById("statGCCount").innerText = "N/A";
  }

  document.getElementById("statMemoryAlerts").innerText = pipelineData.memory_alerts.length;

  // Determine active services list
  let services = pipelineData.active_services;
  if (!services || services.length === 0) {
    services = Array.from(new Set(pipelineData.raw_metrics.map(m => m.service_name).filter(Boolean)));
  }

  // Update hero stats
  document.getElementById("heroServices").innerText = services.length;
  document.getElementById("heroAlerts").innerText = pipelineData.memory_alerts.length;

  // Update memory status
  const alertCount = pipelineData.memory_alerts.length;
  const memStatus = alertCount > 2 ? "CRITICAL" : alertCount > 0 ? "WARNING" : "HEALTHY";
  document.getElementById("memoryStatus").innerText = memStatus;

  // Update status badge
  const badge = document.querySelector(".status-badge");
  if (badge) {
    badge.className = "status-badge " + (alertCount > 2 ? "critical" : alertCount > 0 ? "warning" : "healthy");
    badge.textContent = memStatus;
  }

  // Update service health
  const failedServices = new Set(pipelineData.memory_alerts.map(a => a.service));
  const totalServices = services.length;
  const healthyServices = Math.max(0, totalServices - failedServices.size);
  document.getElementById("failedCount").innerText = failedServices.size;
  document.getElementById("healthyCount").innerText = healthyServices;
  const healthPercent = totalServices > 0 ? (healthyServices / totalServices) * 100 : 100;
  document.getElementById("healthBar").style.width = healthPercent + "%";

  // Dynamic Active Services List
  const servicesListEl = document.getElementById("servicesList");
  if (servicesListEl) {
    if (services.length === 0) {
      servicesListEl.innerHTML = '<div class="empty-cell" style="padding:1rem;">Waiting for active services...</div>';
    } else {
      const latestBySvc = {};
      pipelineData.raw_metrics.forEach(m => {
        if (!latestBySvc[m.service_name]) latestBySvc[m.service_name] = m;
      });
      const avatarColors = ["blue", "purple", "orange", "teal", "red"];
      servicesListEl.innerHTML = services.map((svcName, idx) => {
        const m = latestBySvc[svcName] || {};
        const stack = m.stack || "Microservice";
        const initial = svcName.charAt(0).toUpperCase();
        const color = avatarColors[idx % avatarColors.length];
        const isAlert = m.hybrid_label === "FAILURE" || (m.ram_percent && m.ram_percent > 80);
        const statusClass = isAlert ? "warning" : "active";
        const meta = `${stack} • ${m.ram_percent != null ? m.ram_percent + "% RAM" : "Active"}`;
        return `
          <div class="service-item">
            <div class="service-avatar ${color}">${initial}</div>
            <div class="service-info">
              <div class="service-name">${escapeHtml(svcName)}</div>
              <div class="service-meta">${escapeHtml(meta)}</div>
            </div>
            <span class="service-status ${statusClass}" title="${isAlert ? 'Alert / High RAM' : 'Healthy'}"></span>
          </div>
        `;
      }).join("");
    }
  }

  // Update health items
  const hasTemplates = pipelineData.templates && pipelineData.templates.length > 0;
  const hasClassifier = (pipelineData.raw_metrics && pipelineData.raw_metrics.some(m => m.hybrid_label)) ||
                        (pipelineData.windows && pipelineData.windows.length > 0) ||
                        (pipelineData.stats && pipelineData.stats.total_windows > 0);
  const isStreaming = (pipelineData.feed && pipelineData.feed.length > 0) ||
                      (pipelineData.windows && pipelineData.windows.length > 0);

  const elIngestion = document.getElementById("healthIngestion");
  if (elIngestion) {
    elIngestion.innerText = pipelineData.status === "stopped" ? "Offline" : "Live";
    elIngestion.className = "health-value " + (pipelineData.status === "stopped" ? "bad" : "good");
  }

  const elDrain3 = document.getElementById("healthDrain3");
  if (elDrain3) {
    elDrain3.innerText = hasTemplates ? "Active" : (pipelineData.total_rows > 0 ? "Active" : "Standby");
    elDrain3.className = "health-value " + (hasTemplates ? "good" : "good");
  }

  const elClassifier = document.getElementById("healthClassifier");
  if (elClassifier) {
    elClassifier.innerText = hasClassifier ? "Active" : (pipelineData.total_rows > 0 ? "Active" : "Standby");
    elClassifier.className = "health-value " + (hasClassifier ? "good" : "warning");
  }

  const elExport = document.getElementById("healthExport");
  if (elExport) {
    elExport.innerText = isStreaming ? "Streaming" : "Standby";
    elExport.className = "health-value " + (isStreaming ? "good" : "warning");
  }

  updateCharts();
}

// Dropdown filter population
function updateServiceFilterDropdown() {
  const sel = document.getElementById("serviceSelect");
  if (!sel) return;
  const currVal = sel.value || "ALL";

  const rawSvcs = (pipelineData.raw_metrics || []).map(m => m.service_name).filter(Boolean);
  const activeSvcs = pipelineData.active_services || [];
  const allServices = Array.from(new Set([...activeSvcs, ...rawSvcs])).sort();

  const existingValues = Array.from(sel.options).map(o => o.value).filter(v => v !== "ALL");
  const isSame = existingValues.length === allServices.length &&
    existingValues.every((val, i) => val === allServices[i]);

  if (!isSame) {
    sel.innerHTML = '<option value="ALL">All Services</option>';
    allServices.forEach(svc => {
      const opt = document.createElement("option");
      opt.value = svc;
      opt.innerText = svc;
      if (svc === currVal) opt.selected = true;
      sel.appendChild(opt);
    });
    if (currVal && Array.from(sel.options).some(o => o.value === currVal)) {
      sel.value = currVal;
    }
  }
}

// Main Render Function for Active Tab
function renderStageData() {
  const selectedSvc = document.getElementById("serviceSelect")?.value || "ALL";
  let metrics = pipelineData.raw_metrics || [];
  if (selectedSvc !== "ALL") {
    metrics = metrics.filter(m => m.service_name === selectedSvc);
  }

  // Always update memory trend chart live whenever filter or telemetry changes
  try {
    updateMemoryTrendChart(selectedSvc);
  } catch (e) {
    console.error("Error updating trend chart:", e);
  }

  // Always keep memory and windows tables populated with live data
  try {
    renderMemoryTable(metrics);
  } catch (e) {
    console.error("Error rendering memory table:", e);
  }

  try {
    renderWindowsTable(selectedSvc);
  } catch (e) {
    console.error("Error rendering windows table:", e);
  }

  if (currentTab === "overview") {
    const latestForSvc = metrics[0];
    if (selectedSvc !== "ALL" && latestForSvc) {
      if (latestForSvc.ram_percent != null) {
        setTextContent("statAvgMemory", latestForSvc.ram_percent.toFixed(1) + "%");
      }
      if (latestForSvc.heap_mb_used != null) {
        setTextContent("statHeapMemory", latestForSvc.heap_mb_used.toFixed(0) + " MB");
      }
      if (latestForSvc.gc_count != null) {
        setTextContent("statGCCount", String(latestForSvc.gc_count));
      }
    }
  }
}

// Render memory usage table
function renderMemoryTable(metrics) {
  const tbody = document.getElementById("tableMemoryBody");
  if (!tbody) return;

  if (!metrics || metrics.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-cell">No memory data available</td></tr>';
    return;
  }

  tbody.innerHTML = metrics.slice(0, 25).map(m => {
    const isFail = m.hybrid_label === "FAILURE" || (m.ram_percent != null && m.ram_percent >= 85);
    const ramDisplay = m.ram_percent != null ? m.ram_percent + "%" : "N/A";
    const heapDisplay = m.heap_mb_used != null ? m.heap_mb_used + " MB" : "N/A";
    const gcDisplay = m.gc_count != null ? m.gc_count : "N/A";
    const statusBadge = isFail
      ? '<span class="badge badge-failure">ALERT</span>'
      : '<span class="badge badge-normal">Normal</span>';

    return '<tr>' +
      '<td><strong>' + escapeHtml(m.service_name) + '</strong></td>' +
      '<td style="color: ' + (m.ram_percent > 80 ? 'var(--accent-red)' : 'var(--text-primary)') + '">' + ramDisplay + '</td>' +
      '<td>' + heapDisplay + '</td>' +
      '<td>' + gcDisplay + '</td>' +
      '<td>' + statusBadge + '</td>' +
      '</tr>';
  }).join("");
}

// Render windows table
function renderWindowsTable(selectedSvc) {
  const tbody = document.getElementById("tableWindowsBody");
  if (!tbody) return;

  let wins = pipelineData.windows;
  if (selectedSvc !== "ALL") {
    wins = wins.filter(w => w.service_name === selectedSvc);
  }

  if (!wins || wins.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-cell">No window data</td></tr>';
    return;
  }

  tbody.innerHTML = wins.map(w => {
    const slope = w.trend_slope_5m != null 
      ? w.trend_slope_5m.toFixed(3) 
      : (w.memory_growth != null ? (w.memory_growth > 0 ? "+" : "") + w.memory_growth.toFixed(1) + "%" : "0.000");
    const variance = w.trend_variance_5m != null 
      ? w.trend_variance_5m.toFixed(3) 
      : (w.ram_std != null ? w.ram_std.toFixed(2) : "0.000");
    const isFail = w.hybrid_label === 'FAILURE' || w.label === 'FAILURE';
    return '<tr>' +
      '<td><strong>#' + w.window_id + '</strong></td>' +
      '<td>' + escapeHtml(w.service_name) + '</td>' +
      '<td style="color: ' + ((w.trend_slope_5m > 0.2 || w.memory_growth > 10) ? 'var(--accent-red)' : 'var(--text-primary)') + '">' + slope + '</td>' +
      '<td>' + variance + '</td>' +
      '<td>' + (isFail ? '<span class="badge badge-failure">FAILURE</span>' : '<span class="badge badge-normal">NORMAL</span>') + '</td>' +
      '</tr>';
  }).join("");
}

// Filter logs based on search input
function filterLogs() {
  renderStageData();
}

const SERVICE_COLORS = [
  "#3b82f6", // blue
  "#10b981", // emerald
  "#f59e0b", // amber
  "#a855f7", // purple
  "#ec4899", // pink
  "#06b6d4", // cyan
  "#f43f5e", // rose
  "#6366f1"  // indigo
];

function hexToRgba(hex, alpha) {
  const c = hex.replace("#", "");
  const r = parseInt(c.substring(0, 2), 16);
  const g = parseInt(c.substring(2, 4), 16);
  const b = parseInt(c.substring(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function updateMemoryTrendChart(selectedSvc) {
  if (!chartMemoryTrend) return;
  const metrics = pipelineData.raw_metrics || [];
  if (metrics.length === 0) return;

  const sel = selectedSvc || document.getElementById("serviceSelect")?.value || "ALL";

  if (sel !== "ALL") {
    // Focused single-service view
    const filtered = metrics
      .filter(m => m.service_name === sel)
      .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
      .slice(-30);

    if (filtered.length > 0) {
      chartMemoryTrend.data.labels = filtered.map(m => formatTime(m.timestamp));
      chartMemoryTrend.data.datasets = [{
        label: `${sel} Memory Usage (%)`,
        data: filtered.map(m => m.ram_percent != null ? m.ram_percent : 0),
        borderColor: "#2563eb",
        backgroundColor: "rgba(37, 99, 235, 0.15)",
        fill: true,
        tension: 0.35,
        pointRadius: 4,
        pointHoverRadius: 6,
        borderWidth: 2
      }];
    } else {
      chartMemoryTrend.data.labels = [];
      chartMemoryTrend.data.datasets = [{
        label: `${sel} (Waiting for telemetry...)`,
        data: [],
        borderColor: "#64748b",
        backgroundColor: "transparent"
      }];
    }
  } else {
    // Multi-service view: separate clean line per service so data points do not collide
    const sorted = [...metrics].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    const uniqueTimes = Array.from(new Set(sorted.map(m => formatTime(m.timestamp)))).slice(-20);

    const services = Array.from(new Set(sorted.map(m => m.service_name).filter(Boolean)));
    const datasets = services.map((svcName, idx) => {
      const color = SERVICE_COLORS[idx % SERVICE_COLORS.length];
      const timeToVal = {};
      sorted.filter(m => m.service_name === svcName).forEach(m => {
        timeToVal[formatTime(m.timestamp)] = m.ram_percent != null ? m.ram_percent : 0;
      });

      return {
        label: svcName,
        data: uniqueTimes.map(t => timeToVal[t] !== undefined ? timeToVal[t] : null),
        borderColor: color,
        backgroundColor: hexToRgba(color, 0.1),
        fill: false,
        tension: 0.35,
        pointRadius: 3,
        pointHoverRadius: 5,
        borderWidth: 2,
        spanGaps: true
      };
    });

    chartMemoryTrend.data.labels = uniqueTimes;
    chartMemoryTrend.data.datasets = datasets.length > 0 ? datasets : [{
      label: "All Services Memory Usage (%)",
      data: [],
      borderColor: "#2563eb"
    }];
  }

  chartMemoryTrend.update();
}

// Initialize Chart.js Instances
function initCharts() {
  // Hero Memory Chart (small sparkline in banner)
  const ctxHero = document.getElementById("chartHeroMemory")?.getContext("2d");
  if (ctxHero) {
    chartHeroMemory = new Chart(ctxHero, {
      type: "line",
      data: {
        labels: ["", "", "", "", "", "", ""],
        datasets: [{
          data: [45, 52, 48, 65, 72, 68, 75],
          borderColor: "#ffffff",
          backgroundColor: "rgba(255,255,255,0.2)",
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { display: false },
          y: { display: false, min: 0, max: 100 }
        }
      }
    });
  }

  // Memory Trend Chart
  const ctxTrend = document.getElementById("chartMemoryTrend")?.getContext("2d");
  if (ctxTrend) {
    chartMemoryTrend = new Chart(ctxTrend, {
      type: "line",
      data: {
        labels: [],
        datasets: []
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: "index",
          intersect: false
        },
        scales: {
          x: { ticks: { color: "#94a3b8", maxTicksLimit: 8 }, grid: { color: "rgba(255,255,255,0.05)" } },
          y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" }, beginAtZero: true, max: 100 }
        },
        plugins: {
          legend: {
            display: true,
            position: "top",
            labels: { color: "#cbd5e1", boxWidth: 12, padding: 8, font: { family: "Inter", size: 11 } }
          },
          tooltip: {
            mode: "index",
            intersect: false
          }
        }
      }
    });
  }

  // Heap Memory Chart
  const ctxHeap = document.getElementById("chartHeapMemory")?.getContext("2d");
  if (ctxHeap) {
    chartHeapMemory = new Chart(ctxHeap, {
      type: "doughnut",
      data: {
        labels: [],
        datasets: [{
          data: [],
          backgroundColor: ["#2563eb", "#7c3aed", "#ea580c", "#0d9488", "#16a34a"],
          borderWidth: 2,
          borderColor: "#ffffff"
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "right", labels: { color: "#64748b", font: { family: "Inter", size: 11 } } }
        }
      }
    });
  }

  // Failure Trend Chart
  const ctxFailTrend = document.getElementById("chartTrend")?.getContext("2d");
  if (ctxFailTrend) {
    chartTrend = new Chart(ctxFailTrend, {
      type: "line",
      data: {
        labels: [],
        datasets: [{
          label: "Memory Failures (5m)",
          data: [],
          borderColor: "#dc2626",
          backgroundColor: "rgba(220, 38, 38, 0.1)",
          fill: true,
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: "#64748b" }, grid: { color: "rgba(0,0,0,0.06)" } },
          y: { ticks: { color: "#64748b", precision: 0 }, grid: { color: "rgba(0,0,0,0.06)" }, beginAtZero: true }
        },
        plugins: {
          legend: { labels: { color: "#64748b" } }
        }
      }
    });
  }
}

// Update Chart Data
function updateCharts() {
  const metrics = pipelineData.raw_metrics;

  // Memory Trend Chart with per-service filter
  const selectedSvc = document.getElementById("serviceSelect")?.value || "ALL";
  updateMemoryTrendChart(selectedSvc);

  // Heap Memory Chart
  if (chartHeapMemory && metrics && metrics.length > 0) {
    const serviceHeap = {};
    metrics.forEach(m => {
      if (m.heap_mb_used != null) {
        serviceHeap[m.service_name] = (serviceHeap[m.service_name] || 0) + m.heap_mb_used;
      }
    });
    const labels = Object.keys(serviceHeap);
    const data = Object.values(serviceHeap);
    chartHeapMemory.data.labels = labels;
    chartHeapMemory.data.datasets[0].data = data;
    chartHeapMemory.update();
  }

  // Failure Trend Chart
  if (chartTrend) {
    let trend = (pipelineData.stats && pipelineData.stats.trend_5m) || [];
    if (trend.length === 0 && pipelineData.windows && pipelineData.windows.length > 0) {
      const minMap = {};
      pipelineData.windows.forEach(w => {
        if (w.timestamp) {
          const t = new Date(w.timestamp);
          const minStr = String(t.getHours()).padStart(2, '0') + ':' + String(t.getMinutes()).padStart(2, '0');
          minMap[minStr] = (minMap[minStr] || 0) + (w.hybrid_label === 'FAILURE' || w.label === 'FAILURE' ? 1 : 0);
        }
      });
      const keys = Object.keys(minMap).slice(-8);
      trend = keys.map(k => ({ time: k, failures: minMap[k] }));
    }
    if (trend.length > 0) {
      chartTrend.data.labels = trend.map(t => t.time);
      chartTrend.data.datasets[0].data = trend.map(t => t.failures);
      chartTrend.update();
    }
  }

  // Hero chart update
  if (chartHeroMemory && metrics && metrics.length > 0) {
    const sorted = [...metrics].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    const recentData = sorted.slice(-7).map(m => m.ram_percent || 0);
    while (recentData.length < 7) recentData.unshift(0);
    chartHeroMemory.data.datasets[0].data = recentData;
    chartHeroMemory.update();
  }
}

// Helper functions
function formatTime(timestamp) {
  if (!timestamp) return "N/A";
  try {
    return new Date(timestamp).toLocaleTimeString();
  } catch {
    return "N/A";
  }
}

function escapeHtml(str) {
  if (str == null) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Add a new service to the configuration
async function addNewMicroservice() {
  const nameInput = document.getElementById("newServiceName");
  const urlInput = document.getElementById("newServiceUrl");
  const messageEl = document.getElementById("newServiceMessage");

  const name = nameInput.value.trim();
  const url = urlInput.value.trim();

  // Validation
  if (!name) {
    messageEl.innerHTML = '<span style="color:#dc2626;">Please enter a service name</span>';
    return;
  }
  if (!url || !url.startsWith("http")) {
    messageEl.innerHTML = '<span style="color:#dc2626;">Please enter a valid URL</span>';
    return;
  }

  try {
    messageEl.innerHTML = '<span style="color:#2563eb;">Connecting to backend...</span>';

    const payload = {
      name: name,
      base_url: url,
      enabled: true
    };

    const response = await fetch(`${API_BASE}/api/azure-services`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Unable to add service");

    messageEl.innerHTML = '<span style="color:#16a34a;">' + escapeHtml(result.message || `Service "${name}" added successfully!`) + '</span>';

    nameInput.value = "";
    urlInput.value = "";

    await fetchBackendState();
    if (typeof reloadConfiguration === "function") reloadConfiguration();
  } catch (err) {
    messageEl.innerHTML = '<span style="color:#dc2626;">Error: ' + escapeHtml(err.message) + '</span>';
  }
}

// Detect platform from URL pattern
function detectPlatformFromUrl(url) {
  const u = url.toLowerCase();
  if (u.includes("python") || u.includes("flask") || u.includes("django")) return "python";
  if (u.includes("node") || u.includes("express")) return "node";
  if (u.includes("java") || u.includes("spring")) return "java";
  if (u.includes("golang") || u.includes("-go")) return "go";
  if (u.includes("dotnet") || u.includes("azure")) return "azure";
  if (u.includes("php")) return "php";
  if (u.includes("ruby") || u.includes("rails")) return "ruby";
  return "generic";
}
