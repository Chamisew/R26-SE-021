/* ── STATE ──────────────────────────────────────────────────── */
const services  = {};         // service_name -> latest prediction dict
let   filter    = "all";
let   activeAlarm = null;     // { alarm_id, service }
let   timelineChart = null;
let   mttaChart     = null;
let   alarmCountdownInterval = null; // For the countdown timer

/* ── WEBSOCKET ──────────────────────────────────────────────── */
const socket = io();

socket.on("connect", () => {
  setConnBadge(true);
  // Load initial snapshot
  fetchServices();
  loadHistory();
  loadDrift();
  loadMtta();
});

socket.on("disconnect", () => setConnBadge(false));

socket.on("service_update", (data) => {
  services[data.service] = data;
  renderCard(data);
  updateStats();
  updateClock();
});

socket.on("alarm", (data) => {
  activeAlarm = { alarm_id: data.alarm_id, service: data.service };
  showAlarmBanner(data);
  loadHistory();
});

socket.on("alarm_resolved", () => {
  loadHistory();
  loadDrift();
});

socket.on("mtta_update", () => {
  loadMtta();
});

/* ── INITIAL LOAD ───────────────────────────────────────────── */
async function fetchServices() {
  try {
    const r = await fetch("/api/services");
    const d = await r.json();
    d.services.forEach(s => {
      services[s.service] = s;
      renderCard(s);
    });
    updateStats();
  } catch (e) {
    console.warn("fetchServices failed:", e);
  }
}

/* ── CARD RENDERING ─────────────────────────────────────────── */
function renderCard(s) {
  const grid = document.getElementById("cards-grid");
  const ns   = document.getElementById("no-services");
  if (ns) ns.style.display = "none";

  const id  = "card-" + s.service.replace(/[^a-z0-9]/gi, "-");
  let   el  = document.getElementById(id);

  if (!el) {
    el = document.createElement("div");
    el.id        = id;
    el.className = "service-card";
    el.onclick   = () => openTimeline(s.service);
    grid.appendChild(el);
  }

  // Apply filter visibility
  el.style.display = (filter === "all" || filter === s.status) ? "" : "none";

  // Status class
  el.className = `service-card ${s.status}`;

  const cpu     = (s.latest_cpu || 0).toFixed(1);
  const prob    = s.prob != null ? (s.prob * 100).toFixed(1) + "%" : "—";
  const probVal = s.prob != null ? s.prob * 100 : 0;
  const cpuColor = cpuToColor(s.latest_cpu || 0);
  const probClass = s.status;

  const queue   = (s.latest_queue       || 0).toFixed(0);
  const inRate  = (s.latest_incoming    || 0).toFixed(0);
  const outRate = (s.latest_processing  || 0).toFixed(0);
  const qGrowth = (s.latest_queue_growth|| 0).toFixed(1);

  const statusLabel = s.status === "warmup"
    ? `WARMING UP (${s.row_count}/${10} rows)`
    : s.status.toUpperCase();

  el.innerHTML = `
    <div class="card-top">
      <div class="card-name">${escHtml(s.service)}</div>
      <div class="status-badge ${s.status}">${statusLabel}</div>
    </div>
    <div class="cpu-bar-wrap">
      <div class="cpu-bar-label">
        <span>CPU</span>
        <span>${cpu}%</span>
      </div>
      <div class="cpu-bar-track">
        <div class="cpu-bar-fill" style="width:${Math.min(cpu,100)}%; background:${cpuColor}"></div>
      </div>
    </div>
    <div class="prob-row">
      <div class="prob-label">Failure probability</div>
      <div class="prob-value ${probClass}">${prob}</div>
    </div>
    <div class="card-metrics">
      <div class="metric-item">Queue: <span>${queue}</span></div>
      <div class="metric-item">Growth: <span>${qGrowth}/s</span></div>
      <div class="metric-item">In: <span>${inRate}/s</span></div>
      <div class="metric-item">Out: <span>${outRate}/s</span></div>
    </div>
  `;
}

function cpuToColor(cpu) {
  if (cpu >= 80) return "#f85149";
  if (cpu >= 50) return "#d29922";
  return "#3fb950";
}

function escHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

/* ── STATS ──────────────────────────────────────────────────── */
function updateStats() {
  const all     = Object.values(services);
  const alarms  = all.filter(s => s.status === "alarm").length;
  const watches = all.filter(s => s.status === "watch").length;
  const warmups = all.filter(s => s.status === "warmup").length;

  document.getElementById("stat-services").textContent = all.length;
  document.getElementById("stat-alarm").textContent    = alarms;
  document.getElementById("stat-watch").textContent    = watches;
  document.getElementById("stat-warmup").textContent   = warmups;
}

/* ── FILTER ─────────────────────────────────────────────────── */
function setFilter(f) {
  filter = f;
  document.querySelectorAll(".filter-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.filter === f);
  });
  Object.values(services).forEach(s => renderCard(s));
}

/* ── ALARM BANNER ───────────────────────────────────────────── */
function showAlarmBanner(data) {
  const banner = document.getElementById("alarm-banner");
  
  // Update the banner content
  document.getElementById("alarm-title").textContent = `CPU spike imminent — ${data.service}`;
  document.getElementById("alarm-detail").textContent = 
    `Failure prob ${(data.prob * 100).toFixed(1)} exceeded threshold 0.55 • Smoothed signal confirmed • CPU ${(data.cpu || 0).toFixed(1)}%`;
  document.getElementById("alarm-prob").textContent = (data.prob).toFixed(3);
  document.getElementById("alarm-cpu").textContent = `${(data.cpu || 0).toFixed(0)}%`;
  
  // Start the countdown from 3.5 minutes (210 seconds)
  let countdownSeconds = 210;
  
  // Clear any existing countdown
  if (alarmCountdownInterval) {
    clearInterval(alarmCountdownInterval);
  }
  
  // Update the countdown display initially
  updateCountdownDisplay(countdownSeconds);
  
  // Start the countdown interval
  alarmCountdownInterval = setInterval(() => {
    countdownSeconds -= 1;
    updateCountdownDisplay(countdownSeconds);
    
    // Hide the banner when countdown reaches 0
    if (countdownSeconds <= 0) {
      clearInterval(alarmCountdownInterval);
      alarmCountdownInterval = null;
      document.getElementById("alarm-banner").classList.add("hidden");
      activeAlarm = null;
    }
  }, 1000);
  
  // Show the banner
  banner.classList.remove("hidden");
}

function updateCountdownDisplay(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  document.getElementById("alarm-time").textContent = 
    `${minutes}.${remainingSeconds.toString().padStart(2, '0')} min`;
}

function handleAlarm(action) {
  if (!activeAlarm) return;
  const { alarm_id, service } = activeAlarm;

  if (action === "dismiss") {
    fetch(`/api/outcomes/${alarm_id}/dismiss`, { method: "POST" });
  } else {
    // "restart" or "shed" = treating as a real event (true positive)
    fetch(`/api/outcomes/${alarm_id}/confirm`, { method: "POST" });
  }

  // Clear the countdown interval
  if (alarmCountdownInterval) {
    clearInterval(alarmCountdownInterval);
    alarmCountdownInterval = null;
  }
  
  document.getElementById("alarm-banner").classList.add("hidden");
  activeAlarm = null;
}

/* ── TIMELINE MODAL ─────────────────────────────────────────── */
async function openTimeline(service) {
  try {
    const r = await fetch(`/api/timeline/${encodeURIComponent(service)}`);
    const d = await r.json();
    if (d.error) return;

    document.getElementById("modal-title").textContent = service + " — Timeline";
    document.getElementById("modal-overlay").classList.remove("hidden");

    const rows   = d.rows || [];
    const labels = rows.map((_, i) => i);
    const cpus   = rows.map(r => parseFloat(r.cpu_percent || 0).toFixed(1));
    const queues = rows.map(r => parseFloat(r.in_flight_queue || 0).toFixed(1));

    document.getElementById("modal-meta").innerHTML =
      `<span>Rows buffered: <strong>${d.count}</strong></span>
       <span>Service: <strong>${escHtml(service)}</strong></span>`;

    if (timelineChart) timelineChart.destroy();

    const ctx = document.getElementById("timeline-chart").getContext("2d");
    timelineChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "CPU %",
            data: cpus,
            borderColor: "#f85149",
            backgroundColor: "rgba(248,81,73,0.1)",
            tension: 0.3,
            pointRadius: 0,
            fill: true,
            yAxisID: "y",
          },
          {
            label: "Queue",
            data: queues,
            borderColor: "#58a6ff",
            backgroundColor: "rgba(88,166,255,0.05)",
            tension: 0.3,
            pointRadius: 0,
            fill: true,
            yAxisID: "y1",
          },
        ],
      },
      options: {
        animation: false,
        responsive: true,
        plugins: { legend: { labels: { color: "#8b949e", font: { size: 11 } } } },
        scales: {
          x:  { display: false },
          y:  { title: { display: true, text: "CPU %",  color: "#f85149" }, grid: { color: "#21262d" }, ticks: { color: "#8b949e" } },
          y1: { position: "right", title: { display: true, text: "Queue", color: "#58a6ff" }, grid: { drawOnChartArea: false }, ticks: { color: "#8b949e" } },
        },
      },
    });
  } catch (e) {
    console.warn("Timeline error:", e);
  }
}

function closeModal() {
  document.getElementById("modal-overlay").classList.add("hidden");
  if (timelineChart) { timelineChart.destroy(); timelineChart = null; }
}

/* ── ALARM HISTORY ──────────────────────────────────────────── */
async function loadHistory() {
  try {
    const r   = await fetch("/api/outcomes");
    const d   = await r.json();
    const tbody = document.getElementById("history-body");

    if (!d.alarms || d.alarms.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="empty-row">No alarms recorded yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = d.alarms.map(a => {
      const dt  = new Date(a.fired_at * 1000).toLocaleTimeString();
      const prob = (a.prob * 100).toFixed(1) + "%";
      const cpu  = (a.cpu_percent || 0).toFixed(1) + "%";
      let outcome, actions;

      if (a.outcome === "true_positive") {
        outcome = `<span class="outcome-badge tp">✅ Real failure</span>`;
        actions = "—";
      } else if (a.outcome === "false_positive") {
        outcome = `<span class="outcome-badge fp">False alarm</span>`;
        actions = "—";
      } else {
        outcome = `<span class="outcome-badge pending">Pending</span>`;
        actions = `
          <button class="btn-outcome btn-confirm"      onclick="confirmAlarm(${a.id})">✅ Real</button>
          <button class="btn-outcome btn-dismiss-out"  onclick="dismissAlarm(${a.id})">✕ False</button>`;
      }

      return `<tr>
        <td>${a.id}</td>
        <td><code>${escHtml(a.service)}</code></td>
        <td>${dt}</td>
        <td style="color:#f85149;font-weight:600">${prob}</td>
        <td>${cpu}</td>
        <td>${outcome}</td>
        <td>${actions}</td>
      </tr>`;
    }).join("");
  } catch (e) {
    console.warn("loadHistory error:", e);
  }
}

async function confirmAlarm(id) {
  await fetch(`/api/outcomes/${id}/confirm`, { method: "POST" });
  loadHistory(); loadDrift();
}

async function dismissAlarm(id) {
  await fetch(`/api/outcomes/${id}/dismiss`, { method: "POST" });
  loadHistory(); loadDrift();
}

/* ── DRIFT PANEL ─────────────────────────────────────────────── */
async function loadDrift() {
  try {
    const r = await fetch("/api/drift");
    const d = await r.json();
    const grid = document.getElementById("drift-grid");

    const prec = (d.precision * 100).toFixed(0) + "%";
    const wprec = (d.week_precision * 100).toFixed(0) + "%";

    grid.innerHTML = `
      <div class="drift-card">
        <div class="drift-card-label">Total alarms</div>
        <div class="drift-card-value">${d.total_alarms}</div>
      </div>
      <div class="drift-card">
        <div class="drift-card-label">Resolved</div>
        <div class="drift-card-value">${d.resolved}</div>
      </div>
      <div class="drift-card">
        <div class="drift-card-label">True positives</div>
        <div class="drift-card-value ok">${d.true_positives}</div>
      </div>
      <div class="drift-card">
        <div class="drift-card-label">False positives</div>
        <div class="drift-card-value">${d.false_positives}</div>
      </div>
      <div class="drift-card">
        <div class="drift-card-label">Overall precision</div>
        <div class="drift-card-value ${d.precision < 0.7 ? 'warn' : 'ok'}">${prec}</div>
      </div>
      <div class="drift-card">
        <div class="drift-card-label">This week</div>
        <div class="drift-card-value ${d.week_precision < 0.7 ? 'warn' : 'ok'}">${wprec}</div>
      </div>
      <div class="drift-message ${d.drift_detected ? 'drift' : 'nodrift'}">
        ${escHtml(d.drift_message)}
      </div>
    `;
  } catch (e) {
    console.warn("loadDrift error:", e);
  }
}

/* ── CLOCK ──────────────────────────────────────────────────── */
function updateClock() {
  document.getElementById("hdr-time").textContent = new Date().toLocaleTimeString();
}

/* ── UTILS ──────────────────────────────────────────────────── */
function setConnBadge(connected) {
  const b = document.getElementById("conn-badge");
  b.textContent = connected ? "● LIVE" : "● OFFLINE";
  b.className   = "badge-live " + (connected ? "connected" : "disconnected");
}

/* Reload drift every 30s */
setInterval(loadDrift, 30000);

/* ── MTTA FRAMEWORK ──────────────────────────────────────────── */
async function loadMtta() {
  try {
    const r = await fetch("/api/mtta");
    const d = await r.json();
    
    const results = d.results || [];
    const summary = d.summary || { avg_mtta: 0, target_met_rate: 0, total_events: 0 };
    
    document.getElementById("mtta-avg").textContent = summary.avg_mtta.toFixed(1) + "s";
    document.getElementById("mtta-rate").textContent = (summary.target_met_rate * 100).toFixed(0) + "%";
    document.getElementById("mtta-total").textContent = summary.total_events;
    
    renderMttaChart(results);
    
  } catch (e) {
    console.warn("loadMtta error:", e);
  }
}

function renderMttaChart(events) {
  const ctx = document.getElementById("mtta-chart").getContext("2d");
  
  // Sort events by time if available, or just use as is
  const sorted = events.slice(-20); // show last 20 events
  const labels = sorted.map(e => e.event_id.split("-")[0]); // service names
  const values = sorted.map(e => e.mtta_seconds);
  const colors = sorted.map(e => e.mtta_seconds >= 120 ? "#3fb950" : "#f85149");

  if (mttaChart) mttaChart.destroy();

  mttaChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "MTTA (seconds)",
        data: values,
        backgroundColor: colors,
        borderRadius: 4,
        minBarLength: 5, // Ensure bars are visible even if 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterBody: (items) => {
              const i = items[0].dataIndex;
              const ev = sorted[i];
              return `Target: ${ev.mtta_seconds >= 120 ? 'MET' : 'FAILED'}\nMTTA: ${ev.mtta_seconds.toFixed(1)}s\nFail: ${ev.t_fail}`;
            }
          }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: "#8b949e", font: { size: 10 } } },
        y: { 
          beginAtZero: true, 
          suggestedMax: 150, // Always show the 120s target
          grid: { color: "#30363d" }, 
          ticks: { color: "#8b949e" },
          title: { display: true, text: "Seconds", color: "#8b949e" }
        }
      }
    },
    plugins: [{
      id: 'targetLine',
      afterDraw: (chart) => {
        const {ctx, scales: {y}} = chart;
        const yPos = y.getPixelForValue(120);
        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = "rgba(248, 81, 73, 0.5)";
        ctx.setLineDash([5, 5]);
        ctx.lineWidth = 2;
        ctx.moveTo(chart.chartArea.left, yPos);
        ctx.lineTo(chart.chartArea.right, yPos);
        ctx.stroke();
        ctx.fillStyle = "rgba(248, 81, 73, 0.8)";
        ctx.fillText("120s Target", chart.chartArea.left + 5, yPos - 5);
        ctx.restore();
      }
    }]
  });
}
