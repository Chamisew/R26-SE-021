# Web console (`frontend/`)

React 18 + Vite SPA. The production build (`dist/`) is served directly by
the Python backend (`research_framework/realtime_experiment.py`) — same
origin, no CORS issues in production.

## Source layout

- `index.html` — Vite entry (`<div id="root">` + `/src/main.jsx`)
- `src/main.jsx`, `src/App.jsx` — app shell (sidebar nav, topbar, toasts)
- `src/api.js` — API client, `localStorage` settings store, toast bus
- `src/charts.js` — framework-free canvas painters (smoothed lines, stacked bars)
- `src/components/` — `Sidebar`, `Topbar`, `Toasts`
- `src/pages/` — all 7 pages (Dashboard, Services, Configuration,
  Dataset, Experiments, Benchmark, Settings)
- `src/styles.css` — light theme (ported from the legacy vanilla UI)
- `src/assets/cpu-dash.jpg` — dashboard hero image (bundled by Vite)

## Run

**1. Build the frontend (required after any `src/` change):**

```powershell
cd frontend
npm install
npm run build
# output: dist/index.html + dist/assets/* (served by the backend)
```

**2. Start the backend (serves `dist/` on port 8081):**

```powershell
cd ..\research_framework
python realtime_experiment.py
# open http://localhost:8081
```

**Dev mode (hot reload, `/api` auto-proxied to the backend):**

```powershell
cd frontend
npm run dev
# open http://localhost:5173  (backend must be running on :8081)
```

## Key user flow

Services tab → paste microservice link → live `/api/services/check` probe →
POST `/api/services` writes `services_config.json` → auto-redirect to the
Configuration tab (new row highlighted) → backend `discovery_task` picks it
up within ~10s → telemetry appears on the Dashboard. No restart anywhere.
