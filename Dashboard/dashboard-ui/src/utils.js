export const RISK_ORDER = { CRITICAL: 0, WARNING: 1, HEALTHY: 2, UNKNOWN: 3 };
export const ACTION_SEVERITY = {
  "CRITICAL_RESTART_AND_TRAFFIC_REROUTE": "critical",
  "TRIGGER_PROACTIVE_POD_RESTART": "critical",
  "TRIGGER_LOAD_SHEDDING": "warning",
  "SCALE_OUT": "warning",
  "INCREASE_OBSERVABILITY": "warning",
  "MONITOR": "warning",
  "NO_ACTION": "none",
};

export function smoothSeries(values, window = 3) {
  const n = values.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0, count = 0;
    for (let k = -window; k <= window; k++) {
      const idx = i + k;
      if (idx >= 0 && idx < n && values[idx] !== null && values[idx] !== undefined) {
        sum += values[idx];
        count++;
      }
    }
    out[i] = count ? sum / count : values[i];
  }
  return out;
}

export function smoothPathFromPoints(pts) {
  if (pts.length < 2) return "";
  if (pts.length === 2)
    return `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)} L${pts[1][0].toFixed(1)},${pts[1][1].toFixed(1)}`;
  let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
  for (let i = 1; i < pts.length; i++) {
    const p0 = pts[i - 1], p1 = pts[i];
    const midX = (p0[0] + p1[0]) / 2, midY = (p0[1] + p1[1]) / 2;
    d += ` Q${p0[0].toFixed(1)},${p0[1].toFixed(1)} ${midX.toFixed(1)},${midY.toFixed(1)}`;
  }
  const last = pts[pts.length - 1];
  d += ` L${last[0].toFixed(1)},${last[1].toFixed(1)}`;
  return d;
}

export function buildPath(values, w, h, pad = 4, smooth = false) {
  const clean = values.filter(v => v !== null && v !== undefined);
  if (clean.length < 2) return { line: "", area: "", pts: [] };
  const min = Math.min(...clean), max = Math.max(...clean);
  const range = max - min || 1;
  const drawValues = smooth ? smoothSeries(values, values.length > 40 ? 2 : 1) : values;
  const n = values.length;
  const stepX = (w - pad * 2) / (n - 1);
  const pts = drawValues.map((v, i) => {
    const x = pad + i * stepX;
    const vv = (v === null || v === undefined) ? min : v;
    const y = h - pad - ((vv - min) / range) * (h - pad * 2);
    return [x, y];
  });
  const line = smooth
    ? smoothPathFromPoints(pts)
    : pts.map((p, i) => (i === 0 ? "M" : "L") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");
  const lastPt = pts[pts.length - 1];
  const area = line + ` L${lastPt[0].toFixed(1)},${h - pad} L${pts[0][0].toFixed(1)},${h - pad} Z`;
  return { line, area, min, max, pts, lastPt };
}

export function fmtPct(v) { return (v === null || v === undefined) ? "—" : (Math.round(v * 1000) / 10) + "%"; }
export function fmtNum(v, d = 1) { return (v === null || v === undefined) ? "—" : Number(v).toFixed(d); }
export function pillClass(risk) { return risk === "CRITICAL" ? "critical" : risk === "WARNING" ? "warning" : "healthy"; }
export function barClass(risk) { return risk === "CRITICAL" ? "c-crit" : risk === "WARNING" ? "c-warn" : "c-heal"; }

export function confidenceLabel(prob) {
  if (prob === null || prob === undefined) return { label: "Unknown", note: "no probability available" };
  const margin = Math.abs(prob - 0.5) * 2;
  if (margin >= 0.7) return { label: "High", note: "probability far from decision boundary" };
  if (margin >= 0.3) return { label: "Moderate", note: "probability moderately clear of the threshold" };
  return { label: "Low", note: "probability close to the decision boundary — treat as tentative" };
}

export function esc(v) { return String(v ?? "—").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c])); }
export function time(v) { return v ? new Date(v).toLocaleString() : "Unavailable"; }

export async function fetchDashboard() {
  const res = await fetch("/api/dashboard", { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
