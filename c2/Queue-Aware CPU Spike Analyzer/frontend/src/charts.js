/* Canvas chart painters (ported from the legacy vanilla app.js).
   Framework-free on purpose: called from React effects via canvas refs. */
import { store } from "./api.js";

export const STACK_COLORS = [
  "#2563eb", "#8b5cf6", "#14b8a6", "#f59e0b", "#f43f5e", "#22c55e",
  "#06b6d4", "#fb923c", "#6366f1", "#84cc16", "#ec4899", "#ef4444",
  "#0ea5e9", "#a855f7",
];

export const LINE_PALETTES = {
  cpu: ["#2563eb", "#8b5cf6", "#14b8a6", "#f59e0b", "#f43f5e", "#22c55e"],
  queue: ["#14b8a6", "#8b5cf6", "#2563eb", "#fb923c", "#ec4899", "#22c55e"],
};

/* hex -> rgba helper for gradients / glows */
export function hexA(hex, a) {
  let h = String(hex).replace("#", "");
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  const n = parseInt(h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

/* Round a dynamic axis ceiling up to a pretty 1/2/2.5/5 step */
function niceCeil(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (v <= m * p) return m * p;
  return 10 * p;
}

/* Smooth a polyline with midpoint quadratics (no overshoot like beziers) */
function traceSmooth(ctx, pts) {
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length - 1; i++) {
    const xc = (pts[i].x + pts[i + 1].x) / 2;
    const yc = (pts[i].y + pts[i + 1].y) / 2;
    ctx.quadraticCurveTo(pts[i].x, pts[i].y, xc, yc);
  }
  ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
}

/* Dashed gridlines + y labels. Returns plot box {iw, ih}. */
function paintGrid(ctx, w, h, pad, maxV, unit, dec, labelX = 6, align = "left") {
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  ctx.save();
  ctx.strokeStyle = "#e6edf6";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 5]);
  ctx.fillStyle = "#8a94a6";
  ctx.font = "11px 'Segoe UI', system-ui, sans-serif";
  ctx.textAlign = align;
  for (let g = 0; g <= 4; g++) {
    const y = pad.t + (ih * g) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillText((maxV * (1 - g / 4)).toFixed(dec) + unit, labelX, y + 4);
  }
  ctx.restore();
  return { iw, ih };
}

/* Dot + label legend, wraps to 2 rows then "+N more". */
function paintLegend(ctx, entries, padL, w, h, padR, yFn, colors) {
  const C = colors || STACK_COLORS;
  ctx.font = "11px 'Segoe UI', system-ui, sans-serif";
  let lx = padL, row = 0, shown = 0;
  for (let i = 0; i < entries.length; i++) {
    const label = entries[i].length > 22 ? entries[i].slice(0, 21) + "…" : entries[i];
    const tw = ctx.measureText(label).width;
    if (lx + tw + 18 > w - padR) {
      row++;
      if (row > 1) break;
      lx = padL;
    }
    const ly = yFn(row);
    ctx.fillStyle = C[i % C.length];
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(lx, ly - 9, 10, 10, 5);
    else ctx.rect(lx, ly - 9, 10, 10);
    ctx.fill();
    ctx.fillStyle = "#334155";
    ctx.fillText(label, lx + 14, ly);
    lx += tw + 28;
    shown++;
  }
  if (shown < entries.length) {
    ctx.fillStyle = "#64748b";
    ctx.fillText(`+${entries.length - shown} more`, Math.min(lx, w - 60), yFn(row));
  }
}

/* Relative-time x labels ("-45s … now") from poll cadence */
function paintTimeAxis(ctx, n, padL, slotW, w, labelY) {
  const span = Math.max(0, (n - 1) * (store.refreshMs / 1000));
  const ticks = Math.min(5, n);
  ctx.fillStyle = "#8a94a6";
  ctx.font = "11px 'Segoe UI', system-ui, sans-serif";
  for (let i = 0; i < ticks; i++) {
    const idx = ticks === 1 ? 0 : Math.round((i * (n - 1)) / (ticks - 1));
    const ago = Math.round(span * (1 - idx / Math.max(n - 1, 1)));
    const label = i === ticks - 1 || ago <= 0 ? "now" : `-${ago}s`;
    const x = padL + idx * slotW + slotW / 2;
    ctx.fillText(label, Math.min(Math.max(x - 12, padL), w - 44), labelY);
  }
}

/* Area chart: flowing filled shapes per service (used for queue depth).
   Same point geometry as drawMultiLine, so hitTestLine works for hover. */
export function drawAreaChart(canvas, bucket, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = canvas.clientHeight || 280;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px 'Segoe UI', system-ui, sans-serif";
  const names = Object.keys(bucket);
  if (!names.length) {
    ctx.fillStyle = "#9aa7bd";
    ctx.fillText("Waiting for live telemetry…", 16, 28);
    return;
  }
  const palette = LINE_PALETTES[opts.colorSet === 1 ? "queue" : "cpu"];
  let maxV = opts.max || 0;
  names.forEach((n) => bucket[n].forEach((v) => {
    if (v > maxV) maxV = v;
  }));
  const fixed = opts.max > 0;
  if (maxV <= 0) maxV = 1;
  maxV = fixed ? opts.max : niceCeil(maxV * 1.1);
  const dec = maxV > 10 ? 0 : 2;
  const pad = { l: 48, b: 30, t: 14, r: 12 };
  const { iw, ih } = paintGrid(ctx, w, h, pad, maxV, opts.unit || "", dec);
  const yOf = (v) => pad.t + ih * (1 - Math.min(Math.max(v, 0), maxV) / maxV);
  const drawn = names.slice(0, 12);
  canvas._chartMeta = {
    names: drawn,
    series: Object.fromEntries(drawn.map((n) => [n, bucket[n]])),
    colors: Object.fromEntries(drawn.map((n, i) => [n, palette[i % palette.length]])),
    pad, iw, ih, maxV, unit: opts.unit || "",
  };
  // paint largest-peak series first so smaller shapes stay visible on top
  const order = [...drawn].sort((a, b) => {
    const am = Math.max(...bucket[a], 0), bm = Math.max(...bucket[b], 0);
    return bm - am;
  });
  order.forEach((name) => {
    const vals = bucket[name];
    const color = canvas._chartMeta.colors[name];
    const xOf = (idx) => pad.l + (iw * idx) / Math.max(vals.length - 1, 1);
    const pts = vals.map((v, idx) => ({ x: xOf(idx), y: yOf(v) }));
    if (pts.length === 1) pts.push({ x: pts[0].x + 1, y: pts[0].y });
    // rich gradient fill down to the baseline
    ctx.save();
    ctx.beginPath();
    traceSmooth(ctx, pts);
    ctx.lineTo(pts[pts.length - 1].x, pad.t + ih);
    ctx.lineTo(pts[0].x, pad.t + ih);
    ctx.closePath();
    const ag = ctx.createLinearGradient(0, pad.t, 0, pad.t + ih);
    ag.addColorStop(0, hexA(color, 0.42));
    ag.addColorStop(1, hexA(color, 0.04));
    ctx.fillStyle = ag;
    ctx.fill();
    ctx.restore();
    // crisp top edge (no glow — keeps it visually distinct from CPU lines)
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    traceSmooth(ctx, pts);
    ctx.stroke();
    ctx.restore();
    // live endpoint dot with white ring
    const end = pts[pts.length - 1];
    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(end.x, end.y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(end.x, end.y, 4, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  });
  const longest = Math.max(...names.map((k) => bucket[k].length), 1);
  paintTimeAxis(ctx, longest, pad.l, iw / Math.max(longest - 1, 1), w, h - 14);
}

/* Hit-test a line chart: nearest sampled point within 24px, or null.
   Reads geometry stored by drawMultiLine on canvas._chartMeta. */
export function hitTestLine(canvas, mx, my) {
  const m = canvas._chartMeta;
  if (!m) return null;
  let best = null;
  for (const name of m.names) {
    const vals = m.series[name] || [];
    const n = vals.length;
    for (let i = 0; i < n; i++) {
      const x = m.pad.l + (m.iw * i) / Math.max(n - 1, 1);
      const y = m.pad.t + m.ih * (1 - Math.min(Math.max(vals[i], 0), m.maxV) / m.maxV);
      const d = Math.hypot(x - mx, y - my);
      if (!best || d < best.d) best = { d, name, value: vals[i], x, y, color: m.colors[name] };
    }
  }
  return best && best.d < 24 ? best : null;
}

/* Stable service order for stacked samples (first-seen first) */
export function stackedOrder(samples) {
  const order = [];
  samples.forEach((s) =>
    Object.keys(s.vals).forEach((n) => {
      if (!order.includes(n)) order.push(n);
    })
  );
  return order;
}

/* Hit-test stacked bars: the single segment under the cursor, or null. */
export function hitTestStacked(canvas, mx, my) {
  const m = canvas._stackMeta;
  if (!m || !m.samples.length) return null;
  const n = m.samples.length;
  const slot = m.iw / n;
  const idx = Math.floor((mx - m.pad.l) / slot);
  if (idx < 0 || idx >= n) return null;
  const s = m.samples[idx];
  // walk segments bottom-up (first-seen service at the base), matching the paint order
  let yBase = m.pad.t + m.ih;
  for (let oi = 0; oi < m.order.length; oi++) {
    const name = m.order[oi];
    const value = s.vals[name] || 0;
    if (value <= 0) continue;
    const sh = (Math.min(value, m.maxV) / m.maxV) * m.ih;
    const yTop = yBase - sh;
    if (my >= yTop && my <= yBase) {
      const x = m.pad.l + idx * slot + slot / 2;
      return { name, value, color: STACK_COLORS[oi % STACK_COLORS.length], x, yTop };
    }
    yBase = yTop;
  }
  return null;
}
export function drawStackedBars(canvas, samples) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = canvas.clientHeight || 280;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px 'Segoe UI', system-ui, sans-serif";
  if (!samples.length) {
    ctx.fillStyle = "#9aa7bd";
    ctx.fillText("Waiting for live telemetry…", 16, 28);
    return;
  }
  // stable service order (first-seen first)
  const order = stackedOrder(samples);
  const totals = samples.map((s) => order.reduce((a, n) => a + (s.vals[n] || 0), 0));
  const maxV = niceCeil(Math.max(...totals, 10));
  // no y-axis gridlines/labels/title: values show on hover instead
  const pad = { l: 14, b: 32, t: 14, r: 12 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  // geometry for hover hit-testing (see hitTestStacked)
  canvas._stackMeta = { order, samples, pad: { l: pad.l, t: pad.t }, iw, ih, maxV };
  // bars
  const n = samples.length;
  const slot = iw / n;
  const bw = Math.min(26, Math.max(3, slot - 3));
  samples.forEach((s, idx) => {
    const x = pad.l + idx * slot + (slot - bw) / 2;
    // collect non-zero segments bottom-up so the top one gets rounded corners
    const segs = [];
    order.forEach((name, oi) => {
      const v = s.vals[name] || 0;
      if (v > 0) segs.push({ v, color: STACK_COLORS[oi % STACK_COLORS.length] });
    });
    let yBase = pad.t + ih;
    segs.forEach((sg, si) => {
      const sh = (Math.min(sg.v, maxV) / maxV) * ih;
      const y = yBase - sh;
      const g = ctx.createLinearGradient(0, y, 0, yBase);
      g.addColorStop(0, hexA(sg.color, 0.92));
      g.addColorStop(1, hexA(sg.color, 1));
      ctx.fillStyle = g;
      const isTop = si === segs.length - 1;
      if (isTop && ctx.roundRect && sh > 3) {
        ctx.beginPath();
        ctx.roundRect(x, y, bw, sh, [3, 3, 0, 0]);
        ctx.fill();
      } else {
        ctx.fillRect(x, y, bw, Math.max(sh, 1));
      }
      yBase = y;
    });
  });
  // x time labels from sample timestamps (~6 across)
  ctx.fillStyle = "#8a94a6";
  const ticks = Math.min(6, n);
  for (let i = 0; i < ticks; i++) {
    const idx = ticks === 1 ? 0 : Math.round((i * (n - 1)) / (ticks - 1));
    const x = pad.l + idx * slot + slot / 2;
    ctx.fillText(
      String(samples[idx].t || "").slice(0, 5),
      Math.min(Math.max(x - 12, pad.l), w - 44),
      h - 14
    );
  }
}

export function drawMultiLine(canvas, bucket, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = canvas.clientHeight || 280;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px 'Segoe UI', system-ui, sans-serif";
  const names = Object.keys(bucket);
  if (!names.length) {
    ctx.fillStyle = "#9aa7bd";
    ctx.fillText("Waiting for live telemetry…", 16, 28);
    return;
  }
  const palette = LINE_PALETTES[opts.colorSet === 1 ? "queue" : "cpu"];
  let maxV = opts.max || 0;
  names.forEach((n) => bucket[n].forEach((v) => {
    if (v > maxV) maxV = v;
  }));
  const fixed = opts.max > 0;
  if (maxV <= 0) maxV = 1;
  maxV = fixed ? opts.max : niceCeil(maxV * 1.1);
  const dec = maxV > 10 ? 0 : 2;
  const showLegend = opts.legend !== false;
  const pad = { l: 48, b: showLegend ? 56 : 30, t: 14, r: 12 };
  const { iw, ih } = paintGrid(ctx, w, h, pad, maxV, opts.unit || "", dec);
  // geometry for hover hit-testing (see hitTestLine)
  const drawn = names.slice(0, 12);
  canvas._chartMeta = {
    names: drawn,
    series: Object.fromEntries(drawn.map((n) => [n, bucket[n]])),
    colors: Object.fromEntries(drawn.map((n, i) => [n, palette[i % palette.length]])),
    pad, iw, ih, maxV, unit: opts.unit || "",
  };
  const yOf = (v) => pad.t + ih * (1 - Math.min(Math.max(v, 0), maxV) / maxV);
  names.slice(0, 12).forEach((name, i) => {
    const vals = bucket[name];
    const color = palette[i % palette.length];
    const xOf = (idx) => pad.l + (iw * idx) / Math.max(vals.length - 1, 1);
    const pts = vals.map((v, idx) => ({ x: xOf(idx), y: yOf(v) }));
    if (pts.length === 1) pts.push({ x: pts[0].x + 1, y: pts[0].y });
    // soft area fill under the curve
    ctx.save();
    ctx.beginPath();
    traceSmooth(ctx, pts);
    ctx.lineTo(pts[pts.length - 1].x, pad.t + ih);
    ctx.lineTo(pts[0].x, pad.t + ih);
    ctx.closePath();
    const ag = ctx.createLinearGradient(0, pad.t, 0, pad.t + ih);
    ag.addColorStop(0, hexA(color, 0.22));
    ag.addColorStop(1, hexA(color, 0.02));
    ctx.fillStyle = ag;
    ctx.fill();
    ctx.restore();
    // glowing smoothed stroke
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.shadowColor = hexA(color, 0.45);
    ctx.shadowBlur = 7;
    ctx.beginPath();
    traceSmooth(ctx, pts);
    ctx.stroke();
    ctx.restore();
    // live endpoint dot with white ring
    const end = pts[pts.length - 1];
    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(end.x, end.y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(end.x, end.y, 4, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  });
  const longest = Math.max(...names.map((k) => bucket[k].length), 1);
  paintTimeAxis(ctx, longest, pad.l, iw / Math.max(longest - 1, 1), w, showLegend ? h - 40 : h - 14);
  if (showLegend) {
    paintLegend(ctx, names.slice(0, 12), pad.l, w, h, pad.r, (r) => h - 24 + r * 14, palette);
    if (names.length > 12) {
      ctx.fillStyle = "#64748b";
      ctx.fillText(`+${names.length - 12} more`, w - 70, h - 10);
    }
  }
}
