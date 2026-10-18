/* Shared API client, settings store, and toast bus. */

export const store = {
  get backend() {
    return localStorage.getItem("qa.backend") || window.location.origin;
  },
  set backend(v) {
    localStorage.setItem("qa.backend", v);
  },
  get refreshMs() {
    return parseInt(localStorage.getItem("qa.refreshMs") || "1000", 10);
  },
  set refreshMs(v) {
    localStorage.setItem("qa.refreshMs", String(v));
  },
};

export async function api(path, opts = {}) {
  const url = store.backend.replace(/\/$/, "") + path;
  const res = await fetch(url, {
    ...opts,
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
  });
  const text = await res.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { _raw: text };
  }
  if (!res.ok) throw new Error((body && body.error) || `HTTP ${res.status}`);
  return body;
}

/* Global toast bus: toast(msg, kind) -> Toasts component renders it. */
export function toast(msg, kind = "") {
  window.dispatchEvent(new CustomEvent("qa-toast", { detail: { msg, kind } }));
}

export function stateColor(s) {
  s = String(s || "");
  if (/FAILED|CRITICAL/.test(s)) return "#dc2626";
  if (/DANGEROUS/.test(s)) return "#ea580c";
  if (/QUEUE_BOUND|CPU_BOUND|DEGRADED/.test(s)) return "#d97706";
  if (/HEALTHY/.test(s)) return "#16a34a";
  return "#2563eb";
}

export function pillClass(v) {
  return `pill p-${String(v || "unknown").toLowerCase()}`;
}
