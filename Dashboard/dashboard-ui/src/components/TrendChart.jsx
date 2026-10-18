import { buildPath, fmtNum } from "../utils.js";

export default function TrendChart({ values, color, label, thresholdFrac, svgId }) {
  const w = 560, h = 110, pad = 10;
  const result = buildPath(values, w, h, pad, true);
  const { line, area, min, max, lastPt } = result;
  if (!line) {
    return (
      <div className="trend-panel">
        <div className="trend-title">{label}</div>
        <div className="empty-note" style={{ padding: "10px" }}>Not enough historical samples for a trend line.</div>
      </div>
    );
  }
  const gradId = svgId || `trendGrad_${Math.random().toString(36).slice(2, 9)}`;
  let threshLine = null;
  if (thresholdFrac !== undefined && max > min) {
    const y = h - pad - (thresholdFrac - min) / (max - min) * (h - pad * 2);
    if (y > 0 && y < h) {
      threshLine = (
        <line x1={pad} y1={y.toFixed(1)} x2={w - pad} y2={y.toFixed(1)} className="threshold-line" />
      );
    }
  }
  const gridLines = [0.25, 0.5, 0.75].map((f, i) => {
    const y = pad + f * (h - pad * 2);
    return (
      <line
        key={i}
        x1={pad} y1={y.toFixed(1)}
        x2={w - pad} y2={y.toFixed(1)}
        className="trend-grid-line"
      />
    );
  });

  return (
    <div className="trend-panel">
      <div className="trend-head">
        <div className="trend-title">{label}</div>
        <div className="trend-legend">
          <span><span className="dot" style={{ background: color }}></span>observed</span>
          {threshLine ? <span><span className="dot" style={{ background: "var(--warning)" }}></span>alarm threshold</span> : null}
        </div>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.22" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        {gridLines}
        <path className="area" d={area} fill={`url(#${gradId})`} />
        <path className="line" d={line} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        {lastPt ? (
          <circle cx={lastPt[0].toFixed(1)} cy={lastPt[1].toFixed(1)} r="3.2" fill={color} stroke="#fff" strokeWidth="1.4" />
        ) : null}
        {threshLine}
      </svg>
      <div className="trend-axis-lab">
        <span>oldest sample</span>
        <span>min {fmtNum(min)} · max {fmtNum(max)}</span>
        <span>most recent</span>
      </div>
    </div>
  );
}
