import { useEffect, useRef, useState } from "react";
import { drawMultiLine, hitTestLine } from "../charts.js";

/* Line chart with hover tooltip (no under-chart legend).
   Redraws whenever `tick` changes identity; `bucket` is {name: [vals]}.
   Pass draw={drawAreaChart} for the filled area variant. */
export default function LineChart({ bucket, opts, tick, draw }) {
  const canvasRef = useRef(null);
  const [tip, setTip] = useState(null);
  const drawFn = draw || drawMultiLine;

  const redraw = () => {
    if (canvasRef.current) drawFn(canvasRef.current, bucket, { ...(opts || {}), legend: false });
  };

  useEffect(() => {
    redraw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick]);

  useEffect(() => {
    window.addEventListener("resize", redraw);
    return () => window.removeEventListener("resize", redraw);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick]);

  const onMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const hit = hitTestLine(canvas, e.clientX - rect.left, e.clientY - rect.top);
    if (!hit) {
      setTip(null);
      return;
    }
    const unit = (opts || {}).unit || "";
    const value = unit === "%" ? `${hit.value.toFixed(1)}${unit}` : `${Math.round(hit.value * 100) / 100}`;
    setTip({
      name: hit.name,
      value,
      color: hit.color,
      left: Math.min(Math.max(hit.x, 70), Math.max(rect.width - 70, 70)),
      top: Math.max(hit.y, 30),
    });
  };

  return (
    <div className="chart-wrap">
      <canvas className="chart" ref={canvasRef} onMouseMove={onMove} onMouseLeave={() => setTip(null)} />
      {tip && (
        <div className="chart-tip" style={{ left: tip.left, top: tip.top }}>
          <span className="dot" style={{ background: tip.color }} />
          <b>{tip.name}</b>
          <span>{tip.value}</span>
        </div>
      )}
    </div>
  );
}
