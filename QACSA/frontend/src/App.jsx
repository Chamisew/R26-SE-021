import React, { useState, useEffect, useMemo } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, ComposedChart, Cell, PieChart, Pie, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis 
} from 'recharts';
import { 
  Activity, Database, Server, Cpu, Layers, TrendingUp, ShieldAlert, Zap, Terminal, ShieldCheck, ArrowUpRight, ArrowDownRight, RefreshCw, Download
} from 'lucide-react';
import './index.css';

// --- Components ---

const TopBar = ({ title }) => (
  <header className="topbar">
    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
      <div className="logo-icon" style={{ width: '28px', height: '28px', background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white' }}>
        <Zap size={16} fill="white" />
      </div>
      <h1 style={{ fontSize: '1rem', margin: 0, fontWeight: 800 }}>{title}</h1>
    </div>
    <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
      <div className="status-pill status-normal">
        <RefreshCw size={12} className="spin" /> SYNC: 1000ms
      </div>
      <a href="http://localhost:8081/api/download" className="status-pill" style={{ background: 'var(--accent)', color: 'white', textDecoration: 'none', cursor: 'pointer', gap: '0.5rem' }}>
        <Download size={14} /> EXPORT CSV
      </a>
    </div>
    <style>{`.spin { animation: spin 2s linear infinite; } @keyframes spin { 100% { transform: rotate(360deg); } }`}</style>
  </header>
);

const MetricCard = ({ label, value, subValue, trend, icon: Icon, color }) => (
  <div className="card">
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
      <div>
        <div className="stat-label">{label}</div>
        <div className="stat-value" style={{ color: color || 'var(--text-primary)' }}>{value}</div>
      </div>
      <div style={{ background: 'var(--bg-tertiary)', padding: '0.5rem', borderRadius: '8px', color: color || 'var(--accent)' }}>
        <Icon size={20} />
      </div>
    </div>
    <div style={{ marginTop: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.7rem', fontWeight: 600 }}>
      {trend === 'up' ? <ArrowUpRight size={14} color="#ef4444" /> : <ArrowDownRight size={14} color="#10b981" />}
      <span style={{ color: trend === 'up' ? '#ef4444' : '#10b981' }}>{subValue}</span>
    </div>
  </div>
);

function App() {
  const [dataHistory, setDataHistory] = useState([]);
  const [currentMetrics, setCurrentMetrics] = useState(null);

  useEffect(() => {
    const interval = setInterval(() => {
      fetch('http://localhost:8081/api/data')
        .then(res => res.json())
        .then(data => {
          if (data && data.time) {
            setCurrentMetrics(data);
            setDataHistory(prev => {
              const newHistory = [...prev, data];
              return newHistory.length > 100 ? newHistory.slice(newHistory.length - 100) : newHistory;
            });
          }
        })
        .catch(err => console.error(err));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const stats = useMemo(() => {
    if (!dataHistory.length) return { total: 0, failures: 0, totalIncidents: 0, failureRate: 0 };
    const total = dataHistory.length;
    const failures = dataHistory.filter(h => h.system_state !== 'NORMAL').length;
    const totalIncidents = new Set(dataHistory.map(h => h.incident_id).filter(id => id > 0)).size;
    return { total, failures, totalIncidents, failureRate: ((failures / total) * 100).toFixed(1) };
  }, [dataHistory]);

  if (!currentMetrics) return <div style={{ height: '100vh', background: 'var(--bg-primary)' }}></div>;

  const lastMetrics = dataHistory.length > 1 ? dataHistory[dataHistory.length - 2] : currentMetrics;

  return (
    <div className="app-layout">
      <TopBar title="QUEUE-AWARE RESEARCH COMMAND CENTER" />
      <main className="main-content">
        <div className="command-center">
          
          {/* --- Global Health Overview --- */}
          <div style={{ gridColumn: 'span 3' }}><MetricCard label="Queue Pressure" value={(currentMetrics.queue_pressure_index * 100).toFixed(1) + '%'} subValue="Live Index" trend={currentMetrics.queue_pressure_index > lastMetrics.queue_pressure_index ? 'up' : 'down'} icon={Activity} color={currentMetrics.queue_pressure_index > 0.7 ? '#ef4444' : '#3b82f6'} /></div>
          <div style={{ gridColumn: 'span 3' }}><MetricCard label="System State" value={currentMetrics.system_state} subValue={currentMetrics.incident_phase} trend={currentMetrics.system_state === 'NORMAL' ? 'down' : 'up'} icon={ShieldAlert} color={currentMetrics.system_state === 'FAILED' ? '#ef4444' : (currentMetrics.system_state === 'NORMAL' ? '#10b981' : '#facc15')} /></div>
          <div style={{ gridColumn: 'span 3' }}><MetricCard label="Active Mesh" value={currentMetrics.services?.length || 0} subValue="Containers" trend="down" icon={Server} color="var(--accent)" /></div>
          <div style={{ gridColumn: 'span 3' }}><MetricCard label="ML Label" value={currentMetrics.label === 1 ? 'ANOMALY' : 'NORMAL'} subValue="Prediction" trend={currentMetrics.label === 1 ? 'up' : 'down'} icon={ShieldCheck} color={currentMetrics.label === 1 ? '#ef4444' : '#10b981'} /></div>

          {/* --- Telemetry Analytics Row --- */}
          <div className="card" style={{ gridColumn: 'span 8' }}>
            <div className="card-title"><TrendingUp size={16} color="var(--accent)" /> Resource Dynamics & Velocity</div>
            <div style={{ height: 260 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={dataHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis dataKey="time" hide />
                  <YAxis domain={[0, 100]} stroke="var(--text-muted)" fontSize={11} />
                  <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }} />
                  <Legend verticalAlign="top" align="right" height={36} iconType="circle" />
                  <Area type="monotone" dataKey="cpu_percent" fill="var(--accent)" fillOpacity={0.08} stroke="var(--accent)" name="Global CPU" strokeWidth={2} />
                  <Line type="monotone" dataKey="cpu_velocity" stroke="#ef4444" strokeWidth={1} dot={false} name="Velocity" />
                  <Line type="monotone" dataKey="cpu_trend_5min" stroke="#a855f7" strokeWidth={2} dot={false} name="5m Trend" />
                  <Line type="monotone" dataKey="cpu_trend_10min" stroke="#ec4899" strokeWidth={1.5} dot={false} name="10m Trend" strokeDasharray="4 4" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card" style={{ gridColumn: 'span 4' }}>
            <div className="card-title"><Layers size={16} color="#facc15" /> Workload Balance (λ vs μ)</div>
            <div style={{ height: 260 }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={dataHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis dataKey="time" hide />
                  <YAxis stroke="var(--text-muted)" fontSize={10} />
                  <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }} />
                  <Area type="step" dataKey="incoming_rate" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.1} name="λ" />
                  <Area type="step" dataKey="processing_rate" stroke="#facc15" fill="#facc15" fillOpacity={0.1} name="μ" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* --- Deep Research Insights (Models & Experiments) --- */}
          <div className="card" style={{ gridColumn: 'span 6' }}>
            <div className="card-title"><ShieldCheck size={16} color="var(--accent)" /> Anomaly Prediction Models</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '1rem' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', justifyContent: 'center' }}>
                <div style={{ textAlign: 'center' }}><div style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>F1-SCORE</div><div style={{ fontSize: '1.25rem', fontWeight: 800 }}>0.91</div></div>
                <div style={{ textAlign: 'center' }}><div style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>PRECISION</div><div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#10b981' }}>0.94</div></div>
                <div style={{ textAlign: 'center' }}><div style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>RECALL</div><div style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--accent)' }}>0.88</div></div>
              </div>
              <div style={{ height: 200 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="80%" data={[
                    { subject: 'CPU', A: 85 }, { subject: 'QPI', A: 98 }, { subject: 'λ', A: 70 }, { subject: 'μ', A: 60 }, { subject: 'Growth', A: 90 }
                  ]}>
                    <PolarGrid stroke="var(--border)" />
                    <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                    <Radar name="Importance" dataKey="A" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.4} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="card" style={{ gridColumn: 'span 6' }}>
            <div className="card-title"><Zap size={16} color="#ef4444" /> Fault Injection Analytics</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div style={{ height: 200 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={[
                      { name: 'Healthy', value: stats.total - stats.failures },
                      { name: 'Anomaly', value: stats.failures }
                    ]} innerRadius={50} outerRadius={70} paddingAngle={5} dataKey="value">
                      <Cell fill="#10b981" /><Cell fill="#ef4444" />
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', justifyContent: 'center' }}>
                <div className="status-pill status-normal" style={{ justifyContent: 'space-between', width: '100%' }}><span>Incidents:</span> <b>{stats.totalIncidents}</b></div>
                <div className="status-pill status-critical" style={{ justifyContent: 'space-between', width: '100%' }}><span>Degradation:</span> <b>{stats.failureRate}%</b></div>
              </div>
            </div>
          </div>

          {/* --- Operational Mesh Row --- */}
          <div className="card" style={{ gridColumn: 'span 12' }}>
            <div className="card-title"><Server size={16} color="var(--accent)" /> Service Mesh Cluster Health</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '1rem', maxHeight: '250px', overflowY: 'auto' }}>
              {(currentMetrics.services || []).map((s, i) => (
                <div key={i} style={{ background: 'var(--bg-tertiary)', padding: '0.75rem', borderRadius: '8px', borderLeft: `4px solid ${s.cpu > 80 ? '#ef4444' : '#10b981'}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <span style={{ fontWeight: 700, fontSize: '0.8rem' }}>{s.name}</span>
                    <span style={{ fontSize: '0.8rem', fontWeight: 800 }}>{s.cpu.toFixed(1)}%</span>
                  </div>
                  <div style={{ height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', overflow: 'hidden' }}>
                    <div style={{ width: `${s.cpu}%`, height: '100%', background: s.cpu > 80 ? '#ef4444' : 'var(--accent)', transition: 'width 0.5s ease' }}></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* --- Research Feed Row --- */}
          <div className="card" style={{ gridColumn: 'span 4', background: '#020617', border: '1px solid #1e293b', fontFamily: 'monospace' }}>
            <div className="card-title" style={{ color: '#10b981' }}><Terminal size={16} /> Console</div>
            <div style={{ height: '350px', overflowY: 'auto', fontSize: '0.7rem', color: '#94a3b8' }}>
              {[...dataHistory].reverse().map((h, i) => (
                <div key={i} style={{ marginBottom: '0.2rem' }}>
                  <span style={{ color: '#64748b' }}>[{h.time}]</span> <span style={{ color: h.system_state === 'FAILED' ? '#ef4444' : '#3b82f6' }}>{h.system_state}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="card" style={{ gridColumn: 'span 8', padding: 0, overflow: 'hidden' }}>
            <div className="card-title" style={{ padding: '1rem 1rem 0.5rem 1rem' }}><Database size={16} color="var(--accent)" /> Live Telemetry Dataset Feed</div>
            <div style={{ height: '382px', overflowY: 'auto' }}>
              <table>
                <thead>
                  <tr><th>TIME</th><th>STATE</th><th>CPU%</th><th>VEL</th><th>5M MA</th><th>10M MA</th><th>Q-SIZE</th><th>QPI</th><th>LABEL</th></tr>
                </thead>
                <tbody>
                  {[...dataHistory].reverse().map((row, i) => (
                    <tr key={i} style={{ background: row.system_state === 'FAILED' ? 'rgba(239, 68, 68, 0.05)' : 'transparent' }}>
                      <td>{row.time}</td>
                      <td><span className={`status-pill ${row.system_state === 'FAILED' ? 'status-critical' : 'status-normal'}`} style={{ fontSize: '0.5rem' }}>{row.system_state}</span></td>
                      <td style={{ fontWeight: 800 }}>{row.cpu_percent.toFixed(1)}%</td>
                      <td style={{ color: row.cpu_velocity > 10 ? '#ef4444' : 'var(--text-secondary)' }}>{row.cpu_velocity.toFixed(1)}</td>
                      <td style={{ color: 'var(--accent)' }}>{row.cpu_trend_5min?.toFixed(1) || '0.0'}</td>
                      <td style={{ color: '#8b5cf6' }}>{row.cpu_trend_10min?.toFixed(1) || '0.0'}</td>
                      <td>{row.in_flight_queue}</td>
                      <td>{(row.queue_pressure_index * 100).toFixed(0)}%</td>
                      <td style={{ color: row.label === 1 ? '#ef4444' : '#10b981', fontWeight: 800 }}>{row.label}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}

export default App;
