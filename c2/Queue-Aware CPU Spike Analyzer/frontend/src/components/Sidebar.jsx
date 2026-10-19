export const PAGES = [
  { id: "dashboard", label: "Dashboard", ico: "▨" },
  { id: "services", label: "Services", ico: "✚" },
  { id: "configuration", label: "Configuration", ico: "⚙" },
  { id: "dataset", label: "Your Services data", ico: "☳" },
  { id: "experiments", label: "Experiments", ico: "⚡︎" },
];

export default function Sidebar({ page, onNav, connected }) {
  return (
    <aside className="sidebar">
      <div className="g-logo" title="Sentinel">
        <div className="g-logo-mark">◤</div>
        <span className="lbl">Sentinel</span>
      </div>
      {PAGES.map((p) => (
        <button
          key={p.id}
          className={`nav-btn${page === p.id ? " active" : ""}`}
          id={`nav-${p.id}`}
          title={p.label}
          onClick={() => onNav(p.id)}
        >
          <span className="ico">{p.ico}</span>
          {p.label}
        </button>
      ))}
    </aside>
  );
}
