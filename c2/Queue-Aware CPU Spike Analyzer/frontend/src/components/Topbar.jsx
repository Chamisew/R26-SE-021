export default function Topbar({ query, onQuery, onRefresh, onSettings }) {
  return (
    <div className="topbar">
      <label className="search">
        <span className="s-ico">⌕</span>
        <input
          id="topSearch"
          type="text"
          placeholder="Search services…"
          value={query}
          onChange={(e) => onQuery(e.target.value)}
        />
      </label>
      <div className="spacer" />
      <button className="icon-btn" onClick={onRefresh} title="Refresh now">
        ⟳
      </button>
      <button className="icon-btn" onClick={onSettings} title="Settings">
        ⚙
      </button>
    </div>
  );
}
