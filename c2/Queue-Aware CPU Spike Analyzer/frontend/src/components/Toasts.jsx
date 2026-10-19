import { useEffect, useState } from "react";

let nextId = 1;

export default function Toasts() {
  const [items, setItems] = useState([]);

  useEffect(() => {
    const onToast = (e) => {
      const { msg, kind } = e.detail || {};
      const id = nextId++;
      setItems((prev) => [...prev, { id, msg, kind }]);
      setTimeout(() => {
        setItems((prev) => prev.filter((t) => t.id !== id));
      }, 5200);
    };
    window.addEventListener("qa-toast", onToast);
    return () => window.removeEventListener("qa-toast", onToast);
  }, []);

  return (
    <div id="toasts">
      {items.map((t) => (
        <div key={t.id} className={`toast ${t.kind || ""}`}>
          {t.msg}
        </div>
      ))}
    </div>
  );
}
