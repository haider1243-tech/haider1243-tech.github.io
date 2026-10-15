/* Shared helpers for index.html and resume.html */
window.Site = (() => {
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // Escape, then allow **bold**.
  const rich = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");

  function parseDate(s) {
    if (!s) return null;
    if (s === "present") {
      const d = new Date();
      return { y: d.getFullYear(), m: d.getMonth() + 1, present: true };
    }
    const [y, m] = String(s).split("-").map(Number);
    return { y, m: m || null };
  }

  function fmtDate(s) {
    const d = parseDate(s);
    if (!d) return "";
    if (d.present) return "Present";
    return d.m ? `${MONTHS[d.m - 1]} ${d.y}` : String(d.y);
  }

  const fmtRange = (e) => (e.end ? `${fmtDate(e.start)} – ${fmtDate(e.end)}` : fmtDate(e.start));

  function duration(e) {
    if (!e.end) return "";
    const a = parseDate(e.start), b = parseDate(e.end);
    if (a.m && b.m) {
      const total = (b.y - a.y) * 12 + (b.m - a.m) + 1;
      const y = Math.floor(total / 12), m = total % 12;
      return [y && `${y} yr${y > 1 ? "s" : ""}`, m && `${m} mo${m > 1 ? "s" : ""}`].filter(Boolean).join(" ");
    }
    const y = b.y - a.y;
    return y > 0 ? `${y} yr${y > 1 ? "s" : ""}` : "";
  }

  // Newest first: by start date, ongoing items break ties.
  const byNewest = (a, b) => {
    const k = (e) => { const d = parseDate(e.start); return d.y * 12 + (d.m || 1); };
    return k(b) - k(a) || (b.end === "present") - (a.end === "present");
  };

  return { esc, rich, parseDate, fmtDate, fmtRange, duration, byNewest };
})();
