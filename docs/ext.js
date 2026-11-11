// Open links to other sites in a new tab (also covers links added later by app.js / doc.js).
document.addEventListener("click", (e) => {
  const a = e.target.closest("a[href]");
  if (a && /^https?:$/.test(a.protocol) && a.host !== location.host) {
    a.target = "_blank";
    a.rel = "noopener";
  }
}, true);
