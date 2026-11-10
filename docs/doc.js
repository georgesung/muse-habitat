// Renders a markdown doc into a docs-style page: <article id="doc" data-src="name.md">, <ol id="toc">.
(async function () {
  const doc = document.getElementById("doc");
  const toc = document.getElementById("toc");
  // Same anchors as GitHub's rendering of the markdown, so links work in both places.
  const slug = (t) => t.trim().toLowerCase().replace(/[^\p{L}\p{N}\s-]/gu, "").replace(/\s/g, "-");
  let md;
  try {
    const r = await fetch(doc.dataset.src, { cache: "no-cache" });
    if (!r.ok) throw new Error(r.status);
    md = await r.text();
    if (!window.marked) throw new Error("markdown renderer did not load");
  } catch (e) {
    return; // the fallback link stays
  }
  doc.innerHTML = marked.parse(md);
  const used = {};
  for (const h of doc.querySelectorAll("h1, h2, h3")) {
    let id = slug(h.textContent);
    if (used[id] !== undefined) id = `${id}-${++used[id]}`; else used[id] = 0;
    h.id = id;
    if (h.tagName === "H1") continue;
    const li = document.createElement("li");
    li.className = h.tagName === "H3" ? "sub" : "";
    const a = document.createElement("a");
    a.href = "#" + id;
    a.textContent = h.textContent;
    li.appendChild(a);
    toc.appendChild(li);
  }
  for (const t of doc.querySelectorAll("table")) {
    const box = document.createElement("div");
    box.className = "table-scroll";
    t.replaceWith(box);
    box.appendChild(t);
  }
  for (const a of doc.querySelectorAll('a[href^="http"]')) a.rel = "noopener";
  if (window.matchMedia("(max-width: 900px)").matches) document.getElementById("toc-box").open = false;
  if (location.hash) {
    const el = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (el) el.scrollIntoView();
  }
  // Highlight the section being read.
  const links = [...toc.querySelectorAll("a")];
  const heads = links.map((a) => document.getElementById(a.getAttribute("href").slice(1)));
  const onScroll = () => {
    let cur = 0;
    heads.forEach((h, i) => { if (h.getBoundingClientRect().top < 120) cur = i; });
    links.forEach((a, i) => a.classList.toggle("active", i === cur));
  };
  document.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
})();
