// Fill these in before publishing.
const CONFIG = {
  REPO_URL: "https://github.com/georgesung/muse-habitat",
  HF_URL: "https://huggingface.co/georgesung/Muse-Glimmer-30B-Habitat-PARTNR-Lora",
  DEMO_VIDEO_ID: "yy8IwvziNfA",   // YouTube video id of the 3-minute demo
};

const CATS = {
  rl_win: "RL wins (same result twice)",
  rl_loss: "Original wins (same result twice)",
  noisy: "Mixed / run-to-run noise",
  all_success: "All models succeed",
  all_fail: "All models fail",
};
const MODELS = { original: "Original Glimmer", rl: "Glimmer + RL", rl_longer: "RL, trained longer" };
const PAGE = 50;

let EPISODES = [];
let VIDEOS = [];  // data/videos.json: featured episodes in display order
let filtered = [];
let page = 0;

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
}

function outcome(r) {
  if (!r || r.success === null || r.crashed) return el("span", { class: "res crash", text: "crash" });
  return r.success ? el("span", { class: "res ok", text: "✓ success" }) : el("span", { class: "res bad", text: "✗ fail" });
}

function youtube(id, title) {
  return el("iframe", {
    src: `https://www.youtube-nocookie.com/embed/${encodeURIComponent(id)}`, title, loading: "lazy",
    allow: "accelerometer; encrypted-media; gyroscope; picture-in-picture; fullscreen", allowfullscreen: "",
  });
}

function setLinks() {
  for (const [id, url] of [["repo-link", CONFIG.REPO_URL], ["repo-link-2", CONFIG.REPO_URL], ["hf-link", CONFIG.HF_URL]]) {
    const a = document.getElementById(id);
    if (url) a.href = url;
    else a.setAttribute("aria-disabled", "true");
  }
  if (CONFIG.DEMO_VIDEO_ID) {
    const f = document.getElementById("demo-video");
    f.replaceChildren(youtube(CONFIG.DEMO_VIDEO_ID, "3-minute demo"));
  }
}

function renderVideos() {
  const grid = document.getElementById("video-grid");
  const byId = Object.fromEntries(EPISODES.map((e) => [e.id, e]));
  for (const v of VIDEOS) {
    const id = v.id;
    const ep = byId[id];
    const frame = el("div", { class: "video-frame" });
    const yt = (v.youtube && (v.youtube.stacked || v.youtube.rl || v.youtube.original)) || "";
    frame.append(yt ? youtube(yt, v.title) : el("div", { class: "soon", text: "Video coming soon" }));
    const meta = el("div", { class: "meta" }, el("span", { class: `tag ${v.category}`, text: CATS[v.category] || v.category }));
    if (ep) {
      meta.append(el("span", { class: "tag", text: `House ${ep.house}` }));
    }
    const body = el("div", { class: "body" },
      meta,
      el("h3", { text: v.title }),
      ep ? el("div", { class: "meta" }, el("span", { class: "tag", text: "Original" }), outcome(ep.original),
        el("span", { class: "tag", text: "Glimmer + RL" }), outcome(ep.rl)) : null,
      v.why_original ? el("p", { class: "why" }, el("strong", { text: "Original: " }), v.why_original) : null,
      v.why_rl ? el("p", { class: "why" }, el("strong", { text: "Glimmer + RL: " }), v.why_rl) : null,
      el("button", { class: "btn", type: "button", text: `Episode ${id}: reasoning, turn by turn`, onclick: () => openEpisode(Number(id)) }));
    grid.append(el("div", { class: "vcard" }, frame, body));
  }
}

function initFilters() {
  const cat = document.getElementById("f-cat");
  cat.append(el("option", { value: "", text: "any" }));
  const counts = {};
  for (const e of EPISODES) counts[e.category] = (counts[e.category] || 0) + 1;
  for (const k of ["rl_win", "rl_loss", "noisy", "all_success", "all_fail"]) {
    if (counts[k]) cat.append(el("option", { value: k, text: `${CATS[k]} (${counts[k]})` }));
  }
  const house = document.getElementById("f-house");
  house.append(el("option", { value: "", text: "any" }));
  const hs = [...new Set(EPISODES.map((e) => e.house))].sort((a, b) => a - b);
  for (const h of hs) house.append(el("option", { value: String(h), text: `House ${h}` }));
  for (const id of ["f-search", "f-cat", "f-house", "f-orig", "f-rl"]) {
    document.getElementById(id).addEventListener("input", () => { page = 0; applyFilters(); });
  }
}

function applyFilters() {
  const q = document.getElementById("f-search").value.trim().toLowerCase();
  const cat = document.getElementById("f-cat").value;
  const house = document.getElementById("f-house").value;
  const fo = document.getElementById("f-orig").value;
  const fp = document.getElementById("f-rl").value;
  const ok = (r, want) => want === "" || (want === "1" ? r.success === true : r.success !== true);
  filtered = EPISODES.filter((e) =>
    (!q || e.instruction.toLowerCase().includes(q) || String(e.id) === q) &&
    (!cat || e.category === cat) && (!house || String(e.house) === house) &&
    ok(e.original, fo) && ok(e.rl, fp));
  renderTable();
}

function renderTable() {
  const t = document.getElementById("table");
  const pages = Math.max(1, Math.ceil(filtered.length / PAGE));
  page = Math.min(page, pages - 1);
  document.getElementById("count").textContent =
    `${filtered.length} episode${filtered.length === 1 ? "" : "s"}` +
    (filtered.length ? `; Glimmer + RL succeeds on ${filtered.filter((e) => e.rl.success).length}, original on ${filtered.filter((e) => e.original.success).length}` : "");
  const head = el("div", { class: "row head" },
    el("span", { text: "#" }), el("span", { class: "house", text: "House" }), el("span", { class: "instr", text: "Instruction" }),
    el("span", { text: "Original" }), el("span", { text: "Glimmer + RL" }), el("span", { text: "RL, trained longer" }), el("span", { class: "cat", text: "Category" }));
  const rows = filtered.slice(page * PAGE, (page + 1) * PAGE).map((e) =>
    el("div", { class: "row", role: "button", tabindex: "0", onclick: () => openEpisode(e.id),
      onkeydown: (ev) => { if (ev.key === "Enter") openEpisode(e.id); } },
      el("span", { text: String(e.id) }), el("span", { class: "house", text: String(e.house) }),
      el("span", { class: "instr", text: e.instruction, title: e.instruction }),
      outcome(e.original), outcome(e.rl), outcome(e.rl_longer),
      el("span", { class: "cat" }, el("span", { class: `tag ${e.category}`, text: CATS[e.category] || e.category }))));
  t.replaceChildren(head, ...rows);
  const pager = document.getElementById("pager");
  pager.replaceChildren(
    el("button", { class: "btn", type: "button", text: "← Prev", onclick: () => { page--; renderTable(); } }),
    el("span", { text: `Page ${page + 1} of ${pages}` }),
    el("button", { class: "btn", type: "button", text: "Next →", onclick: () => { page++; renderTable(); } }));
  pager.children[0].disabled = page === 0;
  pager.children[2].disabled = page >= pages - 1;
}

function turnView(t, i) {
  const who = t.agent === 0 ? "Robot" : t.agent === 1 ? "Human" : "Agent ?";
  const head = el("div", {}, el("span", { class: `who a${t.agent}`, text: `${i + 1}. ${who}` }), " ", el("code", { text: t.call }));
  const box = el("div", { class: "turn" }, head);
  if (t.fallback) box.append(el("div", { class: "result", text: "(the free-form call was malformed; constrained fallback)" }));
  if (t.thinking) {
    box.append(el("details", {}, el("summary", { text: "thinking" }), el("div", { class: "think", text: t.thinking })));
  }
  if (t.result) {
    const bad = /fail|error|not /i.test(t.result) && !/^Successful/.test(t.result);
    box.append(el("div", { class: `result ${bad ? "bad" : "ok"}`, text: "→ " + t.result }));
  }
  return box;
}

async function openEpisode(id) {
  const e = EPISODES.find((x) => x.id === id);
  const dlg = document.getElementById("dlg");
  document.getElementById("dlg-title").textContent = `Episode ${id} · House ${e.house} · ${e.instruction}`;
  const body = document.getElementById("dlg-body");
  body.replaceChildren(el("p", { text: "Loading..." }));
  if (!dlg.open) dlg.showModal();
  let data;
  try {
    data = await (await fetch(`data/ep/${id}.json`)).json();
  } catch (err) {
    body.replaceChildren(el("p", { text: "Could not load this episode's transcripts." }));
    return;
  }
  const cards = el("div", { class: "outcomes" });
  const stat = (label, r) => el("div", { class: "card" }, el("div", { class: "k", text: label }), outcome(r),
    r && r.steps != null ? el("div", { class: "k", text: `${r.steps.toLocaleString()} steps · ${Math.round(r.percent * 100)}% of goals` }) : null);
  for (const m of Object.keys(MODELS)) cards.append(stat(MODELS[m], e[m]));
  if (e.recheck) for (const m of Object.keys(MODELS)) cards.append(stat(`${MODELS[m]} (re-run)`, e.recheck[m]));
  const parts = [el("div", { class: "meta" }, el("span", { class: `tag ${e.category}`, text: CATS[e.category] || e.category })), cards];
  const v = VIDEOS.find((x) => x.id === id);
  const yt = v && v.youtube && (v.youtube.stacked || v.youtube.rl || v.youtube.original);
  if (yt) parts.push(el("div", { class: "video-frame", style: "aspect-ratio:16/9;max-width:720px;margin:0 auto 14px" }, youtube(yt, v.title)));
  const runs = Object.keys(data.runs);
  const label = (k) => k.replace(/^original/, MODELS.original).replace(/^rl_longer/, MODELS.rl_longer).replace(/^rl/, MODELS.rl);
  const col = (initial) => {
    const sel = el("select", {});
    for (const k of runs) sel.append(el("option", { value: k, text: `${label(k)} (${data.runs[k].length} turns)` }));
    sel.value = runs.includes(initial) ? initial : runs[0];
    const list = el("div", {});
    const draw = () => list.replaceChildren(...(data.runs[sel.value] || []).map(turnView));
    sel.addEventListener("input", draw);
    draw();
    return el("div", { class: "col" }, sel, list);
  };
  parts.push(el("div", { class: "cols" }, col("original"), col("rl")));
  body.replaceChildren(...parts);
}

async function main() {
  setLinks();
  document.getElementById("dlg-close").addEventListener("click", () => document.getElementById("dlg").close());
  document.getElementById("dlg").addEventListener("click", (ev) => { if (ev.target.id === "dlg") ev.target.close(); });
  try {
    const [eps, vids] = await Promise.all([fetch("data/episodes.json").then((r) => r.json()),
      fetch("data/videos.json").then((r) => r.json())]);
    EPISODES = eps.episodes;
    VIDEOS = vids;
  } catch (err) {
    document.getElementById("count").textContent = "Could not load the episode data (serve this folder over HTTP).";
    return;
  }
  renderVideos();
  initFilters();
  applyFilters();
  const m = location.hash.match(/^#ep(\d+)$/);
  if (m) openEpisode(Number(m[1]));
}

main();
