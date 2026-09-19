/* ================================================================================
   CivicProof — single-file frontend
   -------------------------------------------------------------------------------
   1.  Utilities                  6.  Charts (donut / bars)
   2.  API client                 7.  Upvotes
   3.  Meta + HTML builders       8.  Shared report card
   4.  Toasts & modal             9-12. Citizen pages (home / explore / report / issue)
   5.  City map engine            13. Authority console   14. Router & boot
   ================================================================================ */
"use strict";

/* ============================== 1. UTILITIES ================================ */
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const ic = (name, cls = "") => `<svg class="ic ${cls}"><use href="#i-${name}"/></svg>`;
const debounce = (fn, ms = 300) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

function timeAgo(ts) {
  const s = Math.max(1, Date.now() / 1000 - ts);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  if (s < 86400 * 30) return `${Math.floor(s / 86400)}d ago`;
  return new Date(ts * 1000).toLocaleDateString();
}
const fmtNum = (n) => (n ?? 0).toLocaleString("en-IN");

/* ============================== 2. API CLIENT =============================== */
async function _json(r) { const d = await r.json(); if (!d.ok) throw new Error(d.error || "Request failed"); return d; }
const api = {
  get: (url) => fetch(url).then(_json),
  postJSON: (url, body) => fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(_json),
  postForm: (url, formData) => fetch(url, { method: "POST", body: formData }).then(_json),
};

/* ====================== 3. META + HTML BUILDERS ============================= */
const CAT = {
  pothole:     { label: "Pothole / Road Damage", short: "Pothole",       color: "#c2410c", icon: "road" },
  garbage:     { label: "Garbage & Sanitation",  short: "Garbage",       color: "#a16207", icon: "trash" },
  streetlight: { label: "Streetlight Fault",     short: "Streetlight",   color: "#6d28d9", icon: "lamp" },
  water_leak:  { label: "Water Leak / Flooding", short: "Water",         color: "#0369a1", icon: "drop" },
  road_crack:  { label: "Road Crack / Cave-in",  short: "Crack",         color: "#be123c", icon: "crack" },
  other:       { label: "Other Civic Issue",     short: "Other",         color: "#475569", icon: "flag" },
};
const catOf = (c) => CAT[c] || CAT.other;

const STATUS = {
  open:        { label: "Open",        color: "#0369a1", icon: "zap",          badge: "badge-open" },
  in_progress: { label: "In Progress", color: "#b45309", icon: "wrench",       badge: "badge-in_progress" },
  reopened:    { label: "Reopened",    color: "#dc2626", icon: "alert",        badge: "badge-reopened" },
  resolved:    { label: "Resolved",    color: "#059669", icon: "check-circle", badge: "badge-resolved" },
  duplicate:   { label: "Duplicate",   color: "#64748b", icon: "merge",        badge: "badge-duplicate" },
};
const statusOf = (s) => STATUS[s] || STATUS.open;

const PRIORITY = { Critical: "#b91c1c", High: "#c2410c", Medium: "#b45309", Low: "#059669" };
const priColor = (label) => PRIORITY[label] || PRIORITY.Medium;

const badgeHTML = (rep) =>
  `<span class="badge ${statusOf(rep.status).badge}"><span class="dot"></span>${statusOf(rep.status).label}</span>${rep.flagged ? ' <span class="badge badge-flagged"><span class="dot"></span>Field check</span>' : ""}`;

const ringHTML = (score, label, lg = false) =>
  `<div class="ring ${lg ? "lg" : ""}" style="--p:${Math.round(score)};--pc:${priColor(label)}" title="AI priority: ${label} (${Math.round(score)})"><span>${Math.round(score)}</span></div>`;

const catChipHTML = (rep, withAI = true) =>
  `<span class="chip chip-cat" style="--c:${catOf(rep.category).color}">${ic(catOf(rep.category).icon)}${catOf(rep.category).short}</span>${withAI && rep.ai_confidence ? `<span class="chip" title="AI classification confidence">${ic("brain")}${Math.round(rep.ai_confidence)}%</span>` : ""}`;

/* ========================== 4. TOASTS & MODAL =============================== */
function toast(msg, type = "ok", ms = 3600) {
  const map = { ok: ["check-circle", "#059669"], err: ["x-circle", "#dc2626"], info: ["info", "#0369a1"], warn: ["alert", "#b45309"], ai: ["brain", "#6d28d9"] };
  const [name, color] = map[type] || map.ok;
  const el = document.createElement("div");
  el.className = "toast"; el.style.setProperty("--tc", color);
  el.innerHTML = `${ic(name)}<span>${msg}</span>`;
  $("#toasts").appendChild(el);
  setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 400); }, ms);
}
function openModal(title, bodyHTML, wide = false) {
  $("#modal-root").innerHTML = `<div class="modal-veil" onclick="if(event.target===this)CP.closeModal()">
    <div class="modal" ${wide ? 'style="width:min(720px,100%)"' : ""}>
      <div class="modal-head"><h3>${title}</h3><button class="mclose" onclick="CP.closeModal()">✕</button></div>
      <div class="modal-body">${bodyHTML}</div>
    </div></div>`;
}
function closeModal() { $("#modal-root").innerHTML = ""; }

/* ========================= 5. CITY MAP ENGINE =============================== */
/* Procedural top-down map of central Chennai; lat/lng → canvas projection. */
const CITY = { latMin: 12.948, latMax: 13.082, lngMin: 80.19, lngMax: 80.305 };
function cityProject(lat, lng, W, H, pad = 34) {
  return [pad + ((lng - CITY.lngMin) / (CITY.lngMax - CITY.lngMin)) * (W - pad * 2),
          pad + (1 - (lat - CITY.latMin) / (CITY.latMax - CITY.latMin)) * (H - pad * 2)];
}
function mulberry32(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

function drawCityBase(ctx, W, H) {
  const rnd = mulberry32(42);
  ctx.fillStyle = "#e9edf1"; ctx.fillRect(0, 0, W, H);
  for (let i = 0; i < 260; i++) { // city blocks
    const x = rnd() * W, y = rnd() * H, w = 12 + rnd() * 46, h = 10 + rnd() * 34;
    ctx.fillStyle = `rgba(16,24,40,${0.02 + rnd() * 0.035})`;
    ctx.fillRect(x, y, w, h);
  }
  for (let i = 0; i < 16; i++) { // parks
    const x = rnd() * W, y = rnd() * H, r = 12 + rnd() * 26;
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, "rgba(5,150,105,.14)"); g.addColorStop(1, "transparent");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
  }
  // sea (Bay of Bengal, east)
  const [seaX] = cityProject(13.0, 80.284, W, H);
  const sea = ctx.createLinearGradient(seaX, 0, W, 0);
  sea.addColorStop(0, "rgba(191,219,254,.42)"); sea.addColorStop(1, "rgba(147,197,253,.78)");
  ctx.fillStyle = sea; ctx.beginPath(); ctx.moveTo(seaX, 0);
  for (let y = 0; y <= H; y += 24) ctx.lineTo(seaX + Math.sin(y * 0.02) * 14 + y * 0.02, y);
  ctx.lineTo(W, H); ctx.lineTo(W, 0); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = "rgba(59,130,246,.55)"; ctx.lineWidth = 1.4; ctx.beginPath();
  for (let y = 0; y <= H; y += 24) { const x = seaX + Math.sin(y * 0.02) * 14 + y * 0.02; y === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
  ctx.stroke();
  ctx.fillStyle = "rgba(37,99,235,.55)"; ctx.font = "600 10px Inter,sans-serif";
  ctx.fillText("BAY OF BENGAL", W - 106, H - 16);
  // Adyar river
  ctx.strokeStyle = "rgba(147,197,253,.75)"; ctx.lineWidth = 9; ctx.lineCap = "round"; ctx.beginPath();
  const [rx1, ry1] = cityProject(12.995, 80.19, W, H); ctx.moveTo(rx1, ry1);
  const [cx, cy] = cityProject(12.985, 80.235, W, H);
  const [rx2, ry2] = cityProject(13.006, 80.262, W, H);
  ctx.quadraticCurveTo(cx, cy, rx2, ry2); ctx.stroke();
  ctx.lineWidth = 1.2; ctx.strokeStyle = "rgba(59,130,246,.5)"; ctx.stroke();
  // arterials (white roads)
  const roads = [
    [[13.082, 80.199], [13.006, 80.221], [12.966, 80.232]],
    [[13.060, 80.19], [13.058, 80.305]],
    [[13.029, 80.19], [13.031, 80.262]],
    [[12.975, 80.19], [12.979, 80.26]],
    [[13.002, 80.245], [12.992, 80.305]],
    [[13.082, 80.262], [12.966, 80.286]],
    [[13.045, 80.221], [13.010, 80.257]],
  ];
  for (const r of roads) {
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 3.4; ctx.beginPath();
    r.forEach(([la, ln], i) => { const [x, y] = cityProject(la, ln, W, H); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
    ctx.strokeStyle = "rgba(16,24,40,.08)"; ctx.lineWidth = 1; ctx.stroke();
  }
  ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.lineWidth = 1.4;
  for (let i = 0; i < 26; i++) {
    const la = 12.948 + i * 0.0052;
    let [xa, ya] = cityProject(la, 80.19, W, H); const [xb, yb] = cityProject(la, 80.305, W, H);
    ctx.beginPath(); ctx.moveTo(xa, ya); ctx.lineTo(xb, yb); ctx.stroke();
    const ln = 80.19 + i * 0.0048;
    [xa, ya] = cityProject(12.948, ln, W, H); const [xb2, yb2] = cityProject(13.082, ln, W, H);
    ctx.beginPath(); ctx.moveTo(xa, ya); ctx.lineTo(xb2, yb2); ctx.stroke();
  }
  ctx.fillStyle = "rgba(100,116,139,.75)"; ctx.font = "600 10.5px Inter,sans-serif";
  for (const [t, la, ln] of [["T. Nagar", 13.045, 80.228], ["Guindy", 13.012, 80.212], ["Adyar", 13.007, 80.252],
                             ["Velachery", 12.968, 80.214], ["Marina", 13.045, 80.276], ["Nungambakkam", 13.065, 80.24]]) {
    const [x, y] = cityProject(la, ln, W, H); ctx.fillText(t, x, y);
  }
}

function CityMap(canvas, opts = {}) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const ctx = canvas.getContext("2d");
  let W = 0, H = 0, base = null, markers = [], selectedId = opts.selectedId ?? null;
  let hoverId = null, raf = null, dead = false;
  const t0 = performance.now();
  const tip = opts.tooltipEl || null;

  function resize() {
    const r = canvas.getBoundingClientRect();
    W = Math.max(80, r.width); H = Math.max(80, r.height);
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    base = document.createElement("canvas"); base.width = W * dpr; base.height = H * dpr;
    const bctx = base.getContext("2d"); bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawCityBase(bctx, W, H);
  }
  function setReports(reps) {
    markers = (reps || []).filter((r) => r.lat != null && r.lng != null)
      .map((r) => ({ rep: r, critical: r.priority_label === "Critical" && ["open", "reopened"].includes(r.status) }));
  }
  function draw() {
    if (dead) return;
    const t = (performance.now() - t0) / 1000;
    ctx.clearRect(0, 0, W, H);
    if (base) ctx.drawImage(base, 0, 0, W, H);
    for (const m of markers) {
      const [x, y] = cityProject(m.rep.lat, m.rep.lng, W, H);
      m.x = x; m.y = y;
      const col = statusOf(m.rep.status).color;
      if (m.critical) {
        const ph = (t * 1.4 + m.rep.id) % 2;
        for (let k = 0; k < 2; k++) {
          const rr = 9 + ((ph + k) % 2) * 12;
          ctx.strokeStyle = col + Math.max(0, 0.5 - rr * 0.022).toFixed(3); ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(x, y, rr, 0, 7); ctx.stroke();
        }
      }
      const glow = ctx.createRadialGradient(x, y, 0, x, y, 12);
      glow.addColorStop(0, col + "33"); glow.addColorStop(1, "transparent");
      ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(x, y, 12, 0, 7); ctx.fill();
      ctx.fillStyle = "#ffffff"; ctx.beginPath(); ctx.arc(x, y, 5.6, 0, 7); ctx.fill();
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, (m.rep.id === hoverId || m.rep.id === selectedId) ? 4.6 : 3.4, 0, 7); ctx.fill();
      if (m.rep.id === selectedId) { ctx.strokeStyle = "#101828"; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.arc(x, y, 9.5, 0, 7); ctx.stroke(); }
    }
    raf = requestAnimationFrame(draw);
  }
  function pick(e) {
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = null, bd = 18;
    for (const m of markers) { const d = Math.hypot(m.x - mx, m.y - my); if (d < bd) { bd = d; best = m; } }
    return best;
  }
  canvas.addEventListener("mousemove", (e) => {
    const m = pick(e);
    hoverId = m ? m.rep.id : null;
    canvas.style.cursor = m ? "pointer" : opts.pickMode ? "crosshair" : "default";
    if (tip) {
      if (m) {
        const r = m.rep;
        tip.innerHTML = `<div class="flex" style="gap:6px;margin-bottom:5px"><span class="badge ${statusOf(r.status).badge}" style="font-size:9.5px"><span class="dot"></span>${statusOf(r.status).label}</span><span class="mono faint">${r.ticket}</span></div>
        <div style="font-weight:700;line-height:1.35">${esc(r.title)}</div>
        <div class="muted tiny" style="margin-top:4px">${catOf(r.category).short} · Priority ${r.priority_label} ${Math.round(r.priority_score)} · ${fmtNum(r.upvotes)} confirmations</div>`;
        const host = tip.parentElement.getBoundingClientRect();
        tip.style.left = Math.min(host.width - 260, Math.max(8, m.x + 16)) + "px";
        tip.style.top = Math.max(8, m.y - 30) + "px";
        tip.classList.add("show");
      } else tip.classList.remove("show");
    }
  });
  canvas.addEventListener("mouseleave", () => { hoverId = null; if (tip) tip.classList.remove("show"); });
  canvas.addEventListener("click", (e) => {
    if (opts.pickMode && opts.onPick) {
      const r = canvas.getBoundingClientRect();
      const lat = CITY.latMax - ((e.clientY - r.top - 34) / (H - 68)) * (CITY.latMax - CITY.latMin);
      const lng = CITY.lngMin + ((e.clientX - r.left - 34) / (W - 68)) * (CITY.lngMax - CITY.lngMin);
      opts.onPick(+lat.toFixed(5), +lng.toFixed(5));
      return;
    }
    const m = pick(e);
    if (m && opts.onSelect) opts.onSelect(m.rep);
  });
  resize(); setReports(opts.reports || []); draw();
  const ro = new ResizeObserver(resize); ro.observe(canvas);
  return { setReports, setSelected: (id) => (selectedId = id), destroy() { dead = true; ro.disconnect(); cancelAnimationFrame(raf); } };
}

/* ============================== 6. CHARTS =================================== */
function drawDonut(canvas, items, centerTop, centerBottom) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const ctx = canvas.getContext("2d");
  const W = canvas.clientWidth || 220, H = canvas.clientHeight || 220;
  canvas.width = W * dpr; canvas.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const total = items.reduce((s, i) => s + i.value, 0) || 1;
  const cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 10, r2 = R * 0.66;
  const start = performance.now();
  (function frame(now) {
    const p = Math.min(1, (now - start) / 900), e = 1 - Math.pow(1 - p, 3);
    ctx.clearRect(0, 0, W, H);
    let a = -Math.PI / 2;
    for (const it of items) {
      const sweep = (it.value / total) * Math.PI * 2 * e;
      ctx.beginPath(); ctx.strokeStyle = it.color; ctx.lineWidth = R - r2;
      ctx.arc(cx, cy, (R + r2) / 2, a + 0.02, a + Math.max(0.02, sweep - 0.03)); ctx.stroke();
      a += sweep;
    }
    ctx.fillStyle = "#101828"; ctx.font = "800 26px Inter,sans-serif"; ctx.textAlign = "center";
    ctx.fillText(centerTop, cx, cy + 2);
    ctx.fillStyle = "#5b6472"; ctx.font = "11px Inter,sans-serif";
    ctx.fillText(centerBottom, cx, cy + 20);
    if (p < 1) requestAnimationFrame(frame);
  })(performance.now());
}
function drawBars(canvas, series, color = "#059669") {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const ctx = canvas.getContext("2d");
  const W = canvas.clientWidth || 300, H = canvas.clientHeight || 120;
  canvas.width = W * dpr; canvas.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const max = Math.max(1, ...series.map((s) => s.count));
  const start = performance.now();
  (function frame(now) {
    const p = Math.min(1, (now - start) / 800), e = 1 - Math.pow(1 - p, 3);
    ctx.clearRect(0, 0, W, H);
    const bw = W / series.length;
    series.forEach((s, i) => {
      const h = (s.count / max) * (H - 34) * e;
      const x = i * bw + bw * 0.22, w = bw * 0.56;
      const grad = ctx.createLinearGradient(0, H - 24 - h, 0, H - 24);
      grad.addColorStop(0, color); grad.addColorStop(1, color + "33");
      ctx.fillStyle = grad; ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(x, H - 24 - h, w, h, 5) : ctx.rect(x, H - 24 - h, w, h); ctx.fill();
      ctx.fillStyle = "#98a1b0"; ctx.font = "10px Inter,sans-serif"; ctx.textAlign = "center";
      ctx.fillText(s.label, x + w / 2, H - 8);
      if (s.count) { ctx.fillStyle = "#101828"; ctx.font = "700 10.5px Inter,sans-serif"; ctx.fillText(s.count, x + w / 2, H - 30 - h); }
    });
    if (p < 1) requestAnimationFrame(frame);
  })(performance.now());
}

/* ============================= 7. UPVOTES =================================== */
const VOTE_KEY = "cp_voted";
const votedSet = () => { try { return new Set(JSON.parse(localStorage.getItem(VOTE_KEY) || "[]")); } catch { return new Set(); } };
const voted = (id) => votedSet().has(id);
async function upvote(id, btn) {
  const set = votedSet();
  if (set.has(id)) return toast("You already confirmed this issue", "info");
  try {
    const d = await api.postJSON(`/api/reports/${id}/upvote`, {});
    set.add(id); localStorage.setItem(VOTE_KEY, JSON.stringify([...set]));
    if (btn) { btn.classList.add("voted"); const s = btn.querySelector(".num"); if (s) s.textContent = fmtNum(d.upvotes); }
    toast(`Confirmed — priority is now ${d.priority.label} (${Math.round(d.priority.score)})`, "ok");
    document.dispatchEvent(new CustomEvent("cp:upvoted", { detail: { id, ...d } }));
  } catch (e) { toast(e.message, "err"); }
}

/* ========================= 8. SHARED REPORT CARD ============================ */
function repCardHTML(rep) {
  const dups = (rep.duplicates || []).length;
  return `
  <article class="card card-hover rep-card" onclick="location.hash='#/issue/${rep.id}'">
    ${rep.image_url ? `<img class="rep-thumb" loading="lazy" src="${rep.image_url}" alt="">` : `<div class="rep-thumb" style="display:grid;place-items:center;color:${catOf(rep.category).color}">${ic(catOf(rep.category).icon, "ic-lg")}</div>`}
    <div class="rep-body">
      <div class="row-wrap" style="margin-bottom:6px">${badgeHTML(rep)}${catChipHTML(rep, false)}</div>
      <h4 class="rep-title">${esc(rep.title)}</h4>
      <div class="rep-meta">
        <span title="${esc(rep.address)}">${ic("pin")}${esc((rep.address || "").split(",")[0] || "Chennai")}</span>
        <span>${ic("clock")}${timeAgo(rep.created_at)}</span>
        <span style="color:${priColor(rep.priority_label)};font-weight:700"><b style="display:inline-block;width:6px;height:6px;border-radius:50%;background:currentColor;margin-right:5px"></b>${rep.priority_label}</span>
        ${dups ? `<span title="${dups} duplicate report(s) merged into this ticket">${ic("merge")}${dups} merged</span>` : ""}
      </div>
    </div>
    <div class="rep-right">
      <button class="upv ${voted(rep.id) ? "voted" : ""}" data-upv="${rep.id}" title="Confirm this issue — raises its priority" onclick="event.stopPropagation();CP.upvote(${rep.id},this)">${ic("thumbs")}<span class="num">${fmtNum(rep.upvotes)}</span></button>
    </div>
  </article>`;
}

/* ====================== ROUTER + GLOBAL EXPORTS ============================= */
const App = { cleanups: [] };
const addCleanup = (fn) => App.cleanups.push(fn);
const cleanup = () => { App.cleanups.forEach((f) => { try { f(); } catch {} }); App.cleanups = []; };

window.CP = { $, $$, esc, ic, api, debounce, timeAgo, fmtNum,
  CAT, catOf, STATUS, statusOf, priColor, badgeHTML, ringHTML, catChipHTML,
  repCardHTML, toast, openModal, closeModal, CityMap, drawDonut, drawBars,
  voted, upvote, addCleanup, CITY, navigate: (...a) => navigate(...a) };

function setNav(key) { $$("#navlinks a").forEach((a) => a.classList.toggle("active", a.dataset.nav === key)); }

function navigate() {
  cleanup(); closeModal();
  const hash = location.hash.replace(/^#\/?/, "");
  const [route, arg] = hash.split("/");
  const view = $("#view");
  window.scrollTo({ top: 0 });
  if (route === "" || route === "home") { setNav("home"); Pages.home(view); }
  else if (route === "explore") { setNav("explore"); Pages.explore(view); }
  else if (route === "report") { setNav(""); Pages.report(view); }
  else if (route === "issue" && arg) { setNav(""); Pages.issue(view, +arg); }
  else if (route === "authority") { setNav("authority"); Admin.console(view); }
  else { setNav("home"); Pages.home(view); }
}
window.addEventListener("hashchange", navigate);
document.addEventListener("DOMContentLoaded", navigate);

/* ================================================================================
   9-12. CITIZEN PAGES
   ================================================================================ */
const Pages = {};
window.Pages = Pages;
(function () {
  "use strict";
  const { $, $$, esc, ic, api, timeAgo, fmtNum, catOf, statusOf, priColor,
    badgeHTML, ringHTML, catChipHTML, repCardHTML, toast, openModal, closeModal,
    CityMap, addCleanup, voted, upvote } = window.CP;

  /* ------------------------------- HOME ------------------------------------ */
  Pages.home = async function (view) {
    view.innerHTML = `
    <div class="wrap fade-in">
      <section class="hero">
        <div>
          <span class="eyebrow">${ic("spark")}Chennai · civic reporting that follows through</span>
          <h1 class="display" style="margin-top:18px">Report it.<br><span class="grad-text">Prove it.</span><br>Resolve it.</h1>
          <p class="hero-sub">Snap a photo of a civic issue — it's classified, prioritised and deduplicated automatically.
            When the crew claims it's fixed, <b>ProofWatch</b> checks the repair against your photo and reopens anything unfinished.</p>
          <div class="hero-cta">
            <a class="btn btn-primary" href="#/report">${ic("camera")}Report an Issue</a>
            <a class="btn btn-ghost" href="#/explore">${ic("map")}Explore live issues</a>
            <a class="btn btn-ghost" href="#/authority">${ic("shield")}Authority console</a>
          </div>
          <div class="row-wrap" style="margin-top:22px">
            <span class="chip">${ic("zap")}Works fully offline</span>
            <span class="chip">${ic("merge")}Duplicates merge automatically</span>
            <span class="chip">${ic("compare")}Repairs photo-verified</span>
          </div>
        </div>
        <div class="hero-panel">
          <div class="card" style="position:relative;overflow:hidden">
            <canvas id="hero-map" class="hero-map"></canvas>
            <div class="map-caption">Live issue map · generated from open reports</div>
            <div id="hero-float-1" style="position:absolute;top:16px;left:16px"></div>
            <div id="hero-float-2" style="position:absolute;top:16px;right:16px"></div>
          </div>
        </div>
      </section>

      <section class="stat-row" id="home-stats">
        ${"<div class='card stat-card skel' style='height:86px'></div>".repeat(4)}
      </section>

      <section style="margin-top:44px">
        <span class="eyebrow">${ic("layer")}The loop</span>
        <h2 class="sec-title" style="margin-top:12px">From photo to proof, in one loop</h2>
        <div class="how-grid">
          ${[
            ["camera", "#0369a1", "Snap & geo-tag", "Citizen photographs the issue — photo, GPS and description captured in under 20 seconds.", "Citizen"],
            ["brain", "#6d28d9", "AI triage", "Vision + language models classify the issue, score urgency and generate a priority with reasons.", "AI Engine"],
            ["merge", "#0369a1", "Duplicate merging", "Perceptual photo hashes + location + text similarity merge repeat complaints into one louder signal.", "AI Engine"],
            ["shield", "#059669", "ProofWatch", "Before/after comparison verifies the repair. Fakes and unfinished work auto-reopen the ticket.", "AI Engine"],
          ].map(([i, c, t, p, tag], n) => `
          <div class="card card-hover how-card" style="--hc:${c}">
            <span class="how-num">0${n + 1}</span>
            <div class="how-ic">${ic(i, "ic-lg")}</div>
            <h3>${t}</h3><p>${p}</p><span class="ai-tag">${ic("spark")}${tag}</span>
          </div>`).join("")}
        </div>
      </section>

      <section style="margin-top:46px">
        <span class="eyebrow">${ic("activity")}Under the hood</span>
        <h2 class="sec-title" style="margin-top:12px">How every report is processed</h2>
        <p class="sec-sub" style="margin-top:8px">Four deterministic stages — the reasoning is shown on every ticket.</p>
        <div class="pipe">
          ${[
            ["scan", "#6d28d9", "Vision classifier", "Reads the photo and description to label the issue type, with a confidence score."],
            ["gauge", "#c2410c", "Priority engine", "Severity, urgency language, citizen confirmations and age combine into one priority."],
            ["merge", "#0369a1", "Dedupe engine", "Photo hashes, distance and wording similarity find repeat complaints and merge them."],
            ["compare", "#059669", "ProofWatch", "The “after” photo is compared with the original — unfinished work reopens the ticket."],
          ].map(([i, c, t, p], k) => `
          <div class="card pipe-node" style="--pn:${c}"><span class="badge-node">STAGE ${k + 1}</span><h4>${ic(i)}${t}</h4><p>${p}</p></div>
          ${k < 3 ? `<div class="pipe-arrow">${ic("arrow-r", "ic-lg")}</div>` : ""}`).join("")}
        </div>
      </section>

      <section style="margin-top:46px">
        <div class="flex-between">
          <div><span class="eyebrow">${ic("activity")}Fresh from the feed</span>
          <h2 class="sec-title" style="margin-top:12px">Recent reports</h2></div>
          <a class="btn btn-ghost btn-sm" href="#/explore">View all ${ic("arrow-r")}</a>
        </div>
        <div id="home-feed" class="grid" style="margin-top:18px"></div>
      </section>
    </div>`;

    let map = null;
    api.get("/api/stats").then((s) => {
      const t = s.totals;
      $("#home-stats").innerHTML = [
        [fmtNum(t.reports + t.duplicates), "", "file-text", "Community issues filed", "#0369a1"],
        [fmtNum(t.resolved), "", "check-circle", "Verified resolutions", "#059669"],
        [`${t.upvotes}`, "", "users", "Citizen confirmations", "#6d28d9"],
        [s.proofwatch.attempts, ` ${s.proofwatch.verified}✓ ${s.proofwatch.reopened}✗`, "shield", "ProofWatch verdicts", "#c2410c"],
      ].map(([n, em, i, l, c]) => `
        <div class="card stat-card"><div class="stat-num" style="color:${c}">${n}<em>${em}</em></div>
        <div class="stat-label"><span style="color:${c}">${ic(i)}</span>${l}</div></div>`).join("");

      map = CityMap($("#hero-map"), { reports: s.hotspots, onSelect: (r) => (location.hash = `#/issue/${r.id}`) });
      addCleanup(() => map && map.destroy());

      const top = s.hotspots[0];
      if (top) $("#hero-float-1").innerHTML = `<div class="glass" style="padding:10px 13px;max-width:210px;animation:fadein .8s .3s backwards">
          <div class="flex small" style="font-weight:700;gap:7px"><span style="color:${priColor(top.priority)}">${ic("flame")}</span>Top priority right now</div>
          <div class="tiny muted" style="margin-top:4px;line-height:1.4">${esc(top.title.slice(0, 60))}…</div>
          <div class="tiny" style="color:${priColor(top.priority)};font-weight:800;margin-top:5px">${top.priority} · ${Math.round(top.priority_score)}</div>
        </div>`;
      if (s.proofwatch.attempts) $("#hero-float-2").innerHTML = `<div class="glass" style="padding:10px 13px;animation:fadein .8s .5s backwards">
          <div class="flex small" style="font-weight:700;color:#059669;gap:7px">${ic("compare")}ProofWatch</div>
          <div class="tiny muted" style="margin-top:4px">${s.proofwatch.verified} verified · ${s.proofwatch.reopened} auto-reopened</div>
        </div>`;
    }).catch((e) => toast(e.message, "err"));

    api.get("/api/reports?sort=recent").then((d) => {
      const el = $("#home-feed"); if (!el) return;
      const reps = d.reports.filter((r) => r.status !== "duplicate").slice(0, 3);
      el.innerHTML = reps.length ? reps.map((r) => repCardHTML(r)).join("") :
        `<div class="card empty">${ic("camera", "ic-xl")}<div>No reports yet — be the first.</div></div>`;
    }).catch(() => {});
  };

  /* ------------------------------ EXPLORE ---------------------------------- */
  Pages.explore = async function (view) {
    const state = { status: "", category: "", q: "", sort: "priority", showDup: false };
    let map = null;

    view.innerHTML = `
    <div class="wrap wrap-wide fade-in">
      <div class="flex-between" style="flex-wrap:wrap;gap:14px">
        <div>
          <h2 class="sec-title">Explore issues around you</h2>
          <p class="sec-sub" id="explore-count" style="margin-top:4px">Loading live reports…</p>
        </div>
        <div class="flex" style="gap:10px;flex-wrap:wrap">
          <div style="position:relative">
            <span style="position:absolute;left:11px;top:11px;color:var(--faint)">${ic("search")}</span>
            <input id="f-q" class="input" style="padding-left:38px;width:250px" placeholder="Search title, area, ticket…">
          </div>
          <select id="f-sort" class="select" style="width:170px">
            <option value="priority">Sort: AI priority</option>
            <option value="recent">Sort: newest</option>
            <option value="upvotes">Sort: most confirmed</option>
          </select>
        </div>
      </div>

      <div class="row-wrap" style="margin:16px 0 6px" id="f-status">
        ${["", "open", "in_progress", "reopened", "resolved"].map((s) =>
          `<button class="fchip ${s === "" ? "on" : ""}" data-s="${s}">${s === "" ? "All statuses" : statusOf(s).label}</button>`).join("")}
        <button class="fchip" id="f-dup" title="Duplicates are normally merged into their parent ticket">Duplicates</button>
      </div>
      <div class="row-wrap" id="f-cat">
        <button class="fchip on" data-c="">All categories</button>
        ${Object.entries(window.CP.CAT).map(([k, v]) => `<button class="fchip" data-c="${k}">${v.short}</button>`).join("")}
      </div>

      <div class="explore-grid" style="margin-top:18px">
        <div><div class="feed-list" id="feed"></div></div>
        <div class="map-stick">
          <div class="card" style="position:relative;overflow:hidden">
            <canvas id="explore-map" class="big-map"></canvas>
            <div class="map-tip" id="map-tip"></div>
            <div class="card-pad" style="border-top:1px solid var(--border);padding:12px 16px">
              <div class="map-legend">
                ${Object.entries(window.CP.STATUS).filter(([k]) => k !== "duplicate").map(([k, v]) => `<span><b style="background:${v.color}"></b>${v.label}</span>`).join("")}
                <span class="faint" style="margin-left:auto">Click a marker to open the ticket</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>`;

    map = CityMap($("#explore-map"), { tooltipEl: $("#map-tip"), onSelect: (r) => (location.hash = `#/issue/${r.id}`) });
    addCleanup(() => map && map.destroy());

    async function refresh() {
      const feed = $("#feed"); if (!feed) return;
      const params = new URLSearchParams({ sort: state.sort });
      if (state.status) params.set("status", state.status);
      if (state.category) params.set("category", state.category);
      if (state.q) params.set("q", state.q);
      let d;
      try { d = await api.get("/api/reports?" + params); } catch (e) { return toast(e.message, "err"); }
      if (!$("#feed")) return; // user navigated away meanwhile
      let reps = d.reports;
      if (!state.showDup) reps = reps.filter((r) => r.status !== "duplicate");
      $("#explore-count").textContent =
        `${reps.length} issue${reps.length === 1 ? "" : "s"} ${state.q ? `matching “${state.q}”` : "in view"} · merged duplicates are auto-hidden`;
      feed.innerHTML = reps.length ? reps.map((r) => repCardHTML(r)).join("") :
        `<div class="card empty">${ic("search", "ic-xl")}<div style="font-weight:700;color:var(--muted);margin-bottom:5px">Nothing here</div>Try widening filters, or <a href="#/report" style="color:var(--green-2)">report the issue yourself</a>.</div>`;
      map && map.setReports(reps.filter((r) => r.status !== "duplicate"));
    }

    $("#f-q").addEventListener("input", debounce((e) => { state.q = e.target.value.trim(); refresh(); }, 350));
    $("#f-sort").addEventListener("change", (e) => { state.sort = e.target.value; refresh(); });
    $("#f-status").addEventListener("click", (e) => {
      const b = e.target.closest(".fchip"); if (!b) return;
      if (b.id === "f-dup") { state.showDup = !state.showDup; b.classList.toggle("on", state.showDup); refresh(); return; }
      state.status = b.dataset.s;
      $$("#f-status .fchip").forEach((x) => x !== b && x.id !== "f-dup" && x.classList.remove("on"));
      b.classList.add("on"); refresh();
    });
    $("#f-cat").addEventListener("click", (e) => {
      const b = e.target.closest(".fchip"); if (!b) return;
      state.category = b.dataset.c;
      $$("#f-cat .fchip").forEach((x) => x.classList.toggle("on", x === b));
      refresh();
    });
    const onUpv = () => refresh();
    document.addEventListener("cp:upvoted", onUpv);
    addCleanup(() => document.removeEventListener("cp:upvoted", onUpv));
    refresh();
  };
  const STATUS0 = { open: STATUS.open, in_progress: STATUS.in_progress, reopened: STATUS.reopened, resolved: STATUS.resolved };

  /* -------------------------- REPORT WIZARD -------------------------------- */
  Pages.report = function (view) {
    const wiz = { step: 0, photo: null, photoURL: "", ai: null, title: "", desc: "", name: "",
                  lat: 13.0400, lng: 80.2360, address: "", submitting: false };
    const STEPS = ["Photo", "Describe & AI scan", "Location", "Confirm"];
    let locMap = null;

    view.innerHTML = `
    <div class="wrap fade-in" style="max-width:1040px">
      <span class="eyebrow">${ic("camera")}New report</span>
      <h2 class="sec-title" style="margin:12px 0 2px">Report a civic issue</h2>
      <p class="sec-sub" style="margin-bottom:8px">One photo is all it takes. The AI handles classification, prioritisation and duplicate checks.</p>
      <div class="wiz-steps">${STEPS.map((s, i) => `
        <div class="wiz-step ${i === 0 ? "on" : ""}"><span class="ws-dot">${i + 1}</span><span>${s}</span></div>`).join("")}</div>
      <div id="wiz-body"></div>
    </div>`;

    function setStep(n) {
      wiz.step = n;
      $$("#wiz-steps .wiz-step").forEach((el, i) => {
        el.classList.toggle("on", i === n);
        el.classList.toggle("done", i < n);
        el.querySelector(".ws-dot").innerHTML = i < n ? "✓" : String(i + 1);
      });
      renderStep();
    }
    const body = (html) => { $("#wiz-body").innerHTML = html; };

    /* step 0 — photo */
    function stepPhoto() {
      body(`
      <div class="card card-pad">
        <div class="dz" id="dz" tabindex="0"></div>
        <input type="file" id="dz-file" accept="image/*" hidden>
        <div class="flex-between" style="margin-top:18px">
          <div class="muted small" id="dz-name">${wiz.photo ? esc(wiz.photo.name) : "No photo selected yet"}</div>
          <div class="flex">
            <button class="btn btn-ghost" onclick="location.hash='#/'">Cancel</button>
            <button class="btn btn-primary" id="next0" ${wiz.photo ? "" : "disabled"}>Next ${ic("arrow-r")}</button>
          </div>
        </div>
      </div>`);
      const dz = $("#dz"), fi = $("#dz-file");
      const paint = () => {
        dz.innerHTML = wiz.photoURL ? `
          <img class="dz-preview" src="${wiz.photoURL}">
          <div class="dz-veil"><span>${esc(wiz.photo.name)}</span><span style="color:var(--green-2);font-weight:700">Replace photo</span></div>` : `
          <div class="dz-icon">${ic("camera", "ic-xl")}</div>
          <div style="font-weight:700">Drop the issue photo here</div>
          <div class="muted small">click to browse · paste from clipboard · JPG/PNG up to 20 MB</div>
          <div class="tiny faint">Shoot it straight, in good light — ProofWatch will compare the repair against this exact photo later.</div>`;
      };
      const setFile = (f) => {
        if (!f || !f.type.startsWith("image/")) return toast("Please choose an image file", "warn");
        wiz.photo = f; wiz.photoURL = URL.createObjectURL(f); wiz.ai = null;
        paint(); $("#dz-name").textContent = f.name; $("#next0").disabled = false;
      };
      paint();
      dz.onclick = () => fi.click();
      fi.onchange = () => setFile(fi.files[0]);
      ["dragover", "dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => {
        e.preventDefault(); dz.classList.toggle("drag", ev === "dragover");
        if (ev === "drop") setFile(e.dataTransfer.files[0]);
      }));
      const paste = (e) => { const f = [...(e.clipboardData?.files || [])][0]; if (f) setFile(f); };
      document.addEventListener("paste", paste); addCleanup(() => document.removeEventListener("paste", paste));
      $("#next0").onclick = () => setStep(1);
    }

    /* step 1 — describe + AI analysis */
    const URGENT_LOCAL = ["injured", "accident", "school", "hospital", "children", "elderly", "danger", "dengue", "mosquito", "urgent", "wire", "fire", "emergency"];
    function stepDescribe() {
      body(`
      <div class="grid grid-2" style="align-items:start">
        <div class="card card-pad">
          <div class="field">
            <label class="label">What happened? (title)</label>
            <input id="w-title" class="input" maxlength="120" placeholder="e.g. Deep pothole near the market gate" value="${esc(wiz.title)}">
          </div>
          <div class="field">
            <label class="label">Describe it briefly</label>
            <textarea id="w-desc" class="textarea" maxlength="600" placeholder="Landmarks, how long it's been this way, who is affected…">${esc(wiz.desc)}</textarea>
            <div class="hint" id="w-urgency">${ic("info")}The AI also reads urgency from your words — “near a school”, “children”, “accident” raise the priority.</div>
          </div>
          <div class="field">
            <label class="label">Your name (optional)</label>
            <input id="w-name" class="input" maxlength="60" placeholder="Anonymous Citizen" value="${esc(wiz.name)}">
          </div>
          <div class="flex-between">
            <button class="btn btn-ghost" id="back1">← Back</button>
            <div class="flex">
              ${wiz.ai ? `<button class="btn btn-ghost" id="rerun">${ic("brain")}Re-run AI</button>` : ""}
              <button class="btn btn-primary" id="analyze">${ic("brain")}${wiz.ai ? "Continue" : "Run AI analysis"}</button>
            </div>
          </div>
        </div>
        <div id="ai-side">${wiz.ai ? "" : aiPlaceholderHTML()}</div>
      </div>`);
      $("#back1").onclick = () => setStep(0);
      const syncFields = () => { wiz.title = $("#w-title").value; wiz.desc = $("#w-desc").value; wiz.name = $("#w-name").value; };
      $("#w-desc").addEventListener("input", (e) => {
        const hits = URGENT_LOCAL.filter((k) => e.target.value.toLowerCase().includes(k));
        $("#w-urgency").innerHTML = hits.length
          ? `<span style="color:var(--amber)">${ic("zap")}Urgency language detected: ${hits.slice(0, 4).join(", ")} — AI will boost this.</span>`
          : `${ic("info")}The AI also reads urgency from your words — “near a school”, “children”, “accident” raise the priority.`;
      });
      if (wiz.ai) renderAISide();
      $("#analyze").onclick = () => {
        syncFields();
        if (wiz.ai) return setStep(2);
        if (!wiz.title.trim()) { $("#w-title").focus(); return toast("Give the issue a short title first", "warn"); }
        runAnalysis();
      };
      const rerun = $("#rerun");
      if (rerun) rerun.onclick = () => { syncFields(); wiz.ai = null; $("#ai-side").innerHTML = aiPlaceholderHTML(); runAnalysis(); };
    }
    const aiPlaceholderHTML = () => `
      <div class="ai-panel" style="min-height:280px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center">
        <div class="dz-icon" style="background:rgba(167,139,250,.1);border-color:rgba(167,139,250,.35);color:#c4b5fd">${ic("brain", "ic-xl")}</div>
        <div style="font-weight:700;margin-top:14px">AI triage awaits</div>
        <div class="muted small" style="max-width:260px;margin-top:6px">Classification, confidence, priority scoring and duplicate search appear here after you run the analysis.</div>
      </div>`;

    async function runAnalysis() {
      const side = $("#ai-side");
      side.innerHTML = `<div class="ai-panel">
        <div class="flex" style="gap:8px"><span class="ai-badge">${ic("brain")}AI engine</span><span class="tiny faint mono">vision·nlp·dedupe</span></div>
        <div class="scan-steps" id="scan-steps">
          ${["Extracting visual features (edges, colour, texture)", "Classifying issue category", "Scoring priority from severity & language", "Sweeping for duplicate reports nearby"].map((s) =>
            `<div class="scan-step"><span class="ss-dot"></span><span>${s}</span></div>`).join("")}
        </div></div>`;
      const steps = $$("#scan-steps .scan-step");
      steps.forEach((s, i) => setTimeout(() => s.classList.add("active"), i * 750));
      const minWait = new Promise((r) => setTimeout(r, 4 * 750 + 300));
      const fd = new FormData();
      fd.append("photo", wiz.photo); fd.append("title", wiz.title); fd.append("description", wiz.desc);
      let d = null, err = null;
      try { d = await api.postForm("/api/analyze", fd); } catch (e) { err = e; }
      await minWait;
      steps.forEach((s) => { s.classList.remove("active"); s.classList.add("done"); s.querySelector(".ss-dot").innerHTML = "✓"; });
      if (err) { side.innerHTML = aiPlaceholderHTML(); return toast(err.message, "err"); }
      wiz.ai = { ...d.ai, staged: d.staged_photo };
      setTimeout(renderAISide, 350);
      const btn = $("#analyze");
      if (btn) { btn.innerHTML = `${ic("arrow-r")}Continue`; btn.onclick = () => setStep(2); }
    }

    function renderAISide() {
      const ai = wiz.ai; if (!ai) return;
      const c = ai.classification, p = ai.priority, dup = ai.duplicate;
      const meta = catOf(c.category);
      const arc = 2 * Math.PI * 30;
      $("#ai-side").innerHTML = `
      <div class="ai-panel">
        <div class="flex-between">
          <span class="ai-badge">${ic("brain")}AI triage complete</span>
          <span class="tiny faint mono">${c.agreement ? "vision+text agree" : "vision-led"}</span>
        </div>
        <div class="flex" style="margin-top:16px;gap:16px">
          <svg width="76" height="76" class="conf-arc" viewBox="0 0 76 76">
            <circle cx="38" cy="38" r="30" stroke="#eceef2" stroke-width="7" fill="none"/>
            <circle cx="38" cy="38" r="30" stroke="${meta.color}" stroke-width="7" fill="none" stroke-linecap="round"
              stroke-dasharray="${arc}" data-target="${arc * (1 - c.confidence / 100)}" style="stroke-dashoffset:${arc};transition:stroke-dashoffset 1s cubic-bezier(.22,1,.36,1)"/>
          </svg>
          <div>
            <div class="flex" style="gap:8px"><span style="color:${meta.color}">${ic(meta.icon, "ic-lg")}</span><b style="font-size:16.5px">${meta.label}</b></div>
            <div class="tiny muted" style="margin-top:3px">detected with <b style="color:${meta.color}">${c.confidence}%</b> confidence</div>
          </div>
        </div>
        ${(c.text_keywords || []).length ? `<div class="row-wrap" style="margin-top:12px">${c.text_keywords.slice(0, 5).map((k) => `<span class="kw-chip">${esc(k)}</span>`).join("")}</div>` : ""}
        ${(c.urgency_hits || []).length ? `<div class="row-wrap" style="margin-top:7px">${c.urgency_hits.slice(0, 4).map((k) => `<span class="kw-chip kw-urgent">${ic("zap")}${esc(k)}</span>`).join("")}</div>` : ""}
        <div class="hr"></div>
        <div class="flex-between tiny muted" style="margin-bottom:7px"><span>AI priority</span>
          <b style="color:${priColor(p.label)}">${p.label} · ${Math.round(p.score)}</b></div>
        <div class="meter"><i style="--mc:${priColor(p.label)};width:0%;transition:width 1s" data-w="${p.score}"></i></div>
        <div class="tiny muted" style="margin-top:9px;line-height:1.65">
          ${p.reasons.slice(0, 3).map((r) => `<div class="flex" style="gap:6px"><span style="color:var(--green-2)">+${r.delta}</span><span>${esc(r.text)}</span></div>`).join("")}
        </div>
        ${dup && dup.match ? `
        <div class="hr"></div>
        <div class="flex" style="gap:8px;color:#b91c1c;font-weight:700">${ic("merge")}Possible duplicate found</div>
        <div class="small muted" style="margin-top:5px;line-height:1.5">Matches <a href="#/issue/${dup.match.id}" class="mono" style="color:#0369a1">${dup.match.ticket}</a> (score ${dup.score})
        ${dup.distance_m != null ? ` · ${dup.distance_m}m away` : ""}. If submitted, it auto-merges and boosts the original.</div>` : `
        <div class="hr"></div>
        <div class="flex small" style="gap:8px;color:#059669">${ic("check-circle")}No duplicates found — this is a fresh signal.</div>`}
      </div>`;
      requestAnimationFrame(() => {
        const circ = $("#ai-side circle[data-target]");
        if (circ) circ.style.strokeDashoffset = circ.dataset.target;
        const m = $("#ai-side .meter i");
        if (m) m.style.width = m.dataset.w + "%";
      });
    }

    /* step 2 — location */
    function stepLocation() {
      body(`
      <div class="grid" style="grid-template-columns:1.25fr .9fr;gap:18px;align-items:start">
        <div class="card" style="position:relative;overflow:hidden">
          <canvas id="loc-map" style="width:100%;height:430px;display:block;border-radius:var(--r-lg)"></canvas>
          <div class="map-caption">Click anywhere on the map to pin the issue</div>
        </div>
        <div class="card card-pad">
          <h3 style="font-size:16px;margin-bottom:12px">Where is the issue?</h3>
          <div class="field">
            <label class="label">Coordinates</label>
            <div class="mono small" id="loc-coords" style="padding:11px 13px;background:rgba(2,6,23,.5);border:1px solid var(--border);border-radius:11px;color:var(--green-2)">
              ${wiz.lat.toFixed(5)}, ${wiz.lng.toFixed(5)}</div>
          </div>
          <button class="btn btn-cyan btn-block" id="btn-geo">${ic("target")}Use my current location</button>
          <div class="field" style="margin-top:16px">
            <label class="label">Area / landmark (optional)</label>
            <input id="w-address" class="input" maxlength="90" placeholder="e.g. Anna Salai, near Guindy Metro" value="${esc(wiz.address)}">
            <div class="hint">Shown publicly on the ticket.</div>
          </div>
          <div class="flex-between" style="margin-top:8px">
            <button class="btn btn-ghost" id="back2">← Back</button>
            <button class="btn btn-primary" id="next2">Review ${ic("arrow-r")}</button>
          </div>
        </div>
      </div>`);
      const pinRep = () => ({ id: -1, lat: wiz.lat, lng: wiz.lng, status: "open",
        priority_label: wiz.ai ? wiz.ai.priority.label : "Medium",
        priority_score: wiz.ai ? wiz.ai.priority.score : 50,
        category: wiz.ai ? wiz.ai.classification.category : "other",
        title: wiz.title || "New issue", ticket: "YOU", upvotes: 0 });
      const paintPin = () => {
        $("#loc-coords").textContent = `${wiz.lat.toFixed(5)}, ${wiz.lng.toFixed(5)}`;
        if (locMap) locMap.setReports([pinRep(), ...(locMap.ctxReps || [])]);
      };
      api.get("/api/reports?sort=recent").then((d) => {
        if (!locMap) return;
        locMap.ctxReps = d.reports.filter((r) => r.status !== "duplicate").slice(0, 24);
        locMap.setReports([pinRep(), ...locMap.ctxReps]);
      }).catch(() => {});
      locMap = CityMap($("#loc-map"), { pickMode: true, onPick: (lat, lng) => { wiz.lat = lat; wiz.lng = lng; paintPin(); } });
      addCleanup(() => locMap && (locMap.destroy(), (locMap = null)));
      $("#btn-geo").onclick = () => {
        if (!navigator.geolocation) return toast("Geolocation unavailable in this browser", "warn");
        toast("Locating…", "info", 1800);
        navigator.geolocation.getCurrentPosition(
          (pos) => { wiz.lat = +pos.coords.latitude.toFixed(5); wiz.lng = +pos.coords.longitude.toFixed(5); paintPin(); toast("Location locked", "ok"); },
          () => toast("Couldn't get GPS — pin it on the map instead", "warn"),
          { timeout: 8000 });
      };
      $("#back2").onclick = () => { wiz.address = $("#w-address").value; setStep(1); };
      $("#next2").onclick = () => { wiz.address = $("#w-address").value; setStep(3); };
    }

    /* step 3 — review & submit */
    function stepReview() {
      const ai = wiz.ai, c = ai ? ai.classification : null, p = ai ? ai.priority : null;
      const meta = catOf(c ? c.category : "other");
      body(`
      <div class="card card-pad">
        <div class="flex" style="gap:18px;align-items:flex-start;flex-wrap:wrap">
          <img src="${wiz.photoURL}" style="width:210px;height:150px;object-fit:cover;border-radius:12px;border:1px solid var(--border)">
          <div class="grow" style="min-width:240px">
            <div class="row-wrap" style="margin-bottom:8px">
              <span class="chip chip-cat" style="--c:${meta.color}">${ic(meta.icon)}${meta.label}</span>
              ${c ? `<span class="chip">${ic("brain")}${c.confidence}% AI</span>` : ""}
              ${p ? `<span class="chip" style="color:${priColor(p.label)};border-color:${priColor(p.label)}55">${ic("flame")}${p.label} ${Math.round(p.score)}</span>` : ""}
            </div>
            <h3 style="font-size:17px;line-height:1.4">${esc(wiz.title)}</h3>
            ${wiz.desc ? `<p class="muted small" style="margin:6px 0 0">${esc(wiz.desc)}</p>` : ""}
            <div class="rep-meta" style="margin-top:10px">
              <span>${ic("pin")}${esc(wiz.address || "Pinned on map")}</span>
              <span class="mono">${wiz.lat.toFixed(5)}, ${wiz.lng.toFixed(5)}</span>
              <span>${ic("users")}${esc(wiz.name || "Anonymous Citizen")}</span>
            </div>
          </div>
        </div>
        <div class="hr"></div>
        <div class="muted tiny" style="line-height:1.7">
          ${ic("shield")} On submit, the report is geo-indexed, checked once more against every open ticket for duplicates,
          and queued for the ward authority. When a crew uploads the repair photo, <b>ProofWatch</b> verifies it against your photo.
        </div>
        <div class="flex-between" style="margin-top:16px">
          <button class="btn btn-ghost" id="back3">← Back</button>
          <button class="btn btn-primary" id="submit" style="min-width:230px">${ic("zap")}Submit report</button>
        </div>
      </div>`);
      $("#back3").onclick = () => setStep(2);
      $("#submit").onclick = submit;
    }

    async function submit() {
      if (wiz.submitting) return;
      wiz.submitting = true;
      const btn = $("#submit");
      btn.disabled = true; btn.innerHTML = `${ic("scan")}Running full AI triage…`;
      const fd = new FormData();
      fd.append("photo", wiz.photo); fd.append("title", wiz.title); fd.append("description", wiz.desc);
      fd.append("reporter", wiz.name); fd.append("address", wiz.address);
      fd.append("lat", wiz.lat); fd.append("lng", wiz.lng);
      try {
        const d = await api.postForm("/api/reports", fd);
        d.outcome === "duplicate_linked" ? renderMerged(d) : renderSuccess(d);
      } catch (e) {
        toast(e.message, "err"); btn.disabled = false; btn.innerHTML = `${ic("zap")}Submit report`; wiz.submitting = false;
      }
    }

    function renderMerged(d) {
      const dup = d.ai.duplicate, rep = d.report;
      body(`
      <div class="card card-pad" style="text-align:center;padding:38px">
        <div class="success-burst" style="background:rgba(56,189,248,.12);border-color:rgba(56,189,248,.5);color:#0369a1;margin:0 auto;box-shadow:0 0 40px -6px rgba(56,189,248,.5)">${ic("merge", "ic-xl")}</div>
        <h2 class="sec-title" style="margin-top:18px">Already reported — merged automatically</h2>
        <p class="muted" style="max-width:560px;margin:10px auto 0">Our dedupe engine matched your photo and location to an existing ticket
        (score <b>${dup.score}</b>${dup.distance_m != null ? `, only ${dup.distance_m} m away` : ""}). Instead of creating noise,
        your report was merged into <a href="#/issue/${rep.parent ? rep.parent.id : dup.match.id}" class="mono" style="color:#0369a1">${dup.match.ticket}</a>
        — and your voice <b>raised its priority</b>.</p>
        <div class="row-wrap" style="justify-content:center;margin:18px 0">
          ${dup.signals.map((s) => `<span class="kw-chip">${esc(s)}</span>`).join("")}
        </div>
        <div class="flex" style="justify-content:center">
          <a class="btn btn-primary" href="#/issue/${rep.parent ? rep.parent.id : dup.match.id}">${ic("eye")}Track the original ticket</a>
          <a class="btn btn-ghost" href="#/explore">Back to explore</a>
        </div>
      </div>`);
      toast("Merged — no duplicate ticket created", "ai");
    }

    function renderSuccess(d) {
      const rep = d.report, ai = d.ai;
      body(`
      <div class="card card-pad" style="text-align:center;padding:38px">
        <div class="success-burst" style="margin:0 auto">${ic("check-circle", "ic-xl")}</div>
        <h2 class="sec-title" style="margin-top:18px">Report filed &amp; triaged</h2>
        <div class="ticket-big" style="margin:14px 0 4px">${rep.ticket}</div>
        <p class="muted" style="max-width:520px;margin:6px auto 0">The AI assigned it <b style="color:${priColor(ai.priority.label)}">${ai.priority.label} priority (${Math.round(ai.priority.score)})</b>
        as <b>${catOf(ai.classification.category).label}</b> (${ai.classification.confidence}% confidence). It's live on the public map now.</p>
        <div class="grid grid-3" style="margin:22px 0;text-align:left">
          <div class="glass" style="padding:14px"><div class="tiny faint">AI CATEGORY</div><div class="small" style="font-weight:700;margin-top:4px">${catOf(ai.classification.category).label}</div></div>
          <div class="glass" style="padding:14px"><div class="tiny faint">PRIORITY</div><div class="small" style="font-weight:700;margin-top:4px;color:${priColor(ai.priority.label)}">${ai.priority.label} · ${Math.round(ai.priority.score)}</div></div>
          <div class="glass" style="padding:14px"><div class="tiny faint">DUPLICATES</div><div class="small" style="font-weight:700;margin-top:4px">${ai.duplicate.match ? "Possible match flagged" : "None — fresh signal"}</div></div>
        </div>
        <div class="flex" style="justify-content:center;flex-wrap:wrap">
          <button class="btn btn-ghost" id="copy-ticket">${ic("copy")}Copy ticket ID</button>
          <a class="btn btn-primary" href="#/issue/${rep.id}">${ic("eye")}View ticket</a>
          <a class="btn btn-ghost" href="#/report">Report another</a>
        </div>
      </div>`);
      const cpb = $("#copy-ticket");
      if (cpb) cpb.onclick = () => { navigator.clipboard?.writeText(rep.ticket); toast("Ticket ID copied", "ok"); };
      toast(`Report ${rep.ticket} created`, "ok");
    }

    const renderStep = () => [stepPhoto, stepDescribe, stepLocation, stepReview][wiz.step]();
    renderStep();
  };

  /* --------------------------- ISSUE DETAIL --------------------------------- */
  function fusedBarsHTML(rep) {
    const cls = (rep.ai || {}).classification || {};
    const order = Object.entries(cls.fused_scores || {}).sort((a, b) => b[1] - a[1]).slice(0, 4);
    const max = Math.max(0.001, ...order.map(([, v]) => v));
    return order.map(([k, v]) => `
      <div class="meter-row"><span style="width:86px;white-space:nowrap">${catOf(k).short}</span>
      <div class="meter"><i style="--mc:${k === rep.category ? catOf(k).color : "#475569"};width:${Math.round((v / max) * 100) * (v ? 1 : 0)}%"></i></div>
      <span class="num" style="width:34px;text-align:right">${v.toFixed(2)}</span></div>`).join("");
  }

  function proofwatchHTML(rep) {
    const pw = rep.pw || {};
    if (!pw.verdict) return "";
    const M = pw.metrics || {};
    const cfg = { verified: ["check-circle", "#059669", "pw-verified", "badge-verified", "REPAIR VERIFIED"],
                  reopened: ["x-circle", "#dc2626", "pw-reopened", "badge-rejected", "REJECTED — AUTO-REOPENED"],
                  flagged: ["flag", "#c2410c", "pw-flagged", "badge-flagged", "FLAGGED FOR FIELD CHECK"] }[pw.verdict];
    const [icon, color, cls, badge, label] = cfg;
    const sig = M.issue_signature_before != null ? `
      <div class="sig-arrow">
        <div><div class="tiny faint" style="margin-bottom:4px">ISSUE VISIBLE BEFORE</div>
          <div class="meter"><i style="--mc:#dc2626;width:${Math.round(M.issue_signature_before * 100)}%"></i></div>
          <div class="tiny num" style="color:#b91c1c;margin-top:3px">${M.issue_signature_before.toFixed(2)}</div></div>
        <div style="color:${color};font-weight:800">→ −${Math.round((M.issue_signature_drop || 0) * 100)}%</div>
        <div><div class="tiny faint" style="margin-bottom:4px">AFTER</div>
          <div class="meter"><i style="--mc:#059669;width:${Math.round((M.issue_signature_after || 0) * 100)}%"></i></div>
          <div class="tiny num" style="color:#059669;margin-top:3px">${(M.issue_signature_after || 0).toFixed(2)}</div></div>
      </div>` : "";
    const metric = (k, v) => `<div class="pw-metric"><div class="v">${v}</div><div class="k">${k}</div></div>`;
    return `
    <section class="card pw-verdict ${cls} fade-in">
      <div class="flex-between" style="flex-wrap:wrap;gap:10px">
        <div class="flex" style="gap:11px">
          <span style="color:${color}">${ic(icon, "ic-xl")}</span>
          <div>
            <div class="flex" style="gap:9px"><b style="font-size:16px;">ProofWatch</b><span class="badge ${badge}">${label}</span></div>
            <div class="tiny faint">compared citizen &amp; contractor photos ${timeAgo(pw.checked_at)}</div>
          </div>
        </div>
        <span class="ai-badge">${ic("scan")}machine verification</span>
      </div>
      <p class="small" style="margin:13px 0 0;line-height:1.7">${esc(pw.reason)}</p>
      ${sig}
      <details style="margin-top:13px">
        <summary style="cursor:pointer;font-size:12.5px;font-weight:650;color:var(--muted);list-style:none;display:flex;justify-content:space-between;align-items:center">Verification details &amp; heatmap<span class="chev" style="color:var(--faint)">▾</span></summary>
        <div class="pw-metrics" style="margin-top:11px">
          ${metric("scene match", (M.ssim_block ?? 0).toFixed(2))}
          ${metric("colour match", (M.histogram_correlation ?? 0).toFixed(2))}
          ${metric("image changed", Math.round((M.change_ratio ?? 0) * 100) + "%")}
          ${metric("photo hash Δ", M.photo_hash_distance ?? 0)}
          ${metric("issue still visible", M.after_issue_confidence ? M.after_issue_confidence + "%" : "no")}
          ${metric("after-photo reads", catOf(M.after_top_category).short)}
        </div>
        ${(pw.notes || []).map((n) => `<div class="tiny muted" style="margin-top:8px">${ic("info")} ${esc(n)}</div>`).join("")}
        ${pw.diff_image ? `
          <div style="margin-top:14px">
            <div class="tiny faint" style="margin-bottom:6px;text-transform:uppercase;letter-spacing:.1em">Change heatmap</div>
            <img class="diff-img" src="${pw.diff_image}" alt="ProofWatch change heatmap">
          </div>` : ""}
      </details>
    </section>`;
  }

  function aiTriageHTML(rep) {
    const ai = rep.ai || {}, cls = ai.classification || {}, pri = ai.priority || {};
    const meta = catOf(rep.category);
    return `
    <details class="card fade-in" style="overflow:hidden">
      <summary>${ic("brain")}&nbsp; How this was triaged <span style="color:var(--faint);font-weight:500">· AI breakdown</span><span class="chev" style="color:var(--faint)">▾</span></summary>
      <div style="padding:2px 20px 18px">
      <div class="flex" style="gap:14px;margin-top:6px">
        ${ringHTML(rep.ai_confidence, "Low")}
        <div>
          <div class="flex" style="gap:8px"><span style="color:${meta.color}">${ic(meta.icon)}</span><b>${meta.label}</b></div>
          <div class="tiny muted" style="margin-top:2px">classified at ${rep.ai_confidence}% confidence</div>
          ${(cls.text_keywords || []).length ? `<div class="row-wrap" style="margin-top:7px">${cls.text_keywords.slice(0, 4).map((k) => `<span class="kw-chip">${esc(k)}</span>`).join("")}</div>` : ""}
        </div>
      </div>
      <div class="hr"></div>
      <div class="tiny faint" style="text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px">Category affinities (vision ⊕ language)</div>
      ${fusedBarsHTML(rep)}
      <div class="hr"></div>
      <div class="flex-between"><span class="tiny faint" style="text-transform:uppercase;letter-spacing:.1em">Priority reasoning</span>
        <b style="color:${priColor(rep.priority_label)}">${rep.priority_label} · ${Math.round(rep.priority_score)}</b></div>
      <div class="meter" style="margin:8px 0"><i style="--mc:${priColor(rep.priority_label)};width:${Math.round(rep.priority_score)}%"></i></div>
      <div class="tiny muted" style="line-height:1.8">
        ${(pri.reasons || []).map((r) => `<div class="flex" style="gap:7px"><span style="color:var(--green-2);width:40px;text-align:right" class="num">+${r.delta}</span><span>${esc(r.text)}</span></div>`).join("")}
      </div>
      ${rep.status === "duplicate" && rep.parent ? `
        <div class="hr"></div>
        <div class="flex" style="gap:8px;color:#0369a1;font-weight:700">${ic("merge")}Merged into <a href="#/issue/${rep.parent.id}" class="mono">${rep.parent.ticket}</a></div>
        <div class="tiny muted" style="margin-top:6px;line-height:1.6">${((ai.duplicate || {}).signals || []).map((s) => `• ${esc(s)}`).join("<br>")}</div>` : ""}
      ${(ai.duplicate && ai.duplicate.possible && ai.duplicate.match) ? `
        <div class="hr"></div>
        <div class="tiny" style="color:#c2410c">${ic("alert")} Possible overlap with <a class="mono" href="#/issue/${ai.duplicate.match.id}" style="color:#0369a1">${ai.duplicate.match.ticket}</a> — kept as an independent ticket.</div>` : ""}
      </div>
    </details>`;
  }

  function timelineHTML(activity) {
    const evCfg = {
      created: ["#0369a1", "citizen"], ai_classified: ["#6d28d9", "AI"], upvoted: ["#0369a1", "community"],
      status_changed: ["#b45309", "authority"], duplicate_merged: ["#64748b", "AI"],
      proofwatch_verified: ["#059669", "ProofWatch"], proofwatch_reopened: ["#dc2626", "ProofWatch"],
      proofwatch_flagged: ["#c2410c", "ProofWatch"], reopened: ["#dc2626", "citizen"],
    };
    return `<div class="tl">${(activity || []).map((a) => {
      const [c, actor] = evCfg[a.event] || ["#475569", a.actor];
      return `<div class="tl-item" style="--tlc:${c}">
        <div class="flex-between"><span class="tl-actor" style="color:${c}">${actor}</span><span class="tl-time">${timeAgo(a.created_at)}</span></div>
        <div class="tl-text">${esc(a.detail)}</div>
      </div>`; }).join("") || `<div class="muted small">No events yet.</div>`}
    </div>`;
  }

  function compareSliderHTML(rep) {
    if (!rep.after_image_url) return `<img class="detail-img" src="${rep.image_url}" alt="Issue photo">`;
    return `
    <div class="cmp-wrap" id="cmp">
      <img src="${rep.image_url}" alt="Before">
      <img class="cmp-after" src="${rep.after_image_url}" alt="After">
      <span class="cmp-tag before">BEFORE · CITIZEN</span>
      <span class="cmp-tag after">AFTER · CREW</span>
      <div class="cmp-bar"></div>
      <div class="cmp-knob">${ic("compare")}</div>
    </div>
    <div class="tiny faint" style="margin-top:8px;text-align:center">drag the handle to compare before / after</div>`;
  }

  Pages.issue = async function (view, id) {
    view.innerHTML = `<div class="wrap"><div class="card skel" style="height:320px"></div>
      <div class="grid grid-3" style="margin-top:16px">${"<div class='card skel' style='height:160px'></div>".repeat(3)}</div></div>`;
    let d;
    try { d = await api.get(`/api/reports/${id}`); } catch (e) {
      view.innerHTML = `<div class="wrap"><div class="card empty">${ic("search", "ic-xl")}Ticket not found — <a href="#/explore" style="color:var(--green-2)">browse the map</a>.</div></div>`; return;
    }
    const rep = d.report;
    const meta = catOf(rep.category);
    const track = ["open", "in_progress", "resolved"];
    const trIdx = rep.status === "reopened" ? 0 : track.indexOf(rep.status === "duplicate" ? "open" : rep.status);
    const dupKids = rep.duplicates || [];

    view.innerHTML = `
    <div class="wrap fade-in" style="max-width:1180px">
      <div class="tiny faint" style="margin-bottom:10px"><a href="#/explore" style="color:inherit">Explore</a> &nbsp;/&nbsp; <span class="mono">${rep.ticket}</span></div>
      <div class="flex-between" style="flex-wrap:wrap;gap:12px">
        <div>
          <div class="row-wrap" style="margin-bottom:9px">${badgeHTML(rep)}${catChipHTML(rep)}</div>
          <h2 style="font-size:clamp(20px,3vw,28px);max-width:760px;line-height:1.25">${esc(rep.title)}</h2>
          <div class="rep-meta" style="margin-top:9px">
            <span class="mono">${rep.ticket}</span>
            <span>${ic("users")}${esc(rep.reporter)}</span>
            <span>${ic("clock")}reported ${timeAgo(rep.created_at)}</span>
            <span>${ic("pin")}${esc(rep.address || "Chennai")} <span class="mono faint">${(rep.lat ?? 0).toFixed(4)}, ${(rep.lng ?? 0).toFixed(4)}</span></span>
            ${rep.status === "resolved" && rep.resolved_at ? `<span style="color:#059669">${ic("check-circle")}verified ${timeAgo(rep.resolved_at)}</span>` : ""}
          </div>
        </div>
        <div class="flex" style="gap:12px;align-items:center">
          ${ringHTML(rep.priority_score, rep.priority_label, true)}
          <button class="upv ${voted(rep.id) ? "voted" : ""}" id="upv-btn" style="font-size:14px;padding:11px 16px" onclick="CP.upvote(${rep.id},this)">${ic("thumbs")}<span class="num">${fmtNum(rep.upvotes)}</span>&nbsp;confirm</button>
        </div>
      </div>

      <div class="detail-grid" style="margin-top:20px">
        <div class="flex-col" style="gap:18px">
          <section class="card card-pad">
            ${compareSliderHTML(rep)}
            ${rep.description ? `<p class="small" style="margin:14px 4px 0;line-height:1.7">${esc(rep.description)}</p>` : ""}
          </section>
          <section class="card card-pad">
            <h3 style="font-size:15.5px;margin-bottom:14px">${ic("activity")}&nbsp; Status timeline</h3>
            ${timelineHTML(rep.activity)}
          </section>
          ${proofwatchHTML(rep)}
          ${aiTriageHTML(rep)}
        </div>

        <div class="flex-col" style="gap:18px">
          <section class="card card-pad">
            <h3 style="font-size:14px;margin-bottom:12px">${ic("gauge")}&nbsp; Progress</h3>
            <div class="flex">
              ${track.map((s, i) => `
                <div style="flex:1;text-align:center">
                  <div style="width:24px;height:24px;margin:0 auto;border-radius:50%;display:grid;place-items:center;
                    background:${i <= trIdx ? "rgba(34,197,94,.18)" : "rgba(148,163,184,.1)"};
                    border:1.5px solid ${i <= trIdx ? "#059669" : "var(--border-2)"};color:${i <= trIdx ? "#059669" : "var(--faint)"};font-size:11px">${i <= trIdx ? "✓" : i + 1}</div>
                  <div class="tiny" style="margin-top:5px;color:${i <= trIdx ? "var(--text)" : "var(--faint)"}">${statusOf(s).label}</div>
                </div>`).join("")}
            </div>
            ${rep.status === "reopened" ? `<div class="tiny" style="color:#b91c1c;margin-top:12px;line-height:1.6">${ic("alert")} ProofWatch rejected the completion photo — this ticket was automatically reopened.</div>` : ""}
            ${rep.flagged ? `<div class="tiny" style="color:#c2410c;margin-top:12px;line-height:1.6">${ic("flag")} Completion photo looked wrong for this location — a field re-inspection is required.</div>` : ""}
            ${rep.status === "resolved" ? `<button class="btn btn-danger btn-sm btn-block" style="margin-top:14px" id="btn-reopen">${ic("alert")}Still broken? Re-open this ticket</button>` : ""}
          </section>

          ${dupKids.length ? `
          <section class="card card-pad">
            <h3 style="font-size:14px;margin-bottom:4px">${ic("merge")}&nbsp; Merged duplicates <span class="cnt">${dupKids.length}</span></h3>
            <p class="tiny faint" style="margin:0 0 10px">These reports were auto-identified as the same issue. Each merge boosts priority.</p>
            ${dupKids.map((k) => `<a href="#/issue/${k.id}" style="text-decoration:none;color:inherit">
              <div class="glass" style="padding:10px 12px;margin-bottom:8px;display:flex;gap:10px;align-items:center">
                <span class="mono tiny">${k.ticket}</span><span class="tiny grow" style="line-height:1.35">${esc(k.title)}</span>
                <span class="tiny faint">${timeAgo(k.created_at)}</span></div></a>`).join("")}
          </section>` : ""}

          <section class="card card-pad">
            <h3 style="font-size:14px;margin-bottom:12px">${ic("shield")}&nbsp; Authority actions</h3>
            ${rep.assigned_to ? `<div class="tiny muted" style="margin-bottom:10px">${ic("wrench")} Crew: <b>${esc(rep.assigned_to)}</b></div>` : ""}
            <div id="authority-actions"></div>
          </section>

          <section class="card card-pad">
            <h3 style="font-size:14px;margin-bottom:10px">${ic("map")}&nbsp; Location</h3>
            <div style="position:relative;overflow:hidden;border-radius:12px">
              <canvas id="mini-map" style="width:100%;height:210px;display:block"></canvas>
            </div>
            ${rep.address ? `<div class="tiny muted" style="margin-top:8px">${ic("pin")} ${esc(rep.address)}</div>` : ""}
          </section>
        </div>
      </div>
    </div>`;

    // before/after compare slider
    const cmp = $("#cmp");
    if (cmp) {
      let drag = false;
      const set = (clientX) => {
        const r = cmp.getBoundingClientRect();
        const x = Math.min(Math.max(clientX - r.left, 12), r.width - 12);
        cmp.style.setProperty("--cx", `${(x / r.width) * 100}%`);
      };
      ["pointerdown", "pointermove", "pointerup"].forEach((ev) =>
        cmp.addEventListener(ev, (e) => {
          if (ev === "pointerdown") { drag = true; try { cmp.setPointerCapture(e.pointerId); } catch {} }
          if (ev === "pointerup") drag = false;
          if (ev === "pointerdown" || (ev === "pointermove" && drag)) set(e.clientX);
        }));
    }

    // mini map
    let mini = null;
    const mm = $("#mini-map");
    if (mm) { mini = CityMap(mm, { reports: [rep] }); addCleanup(() => mini && mini.destroy()); }

    // authority actions
    const aa = $("#authority-actions");
    function renderAuthorityActions() {
      const canStart = ["open", "reopened"].includes(rep.status);
      const canResolve = ["in_progress", "reopened"].includes(rep.status);
      aa.innerHTML = `
        <div class="field" style="margin-bottom:12px">
          <label class="label">Assigned crew</label>
          <input id="crew-name" class="input" placeholder="e.g. Ward 174 — Roads Team B" value="${esc(rep.assigned_to || "")}">
        </div>
        <div class="flex-col" style="gap:9px">
          ${canStart ? `<button class="btn btn-cyan btn-block" id="btn-start">${ic("wrench")}Start work (mark in-progress)</button>` : ""}
          ${canResolve ? `<button class="btn btn-primary btn-block" id="btn-resolve">${ic("compare")}Complete work — upload after photo</button>` : ""}
          ${rep.status === "resolved" ? `<div class="tiny" style="color:#059669;line-height:1.6">${ic("check-circle")} This repair is ProofWatch-verified. No further action needed.</div>` : ""}
          ${rep.status === "duplicate" ? `<div class="tiny muted">Managed under parent ticket <a class="mono" style="color:#0369a1" href="#/issue/${rep.parent ? rep.parent.id : ""}">${rep.parent ? rep.parent.ticket : ""}</a>.</div>` : ""}
        </div>`;
      const start = $("#btn-start");
      if (start) start.onclick = async () => {
        try {
          await api.postJSON(`/api/reports/${rep.id}/status`, { status: "in_progress", assigned_to: $("#crew-name").value });
          toast("Marked in-progress and assigned", "ok"); Pages.issue(view, id);
        } catch (e) { toast(e.message, "err"); }
      };
      const resolveBtn = $("#btn-resolve");
      if (resolveBtn) resolveBtn.onclick = openResolve;
    }

    function openResolve() {
      openModal(`${ic("compare")} Complete work — ProofWatch check`, `
        <p class="small muted" style="margin-top:0">Upload the crew's <b>after repair</b> photo. ProofWatch compares it with the citizen's
        original photo: same scene? is the issue signature gone? Fake or unfinished closures are auto-reopened.</p>
        <div class="dz" id="pw-dz" style="min-height:170px">
          <div class="dz-icon" style="width:44px;height:44px;border-radius:11px">${ic("camera", "ic-lg")}</div>
          <div style="font-weight:700;font-size:13.5px">Drop after-repair photo</div>
          <div class="tiny faint">JPG/PNG — same spot, similar angle for best results</div>
        </div>
        <input type="file" id="pw-file" accept="image/*" hidden>
        <div class="field" style="margin-top:12px;margin-bottom:0">
          <label class="label">Completion notes (optional)</label>
          <input id="pw-notes" class="input" placeholder="e.g. Re-laid 6 m² of asphalt, compacted">
        </div>
        <button class="btn btn-primary btn-block" id="pw-submit" style="margin-top:14px" disabled>${ic("scan")}Verify with ProofWatch</button>`);
      let file = null;
      const dz = $("#pw-dz"), fi = $("#pw-file"), sub = $("#pw-submit");
      const setFile = (f) => {
        if (!f || !f.type.startsWith("image/")) return;
        file = f;
        dz.innerHTML = `<img class="dz-preview" src="${URL.createObjectURL(f)}"><div class="dz-veil"><span>${esc(f.name)}</span><span style="color:var(--green-2)">Replace</span></div>`;
        sub.disabled = false;
      };
      dz.onclick = () => fi.click();
      fi.onchange = () => setFile(fi.files[0]);
      ["dragover", "dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => {
        e.preventDefault(); dz.classList.toggle("drag", ev === "dragover");
        if (ev === "drop") setFile(e.dataTransfer.files[0]);
      }));
      sub.onclick = () => runProofWatch(file);
    }

    async function runProofWatch(file) {
      const mb = $(".modal-body"); if (!mb) return;
      mb.innerHTML = `
        <div class="scan-steps" id="pw-steps" style="margin-top:4px">
          ${["Aligning before/after photos & measuring scene match", "Extracting issue signature from both photos", "Rendering change heatmap & deciding verdict"].map((s) =>
            `<div class="scan-step"><span class="ss-dot"></span><span>${s}</span></div>`).join("")}
        </div>
        <div class="tiny faint" style="margin-top:16px;text-align:center">ProofWatch is comparing pixel structure, colour and the issue's visual signature…</div>`;
      const steps = $$("#pw-steps .scan-step");
      steps.forEach((s, i) => setTimeout(() => s.classList.add("active"), i * 900));
      const minWait = new Promise((r) => setTimeout(r, 3 * 900 + 400));
      const fd = new FormData();
      fd.append("after_photo", file); fd.append("notes", $("#pw-notes")?.value || "");
      let d = null, err = null;
      try { d = await api.postForm(`/api/reports/${rep.id}/resolve`, fd); } catch (e) { err = e; }
      await minWait;
      if (err) { closeModal(); return toast(err.message, "err"); }
      steps.forEach((s) => { s.classList.remove("active"); s.classList.add("done"); s.querySelector(".ss-dot").innerHTML = "✓"; });
      const pw = d.proofwatch, M = pw.metrics || {};
      const cfg = { verified: ["#059669", "check-circle", "REPAIR VERIFIED"], reopened: ["#dc2626", "x-circle", "REJECTED — TICKET REOPENED"], flagged: ["#c2410c", "flag", "FLAGGED FOR FIELD CHECK"] }[pw.verdict];
      setTimeout(() => {
        const mb2 = $(".modal-body"); if (!mb2) return;
        mb2.innerHTML = `
          <div style="text-align:center;padding:8px 0 4px">
            <div class="success-burst" style="margin:0 auto 14px;background:${cfg[0]}22;border-color:${cfg[0]}88;color:${cfg[0]};box-shadow:0 0 40px -6px ${cfg[0]}88">${ic(cfg[1], "ic-xl")}</div>
            <b style="font-size:17px;color:${cfg[0]}">${cfg[2]}</b>
            <p class="small muted" style="margin:12px auto 0;max-width:440px;line-height:1.7;text-align:left">${esc(pw.reason)}</p>
            <div class="grid grid-3" style="margin:18px 0">
              <div class="glass" style="padding:12px"><div class="tiny faint">SCENE MATCH</div><b>${(M.ssim_block ?? 0).toFixed(2)}</b></div>
              <div class="glass" style="padding:12px"><div class="tiny faint">SIGNATURE DROP</div><b>−${Math.round((M.issue_signature_drop || 0) * 100)}%</b></div>
              <div class="glass" style="padding:12px"><div class="tiny faint">FRAME CHANGED</div><b>${Math.round((M.change_ratio || 0) * 100)}%</b></div>
            </div>
            <button class="btn btn-primary btn-block" onclick="CP.closeModal();CP.navigate()">${pw.verdict === "verified" ? "Done — ticket closed" : "Open ticket"}</button>
          </div>`;
      }, 400);
      if (d.report) Object.assign(rep, d.report);
    }

    renderAuthorityActions();

    const reopen = $("#btn-reopen");
    if (reopen) reopen.onclick = async () => {
      try { await api.postJSON(`/api/reports/${rep.id}/reopen`, {}); toast("Ticket reopened — authority notified", "warn"); Pages.issue(view, id); }
      catch (e) { toast(e.message, "err"); }
    };
    const onUpv = () => { if ($("#view") && location.hash === `#/issue/${id}`) Pages.issue(view, id); };
    document.addEventListener("cp:upvoted", onUpv, { once: true });
    addCleanup(() => document.removeEventListener("cp:upvoted", onUpv));
  };
})();

/* ================================================================================
   13. AUTHORITY CONSOLE
   ================================================================================ */
const Admin = {};
window.Admin = Admin;
(function () {
  "use strict";
  const { $, $$, esc, ic, api, timeAgo, fmtNum, catOf, priColor,
    badgeHTML, ringHTML, toast, CityMap, drawDonut, drawBars, addCleanup } = window.CP;

  const TABS = [["triage", "Triage queue", "list"], ["proofwatch", "ProofWatch", "compare"],
                ["map", "City map", "map"], ["analytics", "Analytics", "chart"], ["activity", "Activity", "activity"]];

  Admin.console = async function (view) {
    let tab = "triage";
    let data = null;

    view.innerHTML = `
    <div class="wrap wrap-wide fade-in">
      <div class="flex-between" style="flex-wrap:wrap;gap:12px">
        <div>
          <span class="eyebrow">${ic("shield")}Greater Chennai Corporation · Control room</span>
          <h2 class="sec-title" style="margin-top:10px">Authority Console</h2>
        </div>
        <div class="flex">
          <a class="btn btn-ghost btn-sm" href="/api/reports.csv">${ic("download")}Export CSV</a>
          <button class="btn btn-ghost btn-sm" id="btn-refresh">${ic("activity")}Refresh</button>
        </div>
      </div>

      <div class="grid" id="kpi-row" style="grid-template-columns:repeat(6,1fr);margin:20px 0 8px">
        ${"<div class='card kpi skel' style='height:76px'></div>".repeat(6)}
      </div>

      <div class="tabs" id="tabs" style="margin-top:16px">
        ${TABS.map(([k, l, i], n) => `<button class="tab ${n === 0 ? "on" : ""}" data-t="${k}">${ic(i)}&nbsp;${l}<span class="cnt" data-cnt="${k}" style="display:none"></span></button>`).join("")}
      </div>
      <div id="tab-body"></div>
    </div>`;

    $("#tabs").addEventListener("click", (e) => {
      const b = e.target.closest(".tab"); if (!b) return;
      tab = b.dataset.t;
      $$("#tabs .tab").forEach((x) => x.classList.toggle("on", x === b));
      renderTab();
    });
    $("#btn-refresh").onclick = load;

    async function load() {
      try { data = await api.get("/api/stats"); } catch (e) { return toast(e.message, "err"); }
      if (!$("#kpi-row")) return; // navigated away
      renderKPIs();
      renderTab();
    }

    function renderKPIs() {
      const t = data.totals, pw = data.proofwatch;
      const cards = [
        [t.open + t.reopened, "Needs action", "flame", "#c2410c"],
        [t.in_progress, "Crews on ground", "wrench", "#b45309"],
        [t.flagged, "Field checks pending", "flag", "#c2410c"],
        [t.resolved, "Verified resolved", "check-circle", "#059669"],
        [`${pw.accuracy ?? "—"}%`, "ProofWatch verify rate", "compare", "#0369a1"],
        [t.avg_resolution_hours ? `${Math.round(t.avg_resolution_hours)}h` : "—", "Avg. resolution time", "clock", "#6d28d9"],
      ];
      $("#kpi-row").innerHTML = cards.map(([v, l, i, c]) => `
        <div class="card kpi card-hover" style="--kc:${c}">
          <div class="kpi-ic">${ic(i, "ic-lg")}</div>
          <div><div class="kpi-num">${v}</div><div class="kpi-label">${l}</div></div>
        </div>`).join("") +
        `<style>@media(max-width:1100px){#kpi-row{grid-template-columns:1fr 1fr 1fr!important}}@media(max-width:640px){#kpi-row{grid-template-columns:1fr 1fr!important}}</style>`;
      $$("#tabs .cnt").forEach((el) => {
        const k = el.dataset.cnt;
        const m = { triage: t.open + t.reopened + t.in_progress, proofwatch: pw.attempts - pw.verified }[k];
        el.style.display = m ? "" : "none";
        if (m) el.textContent = m;
      });
    }

    function renderTab() {
      ({ triage, proofwatch, map: mapTab, analytics, activity })[tab]();
    }

    /* ----------------------------- triage -------------------------------- */
    let triFilter = "active";
    function triage() {
      $("#tab-body").innerHTML = `
        <div class="row-wrap" id="tri-f" style="margin-bottom:14px">
          ${[["active", "Active queue"], ["resolved", "Resolved"], ["duplicate", "Merged duplicates"]].map(([k, l]) =>
            `<button class="fchip ${triFilter === k ? "on" : ""}" data-k="${k}">${l}</button>`).join("")}
        </div>
        <div id="tri-table"><div class="card card-pad muted small">Loading…</div></div>`;
      $("#tri-f").onclick = (e) => { const b = e.target.closest(".fchip"); if (!b) return;
        triFilter = b.dataset.k;
        $$("#tri-f .fchip").forEach((x) => x.classList.toggle("on", x === b));
        drawTable(); };
      drawTable();
    }
    async function drawTable() {
      const wrapEl = $("#tri-table"); if (!wrapEl) return;
      const status = triFilter === "active" ? "" : triFilter;
      const d = await api.get(`/api/reports?sort=priority${status ? "&status=" + status : ""}`);
      if (!$("#tri-table")) return;
      let reps = d.reports;
      if (triFilter === "active") reps = reps.filter((r) => ["open", "in_progress", "reopened"].includes(r.status));
      if (!reps.length) { wrapEl.innerHTML = `<div class="card empty">${ic("check-circle", "ic-xl")}Queue is clear. Excellent.</div>`; return; }
      wrapEl.innerHTML = `
      <div class="card" style="overflow-x:auto">
        <table class="tbl">
          <thead><tr>
            <th>Priority</th><th>Ticket</th><th>Issue</th><th>Category</th><th>Status</th>
            <th>Confirms</th><th>Age</th><th>Crew</th><th></th>
          </tr></thead>
          <tbody>
            ${reps.map((r) => `
            <tr onclick="location.hash='#/issue/${r.id}'">
              <td>${ringHTML(r.priority_score, r.priority_label)}</td>
              <td class="mono tiny">${r.ticket}</td>
              <td class="t-title">${esc(r.title)}<small>${esc(r.address || "Chennai")}</small></td>
              <td><span class="chip chip-cat" style="--c:${catOf(r.category).color}">${ic(catOf(r.category).icon)}${catOf(r.category).short}</span></td>
              <td>${badgeHTML(r)}</td>
              <td class="num">${fmtNum(r.upvotes)}${(r.duplicates || []).length ? ` <span class="tiny" title="merged duplicates" style="color:#0369a1">+${r.duplicates.length}</span>` : ""}</td>
              <td class="tiny muted">${r.age_days}d</td>
              <td class="tiny muted">${esc(r.assigned_to || "—")}</td>
              <td><button class="btn btn-ghost btn-xs">Open →</button></td>
            </tr>`).join("")}
          </tbody>
        </table>
      </div>`;
    }

    /* ---------------------------- proofwatch ------------------------------ */
    function proofwatch() {
      const el = $("#tab-body");
      el.innerHTML = `<div class="card card-pad muted small">Reviewing ProofWatch log…</div>`;
      api.get("/api/reports?sort=recent").then((d) => {
        if (!$("#tab-body")) return;
        const withPw = d.reports.filter((r) => (r.pw || {}).verdict);
        const pw = data.proofwatch;
        el.innerHTML = `
        <div class="grid grid-4" style="margin-bottom:18px">
          ${[["Attempts", pw.attempts, "scan", "#64748b"], ["Verified repairs", pw.verified, "check-circle", "#059669"],
             ["Auto-reopened", pw.reopened, "x-circle", "#dc2626"], ["Flagged for field check", pw.flagged, "flag", "#c2410c"]].map(([l, v, i, c]) => `
          <div class="card kpi card-hover" style="--kc:${c}"><div class="kpi-ic">${ic(i, "ic-lg")}</div>
          <div><div class="kpi-num">${v}</div><div class="kpi-label">${l}</div></div></div>`).join("")}
        </div>
        ${withPw.length ? withPw.map((r) => {
          const M = (r.pw.metrics || {});
          const vcfg = { verified: ["#059669", "check-circle", "VERIFIED"], reopened: ["#dc2626", "x-circle", "REJECTED"], flagged: ["#c2410c", "flag", "FLAGGED"] }[r.pw.verdict];
          return `
          <div class="card card-pad card-hover" style="margin-bottom:12px;cursor:pointer" onclick="location.hash='#/issue/${r.id}'">
            <div class="flex-between" style="flex-wrap:wrap;gap:10px">
              <div class="flex" style="gap:13px;min-width:0">
                ${r.image_url && r.after_image_url ? `
                <div class="flex" style="gap:4px;flex:none">
                  <img src="${r.image_url}" style="width:62px;height:62px;object-fit:cover;border-radius:10px;border:1px solid var(--border)">
                  <span class="faint" style="align-self:center">→</span>
                  <img src="${r.after_image_url}" style="width:62px;height:62px;object-fit:cover;border-radius:10px;border:1px solid var(--border)">
                </div>` : ""}
                <div style="min-width:0">
                  <div class="flex" style="gap:8px;flex-wrap:wrap">
                    <span class="mono tiny">${r.ticket}</span>
                    <span class="badge ${r.pw.verdict === "verified" ? "badge-verified" : r.pw.verdict === "reopened" ? "badge-rejected" : "badge-flagged"}">${vcfg[2]}</span>
                  </div>
                  <div class="small" style="font-weight:650;margin-top:5px;max-width:520px;line-height:1.35">${esc(r.title)}</div>
                  <div class="tiny muted" style="margin-top:4px">${catOf(r.category).short} · checked ${timeAgo(r.pw.checked_at)} · SSIM ${(M.ssim_block ?? 0).toFixed(2)} · signature −${Math.round((M.issue_signature_drop || 0) * 100)}% · changed ${Math.round((M.change_ratio || 0) * 100)}%</div>
                </div>
              </div>
              <span style="color:${vcfg[0]}">${ic(vcfg[1], "ic-xl")}</span>
            </div>
          </div>`; }).join("") :
          `<div class="card empty">${ic("compare", "ic-xl")}No completion photos processed yet.</div>`}
        <div class="hint" style="margin-top:8px">${ic("info")} ProofWatch runs automatically the moment a crew uploads an “after” photo on any ticket.</div>`;
      });
    }

    /* ------------------------------- map ----------------------------------- */
    function mapTab() {
      const el = $("#tab-body");
      el.innerHTML = `
      <div class="card" style="position:relative;overflow:hidden">
        <canvas id="admin-map" style="width:100%;height:600px;display:block"></canvas>
        <div class="map-tip" id="admin-map-tip"></div>
      </div>
      <div style="margin-top:16px" id="hot-list"></div>`;
      const m = CityMap($("#admin-map"), {
        reports: data.hotspots, tooltipEl: $("#admin-map-tip"),
        onSelect: (r) => (location.hash = `#/issue/${r.id}`),
      });
      addCleanup(() => m.destroy());
      $("#hot-list").innerHTML = `<div class="card card-pad">
        <div class="flex-between" style="flex-wrap:wrap;gap:10px">
          <h3 style="font-size:15px">${ic("flame")}&nbsp; Highest-priority unverified locations</h3>
          <span class="tiny faint">sorted by AI priority score</span>
        </div>
        <div style="margin-top:12px;display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px">
          ${data.hotspots.filter((r) => r.status !== "resolved").slice(0, 6).map((r) => `
            <div class="glass card-hover" style="padding:12px;cursor:pointer" onclick="location.hash='#/issue/${r.id}'">
              <div class="flex-between"><span class="mono tiny">${r.ticket}</span>
                <b style="color:${priColor(r.priority)};font-size:12px">${Math.round(r.priority_score)}</b></div>
              <div class="small" style="font-weight:650;margin-top:6px;line-height:1.35">${esc(r.title)}</div>
              <div class="flex" style="margin-top:8px;gap:6px">${badgeHTML(r)}</div>
            </div>`).join("")}
        </div></div>`;
    }

    /* ----------------------------- analytics -------------------------------- */
    function analytics() {
      const el = $("#tab-body");
      const byCat = Object.entries(data.by_category).map(([k, v]) => ({ label: catOf(k).short, value: v, color: catOf(k).color }));
      const byPri = ["Critical", "High", "Medium", "Low"].filter((p) => data.by_priority[p])
        .map((p) => ({ label: p, value: data.by_priority[p], color: priColor(p) }));
      const maxPri = Math.max(1, ...byPri.map((p) => p.value));
      const pw = data.proofwatch;
      el.innerHTML = `
      <div class="grid grid-3">
        <div class="card card-pad">
          <h3 style="font-size:14.5px;margin-bottom:14px">Reports by category</h3>
          <canvas id="an-donut" class="chart" style="height:210px"></canvas>
          <div class="row-wrap" style="margin-top:12px;justify-content:center">
            ${byCat.map((c) => `<span class="tiny muted"><i class="legend-dot" style="background:${c.color}"></i> ${c.label} (${c.value})</span>`).join("")}
          </div>
        </div>
        <div class="card card-pad">
          <h3 style="font-size:14.5px;margin-bottom:14px">Priority distribution</h3>
          ${byPri.map((p) => `
            <div class="bar-row"><span style="color:${p.color};font-weight:700">${p.label}</span>
            <div class="meter"><i style="--mc:${p.color};width:${Math.round((p.value / maxPri) * 100)}%"></i></div>
            <b class="num" style="text-align:right">${p.value}</b></div>`).join("")}
          <div class="hr"></div>
          <h3 style="font-size:14.5px;margin-bottom:12px">ProofWatch outcomes</h3>
          <div class="flex" style="gap:8px;flex-wrap:wrap">
            <span class="badge badge-verified">${pw.verified} verified</span>
            <span class="badge badge-rejected">${pw.reopened} reopened</span>
            <span class="badge badge-flagged">${pw.flagged} flagged</span>
          </div>
          <div class="tiny muted" style="margin-top:12px;line-height:1.7">
            Verification rate <b style="color:#059669">${pw.accuracy ?? "—"}%</b> ·
            ${pw.reopened ? `<b style="color:#b91c1c">${pw.reopened}</b> fraudulent/unfinished closure${pw.reopened > 1 ? "s" : ""} auto-reopened without human effort.` : "no invalid closures detected."}
          </div>
        </div>
        <div class="card card-pad">
          <h3 style="font-size:14.5px;margin-bottom:14px">Report volume · last 7 days</h3>
          <canvas id="an-bars" class="chart" style="height:180px"></canvas>
          <div class="hr"></div>
          <div class="flex-between small"><span class="muted">Community confirmations</span><b class="num">${fmtNum(data.totals.upvotes)}</b></div>
          <div class="flex-between small" style="margin-top:8px"><span class="muted">Duplicates auto-merged</span><b class="num">${fmtNum(data.totals.duplicates)}</b></div>
          <div class="flex-between small" style="margin-top:8px"><span class="muted">Avg. resolution time</span><b class="num">${data.totals.avg_resolution_hours ? Math.round(data.totals.avg_resolution_hours) + "h" : "—"}</b></div>
          <a class="btn btn-ghost btn-sm btn-block" style="margin-top:14px" href="/api/reports.csv">${ic("download")}Download full CSV</a>
        </div>
      </div>`;
      drawDonut($("#an-donut"), byCat, String(data.totals.reports), "live reports");
      drawBars($("#an-bars"), data.series7, "#059669");
    }

    /* ------------------------------ activity ------------------------------- */
    function activity() {
      const el = $("#tab-body");
      const evCfg = {
        created: ["camera", "#0369a1"], upvoted: ["thumbs", "#0369a1"], status_changed: ["wrench", "#b45309"],
        duplicate_merged: ["merge", "#64748b"], proofwatch_verified: ["check-circle", "#059669"],
        proofwatch_reopened: ["x-circle", "#dc2626"], proofwatch_flagged: ["flag", "#c2410c"], reopened: ["alert", "#dc2626"],
      };
      el.innerHTML = `
      <div class="card card-pad">
        <h3 style="font-size:15px;margin-bottom:16px">Live system events</h3>
        <div class="flex-col">
          ${data.activity.map((a) => {
            const [i, c] = evCfg[a.event] || ["activity", "#475569"];
            return `<div class="flex" style="gap:13px;padding:11px 0;border-bottom:1px solid var(--border)">
              <span style="width:32px;height:32px;border-radius:9px;display:grid;place-items:center;flex:none;
                background:${c}1c;color:${c};border:1px solid ${c}44">${ic(i)}</span>
              <div class="grow" style="min-width:0">
                <div class="small" style="line-height:1.45">${esc(a.detail)}</div>
                <div class="tiny faint" style="margin-top:2px">
                  ${a.ticket && a.report_id ? `<a class="mono" href="#/issue/${a.report_id}" style="color:#0369a1">${a.ticket}</a> · ` : ""}${a.actor} · ${timeAgo(a.created_at)}</div>
              </div>
            </div>`; }).join("") || `<div class="muted small">Quiet for now.</div>`}
        </div>
      </div>`;
    }

    load();
    const iv = setInterval(load, 45000);
    addCleanup(() => clearInterval(iv));
  };
})();

