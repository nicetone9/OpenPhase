/* Systems mode of the Records space: the system-level maps of the multi-protein systems (Fig. S1).
 * One map per group of system types, one dot per unique system (its components without concentrations). Positions are the stored Fig. S1 coordinates.
 * Needs: D (systems data from rs21), $ (getElementById), el (element helper), css (CSS variable reader). Builds nothing outside #sysPane. */
function initSystems(api) {
  "use strict";
  const { D, $, el, css } = api;
  const fmtn = (n) => n.toLocaleString("en-US");
  const buf = (s) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u.buffer; };
  const isDark = () => { const t = document.documentElement.getAttribute("data-theme"); return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches; };
  const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const mix = (a, b, t) => { const x = hex(a), y = hex(b); return "rgb(" + x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(",") + ")"; };
  const rgba = (h, a) => { const x = hex(h); return "rgba(" + x[0] + "," + x[1] + "," + x[2] + "," + a + ")"; };
  const sub = (tag, cls, text) => {       // text with C_sat written as C<sub>sat</sub>
    const e = document.createElement(tag); if (cls) e.className = cls;
    String(text).split("C_sat").forEach((p, i) => { if (i) { e.append("C"); const s = document.createElement("sub"); s.textContent = "sat"; e.appendChild(s); } if (p) e.append(p); });
    return e;
  };
  const TYPES = D.types, SRC = D.srcLab, UNI = /^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?$/;
  const G = D.groups.map((g) => Object.assign({}, g, { X: new Uint16Array(buf(g.x)), Y: new Uint16Array(buf(g.y)), REC: new Uint16Array(buf(g.rec)), POS: new Uint16Array(buf(g.p)), NEG: new Uint16Array(buf(g.q)),
    ANC: new Uint8Array(buf(g.anc)), FT: new Uint8Array(buf(g.ft)), sx: new Float32Array(g.n), sy: new Float32Array(g.n), match: new Uint8Array(g.n), maxRec: 1, ordA: null, ordR: null, types: [] }));
  G.forEach((g) => {
    for (let i = 0; i < g.n; i++) if (g.REC[i] > g.maxRec) g.maxRec = g.REC[i];
    const cnt = {}; g.anchors.forEach((a) => { cnt[a.rank] = a.n; });
    const idx = Array.from({ length: g.n }, (_, i) => i);
    g.ordA = idx.slice().sort((a, b) => (g.ANC[b] === 255 ? 1e9 : cnt[g.ANC[b]]) - (g.ANC[a] === 255 ? 1e9 : cnt[g.ANC[a]]));   // grey first, big anchors, small anchors on top
    g.ordR = idx.slice().sort((a, b) => g.REC[a] - g.REC[b]);
    g.types = [...new Set(Array.from(g.FT))].sort((a, b) => a - b);
  });
  const GK = G.map((g) => g.key);
  const S = { g: 0, sel: 0, t: -1, list: -1, by: "anchor", k: 1, tx: 0, ty: 0, q: "", hover: -1, pin: -1 };
  const cv = $("syCv"), c2 = cv.getContext("2d"), wrap = $("syWrap");
  let W = 0, H = 0, dpr = 1, raf = 0;
  const cur = () => G[S.g];

  // ---------------------------------------------------------------- colours: the system colour table for types; vivid anchors that avoid the system hues
  const ANCH = ["#D6338C", "#F2B705", "#14A098", "#9C5B2E", "#F28C28", "#4D6075"];
  const anchCol = (r) => (isDark() && r === 5 ? "#9FB3C8" : isDark() && r === 3 ? "#C98B5A" : ANCH[r]);
  const REC = ["#FBE3A6", "#F5A83F", "#D2601A", "#7A2E0B"];
  const posStops = () => (isDark() ? ["#8FA6BF", "#4A5663", "#F28C28"] : ["#4D6075", "#E3E7EB", "#F28C28"]);
  function ramp(stops, t) { t = Math.min(1, Math.max(0, t)); const p = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(p)); return mix(stops[i], stops[i + 1], p - i); }
  const posShare = (g, i) => { const l = g.POS[i] + g.NEG[i]; return l ? g.POS[i] / l : -1; };
  const recT = (g, i) => Math.log10(Math.max(1, g.REC[i])) / Math.max(Math.log10(g.maxRec), 0.3);
  function colorOf(g, i, pal) {
    if (S.by === "anchor") return g.ANC[i] === 255 ? pal.grey : anchCol(g.ANC[i]);
    if (S.by === "type") return TYPES[g.FT[i]].point;
    if (S.by === "pos") { const t = posShare(g, i); return t < 0 ? pal.none : ramp(pal.pos, t); }
    return ramp(REC, recT(g, i));
  }
  const typeOn = (g, i) => S.t < 0 || g.FT[i] === S.t;

  // ---------------------------------------------------------------- legend
  function buildLegend() {
    const box = $("syLegend"), g = cur(); box.textContent = ""; box.hidden = S.list >= 0; if (S.list >= 0) return;
    const item = (color, text, n) => { const it = el("span", "sy-lg"), d = el("i"); d.style.background = color; it.append(d, sub("span", null, text), el("b", null, fmtn(n))); box.appendChild(it); };
    const bar = (mid, ticks, nonum) => {
      const w = el("div", "sy-bar"), c = document.createElement("canvas"); c.width = 320; c.height = 10; const x = c.getContext("2d");
      for (let k = 0; k < 320; k++) { x.fillStyle = mid(k / 319); x.fillRect(k, 0, 1, 10); }
      const t = el("div", "sy-ticks"); ticks.forEach((s) => t.appendChild(el("span", null, s))); w.append(c, t); box.appendChild(w);
      if (nonum) item(css("--neutral"), "no labelled records", nonum);
    };
    if (S.by === "type") {
      box.appendChild(el("span", "sy-lgh", "System type"));
      const cnt = {}; for (let i = 0; i < g.n; i++) cnt[g.FT[i]] = (cnt[g.FT[i]] || 0) + 1;
      Object.keys(cnt).map(Number).sort((a, b) => a - b).forEach((k) => item(TYPES[k].point, TYPES[k].name, cnt[k]));
    } else if (S.by === "anchor") {
      box.appendChild(el("span", "sy-lgh", "Most frequent named component"));
      let named = 0; g.anchors.forEach((a) => { item(anchCol(a.rank), a.name, a.n); named += a.n; });
      item(css("--neutral2"), "other or unnamed", g.n - named);
    } else if (S.by === "pos") {
      let nl = 0; for (let i = 0; i < g.n; i++) if (g.POS[i] + g.NEG[i] === 0) nl++;
      bar((t) => ramp(posStops(), t), ["0%", "positive share of labelled records", "100%"], nl);
    } else {
      bar((t) => ramp(REC, t), ["1", "records per system (log scale)", fmtn(g.maxRec)], 0);
    }
  }

  // ---------------------------------------------------------------- geometry
  function resize() {
    const r = wrap.getBoundingClientRect(); dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = Math.max(1, Math.round(r.width * dpr)); H = Math.max(1, Math.round(r.height * dpr)); cv.width = W; cv.height = H; render();
  }
  function project(g) {
    const pad = 22 * dpr, w = W - 2 * pad, h = H - 2 * pad, k = S.k, tx = S.tx, ty = S.ty, f = 1 / 65535;
    for (let i = 0; i < g.n; i++) { g.sx[i] = (pad + g.X[i] * f * w) * k + tx; g.sy[i] = (pad + g.Y[i] * f * h) * k + ty; }
  }
  const baseR = () => Math.min(6.5, 3.4 + 0.6 * Math.log2(S.k)) * dpr;
  function zoomAt(px, py, f) {
    const k2 = Math.min(40, Math.max(1, S.k * f)), r = k2 / S.k; S.tx = px - (px - S.tx) * r; S.ty = py - (py - S.ty) * r; S.k = k2;
    if (S.k === 1) { S.tx = 0; S.ty = 0; }
    render();
  }
  function viewTo(i, k) {
    const g = cur(), pad = 22 * dpr, f = 1 / 65535; S.k = Math.max(S.k, k);
    S.tx = W / 2 - (pad + g.X[i] * f * (W - 2 * pad)) * S.k; S.ty = H / 2 - (pad + g.Y[i] * f * (H - 2 * pad)) * S.k;
  }

  // ---------------------------------------------------------------- drawing
  function render() { if (!raf) raf = requestAnimationFrame(draw); }
  function draw() {
    raf = 0; if (!W || S.list >= 0) return;
    const g = cur(), bg = css("--canvas"), ink = css("--ink"), base = baseR(), q = S.q.length >= 2;
    const pal = { grey: css("--neutral2"), none: css("--neutral"), pos: posStops() };
    const on = (i) => typeOn(g, i) && (!q || g.match[i]);
    c2.fillStyle = bg; c2.fillRect(0, 0, W, H); project(g);
    if (S.by === "anchor") {                                  // soft colour under each named anchor, as in Fig. S1
      const R = 24 * dpr * Math.sqrt(S.k);
      for (const i of g.ordA) {
        if (g.ANC[i] === 255 || !on(i)) continue;
        const x = g.sx[i], y = g.sy[i]; if (x < -R || y < -R || x > W + R || y > H + R) continue;
        const c = anchCol(g.ANC[i]), gr = c2.createRadialGradient(x, y, 0, x, y, R); gr.addColorStop(0, rgba(c, 0.20)); gr.addColorStop(1, rgba(c, 0));
        c2.fillStyle = gr; c2.beginPath(); c2.arc(x, y, R, 0, Math.PI * 2); c2.fill();
      }
    }
    c2.lineWidth = 0.7 * dpr; c2.strokeStyle = bg;
    for (const i of (S.by === "rec" ? g.ordR : g.ordA)) {
      const x = g.sx[i], y = g.sy[i]; if (x < -8 || y < -8 || x > W + 8 || y > H + 8) continue;
      c2.globalAlpha = on(i) ? 0.95 : 0.16; c2.fillStyle = colorOf(g, i, pal);
      c2.beginPath(); c2.arc(x, y, base, 0, Math.PI * 2); c2.fill(); c2.stroke();
    }
    c2.globalAlpha = 1;
    if (q || S.t >= 0) { c2.lineWidth = 1.4 * dpr; c2.strokeStyle = ink; for (let i = 0; i < g.n; i++) if (on(i) && (q || g.types.length > 1)) { c2.beginPath(); c2.arc(g.sx[i], g.sy[i], base * 1.7, 0, Math.PI * 2); c2.stroke(); } }
    for (const j of [S.pin, S.hover]) if (j >= 0) { c2.beginPath(); c2.arc(g.sx[j], g.sy[j], base * 2.4, 0, Math.PI * 2); c2.lineWidth = 1.8 * dpr; c2.strokeStyle = ink; c2.stroke(); }
    let nm = 0; for (let i = 0; i < g.n; i++) nm += on(i) ? 1 : 0;
    $("syCount").textContent = fmtn(g.n) + " systems · " + fmtn(g.records) + " records" + ((q || S.t >= 0) ? " · " + fmtn(nm) + (q ? " match" : " of " + TYPES[S.t].name) : "");
  }

  // ---------------------------------------------------------------- card and list
  const compLabel = (s) => (s === "unnamed sequences" ? "unnamed sequences (no name in the source)" : s === "-" ? "unnamed" : s);
  function pill(t, text) { const p = sub("span", "sy-pill", text || t.name); p.style.setProperty("--f", t.fill); p.style.setProperty("--p", t.point); return p; }
  function buildCard() {
    const card = $("syCard"), g = cur(); card.textContent = "";
    const T = TYPES[S.sel];
    if (S.list >= 0) {
      const lst = D.lists[T.name] || [], lab = lst.reduce((s, r) => s + r.pos + r.neg, 0), rec = lst.reduce((s, r) => s + r.rec, 0), pos = lst.reduce((s, r) => s + r.pos, 0);
      card.appendChild(el("h2", null, T.name)); card.appendChild(el("p", null, T.name === "RNA only" ? "Nucleic acid with no protein." : "DNA with no protein."));
      const t = el("div", "sy-tiles");
      for (const [b, l] of [[fmtn(lst.length), "systems"], [fmtn(rec), "records"], [lab ? Math.round(100 * pos / lab) + "%" : "–", "positive among " + fmtn(lab) + " labelled records"], [fmtn(rec - lab), "records with no binary label"]]) { const d = el("div", "sy-tile"); d.append(el("b", null, b), el("span", null, l)); t.appendChild(d); }
      card.appendChild(t);
      card.appendChild(el("p", null, "These systems have no protein sequence, so there is no embedding to place them on a map; they are listed instead."));
      return;
    }
    const i = S.pin >= 0 ? S.pin : S.hover;
    if (i < 0) {
      card.appendChild(el("h2", null, g.name)); card.appendChild(el("p", null, g.desc));
      const lab = g.pos + g.neg, t = el("div", "sy-tiles");
      for (const [b, l] of [[fmtn(g.n), "systems on the map"], [fmtn(g.records), "records"], [lab ? Math.round(100 * g.pos / lab) + "%" : "–", "positive among " + fmtn(lab) + " labelled records"], [fmtn(g.records - lab), "records with no binary label"]]) { const d = el("div", "sy-tile"); d.append(el("b", null, b), el("span", null, l)); t.appendChild(d); }
      card.appendChild(t);
      const noSeqRec = g.all_records - g.records - g.hid_rec, notes = [];
      if (g.no_seq) notes.push(g.no_seq + (g.no_seq === 1 ? " system has" : " systems have") + " no protein sequence and " + (g.no_seq === 1 ? "is" : "are") + " not drawn (" + fmtn(noSeqRec) + (noSeqRec === 1 ? " record" : " records") + ").");
      if (g.hid_n) notes.push(g.hid_n + (g.hid_n === 1 ? " system whose" : " systems whose") + " records all come from laboratory notebooks " + (g.hid_n === 1 ? "is" : "are") + " not shown (" + fmtn(g.hid_rec) + " records); those notebooks are not part of the published tables.");
      if (g.key === "cx" && (D.lists.Other || []).length) notes.push((D.lists.Other.length) + " system with no component at all is not on the map (" + fmtn(D.lists.Other.reduce((s, r) => s + r.rec, 0)) + " records).");
      if (notes.length) { card.appendChild(el("h2", null, "Not drawn")); notes.forEach((n) => card.appendChild(el("p", null, n))); }
      card.appendChild(el("h2", null, "How to read"));
      const ul = el("ul");
      for (const x of ["One dot is one unique system: the set of its components, without concentrations. Position is a t-SNE of the ESMC-600M embeddings of the components; each map has its own t-SNE, so axes and distances are not comparable across maps.",
        "Colour by the most frequent named component: the six most frequent named components of the group get a colour, every other system is grey. Or colour by system type (the colour table), by the share of labelled records that phase-separate, or by the number of records.",
        "A record is one experimental condition of the system. Systems from phase-diagram sources have many records; some sources give a quantitative readout and no binary label. The conditions themselves are not on these maps: the single-protein records, with conditions, are in the other mode.",
        "Hover a dot to read it, click to pin it, search a component to find every system that contains it."]) ul.appendChild(el("li", null, x));
      card.appendChild(ul);
      return;
    }
    const ft = TYPES[g.FT[i]];
    card.appendChild(el("h2", null, S.pin >= 0 ? "Pinned system" : "System under cursor")); card.appendChild(el("h3", null, compLabel(g.comp[i])));
    const tags = el("div"); tags.appendChild(pill(ft));
    if (g.ANC[i] !== 255) { const a = g.anchors.find((x) => x.rank === g.ANC[i]), p2 = el("span", "sy-pill"), d = el("i"); d.style.background = anchCol(g.ANC[i]); p2.style.setProperty("--f", "var(--chip)"); p2.style.setProperty("--p", "var(--chip)"); p2.append(d, document.createTextNode(a ? a.name : "")); tags.appendChild(p2); }
    card.appendChild(tags);
    const dl = el("dl", "sy-kv"), kv = (k, v) => { if (v != null && v !== "") { dl.appendChild(el("dt", null, k)); const d = el("dd"); if (v instanceof Node) d.appendChild(v); else d.textContent = v; dl.appendChild(d); } };
    kv("System", g.id[i]);
    const cw = document.createElement("span");
    g.comp[i].split(" + ").forEach((p, k) => { if (k) cw.append(" + "); if (UNI.test(p)) { const a = el("a", null, p); a.href = "https://www.uniprot.org/uniprotkb/" + p; a.target = "_blank"; a.rel = "noopener"; cw.appendChild(a); } else cw.append(compLabel(p)); });
    kv("Components", cw); kv("Nucleic acid", g.nuc[i]); kv("Type in the source", g.raw[i]);
    kv("Sources", g.src[i] ? g.src[i].split(";").map((s) => SRC[s] || s).join(", ") : "");
    const lab = g.POS[i] + g.NEG[i], oth = g.REC[i] - lab;
    kv("Records", fmtn(g.REC[i])); kv("Positive", fmtn(g.POS[i])); kv("Negative", fmtn(g.NEG[i])); if (oth > 0) kv("No binary label", fmtn(oth));
    card.appendChild(dl);
    if (lab) {
      const sp = el("div", "sy-split"), a = el("i"), b = el("i"); a.style.flex = String(g.POS[i]); a.style.background = "#F28C28"; b.style.flex = String(g.NEG[i]); b.style.background = "#4D6075"; sp.append(a, b); card.appendChild(sp);
      card.appendChild(el("p", null, g.POS[i] + " of " + lab + " labelled records phase-separate (" + Math.round(100 * g.POS[i] / lab) + "%). Records of one system usually differ in conditions and often come from one paper."));
    } else card.appendChild(el("p", null, "No binary label for this system, so no positive share."));
    const acts = el("div", "sy-acts");
    if (S.pin >= 0) { const b = el("button", null, "Unpin"); b.type = "button"; b.addEventListener("click", () => { S.pin = -1; buildCard(); render(); }); acts.appendChild(b); }
    card.appendChild(acts);
  }
  function showList() {
    const T = TYPES[S.sel], box = $("syList"), lst = (D.lists[T.name] || []).slice().sort((a, b) => b.rec - a.rec); box.textContent = "";
    box.appendChild(el("h2", null, T.name + " · " + lst.length + " systems"));
    box.appendChild(el("p", "sy-hint", "No protein sequence, so no map. One row is one system."));
    const tb = document.createElement("table"), hd = document.createElement("tr");
    ["System", "Nucleic acid", "Type in the source", "Records", "Positive", "Negative", "Sources"].forEach((h) => hd.appendChild(el("th", null, h))); tb.appendChild(hd);
    for (const r of lst) {
      const tr = document.createElement("tr");
      [[r.id], [r.nuc || "–"], [r.raw], [fmtn(r.rec), "n"], [fmtn(r.pos), "n"], [fmtn(r.neg), "n"], [r.src ? r.src.split(";").map((s) => SRC[s] || s).join(", ") : ""]].forEach(([t, c]) => tr.appendChild(el("td", c || null, t)));
      tb.appendChild(tr);
    }
    box.appendChild(tb);
  }

  // ---------------------------------------------------------------- selection, search, parameters
  function applyQuery() {
    const g = cur(), q = S.q.trim().toLowerCase(), box = $("syMatches"); box.textContent = ""; g.match.fill(0);
    if (q.length < 2 || S.list >= 0) return;
    const hits = [];
    for (let i = 0; i < g.n; i++) if (g.comp[i].toLowerCase().includes(q) || g.id[i].toLowerCase() === q || g.nuc[i].toLowerCase().includes(q)) { g.match[i] = 1; hits.push(i); }
    hits.sort((a, b) => g.REC[b] - g.REC[a]);
    for (const i of hits.slice(0, 8)) { const b = el("button", null, compLabel(g.comp[i]) + (g.nuc[i] ? " · " + g.nuc[i] : "") + " · " + fmtn(g.REC[i]) + (g.REC[i] === 1 ? " record" : " records")); b.type = "button"; b.addEventListener("click", () => { S.pin = i; S.hover = i; viewTo(i, 4); buildCard(); render(); }); box.appendChild(b); }
    if (!hits.length) box.appendChild(el("p", "sy-hint", "No system in this map contains it."));
    else if (hits.length > 8) box.appendChild(el("p", "sy-hint", fmtn(hits.length - 8) + " more match on the map."));
  }
  function select(name, keepView) {
    const ti = TYPES.findIndex((t) => t.name === name); if (ti < 0) return;
    const T = TYPES[ti]; S.sel = ti; S.pin = -1; S.hover = -1;
    if (T.kind === "list") {
      S.list = ti; S.t = -1; $("syWrap").hidden = true; $("syList").hidden = false; showList();
      $("syNote").textContent = ""; buildLegend(); buildCard(); return;
    }
    const gi = GK.indexOf(T.group); if (gi < 0) return;
    S.list = -1; S.g = gi; S.t = G[gi].types.length > 1 ? ti : -1; if (!keepView) { S.k = 1; S.tx = 0; S.ty = 0; }
    $("syWrap").hidden = false; $("syList").hidden = true;
    const g = cur();
    $("syNote").textContent = "This map is its own t-SNE of ESMC-600M component embeddings; axes and distances are not comparable across maps." + (S.t >= 0 ? " " + T.name + " systems are ringed; the other types of this map are dimmed." : "");
    applyQuery(); buildLegend(); buildCard(); resize();
  }
  function applyParams(P) {
    S.by = ["anchor", "type", "pos", "rec"].includes(P.get("sby")) ? P.get("sby") : "anchor"; $("syBy").value = S.by;
    S.q = P.get("q") || ""; $("syFq").value = S.q;
    let ti = P.get("t") ? TYPES.findIndex((t) => t.code === P.get("t")) : -1;
    if (ti < 0 && P.get("g")) ti = TYPES.findIndex((t) => t.kind === "map" && t.group === P.get("g"));
    if (ti < 0) ti = TYPES.findIndex((t) => t.name === "protein(2)");
    select(TYPES[ti].name);
    const id = P.get("sys"); if (id && S.list < 0) { const k = cur().id.indexOf(id); if (k >= 0) { S.pin = k; S.hover = k; viewTo(k, 4); buildCard(); render(); } }
  }

  // ---------------------------------------------------------------- interaction
  function nearest(px, py) {
    const g = cur(); let best = -1, bd = (11 * dpr) * (11 * dpr);
    for (let i = 0; i < g.n; i++) { const dx = g.sx[i] - px, dy = g.sy[i] - py, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = i; } }
    return best;
  }
  const pos = (e) => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr]; };
  let drag = null;
  cv.addEventListener("pointerdown", (e) => { cv.setPointerCapture(e.pointerId); drag = { x: e.clientX, y: e.clientY, tx: S.tx, ty: S.ty, moved: false }; });
  cv.addEventListener("pointermove", (e) => {
    if (drag && e.buttons) { const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true; if (drag.moved && S.k > 1) { S.tx = drag.tx + dx * dpr; S.ty = drag.ty + dy * dpr; render(); } return; }
    const p = pos(e), j = nearest(p[0], p[1]); if (j !== S.hover) { S.hover = j; if (S.pin < 0) buildCard(); render(); }
  });
  cv.addEventListener("pointerup", (e) => { const was = drag; drag = null; if (was && !was.moved) { const p = pos(e), j = nearest(p[0], p[1]); S.pin = j; S.hover = j; buildCard(); render(); } });
  cv.addEventListener("pointerleave", () => { if (S.hover >= 0) { S.hover = -1; if (S.pin < 0) buildCard(); render(); } });
  cv.addEventListener("wheel", (e) => { e.preventDefault(); const p = pos(e); zoomAt(p[0], p[1], Math.exp(-e.deltaY * 0.0016)); }, { passive: false });
  $("syZin").addEventListener("click", () => zoomAt(W / 2, H / 2, 1.6)); $("syZout").addEventListener("click", () => zoomAt(W / 2, H / 2, 1 / 1.6)); $("syZfit").addEventListener("click", () => { S.k = 1; S.tx = 0; S.ty = 0; render(); });
  $("syReset").addEventListener("click", () => { S.k = 1; S.tx = 0; S.ty = 0; S.pin = -1; S.hover = -1; S.q = ""; $("syFq").value = ""; S.by = "anchor"; $("syBy").value = "anchor"; applyQuery(); buildLegend(); buildCard(); render(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && S.pin >= 0 && !$("sysPane").hidden) { S.pin = -1; buildCard(); render(); } });
  $("syBy").addEventListener("change", () => { S.by = $("syBy").value; buildLegend(); render(); });
  $("syFq").addEventListener("input", () => { S.q = $("syFq").value; applyQuery(); render(); });
  new ResizeObserver(resize).observe(wrap);
  new MutationObserver(() => { buildLegend(); render(); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  return {
    show() { requestAnimationFrame(() => { resize(); }); },
    select, applyParams,
    state() { return { type: TYPES[S.sel].name, group: cur().key, list: S.list >= 0 }; },
    firstTime() { return !this._done && (this._done = true); }
  };
}
