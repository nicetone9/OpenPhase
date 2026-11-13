/* Guidance tab of the Records space: methods (score field, iso-line and in-region sampling), four main results as interactive maps (hover a point for start, end, the records next to it, sequence change and predictions)
 * and the other examples as GIF rows.
 * Needs: G (guidance data from scripts/rs130_guidance_assets.py), emb ([xs, ys] of the explorer's t-SNE, 0–1 coordinates), rec (record description by row), ps (label code of every record), $ (getElementById), el (element helper).
 * Builds nothing outside #gdPane. GIFs and figures are files next to the page (guidance/…); a GIF is fetched only when its Play button is pressed. */
function initGuidance(api) {
  "use strict";
  const { G, emb, rec, ps: PSC, $, el } = api;
  const BASE = "guidance/", NS = "http://www.w3.org/2000/svg";
  const fmtn = (n) => (n == null ? "–" : n.toLocaleString("en-US"));
  const pct = (v) => (v == null ? "–" : (v > 0 && v < 0.01 ? (v * 100).toFixed(1) : Math.round(v * 100)) + "%");
  const sgn = (v) => (v == null ? "–" : (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2));
  const plural = (n, w) => n + " " + w + (n === 1 ? "" : "s");
  const kb = (k) => (k >= 1000 ? (k / 1000).toFixed(1) + " MB" : k + " KB");
  const root = $("gdPane"), body = $("gdBody"), nav = $("gdNav");
  const T = G.text, B = G.blocks;
  const stops = [];                       // each puts one playing GIF back to its poster
  const maps = [];                        // redraw functions of the point maps
  const stopAll = (except) => stops.forEach((s) => { if (s.owner !== except) s.fn(); });
  const sub = (tag, cls, text) => { const e = el(tag, cls); String(text).split("C_sat").forEach((p, i) => { if (i) { e.append("C"); const s = document.createElement("sub"); s.textContent = "sat"; e.appendChild(s); } if (p) e.append(p); }); return e; };
  const svgEl = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; return e; };
  const kv = (k, v) => { const p = el("p", "gd-kv"); p.append(el("span", null, k + ": "), v); return p; };
  const list = (cls, items) => { const u = el("ul", cls); items.forEach((t) => u.appendChild(el("li", null, t))); return u; };
  const limits = (title, items) => { const d = el("div", "gd-limits"); d.appendChild(el("b", null, title)); d.appendChild(list(null, items)); return d; };
  const col = (n) => getComputedStyle(root).getPropertyValue(n).trim();

  // ---------------- lightbox
  const lb = $("gdLb");
  const openLb = (src, cap) => { $("gdLbImg").src = BASE + src; $("gdLbImg").alt = cap; $("gdLbCap").textContent = cap; lb.hidden = false; };
  const closeLb = () => { lb.hidden = true; $("gdLbImg").removeAttribute("src"); };
  $("gdLbX").addEventListener("click", closeLb);
  lb.addEventListener("click", (e) => { if (e.target === lb) closeLb(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !lb.hidden) closeLb(); });

  const figure = (f) => {
    const fig = el("figure", "gd-fig"), b = el("button"); b.type = "button"; b.title = "Click to enlarge";
    const im = el("img"); im.src = BASE + f.src; im.loading = "lazy"; im.width = f.w; im.height = f.h; im.alt = f.cap; b.appendChild(im);
    b.addEventListener("click", () => openLb(f.src, f.cap));
    fig.append(b, el("figcaption", null, f.cap));
    return fig;
  };

  // ---------------- GIF media: a poster (last frame = final positions); the GIF itself is fetched on request and replayed from a fresh blob URL
  function media(r) {
    const box = el("div", "gd-media"), frame = el("div", "gd-frame"), bar = el("div", "gd-bar"), img = el("img");
    img.alt = r.title + ": final positions of the chains"; img.src = BASE + r.poster; img.loading = "lazy"; img.width = r.gif_info.w; img.height = r.gif_info.h;
    const label = "▶ Play GIF · " + kb(r.gif_kb);
    const play = el("button", "gd-play", label), again = el("button", "gd-again", "↻ Restart"), load = el("span", "gd-load");
    play.type = again.type = "button"; again.hidden = load.hidden = true;
    let blob = null, url = null, on = false;
    const owner = {};
    const revoke = () => { if (url) { URL.revokeObjectURL(url); url = null; } };
    const toPoster = () => { revoke(); img.src = BASE + r.poster; on = false; play.textContent = label; again.hidden = true; };
    const run = () => { revoke(); url = URL.createObjectURL(blob); img.src = url; on = true; play.textContent = "■ Stop"; again.hidden = false; };
    async function start() {
      stopAll(owner);
      if (!blob) {
        load.hidden = false; load.textContent = "loading…";
        try { const resp = await fetch(BASE + r.gif); if (!resp.ok) throw new Error("HTTP " + resp.status); blob = await resp.blob(); }
        catch (e) { load.hidden = true; img.src = BASE + r.gif; on = true; play.textContent = "■ Stop"; return; }
        load.hidden = true;
      }
      run();
    }
    img.addEventListener("error", () => { if (url) { url = null; img.src = BASE + r.gif; } });
    play.addEventListener("click", () => { if (on) toPoster(); else start(); });
    again.addEventListener("click", () => { if (blob) run(); });
    stops.push({ owner, fn: () => { if (on) toPoster(); } });
    frame.appendChild(img); bar.append(play, again, load); box.append(frame, bar);
    box.start = start;
    return box;
  }

  // ---------------- rows
  function rowShell(id) { const row = el("article", "gd-row"); row.id = id; return row; }
  function actions(items) { const d = el("div", "gd-acts"); items.forEach((a) => d.appendChild(a)); return d; }
  function btn(text, fn) { const b = el("button", null, text); b.type = "button"; b.addEventListener("click", fn); return b; }

  function ycRow(r) {
    const row = rowShell("gd-yc-" + r.id), m = media(r), info = el("div", "gd-info");
    info.append(el("h4", null, r.title), el("p", "gd-sub", r.cond));
    info.appendChild(kv("Set", r.set.n1 + " y = 1 · " + r.set.n0 + " y = 0 records · " + plural(r.set.seq, "sequence") + " · " + plural(r.set.papers, "paper")));
    info.appendChild(kv("On the map", "the two labels sit " + r.map.dist.toFixed(2) + " SD apart; mean r, y = 1 minus y = 0: " + sgn(r.map.dr)));
    const t = el("table", "gd-tbl"), h = el("thead"), hr = el("tr");
    ["", "y = 0 → 1", "y = 1 → 0"].forEach((x, i) => hr.appendChild(el("th", i ? "n" : null, x))); h.appendChild(hr); t.appendChild(h);
    const tb = el("tbody"), A = r.dir["0to1"], Bd = r.dir["1to0"];
    const add = (name, f) => { const tr = el("tr"); tr.appendChild(el("td", null, name)); [A, Bd].forEach((d) => { const td = el("td", "n"); f(td, d); tr.appendChild(td); }); tb.appendChild(tr); };
    const withNone = (td, v, n) => { td.append(v + " "); td.appendChild(el("span", "m", "(none " + n + ")")); };
    add("chains", (td, d) => { td.textContent = fmtn(d.chains); });
    add("median Δr, guided", (td, d) => { td.textContent = sgn(d.dr); });
    add("moving toward the target label", (td, d) => withNone(td, pct(d.toward), pct(d.toward_none)));
    add("ending within 0.10 SD of a target record", (td, d) => withNone(td, pct(d.arr), pct(d.arr_none)));
    add("median end → nearest target record (SD)", (td, d) => withNone(td, d.dend.toFixed(2), d.dend_none.toFixed(2)));
    t.appendChild(tb); info.appendChild(t);
    if (r.notes && r.notes.length) info.appendChild(list("gd-notes", r.notes));
    const acts = [];
    if (r.result) acts.push(btn("Final positions and statistics", () => openLb(r.result, r.title + " — final positions of all chains, change of r, share ending near target-label records")));
    acts.push(btn("Copy link", () => copyLink("yc", r.id)));
    info.appendChild(actions(acts));
    row.append(m, info); row.media = m;
    return row;
  }

  function isoRow(r) {
    const row = rowShell("gd-iso-" + r.id), m = media(r), info = el("div", "gd-info"), L = r.line, H = r.hmc;
    info.append(el("h4", null, r.title));
    const bits = []; if (L.arc != null) bits.push("line " + L.arc + " SD long"); if (L.points != null) bits.push(fmtn(L.points) + " points"); if (H.to_y1.pieces != null) bits.push(H.to_y1.pieces + " pieces");
    info.appendChild(el("p", "gd-sub", bits.join(" · ")));
    if (L.n_near != null) info.appendChild(kv("Near the line", fmtn(L.n_near) + " records (" + fmtn(L.n_y1) + " y = 1 / " + fmtn(L.n_y0) + " y = 0) · " + plural(L.papers, "paper") + " · " + plural(L.prots, "protein")));
    if (L.purity != null) info.appendChild(kv("Homogeneity", L.purity.toFixed(2) + " (lower is better)"));
    const t = el("table", "gd-tbl"), h = el("thead"), hr = el("tr");
    ["HMC, 400 proposals", "guided to y = 1", "guided to y = 0"].forEach((x, i) => hr.appendChild(el("th", i ? "n" : null, x))); h.appendChild(hr); t.appendChild(h);
    const tb = el("tbody");
    const add = (name, f) => { const tr = el("tr"); tr.appendChild(el("td", null, name)); ["to_y1", "to_y0"].forEach((k) => tr.appendChild(el("td", "n", f(H[k])))); tb.appendChild(tr); };
    add("distance to the exact distribution (TV)", (d) => d.tv.toFixed(2));
    add("acceptance rate", (d) => d.acc.toFixed(2));
    add("median move along the line (SD)", (d) => d.move.toFixed(1));
    add("chains off the data", (d) => pct(d.off));
    t.appendChild(tb); info.appendChild(t);
    if (r.notes && r.notes.length) info.appendChild(list("gd-notes", r.notes));
    info.appendChild(actions([btn("Copy link", () => copyLink("iso", r.id))]));
    row.append(m, info); row.media = m;
    return row;
  }

  // ---------------- map point -> sequence
  function ptsRow(r) {
    const row = rowShell("gd-pts-" + r.id), info = el("div", "gd-info");
    const mapBox = el("div", "gd-map"), cv = el("canvas"); mapBox.appendChild(cv);
    const key = el("div", "gd-key");
    [["gd-y1", "y = 1 point"], ["gd-y0", "y = 0 point"]].forEach(([c, t]) => { const s = el("span"); const i = el("i"); i.style.background = c === "gd-y1" ? "var(--gd-y1)" : "var(--gd-y0)"; s.append(i, t); key.appendChild(s); });
    const s1 = el("span"); const ring = el("i"); ring.style.cssText = "background:transparent;border:2px solid var(--mute)"; s1.append(ring, "open: the answer is the anchor (k* = 0)"); key.appendChild(s1);
    const s2 = el("span"); const hi = el("i"); hi.style.background = "var(--gd-hi)"; s2.append(hi, "records of " + r.protein); key.appendChild(s2);
    mapBox.appendChild(key);
    const P = r.points, xs = P.map((p) => p.x), ys = P.map((p) => p.yy);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const s = Math.max(x1 - x0, y1 - y0, 0.04) * 1.55, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, view = { x: cx - s / 2, y: cy - s / 2, s };
    const HI = B.points.highlight[r.hi] || [];
    let sel = -1;
    const trs = [];
    const toPx = (x, y, W) => [((x - view.x) / view.s) * W, ((y - view.y) / view.s) * W];
    function draw() {
      const dpr = window.devicePixelRatio || 1, cssW = cv.clientWidth || 420, W = Math.round(cssW * dpr);
      if (cv.width !== W) { cv.width = W; cv.height = W; }
      const g = cv.getContext("2d"); g.clearRect(0, 0, W, W);
      const ex = emb[0], ey = emb[1], n = ex.length, bg = col("--gd-bg"), hc = col("--gd-hi"), c1 = col("--gd-y1"), c0 = col("--gd-y0"), ink = col("--mute");
      g.fillStyle = bg; const rr = 1.5 * dpr;
      for (let i = 0; i < n; i++) { const [px, py] = toPx(ex[i], ey[i], W); if (px < -4 || py < -4 || px > W + 4 || py > W + 4) continue; g.fillRect(px - rr / 2, py - rr / 2, rr, rr); }
      g.fillStyle = hc; g.globalAlpha = 0.7; const hr = 2.2 * dpr;
      for (let k = 0; k < HI.length; k++) { const i = HI[k], [px, py] = toPx(ex[i], ey[i], W); if (px < -4 || py < -4 || px > W + 4 || py > W + 4) continue; g.beginPath(); g.arc(px, py, hr, 0, 6.2832); g.fill(); }
      g.globalAlpha = 1; g.font = Math.round(10 * dpr) + "px system-ui, sans-serif";
      P.forEach((p, i) => {
        const [px, py] = toPx(p.x, p.yy, W), c = p.y ? c1 : c0, R = (i === sel ? 8 : 6) * dpr;
        g.beginPath(); g.arc(px, py, R, 0, 6.2832);
        if (p.k > 0) { g.fillStyle = c; g.fill(); g.lineWidth = 1.2 * dpr; g.strokeStyle = "#fff"; g.stroke(); } else { g.lineWidth = 2.2 * dpr; g.strokeStyle = c; g.stroke(); }
        g.fillStyle = i === sel ? c : ink; g.fillText("y" + p.y + " P" + p.i, px + R + 2 * dpr, py - R + 3 * dpr);
      });
    }
    maps.push(draw);
    const pick = (e) => { const b = cv.getBoundingClientRect(), W = b.width; let best = -1, bd = 14 * 14; P.forEach((p, i) => { const [px, py] = toPx(p.x, p.yy, W), d = (px - (e.clientX - b.left)) ** 2 + (py - (e.clientY - b.top)) ** 2; if (d < bd) { bd = d; best = i; } }); return best; };
    const setSel = (i) => { sel = i; trs.forEach((t, j) => t.classList.toggle("gd-sel", j === i)); draw(); };
    cv.addEventListener("mousemove", (e) => { const i = pick(e); if (i !== sel) setSel(i); });
    cv.addEventListener("mouseleave", () => setSel(-1));
    row.appendChild(mapBox);

    info.appendChild(el("h4", null, r.title));
    const anchors = [...new Set(P.map((p) => p.anchor))];
    info.appendChild(el("p", "gd-sub", r.protein + " · 12 points · " + anchors.length + " anchor sequence" + (anchors.length > 1 ? "s" : "") + " · " + (P.length - r.n_k0) + " of 12 answers carry substitutions"));
    const t = el("table", "gd-tbl"), th = el("thead"), hr = el("tr");
    [["Point", "", "y = 1 / y = 0: guided to; P0\u2013P5: quantile of the exact samples along the line"], ["Anchor", "", "Real sequence with the largest weight near the point (library row number, length)"], ["k*", "n", "Number of substitutions in the answer; 0 means the anchor itself"], ["D to target", "n", "Distance of the ESMC embedding to the target: anchor \u2192 answer"], ["Substitutions", "", "Reference residue, position on the anchor, new residue"]].forEach(([x, c, tip]) => { const h2 = el("th", c, x); h2.title = tip; hr.appendChild(h2); }); th.appendChild(hr); t.appendChild(th);
    const tb = el("tbody");
    P.forEach((p, i) => {
      const tr = el("tr"); trs.push(tr);
      const c0 = el("td"); c0.appendChild(el("span", "y" + p.y, "y" + p.y + " P" + p.i)); tr.appendChild(c0);
      tr.appendChild(el("td", "m", "#" + p.anchor + " · " + p.len + " aa"));
      tr.appendChild(el("td", "n", String(p.k)));
      tr.appendChild(el("td", "n", p.k ? p.dsrc.toFixed(3) + " → " + p.dk.toFixed(3) : p.dsrc.toFixed(3)));
      tr.appendChild(el("td", p.k ? null : "m", p.k ? p.edits.slice().sort((a, b) => a[1] - b[1]).map((e) => e[0] + e[1] + e[2]).join(", ") : "none (the anchor itself)"));
      tr.addEventListener("mouseenter", () => setSel(i)); tr.addEventListener("mouseleave", () => setSel(-1));
      tb.appendChild(tr);
    });
    t.appendChild(tb); info.appendChild(t);
    const st = strip(r); if (st) { info.appendChild(el("div", "gd-sec-h", "Where the substitutions sit (position on the point's anchor)")); info.appendChild(st); }
    const mx = matrix(r); if (mx) { info.appendChild(el("div", "gd-sec-h", "Pairwise differences between the 12 answers (substitutions after a global alignment)")); info.appendChild(mx); }
    const a = el("a", null, "Alignments of the most different pairs (text)"); a.href = BASE + B.points.align_file; a.target = "_blank"; a.rel = "noopener";
    info.appendChild(actions([a, btn("Copy link", () => copyLink("pts", r.id))]));
    row.appendChild(info);
    return row;
  }

  function strip(r) {
    const R = r.points.map((p, i) => [p, i]).filter(([p]) => p.k > 0);
    if (!R.length) return null;
    const L = Math.max(...r.points.map((p) => p.len)), W = 560, padL = 92, padR = 10, rowH = 16, top = 4, H = top + R.length * rowH + 20, sx = (v) => padL + (v / L) * (W - padL - padR);
    const svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, class: "gd-svg", role: "img", "aria-label": "Positions of the substitutions on each anchor" });
    for (let v = 0; v <= L; v += 100) { svg.appendChild(svgEl("line", { x1: sx(v), x2: sx(v), y1: top, y2: H - 16, stroke: col("--line") || "#ccc", "stroke-width": 0.6 })); svg.appendChild(svgEl("text", { x: sx(v), y: H - 5, "text-anchor": "middle" }, String(v))); }
    R.forEach(([p], j) => {
      const y = top + j * rowH + rowH / 2;
      svg.appendChild(svgEl("line", { x1: sx(0), x2: sx(p.len), y1: y, y2: y, stroke: col("--gd-bg"), "stroke-width": 2 }));
      svg.appendChild(svgEl("text", { x: padL - 6, y: y + 3, "text-anchor": "end", class: "gd-ink" }, "y" + p.y + " P" + p.i + " · k* = " + p.k));
      p.edits.forEach((e) => svg.appendChild(svgEl("rect", { x: sx(e[1]) - 1, y: y - 5, width: 2.2, height: 10, fill: col("--gd-hot") })));
    });
    return svg;
  }

  function matrix(r) {
    const M = r.matrix, n = r.points.length;
    if (!M || M.length !== n) return null;
    const cell = 22, padL = 44, padT = 38, W = padL + n * cell + 6, H = padT + n * cell + 6, mx = Math.max(1, ...M.flat());
    const svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, class: "gd-svg", style: "max-width:" + W * 1.25 + "px", role: "img", "aria-label": "Pairwise substitution counts between the 12 answers" });
    r.points.forEach((p, i) => {
      const lab = "y" + p.y + " P" + p.i;
      svg.appendChild(svgEl("text", { x: padL - 4, y: padT + i * cell + cell / 2 + 3, "text-anchor": "end", class: "gd-ink" }, lab));
      svg.appendChild(svgEl("text", { transform: "translate(" + (padL + i * cell + cell / 2 + 3) + "," + (padT - 4) + ") rotate(-90)", class: "gd-ink" }, lab));
    });
    M.forEach((row, i) => row.forEach((v, j) => {
      svg.appendChild(svgEl("rect", { x: padL + j * cell, y: padT + i * cell, width: cell - 1, height: cell - 1, rx: 2, fill: col("--gd-y1"), "fill-opacity": v ? 0.1 + 0.8 * v / mx : 0.04 }));
      if (v) svg.appendChild(svgEl("text", { x: padL + j * cell + (cell - 1) / 2, y: padT + i * cell + cell / 2 + 3, "text-anchor": "middle", class: v / mx > 0.55 ? "" : "gd-ink", style: v / mx > 0.55 ? "fill:#fff" : "" }, String(v)));
    }));
    return svg;
  }

  function failedList(g) {
    const d = el("details", "gd-more"), sm = el("summary", null, "Lines tested but not shown (" + g.failed.length + ")");
    d.appendChild(sm);
    if (g.rule) d.appendChild(el("p", null, "Rule: " + g.rule));
    const t = el("table", "gd-tbl"), tb = el("tbody");
    g.failed.forEach((x) => { const tr = el("tr"); tr.append(el("td", null, x.label), el("td", "m", x.why)); tb.appendChild(tr); });
    t.appendChild(tb); d.appendChild(t);
    return d;
  }

  // ---------------- records: one-line descriptions (from the explorer's own record data)
  const hexRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const sgnN = (v, d) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d == null ? 2 : d);
  const crowdTxt = (c) => (!c || c === "none reported" || c === "not given" ? "" : c);
  function condText(r) {
    const a = [];
    if (r.T != null) a.push(r.T + " °C");
    if (r.pH != null) a.push("pH " + r.pH);
    if (r.NaCl != null) a.push(fmtn(r.NaCl) + " mM NaCl");
    if (r.conc != null) a.push(fmtn(r.conc) + " µM");
    const c = crowdTxt(r.crowder); if (c) a.push(c);
    return a.join(" · ");
  }
  const psWord = (k) => (k === 1 ? "phase separation" : k === 0 ? "no phase separation" : "csat record");
  function refItem(j, d) {
    const r = rec(j), li = el("li", "gd-ref"), dot = el("i");
    dot.style.background = r.ps === 1 ? "var(--gd-y1)" : r.ps === 0 ? "var(--gd-y0)" : "var(--mute)";
    li.append(dot, el("b", null, r.name + (r.uni && r.uni !== r.name ? " · " + r.uni : "")), el("span", "m", " " + r.paper + " · " + psWord(r.ps) + (condText(r) ? " · " + condText(r) : "") + (d != null ? " · " + d.toFixed(2) + " SD" : "")));
    return li;
  }
  function refList(idx, dist, c) { const ul = el("ul", "gd-refs"); for (let k = 0; k < 3; k++) ul.appendChild(refItem(idx[3 * c + k], dist ? dist[3 * c + k] : null)); return ul; }
  const csec = (h, tip) => { const d = el("div", "gd-cs"), b = el("b", "gd-csh", h); if (tip) b.title = tip; d.appendChild(b); return d; };
  const cline = (t, cls) => el("p", cls || "gd-cl", t);
  const R_TIP = "r = log p(u|1) − log p(u|0) of the score field (TorchEBM, σ = 0.16); p(y = 1) = σ(r + log(6,613 / 4,050)). The field describes where each label sits on this map; with papers held out it has no skill for new papers.";
  const fieldTxt = (r, p) => "r = " + sgnN(r) + " · p(y = 1) = " + p.toFixed(2);

  // ---------------- interactive result maps
  let terCache = null, terDark = null;
  function territory() {
    const th = document.documentElement.getAttribute("data-theme"), dark = th ? th === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    if (terCache && terDark === dark) return terCache;
    const Gn = 220, c1 = new Float32Array(Gn * Gn), c0 = new Float32Array(Gn * Gn), ex = emb[0], ey = emb[1];
    for (let i = 0; i < ex.length; i++) { const k = PSC[i]; if (k > 1) continue; const gx = Math.min(Gn - 1, Math.max(0, Math.floor(ex[i] * Gn))), gy = Math.min(Gn - 1, Math.max(0, Math.floor(ey[i] * Gn))); (k === 1 ? c1 : c0)[gy * Gn + gx] += 1; }
    const kern = [1, 4, 6, 4, 1];
    const blur = (a) => {
      const t = new Float32Array(Gn * Gn), o = new Float32Array(Gn * Gn);
      for (let y = 0; y < Gn; y++) for (let x = 0; x < Gn; x++) { let s = 0; for (let d = -2; d <= 2; d++) { const xx = x + d; if (xx >= 0 && xx < Gn) s += a[y * Gn + xx] * kern[d + 2]; } t[y * Gn + x] = s / 16; }
      for (let y = 0; y < Gn; y++) for (let x = 0; x < Gn; x++) { let s = 0; for (let d = -2; d <= 2; d++) { const yy = y + d; if (yy >= 0 && yy < Gn) s += t[yy * Gn + x] * kern[d + 2]; } o[y * Gn + x] = s / 16; }
      return o;
    };
    const b1 = blur(blur(blur(c1))), b0 = blur(blur(blur(c0)));
    let mx = 0; for (let i = 0; i < b1.length; i++) mx = Math.max(mx, b1[i] + b0[i]);
    const cv = document.createElement("canvas"); cv.width = cv.height = Gn;
    const g = cv.getContext("2d"), im = g.createImageData(Gn, Gn), C1 = hexRgb(col("--gd-t1")), C0 = hexRgb(col("--gd-t0"));
    for (let i = 0; i < Gn * Gn; i++) { const a = Math.min(1, Math.max(0, ((b1[i] + b0[i]) / mx - 0.02) / 0.09)), c = b1[i] >= b0[i] ? C1 : C0; im.data[4 * i] = c[0]; im.data[4 * i + 1] = c[1]; im.data[4 * i + 2] = c[2]; im.data[4 * i + 3] = Math.round(255 * a * 0.9); }
    g.putImageData(im, 0, 0);
    terCache = cv; terDark = dark;
    return cv;
  }

  function resultMap(M) {
    const kind = M.kind, box = el("article", "gd-rm");
    box.id = "gd-" + M.id;
    box.appendChild(el("h4", null, M.title));
    (M.desc || []).forEach((t) => box.appendChild(el("p", "gd-rm-desc", t)));
    const body = el("div", "gd-rm-body"), left = el("div", "gd-rm-left"), card = el("aside", "gd-rm-card");
    const wrap = el("div", "gd-map gd-rm-map"), cv = el("canvas"), zb = el("div", "gd-zoom");
    cv.setAttribute("role", "img"); cv.setAttribute("aria-label", M.title + ": interactive map; hover a point for its details");
    const zin = el("button", null, "+"), zout = el("button", null, "−"), zfit = el("button", null, "Fit");
    const zfull = el("button", null, "Full map");
    [zin, zout, zfit].concat(kind === "pts" ? [zfull] : []).forEach((b) => { b.type = "button"; zb.appendChild(b); });
    zin.setAttribute("aria-label", "Zoom in"); zout.setAttribute("aria-label", "Zoom out");
    wrap.append(cv, zb);
    const key = el("div", "gd-rm-key");
    left.append(wrap, key); body.append(left, card); box.appendChild(body);

    const GR = (M.groups || []).map((g) => ({ g, vis: true, n: g.n, S: g.S ? Float32Array.from(g.S) : null, E: g.E ? Float32Array.from(g.E) : null, C: g.control ? Float32Array.from(g.control.E) : null }));
    const IT = M.items || [];
    let fitV = { x: 0, y: 0, s: 1 };
    if (kind === "pts" && IT.length) {          // the 12 points sit in a small part of the map: start zoomed on them
      const xs = IT.map((p) => p.x).concat(IT.filter((p) => p.start >= 0).map((p) => emb[0][p.start])), ys = IT.map((p) => p.yy).concat(IT.filter((p) => p.start >= 0).map((p) => emb[1][p.start]));
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), s0 = Math.max(x1 - x0, y1 - y0, 0.05) * 1.7;
      fitV = { x: (x0 + x1) / 2 - s0 / 2, y: (y0 + y1) / 2 - s0 / 2, s: s0 };
    }
    const view = Object.assign({}, fitV);
    let hov = null, pin = null, ctrl = false, drag = null;
    const same = (a, b) => a && b && a.gi === b.gi && a.c === b.c && a.role === b.role;
    const cur = () => pin || hov;

    // legend and toggles
    const addToggle = (color, text, get, set) => {
      const b = el("button", "gd-tg on"), i = el("i"); b.type = "button"; i.style.background = color; b.append(i, text);
      b.addEventListener("click", () => { const v = !get(); set(v); b.classList.toggle("on", v); draw(); });
      key.appendChild(b);
    };
    GR.forEach((o, gi) => addToggle(o.g.color === "y1" ? "var(--gd-y1)" : o.g.color === "y0" ? "var(--gd-y0)" : "var(--mute)", o.g.label + " (" + fmtn(o.n) + ")", () => o.vis, (v) => { o.vis = v; if (!v && cur() && cur().gi === gi) { pin = null; hov = null; showCard(); } }));
    if (GR.some((o) => o.C)) { const b = el("button", "gd-tg"), i = el("i"); b.type = "button"; i.style.background = "var(--mute)"; b.append(i, "unguided control ends"); b.addEventListener("click", () => { ctrl = !ctrl; b.classList.toggle("on", ctrl); draw(); }); key.appendChild(b); }
    if (kind === "pts") {
      const s1 = el("span", "gd-kk"), ring = el("i"); ring.style.cssText = "background:transparent;border:2px solid var(--mute)"; s1.append(ring, "open: the answer is the anchor (k* = 0)"); key.appendChild(s1);
      const s2 = el("span", "gd-kk"), fill = el("i"); fill.style.background = "var(--mute)"; s2.append(fill, "filled: the answer carries substitutions"); key.appendChild(s2);
    }
    const sk = el("span", "gd-kk"), sd = el("i"); sd.style.cssText = "background:var(--gd-hot);border:1px solid var(--ink)"; sk.append(sd, kind === "region" ? "region (shaded)" : kind === "pts" ? "start: closest record of the anchor" : "start"); key.appendChild(sk);

    // drawing
    function toPx(x, y, Wp) { return [((x - view.x) / view.s) * Wp, ((y - view.y) / view.s) * Wp]; }
    function draw() {
      const dpr = window.devicePixelRatio || 1, cssW = cv.clientWidth || 560, Wd = Math.round(cssW * dpr);
      if (cv.width !== Wd) { cv.width = Wd; cv.height = Wd; }
      const g = cv.getContext("2d"), Wp = cv.width;
      g.clearRect(0, 0, Wp, Wp);
      const T = territory(), Gn = T.width; g.imageSmoothingEnabled = true; g.drawImage(T, view.x * Gn, view.y * Gn, view.s * Gn, view.s * Gn, 0, 0, Wp, Wp);
      const ex = emb[0], ey = emb[1], rr = (1.3 + 0.7 * Math.min(3, 1 / view.s)) * dpr;
      g.fillStyle = col("--gd-bg");
      for (let i = 0; i < ex.length; i++) { const px = ((ex[i] - view.x) / view.s) * Wp, py = ((ey[i] - view.y) / view.s) * Wp; if (px < -3 || py < -3 || px > Wp + 3 || py > Wp + 3) continue; g.fillRect(px - rr / 2, py - rr / 2, rr, rr); }
      const c1 = col("--gd-y1"), c0 = col("--gd-y0"), ink = col("--ink"), hot = col("--gd-hot"), colOf = (k) => (k === "y1" ? c1 : k === "y0" ? c0 : col("--mute"));
      if (kind === "region") {
        const cs = (M.cell_size / view.s) * Wp + 0.6; g.fillStyle = col("--gd-region");
        const c = M.cells; for (let i = 0; i < c.length; i += 2) { const [px, py] = toPx(c[i], c[i + 1], Wp); g.fillRect(px - cs / 2, py - cs / 2, cs, cs); }
      }
      if (kind === "iso") {
        g.strokeStyle = ink; g.lineWidth = 1.5 * dpr; g.globalAlpha = 0.8;
        M.line.forEach((pl) => { g.beginPath(); for (let i = 0; i < pl.length; i += 2) { const [px, py] = toPx(pl[i], pl[i + 1], Wp); if (i) g.lineTo(px, py); else g.moveTo(px, py); } g.stroke(); });
        g.globalAlpha = 1;
      }
      // connectors
      GR.forEach((o) => {
        if (!o.vis || !o.S || !o.E) return;
        g.strokeStyle = colOf(o.g.color); g.globalAlpha = o.n > 800 ? 0.07 : 0.14; g.lineWidth = 1 * dpr; g.beginPath();
        for (let c = 0; c < o.n; c++) { const s = o.g.sid[c], [ax, ay] = toPx(o.S[2 * s], o.S[2 * s + 1], Wp), [bx, by] = toPx(o.E[2 * c], o.E[2 * c + 1], Wp); g.moveTo(ax, ay); g.lineTo(bx, by); }
        g.stroke(); g.globalAlpha = 1;
      });
      // unguided control ends
      if (ctrl) GR.forEach((o) => { if (!o.vis || !o.C) return; g.fillStyle = col("--mute"); g.globalAlpha = 0.45; for (let c = 0; c < o.n; c++) { const [px, py] = toPx(o.C[2 * c], o.C[2 * c + 1], Wp); g.fillRect(px - 1.2 * dpr, py - 1.2 * dpr, 2.4 * dpr, 2.4 * dpr); } g.globalAlpha = 1; });
      // starts
      g.lineWidth = 1 * dpr;
      GR.forEach((o) => { if (!o.vis || !o.S) return; g.fillStyle = hot; g.strokeStyle = ink; for (let s = 0; s < o.S.length / 2; s++) { const [px, py] = toPx(o.S[2 * s], o.S[2 * s + 1], Wp); g.beginPath(); g.arc(px, py, 2.4 * dpr, 0, 6.2832); g.fill(); g.stroke(); } });
      // ends / samples
      GR.forEach((o) => { if (!o.vis || !o.E) return; g.fillStyle = colOf(o.g.color); g.strokeStyle = "rgba(255,255,255,0.85)"; g.lineWidth = 0.7 * dpr; for (let c = 0; c < o.n; c++) { const [px, py] = toPx(o.E[2 * c], o.E[2 * c + 1], Wp); g.beginPath(); g.arc(px, py, (kind === "region" ? 2.5 : o.n > 800 ? 2.1 : 2.8) * dpr, 0, 6.2832); g.fill(); g.stroke(); } });
      // decoded points
      if (kind === "pts") {
        g.font = Math.round(10 * dpr) + "px system-ui, sans-serif";
        IT.forEach((p, i) => {
          const [px, py] = toPx(p.x, p.yy, Wp), col_ = p.y ? c1 : c0, a = p.start >= 0 ? toPx(ex[p.start], ey[p.start], Wp) : null;
          if (a) { g.strokeStyle = col_; g.globalAlpha = 0.5; g.lineWidth = 1 * dpr; g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(px, py); g.stroke(); g.globalAlpha = 1; g.fillStyle = hot; g.strokeStyle = ink; g.beginPath(); g.arc(a[0], a[1], 2.6 * dpr, 0, 6.2832); g.fill(); g.stroke(); }
          g.beginPath(); g.arc(px, py, 6 * dpr, 0, 6.2832);
          if (p.k > 0) { g.fillStyle = col_; g.fill(); g.lineWidth = 1.4 * dpr; g.strokeStyle = "#fff"; g.stroke(); } else { g.lineWidth = 2.2 * dpr; g.strokeStyle = col_; g.stroke(); }
          g.fillStyle = col("--mute"); g.fillText("y" + p.y + " P" + p.i, px + 8 * dpr, py - 6 * dpr);
        });
      }
      // hover / pin
      const sel = cur();
      if (sel) {
        g.lineWidth = 2.2 * dpr; g.strokeStyle = ink;
        if (sel.role === "pt") { const p = IT[sel.c], [px, py] = toPx(p.x, p.yy, Wp); g.beginPath(); g.arc(px, py, 10 * dpr, 0, 6.2832); g.stroke(); }
        else {
          const o = GR[sel.gi];
          if (sel.role === "end") {
            const [bx, by] = toPx(o.E[2 * sel.c], o.E[2 * sel.c + 1], Wp);
            if (o.S) { const s = o.g.sid[sel.c], [ax, ay] = toPx(o.S[2 * s], o.S[2 * s + 1], Wp); g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.stroke(); g.fillStyle = hot; g.beginPath(); g.arc(ax, ay, 4.2 * dpr, 0, 6.2832); g.fill(); g.stroke(); }
            g.beginPath(); g.arc(bx, by, 6.5 * dpr, 0, 6.2832); g.stroke();
          } else {
            const [ax, ay] = toPx(o.S[2 * sel.c], o.S[2 * sel.c + 1], Wp);
            g.beginPath(); g.arc(ax, ay, 7 * dpr, 0, 6.2832); g.stroke();
            for (let c = 0; c < o.n; c++) if (o.g.sid[c] === sel.c) { const [bx, by] = toPx(o.E[2 * c], o.E[2 * c + 1], Wp); g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.stroke(); }
          }
        }
      }
    }
    maps.push(draw);

    // hit testing
    function hit(e) {
      const b = cv.getBoundingClientRect(), mx = e.clientX - b.left, my = e.clientY - b.top, Wc = b.width, X = (x) => ((x - view.x) / view.s) * Wc, Y = (y) => ((y - view.y) / view.s) * Wc;
      let best = null, bd = 12 * 12;
      GR.forEach((o, gi) => { if (!o.vis || !o.E) return; for (let c = 0; c < o.n; c++) { const d = (X(o.E[2 * c]) - mx) ** 2 + (Y(o.E[2 * c + 1]) - my) ** 2; if (d < bd) { bd = d; best = { gi, c, role: kind === "region" ? "end" : "end" }; } } });
      if (!best) GR.forEach((o, gi) => { if (!o.vis || !o.S) return; for (let s = 0; s < o.S.length / 2; s++) { const d = (X(o.S[2 * s]) - mx) ** 2 + (Y(o.S[2 * s + 1]) - my) ** 2; if (d < bd) { bd = d; best = { gi, c: s, role: "start" }; } } });
      if (kind === "pts") IT.forEach((p, c) => { const d = (X(p.x) - mx) ** 2 + (Y(p.yy) - my) ** 2; if (d < bd) { bd = d; best = { gi: 0, c, role: "pt" }; } });
      return best;
    }

    // cards
    function showCard() {
      card.textContent = "";
      const s = cur();
      if (!s) { card.appendChild(cardDefault()); return; }
      card.appendChild(s.role === "pt" ? cardPt(IT[s.c]) : kind === "region" ? cardSample(GR[s.gi].g, s.c) : s.role === "start" ? cardStart(GR[s.gi], s.c) : cardChain(GR[s.gi].g, s.c));
      if (pin) { const b = el("button", "gd-unpin", "Unpin"); b.type = "button"; b.addEventListener("click", () => { pin = null; hov = null; showCard(); draw(); }); card.appendChild(b); }
    }
    function cardDefault() {
      const d = el("div", "gd-cd"); d.appendChild(el("h5", null, "Hover a point"));
      d.appendChild(cline("Click to pin a point, drag to pan, scroll to zoom.", "gd-cl m"));
      (M.reading || []).forEach((t) => d.appendChild(cline(t)));
      return d;
    }
    const ttl = (t, sub, color) => { const h = el("h5"), i = el("i"); i.style.background = color === "y1" ? "var(--gd-y1)" : color === "y0" ? "var(--gd-y0)" : "var(--mute)"; h.append(i, t); const d = el("div", "gd-cd"); d.appendChild(h); if (sub) d.appendChild(cline(sub, "gd-cl m")); return d; };
    function predBlock(g, s, c) {
      const d = csec("Prediction · score field", R_TIP);
      const rs = s != null ? g.rs[s] : null, ps = s != null ? g.ps[s] : null;
      if (rs != null) d.appendChild(cline("p(y = 1): " + ps.toFixed(2) + " → " + g.pe[c].toFixed(2) + "  (r " + sgnN(rs) + " → " + sgnN(g.re[c]) + ", Δr " + sgnN(g.re[c] - rs) + ")"));
      else d.appendChild(cline("p(y = 1) = " + g.pe[c].toFixed(2) + "  (r " + sgnN(g.re[c]) + ")"));
      return d;
    }
    function protBlock(g, s, c) {
      const d = csec("Sequence", "The chain ends are map positions; the sequence is not decoded here (the decoded examples below do that)");
      const a = rec(g.srec ? g.srec[s] : g.ns[3 * s]), b = rec(g.ne[3 * c]);
      const same = a.uni && b.uni && a.uni === b.uni;
      d.appendChild(cline("Protein at the start: " + a.name + (a.uni && a.uni !== a.name ? " (" + a.uni + ")" : "") + " → next to the end: " + b.name + (b.uni && b.uni !== b.name ? " (" + b.uni + ")" : "") + (same ? " — the same protein" : " — a different protein")));
      d.appendChild(cline("Not decoded: a chain end is a map position, not a sequence.", "gd-cl m"));
      return d;
    }
    function cardChain(g, c) {
      const s = g.sid[c], d = ttl("Chain " + (c + 1) + " · " + g.label, kind === "iso" ? M.facts.proposals + " HMC proposals on the line" : fmtn(M.facts.steps) + " Langevin steps, w = " + M.facts.w, g.color);
      const a = csec("Start");
      if (kind === "iso") a.appendChild(cline("On the line, piece " + (g.seg0[s] + 1) + " of " + M.facts.pieces + ", arc " + g.arc0[s].toFixed(1) + " SD"));
      else a.appendChild(cline("A record of the start label (" + M.facts.K + " chains start from each record, 0.03 SD jitter)"));
      a.appendChild(cline(fieldTxt(g.rs[s], g.ps[s]), "gd-cl m"));
      if (kind === "iso") { a.appendChild(el("p", "gd-cl m", "Nearest records (reference). They need not meet the line's condition: the line is a smoothed contour.")); a.appendChild(refList(g.ns, g.nsd, s)); } else { const u = el("ul", "gd-refs"); u.appendChild(refItem(g.srec[s], null)); a.appendChild(u); }
      const b = csec("End");
      if (kind === "iso") b.appendChild(cline("Piece " + (g.seg1[c] + 1) + ", arc " + g.arc1[c].toFixed(1) + " SD (" + (g.seg1[c] === g.seg0[s] ? "moved " + sgnN(g.arc1[c] - g.arc0[s], 1) + " SD along the piece" : "a different piece than the start") + ")"));
      b.appendChild(cline(fieldTxt(g.re[c], g.pe[c]) + " · Δr " + sgnN(g.re[c] - g.rs[s]), "gd-cl m"));
      b.appendChild(el("p", "gd-cl m", "Records next to the end (reference):")); b.appendChild(refList(g.ne, g.ned, c));
      d.append(a, b, predBlock(g, s, c), protBlock(g, s, c));
      return d;
    }
    function cardStart(o, s) {
      const g = o.g, d = ttl("Start " + (s + 1) + " · " + g.label, null, g.color), a = csec("Start");
      if (kind === "iso") a.appendChild(cline("On the line, piece " + (g.seg0[s] + 1) + ", arc " + g.arc0[s].toFixed(1) + " SD"));
      else a.appendChild(cline("A record of the start label"));
      a.appendChild(cline(fieldTxt(g.rs[s], g.ps[s]), "gd-cl m"));
      if (kind === "iso") a.appendChild(refList(g.ns, g.nsd, s)); else { const u = el("ul", "gd-refs"); u.appendChild(refItem(g.srec[s], null)); a.appendChild(u); }
      const dr = []; for (let c = 0; c < o.n; c++) if (g.sid[c] === s) dr.push(g.re[c] - g.rs[s]);
      dr.sort((x, y) => x - y);
      d.appendChild(a); d.appendChild(cline(dr.length + " chains start here; median Δr " + sgnN(dr[Math.floor(dr.length / 2)]) + " (" + sgnN(dr[0]) + " to " + sgnN(dr[dr.length - 1]) + ")"));
      return d;
    }
    function cardSample(g, c) {
      const d = ttl("Sample " + (c + 1) + " · " + g.label, "Independent exact sample: no start or end", g.color), a = csec("Position");
      a.appendChild(cline("Piece " + g.piece[c] + " of " + M.facts.pieces + " of the region"));
      const p = csec("Prediction · score field", R_TIP); p.appendChild(cline("p(y = 1) = " + g.pe[c].toFixed(2) + "  (r " + sgnN(g.re[c]) + ")"));
      const r = csec("Records next to the sample (reference)"); r.appendChild(refList(g.ne, g.ned, c));
      const q = csec("Sequence"); q.appendChild(cline("Not decoded: a sample is a map position. Decoding is shown below for points on a line.", "gd-cl m"));
      d.append(a, p, r, q);
      return d;
    }
    function finchesBlock(f, k) {
      const d = csec("Prediction · FINCHES", "FINCHES homotypic interaction parameter ε (Mpipi and CALVADOS); more negative = more attractive = more prone to phase separation. Designed for disordered regions and checked on A1-LCD variants (Spearman +0.79 with C_sat for Mpipi); for tau, SARS-CoV-2 N and TDP-43 read it as a rough indicator.");
      if (!f) { d.appendChild(cline("Not computed.", "gd-cl m")); return d; }
      const row = (lab, o) => { if (!o) return; const t = lab + ": " + o.src.toFixed(2) + (k ? " → " + o.dec.toFixed(2) + " (Δ " + sgnN(o.dec - o.src) + (o.rnd != null ? "; random substitutions at the same sites: Δ " + sgnN(o.rnd - o.src) : "") + ")" : " (the anchor itself)"); d.appendChild(cline(t)); };
      row("Mpipi, full length", f.full && f.full.mpipi); row("CALVADOS, full length", f.full && f.full.calvados);
      if (f.idr) { row("Mpipi, disordered residues (" + f.n_idr + ")", f.idr.mpipi); row("CALVADOS, disordered residues", f.idr.calvados); }
      d.appendChild(cline("ε: more negative = more attractive. The anchor is the sequence the answer was edited from.", "gd-cl m"));
      return d;
    }
    function cardPt(p) {
      const d = ttl("y = " + p.y + " · point P" + p.i, "answer: " + (p.k ? p.k + " substitution" + (p.k === 1 ? "" : "s") + " of the anchor" : "the anchor itself"), p.y ? "y1" : "y0");
      const a = csec("Start · anchor sequence");
      const an = p.start >= 0 ? rec(p.start) : null;
      a.appendChild(cline("Library sequence #" + p.anchor + (an ? " (" + an.name + ")" : "") + ", " + p.len + " aa"));
      if (p.start >= 0) { a.appendChild(el("p", "gd-cl m", "Its closest record to this point:")); const u = el("ul", "gd-refs"); u.appendChild(refItem(p.start, p.start_d)); a.appendChild(u); }
      const b = csec("End · map point and answer");
      b.appendChild(cline(p.k ? "Substitutions (reference residue, position on the anchor, new residue): " + p.edits.slice().sort((x, y) => x[1] - y[1]).map((e) => e[0] + e[1] + e[2]).join(", ") : "No substitution: the best answer is the anchor itself (k* = 0)."));
      b.appendChild(cline("Distance of the ESMC embedding to the target (D): " + p.dsrc.toFixed(4) + (p.k ? " → " + p.dk.toFixed(4) : ""), "gd-cl m"));
      const r = csec("Records next to the map point (reference)"); r.appendChild(refList(p.ne, p.ned, 0));
      const pr = csec("Prediction · score field", R_TIP); pr.appendChild(cline("p(y = 1) = " + p.p.toFixed(2) + "  (r " + sgnN(p.r) + ") at the map point"));
      d.append(a, b, r, pr, finchesBlock(p.fin, p.k));
      return d;
    }
    function setHover(h) { if (pin) return; if (same(h, hov)) return; hov = h; showCard(); draw(); }

    // events
    cv.addEventListener("pointermove", (e) => {
      if (drag) { const w = cv.getBoundingClientRect().width, dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true; if (drag.moved) { view.x = drag.vx - (dx / w) * view.s; view.y = drag.vy - (dy / w) * view.s; clampView(); draw(); } return; }
      if (e.pointerType !== "touch") setHover(hit(e));
    });
    cv.addEventListener("pointerleave", () => { if (!drag) setHover(null); });
    cv.addEventListener("pointerdown", (e) => { cv.setPointerCapture(e.pointerId); drag = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, moved: false }; });
    cv.addEventListener("pointerup", (e) => { const was = drag; drag = null; if (was && !was.moved) { const h = hit(e); if (pin && same(h, pin)) { pin = null; hov = h; } else if (h) { pin = h; hov = h; } else { pin = null; hov = null; } showCard(); draw(); } });
    const clampView = () => { view.x = Math.min(Math.max(view.x, -0.15), 1.15 - view.s); view.y = Math.min(Math.max(view.y, -0.15), 1.15 - view.s); };
    const zoomAt = (fx, fy, k) => { const s2 = Math.min(1.3, Math.max(0.05, view.s / k)); view.x += (fx - view.x) * (1 - s2 / view.s); view.y += (fy - view.y) * (1 - s2 / view.s); view.s = s2; clampView(); draw(); };
    cv.addEventListener("wheel", (e) => { e.preventDefault(); const b = cv.getBoundingClientRect(); zoomAt(view.x + ((e.clientX - b.left) / b.width) * view.s, view.y + ((e.clientY - b.top) / b.height) * view.s, Math.exp(-e.deltaY * 0.0016)); }, { passive: false });
    zin.addEventListener("click", () => zoomAt(view.x + view.s / 2, view.y + view.s / 2, 1.6));
    zout.addEventListener("click", () => zoomAt(view.x + view.s / 2, view.y + view.s / 2, 1 / 1.6));
    zfit.addEventListener("click", () => { Object.assign(view, fitV); draw(); });
    zfull.addEventListener("click", () => { view.x = 0; view.y = 0; view.s = 1; draw(); });
    box.showCard = showCard;
    showCard();
    return box;
  }

  // ---------------- methods
  function steps(items) {
    const ol = el("ol", "gd-steps");
    items.forEach(([h, t]) => { const li = el("li"); li.append(el("b", null, h + " "), t); ol.appendChild(li); });
    return ol;
  }
  function regionTable(t) {
    const tb = el("table", "gd-tbl gd-plain"), th = el("thead"), hr = el("tr");
    ["Sampler"].concat(t.regions).concat(["largest leak"]).forEach((x, i) => hr.appendChild(el("th", i ? "n" : null, x)));
    th.appendChild(hr); tb.appendChild(th);
    const b = el("tbody");
    t.rows.forEach((r) => { const tr = el("tr"); tr.appendChild(el("td", null, r.method)); r.tv.forEach((v) => tr.appendChild(el("td", "n", v[0].toFixed(2) + " / " + v[1].toFixed(2)))); tr.appendChild(el("td", "n", pct(r.leak))); b.appendChild(tr); });
    tb.appendChild(b);
    const w = el("div", "gd-scroll"); w.appendChild(tb);
    return w;
  }
  function methodGroup(id, m) {
    const d = el("div", "gd-group"); d.id = id; d.appendChild(el("h3", null, m.title));
    if (m.intro) d.appendChild(el("p", null, m.intro));
    d.appendChild(steps(m.steps));
    if (m.table) { d.appendChild(el("p", "gd-cap", m.table.caption)); d.appendChild(regionTable(m.table)); }
    if (m.fig) { const f = el("div", "gd-figs one"); f.appendChild(figure(m.fig)); d.appendChild(f); }
    if (m.recommend) d.appendChild(limits(m.recommend_title || "Recommendation", m.recommend));
    return d;
  }

  // ---------------- blocks
  function section(id, title) { const s = el("section", "gd-block"); s.id = id; if (title) s.appendChild(el("h3", null, title)); return s; }
  function findRow(key) {
    for (const g of B.sampling.groups) for (const r of g.rows) if (g.id + "-" + r.id === key) return r;
    for (const r of B.points.rows) if ("pts-" + r.id === key) return r;
    return null;
  }
  function details(id, summary, cls) { const d = el("details", cls || "gd-det"); if (id) d.id = id; d.appendChild(el("summary", null, summary)); return d; }
  function build() {
    $("gdTitle").textContent = T.title; $("gdIntro").textContent = T.intro;
    const Rs = G.results, MM = B.methods, main = new Set(Rs.maps.map((m) => m.row).filter(Boolean));

    // 1. methods
    const sec1 = section("gd-sec-methods", "Methods");
    const f = B.field, fg = el("div", "gd-group"); fg.id = "gd-sec-field"; fg.appendChild(el("h3", null, f.title));
    const ps = el("div", "gd-paras"); f.paras.forEach((p) => ps.appendChild(el("p", null, p))); fg.appendChild(ps);
    fg.appendChild(limits("What it is, and what it is not", f.facts));
    const ff = el("div", "gd-figs"); f.figs.forEach((x) => ff.appendChild(figure(x))); fg.appendChild(ff);
    sec1.appendChild(fg); sec1.appendChild(methodGroup("gd-sec-m-iso", MM.iso)); sec1.appendChild(methodGroup("gd-sec-m-region", MM.region));
    body.appendChild(sec1);

    // 2. main results: four interactive maps
    const sec2 = section("gd-sec-results", "Main results"); sec2.appendChild(el("p", null, Rs.intro));
    Rs.maps.forEach((M) => {
      const box = resultMap(M); sec2.appendChild(box);
      const r = M.row ? findRow(M.row) : null;
      if (r && (M.kind === "iso" || M.kind === "yc")) { const d = details("gd-" + M.row, "Watch the chains move (GIF · " + kb(r.gif_kb) + ")"); const m = media(r); d.appendChild(m); d.media = m; box.appendChild(d); }
      if (r && M.kind === "pts") { const d = details(null, "All 12 points as a table, with the positions of the substitutions and the pairwise differences"); d.appendChild(ptsRow(r)); box.appendChild(d); }
    });
    body.appendChild(sec2);

    // 3. the other examples, collapsed
    const sm = B.sampling, more = details("gd-sec-more", "More examples (GIF rows): " + Rs.more_text, "gd-det gd-more-all");
    sm.groups.forEach((g) => {
      const rows = g.rows.filter((r) => !main.has(g.id + "-" + r.id));
      const d = el("div", "gd-group"); d.id = "gd-sec-" + g.id; d.appendChild(el("h3", null, g.title + " · " + rows.length + " more example" + (rows.length === 1 ? "" : "s")));
      d.appendChild(limits(g.id === "yc" ? "What the examples show, and what they do not" : "Which sampler to use, and what to keep in mind", g.limits));
      rows.forEach((r) => d.appendChild(g.id === "yc" ? ycRow(r) : isoRow(r)));
      if (g.failed && g.failed.length) d.appendChild(failedList(g));
      more.appendChild(d);
    });
    const pt = B.points, pg = el("div", "gd-group"), prow = pt.rows.filter((r) => !main.has("pts-" + r.id)); pg.id = "gd-sec-pts"; pg.appendChild(el("h3", null, pt.title + " · " + prow.length + " more combination" + (prow.length === 1 ? "" : "s")));
    pg.appendChild(el("p", null, pt.intro)); pg.appendChild(limits("What to keep in mind", pt.limits)); prow.forEach((r) => pg.appendChild(ptsRow(r)));
    more.appendChild(pg);
    body.appendChild(more);

    [["Methods", "gd-sec-methods"], ["Main results", "gd-sec-results"], ["More examples", "gd-sec-more"]].forEach(([t, id]) => {
      const b = el("button", null, t); b.type = "button"; b.addEventListener("click", () => { const e = document.getElementById(id); if (e && e.tagName === "DETAILS") e.open = true; goto(id, true); }); nav.appendChild(b);
    });
  }
  function openUp(e) { if (e.tagName === "DETAILS") e.open = true; for (let p = e.parentElement && e.parentElement.closest("details"); p; p = p.parentElement && p.parentElement.closest("details")) p.open = true; }
  function goto(id, smooth) { const e = document.getElementById(id); if (!e) return; openUp(e); e.scrollIntoView(smooth ? { behavior: "smooth", block: "start" } : { block: "start" }); }
  function copyLink(kind, id) {
    const u = location.href.split("#")[0] + "#mode=guidance&ex=" + kind + "-" + id;
    if (navigator.clipboard) navigator.clipboard.writeText(u).catch(() => {});
    history.replaceState(null, "", "#mode=guidance&ex=" + kind + "-" + id);
  }

  let built = false;
  const ensure = () => { if (!built) { build(); built = true; } };
  const redraw = () => maps.forEach((f) => { try { f(); } catch (e) { /* a map that is not laid out yet */ } });
  window.addEventListener("resize", () => { if (!root.hidden) redraw(); });
  new MutationObserver(() => { if (!root.hidden) redraw(); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  return {
    show() { ensure(); setTimeout(redraw, 30); },
    hide() { stopAll(null); },
    applyParams(P) {
      ensure();
      const ex = P.get("ex"), sec = P.get("sec");
      setTimeout(() => {
        redraw();
        if (ex) {
          const row = document.getElementById("gd-" + ex); if (!row) return;
          openUp(row); redraw();
          row.scrollIntoView({ block: "start" }); row.classList.remove("gd-flash"); void row.offsetWidth; row.classList.add("gd-flash");
          if (P.get("play") === "1" && row.media) row.media.start();
        } else if (sec) goto("gd-sec-" + sec);
      }, 60);
    }
  };
}
