/* Guidance tab of the Records space: the score field on the map, guided sampling (one row per GIF example) and the decoding of map points into sequences.
 * Needs: G (guidance data from scripts/rs130_guidance_assets.py), emb ([xs, ys] of the explorer's t-SNE, 0–1 coordinates), $ (getElementById), el (element helper).
 * Builds nothing outside #gdPane. GIFs and figures are files next to the page (guidance/…); a GIF is fetched only when its Play button is pressed. */
function initGuidance(api) {
  "use strict";
  const { G, emb, $, el } = api;
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

  // ---------------- blocks
  function section(id, title) { const s = el("section", "gd-block"); s.id = id; if (title) s.appendChild(el("h3", null, title)); return s; }
  function build() {
    $("gdTitle").textContent = T.title; $("gdIntro").textContent = T.intro;
    const f = B.field, sec1 = section("gd-sec-field", f.title);
    const ps = el("div", "gd-paras"); f.paras.forEach((p) => ps.appendChild(el("p", null, p))); sec1.appendChild(ps);
    sec1.appendChild(limits("What it is, and what it is not", f.facts));
    const fg = el("div", "gd-figs"); f.figs.forEach((x) => fg.appendChild(figure(x))); sec1.appendChild(fg);
    body.appendChild(sec1);

    const sm = B.sampling, sec2 = section("gd-sec-sampling", sm.title);
    sm.groups.forEach((g) => {
      const d = el("div", "gd-group"); d.id = "gd-sec-" + g.id; d.appendChild(el("h3", null, g.title + " · " + g.rows.length + " example" + (g.rows.length === 1 ? "" : "s"))); d.appendChild(el("p", null, g.intro));
      if (g.fig) { const fg2 = el("div", "gd-figs one"); fg2.appendChild(figure(g.fig)); d.appendChild(fg2); }
      d.appendChild(limits(g.id === "yc" ? "What the examples show, and what they do not" : "Which sampler to use, and what to keep in mind", g.limits));
      g.rows.forEach((r) => d.appendChild(g.id === "yc" ? ycRow(r) : isoRow(r)));
      if (g.failed && g.failed.length) d.appendChild(failedList(g));
      sec2.appendChild(d);
    });
    body.appendChild(sec2);

    const pt = B.points, sec3 = section("gd-sec-pts", pt.title);
    sec3.appendChild(el("p", null, pt.intro)); sec3.appendChild(limits("What to keep in mind", pt.limits));
    pt.rows.forEach((r) => sec3.appendChild(ptsRow(r)));
    body.appendChild(sec3);

    [["Score field", "gd-sec-field"], ["(y, c) sets", "gd-sec-yc"], ["Iso-lines", "gd-sec-iso"], ["Map point → sequence", "gd-sec-pts"]].forEach(([t, id]) => {
      const b = el("button", null, t); b.type = "button"; b.addEventListener("click", () => goto(id, true)); nav.appendChild(b);
    });
  }
  function goto(id, smooth) { const e = document.getElementById(id); if (e) e.scrollIntoView(smooth ? { behavior: "smooth", block: "start" } : { block: "start" }); }
  function copyLink(kind, id) {
    const u = location.href.split("#")[0] + "#mode=guidance&ex=" + kind + "-" + id;
    if (navigator.clipboard) navigator.clipboard.writeText(u).catch(() => {});
    history.replaceState(null, "", "#mode=guidance&ex=" + kind + "-" + id);
  }

  let built = false;
  const ensure = () => { if (!built) { build(); built = true; } };
  const redraw = () => maps.forEach((f) => f());
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
          row.scrollIntoView({ block: "start" }); row.classList.remove("gd-flash"); void row.offsetWidth; row.classList.add("gd-flash");
          if (P.get("play") === "1" && row.media) row.media.start();
        } else if (sec) goto("gd-sec-" + sec);
      }, 60);
    }
  };
}
