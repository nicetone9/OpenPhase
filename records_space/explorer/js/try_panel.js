/* "试新实验"面板（简化版）：左边一个短表单，结果显示在右侧卡片里。
 * 条件可以用一句话快速填表（CondParser 解析后把值填进各个框），也可以直接手填；空着 = 未知。
 * 依赖：CondParser、RecordsSpace；主页面通过 ctx 提供数据、元素工具和地图回调。 */
function initTryPanel(ctx) {
  "use strict";
  const { D, I, SP, CP, $, el, showRecord, fmt } = ctx;
  const BUF = [["", "—"], ["Tris", "Tris"], ["HEPES", "HEPES"], ["phosphate", "phosphate / PBS"], ["goodsMES", "MES / MOPS / PIPES"], ["citrate", "citrate"], ["acetate", "acetate"], ["imidazole", "imidazole"], ["other_buf", "other"]];
  const CROWD = [["", "—"], ["none", "none"], ["PEG1k", "PEG 300–1000"], ["PEG6k", "PEG 3000–6000"], ["PEG20k", "PEG 8000–20000"], ["Ficoll", "Ficoll"], ["Dex40", "Dextran ≤ 40 k"], ["Dex70", "Dextran ≥ 70 k"], ["Glycerol", "Glycerol"]];
  const CROWD_KEYS = ["PEG1k", "PEG6k", "PEG20k", "Ficoll", "Dex40", "Dex70", "Glycerol"];
  const NA_KIND = [["none", "none"], ["RNA", "RNA"], ["DNA", "DNA"]];
  const NA_TYPE = [["", "not given"], ["polyU", "polyU"], ["polyA", "polyA"], ["polyC", "polyC"], ["polyG", "polyG"], ["tRNA", "tRNA"], ["total_RNA", "total RNA"], ["mRNA", "mRNA"], ["G4", "G-quadruplex"], ["ssDNA", "ssDNA"], ["dsDNA", "dsDNA"], ["plasmid_or_long_DNA", "plasmid / long DNA"]];
  const ADD_LAB = { reducing: "DTT / TCEP / BME", chelator: "EDTA / EGTA", glycerol: "glycerol", nucleotide: "ATP / NTP", denaturant: "urea / GdnHCl / TMAO", hexanediol: "1,6-hexanediol", detergent: "detergent", protein_add: "BSA / other protein", polyanion: "heparin / polyanion", metal: "metal ion", dye_probe: "dye / probe" };
  const blank = () => ({ temp: "", rt: false, pH: "", conc: "", buf: "", NaCl: "", KCl: "", MgCl2: "", crowd: "", pct: "", na: "none", naType: "", naLen: "", naConc: "" });
  const TOOLS = D.tools ? D.tools.list : [], ENTRIES = D.tools ? D.tools.entries : {};
  const T = { entry: null, condEdited: false, seq: "", text: "", unseen: false, approx: false, dirty: false, seqInfo: null, zn: null, similar: [], exactIdx: -1, result: null, add: new Set(), f: blank() };
  const examples = D.space.exampleDefs, exacts = examples.filter((e) => e.zn_exact);
  const root = $("tryPane"), out = el("div", "tres"), X = {};
  let seqDirty = true, textDirty = false, timer = null;

  // ---------------------------------------------------------------- 界面
  const mkSelect = (id, opts) => { const s = el("select"); s.id = id; for (const [v, t] of opts) { const o = el("option", null, t); o.value = v; s.appendChild(o); } return s; };
  const mkText = (id, ph) => { const i = el("input"); i.type = "text"; i.inputMode = "decimal"; i.id = id; i.placeholder = ph || ""; i.autocomplete = "off"; i.spellcheck = false; return i; };
  const field = (text, child, id) => { const d = el("div", "fld"), l = el("label", null, text); if (id) l.htmlFor = id; d.append(l, child); return d; };
  const sec = (title) => { const s = el("section", "tsec"); if (title) s.appendChild(el("h4", null, title)); root.appendChild(s); return s; };
  root.textContent = "";

  let s = sec("1 · Protein"); const secP = s;
  X.ex = mkSelect("exSel", [["", "Start from an example…"]]); X.ex.setAttribute("aria-label", "Example experiment");
  { const groups = {}; for (const ex of examples) (groups[ex.group] = groups[ex.group] || []).push(ex);
    for (const g of Object.keys(groups)) { const og = document.createElement("optgroup"); og.label = g; for (const ex of groups[g]) { const o = el("option", null, ex.title); o.value = ex.id; og.appendChild(o); } X.ex.appendChild(og); } }
  X.exNote = el("p", "muted");
  X.seq = el("textarea"); X.seq.id = "tSeq"; X.seq.rows = 3; X.seq.spellcheck = false; X.seq.placeholder = "Paste an amino-acid sequence (FASTA is fine)";
  X.seqInfo = el("div", "seqinfo");
  const unLab = el("label", "chk"); X.unseen = el("input"); X.unseen.type = "checkbox"; X.unseen.id = "tUnseen";
  unLab.append(X.unseen, document.createTextNode(" Leave this protein and its relatives out (treat as unseen)"));
  X.tool = mkSelect("tToolSel", [["", "Or load an entry from a tool…"]]); X.tool.setAttribute("aria-label", "Tool");
  { const groups = {}; for (const t of TOOLS) (groups[t.group] = groups[t.group] || []).push(t);
    for (const g of Object.keys(groups)) { const og = document.createElement("optgroup"); og.label = g; for (const t of groups[g]) { const o = el("option", null, t.label + " (" + t.n_ent.toLocaleString("en-US") + (t.kind === "prot" ? " proteins)" : " entries)")); o.value = t.id; og.appendChild(o); } X.tool.appendChild(og); } }
  X.entry = el("input"); X.entry.type = "text"; X.entry.id = "tEntry"; X.entry.autocomplete = "off"; X.entry.spellcheck = false; X.entry.placeholder = "Type to search this tool's entries"; X.entry.setAttribute("list", "tEntryList"); X.entry.hidden = true;
  X.entryList = el("datalist"); X.entryList.id = "tEntryList"; X.entryNote = el("p", "muted");
  X.entryRow = el("div", "entryrow"); X.entryRow.append(X.tool, X.entry, X.entryList, X.entryNote);
  s.append(X.ex, X.exNote, X.entryRow, X.seq, X.seqInfo, unLab);

  s = sec("2 · Conditions"); const secC = s;
  X.text = el("textarea"); X.text.id = "tText"; X.text.rows = 2; X.text.spellcheck = false;
  X.text.placeholder = "Quick fill: describe it in words, e.g. 20 mM Tris pH 7.5, 150 mM NaCl, 25 °C, 20 µM protein, 1 mM DTT";
  X.read = el("div", "readbox");
  s.append(X.text, X.read);
  X.temp = mkText("f_temp", "25"); X.rt = el("button", "chip"); X.rt.type = "button"; X.rt.textContent = "RT"; X.rt.title = "Room temperature only (no number given)";
  const tw = el("div", "inl"); tw.append(X.temp, X.rt);
  X.pH = mkText("f_pH", "7.5"); X.conc = mkText("f_conc", "20"); X.buf = mkSelect("f_buf", BUF);
  const g1 = el("div", "fg2"); g1.append(field("Temperature (°C)", tw, "f_temp"), field("pH", X.pH, "f_pH"), field("Protein (µM)", X.conc, "f_conc"), field("Buffer", X.buf, "f_buf")); s.appendChild(g1);
  X.NaCl = mkText("f_NaCl", "150"); X.KCl = mkText("f_KCl", ""); X.MgCl2 = mkText("f_MgCl2", "");
  const g2 = el("div", "fg3"); g2.append(field("NaCl (mM)", X.NaCl, "f_NaCl"), field("KCl (mM)", X.KCl, "f_KCl"), field("MgCl₂ (mM)", X.MgCl2, "f_MgCl2")); s.appendChild(g2);
  X.crowd = mkSelect("f_crowd", CROWD); X.pct = mkText("f_pct", "10");
  const g3 = el("div", "fg2"); g3.append(field("Crowder", X.crowd, "f_crowd"), field("Crowder (%)", X.pct, "f_pct")); s.appendChild(g3);
  X.na = mkSelect("f_na", NA_KIND); X.naType = mkSelect("f_naType", NA_TYPE); X.naLen = mkText("f_naLen", "20"); X.naConc = mkText("f_naConc", "10");
  X.naTypeF = field("NA type", X.naType, "f_naType");
  const g4 = el("div", "fg2"); g4.append(field("Nucleic acid", X.na, "f_na"), X.naTypeF); s.appendChild(g4);
  X.naMore = el("div", "fg2"); X.naMore.append(field("NA length (nt / bp)", X.naLen, "f_naLen"), field("NA conc. (µM)", X.naConc, "f_naConc")); s.appendChild(X.naMore);
  const det = el("details", "more"); X.addSum = el("summary", null, "Additives (optional)"); det.appendChild(X.addSum);
  const chips = el("div", "chips2"); X.chips = {};
  for (const k of Object.keys(ADD_LAB)) {
    const b = el("button", "chip"); b.type = "button"; b.textContent = ADD_LAB[k];
    b.addEventListener("click", () => { if (T.add.has(k)) T.add.delete(k); else T.add.add(k); T.dirty = true; markEdited(); syncChips(); later(150); });
    chips.appendChild(b); X.chips[k] = b;
  }
  det.appendChild(chips); s.appendChild(det);
  s.appendChild(el("p", "muted", "Leave a box empty if unknown. Once any salt is given, empty salts count as none."));

  s = sec("");
  const adv = el("details", "more"); adv.appendChild(el("summary", null, "Advanced"));
  const apxLab = el("label", "chk"); X.approx = el("input"); X.approx.type = "checkbox"; X.approx.id = "tApprox";
  apxLab.append(X.approx, document.createTextNode(" Use the in-browser approximation even where an exact embedding is built in"));
  X.req = el("button", null, "Copy request for an exact ESMC placement"); X.req.type = "button";
  adv.append(apxLab, X.req); s.appendChild(adv);
  X.clear = el("button", null, "Clear all"); X.clear.type = "button"; s.appendChild(X.clear);

  // ---------------------------------------------------------------- 小工具
  const num = (v) => { const x = parseFloat(String(v).replace(",", ".")); return Number.isFinite(x) ? x : null; };
  const sig = (v) => (v == null || Number.isNaN(v) ? "" : String(Number(Number(v).toPrecision(4))));
  const wilson = (k, n) => { if (!n) return [0, 1]; const z = 1.96, p = k / n, d = 1 + z * z / n, c = p + z * z / (2 * n), a = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return [(c - a) / d, (c + a) / d]; };
  const exactZ = (seq) => { const e = exacts.find((x) => x.sequence === seq); return e ? { zn: Float32Array.from(e.zn_exact) } : null; };
  const seqName = (i) => { const m = SP.seqMeta[i]; return (m.name && m.name !== "unnamed" ? m.name : "unnamed construct") + (m.uniprot ? ", " + m.uniprot : "") + ", " + m.n_records + (m.n_records === 1 ? " record" : " records"); };
  const share = (b) => Math.round(100 * Math.max(b.containQ, b.containK)) + "%";
  const later = (ms) => { clearTimeout(timer); timer = setTimeout(flush, ms || 250); };

  // ---------------------------------------------------------------- 序列
  function sequenceStep() {
    const box = X.seqInfo; box.textContent = "";
    T.zn = null; T.similar = []; T.exactIdx = -1; T.seqInfo = null;
    if (T.entry) {
      const p = SP.rec.pidx[T.entry.rec], tl = TOOLS[T.entry.tool];
      if (p >= 0) {
        const m = SP.seqMeta[p];
        T.zn = SP.zn(p); T.seqInfo = { cls: "exact", length: m.length, mw: null, seq: null, exact: p, similarSeqs: [] }; T.similar = [p]; T.exactIdx = p;
        box.appendChild(el("p", "muted", I.name[T.entry.rec] + (I.uni[T.entry.rec] ? " (" + I.uni[T.entry.rec] + ")" : "") + " · " + m.length + " aa · from a " + tl.label + " entry."));
        box.appendChild(el("p", "muted", "Protein part: the dataset's exact ESMC embedding. The sequence itself is not stored in this page."));
      } else box.appendChild(el("p", "muted", "This entry has no sequence in the dataset: the protein part stays unknown."));
      return;
    }
    if (!T.seq.trim()) { box.appendChild(el("p", "muted", "No sequence: the protein part of the distance stays unknown.")); return; }
    const a = SP.analyzeSequence(T.seq);
    if (a.tooShort) { box.appendChild(el("p", "muted", "Fewer than 10 standard residues: the sequence is not used.")); return; }
    T.seqInfo = a;
    const ex = exactZ(a.seq);
    let status, how, approx = false;
    if (a.cls === "exact") status = "In the dataset: identical sequence (" + seqName(a.exact) + ")";
    else if (a.cls === "variant") status = "Close variant of " + seqName(a.best.s) + " (" + share(a.best) + " of 5-mers shared)";
    else if (a.cls === "related") status = "Related to " + seqName(a.best.s) + " (a few shared 5-mers)";
    else status = "New protein: no similar sequence in the dataset";
    if (a.cls === "exact") { T.zn = SP.zn(a.exact); how = "exact ESMC embedding"; }
    else if (ex && !T.approx) { T.zn = ex.zn; how = "exact ESMC embedding (computed in Colab)"; }
    else if (a.cls === "variant" && !T.approx) { T.zn = SP.zn(a.best.s); how = "ESMC embedding of the closest known construct"; }
    else { T.zn = SP.approxZ(a.seq); how = "approximate embedding (rough)"; approx = true; }
    box.appendChild(el("p", "muted", a.length + " aa · " + (a.mw / 1000).toFixed(1) + " kDa" + (a.nonstandard ? " · " + a.nonstandard + " symbols dropped" : "")));
    box.appendChild(el("p", "muted", status + ". Protein part: " + how + "."));
    if (approx) {
      const A = SP.approx.eval;
      box.appendChild(el("p", "warn", "No ESMC model runs in the browser, so this is a rough stand-in. On held-out homology clusters it recovers " + Math.round(100 * A.overlap_mean) + "% of the 15 nearest sequences (chance " + (100 * A.random_overlap).toFixed(1) + "%), and about " + Math.round(100 * (1 - A.placement_full_lens.overlap_mean)) + "% of the nearest records change once conditions are added."));
    }
    if (ex && T.approx) { let d = 0; for (let t = 0; t < T.zn.length; t++) d += T.zn[t] * ex.zn[t]; box.appendChild(el("p", "muted", "Approximation vs exact for this example: cosine " + d.toFixed(2) + ".")); }
    T.similar = (a.cls === "exact" ? [a.exact] : []).concat(a.similarSeqs || []); T.exactIdx = a.exact;
  }

  // ---------------------------------------------------------------- 条件：表单 <-> 解析结果
  function syncChips() { for (const k in X.chips) X.chips[k].classList.toggle("on", T.add.has(k)); X.addSum.textContent = "Additives (optional)" + (T.add.size ? " — " + T.add.size + " selected" : ""); }
  function writeForm() {
    const f = T.f;
    for (const k of ["temp", "pH", "conc", "NaCl", "KCl", "MgCl2", "pct", "naLen", "naConc"]) X[k].value = f[k];
    for (const k of ["buf", "crowd", "na", "naType"]) X[k].value = f[k];
    X.rt.classList.toggle("on", f.rt); X.temp.disabled = f.rt;
    X.pct.disabled = f.crowd === "" || f.crowd === "none";
    const na = f.na !== "none"; X.naTypeF.hidden = !na; X.naMore.hidden = !na;
    syncChips();
  }
  function showRead(P) {
    X.read.textContent = "";
    if (!P) return;
    if (P.found.length) { const w = el("div", "found"); for (const f of P.found) w.appendChild(el("span", "fchip", f.kind + ": " + f.text)); X.read.appendChild(w); }
    const notes = P.notes.slice();
    if (P.fields.conc.status && P.fields.conc.status.includes("mass_unit_unconverted")) notes.push("Protein concentration in mg/mL needs a sequence to become µM.");
    if (P.unmatched.length) notes.push("Not recognised: " + P.unmatched.join(" · "));
    for (const n of notes) X.read.appendChild(el("p", "muted", n));
  }
  function fillFromText() {
    if (!T.text.trim()) { showRead(null); return; }
    const P = CP.parseExperiment(T.text, { mw: T.seqInfo ? T.seqInfo.mw : null, unmentionedSalts: "unknown", unmentionedCrowders: "zero", expandPBS: true });
    const F = P.fields, f = T.f = blank();
    f.conc = F.conc.state === "obs" ? sig(Math.pow(10, F.conc.value)) : "";
    f.pH = F.pH.state === "obs" ? sig(F.pH.value) : "";
    if (F.temp.state === "rt") f.rt = true; else if (F.temp.state === "obs") f.temp = sig(F.temp.value);
    for (const k of ["NaCl", "KCl", "MgCl2"]) if (F[k].state === "obs" && F[k].value > 0) f[k] = sig(F[k].value);
    if (!f.NaCl && !f.KCl && !f.MgCl2 && ["NaCl", "KCl", "MgCl2"].some((k) => F[k].state === "zero")) f.NaCl = "0";
    const cr = CROWD_KEYS.find((k) => F[k].state === "obs" && F[k].value > 0);
    if (cr) { f.crowd = cr; f.pct = sig(F[cr].value); } else if (CROWD_KEYS.every((k) => F[k].state === "zero")) f.crowd = "none";
    f.buf = P.buffer.families.find((k) => BUF.some((b) => b[0] === k)) || "";
    T.add = new Set(P.additive.families.filter((k) => ADD_LAB[k]));
    const pn = P.na;
    if (pn) { f.na = pn.kind; f.naType = pn.names.find((n) => NA_TYPE.some((t) => t[0] === n)) || ""; f.naLen = sig(pn.length); f.naConc = sig(pn.concUM); }
    T.dirty = false; writeForm(); showRead(P);
  }
  // 表单 -> 距离用的查询（空着 = 缺失；给了任何一种盐，其余盐记 0；拥挤剂选了"none"全部记 0）
  function buildSpec() {
    const f = T.f, F = {}, miss = { value: null, state: "miss" };
    const obs = (x, tf) => (x == null ? miss : { value: tf ? tf(x) : x, state: "obs" });
    F.conc = obs(num(f.conc), (x) => Math.log10(Math.max(x, 1e-4)));
    F.pH = obs(num(f.pH));
    F.temp = f.rt ? { value: 22, state: "rt" } : obs(num(f.temp));
    const salts = ["NaCl", "KCl", "MgCl2"], given = salts.some((k) => num(f[k]) != null);
    for (const k of salts) F[k] = num(f[k]) != null ? obs(num(f[k])) : (given ? { value: 0, state: "zero" } : miss);
    for (const k of CROWD_KEYS) F[k] = f.crowd === "" ? miss : (f.crowd === k ? obs(num(f.pct)) : { value: 0, state: "zero" });
    const buffer = { families: f.buf ? [f.buf] : [], state: f.buf ? "obs" : "miss" }, af = Array.from(T.add);
    const additive = { families: af, state: af.length ? "obs" : (f.buf ? "none" : "miss") };
    let na = null;
    if (f.na !== "none") {
      const len = num(f.naLen), conc = num(f.naConc), cp = F.conc.state === "obs" ? Math.pow(10, F.conc.value) : null;
      na = { kind: f.na, names: f.naType ? [f.naType] : [], length: len, concUM: conc, ratioLog10: conc && cp ? Math.log10(conc / cp) : null, seqFeatures: null };
    }
    return { fields: F, buffer, additive, na, zn: T.zn };
  }
  const isEmpty = () => { const f = T.f; return !T.entry && !T.seq.trim() && !T.text.trim() && !T.add.size && f.na === "none" && !f.rt && ["temp", "pH", "conc", "buf", "NaCl", "KCl", "MgCl2", "crowd", "pct"].every((k) => f[k] === ""); };

  // ---------------------------------------------------------------- 距离与结果
  function compute() {
    const spec = buildSpec(), q = SP.buildQuery(spec), dist = SP.distances(q);
    let exclude = T.unseen ? SP.excludeMask(T.similar.concat(T.exactIdx >= 0 ? [T.exactIdx] : [])) : null;
    if (T.entry) { exclude = exclude || new Uint8Array(SP.N); exclude[T.entry.rec] = 1; }
    const nn = {}; for (const lens of ["full", "prot", "cond", "fullM"]) nn[lens] = SP.nearest(dist[lens], 15, exclude);
    const mean15 = (lens) => nn[lens].reduce((a, j) => a + dist[lens][j], 0) / nn[lens].length;
    const cls = T.seqInfo ? T.seqInfo.cls : "none";
    const novel = T.unseen || cls === "novel" || cls === "related" || cls === "none";
    const m15 = { full: mean15("full"), fullM: mean15("fullM") };
    T.result = { dist, nn, m15, covFull: SP.coverageLevel("full", m15.full, novel), covM: SP.coverageLevel("fullM", m15.fullM, novel), novel, spec, cls };
  }
  const QUANT_TXT = (p) => (p <= 0.25 ? "a well-covered region" : p <= 0.5 ? "typical coverage" : p <= 0.9 ? "less covered than most records" : p <= 0.99 ? "sparsely covered (beyond the 90th percentile)" : "outside what the dataset covers (beyond the 99th percentile)");
  function showEmpty() {
    out.textContent = ""; out.appendChild(el("h2", null, "Nearest records"));
    out.appendChild(el("p", "muted", "Pick an example, or paste a sequence and fill in the conditions. The nearest records, a coverage check and the position on the map appear here."));
  }
  function showResult() {
    const R = T.result; out.textContent = "";
    out.appendChild(el("h2", null, "Nearest records"));
    const verdict = R.covFull.pct <= 0.5 ? "good" : R.covFull.pct <= 0.9 ? "moderate" : "poor";
    const v = el("div", "verdict " + verdict);
    v.append(el("b", null, "Coverage: " + verdict + ". "), document.createTextNode("The 15 nearest records are at mean distance " + R.m15.full.toFixed(3) + ": " + QUANT_TXT(R.covFull.pct) + "."));
    out.appendChild(v);
    if (T.entry) out.appendChild(entryInfo());
    const list = el("div", "nlist");
    R.nn.full.slice(0, 8).forEach((j, r) => {
      const b = el("button", "nbr"); b.type = "button";
      const lab = I.csat[j] ? ["csat", "csat"] : I.lab[j] === "phase separation" ? ["PS", "ps"] : I.lab[j] === "no phase separation" ? ["no PS", "nps"] : [I.lab[j], "oth"];
      const l1 = el("div", "l1"); l1.append(el("b", null, String(r + 1)), el("span", "nm", nm(j)), el("span", "pill " + lab[1], lab[0]));
      const bits = [I.T[j] != null ? I.T[j] + " °C" : (I.tstate[j] === "room temp. only" ? "RT" : null), I.pH[j] != null ? "pH " + I.pH[j] : null, I.NaCl[j] != null ? fmt(I.NaCl[j]) + " mM NaCl" : null, I.paper[j], "d " + R.dist.full[j].toFixed(3)].filter(Boolean);
      b.append(l1, el("div", "l2", bits.join(" · "))); b.addEventListener("click", () => showRecord(j)); list.appendChild(b);
    });
    out.appendChild(list);
    const nn = R.nn.full, pos = nn.filter((j) => I.lab[j] === "phase separation").length, neg = nn.filter((j) => I.lab[j] === "no phase separation").length, oth = nn.length - pos - neg;
    const papers = new Set(nn.map((j) => I.paper[j])), prots = new Set(nn.map((j) => I.name[j]));
    const bar = el("div", "sbar"); [[pos, "ps"], [neg, "nps"], [oth, "oth"]].forEach(([n, c]) => { if (n) { const i = el("i", c); i.style.flex = String(n); bar.appendChild(i); } });
    out.appendChild(bar);
    const [lo, hi] = wilson(pos, pos + neg);
    out.appendChild(el("p", "muted", "Of the 15 nearest: " + pos + " phase-separating, " + neg + " not" + (oth ? ", " + oth + " csat" : "") + (pos + neg ? " (" + Math.round(100 * pos / (pos + neg)) + "% PS, 95% interval " + Math.round(100 * lo) + "–" + Math.round(100 * hi) + "%)" : "") + " · " + papers.size + " paper(s), " + prots.size + " protein(s)."));
    out.appendChild(el("p", "warn", "Not a prediction. Neighbours from the same paper share the label (AUROC 0.84); neighbours from other papers and proteins barely do (0.50 with protein similarity, 0.59 with conditions only)."));
    const at = (c, p) => c.arr[c.q.indexOf(p)].toFixed(3);
    const dt = el("details", "more"); dt.appendChild(el("summary", null, "How to read this"));
    const ul = el("ul"); [
      "The star on the map is the distance-weighted centre of the 8 nearest records; rings mark the 15 nearest. It is not a fitted coordinate: t-SNE and UMAP cannot place a new point exactly.",
      "Coverage compares the mean distance to your 15 nearest records with " + (R.novel ? "records that lose their own protein and homology cluster" : "the dataset's own records") + ": median " + at(R.covFull, 0.5) + ", 90th percentile " + at(R.covFull, 0.9) + ", 99th " + at(R.covFull, 0.99) + ". With missing values ignored your mean distance is " + R.m15.fullM.toFixed(3) + " (median " + at(R.covM, 0.5) + ").",
      "An empty box is unknown and counts as its own state, not as zero."].forEach((t) => ul.appendChild(el("li", null, t)));
    dt.appendChild(ul); out.appendChild(dt);
  }
  function refreshAll() {
    try {
      if (isEmpty()) { showEmpty(); T.result = null; ctx.setTryResult(null); return; }
      compute(); showResult(); ctx.setTryResult(T.result);
    } catch (e) {
      console.error(e); out.textContent = ""; out.appendChild(el("p", "warn", "Something went wrong while reading this input: " + e.message)); T.result = null; ctx.setTryResult(null);
    }
  }
  function flush() {
    clearTimeout(timer);
    try {
      if (seqDirty) { sequenceStep(); seqDirty = false; if (T.text.trim() && !T.dirty) textDirty = true; }
      if (textDirty) { fillFromText(); textDirty = false; }
    } catch (e) { console.error(e); }
    refreshAll();
  }

  // ---------------------------------------------------------------- 工具条目：载入与高亮
  let entryMap = new Map();
  const nm = (j) => (I.name[j] && I.name[j] !== "unnamed" ? I.name[j] : (I.uni[j] || "unnamed"));
  const tagOf = (j, tl) => {
    if (tl.kind === "prot") return I.uni[j] || "";
    const p = I.rid[j].split(":").slice(1).join(":");
    return /^\d+$/.test(p) ? "row " + p + " · " + I.paper[j] : (p.length > 26 ? p.slice(0, 25) + "…" : p);
  };
  const labShort = (j) => (I.csat[j] ? "csat" : I.lab[j] === "phase separation" ? "PS" : I.lab[j] === "no phase separation" ? "no PS" : "");
  const condBits = (j) => [I.T[j] != null ? I.T[j] + " °C" : null, I.pH[j] != null ? "pH " + I.pH[j] : null].filter(Boolean).join(" ");
  const entryText = (j, tl) => nm(j) + " · " + tagOf(j, tl) + (tl.kind === "prot" ? "" : " · " + labShort(j) + (condBits(j) ? " · " + condBits(j) : ""));
  const recLabel = (j) => { if (I.csat[j]) { const y = D.cont.y_csat.vals[j]; return "csat" + (y != null ? " " + fmt(Math.pow(10, y)) + " µM" : "") + " (" + I.csat[j] + ")"; } return I.lab[j]; };
  function fillEntryList(tid) {
    X.entryList.textContent = ""; entryMap = new Map(); X.entry.value = ""; X.entryNote.textContent = "";
    const tl = TOOLS.find((t) => t.id === tid); X.entry.hidden = !tl;
    if (!tl) return;
    const frag = document.createDocumentFragment();
    for (const j of ENTRIES[tid]) { let txt = entryText(j, tl); if (entryMap.has(txt)) txt += " #" + j; entryMap.set(txt, j); const o = document.createElement("option"); o.value = txt; frag.appendChild(o); }
    X.entryList.appendChild(frag);
    X.entryNote.textContent = tl.kind === "prot" ? "Proteins only: choosing one fills the protein and leaves the conditions empty." : "Choosing an entry fills the protein and its conditions.";
  }
  function setTool(id) { X.tool.value = id || ""; fillEntryList(id || ""); }
  // 记录 -> 表单：把 records.bin 里缩放过的数值还原成毫摩尔、百分比等（数据集序列不放进页面，蛋白用它的嵌入）
  function formFromRecord(j) {
    const R = SP.rec, P = SP.params, Sc = P.scales, f = blank(), fin = Number.isFinite;
    if (fin(R.conc[j])) f.conc = sig(Math.pow(10, R.conc[j] * Sc.conc));
    if (fin(R.pH[j])) f.pH = sig(R.pH[j] * Sc.pH);
    if (R.temp_code[j] === 1) f.rt = true; else if (fin(R.temp[j])) f.temp = sig(R.temp[j] * Sc.temp);
    const nS = P.salt.length, nC = P.crowd.length, sv = P.salt.map((k, t) => R.salt[j * nS + t]);
    P.salt.forEach((k, t) => { if (fin(sv[t])) { const x = Math.pow(10, sv[t] * Sc[k]) - 1; if (x > 1e-6) f[k] = sig(x); } });
    if (!f.NaCl && !f.KCl && !f.MgCl2 && sv.some(fin)) f.NaCl = "0";
    const cv = P.crowd.map((k, t) => R.crowd[j * nC + t]);
    if (cv.some(fin)) {
      let hit = -1; cv.forEach((v, t) => { if (hit < 0 && fin(v) && Math.pow(10, v * Sc[P.crowd[t]]) - 1 > 1e-6) hit = t; });
      if (hit >= 0) { f.crowd = P.crowd[hit]; f.pct = sig(Math.pow(10, cv[hit] * Sc[P.crowd[hit]]) - 1); } else f.crowd = "none";
    }
    const bits = (mask, names) => names.filter((_, i) => (mask >> i) & 1);
    if (R.buf_obs[j] === 1) f.buf = bits(R.buf[j], P.buf).find((k) => BUF.some((b) => b[0] === k)) || "";
    if (R.na_cls[j] > 0) {
      f.na = R.na_cls[j] === 2 ? "DNA" : "RNA"; f.naType = bits(R.na_flags[j], P.na_flags).find((k) => NA_TYPE.some((t) => t[0] === k)) || "";
      if (fin(R.na_len[j])) f.naLen = sig(Math.pow(10, R.na_len[j] * Sc.na_len));
      if (fin(R.na_conc[j])) f.naConc = sig(Math.pow(10, R.na_conc[j] * Sc.na_conc));
    }
    return f;
  }
  const addFromRecord = (j) => new Set(SP.rec.add_obs[j] === 1 ? SP.params.add.filter((_, i) => (SP.rec.add[j] >> i) & 1).filter((k) => ADD_LAB[k]) : []);
  function highlightSections() {
    for (const sc of [secP, secC]) { sc.classList.remove("hl", "light", "nocond"); sc.style.removeProperty("--hl"); const old = sc.querySelector(".sidenote"); if (old) old.remove(); }
    if (!T.entry) return;
    const tl = TOOLS[T.entry.tool], note = (sc, txt) => sc.querySelector("h4").appendChild(el("span", "sidenote", txt));
    secP.classList.add("hl"); secP.style.setProperty("--hl", tl.color);
    if (tl.kind === "prot") { secP.classList.add("light"); note(secP, "from " + tl.label); secC.classList.add("nocond"); note(secC, "no conditions in this tool's entries"); }
    else { note(secP, "from a " + tl.label + " entry"); secC.classList.add("hl"); secC.style.setProperty("--hl", tl.color); note(secC, "from the same entry"); }
  }
  function markEdited() {
    if (!T.entry || T.condEdited) return;
    T.condEdited = true;
    if (TOOLS[T.entry.tool].kind !== "prot") { secC.classList.remove("hl"); const n = secC.querySelector(".sidenote"); if (n) n.textContent = "edited after loading the entry"; }
  }
  function entryInfo() {
    const j = T.entry.rec, tl = TOOLS[T.entry.tool], d = el("div", "entryinfo");
    d.style.setProperty("--hl", tl.color);
    if (tl.kind === "prot") d.append(el("b", null, tl.label + ": "), document.createTextNode("a protein-level entry with no conditions. The neighbours are found from the protein alone; every condition counts as unknown."));
    else d.append(el("b", null, "Entry from " + tl.label + ": "), document.createTextNode("dataset label " + recLabel(j) + " · " + I.paper[j] + ". This record itself is left out of the neighbours."));
    return d;
  }
  function loadEntry(j, ti) {
    const tl = TOOLS[ti]; if (!tl) return;
    T.entry = { rec: j, tool: ti }; T.condEdited = false;
    X.ex.value = ""; X.exNote.textContent = ""; T.seq = ""; X.seq.value = ""; T.text = ""; X.text.value = ""; showRead(null);
    T.f = tl.kind === "prot" ? blank() : formFromRecord(j); T.add = tl.kind === "prot" ? new Set() : addFromRecord(j);
    T.dirty = false; seqDirty = true; textDirty = false;
    writeForm(); highlightSections(); flush();
  }
  function dropEntry() { if (!T.entry) return; T.entry = null; X.entry.value = ""; highlightSections(); }

  // ---------------------------------------------------------------- 事件
  X.tool.addEventListener("change", () => { ctx.setHighlight(X.tool.value, true); fillEntryList(X.tool.value); });
  X.entry.addEventListener("input", () => { const j = entryMap.get(X.entry.value); if (j != null) loadEntry(j, TOOLS.findIndex((t) => t.id === X.tool.value)); });
  X.seq.addEventListener("input", () => { dropEntry(); T.seq = X.seq.value; seqDirty = true; later(350); });
  X.text.addEventListener("input", () => { T.text = X.text.value; textDirty = true; later(450); });
  X.unseen.addEventListener("change", () => { T.unseen = X.unseen.checked; flush(); });
  X.approx.addEventListener("change", () => { T.approx = X.approx.checked; seqDirty = true; flush(); });
  for (const k of ["temp", "pH", "conc", "NaCl", "KCl", "MgCl2", "pct", "naLen", "naConc"]) X[k].addEventListener("input", () => { T.f[k] = X[k].value; T.dirty = true; markEdited(); later(250); });
  for (const k of ["buf", "crowd", "na", "naType"]) X[k].addEventListener("change", () => { T.f[k] = X[k].value; T.dirty = true; markEdited(); writeForm(); flush(); });
  X.rt.addEventListener("click", () => { T.f.rt = !T.f.rt; if (T.f.rt) { T.f.temp = ""; X.temp.value = ""; } T.dirty = true; markEdited(); writeForm(); flush(); });
  function loadExample(id) {
    const ex = examples.find((e) => e.id === id); if (!ex) return;
    X.ex.value = id; X.exNote.textContent = ex.note || "";
    T.seq = ex.sequence || ""; X.seq.value = T.seq; T.text = ex.text || ""; X.text.value = T.text;
    T.unseen = !!ex.unseen; X.unseen.checked = T.unseen;
    dropEntry(); T.f = blank(); T.add = new Set(); T.dirty = false; seqDirty = true; textDirty = !!T.text.trim();
    writeForm(); showRead(null); flush();
  }
  X.ex.addEventListener("change", () => { if (X.ex.value) loadExample(X.ex.value); });
  X.clear.addEventListener("click", () => {
    X.ex.value = ""; X.exNote.textContent = ""; T.seq = ""; X.seq.value = ""; T.text = ""; X.text.value = ""; T.unseen = false; X.unseen.checked = false;
    dropEntry(); T.f = blank(); T.add = new Set(); T.dirty = false; seqDirty = true; textDirty = false; writeForm(); showRead(null); flush();
  });
  X.req.addEventListener("click", () => {
    const f = Object.fromEntries(Object.entries(T.f).filter(([, v]) => v !== "" && v !== false && v !== "none"));
    const req = "Please place this experiment in the records space with an exact ESMC-600M embedding.\nSequence:\n" + (T.seqInfo ? T.seqInfo.seq : "(none)") + "\nConditions: " + JSON.stringify(f) + (T.add.size ? "\nAdditives: " + Array.from(T.add).join(", ") : "") + (T.text ? "\nOriginal text: " + T.text : "");
    const done = () => { X.req.textContent = "Copied. Paste it to Claude."; setTimeout(() => { X.req.textContent = "Copy request for an exact ESMC placement"; }, 2500); };
    const fallback = (t) => { const ta = document.createElement("textarea"); ta.value = t; ta.style.position = "fixed"; ta.style.opacity = "0"; document.body.appendChild(ta); ta.select(); try { document.execCommand("copy"); done(); } catch (e2) { X.req.textContent = "Copy failed"; } document.body.removeChild(ta); };
    try { navigator.clipboard.writeText(req).then(done, () => fallback(req)); } catch (e) { fallback(req); }
  });
  writeForm(); fillEntryList(""); flush();
  return { load: loadExample, state: T, refresh: flush, resultEl: out, setTool, loadEntry };
}
