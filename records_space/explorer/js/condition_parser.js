/* 实验条件解析器（浏览器 / node 通用，不联网）。
 *
 * 第一层：把数据集构造时用的 Python 解析引擎（condition_phase/scripts/world_model/prepare_llpsdb_p0.py）逐函数移植过来，
 *         数值约定、单位表、PEG 分档、状态码都与数据集一致（norm / parse_amount / concentration_um / parse_salts / parse_ph /
 *         parse_temperature / parse_crowders / parse_nucleic / parse_solutes）。
 * 第二层：自由文本 → 各"栏"的路由（数据集的输入是表格的分栏，用户给的是一段话）：找出温度、pH、盐、拥挤剂、蛋白浓度、核酸、缓冲液与添加剂，
 *         把每一项还原成"数值 + 单位"的小片段，再交给第一层的函数。
 * 输出的状态沿用数据集：obs 有数值 / zero 明确为无 / rt 只写室温（记 22 °C）/ miss 缺失。
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.CondParser = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ------------------------------------------------------------------ 常量（与 Python 一致）
  const FIELDS = ["Conc", "pH", "temp", "MgCl2", "NaCl", "KCl", "PEG300-1k", "PEG3k-6k", "PEG8k-20k", "Ficoll", "Dextran -40", "Dextran 70-", "Glycerol"];
  const SHORT = ["conc", "pH", "temp", "MgCl2", "NaCl", "KCl", "PEG1k", "PEG6k", "PEG20k", "Ficoll", "Dex40", "Dex70", "Glycerol"];
  const AA = "ACDEFGHIKLMNPQRSTVWY";
  const RESIDUE_MASS = { A: 71.0788, R: 156.1875, N: 114.1038, D: 115.0886, C: 103.1388, E: 129.1155, Q: 128.1307, G: 57.0519, H: 137.1411, I: 113.1594, L: 113.1594, K: 128.1741,
    M: 131.1926, F: 147.1766, P: 97.1167, S: 87.0782, T: 101.1051, W: 186.2132, Y: 163.1760, V: 99.1326 };
  const MOLAR_TO_UM = { m: 1e6, mm: 1e3, um: 1.0, nm: 1e-3, pm: 1e-6 };
  const MOLAR_TO_MM = {}; for (const k in MOLAR_TO_UM) MOLAR_TO_MM[k] = MOLAR_TO_UM[k] / 1e3;
  const MASS_TO_MG_ML = { "g/l": 1.0, "mg/ml": 1.0, "ug/ul": 1.0, "ug/ml": 1e-3, "ng/ul": 1e-3, "mg/l": 1e-3, "ng/ml": 1e-6 };
  const SALT_ALIASES = { nacl: "NaCl", kcl: "KCl", mgcl2: "MgCl2", licl: "LiCl", nh4cl: "NH4Cl", kglu: "KGlu", kglutamate: "KGlu", "k-glutamate": "KGlu", potassiumglutamate: "KGlu",
    naglu: "NaGlu", sodiumglutamate: "NaGlu", koac: "KOAc", potassiumacetate: "KOAc", naoac: "NaOAc", sodiumacetate: "NaOAc", mgoac: "MgOAc", "mg(oac)2": "MgOAc",
    kscn: "KSCN", nascn: "NaSCN", nabr: "NaBr", kbr: "KBr", nai: "NaI", cacl2: "CaCl2", zncl2: "ZnCl2", mncl2: "MnCl2", nicl2: "NiCl2",
    "(nh4)2so4": "(NH4)2SO4", na2so4: "Na2SO4", k2so4: "K2SO4", mgso4: "MgSO4" };
  const NUM = "(?:\\d+(?:\\.\\d+)?|\\.\\d+)";
  const CONC_UNIT = "(mg/ml|ug/ul|ug/ml|ng/ul|ng/ml|mg/l|g/l|pm|nm|um|mm|m|%)";
  const REPL = [["µ", "u"], ["μ", "u"], ["℃", "C"], ["°C", "C"], ["˚C", "C"], ["° C", "C"], ["＞", ">"], ["＜", "<"], ["–", "-"], ["—", "-"], ["－", "-"],
    ["\u00a0", " "], ["\u2009", " "], ["\u202f", " "], ["≤", "<="], ["≥", ">="], ["％", "%"], ["；", ";"], ["，", ","]];
  const NAME_CLASSES = [["polyU", /poly\s*-?\(?\s*r?u\)?(?![a-z])|polyu/i], ["polyA", /poly\s*-?\(?\s*r?a\)?(?![a-z])|polya/i], ["polyC", /polyc|poly\s*-?\(?\s*r?c\)?(?![a-z])/i], ["polyG", /polyg|poly\s*-?\(?\s*r?g\)?(?![a-z])/i], ["tRNA", /trna/i], ["rRNA", /rrna/i],
    ["total_RNA", /total rna/i], ["mRNA", /mrna/i], ["G4", /quadruplex|g4|telomer/i], ["ssDNA", /ssdna|single.?strand/i], ["dsDNA", /dsdna|double.?strand|bp\b/i],
    ["plasmid_or_long_DNA", /plasmid|lambda|601/i], ["methylated", /methyl|5me/i]];
  // 缓冲液 / 添加剂词表（与 rs01_build_inputs.py 同一套）
  const BUF = [["Tris", /\btris\b|tris-|trizma|tris\(|\btris(?=[\s,(])/i], ["HEPES", /hepes/i], ["phosphate", /phosphate|\bpbs\b|\bdpbs\b|\bnapi\b|na2hpo4|nah2po4|kh2po4|k2hpo4|\bpi\b/i],
    ["goodsMES", /\bmes\b|\bmops\b|pipes|\bmes-/i], ["citrate", /citrate/i], ["acetate", /acetate|\bnaac\b|\boac\b/i], ["imidazole", /imidazole/i],
    ["other_buf", /glycine|borate|\bcaps\b|\bches\b|bicine|bis-?tris|cacodylate|ammonium bicarbonate/i]];
  const ADD = [["reducing", /\bdtt\b|tcep|\bbme\b|mercaptoethanol|2-me\b|beta-?me\b/i], ["chelator", /edta|egta/i], ["glycerol", /glycerol/i],
    ["nucleotide", /\batp\b|\badp\b|\bgtp\b|\bamp\b|camp|nadh|\bfad\b|accoa|acetyl-?coa/i], ["denaturant", /urea|guanidin|gdm|gdn|tmao|trimethylamine|betaine|sorbitol|trehalose|sucrose/i],
    ["hexanediol", /hexane?diol|hexanodiol/i], ["detergent", /triton|np-?40|tween|\bsds\b|chaps|brij|igepal/i], ["protein_add", /\bbsa\b|albumin|lysozyme|\btev\b/i],
    ["polyanion", /heparin|\bpar\b|poly\(|polyp\b|polyu\b|polya\b|dextran sulfate/i], ["metal", /zncl|znso|cacl|mncl|mgso|cucl|fecl|zinc|calcium/i],
    ["dye_probe", /thioflavin|hoechst|dapi|fluorescein|alexa|\bcy3\b|\bcy5\b|nile red/i]];

  // ------------------------------------------------------------------ 第一层：Python 引擎的移植
  function norm(text) {
    let s = text == null ? "" : String(text);
    for (const [a, b] of REPL) s = s.split(a).join(b);
    return s.replace(/\s+/g, " ").trim();
  }
  function isBlank(text) {
    return ["", "-", "nan", "none", "na", "n/a", "unknow", "unknown", "-;-"].includes(norm(text).toLowerCase());
  }
  function emptyValue(raw, unit) {
    return { raw: raw == null ? null : String(raw), unit, value: null, lower: null, upper: null, values: null, status: "missing" };
  }
  function parseAmount(text, unitPattern) {
    const s = norm(text).toLowerCase();
    const u = unitPattern || CONC_UNIT;
    let m = new RegExp("^(<=|>=|<|>)\\s*(" + NUM + ")\\s*" + u + "(?![a-z])(.*)$").exec(s);
    if (m) {
      const v = parseFloat(m[2]);
      const [lo, hi] = m[1].startsWith(">") ? [v, null] : [null, v];
      return { kind: "open_bound", lower: lo, upper: hi, unit: m[3], rest: m[4] };
    }
    m = new RegExp("^(" + NUM + ")\\s*" + u + "?\\s*(?:-|to)\\s*(" + NUM + ")\\s*" + u + "(?![a-z])(.*)$").exec(s);
    if (m && (m[2] === undefined || m[2] === m[4])) {
      const lo = parseFloat(m[1]), hi = parseFloat(m[3]);
      if (lo <= hi) return { kind: "interval", lower: lo, upper: hi, unit: m[4], rest: m[5] };
      return null;
    }
    m = new RegExp("^((?:" + NUM + "\\s*(?:,|or|and|/)\\s*)+" + NUM + ")\\s*" + u + "(?![a-z])(.*)$").exec(s);
    if (m) {
      const vals = Array.from(new Set((m[1].match(new RegExp(NUM, "g")) || []).map(parseFloat))).sort((a, b) => a - b);
      return { kind: "discrete_set", values: vals, unit: m[2], rest: m[3] };
    }
    m = new RegExp("^(" + NUM + ")\\s*(?:±\\s*" + NUM + "\\s*)?" + u + "(?![a-z])(.*)$").exec(s);
    if (m) return { kind: "numeric", value: parseFloat(m[1]), unit: m[2], rest: m[3] };
    return null;
  }
  const STATUS_OF = { numeric: "source_numeric", interval: "interval", discrete_set: "discrete_set", open_bound: "ambiguous_open_bound" };
  function amountToField(amount, factor, raw, unit) {
    const sc = (x) => (x == null ? null : x * factor);
    return { raw, unit, value: sc(amount.value), lower: sc(amount.lower), upper: sc(amount.upper), values: amount.values == null ? null : amount.values.map((v) => v * factor), status: STATUS_OF[amount.kind] };
  }
  function molWeight(seq) {
    if (!seq) return null;
    let m = 18.015;
    for (const c of seq) { if (!(c in RESIDUE_MASS)) return null; m += RESIDUE_MASS[c]; }
    return m;
  }
  function concentrationUM(amount, raw, mw) {
    const unit = amount.unit;
    if (unit in MOLAR_TO_UM) return amountToField(amount, MOLAR_TO_UM[unit], raw, "uM");
    const mass = unit in MASS_TO_MG_ML ? MASS_TO_MG_ML[unit] : (unit === "%" ? 10.0 : null);
    if (mass != null && mw) {
      const out = amountToField(amount, (mass / mw) * 1e6, raw, "uM");
      out.status += "+mass_converted_by_sequence_mw";
      return out;
    }
    const out = amountToField(amount, mass || 1.0, raw, mass ? "mg/ml" : unit);
    out.status += "+mass_unit_unconverted";
    return out;
  }
  function splitPieces(text) {
    return norm(text).replace(/\],/g, "]\u0001").split(/[;|\u0001]/).map((p) => p.trim()).filter((p) => p && !isBlank(p));
  }
  function parseSolutes(text, mws, nucleicCount) {
    const proteinConc = [], other = [], issues = [];
    for (const piece of splitPieces(text)) {
      const amount = parseAmount(piece);
      const bracket = /\[([^\]]+)\]/.exec(piece);
      if (bracket && amount == null) { issues.push({ field: "solute", piece, reason: "protein_amount_unparsed" }); proteinConc.push(null); continue; }
      if (amount == null) { issues.push({ field: "solute", piece, reason: "unparsed" }); continue; }
      if (bracket) proteinConc.push([piece, amount, bracket[1]]);
      else other.push([piece, amount, amount.rest.replace(/^[ ,]+|[ ,]+$/g, "")]);
    }
    if (!proteinConc.length && mws.length && other.length) {
      for (const [p, a] of other.slice(0, mws.length)) proteinConc.push([p, a, null]);
      other.splice(0, mws.length);
      issues.push({ field: "solute", reason: "protein_conc_unbracketed_assigned_by_order" });
    }
    const concs = mws.map((mw, i) => {
      const item = proteinConc[i];
      if (!item) { const v = emptyValue(isBlank(text) ? null : String(text), "uM"); v.status = isBlank(text) ? "missing" : "unparsed"; return v; }
      return concentrationUM(item[1], item[0], mw);
    });
    const nucleic = [], extra = [];
    other.forEach(([piece, amount, name], j) => {
      const f = Object.assign(concentrationUM(amount, piece, null), { name });
      (j < nucleicCount ? nucleic : extra).push(f);
    });
    return { concs, nucleic, extra, issues };
  }
  function canonicalSalt(name) {
    let key = name.toLowerCase().replace(/\s+/g, "");
    if (!key.startsWith("(")) key = key.replace(/\(.*?\)$/, "");
    key = key.replace(/₂/g, "2").replace(/₄/g, "4");
    return SALT_ALIASES[key] || null;
  }
  function parseSalts(text) {
    const fields = { NaCl: null, KCl: null, MgCl2: null }, others = [], issues = [];
    for (let piece of norm(text).split(/[;,|]/)) {
      piece = piece.replace(/\(.*?\)/g, "").trim();
      if (!piece || isBlank(piece)) continue;
      let amount;
      const m = new RegExp("^(" + NUM + ")\\s*(m|mm|um|nm)\\s*(?:or)\\s*(" + NUM + ")\\s*(m|mm|um|nm)\\s+(.*)$").exec(norm(piece).toLowerCase());
      if (m && m[2] === m[4]) amount = { kind: "discrete_set", values: Array.from(new Set([parseFloat(m[1]), parseFloat(m[3])])).sort((a, b) => a - b), unit: m[2], rest: m[5] };
      else amount = parseAmount(piece, "(m|mm|um|nm)");
      if (amount == null) { issues.push({ field: "salt", piece, reason: "unparsed" }); continue; }
      const name = amount.rest.trim();
      const salt = canonicalSalt(name);
      const value = amountToField(amount, MOLAR_TO_MM[amount.unit], piece, "mM");
      if (salt in fields) {
        if (fields[salt] != null) issues.push({ field: salt, piece, reason: "salt_listed_twice" });
        fields[salt] = value;
      } else others.push(Object.assign(value, { name: salt || name, recognized: salt != null }));
    }
    return { fields, others, issues };
  }
  function parsePh(bufferText) {
    const raw = isBlank(bufferText) ? null : String(bufferText);
    const out = emptyValue(raw, "pH");
    if (raw == null) return out;
    const s = norm(raw);
    const hits = []; const rx = new RegExp("p\\s*h\\s*[:=]?\\s*(" + NUM + ")(?:\\s*-\\s*(" + NUM + "))?", "gi");
    let mm; while ((mm = rx.exec(s))) hits.push([parseFloat(mm[1]), mm[2] ? parseFloat(mm[2]) : null]);
    const vals = hits.filter(([a, b]) => a > 0 && a <= 14 && (b == null || (b > 0 && b <= 14)));
    if (!vals.length) {
      const m = new RegExp("(tris(?:-hcl)?|hepes(?:-koh|-naoh)?|pipes(?:/koh)?|mops|mes|bis-tris|citrate|phosphate|acetate)\\s*,?\\s*(" + NUM + ")(?![\\d.])(?!\\s*(?:mm|m\\b|%|um|nm))", "i").exec(s);
      if (m && parseFloat(m[2]) >= 3 && parseFloat(m[2]) <= 11) { out.value = parseFloat(m[2]); out.status = "inferred_from_buffer_number"; }
      else if (/\bpbs\b/i.test(s)) out.status = "missing_pbs_unspecified";
      return out;
    }
    if (vals.length === 1 && vals[0][1] == null) { out.value = vals[0][0]; out.status = "source_numeric"; }
    else if (vals.length === 1) { out.lower = Math.min(vals[0][0], vals[0][1]); out.upper = Math.max(vals[0][0], vals[0][1]); out.status = "interval"; }
    else {
      const flat = Array.from(new Set(vals.flatMap((p) => p.filter((x) => x != null)))).sort((a, b) => a - b);
      out.lower = flat[0]; out.upper = flat[flat.length - 1]; out.values = flat;
      out.status = vals.every(([, b]) => b == null) ? "discrete_set" : "interval";
    }
    return out;
  }
  function parseTemperature(text) {
    const raw = isBlank(text) ? null : String(text);
    const out = emptyValue(raw, "degC");
    if (raw == null) return out;
    const s = norm(raw);
    const sCore = s.replace(/\(.*?\)/g, "").trim();
    if (/^(?:rt|r\.t\.|room temperature|room temp)$/i.test(sCore)) { out.status = "room_temperature_unspecified"; return out; }
    const kelvin = /\d\s*K\b/.test(sCore);
    const shift = kelvin ? -273.15 : 0.0;
    const core = sCore.replace(/\s*[CK]\b/g, "");
    let m;
    if ((m = new RegExp("^(<=|>=|<|>)\\s*(" + NUM + ")$").exec(core))) {
      const v = parseFloat(m[2]) + shift;
      out.lower = m[1].startsWith(">") ? v : null; out.upper = m[1].startsWith("<") ? v : null; out.status = "ambiguous_open_bound";
    } else if ((m = new RegExp("^(-?" + NUM + ")\\s*-{1,2}\\s*(-?" + NUM + ")$").exec(core)) && parseFloat(m[1]) <= parseFloat(m[2])) {
      out.lower = parseFloat(m[1]) + shift; out.upper = parseFloat(m[2]) + shift; out.status = "interval";
    } else if ((m = new RegExp("^(" + NUM + ")\\s*(?:±\\s*" + NUM + ")?$").exec(core))) {
      out.value = parseFloat(m[1]) + shift; out.status = "source_numeric";
    } else if (new RegExp("^(" + NUM + ")(?:\\s*(?:,|and|/)\\s*" + NUM + ")+$").test(core)) {
      const vals = Array.from(new Set((core.match(new RegExp(NUM, "g")) || []).map((x) => parseFloat(x) + shift))).sort((a, b) => a - b);
      if (vals.length === 1) { out.value = vals[0]; out.status = "source_numeric"; }
      else { out.values = vals; out.lower = vals[0]; out.upper = vals[vals.length - 1]; out.status = "discrete_set"; }
    } else out.status = "unparsed";
    return out;
  }
  function pegBin(mw) {
    if (mw == null) return null;
    if (mw <= 1000) return "PEG300-1k";
    if (mw <= 6000) return "PEG3k-6k";
    if (mw <= 20000) return "PEG8k-20k";
    return null;
  }
  function parseCrowders(textsWithSource) {
    const fields = {}; for (const k of FIELDS.slice(6)) fields[k] = null;
    const others = [], issues = [];
    for (const [text, source] of textsWithSource) {
      for (let piece of norm(text).split(/[;|]|,(?![^()]*\))/)) {
        piece = piece.trim();
        if (!piece || isBlank(piece)) continue;
        const low = piece.toLowerCase();
        let agent = null, m;
        if (/peg|polyethylene glycol/.test(low)) {
          m = new RegExp("(?:peg|polyethylene glycol)\\s*-?\\s*(" + NUM + ")\\s*(k|kda)?").exec(low);
          let mw = null;
          if (m) mw = parseFloat(m[1]) * (m[2] || parseFloat(m[1]) < 100 ? 1000 : 1);
          agent = pegBin(mw) || (mw == null ? "PEG_mw_unspecified" : "PEG_" + Math.trunc(mw));
        } else if (low.includes("ficoll")) agent = "Ficoll";
        else if (low.includes("dextran")) {
          m = new RegExp("dextran\\s*(?:t-?\\s*)?(" + NUM + ")\\s*(k|kda)?").exec(low);
          const mw = !m ? null : parseFloat(m[1]) * (m[2] || parseFloat(m[1]) < 1000 ? 1000 : 1);
          agent = mw == null ? null : (mw <= 40000 ? "Dextran -40" : mw >= 70000 ? "Dextran 70-" : null);
          agent = agent || "Dextran_mw_unspecified";
        } else if (low.includes("glycerol")) agent = "Glycerol";
        if (agent == null) { if (source === "Crowding agent") others.push({ raw: piece, source_column: source, name: "other_crowder" }); continue; }
        let amount = parseAmount(low.replace(/\((?:w\/v|v\/v|w\/w)\)/g, ""), "(%|mg/ml|g/l|mm|um)");
        if (amount == null && (m = new RegExp("(" + NUM + ")\\s*(%|mg/ml|g/l)").exec(low))) amount = { kind: "numeric", value: parseFloat(m[1]), unit: m[2], rest: "" };
        let value;
        if (amount == null) { value = emptyValue(piece, "percent_basis_unspecified"); value.status = "present_amount_unparsed"; }
        else if (["%", "mg/ml", "g/l"].includes(amount.unit)) {
          value = amountToField(amount, amount.unit === "%" ? 1.0 : 0.1, piece, "percent_basis_unspecified");
          if (amount.unit !== "%") value.status += "+mass_per_volume_to_percent";
        } else { value = amountToField(amount, 1.0, piece, amount.unit); value.status += "+unit_not_percent"; }
        if (agent in fields) {
          if (fields[agent] != null) { issues.push({ field: agent, piece, reason: "listed_twice_kept_first" }); continue; }
          fields[agent] = value;
        } else others.push(Object.assign(value, { name: agent }));
      }
    }
    return { fields, others, issues };
  }
  function parseNucleicPiece(piece) {
    if (!piece) return null;
    const p = norm(piece);
    let m = /([ACGTU]{8,})/.exec(p.replace(/ /g, ""));
    const seq = m ? m[1] : null;
    m = new RegExp("(" + NUM + ")\\s*-?\\s*(nt|bp|mer)\\b", "i").exec(p);
    const length = m ? parseFloat(m[1]) : null;
    m = /poly\s*-?\(?r?\s*([ACGUT])\)?(?![a-z])/i.exec(p);
    const homopolymer = m ? m[1].toUpperCase() : null;
    return { raw: p, sequence: seq, length, homopolymer, identity_key: seq || (homopolymer ? "poly" + homopolymer : p.toLowerCase()) };
  }

  // 数据集里把一个字段变成模型用的数值（与 s10_build_main_table.val 同口径）
  function fieldValue(name, f) {
    if (!f) return NaN;
    if ((f.status || "").startsWith("room_temperature")) return 22.0;
    if (!["uM", "mM", "pH", "degC", "percent_basis_unspecified", "mg/ml"].includes(f.unit) || f.unit === "mg/ml") return NaN;
    let x;
    if (f.value != null) x = f.value;
    else if (f.values && f.values.length) { const v = f.values.slice().sort((a, b) => a - b), n = v.length; x = n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2; }
    else if (f.lower != null && f.upper != null) x = (f.lower + f.upper) / 2;
    else x = f.lower != null ? f.lower : (f.upper != null ? f.upper : NaN);
    if (name === "Conc" && !Number.isNaN(x)) return Math.log10(Math.max(x, 1e-4));
    return x;
  }
  function stateOf(status, v) {
    if (v == null || Number.isNaN(v)) return "miss";
    if ((status || "").startsWith("absent_from_reported")) return "zero";
    if (status === "room_temperature_unspecified") return "rt";
    return "obs";
  }

  // ------------------------------------------------------------------ 第二层：自由文本路由
  const SALT_NAMES = "nacl|kcl|mgcl2|licl|nh4cl|k-?glutamate|kglu|sodium glutamate|naglu|koac|potassium acetate|naoac|sodium acetate|mg\\(oac\\)2|mgoac|kscn|nascn|nabr|kbr|nai|cacl2|zncl2|mncl2|nicl2|\\(nh4\\)2so4|na2so4|k2so4|mgso4|potassium glutamate";
  const SYNONYMS = [[/sodium chloride/gi, "NaCl"], [/potassium chloride/gi, "KCl"], [/magnesium chloride/gi, "MgCl2"], [/calcium chloride/gi, "CaCl2"], [/ammonium sulfate|ammonium sulphate/gi, "(NH4)2SO4"],
    [/MgCl₂/g, "MgCl2"], [/CaCl₂/g, "CaCl2"], [/degrees? ?c(?:elsius)?\b/gi, " C"], [/deg ?c\b/gi, " C"], [/\bmicromolar\b/gi, "uM"], [/\bmillimolar\b/gi, "mM"], [/\bnanomolar\b/gi, "nM"]];
  const AMT = "(?:(?:<=|>=|<|>)\\s*)?" + NUM + "(?:\\s*(?:-|to)\\s*" + NUM + ")?";
  const RE_TEMP = new RegExp("(^|[^\\w.\\-])((?:(?:<=|>=|<|>)\\s*)?-?" + NUM + "(?:\\s*[CK]?\\s*(?:-|to)\\s*-?" + NUM + ")?)\\s*([CK])(?![A-Za-z])", "g");
  const RE_RT = /\b(?:rt|r\.t\.|room temp(?:erature)?|ambient(?: temp(?:erature)?)?)\b/i;
  const RE_CROWD_KW = "(?:peg|polyethylene glycol)\\s*-?\\s*" + NUM + "?\\s*(?:k|kda)?|ficoll\\s*-?\\s*\\d*|dextran\\s*(?:t-?\\s*)?\\d*\\s*(?:k|kda)?|glycerol";
  const UNIT_SEP = "(?:\\s*(?:mm|um|nm|m)(?=\\s*(?:-|to|or)\\s*[\\d.]))?";     // 区间里第一个数后面的单位，只在后面紧跟分隔符时才吃掉
  const SAMT = "(?:(?:<=|>=|<|>)\\s*)?" + NUM + UNIT_SEP + "(?:\\s*(?:-|to|or)\\s*" + NUM + UNIT_SEP + ")*";
  const AMT_C = "(?:(?:<=|>=|<|>)\\s*)?" + NUM + "(?:\\s*(?:%|mg/ml|g/l)?\\s*(?:-|to)\\s*" + NUM + ")?\\s*(?:%|mg/ml|g/l)(?:\\s*[\\(\\[]?(?:w/v|v/v|w/w|wt|vol)[\\)\\]]?)?";
  const NA_KW = "poly\\s*-?\\(?[ACGUTacgut]\\)?(?![a-z])|trna|rrna|mrna|total rna|ssdna|dsdna|g-?quadruplex|\\bg4\\b|plasmid|oligo(?:nucleotide)?|aptamer|\\brna\\b|\\bdna\\b";

  function preprocess(text) {
    let s = norm(text);
    for (const [rx, to] of SYNONYMS) s = s.replace(rx, to);
    return norm(s);
  }
  function numOrNull(x) { return x == null || Number.isNaN(x) ? null : x; }

  function parseExperiment(rawText, opts) {
    opts = Object.assign({ unmentionedSalts: "unknown", unmentionedCrowders: "zero", expandPBS: true, mw: null, proteinName: "" }, opts || {});
    let text = preprocess(rawText);
    const pname = (opts.proteinName || "").trim();
    if (pname.length >= 2) text = norm(text.split(new RegExp(pname.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "ig")).join("PROTEIN"));
    const found = [];      // [{kind, text}]
    const claimed = [];    // [start, end] 已被某一项占用的片段
    const claim = (s, e, kind, shown) => { claimed.push([s, e]); found.push({ kind, text: shown != null ? shown : text.slice(s, e).trim() }); };
    const isClaimed = (s, e) => claimed.some(([a, b]) => s < b && e > a);
    const res = { text, fields: {}, buffer: {}, additive: {}, na: null, found, notes: [], unmatched: [] };

    // ---- 温度
    let tField = emptyValue(null, "degC"), m, rx;
    RE_TEMP.lastIndex = 0;
    while ((m = RE_TEMP.exec(text))) {
      const s = m.index + m[1].length, e = m.index + m[0].length;
      if (m[3] === "K") { const kv = (m[2].match(new RegExp(NUM, "g")) || []).map(parseFloat); if (!kv.length || kv.some((x) => x < 150 || x > 400)) continue; }
      tField = parseTemperature(m[2] + " " + m[3]); claim(s, e, "temperature", text.slice(s, e)); break;
    }
    if (tField.status === "missing") { const r = RE_RT.exec(text); if (r) { tField = parseTemperature("RT"); claim(r.index, r.index + r[0].length, "temperature", r[0]); } }

    // ---- pH 与缓冲液文本
    const mPh = new RegExp("p\\s*h\\s*[:=]?\\s*(" + NUM + ")(?:\\s*-\\s*(" + NUM + "))?", "i").exec(text);
    let phField = parsePh(text);
    if (mPh) claim(mPh.index, mPh.index + mPh[0].length, "pH", mPh[0]);
    else if (phField.status === "inferred_from_buffer_number") found.push({ kind: "pH", text: "inferred from buffer number" });

    // ---- 盐：找盐名，向左 / 向右取相邻的"数值 + 摩尔单位"
    const saltPieces = [];
    rx = new RegExp("(^|[^A-Za-z0-9])(" + SALT_NAMES + ")(?![A-Za-z0-9])", "gi");
    while ((m = rx.exec(text))) {
      const s = m.index + m[1].length, e = m.index + m[0].length, saltName = m[2];
      const left = new RegExp("(" + SAMT + ")\\s*(mm|um|nm|m)\\s*$", "i").exec(text.slice(0, s));
      let piece = null, a0 = s, a1 = e;
      if (left) { piece = left[1] + " " + left[2] + " " + saltName; a0 = s - left[0].length; }
      else {
        const right = new RegExp("^\\s*(?:[:=]|at)?\\s*(" + SAMT + ")\\s*(mm|um|nm|m)(?![a-z])", "i").exec(text.slice(e));
        if (right) { piece = right[1] + " " + right[2] + " " + saltName; a1 = e + right[0].length; }
      }
      if (piece && !isClaimed(a0, a1)) { saltPieces.push(piece); claim(a0, a1, "salt", piece); }
    }
    let saltFields = { NaCl: null, KCl: null, MgCl2: null }, saltOthers = [];
    if (/\bpbs\b|\bdpbs\b/i.test(text) && opts.expandPBS && !saltPieces.some((p) => /nacl/i.test(p))) {
      const pbs = "137 mM NaCl; 2.7 mM KCl";
      const r = parseSalts(pbs); saltFields = r.fields; res.notes.push("PBS expanded to 137 mM NaCl + 2.7 mM KCl (standard 1x composition; assumed).");
      found.push({ kind: "salt", text: "PBS → 137 mM NaCl, 2.7 mM KCl (assumed)" });
    }
    if (saltPieces.length) { const r = parseSalts(saltPieces.join("; ")); for (const k of ["NaCl", "KCl", "MgCl2"]) if (r.fields[k]) saltFields[k] = r.fields[k]; saltOthers = r.others; }
    const saltReported = saltPieces.length > 0 || saltFields.NaCl != null || /\b(?:no|without|zero) salt\b|salt-free/i.test(text) || opts.unmentionedSalts === "zero";
    res.saltsOther = saltOthers;

    // ---- 拥挤剂
    const crowdPieces = [];
    const clauses = [];
    rx = new RegExp("(" + AMT_C + ")\\s*(?:[\\(\\[]?(?:w/v|v/v|w/w)[\\)\\]]?\\s*)?(?:of\\s+)?(" + RE_CROWD_KW + ")(?![a-z])", "gi");
    while ((m = rx.exec(text))) {
      const s = m.index, e = s + m[0].length;
      if (isClaimed(s, e)) continue;
      clauses.push([s, e]); crowdPieces.push(m[1].replace(/\s*[\(\[]?(?:w\/v|v\/v|w\/w|wt|vol)[\)\]]?/gi, "") + " " + m[2]);
    }
    rx = new RegExp("(" + RE_CROWD_KW + ")(?![a-z])\\s*(?:[:=(]|at)?\\s*(" + AMT_C + ")", "gi");
    while ((m = rx.exec(text))) {
      const s = m.index, e = s + m[0].length;
      if (isClaimed(s, e) || clauses.some(([a, b]) => s < b && e > a)) continue;
      clauses.push([s, e]); crowdPieces.push(m[2].replace(/\s*[\(\[]?(?:w\/v|v\/v|w\/w|wt|vol)[\)\]]?/gi, "") + " " + m[1]);
    }
    for (const [s, e] of clauses) claim(s, e, "crowder", text.slice(s, e));
    const cr = parseCrowders([[crowdPieces.join("; "), "Crowding agent"]]);
    const crowdReported = crowdPieces.length > 0 || /\b(?:no|without|zero) (?:crowder|crowding|peg|ficoll|dextran|glycerol)\b/i.test(text) || opts.unmentionedCrowders === "zero";

    // ---- 核酸
    let na = null; const c0 = claimed.length;
    rx = new RegExp("(" + NA_KW + ")", "gi");
    const naHits = []; while ((m = rx.exec(text))) naHits.push([m.index, m.index + m[0].length, m[0]]);
    if (naHits.length) {
      const s0 = Math.max(0, naHits[0][0] - 70), e0 = Math.min(text.length, naHits[naHits.length - 1][1] + 70);
      // 用逗号 / 分号切出含核酸关键词的从句
      const parts = []; let pos = 0;
      for (const seg of text.split(/[;]|,(?![^()]*\))/)) { const a = text.indexOf(seg, pos); pos = a + seg.length; parts.push([a, a + seg.length, seg]); }
      const segs = parts.filter(([a, b]) => naHits.some(([x, y]) => x >= a && y <= b));
      const seg = segs.length ? segs.map((p) => p[2]).join(" ") : text.slice(s0, e0);
      const hit = naHits[0][2];
      const kind = /\bdna\b|dsdna|ssdna|plasmid|oligo/i.test(seg) && !/\brna\b|polyu|poly\(?a\)?|trna|rrna|mrna/i.test(seg) ? "DNA" : "RNA";
      const piece = parseNucleicPiece(seg);
      const mConc = new RegExp("(" + AMT + ")\\s*(um|nm|pm|mm|ug/ml|ng/ul|ng/ml|mg/ml|ug/ul|g/l|mg/l)(?![a-z])", "i").exec(seg);
      const mLen = new RegExp("(" + NUM + ")\\s*-?\\s*(nt|bp|mer|kb|kbp)\\b", "i").exec(seg);
      let length = piece.length;
      if (mLen && /^kb/i.test(mLen[2])) length = parseFloat(mLen[1]) * 1000; else if (mLen) length = parseFloat(mLen[1]);
      const mRatio = new RegExp("(" + NUM + ")\\s*:\\s*(" + NUM + ")").exec(seg);
      const names = NAME_CLASSES.filter(([, r]) => r.test(seg)).map(([k]) => k);
      let conc = null, concNote = "";
      if (mConc) {
        const am = parseAmount(mConc[1] + " " + mConc[2]);
        if (am) {
          const f = concentrationUM(am, mConc[0], null);
          if (f.status.includes("mass_unit_unconverted")) {
            // 质量单位：核酸的长度已知时按平均每个核苷酸（碱基对）质量换算（数据集没有换算；这里换算并标记）
            const perNt = kind === "DNA" ? (names.includes("ssDNA") ? 330 : 650) : 340;
            const massUgMl = fieldValue("x", Object.assign({}, f, { unit: "uM" })) * 1; // f 里 value 以 mg/ml 单位保存
            if (length && !Number.isNaN(massUgMl)) { const ugml = massUgMl * 1000; conc = ugml / (length * perNt) * 1000; concNote = "mass concentration converted with length × " + perNt + " g/mol per " + (kind === "DNA" && !names.includes("ssDNA") ? "bp" : "nt") + " (not done in the dataset)"; }
            else concNote = "mass concentration without length: left unknown (as in the dataset)";
          } else { const v = fieldValue("x", f); conc = Number.isNaN(v) ? null : v; }
        }
      }
      na = { kind, names, length: length || null, concUM: conc, concNote, sequence: piece.sequence, ratio: mRatio ? parseFloat(mRatio[1]) / parseFloat(mRatio[2]) : null, rawText: seg.trim(), hit };
      const base = Math.max(0, text.indexOf(seg.trim()));
      const claimIn = (sub, kind) => { if (!sub) return; const i = text.indexOf(sub, base); if (i >= 0 && !isClaimed(i, i + sub.length)) claim(i, i + sub.length, kind, sub); };
      for (const [a, b, shown] of naHits) if (a >= base && b <= base + seg.trim().length + 1 && !isClaimed(a, b)) claim(a, b, "nucleic acid", shown);
      claimIn(mConc && mConc[0], "nucleic acid amount"); claimIn(mLen && mLen[0], "nucleic acid length"); claimIn(mRatio && mRatio[0], "nucleic acid ratio");
    }
    res.na = na; const naSpans = claimed.slice(c0);

    // ---- 蛋白浓度
    let conc = emptyValue(null, "uM");
    const nm = (opts.proteinName || "").trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const cue = "(?:protein|peptide|conc(?:entration)?(?: of protein)?|idr|lcd|domain|construct" + (nm ? "|" + nm : "") + ")";
    const PU = "(?:um|nm|pm|mm|mg/ml|ug/ml|g/l)";
    const AM_P = "(?:(?:<=|>=|<|>)\\s*)?" + NUM + "(?:\\s*" + PU + ")?(?:\\s*(?:-|to)\\s*" + NUM + ")?\\s*" + PU;
    const pats = [new RegExp("\\[?" + cue + "\\]?\\s*(?:conc(?:entration)?)?\\s*(?:[:=(]|at|of|is)?\\s*(" + AM_P + ")(?![a-z/])", "ig"), new RegExp("(" + AM_P + ")(?![a-z/])\\s*(?:of\\s+)?\\[?" + cue + "\\]?", "ig")];
    outer: for (const p of pats) {
      p.lastIndex = 0;
      while ((m = p.exec(text))) {
        const s = m.index, e = s + m[0].length;
        if (isClaimed(s, e) && !/protein|conc/i.test(m[0])) continue;
        const am = parseAmount(m[1]);
        if (am) { conc = concentrationUM(am, m[1], opts.mw); claim(s, e, "protein concentration", m[0]); break outer; }
      }
    }
    if (conc.status === "missing") {
      // 没有明确的"protein"提示：取未被占用的、微摩尔到纳摩尔量级的唯一数值
      const cands = []; rx = new RegExp("(" + AMT + ")\\s*(um|nm|pm)(?![a-z/])", "ig");
      while ((m = rx.exec(text))) { const s = m.index, e = s + m[0].length; if (!isClaimed(s, e)) cands.push([s, e, m[1] + " " + m[2]]); }
      if (cands.length === 1) { const am = parseAmount(cands[0][2]); if (am) { conc = concentrationUM(am, cands[0][2], opts.mw); claim(cands[0][0], cands[0][1], "protein concentration", cands[0][2]); res.notes.push("Protein concentration taken from the only free amount in µM/nM; check it."); } }
    }

    // ---- 缓冲液与添加剂（词表与 rs01 相同）
    const famHits = (list, kind) => list.filter(([, r]) => r.test(text)).map(([k, r]) => {
      const mm = r.exec(text);
      if (mm && naSpans.some(([x, y]) => mm.index < y && mm.index + mm[0].length > x)) return null;   // 核酸名里的 poly( 不是添加剂
      if (mm) {
        let a = mm.index; let b = mm.index + mm[0].length;
        const left = new RegExp("(" + AMT + ")\\s*(?:mM|M|uM|nM|%|mg/ml|ug/ml)\\s*(?:(?:sodium|potassium)\\s+)?$", "i").exec(text.slice(0, a));
        if (left) a -= left[0].length;
        const tail = /^\s*-?\s*hcl(?![a-z])/i.exec(text.slice(b)); if (tail && kind === "buffer") b += tail[0].length;
        if (!isClaimed(a, b)) claim(a, b, kind, text.slice(a, b).trim());
      }
      return k;
    }).filter(Boolean);
    const bufFam = famHits(BUF, "buffer");
    const addFam = famHits(ADD, "additive");
    const hasBufText = /\bbuffer(?:ed)?\b/i.test(text) || mPh != null || bufFam.length > 0;
    const bufState = bufFam.length ? "obs" : (hasBufText ? "other" : "miss");
    const addState = addFam.length ? "obs" : (hasBufText ? "none" : "miss");
    res.buffer = { families: bufFam, state: bufState };
    res.additive = { families: addFam, state: addState };

    // ---- 汇总 13 个字段（值 + 状态），与数据集同一套语义
    const F = {};
    const put = (short, name, f) => { const v = fieldValue(name, f); F[short] = { value: Number.isNaN(v) ? null : v, state: stateOf(f && f.status, v), status: f ? f.status : "missing", raw: f ? f.raw : null, lower: f ? f.lower : null, upper: f ? f.upper : null }; };
    put("conc", "Conc", conc);
    put("pH", "pH", phField);
    put("temp", "temp", tField);
    for (const k of ["MgCl2", "NaCl", "KCl"]) {
      let f = saltFields[k];
      if (!f) { f = emptyValue(null, "mM"); if (saltReported) { f.value = 0.0; f.status = "absent_from_reported_salts"; } }
      put(k, k, f);
    }
    const crowdMap = { "PEG300-1k": "PEG1k", "PEG3k-6k": "PEG6k", "PEG8k-20k": "PEG20k", Ficoll: "Ficoll", "Dextran -40": "Dex40", "Dextran 70-": "Dex70", Glycerol: "Glycerol" };
    for (const k of FIELDS.slice(6)) {
      let f = cr.fields[k];
      if (!f) { f = emptyValue(null, "percent_basis_unspecified"); if (crowdReported) { f.value = 0.0; f.status = "absent_from_reported_crowders"; } }
      put(crowdMap[k], k, f);
    }
    res.fields = F;
    res.crowdOthers = cr.others;
    // 没认出的文字：去掉已占用的片段后剩下的非空白
    let rest = text; claimed.slice().sort((a, b) => b[0] - a[0]).forEach(([s, e]) => { rest = rest.slice(0, s) + " " + rest.slice(e); });
    res.unmatched = rest.split(/[;,]/).map((x) => x.trim()).filter((x) => x && !/^(?:and|at|in|with|the|of|buffer|protein)$/i.test(x) && !/^[\s\-–]+$/.test(x));
    return res;
  }

  return { norm, isBlank, parseAmount, concentrationUM, parseSolutes, parseSalts, parsePh, parseTemperature, parseCrowders, parseNucleicPiece, pegBin, molWeight, fieldValue, stateOf,
    parseExperiment, preprocess, FIELDS, SHORT, BUF, ADD, NAME_CLASSES, AA, RESIDUE_MASS };
});
