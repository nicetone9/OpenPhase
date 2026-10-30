/* records 空间的浏览器端实现：序列处理、新序列的近似嵌入、与全部记录的距离（复现 rslib.py）、近邻与定位。
 * 不联网；所有输入只在本页面里处理。 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.RecordsSpace = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const AA = "ACDEFGHIKLMNPQRSTVWY";
  const AA_IDX = {}; for (let i = 0; i < 20; i++) AA_IDX[AA[i]] = i;
  const RESIDUE_MASS = { A: 71.0788, R: 156.1875, N: 114.1038, D: 115.0886, C: 103.1388, E: 129.1155, Q: 128.1307, G: 57.0519, H: 137.1411, I: 113.1594, L: 113.1594, K: 128.1741,
    M: 131.1926, F: 147.1766, P: 97.1167, S: 87.0782, T: 101.1051, W: 186.2132, Y: 163.1760, V: 99.1326 };
  const SK_K = 5, SK_THR = Math.floor(4294967296 / 8);

  // ------------------------------------------------------------------ 序列
  function cleanSequence(text) {
    let t = String(text || "");
    t = t.split("\n").filter((l) => !l.trim().startsWith(">")).join("");
    const raw = t.toUpperCase().replace(/[^A-Z*]/g, "").replace(/\*/g, "");
    let seq = "", bad = 0;
    for (const c of raw) { if (c in AA_IDX) seq += c; else bad++; }
    return { seq, nonstandard: bad, rawLength: raw.length };
  }
  function fnv1a32(str, basis) {
    let h = (basis == null ? 2166136261 : basis) >>> 0;
    for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
    return h >>> 0;
  }
  function sketch(seq) {
    const set = new Set();
    for (let i = 0; i + SK_K <= seq.length; i++) { const h = fnv1a32(seq.substr(i, SK_K)); if (h < SK_THR) set.add(h); }
    return Array.from(set).sort((a, b) => a - b);
  }
  function comp24(seq) {
    const n = seq.length, cnt = new Float64Array(20);
    for (const c of seq) cnt[AA_IDX[c]]++;
    const f = (a) => cnt[AA_IDX[a]];
    const out = [Math.log(n)];
    for (let i = 0; i < 20; i++) out.push(cnt[i] / n);
    out.push((f("K") + f("R") - f("D") - f("E")) / n, (f("K") + f("R") + f("D") + f("E")) / n, (f("F") + f("W") + f("Y")) / n);
    return out;
  }
  function dimer400(seq) {
    const v = new Float64Array(400);
    for (let i = 0; i + 1 < seq.length; i++) v[AA_IDX[seq[i]] * 20 + AA_IDX[seq[i + 1]]]++;
    const d = Math.max(seq.length - 1, 1);
    for (let i = 0; i < 400; i++) v[i] /= d;
    return v;
  }
  function molWeight(seq) { let m = 18.015; for (const c of seq) m += RESIDUE_MASS[c]; return m; }
  function normalize(v) { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) + 1e-9; return Float32Array.from(v, (x) => x / s); }

  // ------------------------------------------------------------------ 数据装载
  function views(buf, meta) {
    const T = { float32: Float32Array, int16: Int16Array, uint32: Uint32Array, int32: Int32Array, "<f4": Float32Array, "<i2": Int16Array, "<u4": Uint32Array, "<i4": Int32Array };
    const out = {};
    for (const a of meta.arrays) out[a.name] = new T[a.dtype](buf, a.offset, a.shape.reduce((x, y) => x * y, 1));
    return out;
  }
  function popcount(x) { x = x - ((x >> 1) & 0x55555555); x = (x & 0x33333333) + ((x >> 2) & 0x33333333); return ((((x + (x >> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24); }
  function jaccard(a, b) { const u = popcount(a | b); return u > 0 ? 1 - popcount(a & b) / u : 0; }

  class Space {
    constructor(d) {
      this.params = d.params; this.N = d.recordsMeta.n; this.M = d.seqsMeta.m;
      this.rec = views(d.recordsBuf, d.recordsMeta);
      this.sq = views(d.seqsBuf, d.seqsMeta);
      this.seqMeta = d.seqsMeta.seqs; this.approx = d.approx; this.coverage = d.coverage;
      this.zdim = this.sq.zn.length / this.M;
      this.inv = new Map();
      for (let s = 0; s < this.M; s++) for (let k = this.sq.sk_off[s]; k < this.sq.sk_off[s + 1]; k++) { const h = this.sq.sk_all[k]; let l = this.inv.get(h); if (!l) this.inv.set(h, (l = [])); l.push(s); }
    }
    zn(i) { return this.sq.zn.subarray(i * this.zdim, (i + 1) * this.zdim); }

    // ---- 新序列：精确匹配 / k-mer 相似 / 近似嵌入
    analyzeSequence(seqText) {
      const { seq, nonstandard } = cleanSequence(seqText);
      if (seq.length < 10) return { seq, nonstandard, tooShort: true };
      const h1 = fnv1a32(seq), h2 = fnv1a32(seq, 0x9747b28c);
      let exact = -1;
      for (let s = 0; s < this.M; s++) if (this.sq.h1[s] === h1 && this.sq.h2[s] === h2 && this.seqMeta[s].length === seq.length) { exact = s; break; }
      const q = sketch(seq), counts = new Map();
      for (const h of q) { const l = this.inv.get(h); if (l) for (const s of l) counts.set(s, (counts.get(s) || 0) + 1); }
      const sims = [];
      for (const [s, c] of counts) {
        const ks = this.sq.sk_off[s + 1] - this.sq.sk_off[s];
        sims.push({ s, shared: c, containQ: c / q.length, containK: c / ks, jaccard: c / (q.length + ks - c) });
      }
      sims.sort((a, b) => Math.max(b.containQ, b.containK) - Math.max(a.containQ, a.containK));
      const best = sims[0] || null;
      let cls = "novel";
      if (exact >= 0) cls = "exact";
      else if (best && (best.containQ >= 0.5 || best.containK >= 0.5 || best.jaccard >= 0.3) && q.length >= 8) cls = "variant";
      else if (best && best.shared >= 3 && Math.max(best.containQ, best.containK) >= 0.1) cls = "related";
      return { seq, nonstandard, length: seq.length, mw: molWeight(seq), comp: comp24(seq), sketchSize: q.length, exact, best, sims: sims.slice(0, 5), cls, similarSeqs: sims.filter((x) => x.containQ >= 0.5 || x.containK >= 0.5 || x.jaccard >= 0.3).map((x) => x.s) };
    }
    approxZ(seq) {
      const A = this.approx;
      let x = comp24(seq);
      if (A.feature === "comp24+dimer") x = x.concat(Array.from(dimer400(seq)));
      const F = A.W.length, D = A.b.length, z = new Float64Array(D);
      for (let j = 0; j < D; j++) z[j] = A.b[j];
      for (let f = 0; f < F; f++) { const v = (x[f] - A.mean[f]) / A.std[f], row = A.W[f]; for (let j = 0; j < D; j++) z[j] += v * row[j]; }
      return normalize(z);
    }

    // ---- 距离：复现 rslib.py（四个透镜）
    distances(q) {
      const N = this.N, R = this.rec, P = this.params, LAM = P.lam, W = P.block_w, PEN = P.temp_state_pen, scP = P.scales.P;
      const out = { full: new Float32Array(N), prot: new Float32Array(N), cond: new Float32Array(N), fullM: new Float32Array(N) };
      const hasQ = q.zn != null, zd = this.zdim, Zn = this.sq.zn;
      const nS = P.salt.length, nC = P.crowd.length;
      const qNA = q.naCls > 0;
      for (let j = 0; j < N; j++) {
        // ---- P
        const pj = R.pidx[j], hasJ = pj >= 0;
        let dP, oP, cP;
        if (hasQ && hasJ) {
          let dot = 0; const off = pj * zd; for (let t = 0; t < zd; t++) dot += q.zn[t] * Zn[off + t];
          const S = Math.max(0, 1 - dot); dP = Math.min(S / scP, 1); oP = dP; cP = 1;
        } else if (hasQ !== hasJ) { dP = LAM; oP = 0; cP = 0; } else { dP = 0; oP = 0; cP = 0; }
        // ---- C：5 个维度
        let sumP = 0, num = 0, den = 0;
        const scal = (a, b) => { const oa = !Number.isNaN(a), ob = !Number.isNaN(b); if (oa && ob) { const d = Math.min(Math.abs(a - b), 1); return [d, d, 1]; } if (oa !== ob) return [LAM, 0, 0]; return [0, 0, 0]; };
        let r = scal(q.conc, R.conc[j]); sumP += r[0]; num += r[1] * r[2]; den += r[2];
        r = scal(q.pH, R.pH[j]); sumP += r[0]; num += r[1] * r[2]; den += r[2];
        r = scal(q.temp, R.temp[j]);
        { const pen = r[2] && q.tempCode !== R.temp_code[j] ? PEN : 0; const p = Math.min(r[0] + pen, 1), o = Math.min(r[1] + pen, 1); sumP += p; num += o * r[2]; den += r[2]; }
        { let sp = 0, so = 0, any = 0; for (let t = 0; t < nS; t++) { const rr = scal(q.salt[t], R.salt[j * nS + t]); sp += rr[0]; so += rr[1]; any = any || rr[2]; } sumP += Math.min(sp, 1); num += Math.min(so, 1) * any; den += any; }
        { let sp = 0, so = 0, any = 0; for (let t = 0; t < nC; t++) { const rr = scal(q.crowd[t], R.crowd[j * nC + t]); sp += rr[0]; so += rr[1]; any = any || rr[2]; } sumP += Math.min(sp, 1); num += Math.min(so, 1) * any; den += any; }
        const dC = sumP / 5, oC = den > 0 ? num / den : 0.5, cC = den / 5;
        // ---- N
        const cj = R.na_cls[j];
        let dN, oN, cN;
        if (!qNA && cj === 0) { dN = 0; oN = 0; cN = 1; }
        else if (qNA !== (cj > 0)) { dN = 1; oN = 1; cN = 1; }
        else {
          let sp = 0, nu = 0, de = 0;
          const dcls = q.naCls !== cj ? 1 : 0; sp += dcls; nu += dcls; de += 1;
          const dn = jaccard(q.naFlags, R.na_flags[j]); sp += dn; nu += dn; de += 1;
          for (const [qa, ra] of [[q.naLen, R.na_len[j]], [q.naConc, R.na_conc[j]], [q.naRatio, R.na_ratio[j]]]) { const rr = scal(qa, ra); sp += rr[0]; nu += rr[1] * rr[2]; de += rr[2]; }
          for (const [qa, ra, n] of [[q.naSeq, R.na_seq, 4], [q.naFold, R.na_fold, 2]]) {
            const oq = !Number.isNaN(qa[0]), orr = !Number.isNaN(ra[j * n]);
            if (oq && orr) { let d2 = 0; for (let t = 0; t < n; t++) { const df = qa[t] - ra[j * n + t]; d2 += df * df; } const d = Math.min(Math.sqrt(d2), 1); sp += d; nu += d; de += 1; }
            else if (oq !== orr) sp += LAM;
          }
          dN = sp / 7; oN = nu / de; cN = de / 7;
        }
        // ---- B
        const setDim = (qm, qo, rm, ro) => { const bo = qo && ro; if (bo) { const d = jaccard(qm, rm); return [d, d, 1]; } if (qo !== ro) return [LAM, 0, 0]; return [0, 0, 0]; };
        const b1 = setDim(q.buf, q.bufObs, R.buf[j], R.buf_obs[j] === 1), b2 = setDim(q.add, q.addObs, R.add[j], R.add_obs[j] === 1);
        const dB = (b1[0] + b2[0]) / 2, deB = b1[2] + b2[2], oB = deB > 0 ? (b1[1] * b1[2] + b2[1] * b2[2]) / deB : 0.5, cB = deB / 2;
        // ---- 透镜
        out.full[j] = W.P * dP + W.N * dN + W.C * dC + W.B * dB;
        out.prot[j] = dP;
        out.cond[j] = (W.N * dN + W.C * dC + W.B * dB) / (W.N + W.C + W.B);
        const nm = W.P * cP * oP + W.N * cN * oN + W.C * cC * oC + W.B * cB * oB, dm = W.P * cP + W.N * cN + W.C * cC + W.B * cB;
        out.fullM[j] = dm > 0 ? nm / dm : 0.5;
      }
      return out;
    }
    queryOfRecord(i) {
      const R = this.rec, P = this.params, nS = P.salt.length, nC = P.crowd.length;
      return { zn: R.pidx[i] >= 0 ? this.zn(R.pidx[i]) : null, conc: R.conc[i], pH: R.pH[i], temp: R.temp[i], tempCode: R.temp_code[i],
        salt: Array.from(R.salt.subarray(i * nS, (i + 1) * nS)), crowd: Array.from(R.crowd.subarray(i * nC, (i + 1) * nC)),
        naCls: R.na_cls[i], naFlags: R.na_flags[i], naLen: R.na_len[i], naConc: R.na_conc[i], naRatio: R.na_ratio[i],
        naSeq: Array.from(R.na_seq.subarray(i * 4, i * 4 + 4)), naFold: Array.from(R.na_fold.subarray(i * 2, i * 2 + 2)),
        buf: R.buf[i], bufObs: R.buf_obs[i] === 1, add: R.add[i], addObs: R.add_obs[i] === 1 };
    }
    // 由解析结果构造查询向量（与 rslib.build_inputs 的缩放一致）
    buildQuery(spec) {
      const P = this.params, S = P.scales, NaN_ = NaN;
      const sc = (x, s) => (x == null || Number.isNaN(x) ? NaN_ : x / s);
      const pos = (x, s) => (x == null ? NaN_ : Math.log10(1 + Math.max(x, 0)) / s);
      const f = spec.fields;
      const q = { zn: spec.zn || null, conc: sc(f.conc.value, S.conc), pH: sc(f.pH.value, S.pH), temp: sc(f.temp.value, S.temp), tempCode: f.temp.state === "rt" ? 1 : (f.temp.state === "miss" ? 2 : 0) };
      q.salt = P.salt.map((k) => (f[k].state === "miss" ? NaN_ : pos(f[k].value, S[k])));
      q.crowd = P.crowd.map((k) => (f[k].state === "miss" ? NaN_ : pos(f[k].value, S[k])));
      const na = spec.na;
      q.naCls = na ? (na.kind === "DNA" ? 2 : 1) : 0;
      q.naFlags = 0; if (na) P.na_flags.forEach((k, i) => { if (na.names.includes(k)) q.naFlags |= 1 << i; });
      q.naLen = na && na.length ? Math.log10(na.length) / S.na_len : NaN_;
      q.naConc = na && na.concUM ? Math.log10(na.concUM) / S.na_conc : NaN_;
      let ratio = NaN_;
      if (na && na.ratioLog10 != null) ratio = na.ratioLog10 / S.na_ratio;
      q.naRatio = ratio;
      q.naSeq = [NaN_, NaN_, NaN_, NaN_]; q.naFold = [NaN_, NaN_];
      if (na && na.seqFeatures) {
        const nn = P.na_norm.na_seq, v = na.seqFeatures;
        q.naSeq = v.map((x, i) => (x - nn.mu[i]) / nn.sd[i] / nn.sc);
      }
      const mask = (list, fam) => { let m = 0; list.forEach((k, i) => { if (fam.includes(k)) m |= 1 << i; }); return m; };
      const bf = spec.buffer.families.slice(); if (spec.buffer.state === "other" && !bf.includes("other_buf")) bf.push("other_buf");
      q.buf = mask(P.buf, bf); q.bufObs = spec.buffer.state !== "miss";
      q.add = mask(P.add, spec.additive.families); q.addObs = spec.additive.state !== "miss";
      return q;
    }
    // k 个最近邻（可排除一组记录）
    nearest(dist, k, exclude) {
      const N = this.N, idx = [];
      for (let j = 0; j < N; j++) { if (exclude && exclude[j]) continue; idx.push(j); }
      idx.sort((a, b) => dist[a] - dist[b] || a - b);
      return idx.slice(0, k);
    }
    // 排除同一蛋白与同源簇：返回长度 N 的 0/1 数组
    excludeMask(seqIdxList) {
      const mask = new Uint8Array(this.N); if (!seqIdxList.length) return mask;
      const set = new Set(seqIdxList), clusters = new Set(seqIdxList.map((s) => this.seqMeta[s].c30).filter(Boolean));
      for (let j = 0; j < this.N; j++) { const pj = this.rec.pidx[j]; if (pj >= 0 && (set.has(pj) || (this.seqMeta[pj].c30 && clusters.has(this.seqMeta[pj].c30)))) mask[j] = 1; }
      return mask;
    }
    coverageLevel(lens, meanDist, unseen) {
      const c = this.coverage[lens], q = this.coverage.quantile_levels, arr = unseen ? c.unseen_mean15 : c.plain_mean15;
      let pct = 1; for (let i = 0; i < q.length; i++) { if (meanDist <= arr[i]) { pct = q[i]; break; } if (i === q.length - 1) pct = 1; }
      return { pct, arr, q };
    }
  }
  return { Space, cleanSequence, fnv1a32, sketch, comp24, dimer400, molWeight, normalize, popcount, jaccard, views };
});
