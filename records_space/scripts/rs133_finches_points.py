"""地图点→序列（rs107 的 5 个组合 × 12 个点）上的 FINCHES 同型相互作用 ε：锚点 vs 解码答案（Guidance tab 的悬停卡片用，用户 11/13：“序列变化和预测”，弹窗里选了“场的预测 + FINCHES”）。
WSL，pixi 项目 /root/t3fin（finches + metapredict），CPU，只读 ~/rs7/ 里别的会话的 rs107_points_decoded.tsv / rs_sequences.tsv，输出新文件 ~/rs7/gd_finches_points.tsv 与 gd_finches_points_summary.json。
ε 是 FINCHES 的同型相互作用参数（Mpipi 与 CALVADOS 两套力场前端；越负 = 越吸引 = 越倾向相分离）。FINCHES 为无序区设计、在 27 个实测 A1-LCD 变体上与 C_sat 的 Spearman 是 Mpipi +0.79、CALVADOS +0.57（task3_guidance，11/12–13）；
tau、SARS-CoV-2 N、TDP-43 含折叠域或结构化区，所以同时算“仅无序区”（metapredict 无序分数 ≥ 0.5 的位点拼接，锚点与答案取同一批位点；至少 30 个无序残基）。
对照：同样的替换位点，替换成的残基按锚点自己的组成随机抽（NR 次，默认 5），Δε_随机 = ε(随机替换序列) − ε(锚点)，用来看解码的 Δε 是否比随机替换更小。
环境变量：NR、NLIM（只算前 N 个，测速）。"""
import os, json, time
import numpy as np, pandas as pd
from finches import Mpipi_frontend, CALVADOS_frontend
import metapredict as meta
D = os.path.expanduser("~/rs7"); E = os.environ.get; NR = int(E("NR", "5")); NLIM = int(E("NLIM", "0")); AA = "ACDEFGHIKLMNPQRSTVWY"; rng = np.random.RandomState(133); T0 = time.time()
sq = pd.read_csv(f"{D}/rs_sequences.tsv", sep="\t", keep_default_na=False, low_memory=False); seqs = ["".join(s.upper().split()) for s in sq.sequence.values]
R = pd.read_csv(f"{D}/rs107_points_decoded.tsv", sep="\t", keep_default_na=False)
if NLIM: R = R.head(NLIM)
fm, fc = Mpipi_frontend(), CALVADOS_frontend()
cache = {}
def eps(s):
    if s not in cache: cache[s] = (float(fm.epsilon(s, s)), float(fc.epsilon(s, s)))
    return cache[s]
dis = {}
rows = []
for k, r in enumerate(R.itertuples()):
    a = int(r.anchor); S = seqs[a]; d = str(r.seq)
    assert len(d) == len(S), (r.combo, r.label, r.point, len(d), len(S))
    if a not in dis: dis[a] = np.array(meta.predict_disorder(S)) >= 0.5
    m = dis[a]; pos = [i for i in range(len(S)) if S[i] != d[i]]
    bg = np.array([S.count(x) for x in AA], float); bg /= bg.sum()
    rnd = []
    for _ in range(NR if pos else 0):
        l = list(S)
        for p in pos: l[p] = AA[rng.choice(20, p=bg)]
        rnd.append("".join(l))
    row = dict(combo=r.combo, label=r.label, point=int(r.point), anchor=a, length=len(S), n_edits=len(pos), n_idr_residues=int(m.sum()), frac_idr=round(float(m.mean()), 3), n_idr_sites=int(sum(m[p] for p in pos)))
    scopes = {"full": (S, d, rnd)}
    if m.sum() >= 30:
        idx = np.where(m)[0]; cut = lambda x: "".join(x[i] for i in idx); scopes["idr"] = (cut(S), cut(d), [cut(x) for x in rnd])
    for sc, (x, y, rr) in scopes.items():
        ea = eps(x); eb = eps(y) if pos else ea; er = np.array([eps(z) for z in rr]) if rr else None
        for j, ff in enumerate(("mpipi", "calvados")):
            row[f"{ff}_{sc}_src"] = ea[j]; row[f"{ff}_{sc}_dec"] = eb[j]; row[f"{ff}_{sc}_rnd"] = float(er[:, j].mean()) if er is not None else None
    rows.append(row)
    if k % 6 == 0: print(f"[{time.time() - T0:6.0f}s] {k}/{len(R)} {r.combo} {r.label}{r.point} k={len(pos)}", flush=True)
F = pd.DataFrame(rows); F.to_csv(f"{D}/gd_finches_points.tsv", sep="\t", index=False)
S = dict(n_points=int(len(F)), n_with_edits=int((F.n_edits > 0).sum()), NR=NR, note="eps = FINCHES homotypic interaction parameter (more negative = more attractive); idr = metapredict >= 0.5 sites of the anchor, >= 30 residues")
for ff in ("mpipi", "calvados"):
    for sc in ("full", "idr"):
        if f"{ff}_{sc}_src" not in F: continue
        g = F[(F.n_edits > 0) & F[f"{ff}_{sc}_src"].notna()]
        if not len(g): continue
        dd = g[f"{ff}_{sc}_dec"] - g[f"{ff}_{sc}_src"]; dr = g[f"{ff}_{sc}_rnd"] - g[f"{ff}_{sc}_src"]
        S[f"{ff}_{sc}"] = dict(n=int(len(g)), d_eps_mean=round(float(dd.mean()), 5), d_eps_median=round(float(dd.median()), 5), frac_stronger_PS=round(float((dd < 0).mean()), 3), d_eps_random_mean=round(float(dr.mean()), 5), abs_d_eps_decoded_median=round(float(dd.abs().median()), 5), abs_d_eps_random_median=round(float(dr.abs().median()), 5))
json.dump(S, open(f"{D}/gd_finches_points_summary.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(S, indent=1)); print("DONE", flush=True)
