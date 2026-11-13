"""Guidance tab 的“更多例子”：6 个新的 (y, c) 集合的引导采样轨迹（用户 11/13 选了“更多 (y,c) 集合，6 个”）。
算法、设定、对照与 rs70_yc_trajectories.py 完全相同（σ=0.16 的 rs25_global_s016.npz 条件能量面、CFG 引导能量 En + w(E_y − En)、Langevin η = 0.25σ²、
温度 T=0.5、1200 步、w=4、K=16、MAXSTART=400、NSHOW=24；主实验 = 引导到相反标签，对照 = 无引导 / 引导到同一标签），只是把集合换成 6 个同一篇论文内 y=1 与 y=0 都不少的条件格。
这份脚本只换集合定义、文件名和输出前缀 gd_，不改任何已有文件。WSL 上用 pixi ~/t3ebm（torch CPU）跑：
  cd ~/rs7 && nice -n 10 env NTHR=4 pixi run --manifest-path ~/t3ebm/pixi.toml python -I rs131_yc_new_trajectories.py
条件格的定义（pH_bin、temp_bin、nacl_bin、crowd、conc_bin）与 rs60 / rs62 / rs70 相同。
输出 ~/rs7/gd_yc_{id}.npz（起点、终点、展示链路径、集合记录索引）、gd_yc_summary.json（与 rs70_summary.json 同格式）、gd_yc_cells_check.json（每个格的记录数核对）。"""
import json, os, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
D = os.path.expanduser("~/rs7"); T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
torch.set_num_threads(int(os.environ.get("NTHR", "4")))
dev = os.environ.get("DEV", "cpu"); SIG = 0.16; ETA = 0.25 * SIG ** 2; TEMP = 0.5; STEPS = int(os.environ.get("STEPS", "1200")); W = 4.0; K = int(os.environ.get("K", "16")); MAXSTART = 400; NSHOW = 24
torch.manual_seed(0); rng = np.random.RandomState(0)
A = np.load(f"{D}/rs25_global_s016.npz"); Zf = np.load(f"{D}/rs22_fields.npz"); U = Zf["U_all"].astype(np.float64); yA = Zf["y_all"]; Pb = U[yA >= 0]; tree = cKDTree(Pb)
ex, ey = A["g_ex"].astype(np.float64), A["g_ey"].astype(np.float64); dx = ex[1] - ex[0]; NE = len(ex); E1, E0, En = A["g_E1"].astype(np.float64), A["g_E0"].astype(np.float64), A["g_En"].astype(np.float64)
c0 = (U.min(0) + U.max(0)) / 2; half = (U.max(0) - U.min(0)).max() / 2 * 1.12; cell = 2 * half / 36; lo = np.array([ex[0], ey[0]]); hi = np.array([ex[-1], ey[-1]])
def norm_logp(E): m = E.min(); return -(E - m) - np.log(np.exp(-(E - m)).sum() * dx * dx)
Rgrid = norm_logp(E1) - norm_logp(E0)
t32 = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
def grad_fields(E): gy, gx = np.gradient(E, dx, dx); return np.stack([gx, gy])
GE = {"1": t32(grad_fields(E1)), "0": t32(grad_fields(E0)), "n": t32(grad_fields(En))}; Rt = t32(Rgrid); lo_t, hi_t = t32(lo), t32(hi)
def interp(field, X): g = (X - lo_t) / (hi_t - lo_t) * 2 - 1; return F.grid_sample(field[None], g[None, :, None, :], mode="bilinear", padding_mode="border", align_corners=True)[0, :, :, 0].T
def score(X, y, w): gn = interp(GE["n"], X); g = gn if w == 0 else gn + w * (interp(GE[str(y)], X) - gn); return -g
r_of = lambda X: interp(Rt[None], X)[:, 0]
REC = 25
def langevin(X0, y, w, steps, keep=None):
    X = X0.clone(); path = [X[keep].clone()] if keep is not None else None; snaps = [X.clone()]
    for t in range(steps):
        X = X + ETA * score(X, y, w) + np.sqrt(2 * ETA * TEMP) * torch.randn_like(X); X = torch.minimum(torch.maximum(X, lo_t), hi_t)
        if keep is not None: path.append(X[keep].clone())
        if (t + 1) % REC == 0: snaps.append(X.clone())
    return X, (torch.stack(path, 1).cpu().numpy() if keep is not None else None), torch.stack(snaps, 1).cpu().numpy()                    # snaps: [链, 步数/REC + 1, 2]

# ---- 记录与条件格（与 rs70 相同）
R = pd.read_csv(f"{D}/rs_records.tsv", sep="\t", low_memory=False)
for c in ("conc", "pH", "temp", "NaCl", "PEG1k", "PEG6k", "PEG20k", "Ficoll", "Dex40", "Dex70"): R[c] = pd.to_numeric(R[c], errors="coerce")
R["pH_bin"] = (R.pH * 2).round() / 2; R["temp_bin"] = pd.cut(R.temp, [-100, 15, 30, 40, 1000], labels=["<15C", "15-30C", "30-40C", ">40C"]).astype(str); R["nacl_bin"] = pd.cut(R.NaCl, [-1, 0.5, 50, 100, 200, 400, 1e9], labels=["0", "<=50", "50-100", "100-200", "200-400", ">400"]).astype(str)
peg = R[["PEG1k", "PEG6k", "PEG20k"]].fillna(0).sum(axis=1); oth = R[["Ficoll", "Dex40", "Dex70"]].fillna(0).sum(axis=1); R["crowd"] = np.where(peg > 0, "PEG", np.where(oth > 0, "Ficoll/Dex", "none")); R["conc_bin"] = np.floor(R.conc * 2) / 2
base = (R.partition == "protein only") & (R.pool == "binary") & R.label.notna() & (R.st_conc == "obs")
def get_cell(mask): g = R[base & mask]; return g[g.label == 1].index.values, g[g.label == 0].index.values, g
def cellmask(paper, pH, temp, nacl, crowd, conc): return (R.paper == paper) & (R.pH_bin == pH) & (R.temp_bin == temp) & (R.nacl_bin == nacl) & (R.crowd == crowd) & (R.conc_bin == conc)
# id: (paper, pH_bin, temp_bin, nacl_bin, crowd, conc_bin, 期望 (n1, n0, 序列数), 说明)
CELLS = {
    "tau_peg": ("pmid:32229582", 7.5, "15-30C", "50-100", "PEG", 1.0, (21, 25, 9), "Tau / MAPT (pmid:32229582), 15–30 °C, pH 7.5, 50–100 mM NaCl, PEG, 10–32 µM"),
    "fus": ("pmid:29961577", 7.5, "15-30C", "0", "none", 0.5, (28, 20, 26), "FUS variants (pmid:29961577), 15–30 °C, pH 7.5, 0 NaCl, no crowder, 3.2–10 µM"),
    "ubqln2": ("pmid:30982635", 7.0, ">40C", "100-200", "none", 1.0, (7, 9, 9), "UBQLN2 (pmid:30982635), > 40 °C, pH 7.0, 100–200 mM NaCl, no crowder, 10–32 µM"),
    "zo1": ("pmid:31675499", 7.5, "15-30C", "0", "PEG", 0.5, (7, 10, 12), "ZO-1 / Q07157 (pmid:31675499), 15–30 °C, pH 7.5, 0 NaCl, PEG, 3.2–10 µM"),
    "g3bp1": ("pmid:32302571", 7.5, "15-30C", "100-200", "Ficoll/Dex", 2.0, (6, 5, 4), "G3BP1 (pmid:32302571), 15–30 °C, pH 7.5, 100–200 mM NaCl, Ficoll/dextran, 100–316 µM"),
    "hnrnpa2b1": ("pmid:33349959", 5.5, "15-30C", "<=50", "none", 1.0, (5, 7, 8), "hnRNPA2B1 (pmid:33349959), 15–30 °C, pH 5.5, ≤ 50 mM NaCl, no crowder, 10–32 µM"),
}
sets = {}; check = {}
for name, (paper, pH, tb, nb, cr, cb, exp, desc) in CELLS.items():
    i1, i0, g = get_cell(cellmask(paper, pH, tb, nb, cr, cb)); sets[name] = (i1, i0, g)
    actual = dict(n1=int(len(i1)), n0=int(len(i0)), n_seq=int(g.seq_row.nunique()), n_paper=int(g.paper.nunique()), n_prot=int(g.uniprot.nunique()), proteins=g.protein_name.fillna("").astype(str).value_counts().head(3).to_dict(), uniprot=g.uniprot.dropna().astype(str).value_counts().head(3).to_dict())
    check[name] = dict(expected=dict(n1=exp[0], n0=exp[1], n_seq=exp[2]), actual=actual, match=(actual["n1"], actual["n0"], actual["n_seq"]) == tuple(exp)); log("核对", name, check[name])
json.dump(check, open(f"{D}/gd_yc_cells_check.json", "w"), indent=1, default=str)
DESC = {n: v[7] for n, v in CELLS.items()}
summ = {}
for name, (i1, i0, g) in sets.items():
    if min(len(i1), len(i0)) < 3: log(name, "两类之一少于 3 条，跳过"); continue
    log(name, "n1", len(i1), "n0", len(i0), "序列", g.seq_row.nunique(), "论文", g.paper.nunique(), "蛋白", g.uniprot.nunique())
    P1, P0 = U[i1], U[i0]; res = dict(desc=DESC[name], n1=int(len(i1)), n0=int(len(i0)), n_seq=int(g.seq_row.nunique()), n_paper=int(g.paper.nunique()), n_prot=int(g.uniprot.nunique()), map_dist_medians=float(np.linalg.norm(np.median(P1, 0) - np.median(P0, 0))), r_y1=float(r_of(t32(P1)).mean()), r_y0=float(r_of(t32(P0)).mean()), on_support_frac=float((tree.query(U[np.r_[i1, i0]])[0] / cell <= 1.2).mean()))
    res["dr_y1_minus_y0"] = res["r_y1"] - res["r_y0"]; out = {"idx1": i1, "idx0": i0, "U1": P1, "U0": P0}
    for a, (ia, Pa, Pt) in {"0to1": (i0, P0, P1), "1to0": (i1, P1, P0)}.items():
        tgt, src = (1, 0) if a == "0to1" else (0, 1); sel = ia if len(ia) <= MAXSTART else rng.choice(ia, MAXSTART, replace=False); S0 = U[sel]; X0 = np.repeat(S0, K, axis=0) + rng.normal(0, 0.03, (len(S0) * K, 2)); X0t = t32(X0); show = np.linspace(0, len(X0) - 1, NSHOW).astype(int)
        trT = cKDTree(Pt); d_start = trT.query(X0)[0]; r0 = r_of(X0t).cpu().numpy(); res[a] = {"d_start_median": float(np.median(d_start)), "r_start_median": float(np.median(r0)), "n_chains": int(len(X0))}
        for key, (y, w) in {"guided": (tgt, W), "stay": (src, W), "none": (1, 0.0)}.items():
            torch.manual_seed(5); Xe, path, snaps = langevin(X0t, y, w, STEPS, keep=show if key == "guided" else None); out[f"{a}_snap_{key}"] = snaps.astype(np.float16); Xn = Xe.cpu().numpy().astype(np.float64); d_end = trT.query(Xn)[0]; re = r_of(Xe).cpu().numpy(); onsup = tree.query(Xn)[0] / cell <= 1.2
            res[a][key] = dict(d_end_median=float(np.median(d_end)), arrived_0p10=float((d_end <= 0.10).mean()), arrived_0p20=float((d_end <= 0.20).mean()), r_end_median=float(np.median(re)), dr_median=float(np.median(re - r0)), dr_p25=float(np.percentile(re - r0, 25)), dr_p75=float(np.percentile(re - r0, 75)), frac_dr_toward=float(np.mean((re - r0) * (1 if tgt == 1 else -1) > 0)), on_support=float(onsup.mean()))
            out[f"{a}_end_{key}"] = Xn.astype(np.float32)
            if key == "guided": out[f"{a}_path"] = path.astype(np.float32)
        out[f"{a}_start"] = X0.astype(np.float32); out[f"{a}_show"] = show
        log(name, a, {k: {x: round(v, 3) for x, v in res[a][k].items()} for k in ("guided", "stay", "none")})
    summ[name] = res; np.savez_compressed(f"{D}/gd_yc_{name}.npz", **out)
    json.dump(summ, open(f"{D}/gd_yc_summary.json", "w"), indent=1)
log("完成", dict(eta=ETA, temp=TEMP, steps=STEPS, w=W, K=K, threads=torch.get_num_threads(), dev=dev)); print("DONE", flush=True)
