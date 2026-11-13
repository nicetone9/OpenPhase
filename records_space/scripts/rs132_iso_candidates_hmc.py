"""Guidance tab 的“更多例子”：更多等值线。先用 rs80 的同一套“有意义”判据评估候选等值线，再对通过的线用 rs95 的同一设定（一维弧长 + TorchEBM HMC）采样。
WSL，pixi ~/t3ebm（torch CPU + torchebm 0.9.1）：
  cd ~/rs7 && nice -n 10 env NTHR=4 pixi run --manifest-path ~/t3ebm/pixi.toml python -I rs132_iso_candidates_hmc.py
只新建文件（前缀 gd_iso_），不改 rs80 / rs95 的任何输出。

A. 候选评估（与 rs80_isocontours.py 逐行相同的场与指标）：每个协变量用 σ=0.16 的核平滑成标量场 g(u)；等值线 g = 水平值（只在核质量 ≥ 3 且离最近记录 ≤ 1.5 格的支撑区内画）；
   局部同质度 purity（线上各点局部标准差的中位数 / 全局标准差，越小越好）、线长 arc（地图 SD）、r(u) 沿线的分布与过零次数、线附近（0.12 SD 内）的记录数、y=1 / y=0 数、论文数、蛋白数；
   rs80 的得分 score = (1 − min(purity, 1)) × min(arc, 6)/6 × tanh(std r) × log(1 + 论文数)，且要求 arc ≥ 1.0、y=1 与 y=0 各 ≥ 10 条（否则 0）。
   “通过” = score > 0（即 rs80 的门槛：purity < 1、arc ≥ 1.0、两类各 ≥ 10 条）且 r 沿线有过零（线上 y 的场有变化）且线附近 ≥ 5 篇论文（rs80 的文档字符串里写了“多篇论文”，代码里只用 log 加权；这里加了这条门槛，已现有的六条线不受影响）。
   已有的六条线（长度 400 aa、NaCl 150 mM、温度 22 °C、pH 7.5、浓度 10 µM、拥挤剂 0.5%）也重新评估一遍，用来核对这里的数字与 rs80_table.tsv 一致，但不再采样。
B. 对通过的新线（局部同质度最小的 8 条）做 HMC：ε = 0.05、10 步蛙跳、T = 0.5、w = 4、400 个提议、60 个等弧长起点 × 8 = 480 条链，两个方向（引导到 y=1 / y=0）。
输出 ~/rs7/gd_iso_candidates.json（全部候选的评估数字与是否通过）、gd_iso_{id}.npz（等值线点、起点、展示链路径、终点）、gd_iso_summary.json（与 rs95_summary.json 同格式，键为 id）。"""
import json, os, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator as RGI
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.expanduser("~/rs7"); T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
torch.set_num_threads(int(os.environ.get("NTHR", "4"))); torch.manual_seed(0)
PART = os.environ.get("PART", "all"); MAXNEW = 8
dev = "cpu"; SIG = 0.16; R_NEAR = 0.12; NSTART = 60
Rr = pd.read_csv(f"{D}/rs_records.tsv", sep="\t", low_memory=False); Zf = np.load(f"{D}/rs22_fields.npz"); U = Zf["U_all"].astype(np.float64); yA = Zf["y_all"]; A = np.load(f"{D}/rs25_global_s016.npz"); T50 = np.load(f"{D}/rs50_contour_traj.npz")
ex, ey = A["g_ex"].astype(np.float64), A["g_ey"].astype(np.float64); dx = ex[1] - ex[0]; NE = len(ex); GX, GY = np.meshgrid(ex, ey); G = np.stack([GX.ravel(), GY.ravel()], 1); Rgrid = T50["R"].astype(np.float64)
c0 = (U.min(0) + U.max(0)) / 2; half = (U.max(0) - U.min(0)).max() / 2 * 1.12; cell = 2 * half / 36
num = lambda c: pd.to_numeric(Rr[c], errors="coerce").values.astype(float)
crowd = np.nansum(np.stack([num(c) for c in ("PEG1k", "PEG6k", "PEG20k", "Ficoll", "Dex40", "Dex70")], 1), 1); crowd = np.where(np.isnan(num("PEG6k")), np.nan, crowd)
COV = {"length": dict(x=np.log10(num("seq_len")), tr=lambda v: np.log10(v), unit="aa"),                                        # 与 rs80 的 COV 相同（去掉 lv 列表）
       "NaCl": dict(x=np.log10(1 + num("NaCl")), tr=lambda v: np.log10(1 + v), unit="mM NaCl"),
       "temp": dict(x=np.clip(num("temp"), 0, 100), tr=lambda v: v, unit="°C"),
       "pH": dict(x=num("pH"), tr=lambda v: v, unit="pH"),
       "conc": dict(x=np.where(Rr.st_conc.values == "obs", num("conc"), np.nan), tr=lambda v: v, unit="log10 µM"),
       "crowder": dict(x=np.log10(1 + crowd), tr=lambda v: np.log10(1 + v), unit="% crowder")}
# 候选水平：任务指定的 10 个（directive=True）+ rs80 网格里其余的水平；existing = 已有 GIF 的六条线
EXISTING = {("length", 400.0), ("NaCl", 150.0), ("temp", 22.0), ("pH", 7.5), ("conc", 1.0), ("crowder", 0.5)}
DIRECTIVE = {("NaCl", 50.0), ("NaCl", 300.0), ("pH", 6.0), ("pH", 8.0), ("temp", 4.0), ("temp", 37.0), ("length", 200.0), ("length", 800.0), ("conc", 0.0), ("conc", 2.0)}
GRID = {"length": [50, 100, 200, 400, 800], "NaCl": [0, 50, 150, 300, 500], "temp": [4, 22, 37, 60], "pH": [5.0, 6.5, 7.5, 8.5], "conc": [0.5, 1.0, 1.5, 2.0, 2.5], "crowder": [0.5, 2, 5, 10]}
ALL = []
for cov in COV:
    for lv in sorted({float(v) for v in GRID[cov]} | {l for c_, l in DIRECTIVE if c_ == cov}): ALL.append((cov, lv))
fmtlv = {"length": lambda v: f"{v:g} aa", "NaCl": lambda v: f"{v:g} mM", "temp": lambda v: f"{v:g} °C", "pH": lambda v: f"pH {v:g}", "conc": lambda v: f"{10 ** v:.3g} µM", "crowder": lambda v: f"{v:g} %"}
ttl = {"length": "Iso-length", "NaCl": "Iso-NaCl", "temp": "Iso-temperature", "pH": "Iso-pH", "conc": "Iso-concentration", "crowder": "Iso-crowder"}
def line_id(cov, lv):
    if cov == "length": return f"len{lv:g}"
    if cov == "NaCl": return f"nacl{lv:g}"
    if cov == "temp": return f"temp{lv:g}"
    if cov == "pH": return "ph" + f"{lv:g}".replace(".", "p")
    if cov == "conc": um = 10 ** lv; return "conc" + (f"{um:.2g}".replace(".", "p") if um < 10 else f"{um:.0f}")                      # 1 µM → conc1，3.2 µM → conc3p2，10 µM → conc10，32 µM → conc32，100 µM → conc100
    return f"crowd{lv:g}".replace(".", "p")
line_id2 = line_id
t32 = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
Gt, Ut = t32(G), t32(U)
def smooth(xv):
    """核平滑：质量 m、均值 mu、局部标准差 sd（σ = SIG）；与 rs80 相同。"""
    ok = ~np.isnan(xv); xt = t32(np.where(ok, xv, 0.0)); okt = t32(ok.astype(float)); m = torch.zeros(len(G), device=dev); s1 = torch.zeros_like(m); s2 = torch.zeros_like(m)
    for i in range(0, len(G), 4096):
        d2 = torch.cdist(Gt[i:i + 4096], Ut) ** 2; w = torch.exp(-d2 / (2 * SIG ** 2)) * okt[None, :]; m[i:i + 4096] = w.sum(1); s1[i:i + 4096] = (w * xt[None, :]).sum(1); s2[i:i + 4096] = (w * xt[None, :] ** 2).sum(1)
    m, s1, s2 = m.cpu().numpy(), s1.cpu().numpy(), s2.cpu().numpy(); mu = s1 / np.maximum(m, 1e-9); sd = np.sqrt(np.maximum(s2 / np.maximum(m, 1e-9) - mu ** 2, 0)); return m.reshape(NE, NE), mu.reshape(NE, NE), sd.reshape(NE, NE)
tree = cKDTree(U); dnn = tree.query(G)[0].reshape(NE, NE) / cell; yb = np.where(yA >= 0, yA, np.nan); paper = Rr.paper.astype(str).values; prot = Rr.uniprot.fillna("").astype(str).values
Rint = lambda P: RGI((ey, ex), Rgrid, bounds_error=False, fill_value=None)(np.stack([P[:, 1], P[:, 0]], 1))

# ================= A. 候选评估
fields = {}
for name in COV:
    xv = COV[name]["x"]; m, mu, sd = smooth(xv); gsd = float(np.nanstd(xv)); sup = (m >= 3.0) & (dnn <= 1.5); fields[name] = (m, mu, sd, sup, gsd); log("场", name, "全局 SD", round(gsd, 3), "支撑格点", int(sup.sum()))
rows = []; lines = {}; fig_, ax_ = plt.subplots()
for cov, lv in ALL:
    m, mu, sd, sup, gsd = fields[cov]; g0 = COV[cov]["tr"](lv); ax_.clear(); cs_ = ax_.contour(GX, GY, np.ma.masked_where(~sup, mu), levels=[g0]); segs = [np.asarray(s_) for s_ in cs_.allsegs[0] if len(s_) > 4]
    row = dict(id=line_id2(cov, lv), cov=cov, level=float(lv), unit=COV[cov]["unit"], label=f"{ttl[cov]} {fmtlv[cov](lv)}", existing=(cov, float(lv)) in EXISTING, directive=(cov, float(lv)) in DIRECTIVE)
    if not segs:
        row.update(status="no supported contour", score=0.0, passes=False, why="no segment inside the data-supported area"); rows.append(row); log(row["id"], "没有等值线"); continue
    pts = np.concatenate(segs); arc = float(sum(np.linalg.norm(np.diff(s_, axis=0), axis=1).sum() for s_ in segs))
    sd_at = RGI((ey, ex), sd, bounds_error=False, fill_value=None)(np.stack([pts[:, 1], pts[:, 0]], 1)); pur = float(np.median(sd_at) / gsd); r = Rint(pts)
    idx = tree.query_ball_point(pts, r=R_NEAR); near = np.unique(np.concatenate([np.asarray(i, int) for i in idx if len(i)])) if len(idx) else np.array([], int)
    n1 = int((yb[near] == 1).sum()) if len(near) else 0; n0 = int((yb[near] == 0).sum()) if len(near) else 0; npap = int(len(set(paper[near]))) if len(near) else 0; npro = int(len(set(prot[near]) - {""})) if len(near) else 0
    cross = 0
    for s_ in segs: rr = Rint(s_); cross += int(np.sum(np.diff(np.sign(rr)) != 0))
    score = (1 - min(pur, 1.0)) * min(arc, 6.0) / 6.0 * float(np.tanh(np.std(r))) * np.log1p(npap) if (arc >= 1.0 and n1 >= 10 and n0 >= 10) else 0.0
    why = []
    if arc < 1.0: why.append(f"line too short ({arc:.2f} SD < 1.0)")
    if n1 < 10 or n0 < 10: why.append(f"too few records near the line (y=1 {n1}, y=0 {n0}; need ≥ 10 each)")
    if pur >= 1.0: why.append(f"not more homogeneous than the whole data (purity {pur:.2f} ≥ 1)")
    if cross < 1: why.append("the y field does not change sign along the line")
    if npap < 5: why.append(f"only {npap} papers near the line (need ≥ 5)")
    passes = score > 0 and cross >= 1 and npap >= 5
    row.update(status="ok", arc=arc, purity=pur, r_std=float(np.std(r)), r_p10=float(np.percentile(r, 10)), r_p90=float(np.percentile(r, 90)), crossings=cross, n_near=int(len(near)), n_y1=n1, n_y0=n0, n_paper=npap, n_prot=npro, score=float(score), n_segments=len(segs), passes=bool(passes), why="; ".join(why) if why else "passes")
    rows.append(row); log(row["id"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k in ("arc", "purity", "crossings", "n_y1", "n_y0", "n_paper", "score", "passes")})
    # 起点：沿线（所有片段）等弧长取 NSTART 个（与 rs80 相同）
    P_ = []; A_ = []; offs = 0.0
    for s_ in segs:
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(s_, axis=0), axis=1))]; n = max(int(d[-1] / 0.02), 2); tt = np.linspace(0, d[-1], n); P_.append(np.stack([np.interp(tt, d, s_[:, 0]), np.interp(tt, d, s_[:, 1])], 1)); A_.append(offs + tt); offs += d[-1] + 0.3
    P_ = np.concatenate(P_); A_ = np.concatenate(A_); sel = np.linspace(0, len(P_) - 1, NSTART).astype(int); lines[row["id"]] = dict(pts=P_, arcpos=A_, S0=P_[sel], Rline=Rint(P_), g0=g0)
plt.close(fig_)
# 与 rs80_table.tsv 核对已有水平的数字
try:
    old = pd.read_csv(f"{D}/rs80_table.tsv", sep="\t"); chk = []
    for _, o in old.iterrows():
        mine = [x for x in rows if x["cov"] == o["cov"] and abs(x["level"] - float(o["level"])) < 1e-9 and x["status"] == "ok"]
        if mine: chk.append(dict(id=mine[0]["id"], d_arc=abs(mine[0]["arc"] - o["arc"]), d_purity=abs(mine[0]["purity"] - o["purity"]), d_n_near=abs(mine[0]["n_near"] - o["n_near"]), d_score=abs(mine[0]["score"] - o["score"])))
    log("与 rs80_table 核对：最大差 arc", max(c["d_arc"] for c in chk), "purity", max(c["d_purity"] for c in chk), "n_near", max(c["d_n_near"] for c in chk), "score", max(c["d_score"] for c in chk), "（行数", len(chk), "）")
    verify = dict(rows=len(chk), max_abs_diff=dict(arc=max(c["d_arc"] for c in chk), purity=max(c["d_purity"] for c in chk), n_near=max(c["d_n_near"] for c in chk), score=max(c["d_score"] for c in chk)))
except Exception as e: verify = dict(error=str(e))
cand_pass = sorted([x for x in rows if x["passes"] and not x["existing"]], key=lambda x: x["purity"]); chosen = [x["id"] for x in cand_pass[:MAXNEW]]
for x in rows: x["selected"] = x["id"] in chosen
json.dump(dict(rule="pass = rs80 score > 0 (purity < 1, arc >= 1.0 SD, >= 10 records of each label within 0.12 SD of the line) AND r changes sign along the line AND >= 5 papers near the line; new lines ranked by purity (lower = more homogeneous), at most 8", verify_against_rs80_table=verify, selected=chosen, candidates=rows), open(f"{D}/gd_iso_candidates.json", "w"), indent=1, default=str)
log("通过且未做过的线", [(x["id"], round(x["purity"], 2)) for x in cand_pass], "→ 选", chosen)
if PART == "cand": raise SystemExit(0)

# ================= B. HMC（rs95 的同一设定）
from torchebm.core import BaseModel
from torchebm.samplers import HamiltonianMonteCarlo
K = 8; TEMP = float(os.environ.get("T", "0.5")); W = float(os.environ.get("W", "4")); NPROP = int(os.environ.get("NPROP", "400")); EPS = 0.05; LEAP = 10
lo_t = torch.as_tensor([ex[0], ey[0]], dtype=torch.float32); hi_t = torch.as_tensor([ex[-1], ey[-1]], dtype=torch.float32); treeR = cKDTree(U)
E = {k: torch.as_tensor(A[f"g_E{k}"], dtype=torch.float32)[None, None] for k in ("1", "0", "n")}
def samp(field, X): g = (X - lo_t) / (hi_t - lo_t) * 2 - 1; return F.grid_sample(field, g[None, :, None, :], mode="bilinear", padding_mode="border", align_corners=True)[0, 0, :, 0]
class ArcEnergy(BaseModel):
    def __init__(self, s, Ug, **kw): super().__init__(prediction_type="energy", **kw); self.register_buffer("s", torch.as_tensor(s, dtype=torch.float32)); self.register_buffer("Ug", torch.as_tensor(Ug, dtype=torch.float32))
    def forward(self, x, y=None, **kw):
        z = x[:, 0]; zc = torch.clamp(z, float(self.s[0]), float(self.s[-1])); i = torch.clamp(torch.bucketize(zc, self.s) - 1, 0, len(self.s) - 2); t = (zc - self.s[i]) / (self.s[i + 1] - self.s[i] + 1e-9)
        return self.Ug[i] * (1 - t) + self.Ug[i + 1] * t + 1e4 * (z - zc) ** 2
summ = {}
for lid in chosen:
    L = lines[lid]; row = [x for x in rows if x["id"] == lid][0]; pts = L["pts"].astype(np.float64); arc = L["arcpos"].astype(np.float64); S0 = L["S0"].astype(np.float64)
    brk = np.r_[False, np.diff(arc) > 0.2]; seg = np.cumsum(brk); nseg = int(seg.max()) + 1; ds = np.r_[np.diff(arc), 0.0]; ds = np.where(np.r_[brk[1:], True], np.median(ds[ds > 0]), ds); bins = np.floor(arc / 0.25).astype(int)
    treeL = cKDTree(pts); X0 = np.repeat(S0, K, axis=0); i0 = treeL.query(X0)[1]; seg0 = seg[i0]; arc0 = arc[i0]; show = np.arange(0, len(X0), K); out = dict(pts=pts, arc=arc, seg=seg, S0=S0, show=show, X0=X0.astype(np.float32)); res = {}
    for dname, ylab in (("to_y1", 1), ("to_y0", 0)):
        En_ = samp(E["n"], torch.as_tensor(pts, dtype=torch.float32)).numpy().astype(np.float64); Ey_ = samp(E[str(ylab)], torch.as_tensor(pts, dtype=torch.float32)).numpy().astype(np.float64); EwT = (En_ + W * (Ey_ - En_)) / TEMP
        ref = np.exp(-(EwT - EwT.min())) * ds; ref /= ref.sum(); paths = np.zeros((len(X0), NPROP + 1, 2), dtype=np.float32); accs = []; t0 = time.time()
        for sg in range(nseg):
            idx = np.where(seg == sg)[0]; m0 = np.where(seg0 == sg)[0]
            if len(m0) == 0: continue
            s_ = arc[idx]; model = ArcEnergy(s_, EwT[idx]); sam = HamiltonianMonteCarlo(model=model, step_size=EPS, n_leapfrog_steps=LEAP); x0 = torch.as_tensor(arc0[m0][:, None], dtype=torch.float32)
            Xs, diag = sam.sample(x=x0, n_steps=NPROP, return_trajectory=True, return_diagnostics=True); Xs = Xs.detach().numpy()[:, :, 0].astype(np.float64)                      # [链, NPROP]
            Xs = np.concatenate([arc0[m0][:, None], Xs], 1); Xs = np.clip(Xs, s_[0], s_[-1])
            paths[m0, :, 0] = np.interp(Xs, s_, pts[idx, 0]); paths[m0, :, 1] = np.interp(Xs, s_, pts[idx, 1])
            if isinstance(diag, dict) and "acceptance_rate" in diag: accs.append(float(torch.as_tensor(diag["acceptance_rate"]).float().mean()))
        Xn = paths[:, -1].astype(np.float64); ia = treeL.query(Xn)[1]; tot = 0.0
        for sg in range(nseg):
            m0 = seg0 == sg
            if m0.sum() == 0: continue
            ins = np.where(seg == sg)[0]; ub = np.unique(bins[ins]); pr = np.array([ref[ins][bins[ins] == b].sum() for b in ub]); pr = pr / pr.sum(); h = np.array([np.sum((seg[ia[m0]] == sg) & (bins[ia[m0]] == b)) for b in ub]) / m0.sum(); tot += 0.5 * (np.abs(h - pr).sum() + (1 - h.sum())) * m0.sum()
        res[dname] = dict(TV_seg=float(tot / len(Xn)), accept_rate=float(np.mean(accs)) if accs else None, arc_move_median=float(np.median(np.abs(arc[ia] - arc0))), off_support=float((treeR.query(Xn)[0] / cell > 1.2).mean()), n_segments=nseg, seconds=float(time.time() - t0))
        out[f"path_{dname}"] = paths[show]; out[f"end_{dname}"] = paths[:, -1]; log(lid, dname, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[dname].items()})
    summ[lid] = dict(level=float(row["level"]), unit=row["unit"], n_line_points=int(len(pts)), proposals=NPROP, eps=EPS, leapfrog=LEAP, T=TEMP, w=W, methods=res); np.savez_compressed(f"{D}/gd_iso_{lid}.npz", **out)
    json.dump(summ, open(f"{D}/gd_iso_summary.json", "w"), indent=1, default=str)
log("完成", dict(selected=chosen, threads=torch.get_num_threads())); print("DONE", flush=True)
