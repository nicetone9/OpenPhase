"""rs134（本机，pixi 环境 ~/pixi/figs；由 rs130 调用）：Guidance tab 里 4 个"主要结果"的交互地图数据（用户 11/13：结果里的点要能悬停看到起点、终点、端点参考信息、序列变化和预测）。
每个点都带：起点与终点（地图坐标）、该处 score 场的估计（r(u) = log p(u|1) − log p(u|0) 与 p(y=1|u) = σ(r + log(π1/π0))，π1/π0 = 6,613/4,050）、最近 3 条记录（行号 + 距离，页面用 explorer 自己的记录信息显示蛋白、论文、标签、条件）。
  iso    等值线 NaCl 150 mM：rs95 的一维弧长 HMC（60 个等弧长起点 × 8 条链，每个方向 480 条链，400 个提议）；起点 = 线上的点，终点 = 链最后的位置
  region 多约束区域 C3（22 ± 4 °C、pH 7.5 ± 0.35、NaCl 150 mM、10 µM；rs82_joint.npz 的 mask_2）：区域内的网格精确采样（概率 ∝ exp(−E_w/T)，w = 4，T = 0.5，格内均匀抖动），每个方向 500 个独立样本（没有起点终点）
  yc     (y,c) 集合 生理格：rs70 的 Langevin 链（1,200 步），起点 = 起始标签的记录，终点 = 链最后的位置；对照 = 无引导
  pts    地图点→序列 tau × 10 µM：rs107 的 12 个点，起点 = 锚点序列在这个点附近最近的记录，终点 = 地图点；序列变化 = 解码答案相对锚点的替换；序列级预测 = FINCHES ε（rs133，锚点 vs 答案）
坐标：采样代码的“标准化 t-SNE”（rs22 的 U_all）就是 rs_coords 的 (xy − mean)/std，页面用 rs05 的归一化（explorer 坐标），这里统一换算。
输入放在 guidance_assets/src/（别的会话的小数组：rs22_fields、rs25_global_s016、rs95_NaCl、rs70_physio、rs82_joint 的拷贝，以及 rs133 的输出）；rs107 解码表在 figs/。"""
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
SRC, FIGS, OUT = ROOT / "guidance_assets" / "src", ROOT / "figs", ROOT / "out"
W_GUIDE, TEMP = 4.0, 0.5


def rnd(v, n=3):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), n)


class Field:
    """score 场的估计、最近记录与坐标换算。"""

    def __init__(self):
        Z = np.load(SRC / "rs22_fields.npz")
        self.U, self.y = Z["U_all"].astype(float), Z["y_all"]
        A = np.load(SRC / "rs25_global_s016.npz")
        self.ex, self.ey = A["g_ex"].astype(float), A["g_ey"].astype(float)
        self.dx = self.ex[1] - self.ex[0]
        self.E = {k: A[f"g_E{k}"].astype(float) for k in "10n"}
        norm = lambda E: -(E - E.min()) - np.log(np.exp(-(E - E.min())).sum() * self.dx * self.dx)
        self.R = norm(self.E["1"]) - norm(self.E["0"])
        self._r = RegularGridInterpolator((self.ey, self.ex), self.R, bounds_error=False, fill_value=None)
        self.prior = float(np.log((self.y == 1).sum() / (self.y == 0).sum()))
        self.tree = cKDTree(self.U)
        co = list(csv.DictReader(open(OUT / "rs_coords.tsv"), delimiter="\t"))
        xs = np.array([float(c["tsne_full_x"]) for c in co]); ys = np.array([float(c["tsne_full_y"]) for c in co])
        self.cx, self.cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
        self.s = max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.04
        self.mx, self.my, self.sd = xs.mean(), ys.mean(), np.nanstd(np.c_[xs, ys])
        self.scale = self.sd / self.s                                  # 标准化坐标 1 个单位 = explorer 坐标 scale

    def r(self, P):
        return self._r(np.c_[P[:, 1], P[:, 0]])

    def p(self, r):
        return 1 / (1 + np.exp(-(r + self.prior)))

    def to_exp(self, P):
        P = np.asarray(P, float).reshape(-1, 2)
        return np.c_[((P[:, 0] * self.sd + self.mx) - self.cx) / self.s + 0.5, 1 - (((P[:, 1] * self.sd + self.my) - self.cy) / self.s + 0.5)]

    def near(self, P, k=3):
        d, i = self.tree.query(np.asarray(P, float).reshape(-1, 2), k=k)
        return d, i


def flat(a, n=4):
    return [round(float(v), n) for v in np.asarray(a).ravel()]


def pack(F, gid, label, color, S, sid, Ee, **extra):
    """一组链：S = 唯一起点（标准化坐标），sid = 每条链的起点序号，Ee = 链终点。"""
    rs, re_ = F.r(S), F.r(Ee)
    ds, ns = F.near(S)
    de, ne = F.near(Ee)
    g = dict(id=gid, label=label, color=color, n=int(len(Ee)), S=flat(F.to_exp(S)), sid=[int(v) for v in sid], E=flat(F.to_exp(Ee)),
             rs=flat(rs, 2), re=flat(re_, 2), ps=flat(F.p(rs), 3), pe=flat(F.p(re_), 3),
             ns=[int(v) for v in ns.ravel()], nsd=flat(ds, 3), ne=[int(v) for v in ne.ravel()], ned=flat(de, 3))
    g.update(extra)
    return g


def overlap(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else None


def build_iso(F, recs):
    Z = np.load(SRC / "rs95_NaCl.npz")
    S95 = json.load(open(SRC / "rs95_summary.json"))["NaCl"]
    pts, arc, seg, S0, X0 = Z["pts"].astype(float), Z["arc"].astype(float), Z["seg"], Z["S0"].astype(float), Z["X0"].astype(float)
    sid = np.repeat(np.arange(len(S0)), 8)
    assert np.allclose(X0, S0[sid], atol=1e-4)
    lt = cKDTree(pts)
    groups = []
    for key, lab, col in (("to_y1", "guided to y = 1", "y1"), ("to_y0", "guided to y = 0", "y0")):
        Ee = Z[f"end_{key}"].astype(float)
        i0, i1 = lt.query(S0)[1], lt.query(Ee)[1]
        m = S95["methods"][key]
        g = pack(F, key, lab, col, S0, sid, Ee, arc0=flat(arc[i0], 1), seg0=[int(v) for v in seg[i0]], arc1=flat(arc[i1], 1), seg1=[int(v) for v in seg[i1]],
                 stats=dict(tv=rnd(m["TV_seg"], 2), acc=rnd(m["accept_rate"], 2), move=rnd(m["arc_move_median"], 1), off=rnd(m["off_support"], 3), pieces=m["n_segments"]))
        groups.append(g)
    prot = lambda g: [recs["uni"][j] or recs["name"][j] for j in g["ne"]]
    jac = overlap(prot(groups[0]), prot(groups[1]))
    pieces = [flat(F.to_exp(pts[seg == k])) for k in sorted(set(seg.tolist()))]
    return dict(id="iso_nacl150", kind="iso", title="Iso-line: NaCl = 150 mM", row="iso-nacl", line=pieces, groups=groups,
                facts=dict(n_chains=480, proposals=S95["proposals"], eps=S95["eps"], leapfrog=S95["leapfrog"], T=S95["T"], w=S95["w"], line_points=S95["n_line_points"], pieces=len(pieces),
                           protein_overlap=rnd(jac, 2), r_end_median=[rnd(np.median(F.r(Z["end_to_y1"].astype(float))), 2), rnd(np.median(F.r(Z["end_to_y0"].astype(float))), 2)]))


def build_region(F, recs):
    J = np.load(SRC / "rs82_joint.npz")
    mask = J["mask_2"].astype(bool)
    lab, ncomp = ndi.label(mask)
    rng = np.random.RandomState(134)
    N = 500
    groups = []
    for y, name, col in ((1, "to_y1", "y1"), (0, "to_y0", "y0")):
        Ew = F.E["n"] + W_GUIDE * (F.E[str(y)] - F.E["n"])
        lp = -(Ew - Ew[mask].min()) / TEMP
        p = np.where(mask, np.exp(lp), 0.0)
        p /= p.sum()
        cell = rng.choice(mask.size, size=N, p=p.ravel())
        ii, jj = np.divmod(cell, mask.shape[1])
        X = np.c_[F.ex[jj] + rng.uniform(-F.dx / 2, F.dx / 2, N), F.ey[ii] + rng.uniform(-F.dx / 2, F.dx / 2, N)]
        r = F.r(X)
        d, n = F.near(X)
        groups.append(dict(id=name, label=f"exact samples guided to y = {y}", color=col, n=N, E=flat(F.to_exp(X)), re=flat(r, 2), pe=flat(F.p(r), 3), ne=[int(v) for v in n.ravel()], ned=flat(d, 3),
                           piece=[int(lab[a, b]) for a, b in zip(ii, jj)]))
    ci, cj = np.nonzero(mask)
    cells = flat(F.to_exp(np.c_[F.ex[cj], F.ey[ci]]))
    # records inside the region
    ri = np.clip(np.round((F.U[:, 1] - F.ey[0]) / F.dx).astype(int), 0, mask.shape[0] - 1)
    rj = np.clip(np.round((F.U[:, 0] - F.ex[0]) / F.dx).astype(int), 0, mask.shape[1] - 1)
    inside = mask[ri, rj]
    n1, n0 = int((inside & (F.y == 1)).sum()), int((inside & (F.y == 0)).sum())
    papers = len({recs["paper"][i] for i in np.nonzero(inside)[0]})
    prots = len({recs["uni"][i] or recs["name"][i] for i in np.nonzero(inside)[0]})
    r1, r0 = F.r(np.array([[0.0, 0.0]])) * 0, None
    return dict(id="region_c3", kind="region", title="Region: 22 °C, pH 7.5, 150 mM NaCl, 10 µM", row=None, cells=cells, cell_size=round(float(F.dx * F.scale), 5), groups=groups,
                facts=dict(cells=int(mask.sum()), pieces=int(ncomp), area_frac=rnd(mask.sum() / (F.ex.size * F.ey.size), 3), records=n1 + n0, n_y1=n1, n_y0=n0, papers=papers, proteins=prots, samples=N, w=W_GUIDE, T=TEMP,
                           r_median=[rnd(np.median(np.array(groups[0]["re"])), 2), rnd(np.median(np.array(groups[1]["re"])), 2)]))


def build_yc(F, recs):
    Z = np.load(SRC / "rs70_physio.npz")
    S70 = json.load(open(FIGS / "rs70_summary.json"))["physio"]
    K = 16
    groups = []
    for a, idx, lab, col in (("0to1", Z["idx0"], "y = 0 records, guided to y = 1", "y1"), ("1to0", Z["idx1"], "y = 1 records, guided to y = 0", "y0")):
        S = F.U[idx]
        sid = np.repeat(np.arange(len(idx)), K)
        Ee = Z[f"{a}_end_guided"].astype(float)
        assert len(Ee) == len(sid)
        m = S70[a]["guided"]
        groups.append(pack(F, a, lab, col, S, sid, Ee, srec=[int(v) for v in idx],
                           control=dict(E=flat(F.to_exp(Z[f"{a}_end_none"].astype(float))), re=flat(F.r(Z[f"{a}_end_none"].astype(float)), 2)),
                           stats=dict(dr=rnd(m["dr_median"], 2), toward=rnd(m["frac_dr_toward"], 3), arr=rnd(m["arrived_0p10"], 3), arr_none=rnd(S70[a]["none"]["arrived_0p10"], 3))))
    return dict(id="yc_physio", kind="yc", title="(y, c) set: physiological cell", row="yc-physio", groups=groups,
                facts=dict(n1=S70["n1"], n0=S70["n0"], seq=S70["n_seq"], papers=S70["n_paper"], desc=S70["desc"], steps=1200, w=W_GUIDE, T=TEMP, K=K, dist=rnd(S70["map_dist_medians"], 2)))


def build_pts(F, recs):
    combo = "conc 10 µM × tau"
    rows = [r for r in csv.DictReader(open(FIGS / "rs107_points_decoded.tsv"), delimiter="\t") if r["combo"] == combo]
    fin = {(r["combo"], r["label"], int(r["point"])): r for r in csv.DictReader(open(SRC / "gd_finches_points.tsv"), delimiter="\t")}
    seq_row = np.array([int(float(x)) if x not in ("", None) else -1 for x in recs["seq_row"]])
    items = []
    for r in rows:
        q = np.array([[float(r["q1"]), float(r["q2"])]])
        rr = F.r(q)
        d, n = F.near(q)
        cand = np.nonzero(seq_row == int(r["anchor"]))[0]
        a = int(cand[np.argmin(np.linalg.norm(F.U[cand] - q, axis=1))]) if len(cand) else -1
        da = float(np.linalg.norm(F.U[a] - q[0])) if a >= 0 else None
        f = fin.get((combo, r["label"], int(r["point"])))
        ff = None
        if f:
            g = lambda k: (float(f[k]) if f.get(k) not in ("", None) else None)
            ff = {sc: {ffd: dict(src=rnd(g(f"{ffd}_{sc}_src"), 3), dec=rnd(g(f"{ffd}_{sc}_dec"), 3), rnd=rnd(g(f"{ffd}_{sc}_rnd"), 3)) for ffd in ("mpipi", "calvados")} for sc in ("full", "idr") if g(f"mpipi_{sc}_src") is not None}
            ff["n_idr"] = int(float(f["n_idr_residues"])); ff["idr_sites"] = int(float(f["n_idr_sites"]))
        items.append(dict(y=1 if r["label"] == "y1" else 0, i=int(r["point"]), x=flat(F.to_exp(q))[0], yy=flat(F.to_exp(q))[1], r=rnd(rr[0], 2), p=rnd(F.p(rr)[0], 3), ne=[int(v) for v in n[0]], ned=flat(d[0], 3),
                          start=a, start_d=rnd(da, 3), anchor=int(r["anchor"]), len=int(r["length"]), k=int(r["kstar"]), dsrc=rnd(r["D_src"], 4), dk=rnd(r["D_k"], 4),
                          edits=[[e[0], int(e[1:-1]), e[-1]] for e in r["edits"].split(";") if e], fin=ff))
    return dict(id="pts_tau10", kind="pts", title="Map point → sequence: tau region ∩ 10 µM line", row="pts-conc_10_m_tau", items=items,
                facts=dict(combo=combo, n_k0=sum(1 for i in items if i["k"] == 0), n_new=sum(1 for i in items if i["k"] > 0)))


def build_results():
    F = Field()
    recs = {k: [] for k in ("name", "uni", "paper", "seq_row")}
    for r in csv.DictReader(open(OUT / "rs_plot_table.tsv"), delimiter="\t"):
        recs["name"].append(r["protein_name"]); recs["uni"].append(r["uniprot"]); recs["paper"].append(r["paper"]); recs["seq_row"].append(r["seq_row"])
    maps = [build_iso(F, recs), build_region(F, recs), build_yc(F, recs), build_pts(F, recs)]
    S = json.load(open(SRC / "gd_finches_points_summary.json"))
    return dict(maps=maps, prior=rnd(float(np.exp(F.prior)), 3), n_records=int(len(F.U)), finches=S)


if __name__ == "__main__":
    out = build_results()
    s = json.dumps(out, separators=(",", ":"))
    print(f"{len(s) / 1e3:.0f} KB;", {m["id"]: len(json.dumps(m)) // 1000 for m in out["maps"]})
    for m in out["maps"]:
        print(m["id"], m.get("facts"))
