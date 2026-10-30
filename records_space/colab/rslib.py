"""records 空间的块距离（numpy / pandas，Colab 上运行；本机没有 numpy，不在本机跑）。

一条 record = 蛋白（中心）+ 核酸（条件）+ 物理化学条件 + 缓冲液/添加剂。读数（label、csat、形态）与来源（source、paper）不进嵌入。

各类数据用各自合适的表示，再合成一个"记录间距离"，最后降维：
  P 蛋白序列      ESMC-600M 第 18 层残基均值（已有 1,299 条 + 新算 39 条）→ z-score → PCA(64) → 余弦距离
  N 核酸          是否有核酸（"无"是合法取值，不是缺失）、RNA/DNA、名称类别（Jaccard）、长度、浓度、核酸:蛋白比、序列特征、折叠特征
  C 物理化学条件  蛋白浓度(log10 µM)、pH、温度、盐（MgCl2/NaCl/KCl 合成一个维度）、拥挤剂（7 个合成一个维度），各自按稳健范围缩放到 [0,1]
  B 缓冲液/添加剂 多热编码上的 Jaccard 距离

缺失当作一种状态（M+，默认）：某维度上两条都观测到 → |Δ|；恰有一条缺失 → LAM（0.5，等于"全库 95% 分位距离"的一半）；两条都缺失 → 0。
对照（M−，只比共同观测的维度）：不共同观测的维度不参与平均。
缺失 ≠ 不适用：无核酸的记录，核酸数值特征是"不适用"（N 块里"有核酸 vs 无核酸"距离 = 1），不记缺失。
"""
import numpy as np
import pandas as pd

LAM = 0.5
BLOCK_W = dict(P=0.40, N=0.15, C=0.35, B=0.10)
SHORT = ["conc", "pH", "temp", "MgCl2", "NaCl", "KCl", "PEG1k", "PEG6k", "PEG20k", "Ficoll", "Dex40", "Dex70", "Glycerol"]
SALT = ["MgCl2", "NaCl", "KCl"]
CROWD = ["PEG1k", "PEG6k", "PEG20k", "Ficoll", "Dex40", "Dex70", "Glycerol"]
NA_FLAGS = ["polyU", "polyA", "polyC", "polyG", "tRNA", "rRNA", "total_RNA", "mRNA", "G4", "ssDNA", "dsDNA", "plasmid_or_long_DNA", "methylated"]
BUF = ["Tris", "HEPES", "phosphate", "goodsMES", "citrate", "acetate", "imidazole", "other_buf"]
ADD = ["reducing", "chelator", "glycerol", "nucleotide", "denaturant", "hexanediol", "detergent", "protein_add", "polyanion", "metal", "dye_probe", "other_add"]
TEMP_STATE_PEN = 0.2          # 温度：一条是实测、一条只写"室温"（记 22 °C）时额外加的距离


# ----------------------------------------------------------------------------- 读表
def load_records(path):
    df = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""], low_memory=False)
    return df


def _q(x, qs):
    x = x[~np.isnan(x)]
    return np.quantile(x, qs) if len(x) else np.array([0.0] * len(qs))


def _scaled(x, mode="range", obs_mask=None):
    """把一列缩放成 v = x / s（只用差的绝对值，不需要平移）。
    range：s = q95 − q05（仅观测值）；pos：x 先取 log10(1+x)，s = 非零值的 q95。"""
    x = np.asarray(x, dtype=np.float64).copy()
    if mode == "pos":
        x = np.log10(1.0 + np.clip(x, 0, None))
        nz = x[(~np.isnan(x)) & (x > 0)]
        s = np.quantile(nz, 0.95) if len(nz) else 1.0
    else:
        base = x if obs_mask is None else np.where(obs_mask, x, np.nan)
        lo, hi = _q(base, [0.05, 0.95])
        s = hi - lo
    s = s if s > 1e-9 else 1.0
    return (x / s).astype(np.float32), float(s)


def build_inputs(df, esmc, seq_df):
    """df：rs_records.tsv；esmc：(n_seq, 1152) 与 seq_df（rs_sequences.tsv）行序一致。返回 numpy 输入字典。"""
    N = len(df)
    inp = dict(N=N, scales={})
    # ---- P：蛋白
    from sklearn.decomposition import PCA
    E = np.asarray(esmc, dtype=np.float32)
    e_mu, e_sd = E.mean(0), E.std(0) + 1e-6
    Z = (E - e_mu) / e_sd
    pca = PCA(n_components=min(64, Z.shape[0] - 1), random_state=0)
    Z = pca.fit_transform(Z).astype(np.float32)
    z_mean = Z.mean(0)
    Z -= z_mean
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    # 供"试新实验"面板把新序列的 ESMC 嵌入投到同一个空间用（只增加保存的状态，不改变任何数值）
    inp["Zn"], inp["esmc_norm"], inp["pca"], inp["z_mean"] = Zn, dict(mu=e_mu, sd=e_sd), pca, z_mean
    S = (1.0 - Zn @ Zn.T).astype(np.float32)
    np.fill_diagonal(S, 0.0)
    S = np.clip(S, 0.0, None)
    pidx = df["seq_row"].to_numpy().astype(int)
    inp["pidx"] = pidx
    inp["S_seq"] = S
    inp["P_has"] = pidx >= 0
    # ---- C：条件
    def num(col):
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    st = {k: df[f"st_{k}"].to_numpy().astype(str) for k in SHORT}
    inp["scales"] = {}
    v, s = _scaled(num("conc"), "range", st["conc"] == "obs"); inp["conc"] = v; inp["scales"]["conc"] = s
    v, s = _scaled(num("pH"), "range", st["pH"] == "obs"); inp["pH"] = v; inp["scales"]["pH"] = s
    tm = num("temp")
    v, s = _scaled(tm, "range", st["temp"] == "obs")
    inp["temp"] = v; inp["scales"]["temp"] = s
    inp["temp_code"] = np.where(st["temp"] == "rt", 1, np.where(st["temp"] == "miss", 2, 0)).astype(np.int8)
    for grp, names in (("salt", SALT), ("crowd", CROWD)):
        mem = []
        for k in names:
            x = num(k)
            x = np.where(st[k] == "miss", np.nan, x)
            v, s = _scaled(x, "pos")
            v = np.where(st[k] == "miss", np.nan, v).astype(np.float32)
            mem.append(v); inp["scales"][k] = s
        inp[grp] = mem
    # ---- N：核酸
    cls = df["na_class"].to_numpy().astype(str)
    inp["na_cls"] = np.where(cls == "RNA", 1, np.where(cls == "DNA", 2, 0)).astype(np.int8)
    pres = inp["na_cls"] > 0

    def na_num(col, tf=None):
        x = num(col)
        if tf:
            x = tf(x)
        x = np.where(pres, x, np.nan)
        v, s = _scaled(x, "range")
        return v, s
    inp["na_len"], inp["scales"]["na_len"] = na_num("na_len_log10")
    inp["na_conc"], inp["scales"]["na_conc"] = na_num("na_conc_log10")
    inp["na_ratio"], inp["scales"]["na_ratio"] = na_num("na_ratio_log10")
    flags = np.stack([np.nan_to_num(pd.to_numeric(df[f"na_name_{k}"], errors="coerce").to_numpy(dtype=np.float32)) for k in NA_FLAGS], 1)
    inp["na_flags"] = np.where(pres[:, None], flags, 0.0).astype(np.float32)
    # 序列特征（GC、嘌呤、log10 最长同聚、G4 标志）与折叠特征（MFE/nt、配对比例）：分别标准化
    seqf = np.stack([num("na_gc"), num("na_purine"), np.log10(np.clip(num("na_maxrun"), 1, None)), num("na_g4")], 1)
    fold = np.stack([num("na_mfe_per_nt"), num("na_paired_frac")], 1)
    for nm, M in (("na_seq", seqf), ("na_fold", fold)):
        M = np.where(pres[:, None], M, np.nan)
        okrow = ~np.isnan(M).any(1)
        mu = np.nanmean(M[okrow], 0) if okrow.any() else np.zeros(M.shape[1])
        sd = np.nanstd(M[okrow], 0) + 1e-6 if okrow.any() else np.ones(M.shape[1])
        Zm = ((M - mu) / sd).astype(np.float32)
        Zm[~okrow] = np.nan
        # 缩放使得随机一对的距离 q95 ≈ 1
        rng = np.random.default_rng(0)
        idx = np.where(okrow)[0]
        if len(idx) > 10:
            a, b = rng.choice(idx, 20000), rng.choice(idx, 20000)
            dd = np.linalg.norm(Zm[a] - Zm[b], axis=1)
            sc = float(np.quantile(dd, 0.95))
        else:
            sc = 1.0
        inp[nm] = (Zm / max(sc, 1e-6)).astype(np.float32)
        inp["scales"][nm] = sc
        inp.setdefault("na_norm", {})[nm] = dict(mu=np.asarray(mu, dtype=np.float64), sd=np.asarray(sd, dtype=np.float64), sc=float(max(sc, 1e-6)))
    # ---- B：缓冲液与添加剂
    bm = np.stack([pd.to_numeric(df[f"buf_{k}"], errors="coerce").fillna(0).to_numpy(dtype=np.float32) for k in BUF], 1)
    bst = df["st_buffer"].to_numpy().astype(str)
    bm[bst == "other", BUF.index("other_buf")] = 1.0
    am = np.stack([pd.to_numeric(df[f"add_{k}"], errors="coerce").fillna(0).to_numpy(dtype=np.float32) for k in ADD], 1)
    ast = df["st_additive"].to_numpy().astype(str)
    inp["buf"], inp["buf_obs"] = bm, bst != "miss"
    inp["add"], inp["add_obs"] = am, ast != "miss"
    inp["bst"], inp["ast"] = bst, ast
    return inp


# ----------------------------------------------------------------------------- 维度距离
def _scal(a, i0, i1, lam=LAM):
    """一维数值：返回 (d_plus, d_obs, both)。d_obs 在不共同观测处为 0（M− 用）。"""
    ac = a[i0:i1, None]
    oc = ~np.isnan(ac)
    oa = ~np.isnan(a)[None, :]
    both = oc & oa
    diff = np.abs(ac - a[None, :])
    d_obs = np.minimum(np.where(both, diff, 0.0), 1.0).astype(np.float32)
    d_plus = np.where(both, d_obs, np.where(oc ^ oa, np.float32(lam), np.float32(0.0))).astype(np.float32)
    return d_plus, d_obs, both


def _group(members, i0, i1, lam=LAM):
    """多成员维度（盐、拥挤剂）：成员距离相加后截断到 1。"""
    dp = db = None
    anyb = None
    for a in members:
        p, o, b = _scal(a, i0, i1, lam)
        dp = p if dp is None else dp + p
        db = o if db is None else db + o
        anyb = b if anyb is None else (anyb | b)
    return np.minimum(dp, 1.0), np.minimum(db, 1.0), anyb


def _jac(A, i0, i1):
    """多热矩阵上的 Jaccard 距离（两个空集合 → 0）。"""
    a = A[i0:i1]
    inter = a @ A.T
    union = a.sum(1)[:, None] + A.sum(1)[None, :] - inter
    return np.where(union > 0, 1.0 - inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def _set_dim(M, obs, i0, i1, lam=LAM):
    """集合型维度（缓冲液/添加剂）：两条都观测 → Jaccard；恰一条缺失 → lam；都缺失 → 0。"""
    j = _jac(M, i0, i1)
    oc = obs[i0:i1, None]
    oa = obs[None, :]
    both = oc & oa
    d_obs = np.where(both, j, 0.0).astype(np.float32)
    d_plus = np.where(both, j, np.where(oc ^ oa, np.float32(lam), np.float32(0.0))).astype(np.float32)
    return d_plus, d_obs, both


# ----------------------------------------------------------------------------- 块距离
def block_P(inp, i0, i1, lam=LAM):
    pidx = inp["pidx"]
    S = inp["S_seq"]
    sc = inp["scales"]["P"]
    has = inp["P_has"]
    ii = pidx[i0:i1]
    safe_i, safe_a = np.where(ii >= 0, ii, 0), np.where(pidx >= 0, pidx, 0)
    d = np.minimum(S[safe_i][:, safe_a] / sc, 1.0).astype(np.float32)
    oc, oa = has[i0:i1, None], has[None, :]
    both = oc & oa
    d_obs = np.where(both, d, 0.0).astype(np.float32)
    d_plus = np.where(both, d, np.where(oc ^ oa, np.float32(lam), np.float32(0.0))).astype(np.float32)
    return d_plus, d_obs, both.astype(np.float32)


def block_C(inp, i0, i1, lam=LAM):
    dims = []
    for k in ("conc", "pH"):
        dims.append(_scal(inp[k], i0, i1, lam))
    p, o, b = _scal(inp["temp"], i0, i1, lam)
    tc = inp["temp_code"]
    pen = TEMP_STATE_PEN * (b & (tc[i0:i1, None] != tc[None, :])).astype(np.float32)
    dims.append((np.minimum(p + pen, 1.0), np.minimum(o + pen, 1.0), b))
    dims.append(_group(inp["salt"], i0, i1, lam))
    dims.append(_group(inp["crowd"], i0, i1, lam))
    K = len(dims)
    dp = sum(d[0] for d in dims) / K
    num = sum(d[1] * d[2] for d in dims)
    den = sum(d[2].astype(np.float32) for d in dims)
    dm = np.where(den > 0, num / np.maximum(den, 1.0), np.float32(0.5)).astype(np.float32)
    return dp.astype(np.float32), dm, (den / K).astype(np.float32)


def block_N(inp, i0, i1, lam=LAM):
    cls = inp["na_cls"]
    cc, ca = cls[i0:i1, None], cls[None, :]
    pres_c, pres_a = cc > 0, ca > 0
    both = pres_c & pres_a
    d_cls = (cc != ca).astype(np.float32)
    d_name = _jac(inp["na_flags"], i0, i1)
    dims = [(d_cls, d_cls, np.ones_like(d_cls, dtype=bool)), (d_name, d_name, np.ones_like(d_name, dtype=bool))]
    for k in ("na_len", "na_conc", "na_ratio"):
        dims.append(_scal(inp[k], i0, i1, lam))
    for k in ("na_seq", "na_fold"):
        Z = inp[k]
        zc = Z[i0:i1]
        okc = ~np.isnan(zc[:, 0])[:, None]
        oka = ~np.isnan(Z[:, 0])[None, :]
        bth = okc & oka
        zz = np.nan_to_num(Z)
        zzc = np.nan_to_num(zc)
        d2 = (zzc ** 2).sum(1)[:, None] + (zz ** 2).sum(1)[None, :] - 2.0 * zzc @ zz.T
        d = np.sqrt(np.clip(d2, 0.0, None))
        d_obs = np.where(bth, np.minimum(d, 1.0), 0.0).astype(np.float32)
        d_plus = np.where(bth, d_obs, np.where(okc ^ oka, np.float32(lam), np.float32(0.0))).astype(np.float32)
        dims.append((d_plus, d_obs, bth))
    K = len(dims)
    dp = sum(d[0] for d in dims) / K
    num = sum(d[1] * d[2] for d in dims)
    den = sum(d[2].astype(np.float32) for d in dims)
    dm = num / np.maximum(den, 1.0)
    one_none = pres_c ^ pres_a
    d_plus = np.where(both, dp, np.where(one_none, np.float32(1.0), np.float32(0.0))).astype(np.float32)
    d_minus = np.where(both, dm, np.where(one_none, np.float32(1.0), np.float32(0.0))).astype(np.float32)
    cov = np.where(both, den / K, np.float32(1.0)).astype(np.float32)
    return d_plus, d_minus, cov


def block_B(inp, i0, i1, lam=LAM):
    p1, o1, b1 = _set_dim(inp["buf"], inp["buf_obs"], i0, i1, lam)
    p2, o2, b2 = _set_dim(inp["add"], inp["add_obs"], i0, i1, lam)
    dp = (p1 + p2) / 2
    den = b1.astype(np.float32) + b2.astype(np.float32)
    dm = np.where(den > 0, (o1 * b1 + o2 * b2) / np.maximum(den, 1.0), np.float32(0.5)).astype(np.float32)
    return dp.astype(np.float32), dm, (den / 2).astype(np.float32)


def calibrate_P(inp, n_pairs=2_000_000, seed=0):
    """P 块的缩放：随机记录对（两条都有序列）的 S 的 95% 分位。"""
    rng = np.random.default_rng(seed)
    pidx = inp["pidx"]
    ok = np.where(pidx >= 0)[0]
    a, b = rng.choice(ok, n_pairs), rng.choice(ok, n_pairs)
    s = inp["S_seq"][pidx[a], pidx[b]]
    sc = float(np.quantile(s, 0.95))
    inp["scales"]["P"] = max(sc, 1e-6)
    return sc


def compute_lens_matrices(inp, chunk=1024, weights=None, verbose=True):
    """一次遍历算出四个透镜的 N×N 距离（float32）：
    full（P+N+C+B，缺失当状态）、prot（只有蛋白）、cond（N+C+B，没有序列）、fullM（M−：缺失不参与）。"""
    import time
    W = dict(BLOCK_W if weights is None else weights)
    N = inp["N"]
    out = {k: np.empty((N, N), np.float32) for k in ("full", "prot", "cond", "fullM")}
    t0 = time.time()
    wsum = sum(W.values())
    wcond = W["N"] + W["C"] + W["B"]
    for i0 in range(0, N, chunk):
        i1 = min(N, i0 + chunk)
        pP, oP, cP = block_P(inp, i0, i1)
        pN, oN, cN = block_N(inp, i0, i1)
        pC, oC, cC = block_C(inp, i0, i1)
        pB, oB, cB = block_B(inp, i0, i1)
        out["full"][i0:i1] = (W["P"] * pP + W["N"] * pN + W["C"] * pC + W["B"] * pB) / wsum
        out["prot"][i0:i1] = pP
        out["cond"][i0:i1] = (W["N"] * pN + W["C"] * pC + W["B"] * pB) / wcond
        num = W["P"] * cP * oP + W["N"] * cN * oN + W["C"] * cC * oC + W["B"] * cB * oB
        den = W["P"] * cP + W["N"] * cN + W["C"] * cC + W["B"] * cB
        out["fullM"][i0:i1] = np.where(den > 0, num / np.maximum(den, 1e-9), np.float32(0.5))
        if verbose and (i0 // chunk) % 4 == 0:
            print(f"  距离矩阵 行 {i1}/{N}  {time.time() - t0:.0f}s", flush=True)
    for k, M in out.items():
        np.fill_diagonal(M, 0.0)
    return out


def missing_dims(df):
    """每条记录在各维度上是否缺失（核酸数值特征只在有核酸时才可能缺失）。返回布尔 DataFrame 与缺失数。"""
    pres = df["na_class"].astype(str).to_numpy() != "none"
    f = pd.DataFrame(index=df.index)
    f["seq"] = df["seq_row"].to_numpy().astype(int) < 0
    f["conc"] = df["st_conc"].to_numpy() == "miss"
    f["pH"] = df["st_pH"].to_numpy() == "miss"
    f["temp"] = df["st_temp"].to_numpy() == "miss"
    f["salt"] = np.all([df[f"st_{k}"].to_numpy() == "miss" for k in SALT], axis=0)
    f["crowder"] = np.all([df[f"st_{k}"].to_numpy() == "miss" for k in CROWD], axis=0)
    f["buffer"] = df["st_buffer"].to_numpy() == "miss"
    f["additive"] = df["st_additive"].to_numpy() == "miss"
    f["na_len"] = pres & pd.to_numeric(df["na_len_log10"], errors="coerce").isna().to_numpy()
    f["na_conc"] = pres & pd.to_numeric(df["na_conc_log10"], errors="coerce").isna().to_numpy()
    f["na_seq"] = pres & pd.to_numeric(df["na_gc"], errors="coerce").isna().to_numpy()
    return f, f.to_numpy().sum(1)


# ----------------------------------------------------------------------------- kNN 与诊断
def knn(D, k, mask_group=None, cand=None, chunk=1024):
    """每行 k 个最近邻（不含自身）。mask_group：同组（如同一篇论文）的候选被排除；cand：只在 cand 为 True 的记录里找。
    返回 (idx, dist)；候选不足 k 个时用 -1 / inf 填充。"""
    N = D.shape[0]
    idx = np.full((N, k), -1, np.int32)
    dst = np.full((N, k), np.inf, np.float32)
    for i0 in range(0, N, chunk):
        i1 = min(N, i0 + chunk)
        blk = D[i0:i1].copy()
        r = np.arange(i1 - i0)
        blk[r, np.arange(i0, i1)] = np.inf
        if mask_group is not None:       # 单个分组数组，或一组数组（任一相同就排除）
            for g in (mask_group if isinstance(mask_group, (list, tuple)) else [mask_group]):
                blk[g[i0:i1, None] == g[None, :]] = np.inf
        if cand is not None:
            blk[:, ~cand] = np.inf
        kk = min(k, N - 1)
        part = np.argpartition(blk, kk - 1, axis=1)[:, :kk]
        pd_ = np.take_along_axis(blk, part, 1)
        o = np.argsort(pd_, axis=1)
        part, pd_ = np.take_along_axis(part, o, 1), np.take_along_axis(pd_, o, 1)
        ok = np.isfinite(pd_)
        idx[i0:i1, :kk] = np.where(ok, part, -1)
        dst[i0:i1, :kk] = np.where(ok, pd_, np.inf)
    return idx, dst


def purity(idx, labels, valid=None):
    """k 近邻里与自己同标签的比例的均值；返回 (purity, chance)。labels 里的 -1 / '' 视为无标签，不计。"""
    lab = pd.factorize(pd.Series(labels))[0] if not np.issubdtype(np.asarray(labels).dtype, np.integer) else np.asarray(labels)
    ok = lab >= 0
    if valid is not None:
        ok &= valid
    nb = idx
    nb_lab = np.where(nb >= 0, lab[np.clip(nb, 0, None)], -2)
    same = (nb_lab == lab[:, None]) & ok[:, None] & (nb_lab >= 0)
    cnt = ((nb >= 0) & (nb_lab >= 0)).sum(1)
    pur = np.where(cnt > 0, same.sum(1) / np.maximum(cnt, 1), np.nan)
    p = np.bincount(lab[ok]) / ok.sum()
    return float(np.nanmean(pur[ok])), float((p ** 2).sum())


def knn_auroc(idx, y, cand_mask):
    """y∈{0,1}（NaN 不计）；score = 邻居（只取 cand_mask 为 True 且有标签的）里正例的比例。"""
    from sklearn.metrics import roc_auc_score
    nb = idx
    good = (nb >= 0)
    yy = np.where(np.isnan(y), np.nan, y)
    nbv = np.where(good, yy[np.clip(nb, 0, None)], np.nan)
    sc = np.nanmean(nbv, axis=1)
    ok = (~np.isnan(y)) & (~np.isnan(sc)) & cand_mask
    if len(np.unique(y[ok])) < 2:
        return float("nan"), int(ok.sum())
    return float(roc_auc_score(y[ok], sc[ok])), int(ok.sum())


def neighbor_overlap(idx_hi, Y, k=15):
    """二维图保持高维近邻的程度：高维 k 近邻中也落在二维 k 近邻里的比例。"""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Y)
    lo = nn.kneighbors(Y, return_distance=False)[:, 1:]
    hi = idx_hi[:, :k]
    hit = 0.0
    for a, b in zip(hi, lo):
        hit += len(set(a.tolist()) & set(b.tolist())) / k
    return hit / len(hi)
