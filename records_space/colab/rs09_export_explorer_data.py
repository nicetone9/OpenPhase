#!/usr/bin/env python3
"""rs09（Colab）：导出"试新实验"面板需要的数据。页面在浏览器里复现 rslib 的距离（不联网、不上传任何输入）。

导出（--export 目录）
  params.json        块权重、λ、各维度缩放、核酸特征归一化、名称表
  records.bin        11,439 条记录的模型输入数组（float32 + int16，小端），布局写在 records_meta.json
  seqs.bin / seqs_meta.json   1,338 条唯一序列：归一化 PCA 坐标 Zn、k-mer 草图、整序列哈希、名称、UniProt、c30 簇
  approx.json        新序列的近似：组成（+ 二肽）→ Zn 的岭回归，留一簇评估，以及"近似 P 块对 full 透镜近邻"的影响
  coverage.json      每个透镜下"到 15 近邻的平均距离"的分位数（含排除同簇的"未见蛋白"版本）
  examples.json      3 条未在数据集中的蛋白（UniProt，md5 已核对）：真实 ESMC 的 Zn 与近似 Zn
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

import rslib

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)
SK_K, SK_BITS = 5, 3                     # FracMinHash：5-mer，保留 hash < 2^32 / 8


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fnv1a32(b, h=2166136261):
    for c in b:
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def sketch(seq):
    thr = (1 << 32) >> SK_BITS
    out = set()
    for i in range(len(seq) - SK_K + 1):
        w = seq[i:i + SK_K]
        if all(c in AA_SET for c in w):
            h = fnv1a32(w.encode("ascii"))
            if h < thr:
                out.add(h)
    return sorted(out)


def comp24(s):
    n = len(s)
    fr = [s.count(a) / n for a in AA]
    return [float(np.log(n))] + fr + [(s.count("K") + s.count("R") - s.count("D") - s.count("E")) / n,
                                      sum(s.count(a) for a in "KRDE") / n, sum(s.count(a) for a in "FWY") / n]


def dimer400(s):
    idx = {a: i for i, a in enumerate(AA)}
    v = np.zeros(400, np.float64)
    for a, b in zip(s[:-1], s[1:]):
        if a in idx and b in idx:
            v[idx[a] * 20 + idx[b]] += 1
    return v / max(len(s) - 1, 1)


def nrm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def topk_overlap(S_hat, S_true, k=15):
    """逐行：两种相似度各取前 k 个，交集占 k 的比例。"""
    a = np.argpartition(-S_hat, k - 1, axis=1)[:, :k]
    b = np.argpartition(-S_true, k - 1, axis=1)[:, :k]
    return np.array([len(set(x.tolist()) & set(y.tolist())) / k for x, y in zip(a, b)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="含 esmc_L18_new.npy")
    ap.add_argument("--export", required=True)
    ap.add_argument("--examples", required=True, help="unseen_examples.fasta")
    ap.add_argument("--skip-esmc", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.export, exist_ok=True)
    warnings.filterwarnings("ignore")
    t_all = time.time()

    df = rslib.load_records(f"{a.data}/rs_records.tsv")
    seqs = pd.read_csv(f"{a.data}/rs_sequences.tsv", sep="\t", keep_default_na=False, na_values=[""])
    e_new = np.load(f"{a.out}/esmc_L18_new.npy").astype(np.float32)
    n_new = int((seqs["origin"] == "csat_only").sum())
    esmc = np.vstack([np.load(f"{a.data}/esmc_L18.npy").astype(np.float32), e_new[:n_new]])
    inp = rslib.build_inputs(df, esmc, seqs)
    rslib.calibrate_P(inp)
    N, M = inp["N"], len(seqs)
    Zn = inp["Zn"].astype(np.float32)
    pidx = inp["pidx"]
    log(f"输入完成：{N} 条记录，{M} 条序列，Zn {Zn.shape}")

    # ---------- 序列元数据、草图、整序列哈希 ----------
    s_list = ["".join(x.upper().split()) for x in seqs["sequence"]]
    uni_of, c30_of = {}, {}
    for r, u, c in zip(df["seq_row"].astype(int), df["uniprot"].fillna("").astype(str), df["c30"].fillna("").astype(str)):
        if r >= 0:
            if u and r not in uni_of:
                uni_of[r] = u
            if c and r not in c30_of:
                c30_of[r] = c
    sk = [sketch(s) for s in s_list]
    sk_off = np.cumsum([0] + [len(x) for x in sk]).astype(np.int32)
    sk_all = np.array([h for x in sk for h in x], dtype="<u4")
    h1 = np.array([fnv1a32(s.encode("ascii", "replace")) for s in s_list], dtype="<u4")
    h2 = np.array([fnv1a32(s.encode("ascii", "replace"), 0x9747B28C) for s in s_list], dtype="<u4")
    seq_meta = [dict(name=str(n) if isinstance(n, str) else "", uniprot=uni_of.get(i, ""), length=int(L), n_records=int(nr), c30=c30_of.get(i, ""))
                for i, (n, L, nr) in enumerate(zip(seqs["name"].fillna(""), seqs["length"], seqs["n_records"]))]
    log(f"草图：共 {len(sk_all)} 个哈希（均 {len(sk_all) / M:.0f} / 序列）")

    # ---------- 近似：组成（+ 二肽）→ Zn ----------
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    C24 = np.array([comp24(s) for s in s_list], np.float64)
    D400 = np.array([dimer400(s) for s in s_list], np.float64)
    gcode = pd.factorize(np.array([c30_of.get(i, f"_u{i}") for i in range(M)]))[0]
    feats = {"comp24": C24, "comp24+dimer": np.hstack([C24, D400])}
    k = 15
    rng = np.random.default_rng(0)
    results, oof_store = {}, {}
    for fname, X in feats.items():
        for alpha in (1.0, 10.0, 100.0, 1000.0, 10000.0):
            Zh = np.zeros_like(Zn)
            for tr, te in GroupKFold(5).split(X, groups=gcode):
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
                m = Ridge(alpha=alpha).fit((X[tr] - mu) / sd, Zn[tr])
                Zh[te] = m.predict((X[te] - mu) / sd)
            ov = []
            cs = []
            for tr, te in GroupKFold(5).split(X, groups=gcode):
                St = Zn[te] @ Zn[tr].T
                Sh = nrm(Zh[te]) @ Zn[tr].T
                ov.append(topk_overlap(Sh, St, k))
                cs.append((nrm(Zh[te]) * Zn[te]).sum(1))
            ov, cs = np.concatenate(ov), np.concatenate(cs)
            results[(fname, alpha)] = dict(overlap_mean=float(ov.mean()), overlap_median=float(np.median(ov)), cos_pred_true_mean=float(cs.mean()))
            oof_store[(fname, alpha)] = Zh
            log(f"  近似 {fname} α={alpha:g}: 前 {k} 近邻重叠 {ov.mean():.3f}，Zh 与 Z 的余弦 {cs.mean():.3f}")
    best = max(results, key=lambda kk: results[kk]["overlap_mean"])
    fname, alpha = best
    X = feats[fname]
    log(f"选定近似：{fname}, α={alpha:g}；重叠 {results[best]['overlap_mean']:.3f}（随机约 {k / (M * 0.8):.3f}）")
    # 无 Z 的基线：组成 z-score 后的余弦近邻与 ESMC 近邻的重叠
    base = []
    for tr, te in GroupKFold(5).split(X, groups=gcode):
        mu, sd = C24[tr].mean(0), C24[tr].std(0) + 1e-6
        Sc = nrm((C24[te] - mu) / sd) @ nrm((C24[tr] - mu) / sd).T
        base.append(topk_overlap(Sc, Zn[te] @ Zn[tr].T, k))
    base = np.concatenate(base)
    mu_f, sd_f = X.mean(0), X.std(0) + 1e-6
    model = Ridge(alpha=alpha).fit((X - mu_f) / sd_f, Zn)
    W, b = model.coef_.T.astype(np.float32), model.intercept_.astype(np.float32)          # (F, 64), (64,)
    Zh_oof = oof_store[best]

    # ---------- 距离、覆盖分布、近似 P 对 full 透镜近邻的影响 ----------
    t0 = time.time()
    D = rslib.compute_lens_matrices(inp, verbose=False)
    log(f"四个透镜的距离矩阵 {time.time() - t0:.0f}s")
    seq_c30 = pd.factorize(np.array([c30_of.get(int(i), f"_s{int(i)}") if i >= 0 else "_none" for i in pidx]))[0]
    cov = {"quantile_levels": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], "n_records": int(N)}
    knn_unseen = {}
    for lens in ("full", "prot", "cond", "fullM"):
        idx0, dst0 = rslib.knn(D[lens], 15)
        idx1, dst1 = rslib.knn(D[lens], 15, mask_group=[seq_c30, pidx])
        knn_unseen[lens] = (idx1, dst1)
        m0 = np.where(np.isfinite(dst0), dst0, np.nan)
        m1 = np.where(np.isfinite(dst1), dst1, np.nan)
        cov[lens] = dict(plain_mean15=[float(x) for x in np.nanquantile(np.nanmean(m0, 1), cov["quantile_levels"])],
                         plain_nn1=[float(x) for x in np.nanquantile(m0[:, 0], cov["quantile_levels"])],
                         unseen_mean15=[float(x) for x in np.nanquantile(np.nanmean(m1, 1), cov["quantile_levels"])],
                         unseen_nn1=[float(x) for x in np.nanquantile(m1[:, 0], cov["quantile_levels"])])
        log(f"覆盖[{lens}] 15 近邻均距中位数：plain {cov[lens]['plain_mean15'][3]:.4f}，unseen {cov[lens]['unseen_mean15'][3]:.4f}")
    # 近似 P：每条记录用其序列的留一簇预测 Zh（归一化）重算 full 行距离
    t0 = time.time()
    Zq = np.zeros((N, Zn.shape[1]), np.float32)
    ok = pidx >= 0
    Zq[ok] = nrm(Zh_oof[pidx[ok]]).astype(np.float32)
    W_ = rslib.BLOCK_W
    wsum = sum(W_.values())
    Dh = np.empty((N, N), np.float32)
    scP = inp["scales"]["P"]
    safe_a = np.where(pidx >= 0, pidx, 0)
    has = inp["P_has"]
    for i0 in range(0, N, 1024):
        i1 = min(N, i0 + 1024)
        S = np.clip(1.0 - Zq[i0:i1] @ Zn.T, 0.0, None)
        dP = np.minimum(S[:, safe_a] / scP, 1.0)
        oc, oa = has[i0:i1, None], has[None, :]
        dP = np.where(oc & oa, dP, np.where(oc ^ oa, np.float32(rslib.LAM), np.float32(0.0))).astype(np.float32)
        pN, _, _ = rslib.block_N(inp, i0, i1)
        pC, _, _ = rslib.block_C(inp, i0, i1)
        pB, _, _ = rslib.block_B(inp, i0, i1)
        Dh[i0:i1] = (W_["P"] * dP + W_["N"] * pN + W_["C"] * pC + W_["B"] * pB) / wsum
    np.fill_diagonal(Dh, 0.0)
    idxh, _ = rslib.knn(Dh, 15, mask_group=[seq_c30, pidx])
    idx1 = knn_unseen["full"][0]
    sel = np.where(ok)[0]
    ov_rec = np.array([len(set(idx1[i][idx1[i] >= 0].tolist()) & set(idxh[i][idxh[i] >= 0].tolist())) / 15 for i in sel])
    log(f"近似 P 对 full 透镜（排除同簇）15 近邻的影响：重叠均值 {ov_rec.mean():.3f}，中位数 {np.median(ov_rec):.3f}，{time.time() - t0:.0f}s")
    del Dh
    approx = dict(feature=fname, alpha=alpha, n_features=int(X.shape[1]), mean=[float(x) for x in mu_f], std=[float(x) for x in sd_f],
                  W=[[round(float(v), 6) for v in row] for row in W], b=[round(float(v), 6) for v in b],
                  eval=dict(k=k, sequences=int(M), grouped_by="30% identity homology cluster (5-fold, held-out clusters)",
                            overlap_mean=results[best]["overlap_mean"], overlap_median=results[best]["overlap_median"],
                            cos_pred_true_mean=results[best]["cos_pred_true_mean"], random_overlap=float(k / (M * 0.8)),
                            composition_cosine_baseline_overlap_mean=float(base.mean()),
                            all_settings={f"{kk[0]}|alpha={kk[1]:g}": vv for kk, vv in results.items()},
                            placement_full_lens=dict(overlap_mean=float(ov_rec.mean()), overlap_median=float(np.median(ov_rec)), n_records=int(len(sel)),
                                                     note="15 nearest records in the full lens with same-cluster records excluded: approximated P block vs exact ESMC P block")))

    # ---------- 例子：真实 ESMC 与近似 ----------
    examples = []
    fa = open(a.examples).read().strip().split(">")[1:]
    ex = []
    for blk in fa:
        head, *lines = blk.strip().splitlines()
        ex.append((head.split()[0], head, "".join(lines)))
    if not a.skip_esmc:
        import torch
        from rs02_embed_new_seqs import load_model, pooled
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, tok = load_model(dev)
        e_mu, e_sd, pca, zm = inp["esmc_norm"]["mu"], inp["esmc_norm"]["sd"], inp["pca"], inp["z_mean"]
        for acc, head, s in ex:
            t1 = time.time()
            e = pooled(model, tok, s, dev).astype(np.float32)
            z = pca.transform(((e - e_mu) / e_sd)[None, :])[0].astype(np.float32) - zm
            zn = z / (np.linalg.norm(z) + 1e-9)
            c = comp24(s)
            xf = np.concatenate([c, dimer400(s)]) if fname == "comp24+dimer" else np.array(c)
            zh = ((xf - mu_f) / sd_f) @ W + b
            zhn = zh / (np.linalg.norm(zh) + 1e-9)
            St, Sh = Zn @ zn, Zn @ zhn
            ov = len(set(np.argsort(-St)[:k].tolist()) & set(np.argsort(-Sh)[:k].tolist())) / k
            examples.append(dict(id=acc, header=head, sequence=s, length=len(s), zn_exact=[round(float(v), 5) for v in zn], zn_approx=[round(float(v), 5) for v in zhn],
                                 cos_approx_exact=float(zn @ zhn), top15_overlap=float(ov),
                                 nearest_known=[dict(i=int(i), cos=float(St[i]), name=seq_meta[i]["name"], uniprot=seq_meta[i]["uniprot"]) for i in np.argsort(-St)[:3]]))
            log(f"例子 {acc}: len {len(s)}，ESMC {time.time() - t1:.0f}s，近似与精确的余弦 {float(zn @ zhn):.3f}，前 15 近邻重叠 {ov:.2f}")

    # ---------- 写出 ----------
    f32 = lambda x: np.ascontiguousarray(np.asarray(x, dtype="<f4"))
    i16 = lambda x: np.ascontiguousarray(np.asarray(x, dtype="<i2"))
    fl = np.asarray(inp["na_flags"] > 0.5, dtype=np.int64)
    pw = lambda m: (m.astype(np.int64) * (1 << np.arange(m.shape[1]))).sum(1)
    cols = [("conc", f32(inp["conc"])), ("pH", f32(inp["pH"])), ("temp", f32(inp["temp"])), ("salt", f32(np.stack(inp["salt"], 1))), ("crowd", f32(np.stack(inp["crowd"], 1))),
            ("na_len", f32(inp["na_len"])), ("na_conc", f32(inp["na_conc"])), ("na_ratio", f32(inp["na_ratio"])), ("na_seq", f32(inp["na_seq"])), ("na_fold", f32(inp["na_fold"])),
            ("pidx", i16(pidx)), ("temp_code", i16(inp["temp_code"])), ("na_cls", i16(inp["na_cls"])), ("na_flags", i16(pw(fl))),
            ("buf", i16(pw(inp["buf"] > 0.5))), ("add", i16(pw(inp["add"] > 0.5))), ("buf_obs", i16(inp["buf_obs"])), ("add_obs", i16(inp["add_obs"]))]
    meta, off, blob = [], 0, bytearray()
    for name, arr in cols:
        meta.append(dict(name=name, dtype=str(arr.dtype), shape=list(arr.shape), offset=off))
        b_ = arr.tobytes()
        blob += b_
        off += len(b_)
        if off % 4:
            pad = 4 - off % 4
            blob += b"\x00" * pad
            off += pad
    open(f"{a.export}/records.bin", "wb").write(bytes(blob))
    json.dump(dict(n=int(N), arrays=meta), open(f"{a.export}/records_meta.json", "w"))
    sblob = bytearray()
    smeta = []
    for name, arr in (("zn", f32(Zn)), ("sk_all", np.ascontiguousarray(sk_all)), ("sk_off", np.ascontiguousarray(sk_off.astype("<i4"))), ("h1", np.ascontiguousarray(h1)), ("h2", np.ascontiguousarray(h2))):
        smeta.append(dict(name=name, dtype=str(arr.dtype), shape=list(arr.shape), offset=len(sblob)))
        sblob += arr.tobytes()
    open(f"{a.export}/seqs.bin", "wb").write(bytes(sblob))
    json.dump(dict(m=int(M), arrays=smeta, seqs=seq_meta, sketch=dict(k=SK_K, bits=SK_BITS)), open(f"{a.export}/seqs_meta.json", "w"))
    na_norm = {k_: dict(mu=[float(x) for x in v["mu"]], sd=[float(x) for x in v["sd"]], sc=float(v["sc"])) for k_, v in inp["na_norm"].items()}
    params = dict(block_w=rslib.BLOCK_W, lam=rslib.LAM, temp_state_pen=rslib.TEMP_STATE_PEN, scales={k_: float(v) for k_, v in inp["scales"].items()},
                  na_norm=na_norm, short=rslib.SHORT, salt=rslib.SALT, crowd=rslib.CROWD, na_flags=rslib.NA_FLAGS, buf=rslib.BUF, add=rslib.ADD, aa=AA)
    json.dump(params, open(f"{a.export}/params.json", "w"), indent=1)
    json.dump(approx, open(f"{a.export}/approx.json", "w"))
    json.dump(cov, open(f"{a.export}/coverage.json", "w"), indent=1)
    json.dump(examples, open(f"{a.export}/examples.json", "w"))
    open(f"{a.export}/EXPORT_DONE", "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    log(f"导出完成 {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
