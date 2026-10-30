"""rs03（Colab）：构建 records 空间 → 四个透镜的距离矩阵 → UMAP / t-SNE → 诊断。

输入  data/rs_records.tsv、data/rs_sequences.tsv、data/esmc_L18.npy（主表 1,299 条）、out/esmc_L18_new.npy（csat 独有序列；rs02 产出）
输出  out/rs_coords.tsv、out/diagnostics.json、out/run_info.json、out/knn_<lens>.npz、out/missing_flags.tsv
透镜  full  = P 0.40 + N 0.15 + C 0.35 + B 0.10，缺失当状态（默认，以蛋白为中心）
      prot  = 只有蛋白序列
      cond  = N + C + B（没有序列：只看核酸与条件）
      fullM = full 的 M− 版本（缺失维度不参与，对照"缺失当状态"的影响）
用法  python rs03_run.py --data /content/rs/data --out /content/rs/out [--limit 1500] [--no-tsne]
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

import rslib


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="冒烟：随机取 N 条 binary + 约 N/5 条 csat")
    ap.add_argument("--no-tsne", action="store_true")
    ap.add_argument("--no-robust", action="store_true")
    ap.add_argument("--umap-k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_all = time.time()
    warnings.filterwarnings("ignore")

    df = rslib.load_records(f"{a.data}/rs_records.tsv")
    seqs = pd.read_csv(f"{a.data}/rs_sequences.tsv", sep="\t", keep_default_na=False, na_values=[""])
    e_main = np.load(f"{a.data}/esmc_L18.npy").astype(np.float32)
    e_new = np.load(f"{a.out}/esmc_L18_new.npy").astype(np.float32)
    n_new = int((seqs["origin"] == "csat_only").sum())
    esmc = np.vstack([e_main, e_new[:n_new]])
    assert esmc.shape[0] == len(seqs), (esmc.shape, len(seqs))
    # 主表行与 rs_sequences 的 main_seq_row 一致
    assert (seqs.loc[seqs["origin"] == "main", "main_seq_row"].astype(int).to_numpy() == np.arange(len(e_main))).all()
    log(f"记录 {len(df)}（binary {int((df.pool == 'binary').sum())}，csat {int((df.pool == 'csat').sum())}）；序列 {len(seqs)}；ESMC {esmc.shape}")

    keep = np.arange(len(df))
    if a.limit:
        rng = np.random.default_rng(a.seed)
        b = np.where(df.pool == "binary")[0]
        c = np.where(df.pool == "csat")[0]
        keep = np.sort(np.concatenate([rng.choice(b, min(a.limit, len(b)), replace=False), rng.choice(c, min(max(a.limit // 5, 50), len(c)), replace=False)]))
        df = df.iloc[keep].reset_index(drop=True)
        log(f"冒烟模式：取 {len(df)} 条")
    N = len(df)

    # ---------------- 输入与距离 ----------------
    t0 = time.time()
    inp = rslib.build_inputs(df, esmc, seqs)
    sc = rslib.calibrate_P(inp)
    log(f"输入构建 {time.time() - t0:.0f}s；P 块缩放（随机对余弦距离 95% 分位）= {sc:.4f}")
    t0 = time.time()
    D = rslib.compute_lens_matrices(inp)
    log(f"四个透镜的距离矩阵 {time.time() - t0:.0f}s；每个 {D['full'].nbytes / 1e6:.0f} MB")
    import sklearn
    import umap
    info = dict(n_records=N, limit=a.limit, seed=a.seed, block_weights=rslib.BLOCK_W, lam=rslib.LAM, temp_state_pen=rslib.TEMP_STATE_PEN,
                scales={k: float(v) for k, v in inp["scales"].items()}, versions=dict(numpy=np.__version__, pandas=pd.__version__,
                sklearn=sklearn.__version__, umap=umap.__version__), umap_k=a.umap_k)
    json.dump(info, open(f"{a.out}/run_info.json", "w"), indent=1)

    # ---------------- kNN ----------------
    t0 = time.time()
    KNN = {}
    for lens in D:
        idx, dst = rslib.knn(D[lens], 50)
        KNN[lens] = (idx, dst)
        np.savez_compressed(f"{a.out}/knn_{lens}.npz", idx=idx, dst=dst)
    log(f"kNN(k=50) {time.time() - t0:.0f}s")

    # ---------------- 降维 ----------------
    coords = pd.DataFrame({"rs_idx": df["rs_idx"].to_numpy(), "record_id": df["record_id"].to_numpy()})
    maps = {}
    for lens in ("full", "prot", "cond", "fullM"):
        t0 = time.time()
        Y = umap.UMAP(n_neighbors=a.umap_k, min_dist=0.3, metric="precomputed", random_state=a.seed, n_components=2).fit_transform(D[lens])
        maps[f"umap_{lens}"] = Y
        coords[f"umap_{lens}_x"], coords[f"umap_{lens}_y"] = Y[:, 0], Y[:, 1]
        coords.to_csv(f"{a.out}/rs_coords.tsv", sep="\t", index=False)
        log(f"UMAP[{lens}] {time.time() - t0:.0f}s")
    if not a.no_robust:
        for k in (10, 60):
            t0 = time.time()
            Y = umap.UMAP(n_neighbors=k, min_dist=0.3, metric="precomputed", random_state=a.seed, n_components=2).fit_transform(D["full"])
            maps[f"umap_full_k{k}"] = Y
            coords[f"umap_full_k{k}_x"], coords[f"umap_full_k{k}_y"] = Y[:, 0], Y[:, 1]
            log(f"UMAP[full, k={k}] {time.time() - t0:.0f}s")
        coords.to_csv(f"{a.out}/rs_coords.tsv", sep="\t", index=False)
    if not a.no_tsne:
        from sklearn.manifold import TSNE
        t0 = time.time()
        init = maps["umap_full"] / maps["umap_full"][:, 0].std() * 1e-4
        Y = TSNE(n_components=2, metric="precomputed", init=init.astype(np.float32), perplexity=30, learning_rate="auto", random_state=a.seed, n_jobs=-1).fit_transform(D["full"])
        maps["tsne_full"] = Y
        coords["tsne_full_x"], coords["tsne_full_y"] = Y[:, 0], Y[:, 1]
        coords.to_csv(f"{a.out}/rs_coords.tsv", sep="\t", index=False)
        log(f"t-SNE[full] {time.time() - t0:.0f}s")

    # ---------------- 诊断 ----------------
    diag = dict(n=N)
    pool = df["pool"].to_numpy().astype(str)
    is_bin, is_csat = pool == "binary", pool == "csat"
    y = pd.to_numeric(df["label"], errors="coerce").to_numpy(dtype=float)
    paper_code = pd.factorize(df["paper"])[0]
    uni = df["uniprot"].fillna("").astype(str).str.split("-").str[0]
    prot_key = pd.factorize(np.where(uni != "", "U:" + uni, "S:" + df["seq_row"].astype(str)))[0]

    # a) 二维图保持高维近邻的程度
    t0 = time.time()
    ov = {}
    for name, Y in maps.items():
        lens = name.split("_")[1] if name.split("_")[1] in D else "full"
        ov[name] = rslib.neighbor_overlap(KNN[lens][0], Y, k=15)
    diag["neighbor_overlap_k15"] = ov
    log(f"近邻保持度 {time.time() - t0:.0f}s：{ov}")

    # b) 标签纯度：k=15 近邻里同标签的比例（含同论文/同蛋白的邻居，所以是"描述"不是"预测"）
    fl, nm = rslib.missing_dims(df)
    fl.to_csv(f"{a.out}/missing_flags.tsv", sep="\t", index=False)
    tstate = df["st_temp"].astype(str).to_numpy()
    buf_cols = [f"buf_{k}" for k in rslib.BUF]
    bprim = np.array([next((k for k in rslib.BUF if r[f"buf_{k}"] == 1), "other/none") if sb != "miss" else "unknown" for (_, r), sb in zip(df[buf_cols].iterrows(), df["st_buffer"].astype(str))])
    labels = {
        "protein (UniProt or sequence)": prot_key,
        "homology cluster c30": pd.Series(df["c30"].fillna("").astype(str)).replace("", np.nan),
        "paper": paper_code,
        "source": pd.Series(df["source"].astype(str)),
        "nucleic acid class": pd.Series(df["na_class"].astype(str)),
        "temperature state": pd.Series(tstate),
        "pool (binary / csat)": pd.Series(pool),
        "phase-separation label": pd.Series(np.where(np.isnan(y), np.nan, y)),
        "morphology": pd.Series(df["morphology"].fillna("").astype(str)).replace("", np.nan),
        "csat link": pd.Series(df["csat_link"].astype(str)),
        "buffer family": pd.Series(bprim),
        "readout class": pd.Series(df["readout_class"].astype(str)),
    }
    pur = {}
    for lab_name, lab in labels.items():
        row = {}
        for lens in D:
            try:
                row[lens] = rslib.purity(KNN[lens][0][:, :15], lab)
            except Exception as e:     # noqa
                row[lens] = [float("nan"), float("nan")]
        pur[lab_name] = row
    diag["purity_k15"] = pur
    log("标签纯度完成")

    # c) 近邻里的 PS 标签能不能预测自己的标签（只在 binary 记录之间找邻居，逐级排除同论文 / 同蛋白）
    t0 = time.time()
    au = {}
    for lens in D:
        r = {}
        for tag, grp in (("any", None), ("cross-paper", paper_code), ("cross-protein", prot_key)):
            idx, _ = rslib.knn(D[lens], 15, mask_group=grp, cand=is_bin)
            sc_, n_ = rslib.knn_auroc(idx, y, is_bin)
            # 置换基线：把候选标签打乱后再算一次
            yp = y.copy()
            ii = np.where(is_bin)[0]
            yp[ii] = np.random.default_rng(a.seed).permutation(y[ii])
            sp_, _ = rslib.knn_auroc(idx, yp, is_bin)
            r[tag] = dict(auroc=sc_, auroc_label_permuted=sp_, n=n_)
        au[lens] = r
        log(f"  kNN-AUROC[{lens}] {json.dumps(r)}")
    diag["knn_auroc_ps_label_k15"] = au
    log(f"kNN-AUROC {time.time() - t0:.0f}s")

    # d) 缺失结构
    miss = {}
    for pl in ("binary", "csat"):
        m = pool == pl
        miss[pl] = {c: float(fl.loc[m, c].mean()) for c in fl.columns}
        miss[pl]["mean_missing_dims"] = float(nm[m].mean())
    miss["by_source_mean_missing_dims"] = {s: float(nm[df["source"].astype(str).to_numpy() == s].mean()) for s in df["source"].astype(str).unique()}
    pat = pd.Series(["".join("1" if v else "0" for v in row) for row in fl.to_numpy()])
    miss["n_distinct_patterns"] = int(pat.nunique())
    miss["top_patterns"] = {k: int(v) for k, v in pat.value_counts().head(12).items()}
    miss["pattern_columns"] = list(fl.columns)
    # 缺失模式与来源/论文的关联：同一论文的记录里模式相同的比例 vs 随机
    same_pat_in_paper = []
    for p_, g in pd.DataFrame({"p": paper_code, "pat": pat}).groupby("p"):
        if len(g) > 1:
            vc = g["pat"].value_counts(normalize=True)
            same_pat_in_paper.append((len(g), float((vc ** 2).sum())))
    w_ = np.array([s for s, _ in same_pat_in_paper], dtype=float)
    miss["within_paper_pattern_agreement"] = float(np.average([v for _, v in same_pat_in_paper], weights=w_))
    pc = pat.value_counts(normalize=True)
    miss["chance_pattern_agreement"] = float((pc ** 2).sum())
    diag["missingness"] = miss
    log(f"缺失结构：{miss['n_distinct_patterns']} 种模式；论文内模式一致 {miss['within_paper_pattern_agreement']:.3f} vs 随机 {miss['chance_pattern_agreement']:.3f}")

    # e) csat 记录覆盖到 binary 空间的哪里
    cov = {}
    ib, ic = np.where(is_bin)[0], np.where(is_csat)[0]
    for lens in ("full", "cond", "prot"):
        Dm = D[lens]
        d_to_csat = Dm[np.ix_(ib, ic)].min(1)
        idx, dst = rslib.knn(Dm, 15, cand=is_bin)
        r15 = dst[ib, 14]
        covered = d_to_csat <= r15
        nn_b = dst[ib, 0]
        d_csat_to_bin = Dm[np.ix_(ic, ib)].min(1)
        cov[lens] = dict(
            frac_binary_records_with_csat_inside_their_15NN_radius=float(covered.mean()),
            by_na_class={c: float(covered[df["na_class"].to_numpy()[ib] == c].mean()) for c in ("none", "RNA", "DNA")},
            csat_to_nearest_binary_median=float(np.median(d_csat_to_bin)),
            binary_to_nearest_binary_median=float(np.median(nn_b)),
            binary_15NN_radius_median=float(np.median(r15)),
            share_csat_rows_nearer_than_binary_median_15NN_radius=float((d_csat_to_bin <= np.median(r15)).mean()))
    diag["csat_coverage"] = cov
    log(f"csat 覆盖：{json.dumps(cov['full'])}")

    json.dump(diag, open(f"{a.out}/diagnostics.json", "w"), indent=1, default=float)
    open(f"{a.out}/DONE", "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    log(f"全部完成 {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
