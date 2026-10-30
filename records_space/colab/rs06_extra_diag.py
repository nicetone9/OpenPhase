"""rs06（Colab）：补充诊断。
1. 近邻里的 PS 标签：同时排除同论文和同蛋白（cross-both），并给出每个透镜的标准误（对记录 bootstrap 200 次）。
2. csat 记录离 binary 记录有多远：四个透镜下的覆盖；把 csat → 最近 binary 的距离按 P/N/C/B 四块拆开，并与 binary 记录到自己第 15 近邻的距离对照。
输出 out/diagnostics_extra.json。
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

import rslib


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def boot_auroc(idx, y, ok, n_boot=200, seed=0):
    from sklearn.metrics import roc_auc_score
    nb = idx
    nbv = np.where(nb >= 0, y[np.clip(nb, 0, None)], np.nan)
    sc = np.nanmean(nbv, axis=1)
    use = np.where(ok & ~np.isnan(sc) & ~np.isnan(y))[0]
    a0 = roc_auc_score(y[use], sc[use])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        s = rng.choice(use, len(use))
        if len(np.unique(y[s])) == 2:
            bs.append(roc_auc_score(y[s], sc[s]))
    return float(a0), float(np.std(bs)), int(len(use))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    df = rslib.load_records(f"{a.data}/rs_records.tsv")
    seqs = pd.read_csv(f"{a.data}/rs_sequences.tsv", sep="\t", keep_default_na=False, na_values=[""])
    esmc = np.vstack([np.load(f"{a.data}/esmc_L18.npy").astype(np.float32), np.load(f"{a.out}/esmc_L18_new.npy").astype(np.float32)[:int((seqs["origin"] == "csat_only").sum())]])
    inp = rslib.build_inputs(df, esmc, seqs)
    rslib.calibrate_P(inp)
    D = rslib.compute_lens_matrices(inp, verbose=False)
    log("距离矩阵完成")
    pool = df["pool"].to_numpy().astype(str)
    is_bin, is_csat = pool == "binary", pool == "csat"
    y = pd.to_numeric(df["label"], errors="coerce").to_numpy(dtype=float)
    paper_code = pd.factorize(df["paper"])[0]
    uni = df["uniprot"].fillna("").astype(str).str.split("-").str[0]
    prot_key = pd.factorize(np.where(uni != "", "U:" + uni, "S:" + df["seq_row"].astype(str)))[0]
    out = {}

    # 1) cross-both 与自助法标准误
    au = {}
    for lens in D:
        r = {}
        for tag, grp in (("any", None), ("cross-paper", paper_code), ("cross-protein", prot_key), ("cross-both", [paper_code, prot_key])):
            idx, _ = rslib.knn(D[lens], 15, mask_group=grp, cand=is_bin)
            m, se, n = boot_auroc(idx, y, is_bin)
            r[tag] = dict(auroc=m, se=se, n=n)
        au[lens] = r
        log(f"kNN-AUROC[{lens}] " + " | ".join(f"{t}: {v['auroc']:.3f}±{v['se']:.3f}" for t, v in r.items()))
    out["knn_auroc_ps_label_k15_boot"] = au

    # 2) csat 覆盖（四个透镜）
    ib, ic = np.where(is_bin)[0], np.where(is_csat)[0]
    cov = {}
    for lens in D:
        Dm = D[lens]
        idx, dst = rslib.knn(Dm, 15, cand=is_bin)
        r15 = dst[ib, 14]
        d_to_csat = Dm[np.ix_(ib, ic)].min(1)
        d_csat_bin = Dm[np.ix_(ic, ib)].min(1)
        src = df["source"].astype(str).to_numpy()[ic]
        cov[lens] = dict(
            frac_binary_with_csat_inside_15NN_radius=float((d_to_csat <= r15).mean()),
            n_binary_with_csat_inside_15NN_radius=int((d_to_csat <= r15).sum()),
            csat_to_nearest_binary_median=float(np.median(d_csat_bin)),
            csat_to_nearest_binary_median_by_source={s: float(np.median(d_csat_bin[src == s])) for s in np.unique(src)},
            binary_15NN_radius_median=float(np.median(r15)),
            share_csat_rows_within_binary_median_15NN_radius=float((d_csat_bin <= np.median(r15)).mean()),
            share_csat_rows_within_binary_p90_15NN_radius=float((d_csat_bin <= np.quantile(r15, 0.9)).mean()))
    out["csat_coverage"] = cov
    log("csat 覆盖：" + json.dumps({k: round(v["csat_to_nearest_binary_median"], 4) for k, v in cov.items()}))

    # 3) 块分解：csat → 最近 binary 的 full 距离按 P/N/C/B 拆开；对照 binary → 自己第 15 近邻
    W = rslib.BLOCK_W
    wsum = sum(W.values())
    i0, i1 = int(ic[0]), int(ic[-1]) + 1
    assert (ic == np.arange(i0, i1)).all(), "csat 记录应当在表尾且连续"
    nearest = np.argmin(np.where(is_bin[None, :], D["full"][i0:i1], np.inf), axis=1)
    rows = np.arange(i1 - i0)
    blk = dict(P=rslib.block_P(inp, i0, i1)[0], N=rslib.block_N(inp, i0, i1)[0], C=rslib.block_C(inp, i0, i1)[0], B=rslib.block_B(inp, i0, i1)[0])
    src = df["source"].astype(str).to_numpy()[ic]
    dec = {}
    for s in ["all"] + list(np.unique(src)):
        m = np.ones(len(rows), bool) if s == "all" else src == s
        dec[s] = {k: float(np.median(W[k] * blk[k][rows[m], nearest[m]] / wsum)) for k in blk}
        dec[s]["total_median"] = float(np.median(D["full"][i0 + rows[m], nearest[m]]))
    # binary 对照（逐块计算，取 15 近邻里最远的那个）
    idx15, _ = rslib.knn(D["full"], 15, cand=is_bin)
    nn15 = idx15[:, 14]
    contrib = {k: np.zeros(len(df), np.float32) for k in "PNCB"}
    for j0 in range(0, len(df), 1024):
        j1 = min(len(df), j0 + 1024)
        for k, fn in (("P", rslib.block_P), ("N", rslib.block_N), ("C", rslib.block_C), ("B", rslib.block_B)):
            arr = fn(inp, j0, j1)[0]
            contrib[k][j0:j1] = W[k] * arr[np.arange(j1 - j0), np.clip(nn15[j0:j1], 0, None)] / wsum
    dec["binary_to_own_15th_neighbour"] = {k: float(np.median(contrib[k][ib])) for k in "PNCB"}
    dec["binary_to_own_15th_neighbour"]["total_median"] = float(np.median(D["full"][ib, nn15[ib]]))
    out["csat_to_nearest_binary_block_decomposition"] = dec
    log("块分解：" + json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in dec.items()}))
    json.dump(out, open(f"{a.out}/diagnostics_extra.json", "w"), indent=1)
    log("EXTRA_DONE")


if __name__ == "__main__":
    main()
