"""rs15（Colab GPU）：负例 / 未标注（PU）页的地图。
新蛋白（清单里 src=new）取序列并算 ESMC-600M 第 18 层嵌入（复用 rs12_universe 的函数），与已有嵌入（蛋白宇宙那次的输出、Swiss-Prot 参考集、PPMC 全长序列）拼起来，
z-score → PCA(64) → L2 归一化 → UMAP（余弦）。
用法：python rs15_pu.py --data /content/rs13/data --out /content/rs13/out [--limit N]
输入：data/pu_manifest.tsv、data/bg_emb_L18.npy、data/emb_data_L18.npy、data/uni_shards/emb_shard_*.npz
输出：out/pu_map.npz（acc、xy、src）、out/pu_new_meta.tsv、out/missing_ids.txt"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rs12_universe as R12                                              # noqa: E402

csv.field_size_limit(10 ** 9)
log = R12.log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    man = list(csv.DictReader(open(f"{a.data}/pu_manifest.tsv"), delimiter="\t"))
    new = [r["uniprot"] for r in man if r["src"] == "new"]
    if a.limit:
        new = new[:a.limit]
        keep = set(new)
        man = [r for r in man if r["src"] != "new" or r["uniprot"] in keep]
    log(f"清单 {len(man)} 个蛋白，要新算 {len(new)}；设备 {dev}")
    have = R12.fetch_all(new, a.out)
    items = [(i, have[i]["seq"]) for i in new if i in have and len(have[i]["seq"]) >= 20]
    done = R12.embed_all(items, a.out, dev)
    with open(f"{a.out}/pu_new_meta.tsv", "w") as f:
        f.write("uniprot\tname\tgene\torganism\tlength\tobs\n")
        for i in new:
            if i in have:
                v = have[i]
                f.write(f"{i}\t{v['name']}\t{v['gene']}\t{v['org']}\t{len(v['seq'])}\t{int(bool(v['obs']))}\n")
    # 已有的嵌入
    uni = {}
    for fn in sorted(glob.glob(f"{a.data}/uni_shards/emb_shard_*.npz")):
        z = np.load(fn)
        for acc, e in zip(z["acc"], z["emb"]):
            uni[str(acc)] = e
    bg = np.load(f"{a.data}/bg_emb_L18.npy", mmap_mode="r")
    sq = np.load(f"{a.data}/emb_data_L18.npy", mmap_mode="r")
    log(f"已有嵌入：宇宙 {len(uni)}，Swiss-Prot 参考集 {bg.shape}，PPMC 序列表 {sq.shape}")
    rows, E = [], []
    for r in man:
        u, s = r["uniprot"], r["src"]
        e = uni.get(u) if s == "uni" else bg[int(r["src_row"])] if s == "bg" else sq[int(r["src_row"])] if s == "seqs" else done.get(u)
        if e is None:
            continue
        rows.append(r)
        E.append(np.asarray(e, np.float32))
    E = np.vstack(E)
    assert not np.isnan(E).any()
    log(f"拼好：{E.shape}（清单里 {len(man) - len(rows)} 个没有嵌入，丢掉）")
    from sklearn.decomposition import PCA
    mu, sd = E.mean(0), E.std(0) + 1e-6
    Z = PCA(n_components=64, random_state=0).fit_transform((E - mu) / sd).astype(np.float32)
    Z -= Z.mean(0)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    import umap
    t0 = time.time()
    xy = umap.UMAP(n_neighbors=30, min_dist=0.25, metric="cosine", random_state=0, low_memory=True).fit_transform(Z).astype(np.float32)
    log(f"UMAP {Z.shape} 用时 {time.time() - t0:.0f}s")
    np.savez(f"{a.out}/pu_map.npz", acc=np.array([r["uniprot"] for r in rows]), xy=xy, src=np.array([r["src"] for r in rows]))
    log("PU_FINISHED")


if __name__ == "__main__":
    main()
