"""rs12（Colab GPU）：蛋白宇宙。给各工具提到的、不在 records 空间里的蛋白：取 UniProt 序列 → ESMC-600M 第 18 层残基均值
→ 用数据集 1,338 条序列拟合的 z-score + PCA(64) 投到同一个 P 块空间（Zn）→ 与数据集序列一起做 UMAP。

口径同 rs02 / rslib.build_inputs：层号 = hidden_states 下标，在残基上取均值，排除特殊符号（id 0,1,2,3,31,32）与 padding；
e_mu/e_sd = 数据集 1,338 条序列的逐维均值与标准差（+1e-6），PCA(64, random_state=0) 在 z-score 后的数据集嵌入上拟合，z_mean 为 PCA 输出的均值，再做 L2 归一化。
可续跑：序列文件和嵌入分片都在 out 下，重跑会跳过已有的。

用法：python rs12_universe.py --data /content/rs12/data --out /content/rs12/out
输入：data/universe_ids.tsv、data/esmc_L18.npy（主表 1,299 条）、data/esmc_L18_new.npy（csat 独有序列）、data/rs_sequences.tsv
输出：out/universe.fasta、out/universe_meta.tsv、out/missing_ids.txt、out/emb_shard_XXXX.npz、out/universe.npz、out/check.json
"""
import argparse
import csv
import json
import os
import re
import time

import numpy as np

csv.field_size_limit(10 ** 9)
LAYER, MAXLEN, TOKEN_BUDGET, SHARD = 18, 2000, 14000, 1500


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------ 1. 取序列
def parse_fasta(text):
    """逐行解析：只有行首的 > 才算一条记录的开头（蛋白名里可能带 >）；头部对不上的记录跳过，不报错。"""
    out, acc, head, seq = {}, None, "", []

    def flush():
        if acc and seq:
            name = head.split(" OS=")[0].strip()
            g = re.search(r" GN=(\S+)", head)
            o = re.search(r" OS=(.+?)(?: OX=| GN=| PE=| SV=| OBS=|$)", head)
            out[acc] = dict(name=name, gene=g.group(1) if g else "", org=o.group(1) if o else "",
                            seq="".join("".join(seq).split()), obs=" OBS=1" in head)
    for line in text.splitlines():
        if line.startswith(">"):
            flush()
            acc, seq = None, []
            m = re.match(r">(?:sp|tr)\|([A-Z0-9]+)\|\S+\s*(.*)", line)
            if m:
                acc, head = m.group(1), m.group(2)
        elif acc:
            seq.append(line.strip())
    flush()
    return out


def uniparc_one(s, acc):
    """UniProtKB 里已经删除的条目（老 TrEMBL 编号）从 UniParc 取序列；返回 (acc, 记录或 None)。"""
    for attempt in range(3):
        try:
            r = s.get("https://rest.uniprot.org/uniparc/search", timeout=60,
                      params={"query": acc, "fields": "upi,organism,length,gene,protein,sequence,accession", "format": "tsv", "size": 3})
            if r.status_code != 200:
                time.sleep(2 * (attempt + 1)); continue
            lines = [l for l in r.text.splitlines() if l.strip()]
            if len(lines) < 2:
                return acc, None
            col = {h: k for k, h in enumerate(lines[0].split("\t"))}
            rows = [l.split("\t") for l in lines[1:]]
            pick = None
            if len(rows) == 1:
                pick = rows[0]
            else:
                hit = [x for x in rows if acc in (x[col["UniProtKB"]].replace(" ", "").split(";") if "UniProtKB" in col and len(x) > col["UniProtKB"] else [])]
                pick = hit[0] if len(hit) == 1 else None
            if pick is None or len(pick) <= col["Sequence"]:
                return acc, None
            live = acc in pick[col["UniProtKB"]].replace(" ", "").split(";") if "UniProtKB" in col and len(pick) > col["UniProtKB"] else False
            name = pick[col["Protein names"]].split(";")[0].strip() if "Protein names" in col and len(pick) > col["Protein names"] else ""
            gene = pick[col["Gene names"]].split(" ")[0].strip().rstrip(";") if "Gene names" in col and len(pick) > col["Gene names"] else ""
            org = pick[col["Organisms"]].split(";")[0].strip() if "Organisms" in col and len(pick) > col["Organisms"] else ""
            return acc, dict(name=name or acc, gene=gene, org=org, seq="".join(pick[col["Sequence"]].split()), obs=not live)
        except Exception as e:                                       # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return acc, None


def fetch_all(ids, out_dir):
    import requests
    from concurrent.futures import ThreadPoolExecutor
    fasta = f"{out_dir}/universe.fasta"
    have = {}
    if os.path.exists(fasta):
        have = parse_fasta(open(fasta).read())
    todo = [i for i in ids if i not in have]
    log(f"序列：已有 {len(have)}，要取 {len(todo)}")
    s = requests.Session()

    def save(got):
        with open(fasta, "a") as f:
            for a, v in got.items():
                f.write(f">sp|{a}|X {v['name']} OS={v['org']} OX=0 GN={v['gene']}{' OBS=1' if v['obs'] else ''}\n{v['seq']}\n")
    # 1) 批量取：每批 100 个；一批失败不影响别的批，漏掉的下面逐个补
    for k in range(0, len(todo), 100):
        batch = todo[k:k + 100]
        got = {}
        for attempt in range(3):
            try:
                r = s.get("https://rest.uniprot.org/uniprotkb/accessions", params={"accessions": ",".join(batch), "format": "fasta"}, timeout=120)
                if r.status_code == 200:
                    got = {a: v for a, v in parse_fasta(r.text).items() if a in set(batch)}
                    break
                log(f"  批 {k // 100}：HTTP {r.status_code}，重试")
            except Exception as e:                                   # noqa: BLE001
                log(f"  批 {k // 100}：{repr(e)[:100]}，重试")
            time.sleep(3 * (attempt + 1))
        have.update(got)
        save(got)
        if (k // 100) % 5 == 0 or k + 100 >= len(todo):
            log(f"  批量 {min(k + 100, len(todo))}/{len(todo)}：累计拿到 {len(have)}")
        time.sleep(0.3)
    # 2) UniProtKB 里没有的（已删除的老 TrEMBL 编号等）：逐个到 UniParc 取
    rest = [i for i in ids if i not in have]
    log(f"批量没取到 {len(rest)} 个，改走 UniParc")
    n_ok = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for k, (a, v) in enumerate(ex.map(lambda x: uniparc_one(s, x), rest)):
            if v:
                have[a] = v; save({a: v}); n_ok += 1
            if (k + 1) % 200 == 0:
                log(f"  UniParc {k + 1}/{len(rest)}：拿到 {n_ok}")
    miss = [i for i in ids if i not in have]
    open(f"{out_dir}/missing_ids.txt", "w").write("\n".join(miss))
    n_obs = sum(1 for i in ids if i in have and have[i]["obs"])
    log(f"序列完成：拿到 {len(ids) - len(miss)}（其中 UniProtKB 已删除、序列来自 UniParc 的 {n_obs} 个），没拿到 {len(miss)}")
    return have


# ------------------------------------------------------------------ 2. 嵌入
def load_model(dev):
    from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
    try:
        model = EsmcForMaskedLM.from_pretrained("biohub/ESMC-600M", device=dev)
    except TypeError:
        model = EsmcForMaskedLM.from_pretrained("biohub/ESMC-600M").to(dev)
    return model.eval(), EsmcTokenizer()


def embed_batch(model, tok, seqs, dev, amp):
    import torch
    enc = {k: v.to(dev) for k, v in tok(seqs, return_tensors="pt", padding=True).items()}
    with torch.inference_mode():
        if amp is not None:
            with torch.autocast(device_type="cuda", dtype=amp):
                hs = model(**enc, output_hidden_states=True).hidden_states
        else:
            hs = model(**enc, output_hidden_states=True).hidden_states
        h = hs[LAYER].float()                                               # [B, T, D]
        special = torch.tensor([0, 1, 2, 3, 31, 32], device=dev)
        m = (enc["attention_mask"].bool() & ~torch.isin(enc["input_ids"], special)).float()[:, :, None]
        return ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()


def embed_all(items, out_dir, dev):
    import torch
    shards = sorted(f for f in os.listdir(out_dir) if f.startswith("emb_shard_") and f.endswith(".npz"))
    done = {}
    for f in shards:
        z = np.load(f"{out_dir}/{f}", allow_pickle=False)
        for a, e in zip(z["acc"], z["emb"]):
            done[str(a)] = e
    log(f"嵌入：已有 {len(done)} 条（{len(shards)} 个分片）")
    todo = sorted([(a, s[:MAXLEN]) for a, s in items if a not in done], key=lambda x: -len(x[1]))
    if not todo:
        return done
    amp = None
    if dev == "cuda":
        amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else None      # T4 没有 bf16：用 fp32，慢但稳
    t0 = time.time()
    model, tok = load_model(dev)
    log(f"模型载入 {time.time() - t0:.0f}s；设备 {dev}，autocast {amp}；待算 {len(todo)} 条，最长 {len(todo[0][1])} aa")
    buf_a, buf_e, n_done, t1, k = [], [], 0, time.time(), len(shards)
    i = 0
    while i < len(todo):
        L = len(todo[i][1]); bs = max(1, TOKEN_BUDGET // (L + 2)); batch = todo[i:i + bs]
        while True:
            try:
                e = embed_batch(model, tok, [s for _, s in batch], dev, amp); break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); bs = max(1, bs // 2); batch = todo[i:i + bs]
                log(f"  显存不够，批量降到 {bs}")
        assert not np.isnan(e).any()
        buf_a += [a for a, _ in batch]; buf_e.append(e.astype(np.float16)); n_done += len(batch); i += len(batch)
        if sum(len(x) for x in buf_e) >= SHARD or i >= len(todo):
            np.savez(f"{out_dir}/emb_shard_{k:04d}.npz", acc=np.array(buf_a), emb=np.vstack(buf_e)); k += 1
            for a, e_ in zip(buf_a, np.vstack(buf_e)):
                done[a] = e_
            buf_a, buf_e = [], []
            el = time.time() - t1
            log(f"  [{n_done}/{len(todo)}] 当前长度 {L}，{n_done / el:.1f} 条/秒，已用 {el / 60:.1f} 分，预计还要 {(len(todo) - n_done) / max(n_done / el, 1e-6) / 60:.1f} 分")
    return done


# ------------------------------------------------------------------ 3. 投到 P 块空间并做 UMAP
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids_rows = list(csv.DictReader(open(f"{a.data}/universe_ids.tsv"), delimiter="\t"))
    if a.limit:
        ids_rows = ids_rows[:a.limit]
    ids = [r["uniprot"] for r in ids_rows]
    log(f"宇宙蛋白 {len(ids)}；设备 {dev}")
    have = fetch_all(ids, a.out)
    items = [(i, have[i]["seq"]) for i in ids if i in have and len(have[i]["seq"]) >= 20]
    # 核对：用 6 条数据集序列的新嵌入对照主表旧嵌入
    seq_rows = list(csv.DictReader(open(f"{a.data}/rs_sequences.tsv"), delimiter="\t"))
    main = np.load(f"{a.data}/esmc_L18.npy").astype(np.float32)
    chk = [r for r in seq_rows if r["origin"] == "main" and 80 <= int(float(r["length"])) <= 300][:6]
    chk_items = [(f"CHK{r['seq_row']}", "".join(r["sequence"].upper().split())) for r in chk]
    done = embed_all(items + chk_items, a.out, dev)
    cs = []
    for r in chk:
        e, ref = done[f"CHK{r['seq_row']}"].astype(np.float32), main[int(r["main_seq_row"])]
        cs.append(float(e @ ref / (np.linalg.norm(e) * np.linalg.norm(ref))))
    json.dump({"check_cosines": cs, "device": dev}, open(f"{a.out}/check.json", "w"))
    log(f"核对（与主表旧嵌入的余弦）：{[round(c, 5) for c in cs]}")

    # 基准：数据集 1,338 条序列的 z-score + PCA(64)（同 rslib.build_inputs）
    from sklearn.decomposition import PCA
    n_new = sum(1 for r in seq_rows if r["origin"] == "csat_only")
    e_new = np.load(f"{a.data}/esmc_L18_new.npy").astype(np.float32)
    Eb = np.vstack([main, e_new[:n_new]])
    assert Eb.shape[0] == len(seq_rows), (Eb.shape, len(seq_rows))
    mu, sd = Eb.mean(0), Eb.std(0) + 1e-6
    pca = PCA(n_components=64, random_state=0)
    Zb = pca.fit_transform((Eb - mu) / sd).astype(np.float32)
    zm = Zb.mean(0)
    Zb -= zm
    Zn_ds = Zb / (np.linalg.norm(Zb, axis=1, keepdims=True) + 1e-9)
    ok = [i for i in ids if i in done]
    Eu = np.vstack([done[i] for i in ok]).astype(np.float32)
    Zu = pca.transform((Eu - mu) / sd).astype(np.float32) - zm
    Zn_u = Zu / (np.linalg.norm(Zu, axis=1, keepdims=True) + 1e-9)
    log(f"投影完成：数据集 {Zn_ds.shape}，宇宙 {Zn_u.shape}")
    # 每个宇宙蛋白最近的数据集序列
    S = Zn_u @ Zn_ds.T
    nn_idx, nn_cos = S.argmax(1).astype(np.int16), S.max(1).astype(np.float16)
    import umap
    X = np.vstack([Zn_ds, Zn_u])
    t0 = time.time()
    xy = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", random_state=0, low_memory=True, verbose=False).fit_transform(X).astype(np.float32)
    log(f"UMAP {X.shape} 用时 {time.time() - t0:.0f}s")
    meta = {i: have[i] for i in ok}
    row_of = {r["uniprot"]: r for r in ids_rows}
    np.savez(f"{a.out}/universe.npz", acc=np.array(ok), name=np.array([meta[i]["name"] for i in ok]), gene=np.array([meta[i]["gene"] for i in ok]),
             org=np.array([meta[i]["org"] for i in ok]), length=np.array([len(meta[i]["seq"]) for i in ok], np.int32),
             trunc=np.array([len(meta[i]["seq"]) > MAXLEN for i in ok]), obs=np.array([bool(meta[i]["obs"]) for i in ok]), mask_lo=np.array([int(row_of[i]["mask_lo"]) for i in ok], np.uint32),
             mask_hi=np.array([int(row_of[i]["mask_hi"]) for i in ok], np.uint32), zn=Zn_u.astype(np.float16), nn_idx=nn_idx, nn_cos=nn_cos,
             xy_u=xy[len(Zn_ds):], xy_ds=xy[:len(Zn_ds)], zn_ds=Zn_ds.astype(np.float16))
    with open(f"{a.out}/universe_meta.tsv", "w") as f:
        f.write("uniprot\tname\tgene\torganism\tlength\n")
        for i in ok:
            f.write(f"{i}\t{meta[i]['name']}\t{meta[i]['gene']}\t{meta[i]['org']}\t{len(meta[i]['seq'])}\n")
    log("UNIVERSE_FINISHED")


if __name__ == "__main__":
    main()
