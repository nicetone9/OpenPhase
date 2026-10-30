"""rs02（Colab）：给 csat 独有的序列算 ESMC-600M 第 18 层残基均值，并用 3 条主表序列核对与旧嵌入的一致性。

口径同 embedding_space_perturbation_2026-10-23/colab/embed_lib.py：层号 = hidden_states 下标，在残基上取均值，排除特殊符号（id 0,1,2,3,31,32）与 padding。
CPU 也能跑（39 条约 137 aa + 3 条检查），逐条前向。esm 包由调用方先装（pip install esm）。
用法：python rs02_embed_new_seqs.py --data /content/rs/data --out /content/rs/out
输出：out/esmc_L18_new.npy（float16，行序 = rs_new_sequences.tsv 的行序，含最后 3 条检查序列）、out/esmc_check.json
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

csv.field_size_limit(10 ** 9)
LAYER = 18


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_model(dev):
    from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
    try:
        model = EsmcForMaskedLM.from_pretrained("biohub/ESMC-600M", device=dev)
    except TypeError:
        model = EsmcForMaskedLM.from_pretrained("biohub/ESMC-600M").to(dev)
    model = model.eval()
    if dev == "cpu":
        model = model.float()
    return model, EsmcTokenizer()


@torch.inference_mode()
def pooled(model, tok, seq, dev):
    enc = {k: v.to(dev) for k, v in tok([seq], return_tensors="pt", padding=True).items()}
    hs = model(**enc, output_hidden_states=True).hidden_states
    hs = hs if torch.is_tensor(hs) else torch.stack(list(hs), 0)                  # [NL, 1, T, D]
    special = torch.tensor([0, 1, 2, 3, 31, 32], device=dev)
    m = (enc["attention_mask"].bool() & ~torch.isin(enc["input_ids"], special)).float()[None, :, :, None]
    return ((hs.float() * m).sum(2) / m.sum(2).clamp(min=1))[LAYER, 0].cpu().numpy()   # [D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--main-npy", default=None, help="主表 esmc_L18.npy（float16，行序 = rs_sequences 的 main_seq_row）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(f"{a.data}/rs_new_sequences.tsv"), delimiter="\t"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"{len(rows)} 条序列（new {sum(r['kind'] == 'new' for r in rows)}，check {sum(r['kind'] == 'check' for r in rows)}）；设备 {dev}")
    t0 = time.time()
    model, tok = load_model(dev)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"模型载入 {time.time() - t0:.0f}s；{nparam:.0f}M 参数，dtype {next(model.parameters()).dtype}")
    out = np.zeros((len(rows), model.config.hidden_size), np.float32)
    for i, r in enumerate(rows):
        t1 = time.time()
        out[i] = pooled(model, tok, "".join(r["sequence"].upper().split()), dev)
        log(f"  [{i + 1}/{len(rows)}] {r['kind']} len={r['length']} {time.time() - t1:.1f}s")
    assert not np.isnan(out).any()
    np.save(f"{a.out}/esmc_L18_new.npy", out.astype(np.float16))
    rep = {"device": dev, "dtype": str(next(model.parameters()).dtype), "n_rows": len(rows), "check": []}
    if a.main_npy and os.path.exists(a.main_npy):
        old = np.load(a.main_npy).astype(np.float32)
        for i, r in enumerate(rows):
            if r["kind"] != "check":
                continue
            ref = old[int(r["main_seq_row"])]
            cos = float(out[i] @ ref / (np.linalg.norm(out[i]) * np.linalg.norm(ref)))
            rep["check"].append(dict(main_seq_row=int(r["main_seq_row"]), length=int(r["length"]), cosine=cos,
                                     max_abs_diff=float(np.abs(out[i] - ref).max()), rel_l2=float(np.linalg.norm(out[i] - ref) / np.linalg.norm(ref))))
        log("一致性检查：" + json.dumps(rep["check"]))
    json.dump(rep, open(f"{a.out}/esmc_check.json", "w"), indent=1)
    log("完成")


if __name__ == "__main__":
    main()
