#!/usr/bin/env python3
"""rs19（本机，纯标准库，只做文本处理和 UniProt 注释下载）：给蛋白宇宙里"没有记录"的工具蛋白推断它们的系统类型。

规则（一眼能解释的那种）：UniProt 关键词或 GO 分子功能里有 RNA-binding / DNA-binding。
  RNA-binding 且非 DNA-binding   -> protein(1) + RNA
  DNA-binding 且非 RNA-binding   -> protein(1) + DNA
  两者都有                        -> protein(1) + RNA（RNA 记录多：2,511 对 662；另记 both 标志）
  都没有（或查不到注释）           -> protein(1)
再用"有记录的蛋白"检验这条规则：UniProt 说 RNA-binding 的蛋白里，真有 + RNA 记录的占多少，对比其余蛋白。

输出（data/universe/）：
  universe_system.json   {"order": [uniprot...], "sys": [0/1/2...], "rna": [0/1...], "dna": [0/1...], "annotated": [0/1...]}
  system_eval.json       规则在有记录的蛋白上的 2x2 表、精确率、召回率、基线率，以及宇宙里的各类数量
用法：python3 -I scripts/rs19_universe_system.py
"""
import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request

ROOT = "/Users/lynn/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Projects/LLPS/condition_llps/records_space_2026-10-29"
UNI = f"{ROOT}/data/universe/universe_ids.tsv"
REC = f"{ROOT}/data/rs_records.tsv"
OUT = f"{ROOT}/data/universe"
ACC = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$")
URL = "https://rest.uniprot.org/uniprotkb/accessions?"
GO_RNA = "GO:0003723"     # RNA binding
GO_DNA = "GO:0003677"     # DNA binding


def base(acc):
    return re.sub(r"-\d+$", "", acc.strip())


def accs_of(field):
    return [base(t) for t in re.split(r"[;,+\s]+", field) if ACC.match(base(t))]


def fetch(batch):
    q = urllib.parse.urlencode({"accessions": ",".join(batch), "fields": "accession,keyword,go_f", "format": "tsv", "size": "500"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(URL + q, timeout=60) as r:
                return r.read().decode("utf-8")
        except Exception as e:  # 网络抖动：退避重试
            print("  retry", attempt + 1, type(e).__name__, e, flush=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("UniProt request failed")


def annotate(accessions):
    """accession -> (rna, dna)；查不到的不在字典里"""
    got, todo = {}, sorted(set(accessions))
    for i in range(0, len(todo), 200):
        text = fetch(todo[i:i + 200])
        for line in text.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            kw, go = parts[1], parts[2]
            got[parts[0]] = (int("RNA-binding" in kw or GO_RNA in go), int("DNA-binding" in kw or GO_DNA in go))
        print(f"  {min(i + 200, len(todo))}/{len(todo)} accessions, {len(got)} annotated", flush=True)
        time.sleep(0.4)
    return got


def main():
    with open(UNI, encoding="utf-8") as f:
        order = [row["uniprot"] for row in csv.DictReader(f, delimiter="\t")]
    # 有记录的蛋白：每个 UniProt 一行，记录它有没有 + RNA / + DNA 的记录
    has = {}
    with open(REC, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            for a in accs_of(row["uniprot"]):
                h = has.setdefault(a, [0, 0, 0])
                h[0] += 1
                if row["na_class"] == "RNA":
                    h[1] += 1
                elif row["na_class"] == "DNA":
                    h[2] += 1
    print("宇宙蛋白", len(order), "；有记录的蛋白", len(has), flush=True)
    ann = annotate([base(a) for a in order] + list(has))

    # 检验：规则对"有记录的蛋白"的预测力（只用有 UniProt 注释的）
    ev = {"proteins_with_records": len(has), "annotated": sum(1 for a in has if a in ann)}
    for name, idx, col in (("RNA", 0, 1), ("DNA", 1, 2)):
        t = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for a, h in has.items():
            if a not in ann:
                continue
            pred, true = ann[a][idx] == 1, h[col] > 0
            t["tp" if pred and true else "fp" if pred else "fn" if true else "tn"] += 1
        n = sum(t.values())
        ev[name] = dict(t, precision=round(t["tp"] / max(t["tp"] + t["fp"], 1), 3), recall=round(t["tp"] / max(t["tp"] + t["fn"], 1), 3),
                        base_rate=round((t["tp"] + t["fn"]) / max(n, 1), 3), rate_if_not_annotated=round(t["fn"] / max(t["fn"] + t["tn"], 1), 3))

    sys_, rna, dna, annotated = [], [], [], []
    for a in order:
        r, d = ann.get(base(a), (0, 0))
        rna.append(r); dna.append(d); annotated.append(int(base(a) in ann))
        sys_.append(1 if r else 2 if d else 0)
    ev["universe"] = {"n": len(order), "annotated": sum(annotated), "protein(1)": sys_.count(0), "protein(1) + RNA": sys_.count(1), "protein(1) + DNA": sys_.count(2), "both": sum(1 for r, d in zip(rna, dna) if r and d)}
    with open(f"{OUT}/universe_system.json", "w") as f:
        json.dump({"order": order, "sys": sys_, "rna": rna, "dna": dna, "annotated": annotated}, f, separators=(",", ":"))
    with open(f"{OUT}/system_eval.json", "w") as f:
        json.dump(ev, f, indent=1)
    print(json.dumps(ev, indent=1), flush=True)


if __name__ == "__main__":
    main()
