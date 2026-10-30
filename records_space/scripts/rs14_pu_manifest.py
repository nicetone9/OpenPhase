#!/usr/bin/env python3
"""rs14（本机，纯标准库）：负例 / 未标注（PU）页的蛋白清单。
集合：28 个——正例 16 个（11 个工具的训练正例、PPMC 的 DE/CE/C_D、LTP、HTP），负例 12 个（10 个工具的训练负例、PPMC 的 NP/ND）。
背景（U）：world_and_cluster_maps 的 Swiss-Prot 参考集 16,000 条（跨界，已算好 ESMC-600M 第 18 层）。
每个蛋白的嵌入来源，优先级 uni（蛋白宇宙那次 Colab 的输出）> bg（Swiss-Prot 参考集）> seqs（PPMC 全长序列）> new（还没算，要上 Colab 取序列并算）。
输出 data/pu/：pu_manifest.tsv、pu_new_ids.tsv（rs12 的取序列与嵌入函数的输入格式）、sets.json、bg_acc.txt、seqs_rows.tsv。"""
import csv
import json
import os
import re
from collections import Counter, defaultdict

csv.field_size_limit(10 ** 9)
BASE = "/Users/lynn/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Projects/LLPS/condition_llps"
ROOT = f"{BASE}/records_space_2026-10-29"
BENCH = f"{BASE}/../benchmarking_llps"
WORLD = f"{BASE}/world_and_cluster_maps_2026-10-27/data"
UNI = re.compile(r"[A-Z][0-9][A-Z0-9]{3}[0-9](?:[A-Z][A-Z0-9]{2}[0-9])?")


def ids_of(u):
    return [x.strip().split("-")[0] for x in re.split(r"[,;+ ]+", u or "") if UNI.fullmatch(x.strip().split("-")[0])]


# ---- 集合
seqs = list(csv.DictReader(open(f"{WORLD}/seqs.tsv"), delimiter="\t"))
ppmc = defaultdict(set)
seq_row = {}                                   # 蛋白 -> emb_data_L18.npy 的行号（只取 PPMC 全长序列那些行）
for i, s in enumerate(seqs):
    if s["family"] == "ppmc":
        for u in ids_of(s["uniprots"]):
            seq_row.setdefault(u, i)
    if s["ppmc_cls"] in ("NP", "ND", "CE", "DE", "C_D"):
        for u in ids_of(s["uniprots"]):
            ppmc[s["ppmc_cls"]].add(u)
bench = {}
for name in ("ltp", "htp"):
    S = set()
    for line in open(f"{BENCH}/Benchmark_general/{name}.fasta"):
        if line.startswith(">"):
            m = UNI.search(line)
            if m:
                S.add(m.group(0))
    bench[name] = S
train = defaultdict(lambda: defaultdict(set))
for r in csv.DictReader(open(f"{BENCH}/Benchmark_figure/htp_independence/training_sets_long.tsv"), delimiter="\t"):
    u = (r["uniprot_acc"] or "").split("-")[0]
    if UNI.fullmatch(u):
        train[r["tool"]][r["role"]].add(u)
LABEL = {"catGRANULE2.0": "catGRANULE 2.0", "Opt_PredLLPS": "OptPredLLPS", "PPT": "PPT-taskA"}
COLOR = {"catGRANULE2.0": "#20B7ED", "FuzDrop": "#EF7182", "LLPhyScore": "#98C93C", "LLPS-predict": "#A47BDD", "Opt_PredLLPS": "#E7AC24", "PPT": "#159C98",
         "ProtGPS": "#D87234", "PSAP": "#427FDB", "PScalpel": "#BE629E", "PSPHunter": "#35A45B", "PSPire": "#E45A62"}
SETS = []        # (id, label, role, color, members)
for cls, lab in (("DE", "PPMC · drivers (DE)"), ("CE", "PPMC · clients (CE)"), ("C_D", "PPMC · both (C_D)")):
    SETS.append(("ppmc_" + cls, lab, "P", "#427FDB", ppmc[cls]))
SETS.append(("ltp", "LTP benchmark set", "P", "#427FDB", bench["ltp"]))
SETS.append(("htp", "HTP benchmark set", "P", "#427FDB", bench["htp"]))
SETS.append(("ppmc_NP", "PPMC · non-phase-separating, protein-level (NP)", "N", "#5B7A93", ppmc["NP"]))
SETS.append(("ppmc_ND", "PPMC · non-phase-separating, disordered (ND)", "N", "#5B7A93", ppmc["ND"]))
for tool in sorted(train, key=lambda x: LABEL.get(x, x).lower()):
    lab = LABEL.get(tool, tool)
    tid = re.sub(r"[^a-z0-9]+", "", tool.lower())
    if train[tool].get("positive"):
        SETS.append(("tr_" + tid + "_pos", lab + " · training positives", "P", COLOR[tool], train[tool]["positive"]))
    if train[tool].get("negative"):
        SETS.append(("tr_" + tid + "_neg", lab + " · training negatives", "N", COLOR[tool], train[tool]["negative"]))
assert len(SETS) <= 32, len(SETS)

# ---- 各来源已有的蛋白
uni_meta = {}
for r in csv.DictReader(open(f"{ROOT}/data/universe/colab_out/universe_meta.tsv"), delimiter="\t"):
    uni_meta[r["uniprot"]] = r
bg = json.load(open(f"{WORLD}/bg.json"))
bg_row = {e["acc"]: i for i, e in enumerate(bg)}
space_ids = set()
for r in csv.DictReader(open(f"{ROOT}/out/rs_plot_table.tsv"), delimiter="\t"):
    space_ids.update(ids_of(r["uniprot"])[:1])

mask = defaultdict(int)
for b, (_, _, _, _, S) in enumerate(SETS):
    for u in S:
        mask[u] |= 1 << b
P_BITS = sum(1 << b for b, s in enumerate(SETS) if s[2] == "P")
N_BITS = sum(1 << b for b, s in enumerate(SETS) if s[2] == "N")
allp = set(mask) | set(bg_row)
rows, new_ids = [], []
for u in sorted(allp):
    m = mask.get(u, 0)
    role = "U" if not m else ("C" if (m & P_BITS and m & N_BITS) else ("P" if m & P_BITS else "N"))
    if u in uni_meta:
        src, srow, nm, gn, og, ln = "uni", -1, uni_meta[u]["name"], uni_meta[u]["gene"], uni_meta[u]["organism"], uni_meta[u]["length"]
    elif u in bg_row:
        e = bg[bg_row[u]]
        src, srow, nm, gn, og, ln = "bg", bg_row[u], e["entry"], "", e["organism"], e["length"]
    elif u in seq_row:
        s = seqs[seq_row[u]]
        src, srow, nm, gn, og, ln = "seqs", seq_row[u], "", s["gene"], "", s["length"]
    else:
        src, srow, nm, gn, og, ln = "new", -1, "", "", "", ""
        new_ids.append(u)
    rows.append((u, src, srow, m, role, 1 if u in bg_row else 0, 1 if u in space_ids else 0, nm, gn, og, ln))
os.makedirs(f"{ROOT}/data/pu", exist_ok=True)
with open(f"{ROOT}/data/pu/pu_manifest.tsv", "w") as f:
    f.write("uniprot\tsrc\tsrc_row\tmask\trole\tbg\tin_records\tname\tgene\torganism\tlength\n")
    for r in rows:
        f.write("\t".join(str(x).replace("\t", " ") for x in r) + "\n")
with open(f"{ROOT}/data/pu/pu_new_ids.tsv", "w") as f:
    f.write("uniprot\tmask_lo\tmask_hi\n")
    for u in new_ids:
        f.write(f"{u}\t{mask.get(u, 0)}\t0\n")
json.dump([{"id": s[0], "label": s[1], "role": s[2], "color": s[3], "bit": b, "n": len(s[4])} for b, s in enumerate(SETS)], open(f"{ROOT}/data/pu/sets.json", "w"), ensure_ascii=False, indent=1)
open(f"{ROOT}/data/pu/bg_acc.txt", "w").write("\n".join(e["acc"] for e in bg))
print("sets", len(SETS), "| proteins", len(rows), "| roles", dict(Counter(r[4] for r in rows)), "| sources", dict(Counter(r[1] for r in rows)))
print("new to embed:", len(new_ids), "| of which in the records space:", sum(1 for u in new_ids if u in space_ids))
for b, s in enumerate(SETS):
    print(f"  bit {b:2d} {s[2]} {s[1]:50s} {len(s[4]):6d}")
