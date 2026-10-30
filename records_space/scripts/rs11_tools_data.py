#!/usr/bin/env python3
"""rs11（本机，纯标准库）：把各工具 / 数据库的条目对到 records 空间，生成 explorer/tools_data.json。
- 带条件的工具：LLPSense、Droppler、RNAPSEC（条目 = 记录：蛋白 + 条件）；还有 LLPSDB v2、PhaSepDB 3 试点、Bremer 2022、Ausserwöger 2025 这几个数据来源。
- 只有蛋白、没有条件的：PPMC（Pintado-Grima 2025，蛋白级置信数据集）及其类别；
  以及基准测试会话（benchmarking_llps）里 11 个预测工具的训练集（training_sets_long.tsv）：正例和（图里至少有 15 个蛋白的）负例。
  记录只按“蛋白在不在这个集合里”对上（用 UniProt 号，两个号的条目只看第一个，因为序列是第一个号的）。**不引入新蛋白**：不在图里的训练集蛋白只统计，不加。
输出：list（下拉选项与说明）、mask（每条记录一个位掩码）、entries（试新实验里可载入的条目 = 记录下标）。"""
import csv
import json
import re
from collections import Counter, defaultdict

csv.field_size_limit(10 ** 9)
BASE = "/Users/lynn/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Projects/LLPS/condition_llps"
ROOT = f"{BASE}/records_space_2026-10-29"

rows = list(csv.DictReader(open(f"{ROOT}/out/rs_plot_table.tsv"), delimiter="\t"))
n = len(rows)
UNI = re.compile(r"[A-Z][0-9][A-Z0-9]{3}[0-9](?:[A-Z][A-Z0-9]{2}[0-9])?")


def first_id(u):
    for x in re.split(r"[,;+ ]+", u or ""):
        x = x.strip().split("-")[0]
        if UNI.fullmatch(x):
            return x
    return None


def all_ids(u):
    return {x.strip().split("-")[0] for x in re.split(r"[,;+ ]+", u or "") if UNI.fullmatch(x.strip().split("-")[0])}


def psid(rid):
    m = re.search(r"(PS\d{8})", rid)
    return m.group(1) if m else None


# Droppler 补充表 S2 里的 LLPSDB 实验编号
dr_ids = set()
for i, line in enumerate(open(f"{BASE}/droppler/btab350_supplementary_data/Suppl_TableS2.csv")):
    if i:
        dr_ids.add(re.split(r"[,\t]", line.strip())[0])

# PPMC 的类别：world_and_cluster_maps 的序列表里 ppmc_cls ∈ NP/ND/CE/DE/C_D
ppmc = defaultdict(set)
for s in csv.DictReader(open(f"{BASE}/world_and_cluster_maps_2026-10-27/data/seqs.tsv"), delimiter="\t"):
    c = s["ppmc_cls"]
    if c in ("NP", "ND", "CE", "DE", "C_D"):
        for u in all_ids(s["uniprots"]):
            ppmc[c].add(u)
ppmc_all = set().union(*ppmc.values())
print("PPMC proteins by class", {k: len(v) for k, v in ppmc.items()}, "all", len(ppmc_all))

CT, DB, PR = "Tools with conditions", "Databases with conditions", "Protein only · benchmark set (no conditions)"
TP, TN = "Protein only · tool training positives (no conditions)", "Protein only · tool training negatives (no conditions)"
TOOLS = [
    ("llpsense", "LLPSense", CT, "cond", "#E45A62", lambda r, i: r["source"] == "LLPSense",
     "Entries are rows of LLPSense's own dataset: a protein plus the conditions it was tested under."),
    ("droppler", "Droppler", CT, "cond", "#E7AC24", lambda r, i: r["source"].startswith("LLPSDB_v2") and psid(r["record_id"]) in dr_ids,
     "Entries are the LLPSDB experiments listed in Droppler's supplementary Table S2 (33 ids, 32 of them are in this map)."),
    ("rnapsec", "RNAPSEC", CT, "cond", "#159C98", lambda r, i: r["source"] == "RNAPSEC",
     "Entries are RNAPSEC rows: a protein, an RNA and the conditions."),
    ("llpsdb", "LLPSDB v2", DB, "cond", "#427FDB", lambda r, i: r["source"].startswith("LLPSDB_v2"),
     "All three forms of LLPSDB v2 in this map (download, site, phase-diagram points)."),
    ("phasepdb3", "PhaSepDB 3 (LLM pilot)", DB, "cond", "#A47BDD", lambda r, i: r["source"] == "PhaSepDB3_pilot_LLM",
     "Conditions read from papers by a language model in a pilot, so they carry extraction errors."),
    ("bremer", "Bremer 2022 (csat)", DB, "cond", "#BE629E", lambda r, i: r["source"] == "Bremer2022",
     "Quantitative saturation concentrations of hnRNPA1 low-complexity variants."),
    ("ausserwoger", "Ausserwöger 2025 (csat)", DB, "cond", "#35A45B", lambda r, i: r["source"] == "Ausserwoger2025",
     "Quantitative phase boundaries measured on a few proteins."),
]
for cls, lab in [("", "all proteins"), ("DE", "drivers (DE)"), ("CE", "clients (CE)"), ("C_D", "both (C_D)"), ("NP", "negatives, PDB (NP)"), ("ND", "negatives, DisProt (ND)")]:
    S = ppmc_all if cls == "" else ppmc[cls]
    TOOLS.append(("ppmc" + ("_" + cls if cls else ""), "PPMC · " + lab, PR, "prot", "#427FDB", (lambda S: (lambda r, i: first_id(r["uniprot"]) in S))(S),
                  "Protein-level labels from Pintado-Grima et al. 2025 (" + lab + "). There are no conditions in these entries; the light dots are all records of the same proteins."))

# 基准测试会话里 11 个工具的训练集
TRAIN = f"{BASE}/../benchmarking_llps/Benchmark_figure/htp_independence/training_sets_long.tsv"
train = defaultdict(lambda: defaultdict(set))
for r in csv.DictReader(open(TRAIN), delimiter="\t"):
    u = (r["uniprot_acc"] or "").split("-")[0]
    if UNI.fullmatch(u):
        train[r["tool"]][r["role"]].add(u)
LABEL = {"catGRANULE2.0": "catGRANULE 2.0", "Opt_PredLLPS": "OptPredLLPS", "PPT": "PPT-taskA"}
COLOR = {"catGRANULE2.0": "#20B7ED", "FuzDrop": "#EF7182", "LLPhyScore": "#98C93C", "LLPS-predict": "#A47BDD", "Opt_PredLLPS": "#E7AC24", "PPT": "#159C98",
         "ProtGPS": "#D87234", "PSAP": "#427FDB", "PScalpel": "#BE629E", "PSPHunter": "#35A45B", "PSPire": "#E45A62"}
space_ids = {first_id(r["uniprot"]) for r in rows} - {None}
for tool in sorted(train, key=lambda x: LABEL.get(x, x).lower()):
    lab = LABEL.get(tool, tool)
    pos, neg = train[tool].get("positive", set()), train[tool].get("negative", set())
    tid = "tr_" + re.sub(r"[^a-z0-9]+", "", tool.lower())
    TOOLS.append((tid + "_pos", lab + " · training positives", TP, "prot", COLOR[tool], (lambda S: (lambda r, i: first_id(r["uniprot"]) in S))(pos),
                  f"Proteins in the training set of {lab} as listed by the benchmark ({len(pos & space_ids)} of its {len(pos):,} training positives are in this map). The tool scores a protein only, so there are no conditions; the light dots are all records of these proteins."))
    if len(neg & space_ids) >= 15:
        TOOLS.append((tid + "_neg", lab + " · training negatives", TN, "prot", COLOR[tool], (lambda S: (lambda r, i: first_id(r["uniprot"]) in S))(neg),
                      f"Training negatives or unlabelled background of {lab} ({len(neg & space_ids)} of its {len(neg):,} are in this map). No conditions."))
    print(f"[train] {lab:16s} positives {len(pos):5d} in map {len(pos & space_ids):4d} | negatives {len(neg):6d} in map {len(neg & space_ids):4d}")

mask = [0] * n
lst, entries = [], {}
for t, (tid, label, group, kind, color, test, note) in enumerate(TOOLS):
    hit = [i for i, r in enumerate(rows) if test(r, i)]
    for i in hit:
        mask[i] |= 1 << t
    prots = {first_id(rows[i]["uniprot"]) for i in hit} - {None}
    # 试新实验里可载入的条目（= 记录下标）：带条件的按"条目"去重，只有蛋白的按蛋白去重
    ent, seen = [], set()
    if kind == "cond":
        for i in hit:
            r = rows[i]
            parts = r["record_id"].split(":")
            key = ":".join(parts[:2]) if tid in ("rnapsec", "phasepdb3", "ausserwoger") else (parts[0] + ":" + psid(r["record_id"]) if tid in ("llpsdb",) and r["source"] == "LLPSDB_v2_phase_diagram" else r["record_id"])
            if key not in seen:
                seen.add(key); ent.append(i)
    else:
        # 每个蛋白取序列记录最多的那条构建体的第一条记录
        nrec = Counter(r["seq_row"] for r in rows)
        best = {}
        for i in hit:
            u = first_id(rows[i]["uniprot"])
            if u is None or rows[i]["seq_row"] == "-1":
                continue
            cur = best.get(u)
            if cur is None or nrec[rows[i]["seq_row"]] > nrec[rows[cur]["seq_row"]]:
                best[u] = i
        ent = sorted(best.values())
    entries[tid] = ent
    lst.append({"id": tid, "label": label, "group": group, "kind": kind, "color": color, "note": note, "n_rec": len(hit), "n_ent": len(ent), "n_prot": len(prots)})
    print(f"{label:28s} {kind:5s} records {len(hit):5d} entries {len(ent):5d} proteins {len(prots):4d}")

assert len(TOOLS) <= 32, len(TOOLS)
json.dump({"list": lst, "mask": mask, "entries": entries}, open(f"{ROOT}/explorer/tools_data.json", "w"), ensure_ascii=False, separators=(",", ":"))
print("tools_data.json written; bytes", len(json.dumps({"list": lst, "mask": mask, "entries": entries})))
