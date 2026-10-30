#!/usr/bin/env python3
"""rs11（本机，纯标准库）：把各工具 / 数据库 / 基准集的条目对到 records 空间，生成 explorer/tools_data.json 与 data/universe/universe_ids.tsv。
- 带条件的工具：LLPSense、Droppler、RNAPSEC（条目 = 记录：蛋白 + 条件）；数据库：LLPSDB v2、PhaSepDB 3 试点、Bremer 2022、Ausserwöger 2025。
- 只有蛋白、没有条件的：PPMC（Pintado-Grima 2025）及其类别，基准会话（benchmarking_llps）里的 LTP / HTP 基准集，
  以及 11 个预测工具的全部训练集蛋白（正例和负例；training_sets_long.tsv）。记录按“蛋白在不在这个集合里”对上
  （用 UniProt 号，两个号的条目只看第一个，因为序列是第一个号的）。
- 不在 records 空间里的集合蛋白写进 universe_ids.tsv（带位掩码），交给 Colab 取序列、算嵌入、画蛋白宇宙。
掩码是两个 32 位字（lo = 工具 0–31，hi = 工具 32–63）。"""
import csv
import json
import os
import re
from collections import Counter, defaultdict

csv.field_size_limit(10 ** 9)
BASE = "/Users/lynn/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Projects/LLPS/condition_llps"
ROOT = f"{BASE}/records_space_2026-10-29"
BENCH = f"{BASE}/../benchmarking_llps"

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


space_ids = {first_id(r["uniprot"]) for r in rows} - {None}

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
ppmc_all = ppmc["DE"] | ppmc["CE"] | ppmc["C_D"]   # 只要 LLPS 宇宙：阴性（NP/ND）不要

# 基准会话的 LTP / HTP 基准集（测试集）
bench = {}
for name in ("ltp", "htp"):
    S = set()
    for line in open(f"{BENCH}/Benchmark_general/{name}.fasta"):
        if line.startswith(">"):
            m = UNI.search(line)
            if m:
                S.add(m.group(0))
    bench[name] = S

# 11 个工具的训练集
train = defaultdict(lambda: defaultdict(set))
for r in csv.DictReader(open(f"{BENCH}/Benchmark_figure/htp_independence/training_sets_long.tsv"), delimiter="\t"):
    u = (r["uniprot_acc"] or "").split("-")[0]
    if UNI.fullmatch(u):
        train[r["tool"]][r["role"]].add(u)

CT, DB = "Tools with conditions", "Databases with conditions"
PB, TP, TN = "Protein only · benchmark sets (no conditions)", "Protein only · tool training positives (no conditions)", "Protein only · tool training negatives (no conditions)"
LABEL = {"catGRANULE2.0": "catGRANULE 2.0", "Opt_PredLLPS": "OptPredLLPS", "PPT": "PPT-taskA"}
COLOR = {"catGRANULE2.0": "#20B7ED", "FuzDrop": "#EF7182", "LLPhyScore": "#98C93C", "LLPS-predict": "#A47BDD", "Opt_PredLLPS": "#E7AC24", "PPT": "#159C98",
         "ProtGPS": "#D87234", "PSAP": "#427FDB", "PScalpel": "#BE629E", "PSPHunter": "#35A45B", "PSPire": "#E45A62"}

# (id, 标签, 分组, 类型, 颜色, 记录判定, 说明)；类型 cond = 条目是记录；prot = 条目是蛋白（只有蛋白、没有条件）
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
PSETS = {}   # 只有蛋白的集合：id -> UniProt 集合
for cls, lab in [("", "all proteins"), ("DE", "drivers (DE)"), ("CE", "clients (CE)"), ("C_D", "both (C_D)")]:
    S = ppmc_all if cls == "" else ppmc[cls]
    tid = "ppmc" + ("_" + cls if cls else "")
    PSETS[tid] = S
    TOOLS.append((tid, "PPMC · " + lab, PB, "prot", "#427FDB", None,
                  "Protein-level labels from Pintado-Grima et al. 2025 (" + lab + "). There are no conditions in these entries."))
for tid, lab, desc in [("ltp", "LTP benchmark set", "Low-throughput literature positives used as the benchmark's confident positives"), ("htp", "HTP benchmark set", "High-throughput screen proteins used as the benchmark's independent validation set")]:
    PSETS[tid] = bench[tid]
    TOOLS.append((tid, lab, PB, "prot", "#427FDB", None, desc + " (benchmarking_llps). Proteins only, no conditions."))
order = sorted(train, key=lambda x: LABEL.get(x, x).lower())
for role, grp, suffix in (("positive", TP, "training positives"),):
    for tool in order:
        S = train[tool].get(role, set())
        if not S:
            continue
        lab = LABEL.get(tool, tool)
        tid = "tr_" + re.sub(r"[^a-z0-9]+", "", tool.lower()) + ("_pos" if role == "positive" else "_neg")
        PSETS[tid] = S
        TOOLS.append((tid, lab + " · " + suffix, grp, "prot", COLOR[tool], None,
                      f"Proteins in the {suffix} of {lab} as listed by the benchmark ({len(S):,} proteins). The tool scores a protein only, so there are no conditions."))
assert len(TOOLS) <= 64, len(TOOLS)

mask_lo, mask_hi = [0] * n, [0] * n
lst, entries = [], {}
nrec_of_seq = Counter(r["seq_row"] for r in rows)
univ = defaultdict(lambda: [0, 0])      # 空间外的集合蛋白 -> [lo, hi]
for t, (tid, label, group, kind, color, test, note) in enumerate(TOOLS):
    word, bit = (0, 1 << t) if t < 32 else (1, 1 << (t - 32))
    if kind == "cond":
        hit = [i for i, r in enumerate(rows) if test(r, i)]
    else:
        S = PSETS[tid]
        hit = [i for i, r in enumerate(rows) if first_id(r["uniprot"]) in S]
        for u in S - space_ids:
            univ[u][word] |= bit
    for i in hit:
        if word == 0:
            mask_lo[i] |= bit
        else:
            mask_hi[i] |= bit
    prots = {first_id(rows[i]["uniprot"]) for i in hit} - {None}
    ent, seen = [], set()
    if kind == "cond":
        for i in hit:
            r = rows[i]
            parts = r["record_id"].split(":")
            key = ":".join(parts[:2]) if tid in ("rnapsec", "phasepdb3", "ausserwoger") else (parts[0] + ":" + psid(r["record_id"]) if tid == "llpsdb" and r["source"] == "LLPSDB_v2_phase_diagram" else r["record_id"])
            if key not in seen:
                seen.add(key); ent.append(i)
    else:
        best = {}
        for i in hit:
            u = first_id(rows[i]["uniprot"])
            if u is None or rows[i]["seq_row"] == "-1":
                continue
            cur = best.get(u)
            if cur is None or nrec_of_seq[rows[i]["seq_row"]] > nrec_of_seq[rows[cur]["seq_row"]]:
                best[u] = i
        ent = sorted(best.values())
    entries[tid] = ent
    n_out = len(PSETS[tid] - space_ids) if kind == "prot" else 0
    lst.append({"id": tid, "label": label, "group": group, "kind": kind, "color": color, "note": note, "n_rec": len(hit), "n_ent": len(ent), "n_prot": len(prots), "n_out": n_out})
    print(f"{label:34s} {kind:5s} records {len(hit):5d} entries {len(ent):5d} proteins in map {len(prots):4d} | outside the map {n_out:6d}")

json.dump({"list": lst, "mask_lo": mask_lo, "mask_hi": mask_hi, "entries": entries}, open(f"{ROOT}/explorer/tools_data.json", "w"), ensure_ascii=False, separators=(",", ":"))
os.makedirs(f"{ROOT}/data/universe", exist_ok=True)
with open(f"{ROOT}/data/universe/universe_ids.tsv", "w") as f:
    f.write("uniprot\tmask_lo\tmask_hi\n")
    for u in sorted(univ):
        f.write(f"{u}\t{univ[u][0]}\t{univ[u][1]}\n")
json.dump([{"id": t[0], "label": t[1], "bit": i} for i, t in enumerate(TOOLS)], open(f"{ROOT}/data/universe/tool_bits.json", "w"), ensure_ascii=False)
print("tools", len(TOOLS), "| proteins outside the map:", len(univ), "| tools_data.json bytes", os.path.getsize(f"{ROOT}/explorer/tools_data.json"))
