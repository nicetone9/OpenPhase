#!/usr/bin/env python3
"""rs01（本机，纯标准库，不做任何嵌入或数值计算）：构建 records 空间的输入表。

范围
  binary 池：Task 1（旧称 Task 2）主表 10,663 条（单蛋白体外二分类，核酸当条件）。
  csat   池：csat_features.tsv 的 776 行（Bremer2022 的温度/pH 扫描、Ausserwöger2025 的线扫），定量读数；与主表 record_id 无交集。
  读数（label / csat / 形态）与来源（source / paper）只做着色标签，不进嵌入输入。

输出（data/）
  rs_records.tsv       每行一条 record：标识、标签、条件数值 + 状态、核酸特征、缓冲液/添加剂 multi-hot、蛋白组成描述符
  rs_sequences.tsv     唯一序列表（主表 1,299 条在前，行号 = work/esmc_L18.npy 的行；csat 独有序列追加在后）
  rs_new_sequences.tsv 需要在 Colab 上新算 ESMC 的序列（csat 独有 + 3 条主表序列用于一致性检查）
  inputs_summary.json  计数与各维度缺失统计
  MD5SUMS

状态码（每个条件维度）：obs 有数值 / zero 作者明确报告为无（absent_from_reported_*）/ rt 只写室温（数值记 22 °C）/ miss 缺失。
缺失 ≠ 不适用：无核酸的记录，核酸数值特征是"不适用"，不是缺失；缺失只指"有核酸但长度/浓度/序列未知"。
"""
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, OrderedDict

csv.field_size_limit(10 ** 9)
ROOT = "/Users/lynn/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Projects/LLPS/condition_llps"
T2 = f"{ROOT}/task2_single_component_condition"
V4 = f"{ROOT}/condition_phase/data/world_model_staging/condition_llps_dataset_v4_2026-10-23/records.jsonl"
OUT = f"{ROOT}/records_space_2026-10-29/data"

FIELDS = ["Conc", "pH", "temp", "MgCl2", "NaCl", "KCl", "PEG300-1k", "PEG3k-6k", "PEG8k-20k", "Ficoll", "Dextran -40", "Dextran 70-", "Glycerol"]
SHORT = ["conc", "pH", "temp", "MgCl2", "NaCl", "KCl", "PEG1k", "PEG6k", "PEG20k", "Ficoll", "Dex40", "Dex70", "Glycerol"]
NA_NAME = ["polyU", "polyA", "polyC", "polyG", "tRNA", "rRNA", "total_RNA", "mRNA", "G4", "ssDNA", "dsDNA", "plasmid_or_long_DNA", "methylated"]
NA_NUM = ["na_len_log10", "na_conc_log10", "na_ratio_log10", "na_charge_ratio_log10", "na_gc", "na_purine", "na_maxrun", "na_g4", "na_mfe_per_nt", "na_paired_frac"]

# 缓冲液 / 添加剂词表（在 buffer_raw + other_molecules_raw 的拼接文本里找）
BUF = OrderedDict([
    ("Tris", r"\btris\b|tris-|trizma|tris\(|\btris(?=[\s,(])"),
    ("HEPES", r"hepes"),
    ("phosphate", r"phosphate|\bpbs\b|\bdpbs\b|\bnapi\b|na2hpo4|nah2po4|kh2po4|k2hpo4|\bpi\b"),
    ("goodsMES", r"\bmes\b|\bmops\b|pipes|\bmes-"),
    ("citrate", r"citrate"),
    ("acetate", r"acetate|\bnaac\b|\boac\b"),
    ("imidazole", r"imidazole"),
    ("other_buf", r"glycine|borate|\bcaps\b|\bches\b|bicine|bis-?tris|cacodylate|ammonium bicarbonate"),
])
ADD = OrderedDict([
    ("reducing", r"\bdtt\b|tcep|\bbme\b|mercaptoethanol|2-me\b|beta-?me\b"),
    ("chelator", r"edta|egta"),
    ("glycerol", r"glycerol"),
    ("nucleotide", r"\batp\b|\badp\b|\bgtp\b|\bamp\b|camp|nadh|\bfad\b|accoa|acetyl-?coa"),
    ("denaturant", r"urea|guanidin|gdm|gdn|tmao|trimethylamine|betaine|sorbitol|trehalose|sucrose"),
    ("hexanediol", r"hexane?diol|hexanodiol"),
    ("detergent", r"triton|np-?40|tween|\bsds\b|chaps|brij|igepal"),
    ("protein_add", r"\bbsa\b|albumin|lysozyme|\btev\b"),
    ("polyanion", r"heparin|\bpar\b|poly\(|polyp\b|polyu\b|polya\b|dextran sulfate"),
    ("metal", r"zncl|znso|cacl|mncl|mgso|cucl|fecl|zinc|calcium"),
    ("dye_probe", r"thioflavin|hoechst|dapi|fluorescein|alexa|\bcy3\b|\bcy5\b|nile red"),
])
BUF_RE = {k: re.compile(v, re.I) for k, v in BUF.items()}
ADD_RE = {k: re.compile(v, re.I) for k, v in ADD.items()}


FRIENDLY = {   # 记录数靠前、且我确定编号对应关系的 UniProt 的通用名；其余用 v4 里的名字，再退到 UniProt 编号
    "P10636": "tau (MAPT)", "P0DTC9": "SARS-CoV-2 N", "P35637": "FUS", "Q13148": "TDP-43", "Q9NQI0": "DDX4",
    "P51608": "MeCP2", "Q9UHD9": "UBQLN2", "D0PV95": "LAF-1", "P09651": "hnRNPA1", "Q06787": "FMRP",
    "Q8N884": "cGAS", "Q13283": "G3BP1", "P0AGE0": "SSB", "P39517": "Ded1p", "P31483": "TIA-1",
}


def nanf(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "na"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return None if v != v else v


def fmt(v, nd=6):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return repr(round(v, nd))
    return str(v)


def sha(seq):
    return hashlib.sha256(seq.encode()).hexdigest()


def state_main(status, v):
    if v is None:
        return "miss"
    if status.startswith("absent_from_reported"):
        return "zero"
    if status == "room_temperature_unspecified":
        return "rt"
    return "obs"


def parse_text(buf_text, other_text):
    """返回 (buffer 命中集合, buffer 状态, 添加剂命中集合, 添加剂状态)。
    buffer 状态：obs（识别出缓冲液）/ other（有文本但词表没命中）/ miss（没有任何文本）。
    添加剂状态：obs（文本里识别出添加剂）/ none（有缓冲液或其他分子文本但没有添加剂，记"无记录"）/ miss（没有任何文本）。"""
    t = " ; ".join(x for x in (buf_text, other_text) if x)
    if not t.strip():
        return set(), "miss", set(), "miss"
    b = {k for k, rx in BUF_RE.items() if rx.search(t)}
    a = {k for k, rx in ADD_RE.items() if rx.search(t)}
    if other_text and not a:
        a = {"other_add"}                  # 有"其他分子"文本但词表没命中
    bs = "obs" if b else ("other" if buf_text else "miss")
    return b, bs, a, ("obs" if a else "none")


def main():
    os.makedirs(OUT, exist_ok=True)
    # ---------------- 读主表与特征 ----------------
    main_rows = OrderedDict()
    with open(f"{T2}/data/staging/main_table.jsonl") as fh:
        for l in fh:
            r = json.loads(l)
            main_rows[r["record_id"]] = r
    feats = {r["record_id"]: r for r in csv.DictReader(open(f"{T2}/work/features_with_vienna.tsv"), delimiter="\t")}
    assert set(feats) == set(main_rows), "features 与主表 record_id 不一致"
    seq_tab = list(csv.DictReader(open(f"{T2}/work/sequences.tsv"), delimiter="\t"))
    assert [int(r["seq_row"]) for r in seq_tab] == list(range(len(seq_tab)))
    # 主表序列：自己的 sha256 与 sequence_id 的对应
    key_of_main_row, id_match = [], 0
    for r in seq_tab:
        s = "".join(r["sequence"].upper().split())
        k = sha(s)
        key_of_main_row.append(k)
        id_match += (k == r["sequence_id"])
    print(f"主表序列 {len(seq_tab)} 条；sha256(sequence) == sequence_id 的有 {id_match} 条")

    # ---------------- v4：csat 记录的缓冲液文本；主表记录的蛋白名 ----------------
    # 主表 record_id 与 v4 的 record_id 写法不同（只有 PhaSepDB3 试点的一致），所以蛋白名按 UniProt 取 v4 里最常见的名字
    ph_recs, aus_recs, name_by_rid, name_by_uniprot, name_by_hash = [], [], {}, {}, {}
    rx_name_uni = re.compile(r'"proteins": \[\{"name": "((?:[^"\\]|\\.)*)", "uniprot": (?:"([A-Za-z0-9_.-]+)"|null)')
    with open(V4) as fh:
        for line in fh:
            if '"source": "Bremer2022"' in line:
                r = json.loads(line)
                if "pH-dependent" in (r.get("detection_method") or ""):
                    ph_recs.append(r)
            elif '"source": "Ausserwoger2025"' in line:
                aus_recs.append(json.loads(line))
            else:
                m = rx_name_uni.search(line)
                if m and m.group(1).strip():
                    nm = m.group(1).strip()
                    if m.group(2):
                        name_by_uniprot.setdefault(m.group(2).split("-")[0], Counter())[nm] += 1
                    seg = line[m.end(): m.end() + 1500]
                    j = seg.find('"sequence_id": "')
                    if j >= 0:
                        name_by_hash.setdefault(seg[j + 16: j + 16 + 64], Counter())[nm] += 1
    print(f"v4：Bremer pH {len(ph_recs)}，Ausserwöger {len(aus_recs)}，UniProt→蛋白名 {len(name_by_uniprot)} 个，序列哈希→蛋白名 {len(name_by_hash)} 个")

    # ---------------- csat 表 ----------------
    cs = list(csv.DictReader(open(f"{T2}/work/csat_features.tsv"), delimiter="\t"))
    assert len(cs) == 776 and len(ph_recs) == 41 and len(aus_recs) == 366
    si = {r["name"]: r["sequence_measured_with_GS"] for r in
          csv.DictReader(open(f"{T2}/data/bremer_si_final_parsed/bremer_si_sequences_merged.tsv"), delimiter="\t") if r["check"] == "ok"}

    # ---------------- 统一序列表 ----------------
    seqs = []           # dict(seq_row, seq_key, length, sequence, origin, main_seq_row)
    key2row = {}
    for r, k in zip(seq_tab, key_of_main_row):
        s = "".join(r["sequence"].upper().split())
        key2row[k] = len(seqs)
        seqs.append(dict(seq_row=len(seqs), seq_key=k, length=len(s), sequence=s, origin="main", main_seq_row=int(r["seq_row"]), names=Counter()))

    def seq_row_for(s):
        if not s:
            return -1
        s = "".join(s.upper().split())
        k = sha(s)
        if k not in key2row:
            key2row[k] = len(seqs)
            seqs.append(dict(seq_row=len(seqs), seq_key=k, length=len(s), sequence=s, origin="csat_only", main_seq_row="", names=Counter()))
        return key2row[k]

    records = []

    # ---------------- 主表记录 ----------------
    for rid, r in main_rows.items():
        f = feats[rid]
        row = OrderedDict()
        row["record_id"] = rid
        row["pool"] = "binary"
        row["source"] = r["source"]
        row["paper"] = r["paper"]
        row["uniprot"] = r.get("uniprot") or ""
        sr = key2row[sha("".join(seq_tab[int(f["seq_row"])]["sequence"].upper().split()))]
        assert sr == int(f["seq_row"])
        row["seq_row"] = sr
        row["seq_len"] = r["seq_len"]
        uni = (r.get("uniprot") or "").split("-")[0]
        # 蛋白名：已知的常用蛋白用手写名（图例要读得懂）；否则取 v4 里同序列哈希最常见的名字，再退到同 UniProt 的，最后退到 UniProt 编号
        nm = (FRIENDLY.get(uni)
              or (name_by_hash.get(key_of_main_row[sr], Counter()).most_common(1) or [("", 0)])[0][0]
              or (name_by_uniprot.get(uni, Counter()).most_common(1) or [("", 0)])[0][0]
              or uni or "unnamed")
        row["protein_name"] = nm
        seqs[sr]["names"][nm] += 1
        row["system_id"] = f["system_id"][:12]
        row["partition"] = "protein+NA" if r["na"]["class"] != "none" else "protein only"
        # 标签（不进嵌入）
        row["label"] = r["label"]
        row["morphology"] = r.get("morphology_class") or ""
        row["readout_class"] = "binary"
        row["y_log10"] = ""
        row["axis"] = ""
        row["phase_state"] = ""
        row["tier_C"] = ""
        row["tier_S"] = ""
        row["seq_state"] = r.get("seq_source") or ""
        # 蛋白注释
        row["nd_type"] = r.get("nd_type") or ""
        row["mutation"] = int(bool(r.get("mutation")))
        row["fusion"] = int(bool(r.get("fusion")))
        row["cleaved"] = int(bool(r.get("cleaved")))
        row["ptm"] = int(bool(r.get("ptm")))
        cl = r.get("cluster") or {}
        row["c30"] = cl.get("c30") or ""
        row["c20"] = cl.get("c20") or ""
        for i in range(24):
            row[f"comp_{i}"] = fmt(nanf(f[f"comp_{i}"]))
        # 条件
        craw = r["condition_raw"]
        for i, (fld, sh) in enumerate(zip(FIELDS, SHORT)):
            v = nanf(f[f"cond_{i}"])
            st = craw[fld][3]
            row[sh] = fmt(v)
            row[f"st_{sh}"] = state_main(st, v)
        row["temp_status_raw"] = craw["temp"][3]
        row["conc_status_raw"] = craw["Conc"][3]
        ex = r.get("extra") or {}
        row["ionic_strength_mM"] = fmt(ex.get("ionic_strength_mM"))
        row["buffer_raw"] = (ex.get("buffer_raw") or "").replace("\t", " ").replace("\n", " ")
        row["other_raw"] = (ex.get("other_molecules_raw") or "").replace("\t", " ").replace("\n", " ")
        b, bs, a, as_ = parse_text(ex.get("buffer_raw"), ex.get("other_molecules_raw"))
        for k in BUF:
            row[f"buf_{k}"] = int(k in b)
        row["st_buffer"] = bs
        for k in list(ADD) + ["other_add"]:
            row[f"add_{k}"] = int(k in a)
        row["st_additive"] = as_
        # 核酸
        nac = r["na"]["class"]
        row["na_class"] = nac
        for k in NA_NUM:
            row[k] = fmt(nanf(f[k])) if nac != "none" else ""
        row["na_len_src"] = f["na_len_src"] if nac != "none" else ""
        row["na_homopolymer"] = f["na_homopolymer"] if nac != "none" else ""
        for k in NA_NAME:
            row[f"na_name_{k}"] = f[f"na_name_{k}"] if nac != "none" else ""
        row["na_has_seq"] = int(bool((f.get("na_seq") or "").strip())) if nac != "none" else ""
        records.append(row)

    # ---------------- csat 记录 ----------------
    uni_of = {"FUS-EGFP": "P35637", "TDP-43-EGFP": "Q13148"}
    for i, c in enumerate(cs):
        row = OrderedDict()
        src = c["source"]
        row["record_id"] = c["record_id"]
        row["pool"] = "csat"
        row["source"] = src
        row["paper"] = "doi:10.1038/s41557-021-00840-w" if src == "Bremer2022" else "doi:10.1038/s41467-025-62437-y"
        if src == "Bremer2022":
            seq = si.get(c["variant_key"])
            ent, uni = c["entity"], "P09651"            # A1-LCD 设计变体的母体 hnRNPA1
            if i < 369:
                text = "20 mM HEPES pH 7.0, 150 mM NaCl"
            else:
                pr = ph_recs[i - 369]
                assert abs(math.log10(pr["quantitative"]["value"]) - float(c["y_log10"])) < 1e-3, ("pH 行对不上", i)
                text = pr["conditions"]["pH"].get("raw") or ""
        else:
            ar = aus_recs[i - 410]
            qv, yl = ar["quantitative"]["value"], nanf(c["y_log10"])
            if qv and qv > 0 and yl is not None:
                assert abs(math.log10(qv) - yl) < 1e-3, ("Ausserwöger 行对不上", i)
            else:
                assert not (qv and qv > 0) and yl is None or qv == 0, ("Ausserwöger 零值行不一致", i, qv, yl)
            seq = (ar["proteins"][0].get("sequence") or "")
            ent = c["entity"]
            uni = ar["proteins"][0].get("uniprot") or uni_of.get(ent, "")
            text = ar["conditions"]["pH"].get("raw") or ""
        sr = seq_row_for(seq)
        row["uniprot"] = uni
        row["seq_row"] = sr
        row["seq_len"] = seqs[sr]["length"] if sr >= 0 else ""
        row["protein_name"] = ent
        if sr >= 0:
            seqs[sr]["names"][ent] += 1
        row["system_id"] = ent[:12]
        row["partition"] = "protein only"
        row["label"] = ""
        row["morphology"] = ""
        row["readout_class"] = c["readout_class"]
        row["y_log10"] = fmt(nanf(c["y_log10"]))
        row["axis"] = c["axis"]
        row["phase_state"] = c["phase_state"]
        row["tier_C"] = c["tier_C_ok"]
        row["tier_S"] = c["tier_S_ok"]
        row["seq_state"] = c["seq_state"]
        row["nd_type"] = ""
        row["mutation"] = 0
        row["fusion"] = 0
        row["cleaved"] = 0
        row["ptm"] = 0
        row["c30"] = ""
        row["c20"] = ""
        for j in range(24):
            row[f"comp_{j}"] = fmt(nanf(c[f"comp_{j}"]))
        for j, sh in enumerate(SHORT):
            v = nanf(c[f"cond_{j}"])
            row[sh] = fmt(v)
            row[f"st_{sh}"] = "miss" if v is None else "obs"
        row["temp_status_raw"] = "source_numeric" if nanf(c["cond_2"]) is not None else "missing_not_stated"
        row["conc_status_raw"] = "label_or_scan_axis_not_input"
        row["ionic_strength_mM"] = ""
        row["buffer_raw"] = text
        row["other_raw"] = ""
        b, bs, a, as_ = parse_text(text, "")
        for k in BUF:
            row[f"buf_{k}"] = int(k in b)
        row["st_buffer"] = bs
        for k in list(ADD) + ["other_add"]:
            row[f"add_{k}"] = int(k in a)
        row["st_additive"] = as_
        row["na_class"] = "none"
        for k in NA_NUM:
            row[k] = ""
        row["na_len_src"] = ""
        row["na_homopolymer"] = ""
        for k in NA_NAME:
            row[f"na_name_{k}"] = ""
        row["na_has_seq"] = ""
        records.append(row)

    # ---------------- 与 csat 的蛋白级关联 ----------------
    # UniProt 比较去掉异构体后缀（P09651-2 → P09651），并按逗号拆开多个编号（Q07352,P35637 含 FUS）
    def ubase(u):
        return {x.strip().split("-")[0] for x in (u or "").split(",") if x.strip()}
    csat_keys = {r["seq_row"] for r in records if r["pool"] == "csat" and r["seq_row"] != -1}
    csat_unis = set().union(*[ubase(r["uniprot"]) for r in records if r["pool"] == "csat"])
    for r in records:
        if r["pool"] == "csat":
            r["csat_link"] = "csat record"
        elif r["seq_row"] in csat_keys:
            r["csat_link"] = "same sequence"
        elif ubase(r["uniprot"]) & csat_unis:
            r["csat_link"] = "same UniProt"
        else:
            r["csat_link"] = "none"

    # ---------------- 写出 ----------------
    for i, r in enumerate(records):
        r["rs_idx"] = i
    cols = ["rs_idx"] + [k for k in records[0].keys() if k != "rs_idx"]
    for r in records:
        assert list(r.keys())[:0] == [] and set(cols) >= set(r.keys()), set(r.keys()) - set(cols)
    with open(f"{OUT}/rs_records.tsv", "w") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in records:
            w.writerow([fmt(r.get(k, "")) if not isinstance(r.get(k, ""), str) else r.get(k, "") for k in cols])
    n_rec_of = Counter(r["seq_row"] for r in records)
    with open(f"{OUT}/rs_sequences.tsv", "w") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["seq_row", "seq_key", "length", "origin", "main_seq_row", "n_records", "name", "sequence"])
        for s in seqs:
            nm = s["names"].most_common(1)[0][0] if s["names"] else ""
            w.writerow([s["seq_row"], s["seq_key"], s["length"], s["origin"], s["main_seq_row"], n_rec_of.get(s["seq_row"], 0), nm, s["sequence"]])
    new = [s for s in seqs if s["origin"] == "csat_only"]
    # 一致性检查序列：主表里长度最接近 150 / 300 / 500 的三条
    chk = []
    for target in (150, 300, 500):
        s = min((q for q in seqs if q["origin"] == "main"), key=lambda q: abs(q["length"] - target))
        chk.append(s)
    with open(f"{OUT}/rs_new_sequences.tsv", "w") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["seq_row", "kind", "main_seq_row", "length", "sequence"])
        for s in new:
            w.writerow([s["seq_row"], "new", "", s["length"], s["sequence"]])
        for s in chk:
            w.writerow([s["seq_row"], "check", s["main_seq_row"], s["length"], s["sequence"]])

    # ---------------- 汇总 ----------------
    summ = OrderedDict()
    summ["records_total"] = len(records)
    summ["by_pool"] = dict(Counter(r["pool"] for r in records))
    summ["by_source"] = dict(Counter(r["source"] for r in records))
    summ["unique_sequences"] = len(seqs)
    summ["sequences_from_main"] = sum(s["origin"] == "main" for s in seqs)
    summ["sequences_csat_only_new"] = len(new)
    summ["records_without_sequence"] = sum(r["seq_row"] == -1 for r in records)
    summ["csat_link"] = dict(Counter(r["csat_link"] for r in records))
    summ["na_class"] = dict(Counter(r["na_class"] for r in records))
    miss = {}
    for pool in ("binary", "csat"):
        rows = [r for r in records if r["pool"] == pool]
        d = {}
        for sh in SHORT:
            d[sh] = dict(Counter(r[f"st_{sh}"] for r in rows))
        d["buffer"] = dict(Counter(r["st_buffer"] for r in rows))
        d["additive"] = dict(Counter(r["st_additive"] for r in rows))
        miss[pool] = d
    summ["state_counts"] = miss
    nar = [r for r in records if r["na_class"] != "none"]
    summ["na_present_unknowns"] = {k: sum(1 for r in nar if r[k] == "") for k in NA_NUM}
    summ["n_na_present"] = len(nar)
    summ["buffer_family_counts"] = {k: sum(int(r[f"buf_{k}"]) for r in records) for k in BUF}
    summ["additive_family_counts"] = {k: sum(int(r[f"add_{k}"]) for r in records) for k in list(ADD) + ["other_add"]}
    json.dump(summ, open(f"{OUT}/inputs_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
