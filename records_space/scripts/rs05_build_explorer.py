#!/usr/bin/env python3
"""rs05（本机，纯标准库）：把 out/rs_plot_table.tsv + out/rs_coords.tsv 打包成一个自包含的交互页面。
输出  explorer/records_space_explorer.html（独立网页，可直接用浏览器打开）和 explorer/page_fragment.html（用于发布为 Artifact 的片段）。
页面只含坐标与标注（蛋白名、UniProt、论文号、条件、读数），不含序列。"""
import base64
import csv
import json
import math
import os
import struct
import sys
from collections import Counter

csv.field_size_limit(10 ** 9)
ROOT = "/Users/lynn/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Projects/LLPS/condition_llps/records_space_2026-10-29"
OUT = f"{ROOT}/out"
EXP = f"{ROOT}/explorer"
XP = f"{ROOT}/export"

EMB = [("tsne_full", "t-SNE · full"), ("umap_full", "UMAP · full"), ("umap_prot", "UMAP · protein only"),
       ("umap_cond", "UMAP · nucleic acid + conditions"), ("umap_fullM", "UMAP · full, observed-only (M−)")]


def f(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() in ("nan", "none"):
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def r(v, nd=3):
    return None if v is None else round(v, nd)


def main():
    rows = list(csv.DictReader(open(f"{OUT}/rs_plot_table.tsv"), delimiter="\t"))
    co = list(csv.DictReader(open(f"{OUT}/rs_coords.tsv"), delimiter="\t"))
    assert len(rows) == len(co) and all(a["rs_idx"] == b["rs_idx"] for a, b in zip(rows, co))
    n = len(rows)
    data = {"n": n, "emb": {}, "emb_labels": {k: v for k, v in EMB}, "cat": {}, "cont": {}, "info": {}}

    # ---- 坐标：每个嵌入归一到 [0,1]（保持纵横比），y 翻转（上为正）
    for key, _ in EMB:
        if f"{key}_x" not in co[0]:
            continue
        xs = [float(c[f"{key}_x"]) for c in co]
        ys = [float(c[f"{key}_y"]) for c in co]
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        s = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.04
        data["emb"][key] = [[round((x - cx) / s + 0.5, 4) for x in xs], [round(1 - ((y - cy) / s + 0.5), 4) for y in ys]]

    # ---- 分类字段
    def cat(name, values, order=None, label=None):
        lv = list(order) if order else [k for k, _ in Counter(values).most_common()]
        for v in set(values):
            if v not in lv:
                lv.append(v)
        idx = {v: i for i, v in enumerate(lv)}
        data["cat"][name] = {"levels": lv, "codes": [idx[v] for v in values], "label": label or name}

    # 蛋白：记录数前 14 的 UniProt 用名字，无 UniProt 与其他各归一类
    pk = [x["pkey"] for x in rows]
    cnt = Counter(k for k in pk if not k.startswith("NA:"))
    top = [k for k, _ in cnt.most_common(14)]
    # 来源条目的 UniProt 栏里写了多个号（RNAPSEC 的几条）：用各自的蛋白名拼出来，序列是哪一个另看（Q07352 的序列是 ZFP36L1 片段，不是 FUS）
    known = {"Q07352": "ZFP36L1"}   # UniProt Q07352 = TISB_HUMAN / ZFP36L1，2026-10-30 核对
    single_name = {}
    for x in rows:
        if "," not in x["pkey"] and not x["pkey"].startswith("NA:"):
            single_name.setdefault(x["pkey"], x["pname"])

    def disp(x):
        if "," in x["pkey"]:
            return " + ".join(known.get(u) or single_name.get(u) or u for u in x["pkey"].split(",")) + " (two UniProt IDs)"
        return x["pname"]
    name_of = {}
    for x in rows:
        name_of.setdefault(x["pkey"], disp(x))
    prot = [name_of[k] if k in top else ("no UniProt" if k.startswith("NA:") else "other proteins") for k in pk]
    cat("protein", prot, [name_of[k] for k in top] + ["no UniProt", "other proteins"], "Protein identity (top 14)")
    cat("na_class", [x["na_class"] for x in rows], ["none", "RNA", "DNA"], "Nucleic acid class")
    cat("construct", [x["construct"] for x in rows], ["full length / wild type", "truncation / domain", "point mutant", "fusion tag"], "Construct type")
    cat("temp_band", [x["temp_band"] for x in rows], ["≤ 10 °C", "10–20 °C", "20–30 °C", "30–40 °C", "> 40 °C", "room temp. only", "missing"], "Temperature band")
    cat("temp_state", [x["temp_state"] for x in rows], ["measured", "room temp. only", "missing"], "Temperature state")
    cat("crowder", [x["crowder"] for x in rows], ["none reported", "PEG", "Ficoll / dextran", "glycerol", "not given"], "Crowder")
    cat("buffer_f", [x["buffer_f"] for x in rows], ["Tris", "HEPES", "phosphate / PBS", "MES / MOPS / PIPES", "citrate / acetate", "other", "not given"], "Buffer family")
    cat("additive_f", [x["additive_f"] for x in rows], ["none recorded", "additive listed", "not given"], "Additives")
    cat("ps", [x["ps"] for x in rows], ["no phase separation", "phase separation", "quantitative only (csat pool)"], "Phase-separation label")
    cat("csat_status", [x["csat_status"] for x in rows], ["no csat", "binary, same UniProt as a csat protein", "binary, identical sequence to a csat protein", "csat record"], "Has csat?")
    cat("readout_class", [x["readout_class"] for x in rows], ["binary", "dilute", "csat_boundary", "dilute_linescan", "cloud_point", "dense"], "Quantitative readout")
    srcs = ["LLPSDB_v2", "LLPSDB_v2_site", "LLPSDB_v2_phase_diagram", "RNAPSEC", "LLPSense", "PhaSepDB3_pilot_LLM", "Bremer2022", "Ausserwoger2025"]
    cat("source", [x["source"] for x in rows], srcs, "Source")

    # ---- 连续字段：vals + kind（0 有值，1 缺失，2 特殊状态）
    def cont(name, label, vals, kinds, lo, hi, cmap="turbo", special=None, missing="missing", discrete=0, unit=""):
        data["cont"][name] = {"label": label, "vals": [r(v) for v in vals], "kind": kinds, "lo": lo, "hi": hi, "map": cmap, "special": special, "missing": missing, "discrete": discrete, "unit": unit}

    def simple(col, **kw):
        vals = [f(x[col]) for x in rows]
        kinds = [0 if v is not None else 1 for v in vals]
        return vals, kinds
    v, k = simple("len_log"); cont("len_log", "Protein length", v, k, 1.3, 3.0, missing="no sequence", unit="log10 residues")
    v, k = simple("ncpr"); cont("ncpr", "Net charge per residue", v, k, -0.15, 0.15, cmap="div", missing="no sequence", unit="NCPR")
    v, k = simple("arom"); cont("arom", "Aromatic residues (F + W + Y)", v, k, 0.0, 0.2, missing="no sequence", unit="fraction of residues")
    tv = [f(x["tempC"]) if x["temp_state"] == "measured" else None for x in rows]
    tk = [0 if x["temp_state"] == "measured" else (2 if x["temp_state"] == "room temp. only" else 1) for x in rows]
    cont("tempC", "Temperature", tv, tk, 0, 45, special="room temp. only (not measured)", unit="°C (measured)")
    v, k = simple("pH_v"); cont("pH_v", "pH", v, k, 5, 9, unit="pH")
    v, k = simple("nacl_log"); cont("nacl_log", "NaCl", v, k, 0, 2.7, unit="log10(1 + mM)")
    v, k = simple("conc_v"); cont("conc_v", "Protein concentration", v, k, -0.5, 2.5, missing="missing (csat: label or scan axis)", unit="log10 µM")
    nv = [f(x["na_ratio"]) for x in rows]
    nk = [2 if x["na_class"] == "none" else (0 if nv[i] is not None else 1) for i, x in enumerate(rows)]
    cont("na_ratio", "Nucleic acid : protein ratio", nv, nk, -2, 3, special="no nucleic acid", missing="nucleic acid present, ratio unknown", unit="log10 molar ratio")
    yv = [f(x["y_csat"]) for x in rows]
    yk = [2 if x["pool"] == "binary" else (0 if yv[i] is not None else 1) for i, x in enumerate(rows)]
    cont("y_csat", "csat / dilute-phase value", yv, yk, 0, 3, special="binary record (no value)", unit="log10 µM (csat pool)")
    mv = [min(float(x["n_miss"]), 6.0) for x in rows]
    cont("n_miss", "Missing condition dimensions", mv, [0] * n, 0, 6, cmap="steps", discrete=7, unit="number missing (6 = six or more)")

    # ---- 悬停卡片的信息
    def s(x, m=72):
        x = (x or "").strip()
        return x if len(x) <= m else x[: m - 1] + "…"
    info = {k: [] for k in ("rid", "name", "uni", "paper", "len", "lab", "morph", "csat", "tstate", "T", "pH", "NaCl", "conc", "buf", "oth", "na", "nalen", "naratio", "nmiss", "ent")}
    for x in rows:
        info["rid"].append(x["record_id"])
        info["name"].append(disp(x))
        info["uni"].append(x["uniprot"])
        info["paper"].append(x["paper"])
        info["len"].append(r(f(x["seq_len"]), 0))
        info["lab"].append(x["ps"])
        info["morph"].append(x["morphology"])
        info["csat"].append(x["readout_class"] if x["readout_class"] != "binary" else "")
        info["tstate"].append(x["temp_state"])
        info["T"].append(r(f(x["tempC"]), 1) if x["temp_state"] == "measured" else None)
        info["pH"].append(r(f(x["pH_v"]), 2))
        nacl = f(x["NaCl"]) if x["st_NaCl"] in ("obs", "zero") else None
        info["NaCl"].append(r(nacl, 1))
        info["conc"].append(r(10 ** f(x["conc_v"]), 3) if f(x["conc_v"]) is not None else None)
        info["buf"].append(s(x["buffer_raw"]))
        info["oth"].append(s(x["other_raw"]))
        info["na"].append(x["na_class"])
        info["nalen"].append(r(10 ** f(x["na_len"]), 0) if f(x["na_len"]) is not None else None)
        info["naratio"].append(r(f(x["na_ratio"]), 2))
        info["nmiss"].append(int(float(x["n_miss"])))
        info["ent"].append(x["protein_name"] if x["pool"] == "csat" else "")
    data["info"] = info
    data["meta"] = {"n": n, "n_binary": sum(1 for x in rows if x["pool"] == "binary"), "n_csat": sum(1 for x in rows if x["pool"] == "csat"),
                    "n_seq": len({x["seq_row"] for x in rows if x["seq_row"] != "-1"}), "n_papers": len({x["paper"] for x in rows}),
                    "source_counts": dict(Counter(x["source"] for x in rows))}
    # ---- “试新实验”面板用的数据：导出的二进制数组、参数、近似嵌入、覆盖度分位数、例子（不含数据集序列，只有草图与 ESMC 嵌入）
    def b64(p):
        return base64.b64encode(open(p, "rb").read()).decode()
    rmeta = json.load(open(f"{XP}/records_meta.json"))
    params = json.load(open(f"{XP}/params.json"))
    assert rmeta["n"] == n, (rmeta["n"], n)
    rb = open(f"{XP}/records.bin", "rb").read()
    off = next(a["offset"] for a in rmeta["arrays"] if a["name"] == "conc")
    bad = 0
    for x, c in zip(rows, struct.unpack_from(f"<{n}f", rb, off)):
        v = f(x["conc_v"])
        if (v is None) != math.isnan(c) or (v is not None and abs(v / params["scales"]["conc"] - c) > 2e-3):
            bad += 1
    print("records.bin 与 rs_plot_table 的蛋白浓度对不上的行数：", bad, "/", n)
    assert bad < 0.01 * n, "records.bin 与 rs_plot_table 的行顺序可能不一致"
    data["space"] = {"params": params, "recordsMeta": rmeta, "recordsB64": b64(f"{XP}/records.bin"), "seqsMeta": json.load(open(f"{XP}/seqs_meta.json")),
                     "seqsB64": b64(f"{XP}/seqs.bin"), "approx": json.load(open(f"{XP}/approx.json")), "coverage": json.load(open(f"{XP}/coverage.json")),
                     "exampleDefs": json.load(open(f"{EXP}/examples/example_defs.json"))}
    data["tools"] = json.load(open(f"{EXP}/tools_data.json"))
    assert len(data["tools"]["mask_lo"]) == n
    # 蛋白宇宙：没有条件、只有蛋白的条目（rs13 由 Colab 的 rs12 输出整理）；记录在这个透镜里的位置 = 它的数据集序列的位置
    up = f"{EXP}/universe_data.json"
    if os.path.exists(up):
        U = json.load(open(up))
        raw = base64.b64decode(U["xyds"])
        q = struct.unpack("<" + "H" * (len(raw) // 2), raw)
        xs, ys, seen = [], [], Counter()
        for x in rows:
            sr = int(float(x["seq_row"])) if x["seq_row"] not in ("", None) else -1
            if 0 <= sr < U["ds_n"]:
                r_ = seen[sr]; seen[sr] += 1                      # 同一蛋白的第 r_ 条记录：向日葵排布，半径 ∝ √r_，只为让重叠的点散成一小团
                rad, ang = 0.0017 * math.sqrt(r_), r_ * 2.399963229728653
                xs.append(round(min(max(q[2 * sr] / 65535 + rad * math.cos(ang), 0.0), 1.0), 4)); ys.append(round(min(max(q[2 * sr + 1] / 65535 + rad * math.sin(ang), 0.0), 1.0), 4))
            else:
                xs.append(-1); ys.append(-1)
        data["emb"]["umap_universe"] = [xs, ys]
        data["emb_labels"]["umap_universe"] = "Protein universe (UMAP)"
        data["universe"] = U
        print("universe proteins", U["n"], "| records placed", sum(1 for v in xs if v >= 0))
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    def js(name):
        t = open(f"{EXP}/js/{name}").read()
        assert "</script" not in t.lower(), name
        return t
    tpl = open(f"{EXP}/explorer_template.html").read()
    for ph in ("/*__DATA__*/null", "/*__CONDPARSER__*/", "/*__RECORDSSPACE__*/", "/*__TRYPANEL__*/"):
        assert tpl.count(ph) == 1, ph
    frag = tpl.replace("/*__CONDPARSER__*/", js("condition_parser.js")).replace("/*__RECORDSSPACE__*/", js("records_space.js")).replace("/*__TRYPANEL__*/", js("try_panel.js")).replace("/*__DATA__*/null", blob)
    open(f"{EXP}/page_fragment.html", "w").write(frag)
    standalone = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + \
        frag.split("</style>", 1)[0].replace("<title>", "<title>", 1) + "</style>\n</head>\n<body>\n" + frag.split("</style>", 1)[1] + "\n</body>\n</html>\n"
    open(f"{EXP}/records_space_explorer.html", "w").write(standalone)
    print("n", n, "json MB", round(len(blob) / 1e6, 2), "file MB", round(len(standalone) / 1e6, 2))


if __name__ == "__main__":
    main()
