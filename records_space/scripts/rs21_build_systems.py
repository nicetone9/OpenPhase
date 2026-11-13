#!/usr/bin/env python3
"""rs21（本机，纯标准库）：多蛋白系统的"系统级地图"数据，供 explorer 的 Systems 模式使用（rs05 调 build_data()）。
一张图一种系统类型（Fig. S1 的七张图），一个点 = 一个唯一系统（组分集合，不含浓度）；坐标沿用 Fig. S1 每张图自己的 t-SNE（ESMC-600M 组分嵌入），不重新计算；只有系统级信息（组分、记录数、阳性 / 阴性数、来源），没有条件。
另外给出系统色表的 16 个类型各自的种类（单蛋白 = 在 Explore 里筛、有地图、只有清单）和计数，没有蛋白序列的 RNA only / DNA only 给清单。
输入  data/systems/（fsys_points.csv、systems_multi.tsv、systems_extra.tsv、sys_summary.csv、sys_tsne_summary.json；缺了就从 world_and_cluster_maps_2026-10-27 复制 / 过滤）
公开页不放的：只来自实验室笔记本（OpenPhase_db2）的系统——仓库 data/README 写明 db2 的记录不在公开表里，这里同样处理（改 EXCLUDE_SOURCES 即可放回）。
用法：python3 -I scripts/rs21_build_systems.py   （只打印汇总）；explorer 的构建会 import 它"""
import base64
import csv
import json
import os
import re
import shutil
import sys
from array import array
from collections import Counter, defaultdict

csv.field_size_limit(10 ** 9)
ROOT = "/Users/lynn/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Projects/LLPS/condition_llps/records_space_2026-10-29"
BASE = "/Users/lynn/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Projects/LLPS/condition_llps/world_and_cluster_maps_2026-10-27"
DATA = f"{ROOT}/data/systems"
EXCLUDE_SOURCES = {"OpenPhase_db2"}           # a system whose sources are all in this set is left out of the public page

# the system colour table (Fill, Point), all 16 types. kind: single = records of the Records space (filter there), map = a Fig. S1 map, list = no protein sequence, so only a list
# (name, fill, point, kind, code, map group)
TABLE = [("protein(1)", "#3FC3FF", "#1799D2", "single", "p1", None), ("protein(2)", "#80ACED", "#4474B7", "map", "p2", "p2"), ("protein(3)", "#68D5D6", "#19858E", "map", "p3", "p3"), ("protein(\u22654)", "#9BB6D1", "#4C637E", "map", "p4", "p4"),
         ("RNA only", "#E9A3A9", "#A43851", "list", "rna", None), ("protein(1) + RNA", "#FFC2C4", "#E45265", "single", "p1r", None), ("protein(2) + RNA", "#F4A48C", "#BA553B", "map", "p2r", "p2r"), ("protein(3) + RNA", "#F4A6CD", "#A84076", "map", "p3r", "p3na"),
         ("protein(\u22654) + RNA", "#EBC0AC", "#91432F", "map", "p4r", "p4"), ("DNA only", "#A6CBA0", "#27613E", "list", "dna", None), ("protein(1) + DNA", "#D4F3B6", "#68A92C", "single", "p1d", None), ("protein(2) + DNA", "#A7D9AD", "#39824E", "map", "p2d", "p2d"),
         ("protein(3) + DNA", "#95D4BE", "#217660", "map", "p3d", "p3na"), ("protein(\u22654) + DNA", "#D4DDA0", "#657626", "map", "p4d", "p4"), ("Other", "#E8E8E8", "#666666", "map", "oth", "cx"), ("C_sat pool", "#DCCBF2", "#8B56BB", "single", "csat", None)]
TYPE_NAMES = [t[0] for t in TABLE]
SINGLE_COUNTS = {"protein(1)": (1163, 7490), "protein(1) + RNA": (241, 2511), "protein(1) + DNA": (99, 662), "C_sat pool": (37, 776)}   # unique systems, records (data/systems.tsv of the repository)


def fine_type(raw):
    """v4 raw system type -> a type of the colour table (complexes are 'Other')."""
    r = raw.strip().lower()
    if "complex" in r or r == "protein(0)":
        return "Other"
    if r == "protein(0) + rna":
        return "RNA only"
    if r in ("dna(4)", "protein(0) + dna"):
        return "DNA only"
    m = re.match(r"protein\((\d+)\)(?: \+ (rna|dna))?$", r)
    n, na = int(m.group(1)), m.group(2)
    return ("protein(%d)" % n if n <= 3 else "protein(\u22654)") + (" + " + na.upper() if na else "")


# (key, S1 group name, display name, family, one-line description)
GROUPS = [("p2", "protein(2)", "protein(2)", "P", "Two proteins, no nucleic acid."),
          ("p3", "protein(3)", "protein(3)", "P", "Three proteins, no nucleic acid."),
          ("p4", "protein(>=4)", "protein(≥4)", "P", "Four or more proteins."),
          ("p2r", "protein(2) + RNA", "protein(2) + RNA", "R", "Two proteins with RNA."),
          ("p2d", "protein(2) + DNA", "protein(2) + DNA", "D", "Two proteins with DNA."),
          ("p3na", "protein(3) + NA", "protein(3) + NA", "C", "Three proteins with RNA or DNA."),
          ("cx", "With complex", "With complex", "C", "Systems that include a complex as a component.")]
SRC_LAB = {"LLPSDB_v2": "LLPSDB v2 (download)", "LLPSDB_v2_site": "LLPSDB v2 (site)", "LLPSDB_v2_phase_diagram": "LLPSDB v2 (phase diagrams)",
           "PhaSepDB3_pilot_LLM": "PhaSepDB 3 (LLM pilot)", "Leurs2025": "Leurs 2025", "Pei2026": "Pei 2026 (miscibility)"}


def prep():
    """Copy / filter the small inputs into data/systems/ so that this folder is self-contained."""
    os.makedirs(DATA, exist_ok=True)
    pts = f"{DATA}/fsys_points.csv"
    if not os.path.exists(pts):
        shutil.copy(f"{BASE}/out/final_fig1_figS1/source_data_E/fsys_points.csv", pts)
    ids = {r["sys_id"] for r in csv.DictReader(open(pts, encoding="utf-8"))}
    if not os.path.exists(f"{DATA}/systems_multi.tsv"):
        with open(f"{DATA}/systems_multi.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t"); w.writerow(["sys_id", "raw_type", "nuc", "sources"])
            for r in csv.DictReader(open(f"{BASE}/data/systems.csv", encoding="utf-8")):
                if r["sys_id"] in ids:
                    w.writerow([r["sys_id"], r["raw_type"], r["nuc"], r["sources"]])
    for n in ("sys_summary.csv", "sys_tsne_summary.json"):
        if not os.path.exists(f"{DATA}/{n}"):
            shutil.copy(f"{BASE}/data/{n}", f"{DATA}/{n}")


def b64(code, vals):
    a = array(code, vals)
    if sys.byteorder != "little":
        a.byteswap()
    return base64.b64encode(a.tobytes()).decode()


def cut(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


def build_data():
    prep()
    pts = list(csv.DictReader(open(f"{DATA}/fsys_points.csv", encoding="utf-8")))
    meta = {r["sys_id"]: r for r in csv.DictReader(open(f"{DATA}/systems_multi.tsv", encoding="utf-8"), delimiter="\t")}
    extra = list(csv.DictReader(open(f"{DATA}/systems_extra.tsv", encoding="utf-8"), delimiter="\t"))
    summ = {r["system"]: r for r in csv.DictReader(open(f"{DATA}/sys_summary.csv", encoding="utf-8"))}
    tsne = json.load(open(f"{DATA}/sys_tsne_summary.json"))
    by = defaultdict(list)
    hidden = Counter(); hidden_by = defaultdict(Counter)
    for r in pts:
        srcs = [s_ for s_ in meta[r["sys_id"]]["sources"].split(";") if s_]
        if srcs and all(s_ in EXCLUDE_SOURCES for s_ in srcs):
            hidden["systems"] += 1; hidden["records"] += int(r["n_records"]); hidden_by[r["system"]]["systems"] += 1; hidden_by[r["system"]]["records"] += int(r["n_records"]); continue
        by[r["system"]].append(r)
    groups, report = [], {}
    tn, tr = Counter(), Counter()                                   # systems and records per type of the colour table (public systems)
    for key, gname, disp, fam, desc in GROUPS:
        rows = by[gname]
        xs = [float(r["tsne_x"]) for r in rows]; ys = [float(r["tsne_y"]) for r in rows]
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        s = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6) * 1.06
        q = lambda v: min(max(round(v * 65535), 0), 65535)
        X = [q((x - cx) / s + 0.5) for x in xs]; Y = [q(1 - ((y - cy) / s + 0.5)) for y in ys]
        # anchors exactly as in Fig. S1: the six most frequent named components of the group (ranks 0-5); "-", "unnamed ..." are not names
        anc, names = [], {}
        for r in rows:
            rank, nm = int(r["anchor_rank"]), r["anchor_label"].strip()
            a = rank if 0 <= rank < 6 and nm not in ("", "-", "unnamed", "unnamed sequences") else 255
            anc.append(a)
            if a != 255:
                assert names.setdefault(a, nm) == nm, (gname, a, nm, names[a])
        cnt = Counter(anc)
        anchors = [{"rank": k, "name": names[k], "n": cnt[k]} for k in sorted(names)]
        nuc = [cut(meta[r["sys_id"]]["nuc"], 44) for r in rows]
        srcs = [";".join(s_ for s_ in meta[r["sys_id"]]["sources"].split(";") if s_ and s_ not in EXCLUDE_SOURCES) for r in rows]
        rec = [int(r["n_records"]) for r in rows]; pos = [int(r["n_pos"]) for r in rows]; neg = [int(r["n_neg"]) for r in rows]
        ft = [TYPE_NAMES.index(fine_type(meta[r["sys_id"]]["raw_type"])) for r in rows]
        for f_, n_ in zip(ft, rec):
            tn[TYPE_NAMES[f_]] += 1; tr[TYPE_NAMES[f_]] += n_
        sm = summ[gname]
        g = {"key": key, "name": disp, "desc": desc, "n": len(rows), "records": sum(rec), "pos": sum(pos), "neg": sum(neg),
             "all_n": int(sm["unique_systems"]), "all_records": int(sm["records"]), "no_seq": tsne[gname]["skipped_no_protein_sequence"], "hid_n": hidden_by[gname]["systems"], "hid_rec": hidden_by[gname]["records"],
             "x": b64("H", X), "y": b64("H", Y), "rec": b64("H", [min(v, 65535) for v in rec]), "p": b64("H", pos), "q": b64("H", neg), "anc": b64("B", anc),
             "anchors": anchors, "id": [r["sys_id"] for r in rows], "comp": [cut(r["components"], 120) for r in rows], "src": srcs,
             "raw": [meta[r["sys_id"]]["raw_type"] for r in rows], "nuc": nuc, "ft": b64("B", ft)}
        groups.append(g)
        report[disp] = {"drawn": len(rows), "records": sum(rec), "pos": sum(pos), "neg": sum(neg), "anchors": [(a["name"], a["n"]) for a in anchors], "systems_in_group": g["all_n"], "no_sequence": g["no_seq"], "records_in_group": g["all_records"]}
    # systems without a protein sequence: RNA only, DNA only, and the one system with no component at all (counted under Other)
    lists = {"RNA only": [], "DNA only": [], "Other": []}
    for r in extra:
        t = fine_type(r["raw_type"]); n_ = int(r["n_records"])
        lists[t].append({"id": r["sys_id"], "raw": r["raw_type"], "nuc": cut(r["nuc"], 60), "src": ";".join(s_ for s_ in r["sources"].split(";") if s_ and s_ not in EXCLUDE_SOURCES), "rec": n_, "pos": int(r["n_pos"]), "neg": int(r["n_neg"])})
        tn[t] += 1; tr[t] += n_
    types = []
    for name, fill, point, kind, code, grp in TABLE:
        if kind == "single":
            n_sys, n_rec = SINGLE_COUNTS[name]
        else:
            n_sys, n_rec = tn[name], tr[name]
        types.append({"name": name, "fill": fill, "point": point, "kind": kind, "code": code, "group": grp, "systems": n_sys, "records": n_rec})
    data = {"types": types, "groups": groups, "lists": lists, "hidden": {"systems": hidden["systems"], "records": hidden["records"]}, "srcLab": SRC_LAB,
            "total": {"systems": sum(g["n"] for g in groups), "records": sum(g["records"] for g in groups)}}
    return data, report


def main():
    data, report = build_data()
    json.dump({"groups": report, "types": [(t["name"], t["systems"], t["records"]) for t in data["types"]], "hidden": data["hidden"], "total": data["total"]}, open(f"{ROOT}/systems_page/summary.json", "w"), ensure_ascii=False, indent=1)
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    print("systems drawn", data["total"], "| hidden (lab notebooks only)", data["hidden"], "| json MB", round(len(blob) / 1e6, 2))
    for t in data["types"]:
        print(f"  {t['name']:22s} {t['kind']:6s} systems {t['systems']:5d} records {t['records']:5d} group {t['group']}")
    for k, v in report.items():
        print(f"  {k:18s} drawn {v['drawn']:4d} records {v['records']:5d} pos {v['pos']:5d} neg {v['neg']:5d} | anchors {v['anchors']}")


if __name__ == "__main__":
    main()
