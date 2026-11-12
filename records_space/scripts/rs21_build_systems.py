#!/usr/bin/env python3
"""rs21（本机，纯标准库）：多蛋白系统的"系统级地图"页。
一张图一种系统类型（protein(2)、protein(3)、protein(≥4)、+ RNA、+ DNA、protein(3) + NA、With complex），一个点 = 一个唯一系统（组分集合，不含浓度）。
坐标沿用 Fig. S1：每张图自己的 t-SNE（ESMC-600M 组分嵌入），不重新计算；这一页只有系统级信息（组分、记录数、阳性 / 阴性数、来源），没有条件。
输入  data/systems/（fsys_points.csv、systems_multi.tsv、sys_summary.csv、sys_tsne_summary.json；缺了就从 world_and_cluster_maps_2026-10-27 复制 / 过滤）
输出  systems_page/page_fragment.html（Artifact 用）、systems_page/index.html（独立文件）、systems_page/summary.json
公开页不放的：只来自实验室笔记本（OpenPhase_db2）的系统——仓库 data/README 写明 db2 的记录不在公开表里，这里同样处理（改 EXCLUDE_SOURCES 即可放回）。
用法：python3 -I scripts/rs21_build_systems.py"""
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

# the system colour table (Fill, Point) for the types that occur in Fig. S1; Fill = tags, Point = dots
FINE = {"protein(2)": ("#80ACED", "#4474B7"), "protein(3)": ("#68D5D6", "#19858E"), "protein(\u22654)": ("#9BB6D1", "#4C637E"),
        "protein(2) + RNA": ("#F4A48C", "#BA553B"), "protein(3) + RNA": ("#F4A6CD", "#A84076"), "protein(\u22654) + RNA": ("#EBC0AC", "#91432F"),
        "protein(2) + DNA": ("#A7D9AD", "#39824E"), "protein(3) + DNA": ("#95D4BE", "#217660"), "protein(\u22654) + DNA": ("#D4DDA0", "#657626"), "Other": ("#E8E8E8", "#666666")}
FINE_NAMES = list(FINE)
TAB_TYPES = {"p2": ["protein(2)"], "p3": ["protein(3)"], "p4": ["protein(\u22654)"], "p2r": ["protein(2) + RNA"], "p2d": ["protein(2) + DNA"], "p3na": ["protein(3) + RNA", "protein(3) + DNA"], "cx": ["Other"]}


def fine_type(raw):
    """v4 raw system type -> a type of the colour table (complexes are 'Other')."""
    r = raw.strip().lower()
    if "complex" in r or r == "protein(0)":
        return "Other"
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


def main():
    prep()
    pts = list(csv.DictReader(open(f"{DATA}/fsys_points.csv", encoding="utf-8")))
    meta = {r["sys_id"]: r for r in csv.DictReader(open(f"{DATA}/systems_multi.tsv", encoding="utf-8"), delimiter="\t")}
    summ = {r["system"]: r for r in csv.DictReader(open(f"{DATA}/sys_summary.csv", encoding="utf-8"))}
    tsne = json.load(open(f"{DATA}/sys_tsne_summary.json"))
    by = defaultdict(list)
    hidden = Counter(); hidden_by = defaultdict(Counter)
    for r in pts:
        srcs = [s for s in meta[r["sys_id"]]["sources"].split(";") if s]
        if srcs and all(s in EXCLUDE_SOURCES for s in srcs):
            hidden["systems"] += 1; hidden["records"] += int(r["n_records"]); hidden_by[r["system"]]["systems"] += 1; hidden_by[r["system"]]["records"] += int(r["n_records"]); continue
        by[r["system"]].append(r)
    groups, report = [], {}
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
        srcs = [";".join(s for s in meta[r["sys_id"]]["sources"].split(";") if s and s not in EXCLUDE_SOURCES) for r in rows]
        rec = [int(r["n_records"]) for r in rows]; pos = [int(r["n_pos"]) for r in rows]; neg = [int(r["n_neg"]) for r in rows]
        sm = summ[gname]
        g = {"key": key, "name": disp, "fam": fam, "desc": desc, "n": len(rows), "records": sum(rec), "pos": sum(pos), "neg": sum(neg),
             "all_n": int(sm["unique_systems"]), "all_records": int(sm["records"]), "no_seq": tsne[gname]["skipped_no_protein_sequence"], "hid_n": hidden_by[gname]["systems"], "hid_rec": hidden_by[gname]["records"],
             "x": b64("H", X), "y": b64("H", Y), "rec": b64("H", [min(v, 65535) for v in rec]), "p": b64("H", pos), "q": b64("H", neg), "anc": b64("B", anc),
             "anchors": anchors, "id": [r["sys_id"] for r in rows], "comp": [cut(r["components"], 120) for r in rows], "src": srcs,
             "raw": [meta[r["sys_id"]]["raw_type"] for r in rows], "nuc": nuc,
             "ft": b64("B", [FINE_NAMES.index(fine_type(meta[r["sys_id"]]["raw_type"])) for r in rows]),
             "tab": [FINE[t][0] for t in TAB_TYPES[key]], "tabp": [FINE[t][1] for t in TAB_TYPES[key]]}
        groups.append(g)
        report[disp] = {"drawn": len(rows), "records": sum(rec), "pos": sum(pos), "neg": sum(neg), "anchors": [(a["name"], a["n"]) for a in anchors], "systems_in_group": g["all_n"], "no_sequence": g["no_seq"], "records_in_group": g["all_records"]}
    data = {"fine": [{"name": n, "fill": FINE[n][0], "point": FINE[n][1]} for n in FINE_NAMES], "groups": groups, "hidden": {"systems": hidden["systems"], "records": hidden["records"]}, "srcLab": SRC_LAB,
            "total": {"systems": sum(g["n"] for g in groups), "records": sum(g["records"] for g in groups)}}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    tpl = open(f"{ROOT}/systems_page/systems_template.html", encoding="utf-8").read()
    assert tpl.count("/*__DATA__*/null") == 1
    frag = tpl.replace("/*__DATA__*/null", blob)
    open(f"{ROOT}/systems_page/page_fragment.html", "w", encoding="utf-8").write(frag)
    head, body = frag.split("</style>", 1)
    standalone = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + head + "</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    open(f"{ROOT}/systems_page/index.html", "w", encoding="utf-8").write(standalone)
    json.dump({"groups": report, "hidden": data["hidden"], "total": data["total"]}, open(f"{ROOT}/systems_page/summary.json", "w"), ensure_ascii=False, indent=1)
    print("systems drawn", data["total"], "| hidden (lab notebooks only)", data["hidden"], "| json MB", round(len(blob) / 1e6, 2), "| file MB", round(len(standalone) / 1e6, 2))
    for k, v in report.items():
        print(f"  {k:18s} drawn {v['drawn']:4d} records {v['records']:5d} pos {v['pos']:5d} neg {v['neg']:5d} | anchors {v['anchors']}")


if __name__ == "__main__":
    main()
