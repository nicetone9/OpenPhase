#!/usr/bin/env python3
"""rs16（本机，纯标准库）：把 Colab 上 rs15 的输出（pu_map.npz）和 rs14 的清单做成"负例 / 未标注"页：pu_page/page_fragment.html（Artifact 用）、pu_page/index.html（独立文件）。
这一页只有负例（N）和未标注（U），没有正例。
用法：python3 -I scripts/rs16_build_pu.py [pu_map.npz 路径，默认 data/pu/colab_out_nu/pu_map.npz]"""
import base64
import csv
import json
import os
import sys
from array import array
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rs13_universe_export import load_npz                       # noqa: E402

csv.field_size_limit(10 ** 9)
ROOT = "/Users/lynn/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Projects/LLPS/condition_llps/records_space_2026-10-29"
SRC = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/data/pu/colab_out_nu/pu_map.npz"
ROLE = {"U": 0, "N": 1}


def b64(code, vals):
    a = array(code, vals)
    if sys.byteorder != "little":
        a.byteswap()
    return base64.b64encode(a.tobytes()).decode()


def cut(x, n=56):
    x = (x or "").strip()
    if len(x) <= n:
        return x
    c = x[:n].rsplit(" ", 1)[0].rstrip(",;:-(") if " " in x[:n] else x[:n]
    return c + "…"


def main():
    z = load_npz(SRC)
    accs, xy = z["acc"][1], z["xy"][1]
    man = {r["uniprot"]: r for r in csv.DictReader(open(f"{ROOT}/data/pu/pu_manifest.tsv"), delimiter="\t")}
    newm = {}
    pm = os.path.join(os.path.dirname(SRC), "pu_new_meta.tsv")
    if os.path.exists(pm):
        for r in csv.DictReader(open(pm), delimiter="\t"):
            newm[r["uniprot"]] = r
    sets = json.load(open(f"{ROOT}/data/pu/sets.json"))
    assert all(s["role"] == "N" for s in sets), "这一页只放负例集合"
    n = len(accs)
    xs, ys = xy[0::2], xy[1::2]
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    s = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.04
    q = lambda v: min(max(round(v * 65535), 0), 65535)
    X = [q((x - cx) / s + 0.5) for x in xs]
    Y = [q(1 - ((y - cy) / s + 0.5)) for y in ys]
    role, flag, mask, ln, name, gene, org = [], [], [], [], [], [], []
    for a in accs:
        r = man[a]
        m = newm.get(a)
        role.append(ROLE[r["role"]])
        obs = 1 if m and m["obs"] == "1" else 0
        flag.append(int(r["bg"]) | (2 if r["in_records"] == "1" else 0) | (4 if obs else 0))
        mask.append(int(r["mask"]))
        nm, gn, og, l = (m["name"], m["gene"], m["organism"], m["length"]) if m else (r["name"], r["gene"], r["organism"], r["length"])
        ln.append(min(int(float(l)), 65535) if l else 0)
        name.append(cut(nm)); gene.append(gn[:24]); org.append(og[:40])
    for st in sets:
        st["nn"] = sum(1 for v in mask if (v >> st["bit"]) & 1)
        st.pop("n", None)
    pairs = Counter()
    for m in mask:
        bs = [st for st in sets if (m >> st["bit"]) & 1]
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                pairs[(bs[i]["label"], bs[j]["label"])] += 1
    data = {"pairs": [[a, b, c] for (a, b), c in pairs.most_common(6)], "n": n, "x": b64("H", X), "y": b64("H", Y), "role": b64("B", role), "flag": b64("B", flag), "mask": b64("I", mask),
            "len": b64("H", ln), "acc": list(accs), "name": name, "gene": gene, "org": org, "sets": sets, "bg_n": sum(1 for f in flag if f & 1)}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    tpl = open(f"{ROOT}/pu_page/pu_template.html", encoding="utf-8").read()
    assert tpl.count("/*__DATA__*/null") == 1
    frag = tpl.replace("/*__DATA__*/null", blob)
    open(f"{ROOT}/pu_page/page_fragment.html", "w", encoding="utf-8").write(frag)
    head, body = frag.split("</style>", 1)
    standalone = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + head + "</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    open(f"{ROOT}/pu_page/index.html", "w", encoding="utf-8").write(standalone)
    miss = os.path.join(os.path.dirname(SRC), "missing_ids.txt")
    k = Counter(bin(v).count("1") for v in mask if v)
    summ = {"n": n, "roles": {c: sum(1 for v in role if v == i) for c, i in ROLE.items()}, "in_records": sum(1 for f in flag if f & 2), "obsolete": sum(1 for f in flag if f & 4),
            "bg": data["bg_n"], "sets": len(sets), "pairs": data["pairs"], "in_k_sets": dict(sorted(k.items())), "multi": sum(c for kk, c in k.items() if kk >= 2),
            "new_listed": sum(1 for r in man.values() if r["src"] == "new"), "new_embedded": len(newm), "missing_ids": len(open(miss).read().split()) if os.path.exists(miss) else None, "manifest": len(man)}
    json.dump(summ, open(f"{ROOT}/pu_page/summary.json", "w"), ensure_ascii=False, indent=1)
    print("proteins", n, "| roles", summ["roles"], "| in records", summ["in_records"], "| obsolete", summ["obsolete"], "| json MB", round(len(blob) / 1e6, 2), "| file MB", round(len(standalone) / 1e6, 2))


if __name__ == "__main__":
    main()
