#!/usr/bin/env python3
"""rs10（本机，纯标准库）：生成“试新实验”面板的例子 explorer/examples/example_defs.json。
序列来自 UniProt 的规范序列（三条取自 export/examples.json，三条从本地序列表按 md5 取出），每条都核对了 UniProt 给出的 md5。
带 zn_exact 的三条是在 Colab 上算过的精确 ESMC-600M 嵌入。"""
import csv
import hashlib
import json

csv.field_size_limit(10 ** 9)
ROOT = "/Users/lynn/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Projects/LLPS/condition_llps/records_space_2026-10-29"
UNIPROT_MD5 = {
    "P04147": "4AA48E260BC060F53A93EBE60CFD8DBF", "P62993": "CE64B43A36A4F318BB8AEF9B9DFDAA55", "P11940": "412581D5F07C91D4387F72B99EBAB1CD",
    "P35637": "81FEF7CAAE3231906FBE85F45E13269B", "Q13148": "D21783F4E4F170AA5D52741BC444BF07", "P09651": "635B53C9FC307302B28FFB8F156A36B4"}

G_NEW = "New proteins (no record in the dataset)"
G_HELD = "Known proteins, held out (Treat as unseen is on)"
G_PARSE = "Condition-parsing demos"
DEFS = [
    dict(id="grb2", group=G_NEW, title="GRB2 · HEPES, KCl, MgCl2, PEG 8000", name="GRB2", uni="P62993",
         text="20 mM HEPES pH 7.4, 150 mM KCl, 2 mM MgCl2, 1 mM DTT, 10% PEG 8000, 25 °C, 50 µM protein",
         note="GRB2 has no record and no similar sequence in the dataset (class: new). The protein part uses an exact ESMC embedding computed in Colab for this example. Tick the approximation box to see how far the in-browser approximation would move it."),
    dict(id="pabpc1_rna", group=G_NEW, title="PABPC1 + poly(A) RNA · Tris, NaCl, TCEP", name="PABPC1", uni="P11940",
         text="25 mM Tris-HCl pH 7.5, 100 mM NaCl, 1 mM TCEP, 5 µM protein, 10 µM poly(A) RNA 30-nt, 20 °C",
         note="PABPC1 shares only a few 5-mers with a few dataset sequences (class: related). The protein part uses an exact ESMC embedding computed in Colab. The RNA enters as a condition: class, type, length and concentration."),
    dict(id="pab1_variant", group=G_NEW, title="Pab1 · close variant of a dataset construct", name="Pab1", uni="P04147",
         text="10 mM HEPES pH 7.0, 50 mM NaCl, 0.5 mM EDTA, 150 µM protein, 25 °C",
         note="The dataset holds a 597-residue Pab1 construct with no UniProt ID (42 records). The page recognises it from shared 5-mers (class: variant). Here its records are not among the 15 nearest: they were measured under other conditions, so records of other proteins with closer conditions rank first."),
    dict(id="fus_heldout", group=G_HELD, title="FUS, full length · Tris, NaCl, 25 °C", name="FUS", uni="P35637", unseen=True,
         text="20 mM Tris-HCl pH 7.4, 150 mM NaCl, 1 mM DTT, 25 °C, 10 µM protein",
         note="The sequence is identical to a dataset sequence. With Treat as unseen on, FUS, its close variants and its 30% homology cluster are left out of the neighbours, and coverage is judged against records seen as a new protein."),
    dict(id="tdp43_rna", group=G_HELD, title="TDP-43 + RNA · HEPES, NaCl, 4 °C", name="TDP-43", uni="Q13148", unseen=True,
         text="20 mM HEPES pH 7.0, 150 mM NaCl, 1 mM DTT, 4 °C, 5 µM protein, 5 µM 12-nt RNA",
         note="Same idea for TDP-43 with a short RNA. With TDP-43 and its homology cluster left out, the neighbours come from other proteins."),
    dict(id="hnrnpa1", group=G_HELD, title="hnRNPA1, full length · MES, NaCl, 4 °C", name="hnRNPA1", uni="P09651", unseen=True,
         text="20 mM MES pH 6.0, 150 mM NaCl, 4 °C, 100 µM protein",
         note="Full-length hnRNPA1 (372 residues). The dataset also holds its low-complexity domain as separate constructs; Treat as unseen leaves those out too."),
    dict(id="parse_pbs", group=G_PARSE, title="FUS · PBS, glycerol, room temperature", name="FUS", uni="P35637",
         text="PBS pH 7.4, 1 mM DTT, 10% glycerol, room temperature, 20 µM protein",
         note="PBS is expanded to 137 mM NaCl + 2.7 mM KCl, a stated assumption you can switch off. Room temperature is its own state, not a measured 22 °C. Crowders not mentioned count as none."),
    dict(id="parse_mass", group=G_PARSE, title="GRB2 · mg/mL, 0.15 M NaCl, Ficoll, Kelvin", name="GRB2", uni="P62993",
         text="Protein at 2 mg/mL in 25 mM sodium phosphate pH 6.8, 0.15 M NaCl, 5% (w/v) Ficoll 70, 310 K",
         note="mg/mL is converted to µM with the sequence mass; 0.15 M becomes 150 mM; 310 K becomes 36.9 °C; Ficoll 70 is filed under Ficoll."),
    dict(id="seq_only", group=G_PARSE, title="PABPC1 · sequence only, no conditions", name="PABPC1", uni="P11940", text="", unmSalts="unknown", unmCrowd="unknown",
         note="With no conditions every condition dimension is unknown, so the distance rests on the protein alone and coverage is judged against that."),
]


def main():
    ex = {e["id"]: e for e in json.load(open(f"{ROOT}/export/examples.json"))}
    seqs = {}
    for r in csv.DictReader(open(f"{ROOT}/data/rs_sequences.tsv"), delimiter="\t"):
        h = hashlib.md5(r["sequence"].upper().encode()).hexdigest().upper()
        seqs.setdefault(h, r["sequence"].upper())
    out = []
    for d in DEFS:
        u = d.pop("uni")
        if u in ex:
            s = ex[u]["sequence"].upper()
        else:
            s = seqs[UNIPROT_MD5[u]]
        assert hashlib.md5(s.encode()).hexdigest().upper() == UNIPROT_MD5[u], u
        e = dict(d, sequence=s, uniprot=u)
        e.setdefault("unseen", False)
        e.setdefault("unmSalts", "unknown")
        e.setdefault("unmCrowd", "zero")
        e.setdefault("pbs", True)
        if u in ex:
            e["zn_exact"] = [round(x, 5) for x in ex[u]["zn_exact"]]
        out.append(e)
    json.dump(out, open(f"{ROOT}/explorer/examples/example_defs.json", "w"), ensure_ascii=False, separators=(",", ":"))
    print("examples", len(out), "with exact ESMC", sum(1 for e in out if "zn_exact" in e), "bytes", len(json.dumps(out)))


if __name__ == "__main__":
    main()
