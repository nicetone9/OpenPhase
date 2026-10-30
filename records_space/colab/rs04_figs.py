"""rs04（Colab）：records 空间的静态图（英文标签；配色沿用 world_and_cluster_maps 第 3 轮的鲜艳配色）。PNG 300 dpi + PDF。
版式：每个面板 = 方形散点图 + 其下的图例区（分类图例，或水平色条 + 特殊状态图例），色条与图例不会挤出画布。
输入  data/rs_records.tsv、out/rs_coords.tsv、out/diagnostics.json、out/diagnostics_extra.json（可选）、out/missing_flags.tsv
输出  figs/FigR1_sequence_views_{umap,tsne}、FigR2_condition_views_*、FigR3_readout_provenance_missingness_*、FigR4_lens_comparison、FigR5_diagnostics
用法  python rs04_figs.py --data /content/rs/data --out /content/rs/out --figs /content/rs/figs
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------- 样式
INK, MUTE = "#242424", "#595959"
NEUTRAL, DARK, PALE = "#D5D9DE", "#2B2D42", "#E3E7EB"
CAT = ["#20B7ED", "#EF7182", "#98C93C", "#A47BDD", "#E7AC24", "#159C98", "#D87234", "#427FDB", "#BE629E", "#35A45B", "#E45A62", "#6B2A6B"]
POS_C, NEG_C = "#6B2A6B", "#008DB8"
NA_COL = {"none": "#159AD6", "RNA": "#E45A62", "DNA": "#81BA36"}
SRC_COL = {"LLPSDB_v2": "#E45A62", "LLPSDB_v2_site": "#D87234", "LLPSDB_v2_phase_diagram": "#E7AC24", "RNAPSEC": "#159C98",
           "LLPSense": "#BE629E", "PhaSepDB3_pilot_LLM": "#427FDB", "Bremer2022": "#6B2A6B", "Ausserwoger2025": "#A47BDD"}
SRC_LAB = {"LLPSDB_v2": "LLPSDB v2 (download)", "LLPSDB_v2_site": "LLPSDB v2 (site)", "LLPSDB_v2_phase_diagram": "LLPSDB v2 (phase diagram)",
           "RNAPSEC": "RNAPSEC", "LLPSense": "LLPSense", "PhaSepDB3_pilot_LLM": "PhaSepDB 3 (LLM pilot)", "Bremer2022": "Bremer 2022 (csat)",
           "Ausserwoger2025": "Ausserwöger 2025 (csat)"}
TURBO = mpl.colormaps["turbo"]
CONT = LinearSegmentedColormap.from_list("turbo_t", TURBO(np.linspace(0.08, 0.93, 256)))
DIV = LinearSegmentedColormap.from_list("ncpr", ["#008DB8", "#F1F4F7", "#E45A62"])
MM = 1 / 25.4
mpl.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "axes.titleweight": "bold", "axes.labelsize": 6, "legend.fontsize": 5.6,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.5, "figure.dpi": 100})


# ----------------------------------------------------------------------------- 版式
class Grid:
    """nrows × ncols 个面板；每个面板 = 方形绘图区 + 图例区；返回 (fig, plots, legs)。"""

    def __init__(self, nrows, ncols, width_mm=183, leg=0.44, gap=0.24, wspace=0.12):
        left, right, top, bottom = 0.018, 0.985, 0.94, 0.004
        cell = width_mm * (right - left) / (ncols + wspace * (ncols - 1))           # mm
        unit = cell                                                                  # 高度单位 = 一个绘图区的边长
        height = unit * (nrows * (1 + leg + gap) - gap + 0.02) / (top - bottom)
        self.fig = plt.figure(figsize=(width_mm * MM, height * MM))
        gs = self.fig.add_gridspec(nrows * 3, ncols, height_ratios=[1, leg, gap] * (nrows - 1) + [1, leg, 0.02], left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=0.0)
        self.plots = [[self.fig.add_subplot(gs[3 * r, c]) for c in range(ncols)] for r in range(nrows)]
        self.legs = [[self.fig.add_subplot(gs[3 * r + 1, c]) for c in range(ncols)] for r in range(nrows)]
        for row in self.legs:
            for a in row:
                a.axis("off")
        self.fig.subplots_adjust()

    def at(self, r, c):
        return self.plots[r][c], self.legs[r][c]


def style_ax(ax, title, letter=None):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("#9AA3AB")
    ax.set_box_aspect(1)
    ax.set_title(title, loc="left", pad=2.5, color=INK)
    if letter:
        ax.text(-0.01, 1.09, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom", color=INK)


def pad_lim(ax, x, y, m=0.03):
    dx, dy = np.ptp(x), np.ptp(y)
    ax.set_xlim(x.min() - m * dx, x.max() + m * dx); ax.set_ylim(y.min() - m * dy, y.max() + m * dy)


def _dot(c):
    return Line2D([], [], marker="o", ls="", ms=3.4, mfc=c, mec="none")


def cat_panel(ax, la, x, y, cats, order, colors, title, letter=None, s=2.4, big=None, ncol=2, labels=None, counts=True, draw_first=None):
    cats = np.asarray(cats)
    draw = list(order) if draw_first is None else list(draw_first) + [o for o in order if o not in draw_first]
    for c in draw:
        m = cats == c
        if m.any():
            ax.scatter(x[m], y[m], s=s * (big or {}).get(c, 1.0), c=colors[c], linewidths=0, rasterized=True, zorder=2 + (big or {}).get(c, 1.0) * 0.1)
    present = [c for c in order if (cats == c).any()]
    hs = [_dot(colors[c]) for c in present]
    ls = [(labels or {}).get(c, c) + (f" ({int((cats == c).sum()):,})" if counts else "") for c in present]
    la.legend(hs, ls, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, frameon=False, handletextpad=0.1, columnspacing=0.8, labelspacing=0.25, borderaxespad=0)
    style_ax(ax, title, letter); pad_lim(ax, x, y)


def cont_panel(fig, ax, la, x, y, v, title, letter=None, cmap=CONT, vmin=None, vmax=None, s=2.4, missing=None, missing_color=DARK, missing_label="missing",
               special=None, cbar_label=None, ticks=None):
    """连续变量；missing：布尔掩码（深色）；special：[(掩码, 颜色, 标签)] 额外状态。色条水平放在图例区上部，状态图例放在下部。"""
    v = np.asarray(v, dtype=float)
    ok = ~np.isnan(v)
    if missing is not None:
        ok &= ~missing
    hs, ls = [], []
    for m, c, lab in (special or []):
        if m.any():
            ax.scatter(x[m], y[m], s=s, c=c, linewidths=0, rasterized=True, zorder=2)
            hs.append(_dot(c)); ls.append(f"{lab} ({int(m.sum()):,})")
            ok &= ~m
    if missing is not None and missing.any():
        ax.scatter(x[missing], y[missing], s=s, c=missing_color, linewidths=0, rasterized=True, zorder=3)
        hs.append(_dot(missing_color)); ls.append(f"{missing_label} ({int(missing.sum()):,})")
    vmin = np.nanquantile(v[ok], 0.02) if vmin is None else vmin
    vmax = np.nanquantile(v[ok], 0.98) if vmax is None else vmax
    sc = ax.scatter(x[ok], y[ok], s=s, c=v[ok], cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, rasterized=True, zorder=4)
    cax = la.inset_axes([0.08, 0.76, 0.84, 0.12])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=ticks)
    cb.ax.tick_params(labelsize=5.2, width=0.4, length=2, pad=1); cb.outline.set_linewidth(0.4)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=5.4, labelpad=1)
    if hs:
        la.legend(hs, ls, loc="upper center", bbox_to_anchor=(0.5, 0.40), ncol=1, frameon=False, handletextpad=0.1, labelspacing=0.25, borderaxespad=0)
    style_ax(ax, title, letter); pad_lim(ax, x, y)


# ----------------------------------------------------------------------------- 数据准备
def prepare(data, out):
    df = pd.read_csv(f"{data}/rs_records.tsv", sep="\t", keep_default_na=False, na_values=[""], low_memory=False)
    co = pd.read_csv(f"{out}/rs_coords.tsv", sep="\t")
    assert (co["rs_idx"].to_numpy() == df["rs_idx"].to_numpy()).all()
    fl = pd.read_csv(f"{out}/missing_flags.tsv", sep="\t")
    d = df.copy()
    num = lambda c: pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float)
    d["pool"] = d["pool"].astype(str)
    uni = d["uniprot"].fillna("").astype(str).str.split("-").str[0]
    d["pkey"] = np.where(uni != "", uni, "NA:" + d["seq_row"].astype(str))
    nm = {}
    for k, g in d.groupby("pkey"):
        gb = g[g["pool"] == "binary"]
        nm[k] = (gb if len(gb) else g)["protein_name"].value_counts().index[0]
    nm["P09651"] = "hnRNPA1 (A1-LCD)"
    nm["P00698"] = "P00698 (lysozyme C)"
    nm["Q07352,P35637"] = "Q07352 + P35637 (two UniProt IDs)"
    d["pname"] = d["pkey"].map(nm)
    d["temp_state"] = d["st_temp"].map({"obs": "measured", "rt": "room temp. only", "miss": "missing"})
    d["tempC"] = num("temp")
    tc = d["tempC"].to_numpy()
    band = np.where(d["st_temp"] == "miss", "missing", np.where(d["st_temp"] == "rt", "room temp. only",
            np.where(tc <= 10, "≤ 10 °C", np.where(tc <= 20, "10–20 °C", np.where(tc <= 30, "20–30 °C", np.where(tc <= 40, "30–40 °C", "> 40 °C"))))))
    d["temp_band"] = band
    d["pH_v"] = np.where(d["st_pH"] == "obs", num("pH"), np.nan)
    nacl = num("NaCl")
    d["nacl_log"] = np.where(d["st_NaCl"].isin(["obs", "zero"]), np.log10(1 + np.clip(nacl, 0, None)), np.nan)
    d["conc_v"] = np.where(d["st_conc"] == "obs", num("conc"), np.nan)
    peg = np.nan_to_num(num("PEG1k")) + np.nan_to_num(num("PEG6k")) + np.nan_to_num(num("PEG20k"))
    fic = np.nan_to_num(num("Ficoll")) + np.nan_to_num(num("Dex40")) + np.nan_to_num(num("Dex70"))
    gly = np.nan_to_num(num("Glycerol"))
    unk = np.all([d[f"st_{k}"] == "miss" for k in ["PEG1k", "PEG6k", "PEG20k", "Ficoll", "Dex40", "Dex70", "Glycerol"]], axis=0)
    d["crowder"] = np.where(unk, "not given", np.where(peg > 0, "PEG", np.where(fic > 0, "Ficoll / dextran", np.where(gly > 0, "glycerol", "none reported"))))
    bf = []
    for _, r in d[[f"buf_{k}" for k in ["Tris", "HEPES", "phosphate", "goodsMES", "citrate", "acetate", "imidazole", "other_buf"]] + ["st_buffer"]].iterrows():
        if r["st_buffer"] == "miss":
            bf.append("not given")
        elif r["buf_Tris"] == 1:
            bf.append("Tris")
        elif r["buf_HEPES"] == 1:
            bf.append("HEPES")
        elif r["buf_phosphate"] == 1:
            bf.append("phosphate / PBS")
        elif r["buf_goodsMES"] == 1:
            bf.append("MES / MOPS / PIPES")
        elif r["buf_citrate"] == 1 or r["buf_acetate"] == 1:
            bf.append("citrate / acetate")
        else:
            bf.append("other")
    d["buffer_f"] = bf
    d["additive_f"] = d["st_additive"].map({"obs": "additive listed", "none": "none recorded", "miss": "not given"})
    lab = pd.to_numeric(d["label"], errors="coerce")
    d["ps"] = np.where(d["pool"] == "csat", "quantitative only (csat pool)", np.where(lab == 1, "phase separation", "no phase separation"))
    d["csat_status"] = d["csat_link"].map({"none": "no csat", "same UniProt": "binary, same UniProt as a csat protein", "same sequence": "binary, identical sequence to a csat protein", "csat record": "csat record"})
    constr = np.where(d["mutation"].astype(int) == 1, "point mutant", np.where(d["fusion"].astype(int) == 1, "fusion tag", np.where(d["cleaved"].astype(int) == 1, "truncation / domain", "full length / wild type")))
    constr = np.where((d["pool"] == "csat") & (d["seq_state"] == "tag_not_in_sequence"), "fusion tag", constr)
    constr = np.where((d["pool"] == "csat") & (d["source"] == "Bremer2022"), "point mutant", constr)
    d["construct"] = constr
    d["len_log"] = np.log10(pd.to_numeric(d["seq_len"], errors="coerce"))
    d["ncpr"] = num("comp_21")
    d["arom"] = num("comp_23")
    d["n_miss"] = fl.sum(1).to_numpy()
    d["y_csat"] = num("y_log10")
    d["na_ratio"] = num("na_ratio_log10")
    d["na_len"] = num("na_len_log10")
    return d, co, fl, nm


def proteins_top(d, k=12):
    cnt = d["pkey"].value_counts()
    keys = [c for c in cnt.index if not c.startswith("NA:")][:k]
    cats = np.where(d["pkey"].isin(keys), d["pname"], np.where(d["pkey"].str.startswith("NA:"), "no UniProt", "other proteins"))
    order = [d.loc[d["pkey"] == c, "pname"].iloc[0] for c in keys]
    colors = {n: CAT[i % len(CAT)] for i, n in enumerate(order)}
    colors["no UniProt"], colors["other proteins"] = "#A9B1B9", NEUTRAL
    return cats, order + ["no UniProt", "other proteins"], colors


def save(fig, figs, name):
    os.makedirs(figs, exist_ok=True)
    fig.savefig(f"{figs}/{name}.png", dpi=300, facecolor="white")
    fig.savefig(f"{figs}/{name}.pdf", facecolor="white")
    plt.close(fig)
    print("saved", name, flush=True)


def xy(co, emb):
    return co[f"{emb}_x"].to_numpy(), co[f"{emb}_y"].to_numpy()


# ----------------------------------------------------------------------------- 图
def fig_R1(d, co, figs, emb, tag):
    x, y = xy(co, emb)
    G = Grid(2, 3)
    cats, order, colors = proteins_top(d)
    cat_panel(*G.at(0, 0), x, y, cats, order, colors, "Protein identity (top 12)", "a", ncol=2, draw_first=["other proteins", "no UniProt"])
    cont_panel(G.fig, *G.at(0, 1), x, y, d["len_log"], "Protein length", "b", cbar_label="log10 residues", ticks=[1.5, 2.0, 2.5])
    cont_panel(G.fig, *G.at(0, 2), x, y, d["ncpr"], "Net charge per residue", "c", cmap=DIV, vmin=-0.15, vmax=0.15, cbar_label="NCPR", ticks=[-0.15, 0, 0.15])
    cont_panel(G.fig, *G.at(1, 0), x, y, d["arom"], "Aromatic residues (F + W + Y)", "d", vmin=0, vmax=0.2, cbar_label="fraction of residues", ticks=[0, 0.1, 0.2])
    cat_panel(*G.at(1, 1), x, y, d["na_class"], ["none", "RNA", "DNA"], NA_COL, "Nucleic acid as a condition", "e", ncol=1, labels={"none": "no nucleic acid"}, draw_first=["none"])
    cc = {"full length / wild type": "#BFC7CF", "truncation / domain": "#20B7ED", "point mutant": "#E7AC24", "fusion tag": "#E45A62"}
    cat_panel(*G.at(1, 2), x, y, d["construct"], list(cc), cc, "Construct type", "f", ncol=1, draw_first=["full length / wild type"])
    save(G.fig, figs, f"FigR1_sequence_views_{tag}")


def fig_R2(d, co, figs, emb, tag):
    x, y = xy(co, emb)
    G = Grid(3, 3)
    cont_panel(G.fig, *G.at(0, 0), x, y, d["tempC"], "Temperature", "a", vmin=0, vmax=45, missing=(d["temp_state"] == "missing").to_numpy(), missing_label="missing",
               special=[((d["temp_state"] == "room temp. only").to_numpy(), "#BFDDF0", "room temp. only (not measured)")], cbar_label="°C (measured)", ticks=[0, 15, 30, 45])
    bc_ = {"≤ 10 °C": "#146DB5", "10–20 °C": "#20B7ED", "20–30 °C": "#98C93C", "30–40 °C": "#E7AC24", "> 40 °C": "#E45A62", "room temp. only": "#CFE3F2", "missing": DARK}
    cat_panel(*G.at(0, 1), x, y, d["temp_band"], list(bc_), bc_, "Temperature band", "b", ncol=2, draw_first=["room temp. only"])
    cont_panel(G.fig, *G.at(0, 2), x, y, d["pH_v"], "pH", "c", vmin=5, vmax=9, missing=d["pH_v"].isna().to_numpy(), missing_label="missing", cbar_label="pH", ticks=[5, 6, 7, 8, 9])
    cont_panel(G.fig, *G.at(1, 0), x, y, d["nacl_log"], "NaCl", "d", vmin=0, vmax=2.7, missing=d["nacl_log"].isna().to_numpy(), missing_label="missing", cbar_label="log10(1 + mM)", ticks=[0, 1, 2])
    cont_panel(G.fig, *G.at(1, 1), x, y, d["conc_v"], "Protein concentration", "e", vmin=-0.5, vmax=2.5, missing=d["conc_v"].isna().to_numpy(),
               missing_label="missing (csat: label / scan axis)", cbar_label="log10 µM", ticks=[0, 1, 2])
    cc = {"none reported": "#BFC7CF", "PEG": "#E45A62", "Ficoll / dextran": "#E7AC24", "glycerol": "#98C93C", "not given": DARK}
    cat_panel(*G.at(1, 2), x, y, d["crowder"], list(cc), cc, "Crowder", "f", ncol=2, draw_first=["none reported"])
    bc = {"Tris": "#20B7ED", "HEPES": "#E45A62", "phosphate / PBS": "#98C93C", "MES / MOPS / PIPES": "#A47BDD", "citrate / acetate": "#E7AC24", "other": "#159C98", "not given": DARK}
    cat_panel(*G.at(2, 0), x, y, d["buffer_f"], list(bc), bc, "Buffer family", "g", ncol=2)
    ac = {"none recorded": "#BFC7CF", "additive listed": "#E7AC24", "not given": DARK}
    cat_panel(*G.at(2, 1), x, y, d["additive_f"], list(ac), ac, "Additives (DTT, EDTA, glycerol, ATP …)", "h", ncol=1, draw_first=["none recorded"])
    has_na = (d["na_class"] != "none").to_numpy()
    cont_panel(G.fig, *G.at(2, 2), x, y, d["na_ratio"], "Nucleic acid : protein ratio", "i", vmin=-2, vmax=3, missing=(has_na & np.isnan(d["na_ratio"].to_numpy())), missing_label="nucleic acid present, ratio unknown",
               special=[(~has_na, PALE, "no nucleic acid")], cbar_label="log10 molar ratio", ticks=[-2, 0, 2])
    save(G.fig, figs, f"FigR2_condition_views_{tag}")


def fig_R3(d, co, figs, emb, tag):
    x, y = xy(co, emb)
    G = Grid(2, 3)
    pc = {"no phase separation": NEG_C, "phase separation": POS_C, "quantitative only (csat pool)": "#BFC7CF"}
    cat_panel(*G.at(0, 0), x, y, d["ps"], ["quantitative only (csat pool)", "no phase separation", "phase separation"], pc, "Phase-separation label", "a", ncol=1, draw_first=["quantitative only (csat pool)"])
    cs = {"no csat": "#D5D9DE", "binary, same UniProt as a csat protein": "#E7AC24", "binary, identical sequence to a csat protein": "#EF7182", "csat record": "#6B2A6B"}
    cat_panel(*G.at(0, 1), x, y, d["csat_status"], list(cs), cs, "Has csat?", "b", ncol=1, big={"csat record": 2.6, "binary, same UniProt as a csat protein": 1.4, "binary, identical sequence to a csat protein": 1.6}, draw_first=["no csat"])
    rc = {"binary": "#E3E7EB", "dilute": "#20B7ED", "csat_boundary": "#6B2A6B", "dilute_linescan": "#98C93C", "cloud_point": "#E7AC24", "dense": "#EF7182", "unclassified": "#A9B1B9"}
    rl = {"binary": "binary record (no csat)", "csat_boundary": "csat (boundary)", "dilute": "dilute phase", "dilute_linescan": "dilute phase (line scan)", "cloud_point": "cloud point", "dense": "dense phase"}
    cat_panel(*G.at(0, 2), x, y, d["readout_class"], ["binary", "dilute", "csat_boundary", "dilute_linescan", "cloud_point", "dense"], rc, "Quantitative readout", "c", ncol=1, labels=rl,
              big={"dilute": 2.2, "csat_boundary": 2.2, "dilute_linescan": 2.2, "cloud_point": 2.2, "dense": 2.2}, draw_first=["binary"])
    cat_panel(*G.at(1, 0), x, y, d["source"].astype(str), list(SRC_COL), SRC_COL, "Source", "d", ncol=1, labels=SRC_LAB, counts=False)
    cont_panel(G.fig, *G.at(1, 1), x, y, d["y_csat"], "csat / dilute-phase value", "e", vmin=0, vmax=3, missing=d["y_csat"].isna().to_numpy(), missing_color=PALE,
               missing_label="no value (binary records; 2 csat rows with value 0)", cbar_label="log10 µM (csat pool)", ticks=[0, 1, 2, 3])
    nmv = d["n_miss"].to_numpy()
    cmap = ListedColormap(["#20B7ED", "#98C93C", "#E7AC24", "#D87234", "#E45A62", "#6B2A6B", "#2B2D42"])
    cont_panel(G.fig, *G.at(1, 2), x, y, np.clip(nmv, 0, 6), "Missing condition dimensions", "f", cmap=cmap, vmin=-0.5, vmax=6.5, cbar_label="number missing (6 = six or more)", ticks=[0, 1, 2, 3, 4, 5, 6])
    save(G.fig, figs, f"FigR3_readout_provenance_missingness_{tag}")


def fig_R4(d, co, figs):
    lenses = [("umap_full", "full"), ("umap_prot", "protein only"), ("umap_cond", "NA + conditions"), ("umap_fullM", "full, observed-only"), ("tsne_full", "full, t-SNE")]
    lenses = [(k, t) for k, t in lenses if f"{k}_x" in co.columns]
    cats, order, colors = proteins_top(d, 8)
    rows = [("Protein identity", cats, order, colors, "Protein identity (top 8)"),
            ("Nucleic acid", d["na_class"].to_numpy(), ["none", "RNA", "DNA"], NA_COL, "Nucleic acid"),
            ("Temperature state", d["temp_state"].to_numpy(), ["measured", "room temp. only", "missing"], {"measured": "#20B7ED", "room temp. only": "#E7AC24", "missing": "#2B2D42"}, "Temperature state"),
            ("Source", d["source"].astype(str).to_numpy(), list(SRC_COL), SRC_COL, "Source")]
    n_c = len(lenses)
    cell = 183 * 0.96 / (n_c + 0.06 * (n_c - 1))
    fig = plt.figure(figsize=(183 * MM, (len(rows) * cell * 1.02 + 38) * MM))
    gs = fig.add_gridspec(len(rows) + 1, n_c, height_ratios=[cell * 1.02] * len(rows) + [38], left=0.03, right=0.99, top=0.965, bottom=0.01, wspace=0.06, hspace=0.12)
    for j, (k, lt) in enumerate(lenses):
        x, y = xy(co, k)
        for i, (rt, cats_, order_, cols, _) in enumerate(rows):
            ax = fig.add_subplot(gs[i, j])
            cats_ = np.asarray(cats_)
            for c in (order_[::-1] if i == 0 else order_):
                m = cats_ == c
                if m.any():
                    ax.scatter(x[m], y[m], s=1.1, c=cols[c], linewidths=0, rasterized=True)
            style_ax(ax, lt if i == 0 else "", None)
            pad_lim(ax, x, y)
            if j == 0:
                ax.set_ylabel(rt, fontsize=6.5, fontweight="bold", color=INK)
    for i, (rt, cats_, order_, cols, ttl) in enumerate(rows):
        la = fig.add_subplot(gs[len(rows), i])
        la.axis("off")
        present = [c for c in order_ if (np.asarray(cats_) == c).any()]
        la.legend([_dot(cols[c]) for c in present], [SRC_LAB.get(c, c) for c in present], title=ttl, title_fontsize=5.8, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=1, frameon=False,
                  handletextpad=0.1, labelspacing=0.22, borderaxespad=0, fontsize=5.2, alignment="left")
    save(fig, figs, "FigR4_lens_comparison")


def fig_R5(diag, extra, fl, d, figs):
    fig, axs = plt.subplots(2, 2, figsize=(183 * MM, 150 * MM), gridspec_kw=dict(left=0.085, right=0.93, top=0.91, bottom=0.12, wspace=0.34, hspace=0.62))
    lens_order = ["full", "prot", "cond", "fullM"]
    lens_lab = {"full": "full", "prot": "protein", "cond": "NA + cond.", "fullM": "full (M−)"}
    letter = lambda ax, t: ax.text(-0.02, 1.2, t, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")
    # a
    ax = axs[0, 0]
    src = (extra or {}).get("knn_auroc_ps_label_k15_boot")
    tags = ["any", "cross-paper", "cross-protein", "cross-both"]
    cols = {"any": "#20B7ED", "cross-paper": "#6B2A6B", "cross-protein": "#E7AC24", "cross-both": "#E45A62"}
    names = {"any": "all neighbours", "cross-paper": "other papers", "cross-protein": "other proteins", "cross-both": "other papers and proteins"}
    w = 0.2
    for k, t in enumerate(tags):
        if src is not None:
            vals = [src[l][t]["auroc"] for l in lens_order]; ses = [src[l][t]["se"] for l in lens_order]
        else:
            if t == "cross-both":
                continue
            vals = [diag["knn_auroc_ps_label_k15"][l][t]["auroc"] for l in lens_order]; ses = None
        ax.bar(np.arange(4) + (k - 1.5) * w, vals, w, color=cols[t], label=names[t], yerr=ses, error_kw=dict(lw=0.5, capsize=1.0, ecolor=INK))
    ax.axhline(0.5, color=INK, lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks(range(4)); ax.set_xticklabels([lens_lab[l] for l in lens_order]); ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC of neighbour-mean label")
    ax.set_title("Do neighbours share the phase-separation label?", loc="left", pad=3)
    ax.legend(frameon=False, loc="upper center", ncol=2, labelspacing=0.2, columnspacing=0.8, handlelength=1.0, borderaxespad=0.1)
    letter(ax, "a")
    # b
    ax = axs[0, 1]
    pur = diag["purity_k15"]
    pn = ["protein (UniProt or sequence)", "paper", "source", "nucleic acid class", "temperature state", "buffer family", "phase-separation label"]
    ps = ["protein", "paper", "source", "nucleic acid", "temperature state", "buffer", "PS label"]
    y_ = np.arange(len(pn))[::-1]
    for l, c, off in (("full", "#20B7ED", 0.18), ("cond", "#E45A62", -0.18)):
        ax.barh(y_ + off, [pur[n][l][0] for n in pn], 0.34, color=c, label={"full": "full", "cond": "NA + conditions"}[l])
    ax.scatter([pur[n]["full"][1] for n in pn], y_, marker="|", s=40, color=INK, label="chance", zorder=5, linewidths=0.9)
    ax.set_yticks(y_); ax.set_yticklabels(ps); ax.set_xlim(0, 1)
    ax.set_xlabel("share of 15 nearest neighbours with the same label")
    ax.set_title("What organises the space?", loc="left", pad=16)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, labelspacing=0.2, columnspacing=0.8, handlelength=1.0, borderaxespad=0.1)
    letter(ax, "b")
    # c
    ax = axs[1, 0]
    ov = diag["neighbor_overlap_k15"]
    keys = [k for k in ["umap_full", "umap_full_k10", "umap_full_k60", "tsne_full", "umap_prot", "umap_cond", "umap_fullM"] if k in ov]
    lab = {"umap_full": "UMAP full\nk = 30", "umap_full_k10": "UMAP full\nk = 10", "umap_full_k60": "UMAP full\nk = 60", "tsne_full": "t-SNE\nfull", "umap_prot": "UMAP\nprotein", "umap_cond": "UMAP\nNA + cond.", "umap_fullM": "UMAP\nfull (M−)"}
    ax.bar(range(len(keys)), [ov[k] for k in keys], color=["#20B7ED", "#7DD3F5", "#7DD3F5", "#98C93C", "#E7AC24", "#E45A62", "#A47BDD"][:len(keys)])
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([lab[k] for k in keys], fontsize=4.9, rotation=35, ha="right"); ax.set_ylim(0, 1)
    ax.set_ylabel("share of the 15 high-dimensional\nneighbours kept in 2-D")
    ax.set_title("How faithful is the 2-D map?", loc="left", pad=3)
    letter(ax, "c")
    # d
    ax = axs[1, 1]
    srcs_ = d["source"].astype(str)
    srcs = [s for s in SRC_COL if (srcs_ == s).any()]
    dims = list(fl.columns)
    M = np.array([[fl.loc[(srcs_ == s).to_numpy(), c].mean() for c in dims] for s in srcs])
    cm = LinearSegmentedColormap.from_list("miss", ["#F4F8FB", "#20B7ED", "#6B2A6B"])
    im = ax.imshow(M, cmap=cm, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(dims))); ax.set_xticklabels(dims, rotation=60, ha="right", fontsize=5.2)
    ax.set_yticks(range(len(srcs))); ax.set_yticklabels([SRC_LAB[s].replace(" (", "\n(") for s in srcs], fontsize=5.0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.ax.tick_params(labelsize=5.2, width=0.4, length=2); cb.set_label("share of records missing", fontsize=5.4)
    ax.set_title("Missingness follows the source", loc="left", pad=3)
    letter(ax, "d")
    for a_ in (axs[0, 0], axs[0, 1], axs[1, 0]):
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    save(fig, figs, "FigR5_diagnostics")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figs", required=True)
    a = ap.parse_args()
    d, co, fl, nm = prepare(a.data, a.out)
    diag = json.load(open(f"{a.out}/diagnostics.json"))
    extra = json.load(open(f"{a.out}/diagnostics_extra.json")) if os.path.exists(f"{a.out}/diagnostics_extra.json") else None
    for emb, tag in (("umap_full", "umap"), ("tsne_full", "tsne")):
        if f"{emb}_x" in co.columns:
            fig_R1(d, co, a.figs, emb, tag)
            fig_R2(d, co, a.figs, emb, tag)
            fig_R3(d, co, a.figs, emb, tag)
    fig_R4(d, co, a.figs)
    fig_R5(diag, extra, fl, d, a.figs)
    d.drop(columns=[c for c in d.columns if c.startswith("comp_")]).to_csv(f"{a.out}/rs_plot_table.tsv", sep="\t", index=False)
    print("FIGS_DONE", flush=True)


if __name__ == "__main__":
    main()
