"""rs18（Colab CPU）：records 空间的"代表性染色"图。
A  系统类型 × 各染色类别的富集点图（点大小 = 该系统类型里落在这个类别的记录占比，颜色 = 相对全部记录的 log2 富集；版式参照用户给的 GO 点图）
B  同一张 t-SNE（11,439 条记录）按 8 个有意义的变量上色的小图并排：系统类型、蛋白、标签、来源、温度、NaCl、蛋白浓度、缺失块数
输入  data/rs_plot_table.tsv、data/rs_coords.tsv      输出  figs/FigR6_coloring_gallery.{png,pdf,svg}、out/coloring_stats.csv
用法  python rs18_coloring_figure.py --data /content/rs6/data --out /content/rs6/out --figs /content/rs6/figs"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

for f in fm.findSystemFonts():
    if "LiberationSans-" in f:
        fm.fontManager.addfont(f)
FONT = "Liberation Sans" if any(x.name == "Liberation Sans" for x in fm.fontManager.ttflist) else "DejaVu Sans"
INK, MUTE, LINE, GRID = "#1F2933", "#5A6877", "#B9C2CB", "#E6EAEE"
mpl.rcParams.update({"font.family": [FONT, "DejaVu Sans"], "font.size": 6.4, "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.5, "figure.dpi": 100})

# ------------------------------------------------------------------ 配色（每个变量自成一套）
SYS = ["protein only", "protein+RNA", "protein+DNA", "Csat pool"]
SYS_COL = {"protein only": "#018AB2", "protein+RNA": "#964A59", "protein+DNA": "#327C5A", "Csat pool": "#6E5722"}      # 与 explorer 的 FIXED.system 一致
CAT = ["#20B7ED", "#EF7182", "#98C93C", "#A47BDD", "#E7AC24", "#159C98", "#D87234", "#427FDB", "#BE629E", "#35A45B"]
PROT = [("FUS", "FUS"), ("tau (MAPT)", "tau"), ("hnRNPA1 (A1-LCD)", "hnRNPA1"), ("SARS-CoV-2 N", "SARS-CoV-2 N"), ("P00698 (lysozyme C)", "lysozyme C"),
        ("TDP-43", "TDP-43"), ("A0A0G2UMW8", "A0A0G2UMW8"), ("DDX4", "DDX4"), ("MeCP2", "MeCP2"), ("UBQLN2", "UBQLN2")]
LAB_COL = {"phase separation": "#E4572E", "no phase separation": "#2E5EAA", "csat only": "#B9B1C9"}
SRC = [("LLPSDB_v2", "LLPSDB v2 files", "#A63D12", "LLPSDB v2 (download)"), ("LLPSDB_v2_site", "LLPSDB v2 site", "#EE8B2F", "LLPSDB v2 (site)"), ("LLPSDB_v2_phase_diagram", "LLPSDB v2 diagrams", "#F2C14E", "LLPSDB v2 (phase diagrams)"),
       ("LLPSense", "LLPSense", "#BE629E", "LLPSense"), ("RNAPSEC", "RNAPSEC", "#159C98", "RNAPSEC"), ("PhaSepDB3_pilot_LLM", "PhaSepDB 3", "#427FDB", "PhaSepDB 3 (pilot)"),
       ("Bremer2022", "Bremer 2022", "#6B2A6B", "Bremer 2022"), ("Ausserwoger2025", "Ausserwöger", "#A47BDD", "Ausserwöger 2025")]
TEMP = LinearSegmentedColormap.from_list("temp", ["#2B5C9E", "#79A6D2", "#EFE4C8", "#F0A25E", "#C23B2E"])
MISS = LinearSegmentedColormap.from_list("miss", ["#E9ECEF", "#B9C0C8", "#7C8794", "#454F5C", "#151B23"])
FC = LinearSegmentedColormap.from_list("fc", ["#1B4F9C", "#F8F8F8", "#B2182B"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--figs", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.figs, exist_ok=True)
    df = pd.read_csv(f"{a.data}/rs_plot_table.tsv", sep="\t", low_memory=False)
    co = pd.read_csv(f"{a.data}/rs_coords.tsv", sep="\t")
    df = df.merge(co[["rs_idx", "tsne_full_x", "tsne_full_y"]], on="rs_idx", how="left")
    assert df.tsne_full_x.notna().all()
    df["system"] = np.where(df.pool == "csat", "Csat pool", df.na_class.map({"none": "protein only", "RNA": "protein+RNA", "DNA": "protein+DNA"}))
    print(df.system.value_counts().to_dict(), flush=True)
    df["label3"] = np.where(df.pool == "csat", "csat only", df.ps)
    obs = lambda c: (df[c] == "obs")

    # ------------------------------------------------------------------ A 的列：(区块, 标签, 布尔)
    cols = []
    for k, _, _, lab in SRC:
        cols.append(("Source", lab, (df.source == k).to_numpy()))
    for k, lab in (("phase separation", "phase separation"), ("no phase separation", "no phase separation"), ("csat only", "csat value only")):
        cols.append(("Label", lab, (df.label3 == k).to_numpy()))
    for k in ("measured", "room temp. only", "missing"):
        cols.append(("Temperature", {"measured": "T measured", "room temp. only": "room temp. only", "missing": "T missing"}[k], (df.temp_state == k).to_numpy()))
    for lab, m in (("pH given", obs("st_pH")), ("protein conc. given", obs("st_conc")), ("NaCl added", obs("st_NaCl")), ("buffer named", obs("st_buffer")),
                   ("additive listed", obs("st_additive")), ("crowder present", df.crowder.isin(["PEG", "glycerol", "Ficoll / dextran"]))):
        cols.append(("Conditions", lab, m.to_numpy()))
    pn = df.pname.replace({"Q07352 + P35637 (two UniProt IDs)": "ZFP36L1 + FUS"})
    for k, lab in PROT:
        cols.append(("Protein", lab, (pn == k).to_numpy()))
    sysv = df.system.to_numpy()
    R = np.zeros((4, len(cols))); O = np.zeros(len(cols)); rows = []
    for j, (blk, lab, m) in enumerate(cols):
        O[j] = m.mean()
        for i, s in enumerate(SYS):
            sel = sysv == s; R[i, j] = m[sel].mean(); rows.append((s, blk, lab, int(sel.sum()), int(m[sel].sum()), R[i, j], O[j], np.log2(R[i, j] / O[j]) if R[i, j] > 0 else -np.inf))
    pd.DataFrame(rows, columns=["system", "block", "category", "n_system", "n_in_category", "ratio", "overall_ratio", "log2fc"]).to_csv(f"{a.out}/coloring_stats.csv", index=False)

    # ------------------------------------------------------------------ 画布（英寸）
    FW, FH = 7.2, 8.7
    fig = plt.figure(figsize=(FW, FH), facecolor="white")
    def ax_in(l, b, w, h, **kw): return fig.add_axes([l / FW, b / FH, w / FW, h / FH], **kw)
    def tx(x, y, s, **kw): fig.text(x / FW, y / FH, s, **kw)

    # ---------------- A
    tx(0.12, FH - 0.2, "A", fontsize=10, fontweight="bold", color=INK, va="center")
    tx(0.42, FH - 0.2, "What each system type is made of", fontsize=7.6, fontweight="bold", color=INK, va="center")
    L0, W0, rh = 1.5, 4.5, 0.28
    colw = W0 / len(cols)
    top_a = FH - 0.62
    axA = ax_in(L0, top_a - 4 * rh, W0, 4 * rh)
    axA.set_xlim(-0.5, len(cols) - 0.5); axA.set_ylim(3.5, -0.5)
    norm = Normalize(-1.5, 1.5)
    for i in range(4):
        axA.axhspan(i - 0.5, i + 0.5, color="#F6F8FA" if i % 2 == 0 else "white", lw=0, zorder=0)
    for j in range(len(cols)):
        for i in range(4):
            r = R[i, j]
            if r <= 0:
                continue
            fc = float(np.clip(np.log2(r / O[j]), -1.5, 1.5))
            axA.scatter(j, i, s=125 * r ** 0.75, c=[FC(norm(fc))], edgecolors="#6B7782", linewidths=0.25, zorder=3)
    # 区块分隔与标题
    blocks, start = [], 0
    for j in range(1, len(cols) + 1):
        if j == len(cols) or cols[j][0] != cols[start][0]:
            blocks.append((cols[start][0], start, j - 1)); start = j
    for k, (name, j0, j1) in enumerate(blocks):
        if k:
            axA.axvline(j0 - 0.5, color=LINE, lw=0.6, zorder=1)
        xm = (j0 + j1) / 2
        axA.annotate(name, xy=((xm + 0.5) / len(cols), 1.0), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=INK)
    axA.set_xticks(range(len(cols))); axA.set_xticklabels([c[1] for c in cols], rotation=48, ha="right", rotation_mode="anchor", fontsize=5.7, color=INK)
    axA.set_yticks(range(4)); axA.set_yticklabels([]); axA.tick_params(length=0, pad=2)
    for sp in axA.spines.values():
        sp.set_color(LINE)
    n_sys = df.system.value_counts()
    for i, s in enumerate(SYS):
        yy = top_a - (i + 0.5) * rh
        fig.patches.append(plt.Rectangle(((L0 - 1.38) / FW, (yy - 0.055) / FH), 0.11 / FW, 0.11 / FH, transform=fig.transFigure, facecolor=SYS_COL[s], edgecolor="none"))
        tx(L0 - 1.22, yy, f"{s}", fontsize=6.6, color=INK, va="center", fontweight="bold")
        tx(L0 - 0.06, yy, f"{n_sys[s]:,}", fontsize=5.6, color=MUTE, va="center", ha="right")
    # 图例：点大小 + 色条（右侧一列，高度控制在 A 之内）
    lx = L0 + W0 + 0.3
    tx(lx, top_a + 0.02, "Ratio", fontsize=6.6, fontweight="bold", color=INK, va="bottom")
    for k, v in enumerate((0.02, 0.1, 0.3, 0.6, 1.0)):
        yy = top_a - 0.13 - k * 0.19
        fig.add_artist(Line2D([(lx + 0.1) / FW], [yy / FH], marker="o", markersize=(125 * v ** 0.75) ** 0.5, markerfacecolor="#B9C2CB", markeredgecolor="#6B7782", markeredgewidth=0.25, linestyle="none", transform=fig.transFigure))
        tx(lx + 0.3, yy, f"{v:g}", fontsize=5.8, color=INK, va="center")
    tx(lx, top_a - 1.2, "Log₂ enrichment\nover all records", fontsize=5.8, color=INK, va="top")
    cax = ax_in(lx + 0.04, top_a - 2.2, 0.13, 0.72)
    cb = fig.colorbar(ScalarMappable(norm, FC), cax=cax); cb.set_ticks([-1, 0, 1]); cb.set_ticklabels(["−1", "0", "1"]); cb.ax.tick_params(labelsize=5.6, length=2, width=0.4); cb.outline.set_linewidth(0.4)
    tx(lx + 0.105, top_a - 1.45, "> 1.5", fontsize=5.4, color=MUTE, va="center", ha="center"); tx(lx + 0.105, top_a - 2.3, "< −1.5", fontsize=5.4, color=MUTE, va="center", ha="center")

    # ---------------- B
    yB = top_a - 4 * rh - 1.28
    tx(0.12, yB, "B", fontsize=10, fontweight="bold", color=INK, va="center")
    tx(0.42, yB, "One map, eight colourings (t-SNE of 11,439 records)", fontsize=7.6, fontweight="bold", color=INK, va="center")
    x, y = df.tsne_full_x.to_numpy(), df.tsne_full_y.to_numpy()
    rng = np.random.default_rng(0); order = rng.permutation(len(df))
    pw, gap, lm = 1.5, 0.2, 0.42
    panels = []

    def base(ax, title, sub):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(LINE); sp.set_linewidth(0.5)
        ax.set_title(title, fontsize=7.2, fontweight="bold", color=INK, loc="left", pad=11)
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 2.2), textcoords="offset points", fontsize=5.4, color=MUTE, va="bottom")

    def cat_leg(lax, items, ncol=2):
        lax.axis("off")
        h = [Line2D([], [], marker="o", linestyle="none", markersize=3.6, markerfacecolor=c, markeredgecolor="none", label=l) for c, l in items]
        lax.legend(handles=h, loc="upper left", ncol=ncol, frameon=False, handletextpad=0.25, columnspacing=0.7, borderaxespad=0, labelspacing=0.22, fontsize=5.3)

    def cont_leg(lax, cmap, vmin, vmax, ticks, label, tick_labels=None, special=()):
        lax.axis("off")
        cb_ax = lax.inset_axes([0.0, 0.7, 0.9, 0.17])
        cbar = fig.colorbar(ScalarMappable(Normalize(vmin, vmax), cmap), cax=cb_ax, orientation="horizontal")
        cbar.set_ticks(ticks); cbar.set_ticklabels(tick_labels or [f"{t:g}" for t in ticks]); cbar.ax.tick_params(labelsize=5.4, length=1.6, width=0.4, pad=1.2); cbar.outline.set_linewidth(0.4)
        lax.text(0.0, 0.3, label, fontsize=5.6, color=INK, va="center", transform=lax.transAxes)
        for k, (c, l) in enumerate(special):
            lax.add_patch(plt.Rectangle((0.0 + 0.5 * k, 0.0), 0.06, 0.17, color=c, transform=lax.transAxes, clip_on=False)); lax.text(0.08 + 0.5 * k, 0.085, l, fontsize=5.4, color=INK, va="center", transform=lax.transAxes)

    specs = []
    # 1 系统类型
    def p_sys(ax, lax):
        for s in SYS[::-1]:
            m = (df.system == s).to_numpy(); ax.scatter(x[m], y[m], s=1.6, c=SYS_COL[s], linewidths=0, rasterized=True, alpha=0.9)
        cat_leg(lax, [(SYS_COL[s], f"{s}  {n_sys[s]:,}") for s in SYS], 1)
    specs.append(("System type", "protein, nucleic acid, csat pool", p_sys))
    # 2 蛋白
    def p_prot(ax, lax):
        known = pn.isin([k for k, _ in PROT]).to_numpy()
        ax.scatter(x[~known], y[~known], s=1.2, c="#D5DAE0", linewidths=0, rasterized=True)
        for (k, lab), c in zip(PROT[:8], CAT):
            m = (pn == k).to_numpy(); ax.scatter(x[m], y[m], s=1.8, c=c, linewidths=0, rasterized=True, alpha=0.95)
        cat_leg(lax, [(c, lab) for (k, lab), c in zip(PROT[:8], CAT)] + [("#D5DAE0", "other proteins")], 2)
    specs.append(("Protein", "records of one protein sit together", p_prot))
    # 3 标签
    def p_lab(ax, lax):
        for k in ("csat only", "no phase separation", "phase separation"):
            m = (df.label3 == k).to_numpy(); ax.scatter(x[m], y[m], s=1.5, c=LAB_COL[k], linewidths=0, rasterized=True, alpha=0.8)
        cat_leg(lax, [(LAB_COL["phase separation"], f"phase separation  {int((df.label3 == 'phase separation').sum()):,}"), (LAB_COL["no phase separation"], f"no phase separation  {int((df.label3 == 'no phase separation').sum()):,}"),
                      (LAB_COL["csat only"], f"csat value only  {int((df.label3 == 'csat only').sum()):,}")], 1)
    specs.append(("Label", "what the paper reported", p_lab))
    # 4 来源
    def p_src(ax, lax):
        for k, lab, c, _ in SRC:
            m = (df.source == k).to_numpy(); ax.scatter(x[m], y[m], s=1.6, c=c, linewidths=0, rasterized=True, alpha=0.9)
        cat_leg(lax, [(c, lab) for k, lab, c, _ in SRC], 2)
    specs.append(("Source", "which resource the record came from", p_src))
    # 5 温度
    def p_temp(ax, lax):
        tc = pd.to_numeric(df.tempC, errors="coerce").to_numpy(); ok = (df.temp_state == "measured").to_numpy() & ~np.isnan(tc)
        rt = (df.temp_state == "room temp. only").to_numpy(); ms = (df.temp_state == "missing").to_numpy()
        ax.scatter(x[rt], y[rt], s=1.3, c="#CFC8B6", linewidths=0, rasterized=True); ax.scatter(x[ms], y[ms], s=1.6, c="#2B2D42", linewidths=0, rasterized=True)
        o = np.argsort(tc[ok]); ax.scatter(x[ok][o], y[ok][o], s=1.7, c=np.clip(tc[ok][o], 0, 90), cmap=TEMP, vmin=0, vmax=90, linewidths=0, rasterized=True)
        cont_leg(lax, TEMP, 0, 90, [0, 30, 60, 90], "Temperature (°C)", ["0", "30", "60", "≥ 90"], special=[("#CFC8B6", "room temp. only"), ("#2B2D42", "missing")])
    specs.append(("Temperature", "neighbours share the temperature", p_temp))
    # 6 NaCl
    def p_nacl(ax, lax):
        v = pd.to_numeric(df.nacl_log, errors="coerce").to_numpy(); ok = ~np.isnan(v)
        ax.scatter(x[~ok], y[~ok], s=1.4, c="#D5DAE0", linewidths=0, rasterized=True)
        o = np.argsort(v[ok]); ax.scatter(x[ok][o], y[ok][o], s=1.7, c=v[ok][o], cmap="viridis", vmin=0, vmax=3.4, linewidths=0, rasterized=True)
        cont_leg(lax, mpl.colormaps["viridis"], 0, 3.4, [0, 1, 2, 3], "log₁₀(1 + NaCl, mM)", ["0", "1", "2", "3"], special=[("#D5DAE0", "not given")])
    specs.append(("NaCl", "ionic strength of the buffer", p_nacl))
    # 7 蛋白浓度
    def p_conc(ax, lax):
        v = pd.to_numeric(df.conc_v, errors="coerce").to_numpy(); ok = ~np.isnan(v)
        ax.scatter(x[~ok], y[~ok], s=1.4, c="#D5DAE0", linewidths=0, rasterized=True)
        o = np.argsort(-v[ok]); ax.scatter(x[ok][o], y[ok][o], s=1.7, c=np.clip(v[ok][o], -1, 3), cmap="magma_r", vmin=-1, vmax=3, linewidths=0, rasterized=True)
        cont_leg(lax, mpl.colormaps["magma_r"], -1, 3, [-1, 0, 1, 2, 3], "log₁₀ protein (µM)", ["≤ −1", "0", "1", "2", "≥ 3"], special=[("#D5DAE0", "not given")])
    specs.append(("Protein concentration", "dilute to concentrated", p_conc))
    # 8 缺失块数
    def p_miss(ax, lax):
        v = pd.to_numeric(df.n_miss, errors="coerce").to_numpy(); o = np.argsort(v)
        ax.scatter(x[o], y[o], s=1.7, c=v[o], cmap=MISS, vmin=0, vmax=6, linewidths=0, rasterized=True)
        cont_leg(lax, MISS, 0, 6, [0, 2, 4, 6], "Condition blocks not given", ["0", "2", "4", "≥ 6"])
    specs.append(("Missing blocks", "missingness follows the source", p_miss))

    top_b = yB - 0.42
    ph = pw + 0.7
    for k, (title, sub, fn) in enumerate(specs):
        r, c = divmod(k, 4)
        l = lm + c * (pw + gap); b = top_b - r * (ph + 0.48) - pw
        ax = ax_in(l, b, pw, pw); lax = ax_in(l, b - 0.62, pw + 0.12, 0.54)
        base(ax, title, sub); fn(ax, lax)
        cx, cy, rr = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2, max(x.max() - x.min(), y.max() - y.min()) / 2 + 3
        ax.set_xlim(cx - rr, cx + rr); ax.set_ylim(cy - rr, cy + rr)
    tx(lm, 0.12, "Dot = one record. A: dot size = share of the system type's records in the category; colour = log₂(share ÷ share among all records), capped at ±1.5.", fontsize=5.4, color=MUTE, va="center")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{a.figs}/FigR6_coloring_gallery.{ext}", dpi=300 if ext == "png" else None, facecolor="white")
    print("FIG6_FINISHED", flush=True)


if __name__ == "__main__":
    main()
