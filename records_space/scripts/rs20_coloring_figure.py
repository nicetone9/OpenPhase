"""FigR6 v2: what each system type is made of (top) and the records map under eight colourings (below).

Style follows Fig. 1 / Fig. S1: family-coloured tags, soft colour blobs with direct labels, 183 mm wide, English only.
System types use the Fig. 1 family colours (blue = protein(1), red = + RNA, green = + DNA, purple = C_sat).
Every other categorical palette avoids those four hues, so a colour never means two things in one figure.
Run:  scripts/run_fig6_local.sh   (builds a throw-away environment, renders, then runs the nature-figure audits)
"""
import math
from collections import deque

import numpy as np
import pandas as pd
import common_style as cs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex, to_rgb
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import ScaledTranslation

MM = cs.MM
INK, MUTE, WHITE, GREY, LINE = "#242424", "#595959", "#FFFFFF", "#C9CED4", "#B9C2CB"
TAG = "#ECEEF0"

# ---------------------------------------------------------------- system types: Fig. 1 / S1 families (llps_palette_cx.FAM_A)
FAM = {"P": dict(halo="#3FC3FF", dot="#1B9AD6", label="#0F6A82"), "R": dict(halo="#FFC2C4", dot="#E4525C", label="#C42B2B"),
       "D": dict(halo="#D4F3B6", dot="#6BBE3A", label="#4A8222"), "C": dict(halo="#DCCBF2", dot="#7A4AA6", label="#6B2A6B")}
SYS = [("protein(1)", "P", 7490), ("protein(1) + RNA", "R", 2511), ("protein(1) + DNA", "D", 662), ("protein(1) · C_sat", "C", 776)]
SYS_SHORT = ["protein(1)", "+ RNA", "+ DNA", "C_sat"]

# ---------------------------------------------------------------- other palettes: no blue / red / green / purple family hues
PROT = [("FUS", "FUS", "#D6338C"), ("tau (MAPT)", "tau", "#F2B705"), ("hnRNPA1 (A1-LCD)", "hnRNPA1", "#14A098"), ("SARS-CoV-2 N", "SARS-CoV-2 N", "#9C5B2E"),
        ("P00698 (lysozyme C)", "lysozyme C", "#F28C28"), ("TDP-43", "TDP-43", "#4D6075"), ("A0A0G2UMW8", "A0A0G2UMW8", "#9AA31A")]
PROT_GREY = ("DDX4", "MeCP2", "UBQLN2")
LAB = [("phase separation", "#F28C28"), ("no phase separation", "#4D6075"), ("csat only", "#C9CED4")]
# the three LLPSDB v2 forms share one colour in the map (the matrix above splits them)
SRC = [("LLPSDB_v2", "LLPSDB v2 files", "#F28C28"), ("LLPSDB_v2_site", "LLPSDB v2 site", "#F28C28"), ("LLPSDB_v2_phase_diagram", "LLPSDB v2 diagrams", "#F28C28"),
       ("LLPSense", "LLPSense", "#D6338C"), ("RNAPSEC", "RNAPSEC", "#14A098"), ("PhaSepDB3_pilot_LLM", "PhaSepDB 3", "#F2B705"),
       ("Bremer2022", "Bremer 2022", "#9C5B2E"), ("Ausserwoger2025", "Ausserwöger 2025", "#4D6075")]
ENR = LinearSegmentedColormap.from_list("enr", ["#2E3B4E", "#8E9BAA", "#F5F5F5", "#F2C45A", "#E08A00"])
# sequential maps start at a visible tint (a white start would vanish on the white page)
TEMP = LinearSegmentedColormap.from_list("temp", ["#FBE3A6", "#F5A83F", "#D2601A", "#7A2E0B"])
NACL = LinearSegmentedColormap.from_list("nacl", [mpl.colormaps["viridis"](v) for v in np.linspace(0.0, 1.0, 64)])
CONC = LinearSegmentedColormap.from_list("conc", [mpl.colormaps["magma_r"](v) for v in np.linspace(0.10, 1.0, 64)])
MISS = LinearSegmentedColormap.from_list("miss", ["#DDE1E6", "#9AA3AD", "#4B5561", "#151B23"])

FINAL_WIDTH_MM = 183.0          # Nature double-column width
FIG_W = FINAL_WIDTH_MM
PLOT_W, PLOT_H = 41.0, 38.0
COLX = [6 + i * 44 for i in range(4)]
HEAT_TOP, ROW_H = 11.5, 6.0
HEAT_X0, HEAT_X1 = 41.0, 141.0
B_TOP, ROW_STEP, GAP = 58.0, 49.0, 4.4
FIG_H = B_TOP + ROW_STEP + GAP + PLOT_H + 14.5


# ---------------------------------------------------------------- density helpers (same as the Fig. 1 / S1 maps)
def gaussian_density(xy, xedges, yedges, bandwidth):
    d = np.histogram2d(xy[:, 1], xy[:, 0], bins=(yedges, xedges))[0]
    for axis, step in ((0, yedges[1] - yedges[0]), (1, xedges[1] - xedges[0])):
        sigma = bandwidth / step
        off = np.arange(-int(math.ceil(4 * sigma)), int(math.ceil(4 * sigma)) + 1)
        k = np.exp(-0.5 * (off / sigma) ** 2); k /= k.sum()
        d = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis, d)
    return d / d.max() if d.max() > 0 else d


def largest_island(mask, xy, xedges, yedges):
    comp = np.zeros(mask.shape, dtype=np.int32); h, w = mask.shape; c = 0
    for y, x in zip(*np.nonzero(mask)):
        if comp[y, x]:
            continue
        c += 1; comp[y, x] = c; q = deque([(y, x)])
        while q:
            cy, cx = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = cy + dy, cx + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not comp[yy, xx]:
                        comp[yy, xx] = c; q.append((yy, xx))
    ix = np.clip(np.searchsorted(xedges, xy[:, 0], side="right") - 1, 0, w - 1)
    iy = np.clip(np.searchsorted(yedges, xy[:, 1], side="right") - 1, 0, h - 1)
    mem = comp[iy, ix]; cnt = np.bincount(mem, minlength=c + 1); area = np.bincount(comp.ravel(), minlength=c + 1)
    el = [i for i in range(1, c + 1) if cnt[i]]
    if not el:
        return None
    best = min(el, key=lambda v: (-cnt[v], -area[v], v))
    centre = xy[mem == best].mean(axis=0); yy, xx = np.nonzero(comp == best)
    pix = np.column_stack(((xedges[xx] + xedges[xx + 1]) / 2, (yedges[yy] + yedges[yy + 1]) / 2))
    return pix[np.argmin(np.sum((pix - centre) ** 2, axis=1))]


def soft_blob(ax, density, colour, alpha):
    rgba = np.empty((*density.shape, 4)); rgba[:, :, :3] = to_rgb(colour)
    f = np.clip((density - 0.08) / 0.04, 0, 1)
    rgba[:, :, 3] = alpha * f * f * (3 - 2 * f)
    ax.imshow(rgba, origin="lower", extent=(*ax.get_xlim(), *ax.get_ylim()), interpolation="bilinear", zorder=0, aspect="auto")


def darker(c, t=0.55):
    return to_hex(t * np.array(to_rgb(c)) + (1 - t) * np.array(to_rgb("#17212B")))


# ---------------------------------------------------------------- text with a readable C_sat subscript (points, never scaled glyphs)
def parts_of(text, size):
    i = text.find("C_sat")
    if i < 0:
        return [(text, size, 0.0)]
    out = [(text[:i + 1], size, 0.0), ("sat", max(5.1, 0.72 * size), -0.25 * size)]
    if text[i + 5:]:
        out.append((text[i + 5:], size, 0.0))
    return out


def draw_parts(ax, x, y, text, size, color, transform=None, weight="normal", center=False, stroke=None, va="baseline", zorder=6):
    """Draw text piece by piece; returns artists. center=True centres the whole run on x (va centre -> baseline offset)."""
    transform = ax.transData if transform is None else transform
    r = ax.figure.canvas.get_renderer(); arts, off = [], 0.0
    for value, sz, lower in parts_of(text, size):
        a = ax.text(x, y, value, transform=transform + ScaledTranslation(off / 72, lower / 72, ax.figure.dpi_scale_trans), fontsize=sz, color=color,
                    fontweight=weight, ha="left", va=va, zorder=zorder, clip_on=False)
        if stroke:
            a.set_path_effects([pe.withStroke(linewidth=stroke, foreground=WHITE)])
        off += a.get_window_extent(r).width * 72 / ax.figure.dpi + 0.2 + 0.28 * sz * (len(value) - len(value.rstrip(" ")))
        arts.append(a)
    if center:
        total = off - 0.2
        for a, (value, sz, lower) in zip(arts, parts_of(text, size)):
            pass
        shift = -total / 2
        for k, a in enumerate(arts):
            tr = a.get_transform(); a.set_transform(tr + ScaledTranslation(shift / 72, 0, ax.figure.dpi_scale_trans))
    return arts


# ---------------------------------------------------------------- map panels
def prepare_ax(ax, xy):
    low, high = xy.min(0), xy.max(0); c = (low + high) / 2
    bb = ax.get_window_extent(); asp = bb.width / bb.height
    half = max(high[1] - low[1], (high[0] - low[0]) / asp) * 0.56
    ax.set_xlim(c[0] - half * asp, c[0] + half * asp); ax.set_ylim(c[1] - half, c[1] + half)
    cs.style_embedding_ax(ax, "en", arrows=False); ax.set_aspect("equal", adjustable="box"); ax.apply_aspect()
    xe = np.linspace(*ax.get_xlim(), 161); ye = np.linspace(*ax.get_ylim(), int(round(160 / asp)) + 1)
    return xe, ye


def place_labels(ax, items, size=5.4):
    """items: dict(text, xy, color). Label on its island if it fits, else the nearest free spot with a dotted leader."""
    fig = ax.figure; r = fig.canvas.get_renderer(); bb = ax.get_window_extent(); mm = bb.width / PLOT_W
    bounds = bb.padded(-0.5 * mm); boxes, placed, skipped = [], [], []
    inv = ax.transData.inverted()
    for it in items:
        base = np.array(ax.transData.transform(it["xy"])); done = False
        for rad, n, phase in ((0, 1, 0.0), (7, 12, 0.0), (11, 16, 0.2), (15, 20, 0.1), (19, 24, 0.3)):
            for a in np.linspace(0, 2 * np.pi, n, endpoint=False) + phase:
                p = base + rad * mm * np.array([math.cos(a), math.sin(a)]); xy = inv.transform(p)
                arts = draw_parts(ax, xy[0], xy[1], it["text"], size, it["color"], center=True, weight="bold", stroke=0.8, va="center")
                box = None
                for t in arts:
                    b = t.get_window_extent(r)
                    box = b if box is None else type(b).union([box, b])
                box = box.padded(0.35 * mm)
                ok = box.x0 >= bounds.x0 and box.x1 <= bounds.x1 and box.y0 >= bounds.y0 and box.y1 <= bounds.y1 and not any(box.overlaps(o) for o in boxes)
                if ok:
                    boxes.append(box); placed.append(it["text"]); done = True
                    if rad:
                        cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2; d = np.hypot(cx - base[0], cy - base[1]); u = np.array([cx - base[0], cy - base[1]]) / d
                        hw = np.array([(box.x1 - box.x0) / 2 * abs(u[0]) ** 0.0, (box.y1 - box.y0) / 2])
                        end = np.array([cx, cy]) - u * min(0.62 * (box.x1 - box.x0) / 2 / max(abs(u[0]), 1e-6), 0.62 * (box.y1 - box.y0) / 2 / max(abs(u[1]), 1e-6))
                        s_, e_ = inv.transform(base), inv.transform(end)
                        ax.plot([s_[0], e_[0]], [s_[1], e_[1]], color=INK, lw=0.35, ls=(0, (1, 2.2)), zorder=5, solid_capstyle="round")
                        ax.plot([s_[0]], [s_[1]], marker="o", ms=1.8, mfc=it["color"], mec=INK, mew=0.25, zorder=6)
                    break
                for t in arts:
                    t.remove()
            if done:
                break
        if not done:
            skipped.append(it["text"])
    return placed, skipped


def cat_panel(ax, xy, codes, names, dot_cols, blob_cols, label_cols, blob_min=25, blob_alpha=0.40, label_names=None):
    xe, ye = prepare_ax(ax, xy); bw = 0.026 * np.ptp(xe)
    grey = codes < 0
    if grey.any():
        soft_blob(ax, gaussian_density(xy[grey], xe, ye, bw), GREY, 0.22)
    items = []
    for k in range(len(names)):
        own = np.flatnonzero(codes == k)
        if not len(own):
            continue
        dens = gaussian_density(xy[own], xe, ye, bw)
        ba = blob_alpha[k] if hasattr(blob_alpha, "__len__") else blob_alpha
        if len(own) >= blob_min and ba > 0:
            soft_blob(ax, dens, blob_cols[k], ba)
        loc = largest_island(dens >= 0.09, xy[own], xe, ye)
        if loc is not None:
            items.append(dict(text=(label_names or names)[k], xy=loc, color=label_cols[k], n=len(own)))
    if grey.any():
        ax.scatter(xy[grey, 0], xy[grey, 1], s=1.7 ** 2, c=GREY, alpha=0.85, edgecolors=WHITE, linewidths=0.15, rasterized=True, zorder=2)
    for k in sorted(range(len(names)), key=lambda v: -(codes == v).sum()):
        m = codes == k
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=1.9 ** 2, c=dot_cols[k], alpha=0.95, edgecolors=WHITE, linewidths=0.15, rasterized=True, zorder=3)
    items.sort(key=lambda d: -d["n"])
    return place_labels(ax, items)


def cont_panel(ax, xy, values, cmap, vmin, vmax, special):
    """values: float array (nan = special); special: list of (mask, colour)."""
    prepare_ax(ax, xy)
    for m, c in special:
        ax.scatter(xy[m, 0], xy[m, 1], s=1.7 ** 2, c=c, edgecolors=WHITE, linewidths=0.12, rasterized=True, zorder=2)
    ok = ~np.isnan(values); o = np.argsort(values[ok])
    ax.scatter(xy[ok][o, 0], xy[ok][o, 1], s=1.9 ** 2, c=np.clip(values[ok][o], vmin, vmax), cmap=cmap, vmin=vmin, vmax=vmax, edgecolors=WHITE, linewidths=0.12, rasterized=True, zorder=3)


def main():
    args = cs.parse_args(); cs.setup_style("en")
    # publication-safe text: Arial-metric font, editable text in PDF (Type 42) and SVG
    mpl.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "svg.fonttype": "none", "pdf.fonttype": 42})
    df = pd.read_csv(cs.DATA / "rs_plot_table.tsv", sep="\t", low_memory=False)
    co = pd.read_csv(cs.DATA / "rs_coords.tsv", sep="\t")
    df = df.merge(co[["rs_idx", "tsne_full_x", "tsne_full_y"]], on="rs_idx", how="left")
    assert df.tsne_full_x.notna().all() and len(df) == 11439
    xy = df[["tsne_full_x", "tsne_full_y"]].to_numpy()
    sysc = np.where(df.pool == "csat", 3, df.na_class.map({"none": 0, "RNA": 1, "DNA": 2}).fillna(0)).astype(int)
    assert [int((sysc == i).sum()) for i in range(4)] == [s[2] for s in SYS], "system-type counts differ from the repository tables"
    label3 = np.where(df.pool == "csat", "csat only", df.ps)
    obs = lambda c: (df[c] == "obs").to_numpy()
    pn = df.pname.replace({"Q07352 + P35637 (two UniProt IDs)": "ZFP36L1 + FUS"})

    # ------------------------------------------------------------ panel a columns: (block, label, boolean mask)
    SHORT = {"LLPSDB v2 files": "files", "LLPSDB v2 site": "site", "LLPSDB v2 diagrams": "diagrams"}
    cols = [("Source", SHORT.get(lab, lab), (df.source == k).to_numpy()) for k, lab, _ in SRC]
    cols += [("Label", "phase sep.", (label3 == "phase separation")), ("Label", "no phase sep.", (label3 == "no phase separation")), ("Label", "quantitative", (label3 == "csat only"))]
    cols += [("Temperature", lab, (df.temp_state == k).to_numpy()) for k, lab in (("measured", "T measured"), ("room temp. only", "RT only"), ("missing", "T missing"))]
    cols += [("Conditions", lab, m) for lab, m in (("pH given", obs("st_pH")), ("conc. given", obs("st_conc")), ("NaCl added", obs("st_NaCl")), ("buffer named", obs("st_buffer")),
                                                  ("additive", obs("st_additive")), ("crowder", df.crowder.isin(["PEG", "glycerol", "Ficoll / dextran"]).to_numpy()))]
    for k, lab in (("FUS", "FUS"), ("tau (MAPT)", "tau"), ("hnRNPA1 (A1-LCD)", "hnRNPA1"), ("SARS-CoV-2 N", "SARS-CoV-2 N"), ("P00698 (lysozyme C)", "lysozyme C"),
                   ("TDP-43", "TDP-43"), ("A0A0G2UMW8", "A0A0G2UMW8"), ("DDX4", "DDX4"), ("MeCP2", "MeCP2"), ("UBQLN2", "UBQLN2")):
        cols.append(("Protein", lab, (pn == k).to_numpy()))
    nC = len(cols); R = np.zeros((4, nC)); O = np.zeros(nC); rows = []
    for j, (blk, lab, m) in enumerate(cols):
        m = np.asarray(m, dtype=bool); O[j] = m.mean()
        for i in range(4):
            sel = sysc == i; R[i, j] = m[sel].mean()
            rows.append((SYS[i][0], blk, lab, int(sel.sum()), int(m[sel].sum()), R[i, j], O[j], np.log2(max(R[i, j], 1e-12) / max(O[j], 1e-12)) if R[i, j] > 0 else np.nan))   # zero share stays blank
    pd.DataFrame(rows, columns=["system", "block", "category", "n_system", "n_in_category", "share", "share_all_records", "log2_enrichment"]).to_csv(f"{args.out}/FigR6_v2.source_data_a.csv", index=False)

    # ------------------------------------------------------------ canvas
    fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM), facecolor=WHITE)
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, FIG_W); ov.set_ylim(FIG_H, 0); ov.axis("off"); ov.set_zorder(5); ov.patch.set_alpha(0)

    def mm_axes(x, top, w, h):
        return fig.add_axes([x / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H])

    def pill(x, y, w, h, fc):
        ov.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.1", fc=fc, ec="none", clip_on=False))

    def txt(x, y, s, size=5.5, color=INK, weight="normal", ha="left", va="center", **kw):
        return ov.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, clip_on=False, **kw)

    # ---------------- a: heat/dot matrix
    cw = (HEAT_X1 - HEAT_X0) / nC; norm = Normalize(-1.5, 1.5)
    ov.add_patch(Rectangle((HEAT_X0, HEAT_TOP), HEAT_X1 - HEAT_X0, 4 * ROW_H, fc="#F7F8FA", ec=LINE, lw=0.4, zorder=0))
    for i in (1, 3):
        ov.add_patch(Rectangle((HEAT_X0, HEAT_TOP + i * ROW_H), HEAT_X1 - HEAT_X0, ROW_H, fc=WHITE, ec="none", zorder=0.2))
    blocks, st = [], 0
    for j in range(1, nC + 1):
        if j == nC or cols[j][0] != cols[st][0]:
            blocks.append((cols[st][0], st, j - 1)); st = j
    for b, (name, j0, j1) in enumerate(blocks):
        if b:
            ov.plot([HEAT_X0 + j0 * cw] * 2, [HEAT_TOP, HEAT_TOP + 4 * ROW_H], color=LINE, lw=0.5, zorder=1)
        txt(HEAT_X0 + (j0 + j1 + 1) / 2 * cw, HEAT_TOP - 6.2, name, 5.8, INK, "bold", ha="center")
        if name == "Source":
            bx0, bx1 = HEAT_X0 + (j0 + 0.12) * cw, HEAT_X0 + (j0 + 2.88) * cw
            ov.plot([bx0, bx1], [HEAT_TOP - 0.9] * 2, color=MUTE, lw=0.5, solid_capstyle="butt", zorder=3)
            txt((bx0 + bx1) / 2, HEAT_TOP - 2.7, "LLPSDB v2", 5.2, INK, ha="center")
    DMAX = 3.5
    for j in range(nC):
        for i in range(4):
            r = R[i, j]
            if r <= 0:
                continue
            fc = float(np.clip(np.log2(max(r, 1e-12) / max(O[j], 1e-12)), -1.5, 1.5)); d = DMAX * r ** 0.375 + 0.35
            ov.scatter([HEAT_X0 + (j + 0.5) * cw], [HEAT_TOP + (i + 0.5) * ROW_H], s=(d * 72 / 25.4) ** 2, c=[ENR(norm(fc))], edgecolors="#6B7782", linewidths=0.25, zorder=3)
    mark = {SHORT.get(lab, lab): c for _, lab, c in SRC}; mark.update({"phase sep.": LAB[0][1], "no phase sep.": LAB[1][1], "quantitative": LAB[2][1]})
    mark.update({p_[1]: p_[2] for p_ in PROT}); mark.update({k: "#9AA3AD" for k in PROT_GREY}); mark["MeCP2"] = "#9AA3AD"
    for j, (blk, lab, _) in enumerate(cols):
        cx = HEAT_X0 + (j + 0.5) * cw
        if lab in mark and blk in ("Source", "Label", "Protein"):
            ov.scatter([cx], [HEAT_TOP + 4 * ROW_H + 2.0], s=(1.7 * 72 / 25.4) ** 2, c=mark[lab], edgecolors="none", zorder=3)
        txt(cx, HEAT_TOP + 4 * ROW_H + 3.4, lab, 5.2, INK, ha="right", va="center", rotation=90, rotation_mode="anchor")
    for i, (name, f, n) in enumerate(SYS):
        y0 = HEAT_TOP + i * ROW_H + 0.6
        pill(2.0, y0, HEAT_X0 - 5.0, ROW_H - 1.2, FAM[f]["halo"])
        ov_ax = ov
        # name left, count right (both inside the pill)
        arts = draw_parts(ov, 3.6, HEAT_TOP + (i + 0.5) * ROW_H + 0.1, name, 5.6, INK, transform=ov.transData, weight="bold", va="center")
        txt(HEAT_X0 - 4.0, HEAT_TOP + (i + 0.5) * ROW_H, f"{n:,}", 5.4, INK, ha="right")

    # key: dot size + enrichment colour bar (right of the matrix)
    kx = HEAT_X1 + 5.0
    txt(kx, HEAT_TOP - 3.4, "Share of records", 5.4, INK, "bold")
    for k, v in enumerate((0.05, 0.25, 1.0)):
        cx = kx + 3.0 + k * 11.0; d = DMAX * v ** 0.375 + 0.35
        ov.scatter([cx], [HEAT_TOP + 2.2], s=(d * 72 / 25.4) ** 2, c="#D5DBE1", edgecolors="#6B7782", linewidths=0.25, zorder=3)
        txt(cx + 2.4, HEAT_TOP + 2.2, f"{int(v * 100)}%", 5.2, INK)
    txt(kx, HEAT_TOP + 9.4, "Enrichment", 5.4, INK, "bold")
    cax = mm_axes(kx, HEAT_TOP + 11.8, 30.0, 2.0); grad = np.linspace(-1.5, 1.5, 256)[None, :]
    cax.imshow(grad, cmap=ENR, norm=norm, aspect="auto", extent=(-1.5, 1.5, 0, 1)); cax.set_yticks([]); cax.set_xticks([-1.5, 0, 1.5]); cax.set_xticklabels(["−1.5", "0", "1.5"])
    cax.tick_params(axis="x", labelsize=5.2, length=1.5, width=0.4, pad=1.0)
    for s_ in cax.spines.values():
        s_.set_linewidth(0.4)
    txt(kx, HEAT_TOP + 19.6, "log₂ (share ÷ share overall)", 5.2, INK)

    # ---------------- b: eight colourings of one map
    map_axes, pids = [], []
    specs = ["System type", "Protein", "Label", "Source", "Temperature (°C)", "NaCl, log₁₀(1 + mM)", "Protein, log₁₀ µM", "Condition blocks not given"]
    for k, title in enumerate(specs):
        r, c = divmod(k, 4); top = B_TOP + r * ROW_STEP
        pill(COLX[c], top - 1.0, PLOT_W, 5.0, TAG)
        txt(COLX[c] + 1.8, top + 1.5, title, 5.6, INK, "bold")
        ax = mm_axes(COLX[c], top + GAP, PLOT_W, PLOT_H); map_axes.append(ax); pids.append("abcdefgh"[k])
        below = top + GAP + PLOT_H + 1.6
        corner = top + GAP + PLOT_H - 1.2            # baseline of the last key row inside the panel (bottom-left corner)
        if k == 0:
            names = [s[0] for s in SYS]; fc = [FAM[s[1]] for s in SYS]
            placed, skipped = cat_panel(ax, xy, sysc, names, [f["dot"] for f in fc], [f["halo"] for f in fc], [f["label"] for f in fc], label_names=SYS_SHORT, blob_alpha=[0.0, 0.42, 0.50, 0.50])
        elif k == 1:
            codes = np.full(len(df), -1)
            for q, (key, lab, col) in enumerate(PROT):
                codes[(pn == key).to_numpy()] = q
            names = [p[1] for p in PROT]; cols_p = [p[2] for p in PROT]
            placed, skipped = cat_panel(ax, xy, codes, names, cols_p, cols_p, [darker(c, 0.6) for c in cols_p], blob_min=12, blob_alpha=0.30)
        elif k == 2:
            prepare_ax(ax, xy)
            lc = {key: col for key, col in LAB}; order = np.random.default_rng(3).permutation(len(df));   # draw order only (no simulated data): neither class hides the other
            order = order[np.argsort(label3[order] != "csat only", kind="stable")]
            ax.scatter(xy[order, 0], xy[order, 1], s=1.8 ** 2, c=[lc[v] for v in label3[order]], alpha=0.9, edgecolors=WHITE, linewidths=0.12, rasterized=True, zorder=3)
            xx = COLX[c] + 0.8
            for key, col, dx in (("phase sep.", LAB[0][1], 0.0), ("no phase sep.", LAB[1][1], 12.6), ("C_sat only", LAB[2][1], 27.6)):
                ov.scatter([xx + dx + 1.0], [below + 1.6], s=2.2 ** 2, c=col, edgecolors="none", zorder=3)
                draw_parts(ov, xx + dx + 2.8, below + 1.6, key, 5.2, INK, transform=ov.transData, va="center")
        elif k == 3:
            codes = np.full(len(df), -1)
            for q, (key, lab, col) in enumerate(SRC):
                codes[(df.source == key).to_numpy()] = q
            codes = np.where(codes <= 2, np.where(codes >= 0, 0, -1), codes - 2)
            snames = ["LLPSDB v2"] + [s[1] for s in SRC[3:]]; scols = [SRC[0][2]] + [s[2] for s in SRC[3:]]
            placed, skipped = cat_panel(ax, xy, codes, snames, scols, scols, [darker(c, 0.5) for c in scols], blob_min=40, blob_alpha=[0.0, 0.34, 0.34, 0.34, 0.34, 0.34])
        else:
            if k == 4:
                tc = pd.to_numeric(df.tempC, errors="coerce").to_numpy(); meas = (df.temp_state == "measured").to_numpy() & ~np.isnan(tc)
                val = np.where(meas, tc, np.nan); cmap, v0, v1, ticks, tl = TEMP, 0, 90, [0, 30, 60, 90], ["0", "30", "60", "≥ 90"]
                special = [((df.temp_state == "room temp. only").to_numpy(), "#D9D2C0"), ((df.temp_state == "missing").to_numpy(), "#2B2D42")]; keys = [("#D9D2C0", "RT only"), ("#2B2D42", "missing")]
            elif k == 5:
                val = pd.to_numeric(df.nacl_log, errors="coerce").to_numpy(); cmap, v0, v1, ticks, tl = NACL, 0, 3.4, [0, 1, 2, 3], ["0", "1", "2", "3"]
                special = [(np.isnan(val), "#D5DAE0")]; keys = [("#D5DAE0", "not given")]
            elif k == 6:
                val = pd.to_numeric(df.conc_v, errors="coerce").to_numpy(); cmap, v0, v1, ticks, tl = CONC, -1, 3, [-1, 0, 1, 2, 3], ["≤ −1", "0", "1", "2", "≥ 3"]
                special = [(np.isnan(val), "#D5DAE0")]; keys = [("#D5DAE0", "not given")]
            else:
                val = pd.to_numeric(df.n_miss, errors="coerce").to_numpy(); cmap, v0, v1, ticks, tl = MISS, 0, 6, [0, 2, 4, 6], ["0", "2", "4", "≥ 6"]
                special = []; keys = []
            cont_panel(ax, xy, val, cmap, v0, v1, special)
            cb = mm_axes(COLX[c] + 0.5, below, 30.0, 1.8); cb.imshow(np.linspace(v0, v1, 256)[None, :], cmap=cmap, aspect="auto", extent=(v0, v1, 0, 1))
            cb.set_yticks([]); cb.set_xticks(ticks); cb.set_xticklabels(tl); cb.tick_params(axis="x", labelsize=5.2, length=1.5, width=0.4, pad=1.0)
            for s_ in cb.spines.values():
                s_.set_linewidth(0.4)
            for q, (col, nm) in enumerate(keys):
                kx2 = COLX[c] + 0.5 + q * 17.0; yy = below + 8.4
                ov.add_patch(Rectangle((kx2, yy - 1.1), 2.2, 2.2, fc=col, ec="none", zorder=3)); txt(kx2 + 3.0, yy, nm, 5.2, INK)
        if k in (0, 1, 3):
            print(f"panel {k}: labels placed {placed}; skipped {skipped}", flush=True)

    pd.DataFrame({"record_id": df.record_id, "tsne_x": xy[:, 0], "tsne_y": xy[:, 1], "system_type": [SYS[i][0] for i in sysc], "protein": pn, "label": label3, "source": df.source,
                  "temp_state": df.temp_state, "temp_C": df.tempC, "nacl_log10_1p_mM": df.nacl_log, "protein_conc_log10_uM": df.conc_v, "condition_blocks_not_given": df.n_miss}).to_csv(f"{args.out}/FigR6_v2.source_data_b.csv", index=False)
    base = f"{args.out}/FigR6_v2"
    print(f"Layout: {FIG_W:g} x {FIG_H:g} mm", flush=True)
    # render-time panel-alignment gate (1.5 pt); raises when a comparable panel edge is off
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig, json_out=f"{base}.alignment.json", overlay_svg=f"{base}.alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                       require_panel_labels=False, strict=True, axes=map_axes, panel_ids=pids, row_groups=[pids[:4], pids[4:]])
    fig.savefig(f"{base}.pdf"); fig.savefig(f"{base}.svg"); fig.savefig(f"{base}.png", dpi=300)
    fig.savefig(f"{base}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("Export:", base, flush=True)


if __name__ == "__main__":
    main()
