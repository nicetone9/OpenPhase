"""FigR6 v2: what each system type is made of (top) and the records map under eight colourings (below).

Style follows Fig. 1 / Fig. S1: family-coloured tags, soft colour blobs with direct labels, 183 mm wide, English only.
System types use the Fig. 1 family colours (blue = protein(1), red = + RNA, green = + DNA, purple = C_sat).
Every other categorical palette avoids those four hues, so a colour never means two things in one figure.
Run:  scripts/run_fig6_local.sh   (builds a throw-away environment, renders, then runs the nature-figure audits)
"""
import math
import re
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
ENR = LinearSegmentedColormap.from_list("enr", ["#E6E9ED", "#BCC3CC", "#7A8693", "#3D4857", "#171F2A"])   # light = fewer than expected, dark = more than expected (colours are kept for identities)
# sequential maps start at a visible tint (a white start would vanish on the white page)
# continuous variables: one hue per ramp (orange, green, purple, grey), from a light tint that stays visible on white to a deep shade; same lightness progression so that they sit together
TEMP = LinearSegmentedColormap.from_list("temp", ["#FDD0A2", "#FDAE6B", "#F16913", "#D94801", "#8C2D04"])
NACL = LinearSegmentedColormap.from_list("nacl", ["#CCECE6", "#99D8C9", "#41AE76", "#238B45", "#005824"])
CONC = LinearSegmentedColormap.from_list("conc", ["#DADAEB", "#BCBDDC", "#807DBA", "#6A51A3", "#4A1486"])
MISS = LinearSegmentedColormap.from_list("miss", ["#E3E3E3", "#BDBDBD", "#969696", "#636363", "#252525"])

FINAL_WIDTH_MM = 183.0          # Nature double-column width
FIG_W = FINAL_WIDTH_MM
PLOT_W, PLOT_H = 41.0, 38.0
COLX = [6 + i * 44 for i in range(4)]
# --- panel a: one row per system type (all 16 of the colour table); columns = label bar, source dots, protein dots
# the system colour table: (Fill, Point) -- Fill for tags and bands, Point for dots
PAL = {"protein(1)": ("#3FC3FF", "#1799D2"), "protein(2)": ("#80ACED", "#4474B7"), "protein(3)": ("#68D5D6", "#19858E"), "protein(\u22654)": ("#9BB6D1", "#4C637E"),
       "RNA only": ("#E9A3A9", "#A43851"), "protein(1) + RNA": ("#FFC2C4", "#E45265"), "protein(2) + RNA": ("#F4A48C", "#BA553B"), "protein(3) + RNA": ("#F4A6CD", "#A84076"), "protein(\u22654) + RNA": ("#EBC0AC", "#91432F"),
       "DNA only": ("#A6CBA0", "#27613E"), "protein(1) + DNA": ("#D4F3B6", "#68A92C"), "protein(2) + DNA": ("#A7D9AD", "#39824E"), "protein(3) + DNA": ("#95D4BE", "#217660"), "protein(\u22654) + DNA": ("#D4DDA0", "#657626"),
       "Other": ("#E8E8E8", "#666666"), "C_sat pool": ("#DCCBF2", "#8B56BB")}
ORDER = list(PAL)
GAP_AFTER = {3, 8, 13, 14}                               # a little extra space between protein-only | RNA | DNA | Other | C_sat pool
SINGLE = {0: "protein(1)", 1: "protein(1) + RNA", 2: "protein(1) + DNA", 3: "C_sat pool"}
SOURCES = [("LLPSDB v2", ("LLPSDB_v2", "LLPSDB_v2_site", "LLPSDB_v2_phase_diagram")), ("LLPSense", ("LLPSense",)), ("RNAPSEC", ("RNAPSEC",)), ("PhaSepDB 3", ("PhaSepDB3_pilot_LLM",)),
           ("Bremer 2022", ("Bremer2022",)), ("Ausserw\u00f6ger 2025", ("Ausserwoger2025",)), ("Leurs 2025", ("Leurs2025",)), ("Pei 2026", ("Pei2026",))]
SRC_MARK = ["#F28C28", "#D6338C", "#14A098", "#F2B705", "#9C5B2E", "#4D6075", "#B9C0C8", "#B9C0C8"]            # the colours used in the Source map below; grey = not drawn there
PROTS = [("FUS", {"fus"}, {"P35637"}), ("TDP-43", {"tdp-43", "tardbp"}, {"Q13148"}), ("tau", {"tau", "mapt", "tau (mapt)"}, {"P10636"}), ("hnRNPA1", {"hnrnpa1", "hnrnpa1 (a1-lcd)"}, {"P09651"}),
         ("SARS-CoV-2 N", {"sars-cov-2 n"}, {"P0DTC9"}), ("MeCP2", {"mecp2"}, {"P51608"}), ("NPM1", {"npm1"}, {"P06748"}), ("HNRNPK", {"hnrnpk"}, {"P61978"}), ("G3BP1", {"g3bp1"}, {"Q13283"})]
PROT_MARK = ["#D6338C", "#4D6075", "#F2B705", "#14A098", "#9C5B2E", "#B9C0C8", "#B9C0C8", "#B9C0C8", "#B9C0C8"]  # the colours used in the Protein map below
PITCH, ROW_H, ROW_GAP, ROWS_TOP = 4.9, 4.6, 1.6, 11.5
PILL_X, PILL_W, NUM_X1, NUM_X2, BAR_X0, BAR_W, SRC_X0 = 3.0, 34.0, 47.0, 58.5, 63.0, 25.0, 93.0
PROT_X0 = SRC_X0 + 8 * PITCH + 3.0
HEAT_BOTTOM = ROWS_TOP + len(ORDER) * ROW_H + len(GAP_AFTER) * ROW_GAP
B_TOP, ROW_STEP, GAP = HEAT_BOTTOM + 25.0, 49.0, 4.4
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
    """values: float array (nan = special); special: list of (mask, face, edge); face None = hollow marker."""
    prepare_ax(ax, xy)
    for m, face, edge in special:
        if face is None:
            ax.scatter(xy[m, 0], xy[m, 1], s=1.7 ** 2, facecolors="none", edgecolors=edge, linewidths=0.35, rasterized=True, zorder=2)
        else:
            ax.scatter(xy[m, 0], xy[m, 1], s=1.7 ** 2, c=face, edgecolors=edge, linewidths=0.12, rasterized=True, zorder=2)
    ok = ~np.isnan(values); o = np.argsort(values[ok])
    ax.scatter(xy[ok][o, 0], xy[ok][o, 1], s=1.9 ** 2, c=np.clip(values[ok][o], vmin, vmax), cmap=cmap, vmin=vmin, vmax=vmax, edgecolors=WHITE, linewidths=0.12, rasterized=True, zorder=3)


def row_y(i):
    return ROWS_TOP + i * ROW_H + ROW_GAP * sum(1 for g in GAP_AFTER if g < i)


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
    pn = df.pname.replace({"Q07352 + P35637 (two UniProt IDs)": "ZFP36L1 + FUS"})

    # ------------------------------------------------------------ panel a: statistics for all system types
    def tok_hits(tokens):
        low = [t.strip().lower() for t in tokens]; acc_ = [re.sub(r"-\d+$", "", t.strip()).upper() for t in tokens]
        return np.array([any(l in names for l in low) or any(a in accs for a in acc_) for _, names, accs in PROTS])

    def fine_type(raw):
        """v4 raw system type -> a row of the colour table."""
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
    stats = {}
    pair = {}                                                       # single-protein records: hits for every (name, accession) pair once
    for a, b in set(zip(df.pname.fillna(""), df.uniprot.fillna(""))):
        pair[(a, b)] = tok_hits([a, b])
    hit = np.array([pair[(a, b)] for a, b in zip(df.pname.fillna(""), df.uniprot.fillna(""))])
    for i, name in SINGLE.items():
        sel = sysc == i
        stats[name] = dict(systems=int(df.system_id[sel].nunique()), records=int(sel.sum()), pos=int((label3[sel] == "phase separation").sum()), neg=int((label3[sel] == "no phase separation").sum()),
                           src=np.array([df.source[sel].isin(names).mean() for _, names in SOURCES]), prot=hit[sel].mean(axis=0))
    pts = pd.read_csv(cs.DATA / "fsys_points.csv"); meta = pd.read_csv(cs.DATA / "systems_multi.tsv", sep="\t", keep_default_na=False).set_index("sys_id")
    extra = pd.read_csv(cs.DATA / "systems_extra.tsv", sep="\t", keep_default_na=False)
    acc = {n: dict(systems=0, records=0, pos=0, neg=0, src=np.zeros(len(SOURCES)), prot=np.zeros(len(PROTS))) for n in ORDER if n not in SINGLE.values()}

    def add(name, n, pos, neg, ss, comps):
        a = acc[name]; a["systems"] += 1; a["records"] += n; a["pos"] += pos; a["neg"] += neg
        groups = [k for k, (_, names) in enumerate(SOURCES) if any(s in names for s in ss)]
        for k in groups:                                             # a system with several source groups splits its records evenly (laboratory notebooks have no column)
            a["src"][k] += n / len(groups)
        if comps:
            a["prot"] += n * tok_hits(comps)
    for _, r in pts.iterrows():
        add(fine_type(meta.loc[r.sys_id, "raw_type"]), int(r.n_records), int(r.n_pos), int(r.n_neg), [s_ for s_ in meta.loc[r.sys_id, "sources"].split(";") if s_], r.components.split(" + "))
    for _, r in extra.iterrows():                                    # systems without a protein sequence (RNA only, DNA only, no component): not on the Fig. S1 maps
        add(fine_type(r.raw_type), int(r.n_records), int(r.n_pos), int(r.n_neg), [s_ for s_ in r.sources.split(";") if s_], [])
    for n, a in acc.items():
        stats[n] = dict(systems=a["systems"], records=a["records"], pos=a["pos"], neg=a["neg"], src=a["src"] / a["records"], prot=a["prot"] / a["records"])
    rec = np.array([stats[n]["records"] for n in ORDER], dtype=float)
    over_src = sum(rec[k] * stats[n]["src"] for k, n in enumerate(ORDER)) / rec.sum()
    over_prot = sum(rec[k] * stats[n]["prot"] for k, n in enumerate(ORDER)) / rec.sum()
    lg = lambda s_, o: float(np.log2(max(s_, 1e-12) / max(o, 1e-12)))
    recs = []
    for n in ORDER:
        st = stats[n]; none = st["records"] - st["pos"] - st["neg"]
        for blk, labs, vals, over in (("Source", [s_[0] for s_ in SOURCES], st["src"], over_src), ("Protein", [p_[0] for p_ in PROTS], st["prot"], over_prot)):
            for lab, v, o in zip(labs, vals, over):
                recs.append((n, st["systems"], st["records"], st["pos"], st["neg"], none, blk, lab, v, o, lg(v, o) if v > 0 else np.nan))   # zero share stays blank
    pd.DataFrame(recs, columns=["system", "systems", "records", "positive", "negative", "no_binary_label", "block", "category", "share", "share_all_records", "log2_enrichment"]).to_csv(f"{args.out}/FigR6.source_data_a.csv", index=False)

    # ------------------------------------------------------------ canvas
    fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM), facecolor=WHITE)
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, FIG_W); ov.set_ylim(FIG_H, 0); ov.axis("off"); ov.set_zorder(5); ov.patch.set_alpha(0)

    def mm_axes(x, top, w, h):
        return fig.add_axes([x / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H])

    def pill(x, y, w, h, fc):
        ov.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.1", fc=fc, ec="none", clip_on=False))

    def txt(x, y, s, size=5.5, color=INK, weight="normal", ha="left", va="center", **kw):
        return ov.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, clip_on=False, **kw)

    # ---------------- a: all system types
    norm = Normalize(-1.5, 1.5); DMAX = 3.3; X_END = PROT_X0 + len(PROTS) * PITCH
    for i, name in enumerate(ORDER):                                # a light band in the row's own fill colour ties the dots to the tag
        ov.add_patch(Rectangle((PILL_X + PILL_W + 1.2, row_y(i)), X_END - (PILL_X + PILL_W + 1.2) + 1.0, ROW_H, fc=PAL[name][0], alpha=0.20, ec="none", zorder=0))
    for x in (BAR_X0 - 2.2, SRC_X0 - 1.5, PROT_X0 - 1.5):
        ov.plot([x, x], [row_y(0), HEAT_BOTTOM], color=LINE, lw=0.5, zorder=1)
    txt(NUM_X1, ROWS_TOP - 3.4, "systems", 5.2, INK, ha="right"); txt(NUM_X2, ROWS_TOP - 3.4, "records", 5.2, INK, ha="right")
    txt(BAR_X0 + BAR_W / 2, ROWS_TOP - 3.4, "Label", 5.8, INK, "bold", ha="center")
    txt(SRC_X0 + 4 * PITCH, ROWS_TOP - 3.4, "Source", 5.8, INK, "bold", ha="center"); txt(PROT_X0 + len(PROTS) / 2 * PITCH, ROWS_TOP - 3.4, "Protein", 5.8, INK, "bold", ha="center")
    LABC = (LAB[0][1], LAB[1][1], LAB[2][1])
    for i, name in enumerate(ORDER):
        st = stats[name]; ry = row_y(i); cy = ry + ROW_H / 2
        pill(PILL_X, ry + 0.4, PILL_W, ROW_H - 0.8, PAL[name][0])
        ov.scatter([PILL_X + 2.3], [cy], s=(2.2 * 72 / 25.4) ** 2, c=PAL[name][1], edgecolors="none", zorder=4)          # the system's point colour, as in the maps
        draw_parts(ov, PILL_X + 4.7, cy, name, 5.4, INK, transform=ov.transData, weight="bold", va="center")
        txt(NUM_X1, cy, f"{st['systems']:,}", 5.4, INK, ha="right"); txt(NUM_X2, cy, f"{st['records']:,}", 5.4, INK, ha="right")
        tot = st["records"]; parts = (st["pos"], st["neg"], tot - st["pos"] - st["neg"]); x = BAR_X0
        for v, c in zip(parts, LABC):                               # label split: phase separation / no phase separation / no binary label
            w = BAR_W * v / tot
            if w > 0:
                ov.add_patch(Rectangle((x, cy - 1.2), w, 2.4, fc=c, ec="none", zorder=3)); x += w
        for blk, x0, vals, over in (("s", SRC_X0, st["src"], over_src), ("p", PROT_X0, st["prot"], over_prot)):
            for j, (v, o) in enumerate(zip(vals, over)):
                if v <= 0:
                    continue
                d = DMAX * v ** 0.375 + 0.3
                ov.scatter([x0 + (j + 0.5) * PITCH], [cy], s=(d * 72 / 25.4) ** 2, c=[ENR(norm(float(np.clip(lg(v, o), -1.5, 1.5))))], edgecolors="#6B7683", linewidths=0.3, zorder=3)
    # column labels, each with the colour it has in the maps below
    for blk, x0, labs, marks in (("s", SRC_X0, [s_[0] for s_ in SOURCES], SRC_MARK), ("p", PROT_X0, [p_[0] for p_ in PROTS], PROT_MARK)):
        for j, (lab, c) in enumerate(zip(labs, marks)):
            cx = x0 + (j + 0.5) * PITCH
            ov.scatter([cx], [HEAT_BOTTOM + 2.2], s=(1.7 * 72 / 25.4) ** 2, c=c, edgecolors="none", zorder=3)
            txt(cx, HEAT_BOTTOM + 3.9, lab, 5.2, INK, ha="right", va="center", rotation=90, rotation_mode="anchor")
    # keys sit in the free corner under the counts and the label bars
    ky = HEAT_BOTTOM + 5.0
    txt(PILL_X, ky, "Share of records", 5.4, INK, "bold")
    for k, v in enumerate((0.05, 0.25, 1.0)):
        cx = 25.0 + k * 11.5; d = DMAX * v ** 0.375 + 0.3
        ov.scatter([cx], [ky], s=(d * 72 / 25.4) ** 2, c="#D5DBE1", edgecolors="#6B7683", linewidths=0.3, zorder=3)
        txt(cx + 2.5, ky, f"{int(v * 100)}%", 5.2, INK)
    txt(PILL_X, ky + 8.0, "Enrichment", 5.4, INK, "bold")
    cax = mm_axes(PILL_X + 17.0, ky + 6.8, 30.0, 2.2); cax.imshow(np.linspace(-1.5, 1.5, 256)[None, :], cmap=ENR, norm=norm, aspect="auto", extent=(-1.5, 1.5, 0, 1))
    cax.set_yticks([]); cax.set_xticks([-1.5, 0, 1.5]); cax.set_xticklabels(["\u22121.5", "0", "1.5"]); cax.tick_params(axis="x", labelsize=5.2, length=1.5, width=0.4, pad=1.0)
    for s_ in cax.spines.values():
        s_.set_linewidth(0.4)
    txt(PILL_X, ky + 14.2, "log\u2082 (share \u00f7 share overall)", 5.2, INK)
    for q, (nm, c) in enumerate((("phase separation", LABC[0]), ("no phase separation", LABC[1]), ("no binary label", LABC[2]))):
        yy = ky + q * 3.6
        ov.add_patch(Rectangle((BAR_X0, yy - 1.1), 2.2, 2.2, fc=c, ec="none", zorder=3)); txt(BAR_X0 + 3.2, yy, nm, 5.2, INK)

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
            names = [SINGLE[q] for q in range(4)]
            placed, skipped = cat_panel(ax, xy, sysc, names, [PAL[n][1] for n in names], [PAL[n][0] for n in names], [darker(PAL[n][1], 0.62) for n in names], label_names=SYS_SHORT, blob_alpha=[0.0, 0.55, 0.60, 0.60])
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
                special = [((df.temp_state == "room temp. only").to_numpy(), "#B9C4D0", WHITE), ((df.temp_state == "missing").to_numpy(), None, "#2B2D42")]; keys = [("#B9C4D0", WHITE, "RT only"), (None, "#2B2D42", "missing")]
            elif k == 5:
                val = pd.to_numeric(df.nacl_log, errors="coerce").to_numpy(); cmap, v0, v1, ticks, tl = NACL, 0, 3.4, [0, 1, 2, 3], ["0", "1", "2", "3"]
                special = [(np.isnan(val), None, "#8E99A6")]; keys = [(None, "#8E99A6", "not given")]
            elif k == 6:
                val = pd.to_numeric(df.conc_v, errors="coerce").to_numpy(); cmap, v0, v1, ticks, tl = CONC, -1, 3, [-1, 0, 1, 2, 3], ["≤ −1", "0", "1", "2", "≥ 3"]
                special = [(np.isnan(val), None, "#8E99A6")]; keys = [(None, "#8E99A6", "not given")]
            else:
                val = pd.to_numeric(df.n_miss, errors="coerce").to_numpy(); cmap, v0, v1, ticks, tl = MISS, 0, 6, [0, 2, 4, 6], ["0", "2", "4", "≥ 6"]
                special = []; keys = []
            cont_panel(ax, xy, val, cmap, v0, v1, special)
            cb = mm_axes(COLX[c] + 0.5, below, 30.0, 1.8); cb.imshow(np.linspace(v0, v1, 256)[None, :], cmap=cmap, aspect="auto", extent=(v0, v1, 0, 1))
            cb.set_yticks([]); cb.set_xticks(ticks); cb.set_xticklabels(tl); cb.tick_params(axis="x", labelsize=5.2, length=1.5, width=0.4, pad=1.0)
            for s_ in cb.spines.values():
                s_.set_linewidth(0.4)
            for q, (face, edge, nm) in enumerate(keys):
                kx2 = COLX[c] + 0.5 + q * 17.0; yy = below + 8.4
                ov.scatter([kx2 + 1.1], [yy], s=(2.3 * 72 / 25.4) ** 2, facecolors=face if face else "none", edgecolors=edge, linewidths=0.5, zorder=3); txt(kx2 + 3.0, yy, nm, 5.2, INK)
        if k in (0, 1, 3):
            print(f"panel {k}: labels placed {placed}; skipped {skipped}", flush=True)

    pd.DataFrame({"record_id": df.record_id, "tsne_x": xy[:, 0], "tsne_y": xy[:, 1], "system_type": [SINGLE[i] for i in sysc], "protein": pn, "label": label3, "source": df.source,
                  "temp_state": df.temp_state, "temp_C": df.tempC, "nacl_log10_1p_mM": df.nacl_log, "protein_conc_log10_uM": df.conc_v, "condition_blocks_not_given": df.n_miss}).to_csv(f"{args.out}/FigR6.source_data_b.csv", index=False)
    base = f"{args.out}/FigR6"
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
