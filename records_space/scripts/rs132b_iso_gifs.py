"""Guidance tab 的“更多例子”：新增等值线的一维弧长 + TorchEBM HMC 采样 GIF。除了读 gd_iso_summary.json / gd_iso_candidates.json / gd_iso_{id}.npz、文件名不带 FigR 编号、标题用候选表里的 label 以外，与 rs96_arc_hmc_gifs.py 完全相同。
运行：cd ~/pixi/figs_work && pixi run --manifest-path ~/pixi/figs/pixi.toml python rs132b_iso_gifs.py --lang en --out out_gd
输出 out_gd/gd_iso_{id}_raw.gif（之后用 ffmpeg 压缩）。"""
import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator
import common_style as cs

args = cs.parse_args(); cs.setup_style("en"); MM = cs.MM; D = cs.DATA; OUT = args.out
C = np.load(D / "rs80_contours.npz"); SM = json.load(open(D / "gd_iso_summary.json")); CAND = {r["id"]: r for r in json.load(open(D / "gd_iso_candidates.json"))["candidates"]}; Zf = np.load(D / "rs22_fields.npz"); U, yA = Zf["U_all"], Zf["y_all"]; Pb = U[yA >= 0]
ex, ey, R, dnn = C["ex"], C["ey"], C["R"], C["dnn"]; GX, GY = np.meshgrid(ex, ey); RINT = RegularGridInterpolator((ey, ex), R, bounds_error=False, fill_value=None); r_of = lambda X: RINT(np.stack([X[..., 1], X[..., 0]], -1))
INK, MUTE, WHITE, GOLD = "#26202B", "#5E6670", "#FFFFFF", "#E0A526"; PS, NP = "#6B2A6B", "#008DB8"
c0 = (U.min(0) + U.max(0)) / 2; half = (U.max(0) - U.min(0)).max() / 2 * 1.12; ext = (c0[0] - half, c0[0] + half, c0[1] - half, c0[1] + half)
cmapr = LinearSegmentedColormap.from_list("r", [NP, "#7CCBE6", "#FFFFFF", "#D79AD7", PS])
def background(ax):
    ax.set_xlim(*ext[:2]); ax.set_ylim(*ext[2:]); ax.set_aspect("equal"); ax.axis("off"); lim = 2.0; rgba = cmapr(np.clip((R + lim) / (2 * lim), 0, 1)); a = np.clip((2.2 - dnn) / 1.0, 0, 1); rgba[..., 3] = 0.55 * a * a * (3 - 2 * a); ax.imshow(rgba, origin="lower", extent=ext, interpolation="bilinear", zorder=0, aspect="auto")
    ax.scatter(Pb[:, 0], Pb[:, 1], s=0.5, c="#8C8C8C", lw=0, alpha=0.3, rasterized=True, zorder=1); ax.contour(GX, GY, np.ma.masked_where(dnn > 2.0, R), levels=[0], colors=[INK], linewidths=0.35, linestyles=[(0, (3, 2))], zorder=2)

def make(lid, per=4, ghosts=5, fps=15, hold_start=8, hold_end=120):
    if lid not in SM: return
    Z = np.load(D / f"gd_iso_{lid}.npz"); label = CAND[lid]["label"]; sm = SM[lid]; pts = Z["pts"]; fig = plt.figure(figsize=(183.0 * MM, 100.0 * MM), dpi=110)
    axes = [fig.add_axes([6 / 183, 4 / 100, 84 / 183, 84 / 100]), fig.add_axes([97 / 183, 4 / 100, 84 / 183, 84 / 100])]; info = []; txts = []
    for ax, key, col, lab in ((axes[0], "path_to_y1", PS, "guided to y = 1"), (axes[1], "path_to_y0", NP, "guided to y = 0")):
        background(ax); P = Z[key].astype(float); ax.scatter(pts[:, 0], pts[:, 1], s=0.8, c=INK, lw=0, zorder=3); ax.scatter(P[:, 0, 0], P[:, 0, 1], s=6, c=GOLD, edgecolor=INK, linewidth=0.3, zorder=9)
        gh = [ax.scatter(P[:, 0, 0], P[:, 0, 1], s=7, c=col, lw=0, alpha=0.0, zorder=8) for _ in range(ghosts)]; sc = ax.scatter(P[:, 0, 0], P[:, 0, 1], s=9, c=col, edgecolor=WHITE, linewidth=0.3, zorder=10); fin = ax.scatter(P[:, -1, 0], P[:, -1, 1], s=14, c=INK, edgecolor=WHITE, linewidth=0.3, zorder=12, visible=False)
        ax.text(0.0, 1.0, f"{label}: {lab}", transform=ax.transAxes, fontsize=7.2, fontweight="bold", color=INK, va="bottom"); txt = ax.text(0.99, 0.01, "", transform=ax.transAxes, fontsize=6.4, color=INK, ha="right", va="bottom"); txts.append(txt); info.append((P, gh, sc, fin, key[5:]))
    acc = [sm["methods"][d]["accept_rate"] for d in ("to_y1", "to_y0") if sm["methods"][d].get("accept_rate") is not None]
    fig.text(6 / 183, 1 - 3.0 / 100, f"1-D arc-length HMC (TorchEBM) on the iso-line · {len(info[0][0])} chains · ε = {sm['eps']}, {sm['leapfrog']} leapfrog steps, T = {sm['T']}, w = {sm['w']} · accept rate {np.mean(acc):.2f} · gold: start, coloured: current + 5 ghosts, black: final".replace("accept rate nan", ""), fontsize=5.4, color=MUTE, va="center")
    nT = info[0][0].shape[1]; fr = list(range(0, nT, per)); sched = [(0, False)] * hold_start + [(t, False) for t in fr[1:]] + [(nT - 1, True)] * hold_end
    def update(fi):
        t, final = sched[fi]; arts = []
        for (P, gh, sc, fin, d), txt in zip(info, txts):
            for j, g in enumerate(gh):
                tj = t - per * (j + 1)
                if final or tj < 0: g.set_alpha(0.0)
                else: g.set_offsets(P[:, tj, :]); g.set_alpha(0.45 * (1 - j / ghosts))
                arts.append(g)
            sc.set_offsets(P[:, t, :]); sc.set_visible(not final); fin.set_visible(final); dr = float(np.median(r_of(P[:, t, :]) - r_of(P[:, 0, :])))
            txt.set_text((f"final positions · proposal {t} · median Δr {dr:+.2f}" if final else f"proposal {t}/{nT - 1} · median Δr {dr:+.2f}").replace("-", "−")); arts += [sc, fin, txt]
        return arts
    anim = FuncAnimation(fig, update, frames=len(sched), blit=False); fn = f"{OUT}/gd_iso_{lid}_raw.gif"; anim.save(fn, writer=PillowWriter(fps=fps), dpi=110); plt.close(fig); print("导出：", fn, f"（{len(sched)} 帧）")

for lid in list(SM): make(lid)
