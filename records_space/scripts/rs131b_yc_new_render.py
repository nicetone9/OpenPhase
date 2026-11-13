"""Guidance tab 的“更多例子”：6 个新 (y, c) 集合的结果图与轨迹 GIF。除了读 gd_yc_summary.json / gd_yc_{id}.npz、文件名不带 FigR 编号以外，与 rs73_yc_result_and_gif.py 完全相同。
运行：cd ~/pixi/figs_work && pixi run --manifest-path ~/pixi/figs/pixi.toml python rs131b_yc_new_render.py --lang en --out out_gd
输出 out_gd/gd_yc_{id}_result.{png,pdf} 和 gd_yc_{id}_raw.gif（之后用 ffmpeg 压缩）。"""
import json, os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
import colorsys
import common_style as cs

args = cs.parse_args(); cs.setup_style("en"); MM = cs.MM; D = cs.DATA; OUT = args.out
SUM = json.load(open(D / "gd_yc_summary.json")); T50 = np.load(D / "rs50_contour_traj.npz"); Zf = np.load(D / "rs22_fields.npz"); U, yA = Zf["U_all"], Zf["y_all"]; Pb = U[yA >= 0]; yb = yA[yA >= 0]; P1, P0 = Pb[yb == 1], Pb[yb == 0]
RINT = RegularGridInterpolator((T50["ey"], T50["ex"]), T50["R"], bounds_error=False, fill_value=None); r_of = lambda X: RINT(np.stack([X[..., 1], X[..., 0]], -1))
INK, MUTE, WHITE, GOLD, GREY = "#26202B", "#5E6670", "#FFFFFF", "#E0A526", "#6B7078"; PS, NP = "#6B2A6B", "#008DB8"
def tint(c, l=0.83): h, _, s = colorsys.rgb_to_hls(*to_rgb(c)); return colorsys.hls_to_rgb(h, l, min(1.0, 0.5 + 0.45 * s))
PINK, CYAN = tint(PS), tint(NP)
c0 = (U.min(0) + U.max(0)) / 2; half = (U.max(0) - U.min(0)).max() / 2 * 1.12; ext = (c0[0] - half, c0[0] + half, c0[1] - half, c0[1] + half)
RES = 360; xe, ye = np.linspace(ext[0], ext[1], RES + 1), np.linspace(ext[2], ext[3], RES + 1); sg_px = 0.016 * 2 * half / ((ext[1] - ext[0]) / RES)
d1 = ndi.gaussian_filter(np.histogram2d(P1[:, 1], P1[:, 0], bins=[ye, xe])[0], sg_px); d0 = ndi.gaussian_filter(np.histogram2d(P0[:, 1], P0[:, 0], bins=[ye, xe])[0], sg_px); tot = d1 + d0
def territories(ax, alpha=0.55):
    rgba = np.zeros(tot.shape + (4,)); rgba[..., :3] = np.where((d1 >= d0)[..., None], np.array(PINK), np.array(CYAN)); f = np.clip((tot / tot.max() - 0.03) / 0.04, 0, 1); rgba[..., 3] = alpha * f * f * (3 - 2 * f); ax.imshow(rgba, origin="lower", extent=ext, interpolation="bilinear", zorder=0, aspect="auto")
def frame(ax): ax.set_xlim(*ext[:2]); ax.set_ylim(*ext[2:]); ax.set_aspect("equal"); ax.axis("off")
def faint(ax): ax.scatter(Pb[:, 0], Pb[:, 1], s=0.5, c="#8C8C8C", lw=0, alpha=0.25, rasterized=True, zorder=1)
def title(ax, t, sub=None, dx=0.0):
    n = 0 if not sub else sub.count("\n") + 1; ty = 1.012 + n * 0.056 + (0.012 if n else 0.06)
    ax.text(dx, ty, t, transform=ax.transAxes, fontsize=6.4, fontweight="bold", color=INK, ha="left", va="bottom")
    if sub: ax.text(dx, 1.012, sub, transform=ax.transAxes, fontsize=5.0, color=MUTE, ha="left", va="bottom", linespacing=1.15)
def letter(ax, s, dx=-0.02, y=1.24): ax.text(dx, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", color=INK, ha="left", va="bottom")
def lighten(c, f): return tuple(np.array(to_rgb(c)) * (1 - f) + f)
def clean(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(length=2, width=0.4, labelsize=5.2)
NSH = 300

# ============================================================ 结果图
W_, H_ = 183.0, 150.0; PW, PG, X0, PH = 50.0, 12.5, 6.0, 48.0
def mm_axes(fig, col, row, left_pad=0.0):
    top = 19.0 if row == 0 else 19.0 + PH + 21.0; x = X0 + col * (PW + PG) + left_pad; return fig.add_axes([x / W_, 1 - (top + PH) / H_, (PW - left_pad) / W_, PH / H_])

def result_fig(name):
    S = SUM[name]; Z = np.load(D / f"gd_yc_{name}.npz"); U1c, U0c = Z["U1"], Z["U0"]; fig = plt.figure(figsize=(W_ * MM, H_ * MM))
    def end_map(ax, a, key, col, ttl, letter_):
        frame(ax); territories(ax); faint(ax); same, tgt = (U0c, U1c) if a == "0to1" else (U1c, U0c); scol, tcol = (NP, PS) if a == "0to1" else (PS, NP)
        ax.scatter(same[:, 0], same[:, 1], s=9, facecolor=scol, edgecolor=WHITE, linewidth=0.3, zorder=5); ax.scatter(tgt[:, 0], tgt[:, 1], s=26, facecolor="none", edgecolor=tcol, linewidth=0.8, zorder=8)
        E = Z[f"{a}_end_{key}"].astype(float); r0 = r_of(Z[f"{a}_start"].astype(float)); re = r_of(E); ax.scatter(E[:, 0], E[:, 1], s=5.5 if len(E) > 300 else 7.5, c=col, edgecolor=WHITE, linewidth=0.15, alpha=0.65, zorder=6, rasterized=True)
        m = S[a][key]; title(ax, ttl, f"{len(E)} chain end points after 1,200 steps\nmedian r {np.median(r0):+.2f} → {np.median(re):+.2f} · ≤ 0.10 SD of a target record: {100 * m['arrived_0p10']:.0f}%"); letter(ax, letter_)
    end_map(mm_axes(fig, 0, 0), "0to1", "guided", PS, "Ends: y = 0 start, guided to y = 1", "a"); end_map(mm_axes(fig, 1, 0), "1to0", "guided", NP, "Ends: y = 1 start, guided to y = 0", "b")
    end_map(mm_axes(fig, 0, 1), "0to1", "none", GREY, "Ends: y = 0 start, no guidance", "d"); end_map(mm_axes(fig, 1, 1), "1to0", "none", GREY, "Ends: y = 1 start, no guidance", "e")
    # c：r 变化箱线
    ax = mm_axes(fig, 2, 0, left_pad=8.0); xs = []; labs = []; pos = 0.0
    for a, col in (("0to1", PS), ("1to0", NP)):
        for key, alpha_, lab in (("guided", 1.0, "guided"), ("stay", 0.45, "to same y"), ("none", 0.45, "none")):
            sn = Z[f"{a}_snap_{key}"].astype(float); dr = r_of(sn[:, -1]) - r_of(sn[:, 0]); q = np.percentile(dr, [5, 25, 50, 75, 95]); c_ = col if key == "guided" else (GREY if key == "none" else lighten(col, 0.55))
            ax.add_patch(plt.Rectangle((pos - 0.32, q[1]), 0.64, q[3] - q[1], facecolor=c_, edgecolor="none", alpha=alpha_ if key == "guided" else 0.8, zorder=3)); ax.plot([pos - 0.32, pos + 0.32], [q[2], q[2]], color=INK, lw=0.9, zorder=4); ax.plot([pos, pos], [q[0], q[1]], color=INK, lw=0.5, zorder=2); ax.plot([pos, pos], [q[3], q[4]], color=INK, lw=0.5, zorder=2)
            xs.append(pos); labs.append(lab); pos += 1.0
        pos += 0.8
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + 0.18 * (hi_ - lo_)); ax.axhline(0, color=INK, lw=0.4); ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=4.6, rotation=35, ha="right"); ax.set_xlim(-0.7, pos - 0.9); ax.set_ylabel("r(end) − r(start)", fontsize=5.2); clean(ax)
    ax.text(1, 0.97, "y = 0 → 1", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.0, color=PS, fontweight="bold"); ax.text(5.8, 0.97, "y = 1 → 0", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.0, color=NP, fontweight="bold")
    title(ax, "Change of r after 1,200 steps", "box: 25–75%, line: median, whiskers: 5–95%\nto same y = guided to the start label", dx=-8.0 / 42.0); letter(ax, "c", dx=-8.0 / 42.0 - 0.02)
    # f：到达比例
    ax = mm_axes(fig, 2, 1, left_pad=8.0); pos = 0.0; xs = []; labs = []
    for a, col in (("0to1", PS), ("1to0", NP)):
        for key, lab in (("guided", "guided"), ("stay", "to same y"), ("none", "none")):
            m = S[a][key]; c_ = col if key == "guided" else (GREY if key == "none" else lighten(col, 0.55)); ax.bar(pos, 100 * m["arrived_0p20"], width=0.7, color=lighten(c_, 0.65), edgecolor=c_, linewidth=0.4, zorder=2); ax.bar(pos, 100 * m["arrived_0p10"], width=0.7, color=c_, zorder=3)
            ax.text(pos, 100 * m["arrived_0p20"] + 0.8, f"{100 * m['arrived_0p10']:.0f}", ha="center", va="bottom", fontsize=4.8, color=INK); xs.append(pos); labs.append(lab); pos += 1.0
        pos += 0.8
    hi_ = ax.get_ylim()[1]; ax.set_ylim(0, hi_ * 1.3); ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=4.6, rotation=35, ha="right"); ax.set_xlim(-0.7, pos - 0.9); ax.set_ylabel("% of chains", fontsize=5.2); clean(ax)
    ax.text(1, 0.97, "target: y = 1", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.0, color=PS, fontweight="bold"); ax.text(5.8, 0.97, "target: y = 0", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.0, color=NP, fontweight="bold")
    title(ax, "Chains ending on target-label records", "solid: within 0.10 SD · light: within 0.20 SD\nnumber = % within 0.10 SD", dx=-8.0 / 42.0); letter(ax, "f", dx=-8.0 / 42.0 - 0.02)
    fig.text(6 / W_, 1 - 2.6 / H_, S["desc"], fontsize=6.8, fontweight="bold", color=INK, va="center")
    fig.text(6 / W_, 1 - 146.0 / H_, "maps: filled dots = records with the start label, rings = records with the target label of this (y, c) set, small dots = chain end points", fontsize=5.0, color=MUTE, va="center")
    fn = f"gd_yc_{name}_result"
    for e in ("png", "pdf"): fig.savefig(f"{OUT}/{fn}.{e}", dpi=300)
    plt.close(fig); print("导出：", fn)

# ============================================================ 轨迹 GIF
def traj_gif(name, stride=10, trail=60, fps=15, hold_start=8, hold_end=120):
    """开头停 hold_start 帧；跑完后停在终点 hold_end 帧（约 8 秒）：只显示黑色的小终点点（用户 11/13：最后的点用黑色、不要那么粗），不再保留轨迹线。"""
    S = SUM[name]; Z = np.load(D / f"gd_yc_{name}.npz"); U1c, U0c = Z["U1"], Z["U0"]; fig = plt.figure(figsize=(183.0 * MM, 100.0 * MM), dpi=110); axes = [fig.add_axes([6 / 183, 4 / 100, 84 / 183, 84 / 100]), fig.add_axes([97 / 183, 4 / 100, 84 / 183, 84 / 100])]
    info = []; txts = []
    for ax, a, col, tgt, same, ttl in ((axes[0], "0to1", PS, U1c, U0c, "From y = 0 records, guided to y = 1"), (axes[1], "1to0", NP, U0c, U1c, "From y = 1 records, guided to y = 0")):
        frame(ax); territories(ax); faint(ax); P = Z[f"{a}_path"].astype(float); ax.scatter(same[:, 0], same[:, 1], s=10, facecolor=(NP if a == "0to1" else PS), edgecolor=WHITE, linewidth=0.3, zorder=4); ax.scatter(tgt[:, 0], tgt[:, 1], s=34, facecolor="none", edgecolor=col, linewidth=0.9, zorder=8)
        ax.scatter(P[:, 0, 0], P[:, 0, 1], s=6, c=GOLD, edgecolor=INK, linewidth=0.3, zorder=9); lc = LineCollection([], linewidths=0.8, zorder=5); ax.add_collection(lc); sc = ax.scatter(P[:, 0, 0], P[:, 0, 1], s=11, c=col, edgecolor=WHITE, linewidth=0.4, zorder=10)
        fin = ax.scatter(P[:, -1, 0], P[:, -1, 1], s=16, c=INK, edgecolor=WHITE, linewidth=0.3, zorder=12, visible=False)                  # 终点：黑色小点
        ax.text(0.0, 1.0, ttl, transform=ax.transAxes, fontsize=7.2, fontweight="bold", color=INK, va="bottom"); txt = ax.text(0.99, 0.01, "", transform=ax.transAxes, fontsize=6.4, color=INK, ha="right", va="bottom"); txts.append(txt); info.append((P, lc, sc, fin, col, a))
    fig.text(6 / 183, 1 - 3.0 / 100, S["desc"] + "  ·  gold: start, small dot: current position, black dot: final position, rings: target-label records", fontsize=6.0, color=MUTE, va="center")
    steps_all = np.arange(0, info[0][0].shape[1], stride); sched = [(0, False)] * hold_start + [(int(t), False) for t in steps_all[1:-1]] + [(int(steps_all[-1]), True)] * hold_end
    def update(fi):
        t, final = sched[fi]; arts = []
        for (P, lc, sc, fin, col, a), txt in zip(info, txts):
            t0 = max(0, t - trail); segs = []; cols = []
            for p in ([] if final else P):
                seg = p[t0:t + 1]
                if len(seg) < 2: continue
                n = len(seg) - 1; segs.extend(np.stack([seg[:-1], seg[1:]], 1)); al = np.linspace(0.05, 0.9, n); rgb = np.array(to_rgb(col)); cols.extend(np.c_[np.tile(rgb, (n, 1)), al])
            lc.set_segments(segs); lc.set_color(cols if cols else col); lc.set_visible(not final); sc.set_offsets(P[:, t, :]); sc.set_visible(not final); fin.set_visible(final); dr = float(np.median(r_of(P[:, t, :]) - r_of(P[:, 0, :])))
            txt.set_text((f"final positions · step {t} · median Δr {dr:+.2f}" if final else f"step {t}/{P.shape[1] - 1} · median Δr {dr:+.2f}").replace("-", "−")); arts += [lc, sc, fin, txt]
        return arts
    anim = FuncAnimation(fig, update, frames=len(sched), blit=False); fn = f"{OUT}/gd_yc_{name}_raw.gif"; anim.save(fn, writer=PillowWriter(fps=fps), dpi=110); plt.close(fig); print("导出：", fn, f"（{len(sched)} 帧，其中结尾停留 {hold_end} 帧）")


for n in (os.environ.get("ONLY", "").split(",") if os.environ.get("ONLY") else list(SUM)): result_fig(n); traj_gif(n)
