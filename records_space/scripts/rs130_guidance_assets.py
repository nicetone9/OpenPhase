#!/usr/bin/env python3
"""rs130（本机；pixi 环境 ~/pixi/figs，需要 ffmpeg）：Records space explorer 里 Guidance tab 的素材与数据。
输入：figs/ 里别的会话做好的 GIF、结果图、json、解码表（Task 3 guidance 空间），guidance_assets/new/ 里 fork 新增的例子（有就读，没有就跳过）。
输出：
  guidance_assets/web/          压缩后的 GIF（调色板 64 色、不抖动，帧数与每帧时长不变）、海报帧（GIF 的最后一帧，WebP）、结果图与背景图（WebP）、alignments 文本；站点里放在 records_space/guidance/
  guidance_page/guidance_data.json   explorer 内嵌的数据（三块：score 场、采样例子（(y,c) 集合与等值线）、地图点→序列）
每个数字都从 json / tsv 读，页面上的说明文字写在下面的 TEXT 里。
运行：~/.pixi/bin/pixi run --manifest-path ~/pixi/figs/pixi.toml python scripts/rs130_guidance_assets.py
"""
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rs134_guidance_results as R134      # the four interactive main results

ROOT = Path(__file__).resolve().parent.parent
FIGS, OUT = ROOT / "figs", ROOT / "out"
NEW, WEB, SRC = ROOT / "guidance_assets" / "new", ROOT / "guidance_assets" / "web", ROOT / "guidance_assets" / "src"
PAGE = ROOT / "guidance_page"
for d in (WEB, PAGE):
    d.mkdir(parents=True, exist_ok=True)
FF = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
MAX_GIF = 4.2e6


def fresh(dst, *srcs):
    return dst.exists() and all(dst.stat().st_mtime >= s.stat().st_mtime for s in srcs)


def gif_compress(src, dst, colors=64):
    """ffmpeg 调色板压缩：不抖动、只存帧间差异；帧数和每帧时长不变（结尾黑点停留那一帧的 7.9 s 也保留）。"""
    if fresh(dst, src):
        return
    for c in (colors, 48, 32):
        vf = f"split[s0][s1];[s0]palettegen=max_colors={c}:stats_mode=diff[p];[s1][p]paletteuse=dither=none:diff_mode=rectangle"
        subprocess.run([FF, "-v", "error", "-y", "-i", str(src), "-vf", vf, "-fps_mode", "passthrough", str(dst)], check=True)
        if dst.stat().st_size <= MAX_GIF:
            break


def gif_poster(gif, dst):
    if fresh(dst, gif):
        return
    im = Image.open(gif)
    im.seek(im.n_frames - 1)
    im.convert("RGB").save(dst, "WEBP", quality=86, method=6)


def to_webp(src, dst, maxw=1600, q=82):
    im = Image.open(src).convert("RGB")
    if not fresh(dst, src):
        if im.width > maxw:
            im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
        im.save(dst, "WEBP", quality=q, method=6)
    w, h = Image.open(dst).size
    return {"w": w, "h": h}


def gif_info(gif):
    im = Image.open(gif)
    return {"w": im.size[0], "h": im.size[1], "frames": im.n_frames}


def fmtn(n):
    return f"{n:,}"


def rnd(v, n=3):
    return None if v is None else round(float(v), n)


# ------------------------------------------------------------------ 文字（英文，页面直接显示）
TEXT = {
    "title": "Guidance on the map",
    "intro": "A score field learned on the 2-D t-SNE of the 11,439 records pushes sampled points toward phase-separating (y = 1) or non-separating (y = 0) regions. "
             "First the methods (the score field, sampling along an iso-line, sampling inside a region), then four main results as interactive maps: hover a point for its start and end, the records next to it, "
             "the sequence change and the predictions. The other examples are collapsed at the end.",
    "field": {
        "title": "The score field",
        "paras": [
            "A TorchEBM energy model E(u | y) is trained on the t-SNE coordinates u of the records only (conditional denoising score matching, 15% label dropout, Fourier features), "
            "with kernel width σ = 0.16 inside the data, rising smoothly to 0.50 outside. Its direction agrees with the analytic Gaussian-KDE score (cosine 0.994). "
            "The guidance direction is s(u | 1) − s(u | 0) = ∇ log p(u | 1) − ∇ log p(u | 0); guided sampling uses the classifier-free energy E_n + w (E_y − E_n).",
        ],
        "facts": [
            "It describes the label layout of this map. With papers held out (checked on the first, σ = 0.12 field), the share of arrows pointing toward y = 1 is 0.469 (labels permuted: 0.42), so it does not predict labels for new papers.",
            "Guidance works, within limits: guiding to y = 1 raises the share of samples that land in the y = 1 territory from 67% (no guidance) to about 75% (the records themselves: 77%), and weights above w = 2 add nothing. "
            "Guiding to y = 0 with w = 4 pushes 28% of the samples outside the data, so w ≤ 2 is the usable range.",
            "The field has no condition c as input: it only knows where the records of each label sit.",
        ],
        "figs": [
            {"id": "field_scorefield", "src": FIGS / "FigR7a_records_y_score_field_s016_blend.png",
             "cap": "Score toward y = 1 on the t-SNE plane (36 × 36 lattice, every arrow drawn). Boxes: six sub-spaces that hold both labels and where y depends on the conditions."},
            {"id": "field_energy_samples", "src": FIGS / "FigR8_energy_and_samples.png",
             "cap": "Energies E(u | y), the log density ratio r(u) = log p(u | 1) − log p(u | 0), and where guided samples land (400 samples per setting)."},
        ],
    },
    "yc": {
        "title": "From a (y, c) set",
        "intro": "A (y, c) set is the group of records that share one label inside one condition cell. Chains start at the map positions of the start-label records (16 chains per record, 0.03 SD jitter) "
                 "and are guided to the opposite label for 1,200 steps (Langevin, temperature 0.5, w = 4); the controls are no guidance and guidance to the same label. "
                 "The GIF shows 24 chains (a frame every 10 steps, 60-step tail), left guided to y = 1 and right to y = 0; the black dots are the final positions and stay for 8 s. "
                 "r is the log density ratio of the field, Δr its change from start to end; “within 0.10 SD” counts chains that end within 0.10 map SD of a record of the target label in the same cell; in the tables “none” is the control without guidance.",
    },
    "iso": {
        "title": "Along an iso-line of one condition",
        "intro": "An iso-line is a smoothed contour of one covariate (sequence length, NaCl, temperature, pH, concentration, crowder) over the map. Chains are sampled on the line, parameterised by arc length, "
                 "with TorchEBM's Hamiltonian Monte Carlo: 10 leapfrog steps of 0.05 SD per proposal, Metropolis acceptance, temperature 0.5, w = 4, 400 proposals, 60 start points × 8 chains. "
                 "A chain cannot leave the line and stays in the piece of the line it starts on (a line has several pieces). "
                 "The GIF shows 60 chains, left guided to y = 1 and right to y = 0, 4 proposals per frame, the black dots are the final positions. "
                 "Homogeneity is how uniform the covariate is near the line (lower is better); TV is the distance of the sampled distribution to the exact one on the line.",
        "fig": {"id": "iso_sampler_comparison", "src": FIGS / "FigR74_sampler_comparison.png",
                "cap": "Samplers compared on two iso-lines (480 chains each, equal cost): TorchEBM Langevin and HMC with the line written as a thin tube, and the same samplers on the 1-D arc-length coordinate."},
        "recommend": [
            "On a line: 1-D arc-length HMC. Distance to the exact distribution (TV) 0.07 on NaCl 150 mM and 0.15 on 22 °C, against 0.28 and 0.52 for the earlier projected Langevin; 0% of the chains leave the line (11% on the NaCl line before).",
            "Inside a region cut out by several constraints: exact sampling on the 160 × 160 energy grid. The region has disconnected pieces and MCMC leaks between them (distance to the exact distribution 0.20–0.66, against a sampling noise floor of 0.05–0.15). HMC is for showing the process.",
            "Both ends of a line sit next to the same proteins (87–99% shared in the Langevin run), so y differences along a line come from other conditions or from the source paper, not from the sequence.",
        ],
    },
    "pts": {
        "title": "From a map point to a sequence",
        "intro": "A map point has no inverse in sequence space, so the decoder goes through the records around it. The anchor is the real sequence with the largest Gaussian weight among the neighbours of the point; "
                 "the target is the ESMC-600M embedding (layer 18, z-scored, PCA to 64) of those neighbours. ESMC's masked language model proposes substitutions at random positions, added one at a time "
                 "(an edit ladder: k substitutions after step k, sequences already in the library excluded). The answer is the step k* closest to the target; k* = 0 means the anchor itself is the best answer. "
                 "D is the distance between the ESMC embedding (64-D, unit sphere) of a sequence and the target. "
                 "The 12 points of a row are quantiles along the line of the exact samples of (region ∩ iso-line), 6 guided to y = 1 and 6 to y = 0.",
        "limits": [
            "Many points return their anchor (k* = 0): the target embedding is then almost the anchor's own. The TDP-43 example is the one where most answers carry substitutions.",
            "Positions are proposed at random and kept when they move the embedding toward the target; they are not chosen for function, and the decoded variants are not scored for phase separation here.",
        ],
    },
}

METHODS = {
    "iso": {
        "title": "Sampling along an iso-line",
        "steps": [
            ["Line.", "Each covariate (sequence length, NaCl, temperature, pH, concentration, crowder) is smoothed over the map with a Gaussian kernel (σ = 0.16 map SD) into a field g(u); the iso-line is the contour g(u) = g₀. "
                      "A line is kept when the covariate is uniform near it (homogeneity < 1), it is at least 1 SD long, at least 10 records of each label lie within 0.12 SD of it, r changes sign along it and at least 5 papers lie near it."],
            ["Coordinate.", "The line is parameterised by arc length s. A line has several pieces; each piece is an interval of s."],
            ["Target.", "On a piece, p(s) ∝ exp(−E_w(u(s)) / T) with the guided energy E_w = E_n + w (E_y − E_n) of the TorchEBM conditional energy (σ = 0.16), w = 4, T = 0.5, y = 1 or y = 0."],
            ["Sampler.", "Hamiltonian Monte Carlo on s (TorchEBM): 10 leapfrog steps of 0.05 SD per proposal, Metropolis accept/reject, steep walls at the ends of a piece, 400 proposals; 60 start points equally spaced in arc length × 8 chains per direction. "
                         "The chain cannot leave the line and stays on the piece where it starts."],
            ["Checks.", "TV distance to the exact distribution on the line, acceptance rate, median move along the line, share of chains that end off the data."],
        ],
        "fig": {"id": "iso_sampler_comparison", "src": FIGS / "FigR74_sampler_comparison.png",
                "cap": "Samplers compared on two iso-lines (NaCl 150 mM and 22 °C; 480 chains each, equal cost): TorchEBM Langevin and HMC with the line written as a thin tube, and the same samplers on the 1-D arc-length coordinate."},
        "recommend_title": "Why this sampler",
        "recommend": ["Distance to the exact distribution (TV) 0.07 on NaCl 150 mM and 0.15 on 22 °C, against 0.28 and 0.52 for the earlier projected Langevin; 0% of the chains leave the line (11% on the NaCl line before). Langevin on the same coordinate gets stuck; HMC moves 0.3–1.3 SD along the line."],
    },
    "region": {
        "title": "Sampling inside a region",
        "steps": [
            ["Region.", "Several conditions at once: F = {u : |g_k(u) − g_k0| ≤ τ_k for every constraint k, the area is data-supported, and the local spread of the covariate is at most 0.8 of its global spread}. "
                        "The example in the results, C3, is 22 ± 4 °C, pH 7.5 ± 0.35, NaCl 150 mM (log₁₀(1 + x) ± 0.25) and 10 µM (log₁₀ ± 0.3): 735 cells of the 160 × 160 grid in 11 separate pieces."],
            ["Target.", "The same guided energy E_w (w = 4, T = 0.5), tabulated on the grid; the probability of a cell inside the region is ∝ exp(−E_w / T)."],
            ["Sampler (recommended): exact grid sampling.", "Draw cells at random with these probabilities and jitter uniformly inside the cell. The samples are independent, there is nothing to tune and nothing leaves the region. HMC is kept for showing the process (the GIFs)."],
            ["Why not MCMC.", "The pieces are disconnected: Langevin or HMC with a wall leaks out of the region (up to 16% of the chains) or cannot change piece; their distance to the exact distribution is 0.20–0.66, "
                              "while exact sampling is as close as two independent exact samples are to each other (0.05–0.15, the sampling-noise floor)."],
        ],
    },
}

# 现有四个 (y,c) 集合：标题、条件、说明（说明来自 Task 3 guidance 空间会话对结果的读法）
YC_OLD = [
    dict(id="physio", gif="FigR20_traj_physio.gif", res="FigR20_result_physio.png", title="Physiological cell, across papers",
         cond="22 °C · pH 7.5 · 100–200 mM NaCl · no crowder · 10–32 µM", paper="43 papers",
         notes=["The two labels interleave on the map (the mean r of the two classes differs by only 0.05), so the field cannot separate them."]),
    dict(id="hnrnpdl", gif="FigR21_traj_hnrnpdl.gif", res="FigR21_result_hnrnpdl.png", title="hnRNPDL variants",
         cond="pmid:31995753 · 150 mM NaCl · 50 µM", paper="pmid:31995753",
         notes=["Both labels sit at the same map point (0.00 SD apart, r differs by 0.00): there is no iso-line to start from, and guidance moves the chains away from the neighbourhood of the variants. Variant edits have to be made in sequence space."]),
    dict(id="brd4", gif="FigR22_traj_brd4.gif", res="FigR22_result_brd4.png", title="BRD4",
         cond="pmid:32203489 · 22 °C · pH 7.5 · 0 NaCl · no crowder · 10–32 µM", paper="pmid:32203489",
         notes=["As for hnRNPDL, both labels sit almost at the same map point (0.02 SD apart, r differs by 0.00): the map cannot separate the variants."]),
    dict(id="elp", gif="FigR23_traj_elp.gif", res="FigR23_result_elp.png", title="Designed repeat polypeptides",
         cond="pmid:26390327 · 30–40 °C · pH 7.5 · 0 NaCl · 32–100 µM", paper="pmid:26390327",
         notes=["The only set whose labels the map separates (r differs by 0.39). These are designed sequences, not natural proteins, and the 30–40 °C bin spans 10 °C."]),
]

ISO_ORDER = ["NaCl", "temp", "pH", "length", "conc", "crowder"]
ISO_TITLE = {"length": "Sequence length = 400 aa", "NaCl": "NaCl = 150 mM", "temp": "Temperature = 22 °C", "pH": "pH = 7.5", "conc": "Protein concentration = 10 µM", "crowder": "Crowder = 0.5%"}
ISO_NOTES = {
    "NaCl": ["The shortest and most homogeneous line: few points, several separate pieces; chains stay on the piece they start on."],
    "temp": ["Covers the most records, but 22 °C mostly means “measured at room temperature”.",
             "Guided to y = 0, the sampled distribution is still 0.37 from the exact one after 400 proposals (not converged)."],
    "length": ["Sequence length is only weakly encoded by the map position, so this line is less homogeneous than the NaCl and temperature lines."],
    "crowder": ["Not a meaningful line: the crowder content near it is not uniform (homogeneity 0.98).",
                "Guided to y = 0, the sampled distribution is 0.32 from the exact one."],
}


def yc_row(row_id, title, cond, paper, S, gif_src, res_src, notes, note_extra=None):
    gif_compress(gif_src, WEB / f"yc_{row_id}.gif")
    gif_poster(WEB / f"yc_{row_id}.gif", WEB / f"yc_{row_id}_poster.webp")
    res = to_webp(res_src, WEB / f"yc_{row_id}_result.webp", maxw=1700) if res_src is not None else None
    d = {}
    for a in ("0to1", "1to0"):
        g, n = S[a]["guided"], S[a]["none"]
        d[a] = dict(dr=rnd(g["dr_median"], 2), toward=rnd(g["frac_dr_toward"], 3), toward_none=rnd(n["frac_dr_toward"], 3), arr=rnd(g["arrived_0p10"], 3), arr_none=rnd(n["arrived_0p10"], 3),
                    dend=rnd(g["d_end_median"], 2), dend_none=rnd(n["d_end_median"], 2), chains=int(S[a]["n_chains"]), onsup=rnd(g["on_support"], 3))
    return dict(id=row_id, kind="yc", title=title, cond=cond, paper=paper,
                set=dict(n1=S["n1"], n0=S["n0"], seq=S["n_seq"], papers=S["n_paper"], prots=S["n_prot"]), map=dict(dist=rnd(S["map_dist_medians"], 2), dr=rnd(S["dr_y1_minus_y0"], 2)),
                dir=d, gif=f"yc_{row_id}.gif", poster=f"yc_{row_id}_poster.webp", result=f"yc_{row_id}_result.webp", result_size=res, gif_info=gif_info(WEB / f"yc_{row_id}.gif"),
                gif_kb=round((WEB / f"yc_{row_id}.gif").stat().st_size / 1e3), notes=list(notes) + (note_extra or []))


def auto_yc_notes(S):
    """Notes that follow directly from the numbers of a set."""
    n = []
    if S["n_paper"] == 1:
        n.append("One paper only: label differences inside the cell are differences between that paper's constructs or conditions.")
    if S["n_prot"] > 1:
        n.append(f"The set mixes {S['n_prot']} UniProt entries.")
    d, dr = S["map_dist_medians"], S["dr_y1_minus_y0"]
    if d < 0.12:
        n.append(f"The two labels sit almost on top of each other on the map ({d:.2f} SD apart, mean r differs by {abs(dr):.2f}), so the field cannot separate them.")
    a = S["0to1"]
    if a["guided"]["dr_median"] < 0.1 and a["none"]["dr_median"] < -0.3:
        n.append(f"Guided \u0394r from y = 0 to y = 1 is about {a['guided']['dr_median']:+.2f} because the unguided chains drift to lower r ({a['none']['dr_median']:+.2f}); guidance only cancels that drift.".replace("-", "\u2212"))
    return n


def build_yc():
    S70 = json.load(open(FIGS / "rs70_summary.json"))
    rows = [yc_row(o["id"], o["title"], o["cond"], o["paper"], S70[o["id"]], FIGS / o["gif"], FIGS / o["res"], o["notes"]) for o in YC_OLD]
    if (NEW / "yc").is_dir():
        for jf in sorted((NEW / "yc").glob("*.json")):
            j = json.load(open(jf))
            rid = j["id"]
            gif, res = jf.with_suffix(".gif"), jf.parent / f"{rid}_result.png"
            if not gif.exists():
                print("缺 GIF，跳过", rid)
                continue
            S = j["summary"]
            notes = list(j.get("notes", [])) + auto_yc_notes(S)
            if min(S["n1"], S["n0"]) < 8:
                notes.append(f"Small set ({S['n1']} y = 1, {S['n0']} y = 0 records).")
            cond = re.sub(r"^.*?\(pmid:\d+\),\s*", "", j.get("desc", ""))
            rows.append(yc_row(rid, j.get("title", rid), cond, j.get("paper", ""), S, gif, res if res.exists() else None, notes))
    return rows


def iso_title(cov, level):
    if cov == "conc":
        return f"Protein concentration = {10 ** level:.3g} µM"
    return {"NaCl": f"NaCl = {level:g} mM", "temp": f"Temperature = {level:g} °C", "pH": f"pH = {level:g}", "length": f"Sequence length = {level:g} aa", "crowder": f"Crowder = {level:g}%"}[cov]


def auto_iso_notes(m, S):
    n = []
    if m.get("purity") is not None and m["purity"] >= 0.9:
        n.append(f"Marginal: homogeneity {m['purity']:.2f}, the line is only {m['arc']:.1f} SD long and lies near {m['n_paper']} papers.")
    for k, lab in (("to_y1", "y = 1"), ("to_y0", "y = 0")):
        mm = S["methods"][k]
        if mm["TV_seg"] >= 0.2:
            n.append(f"Guided to {lab}, the sampled distribution is {mm['TV_seg']:.2f} from the exact one after 400 proposals (not fully converged).")
        if mm["off_support"] >= 0.08:
            n.append(f"Guided to {lab}, {mm['off_support'] * 100:.0f}% of the chains end off the data.")
    return n


def iso_row(cov, level, rid, m, S, gif_src, notes):
    gif_compress(gif_src, WEB / f"iso_{rid}.gif")
    gif_poster(WEB / f"iso_{rid}.gif", WEB / f"iso_{rid}_poster.webp")
    hm = {k: dict(tv=rnd(S["methods"][k]["TV_seg"], 2), acc=rnd(S["methods"][k]["accept_rate"], 2), move=rnd(S["methods"][k]["arc_move_median"], 1), off=rnd(S["methods"][k]["off_support"], 3), pieces=S["methods"][k]["n_segments"]) for k in ("to_y1", "to_y0")}
    return dict(id=rid, kind="iso", cov=cov, val=(10 ** level if cov == "conc" else level), title=iso_title(cov, level),
                line=dict(arc=rnd(m.get("arc"), 1), purity=rnd(m.get("purity"), 2), n_near=m.get("n_near"), n_y1=m.get("n_y1"), n_y0=m.get("n_y0"), papers=m.get("n_paper"), prots=m.get("n_prot"), points=S.get("n_line_points")),
                hmc=hm, gif=f"iso_{rid}.gif", poster=f"iso_{rid}_poster.webp", gif_info=gif_info(WEB / f"iso_{rid}.gif"), gif_kb=round((WEB / f"iso_{rid}.gif").stat().st_size / 1e3), notes=notes)


def build_iso():
    S80 = json.load(open(SRC / "rs80_summary.json"))
    S95 = json.load(open(FIGS / "rs95_summary.json"))
    rows, failed, rule = [], [], None
    for cov in ISO_ORDER:
        gif_src = FIGS / f"FigR76_isoline_hmc_{cov}.gif"
        if cov not in S95 or not gif_src.exists():
            continue
        rows.append(iso_row(cov, S80[cov]["contour"]["level"], cov.lower(), S80[cov]["contour"], S95[cov], gif_src, ISO_NOTES.get(cov, [])))
    if (NEW / "iso").is_dir():
        for jf in sorted((NEW / "iso").glob("*.json")):
            if jf.name.startswith("_"):
                continue
            j = json.load(open(jf))
            gif = jf.with_suffix(".gif")
            if not gif.exists():
                continue
            rows.append(iso_row(j["covariate"], j["level"], j["id"], j["meaningful"], j["summary"], gif, list(j.get("notes", [])) + auto_iso_notes(j["meaningful"], j["summary"])))
        cf = NEW / "iso" / "_candidates.json"
        if cf.exists():
            c = json.load(open(cf))
            rule = ("A line is shown when the covariate is more uniform near it than across the data (homogeneity < 1), it is at least 1 SD long, at least 10 records of each label lie within 0.12 SD of it, "
                    "r changes sign along it and at least 5 papers lie near it. The last two conditions were added when the new lines were chosen.")
            failed = [dict(label=x["label"].replace("Iso-pH pH ", "Iso-pH "), why=x["why"].replace("only 1 papers", "only 1 paper")) for x in c["candidates"] if not x["passes"]]
    rows.sort(key=lambda r: (ISO_ORDER.index(r["cov"]), r["val"]))
    return rows, failed, rule


# ------------------------------------------------------------------ 地图点 → 序列
COMBO_TITLE = {"conc 10 µM × tau": "tau region ∩ protein concentration 10 µM", "NaCl 50 mM × tau": "tau region ∩ NaCl 50 mM",
               "conc 10 µM × SARS-CoV-2 N": "SARS-CoV-2 N region ∩ protein concentration 10 µM", "NaCl 50 mM × SARS-CoV-2 N": "SARS-CoV-2 N region ∩ NaCl 50 mM",
               "conc 10 µM × TDP-43": "TDP-43 region ∩ protein concentration 10 µM"}
COMBO_FIRST = list(COMBO_TITLE)                  # the five decoded so far come first; any combo the decoder adds later follows automatically
LINE_LABEL = {"conc": "protein concentration", "NaCl": "NaCl", "pH": "pH", "temp": "temperature", "length": "sequence length", "crowder": "crowder"}


def combo_title(combo, protein):
    if combo in COMBO_TITLE:
        return COMBO_TITLE[combo]
    m = re.match(r"(\w+) (.+?) × (.+)", combo)
    return f"{m.group(3)} region ∩ {LINE_LABEL.get(m.group(1), m.group(1))} {m.group(2)}" if m else combo


def explorer_map():
    """rs90 / rs107 的“标准化 t-SNE”坐标 → explorer 的 D.emb.tsne_full 坐标（与 rs05_build_explorer 同一套归一化）。"""
    co = list(csv.DictReader(open(OUT / "rs_coords.tsv"), delimiter="\t"))
    xs = np.array([float(c["tsne_full_x"]) for c in co])
    ys = np.array([float(c["tsne_full_y"]) for c in co])
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    s = max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.04
    mx, my, sd = xs.mean(), ys.mean(), np.nanstd(np.c_[xs, ys])
    ex, ey = (xs - cx) / s + 0.5, 1 - ((ys - cy) / s + 0.5)

    def f(q1, q2):
        xr, yr = q1 * sd + mx, q2 * sd + my
        return (xr - cx) / s + 0.5, 1 - ((yr - cy) / s + 0.5)
    return f, ex, ey


def parse_alignments():
    """rs107_points_alignments.txt：每个组合的 12×12 替换数矩阵（Biopython 全局比对）。"""
    txt = open(FIGS / "rs107_points_alignments.txt", encoding="utf-8").read()
    out = {}
    for sec in re.split(r"(?m)^=== ", txt)[1:]:
        combo = sec.split("：", 1)[0].strip()
        i = sec.find("(2)")
        j = sec.find("(3)")
        block = sec[i:j] if j > i else sec[i:]
        rows = []
        for line in block.splitlines():
            m = re.match(r"\s*(y[01]P\d)\s+(.*)", line)
            if m:
                rows.append([int(x) for x in re.findall(r"(\d+)\(\s*\d+\)", m.group(2))])
        out[combo] = rows
    return out


def build_points():
    f, ex, ey = explorer_map()
    pt = list(csv.DictReader(open(FIGS / "rs107_points_decoded.tsv"), delimiter="\t"))
    mats = parse_alignments()
    table = list(csv.DictReader(open(OUT / "rs_plot_table.tsv"), delimiter="\t"))
    seen = []
    for r in pt:
        if r["combo"] not in seen:
            seen.append(r["combo"])
    order = [c for c in COMBO_FIRST if c in seen] + [c for c in seen if c not in COMBO_FIRST]
    rows, check, hi = [], [], {}
    for combo in order:
        P = [r for r in pt if r["combo"] == combo]
        names = [r["anchor_name"] for r in P]
        protein = max(set(names), key=names.count)
        if protein not in hi:
            hi[protein] = [i for i, r in enumerate(table) if r["protein_name"] == protein]
        pts = []
        for r in P:
            x, y = f(float(r["q1"]), float(r["q2"]))
            edits = [[e[0], int(e[1:-1]), e[-1]] for e in r["edits"].split(";") if e]
            pts.append(dict(y=1 if r["label"] == "y1" else 0, i=int(r["point"]), x=round(float(x), 5), yy=round(float(y), 5), anchor=int(r["anchor"]), len=int(r["length"]), k=int(r["kstar"]), dsrc=rnd(r["D_src"], 4), dk=rnd(r["D_k"], 4),
                            edits=edits, new=str(r["new"]).lower() in ("true", "1", "yes", "是")))
        H = np.array(hi[protein])
        d_hi = [float(np.min(np.hypot(ex[H] - p["x"], ey[H] - p["yy"]))) for p in pts] if len(H) else [float("nan")]
        d_all = [float(np.min(np.hypot(ex - p["x"], ey - p["yy"]))) for p in pts]
        check.append((combo, round(float(np.median(d_hi)), 4), round(float(np.median(d_all)), 4)))
        rows.append(dict(id=re.sub(r"[^a-z0-9]+", "_", combo.lower()).strip("_"), kind="pts", title=combo_title(combo, protein), protein=protein, hi=protein, points=pts, matrix=mats.get(combo),
                         n_new=sum(1 for p in pts if p["new"]), n_k0=sum(1 for p in pts if p["k"] == 0)))
    print("坐标换算核对（中位数：到该蛋白最近记录的距离 / 到任一记录的距离，explorer 坐标，地图宽 ≈ 1）：", check)
    shutil.copy(FIGS / "rs107_points_alignments.txt", WEB / "points_alignments.txt")
    return rows, hi


def region_table():
    S = json.load(open(FIGS / "rs97_summary.json"))
    regs = ["C1: 22 °C + pH 7.5", "C3: + 150 mM NaCl + 10 µM", "L1: 400 aa + 150 mM NaCl"]
    meth = [("EXACT grid sampling", "Exact grid sampling"), ("LEM Langevin EM + wall", "Langevin + wall"), ("HMC + wall", "HMC + wall"), ("HMC + stiff wall", "HMC + stiff wall")]
    rows = []
    for key, name in meth:
        tv = [[rnd(S[f"{r} | {d}"]["methods"][key]["TV"], 3) for d in ("to_y1", "to_y0")] for r in regs]
        leak = max(S[f"{r} | {d}"]["methods"][key]["leak"] for r in regs for d in ("to_y1", "to_y0"))
        rows.append(dict(method=name, tv=tv, leak=rnd(leak, 3)))
    return dict(caption="Distance to the exact distribution (TV, guided to y = 1 / to y = 0; the sampling-noise floor is 0.05–0.15) on three regions: C1 = 22 °C and pH 7.5; C3 = C1 plus 150 mM NaCl and 10 µM; L1 = 400 aa and 150 mM NaCl. "
                        "MCMC chains start at random points inside the region; “largest leak” is the largest share of chains that ended outside it.",
                regions=["C1", "C3", "L1"], rows=rows)


def results_block(iso, yc, pts):
    """Intro, description and reading notes of the four main results; every number comes from the data."""
    out = R134.build_results()
    I = {m["id"]: m for m in out["maps"]}
    isor = next(r for r in iso if r["id"] == "nacl")
    ycr = next(r for r in yc if r["id"] == "physio")
    iso_m, reg_m, yc_m, pt_m = I["iso_nacl150"], I["region_c3"], I["yc_physio"], I["pts_tau10"]
    f_iso, f_reg, f_yc, f_pt = iso_m["facts"], reg_m["facts"], yc_m["facts"], pt_m["facts"]
    st = {g["id"]: g["stats"] for g in iso_m["groups"]}
    iso_m["desc"] = [f"Chains start on the NaCl = 150 mM iso-line (60 equally spaced starts × 8 chains) and are guided to y = 1 (purple) or y = 0 (cyan) by 1-D arc-length HMC ({f_iso['proposals']} proposals, w = {f_iso['w']:g}). "
                     f"The line is {isor['line']['arc']} SD long in {f_iso['pieces']} pieces, with {fmtn(isor['line']['n_near'])} records near it from {isor['line']['papers']} papers and {isor['line']['prots']} proteins."]
    iso_m["reading"] = [f"Guided ends sit on stretches of higher r (to y = 1: median r {f_iso['r_end_median'][0]:+.2f}) or lower r (to y = 0: median r {f_iso['r_end_median'][1]:+.2f}).".replace("-", "−"),
                        f"The proteins next to the two kinds of ends largely coincide ({round(100 * f_iso['protein_overlap'])}% shared), so a large part of the y difference along an iso-line is not a sequence effect: it comes from other conditions or from the source paper.",
                        f"Sampler checks: TV to the exact distribution {st['to_y1']['tv']:.2f} / {st['to_y0']['tv']:.2f}, acceptance {st['to_y1']['acc']:.2f} / {st['to_y0']['acc']:.2f}."]
    reg_m["desc"] = [f"{f_reg['samples']} independent exact samples per direction inside the region where all four constraints hold (22 ± 4 °C, pH 7.5, 150 mM NaCl, 10 µM): {f_reg['cells']} grid cells ({100 * f_reg['area_frac']:.1f}% of the grid) in {f_reg['pieces']} pieces, "
                     f"with {f_reg['records']} records inside ({f_reg['n_y1']} y = 1, {f_reg['n_y0']} y = 0) from {f_reg['papers']} papers and {f_reg['proteins']} proteins."]
    reg_m["reading"] = [f"Samples guided to y = 1 sit at high r (median {f_reg['r_median'][0]:+.1f}), those guided to y = 0 near {f_reg['r_median'][1]:+.1f}: the field separates the two inside the region.".replace("-", "−"),
                        "There is no start or end: exact samples are independent. Hover a sample for the records next to it (the reference) and the field's estimate.",
                        "Decoding to sequences is shown for points on a line (the last result)."]
    sy = {g["id"]: g["stats"] for g in yc_m["groups"]}
    yc_m["desc"] = [f"Chains start at the records of one label in the physiological cell ({f_yc['desc'].replace(' (cross-paper)', '')}; {f_yc['n1']} y = 1 and {f_yc['n0']} y = 0 records, {f_yc['seq']} sequences, {f_yc['papers']} papers) "
                    f"and are guided to the other label for {fmtn(f_yc['steps'])} steps (Langevin, T = {f_yc['T']:g}, w = {f_yc['w']:g}; {f_yc['K']} chains per record, 0.03 SD jitter)."]
    yc_m["reading"] = [f"Guided ends move in r in the intended direction (median Δr {sy['0to1']['dr']:+.2f} for y = 0 → 1, {sy['1to0']['dr']:+.2f} for y = 1 → 0), but they end within 0.10 SD of a record of the target label no more often than unguided chains "
                       f"({round(100 * sy['0to1']['arr'])}% against {round(100 * sy['0to1']['arr_none'])}%, {round(100 * sy['1to0']['arr'])}% against {round(100 * sy['1to0']['arr_none'])}%).".replace("-", "−"),
                       f"The two labels interleave on the map (mean r of the two classes differs by {abs(ycr['map']['dr']):.2f}) and the field has no condition c as input, so the cell only decides where the chains start.",
                       "Switch on “unguided control ends” to see where chains without guidance end."]
    fs = out["finches"]
    mp, rp = fs["mpipi_full"]["abs_d_eps_decoded_median"], fs["mpipi_full"]["abs_d_eps_random_median"]
    pt_m["desc"] = ["Twelve points of the exact samples of (tau region ∩ 10 µM iso-line), 6 guided to y = 1 and 6 to y = 0, decoded to sequences through the records around them: an anchor sequence plus ESMC masked-language-model substitutions "
                    "that move the ESMC embedding toward the target. Start = the closest record of the anchor, end = the map point and the answer."]
    pt_m["reading"] = [f"{f_pt['n_k0']} of the 12 answers are the anchor itself (k* = 0); {f_pt['n_new']} carry 2–12 substitutions.",
                       f"FINCHES, over all {fs['n_with_edits']} decoded variants of the five map-point examples: median |Δε| (Mpipi, full length) {mp:.2f} for the decoded answers against {rp:.2f} for random substitutions at the same sites, "
                       f"and {round(100 * fs['mpipi_full']['frac_stronger_PS'])}% become more attractive: no consistent direction.",
                       "Positions are proposed at random and kept when they move the embedding toward the target; they are not chosen for function."]
    n_more = [len(yc) - 1, len(iso) - 1, len(pts) - 1]
    return dict(intro="Four results, each an interactive map on the t-SNE of the records. Hover (or tap) a point for its start and end, the records next to it (protein, paper, label, conditions), the score field's estimate and, for the decoded points, "
                      "the sequence change and the FINCHES prediction. Scroll to zoom, drag to pan, click to pin a point; the legend toggles the groups.",
                maps=out["maps"], more_text=f"{n_more[0]} more (y, c) sets, {n_more[1]} more iso-lines, {n_more[2]} more map-point combinations")


def main():
    SRC.mkdir(parents=True, exist_ok=True)
    if not (SRC / "rs80_summary.json").exists():
        sys.exit("缺 guidance_assets/src/rs80_summary.json（从 WSL ~/rs7/rs80_summary.json 读来的等值线评估）")
    data = {"text": {k: v for k, v in TEXT.items() if k not in ("field", "iso")}, "blocks": {}}
    f = TEXT["field"]
    figs = []
    for g in f["figs"]:
        size = to_webp(g["src"], WEB / f"{g['id']}.webp", maxw=1700)
        figs.append(dict(src=f"{g['id']}.webp", cap=g["cap"], **size))
    data["blocks"]["field"] = dict(title=f["title"], paras=f["paras"], facts=f["facts"], figs=figs)
    yc = build_yc()
    iso, failed, rule = build_iso()
    # 区间由当前显示的例子算出来，页面文字里引用
    lo = min(min(r["dir"][a]["toward"] for a in ("0to1", "1to0")) for r in yc)
    hi_ = max(max(r["dir"][a]["toward"] for a in ("0to1", "1to0")) for r in yc)
    lo0 = min(min(r["dir"][a]["toward_none"] for a in ("0to1", "1to0")) for r in yc)
    hi0 = max(max(r["dir"][a]["toward_none"] for a in ("0to1", "1to0")) for r in yc)
    arr = max(max(r["dir"][a]["arr"] for a in ("0to1", "1to0")) for r in yc)
    yc_t = TEXT["yc"]
    yc_limits = [
        f"Guided chains move the right way ({round(100 * lo)}–{round(100 * hi_)}% of them toward the target label, against {round(100 * lo0)}–{round(100 * hi0)}% without guidance), "
        f"but they do not end next to records of the opposite label in the same cell more often than unguided chains (at most {round(100 * arr)}% within 0.10 SD).",
        "Why: the field has no condition c as input, so “from this (y, c) set” only chooses where the chains start; the guidance itself is the global y field. Fixing c needs a model that takes c as input.",
    ]
    iso_t = TEXT["iso"]
    iso_fig = to_webp(iso_t["fig"]["src"], WEB / f"{iso_t['fig']['id']}.webp", maxw=1700)
    data["blocks"]["sampling"] = dict(title="Guided sampling", groups=[
        dict(id="yc", title=yc_t["title"], intro=yc_t["intro"], limits=yc_limits, rows=yc),
        dict(id="iso", title=iso_t["title"], intro=iso_t["intro"], fig=dict(src=f"{iso_t['fig']['id']}.webp", cap=iso_t["fig"]["cap"], **iso_fig), limits=iso_t["recommend"], rows=iso, failed=failed, rule=rule),
    ])
    pts, hi = build_points()
    MT = dict(iso=dict(METHODS["iso"]), region=dict(METHODS["region"]))
    f2 = MT["iso"].pop("fig"); MT["iso"]["fig"] = dict(src=f"{f2['id']}.webp", cap=f2["cap"], **to_webp(f2["src"], WEB / f"{f2['id']}.webp", maxw=1700))
    MT["region"]["table"] = region_table()
    data["blocks"]["methods"] = MT
    data["results"] = results_block(iso, yc, pts)
    p = TEXT["pts"]
    data["blocks"]["points"] = dict(title=p["title"], intro=p["intro"], limits=p["limits"], rows=pts, align_file="points_alignments.txt", highlight={k: v for k, v in hi.items()})
    json.dump(data, open(PAGE / "guidance_data.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    files = sorted(WEB.iterdir())
    tot = sum(x.stat().st_size for x in files)
    json.dump({x.name: x.stat().st_size for x in files}, open(WEB / "MANIFEST.json", "w"), indent=0)
    print(f"例子：(y,c) {len(yc)} 个，等值线 {len(iso)} 条，地图点组合 {len(pts)} 个；素材 {len(files)} 个文件，共 {tot / 1e6:.1f} MB；json {(PAGE / 'guidance_data.json').stat().st_size / 1e3:.0f} KB")
    for x in files:
        if x.suffix == ".gif":
            print(f"  {x.stat().st_size / 1e6:5.2f} MB  {x.name}")


if __name__ == "__main__":
    main()
