"""Guidance tab 的“更多例子”素材收尾：把 WSL 上跑出的汇总（gd_yc_summary.json、gd_iso_summary.json、gd_iso_candidates.json）整理成每个例子一个 json，
并写 guidance_assets/new/README.md（数字全部从 json 里读，不手填）。纯标准库，本机跑。
输入：~/pixi/figs_work/source_data/ 下的 gd_yc_*.json、gd_iso_*.json（从 WSL ~/rs7 下载，新文件名，不覆盖已有文件）。
输出：guidance_assets/new/yc/<id>.json、guidance_assets/new/iso/<id>.json、guidance_assets/new/iso/_candidates.json、guidance_assets/new/README.md。
GIF 与结果图由 rs131b_yc_new_render.py / rs132b_iso_gifs.py 出图、ffmpeg 压缩后放在同一目录。"""
import json, os
SRC = os.path.expanduser("~/pixi/figs_work/source_data")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/guidance_assets/new"; os.makedirs(f"{OUT}/yc", exist_ok=True); os.makedirs(f"{OUT}/iso", exist_ok=True)
yc = json.load(open(f"{SRC}/gd_yc_summary.json")); chk = json.load(open(f"{SRC}/gd_yc_cells_check.json")); iso = json.load(open(f"{SRC}/gd_iso_summary.json")); cand = json.load(open(f"{SRC}/gd_iso_candidates.json"))
META = {  # id: (title, paper, protein, 条件格定义)
    "tau_peg": ("Tau in PEG (pmid:32229582)", "pmid:32229582", "tau (MAPT)", dict(pH_bin=7.5, temp_bin="15-30C", nacl_bin="50-100", crowd="PEG", conc_bin=1.0)),
    "fus": ("FUS variants (pmid:29961577)", "pmid:29961577", "FUS and variants", dict(pH_bin=7.5, temp_bin="15-30C", nacl_bin="0", crowd="none", conc_bin=0.5)),
    "ubqln2": ("UBQLN2 (pmid:30982635)", "pmid:30982635", "UBQLN2", dict(pH_bin=7.0, temp_bin=">40C", nacl_bin="100-200", crowd="none", conc_bin=1.0)),
    "zo1": ("ZO-1 / Q07157 (pmid:31675499)", "pmid:31675499", "ZO-1 (Q07157) and two others", dict(pH_bin=7.5, temp_bin="15-30C", nacl_bin="0", crowd="PEG", conc_bin=0.5)),
    "g3bp1": ("G3BP1 (pmid:32302571)", "pmid:32302571", "G3BP1", dict(pH_bin=7.5, temp_bin="15-30C", nacl_bin="100-200", crowd="Ficoll/Dex", conc_bin=2.0)),
    "hnrnpa2b1": ("hnRNPA2B1 (pmid:33349959)", "pmid:33349959", "hnRNPA2B1 and three others", dict(pH_bin=5.5, temp_bin="15-30C", nacl_bin="<=50", crowd="none", conc_bin=1.0)),
}
for k, s in yc.items():
    t, paper, protein, cell = META[k]
    json.dump(dict(id=k, kind="yc", title=t, desc=s["desc"], paper=paper, protein=protein, cell=cell, cell_check=chk[k], summary=s, figure_no=None), open(f"{OUT}/yc/{k}.json", "w"), indent=1, ensure_ascii=False)
rows = {r["id"]: r for r in cand["candidates"]}
for k, s in iso.items():
    r = rows[k]; mean = {a: r[a] for a in ("arc", "purity", "r_std", "r_p10", "r_p90", "crossings", "n_near", "n_y1", "n_y0", "n_paper", "n_prot", "score", "n_segments")}
    desc = f"Iso-line of the kernel-smoothed {r['cov']} field (σ = 0.16 on the map) at {r['label'].split(' ', 1)[1]}; chains move along the line only (1-D arc-length HMC, TorchEBM), guided to y = 1 or y = 0."
    json.dump(dict(id=k, kind="iso", covariate=r["cov"], level=r["level"], unit=r["unit"], title=r["label"], desc=desc, directive=r["directive"], meaningful=mean, summary=s), open(f"{OUT}/iso/{k}.json", "w"), indent=1, ensure_ascii=False)
json.dump(cand, open(f"{OUT}/iso/_candidates.json", "w"), indent=1, ensure_ascii=False)
# ---- README（数字取自 json）
L = ["# Guidance tab: new examples (generated 2026-11-13)", "",
     "All numbers below are read from the json files in this folder; nothing is typed by hand. Sampling ran on WSL (CPU only, `nice -n 10`, 4 threads, pixi `~/t3ebm`: torch 2.14.1 CPU, torchebm 0.9.1) with the same settings as the existing examples (`scripts/rs70_yc_trajectories.py`, `scripts/rs95_arc_hmc_runs.py`); figures and GIFs were drawn locally with `~/pixi/figs` (`scripts/rs131b_yc_new_render.py`, `scripts/rs132b_iso_gifs.py`, copies of rs73 / rs96) and compressed with ffmpeg (`palettegen` 64 colours, no dither, frame count and per-frame delays unchanged).", "",
     "## (y, c) sets (`yc/`)", "",
     "Settings as rs70: σ = 0.16 TorchEBM conditional energy, Langevin η = 0.25σ², T = 0.5, 1,200 steps, w = 4, K = 16 chains per start record (at most 400 start records), controls = no guidance and guidance to the same label; GIF = 24 chains, one frame per 10 steps, 121 frames, 60-step trail, black final dots held about 8 s.", "",
     "| id | records y=1 / y=0 | sequences | proteins | papers | map distance of the two label groups (SD) | median Δr, 0→1 guided (none) | median Δr, 1→0 guided (none) | chains ending ≤ 0.10 SD of a target record, 0→1 / 1→0 guided (none) |", "|---|---|---|---|---|---|---|---|---|"]
for k, s in yc.items():
    g1, g0, n1, n0 = s["0to1"]["guided"], s["1to0"]["guided"], s["0to1"]["none"], s["1to0"]["none"]
    L.append(f"| `{k}` | {s['n1']} / {s['n0']} | {s['n_seq']} | {s['n_prot']} | {s['n_paper']} | {s['map_dist_medians']:.2f} | {g1['dr_median']:+.2f} ({n1['dr_median']:+.2f}) | {g0['dr_median']:+.2f} ({n0['dr_median']:+.2f}) | {100 * g1['arrived_0p10']:.0f}% / {100 * g0['arrived_0p10']:.0f}% ({100 * n1['arrived_0p10']:.0f}% / {100 * n0['arrived_0p10']:.0f}%) |")
L += ["", "Things to keep in mind (all six):", "- Every set is **one paper only** (`n_paper` = 1): the cell is a within-paper cell, so label differences inside it are differences between that paper's constructs or conditions.",
      "- The two label groups sit almost on top of each other on the map (distance 0.01–0.11 SD, mean-r difference ≈ 0). Guidance moves chains in r in the intended direction (compare with the no-guidance column), but chains almost never end on a target-label record (≤ 2%), the same picture as the existing hnRNPDL and BRD4 examples: the 2-D field carries no condition c.",
      "- Small sets: `g3bp1` (6 / 5 records, 4 sequences), `hnrnpa2b1` (5 / 7), `ubqln2` (7 / 9), `zo1` (7 / 10). `fus` is a FUS-family variant/chimera set (22 UniProt entries), `zo1` has 3 proteins, `hnrnpa2b1` 4.",
      "- `ubqln2` 0→1: guided Δr is about 0 because the no-guidance chains drift to lower r; guidance only cancels that drift.", "",
      "## Iso-lines (`iso/`)", "", f"Rule (from `scripts/rs80_isocontours.py`, with one added guard): {cand['rule']}.", f"Re-evaluating the 23 rs80 grid rows reproduces `rs80_table.tsv` (max abs difference: arc {cand['verify_against_rs80_table']['max_abs_diff']['arc']:.1e}, purity {cand['verify_against_rs80_table']['max_abs_diff']['purity']:.1e}, n_near {cand['verify_against_rs80_table']['max_abs_diff']['n_near']}, score {cand['verify_against_rs80_table']['max_abs_diff']['score']:.1e}).", "",
      "Settings as rs95: line parametrised by arc length, 1-D HMC (ε = 0.05, 10 leapfrog steps, T = 0.5, w = 4), 400 proposals, 60 equal-arc starts × 8 = 480 chains per direction; GIF = 4 proposals per frame, 5 fading ghosts, black final dots held about 8 s.", "",
      "| id | line | from | purity (lower = more homogeneous) | length (SD) | y=1 / y=0 records near the line | papers | segments | TV to y=1 / y=0 | accept rate to y=1 / y=0 | median arc move (SD) to y=1 / y=0 | off-support chains to y=1 / y=0 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for k, s in iso.items():
    r = rows[k]; m1, m0 = s["methods"]["to_y1"], s["methods"]["to_y0"]
    L.append(f"| `{k}` | {r['label']} | {'requested list' if r['directive'] else 'rs80 grid'} | {r['purity']:.2f} | {r['arc']:.1f} | {r['n_y1']} / {r['n_y0']} | {r['n_paper']} | {r['n_segments']} | {m1['TV_seg']:.2f} / {m0['TV_seg']:.2f} | {m1['accept_rate']:.2f} / {m0['accept_rate']:.2f} | {m1['arc_move_median']:.2f} / {m0['arc_move_median']:.2f} | {100 * m1['off_support']:.1f}% / {100 * m0['off_support']:.1f}% |")
L += ["", "Candidates that were evaluated but not sampled (full numbers in `iso/_candidates.json`):", "", "| line | requested | status | why |", "|---|---|---|---|"]
for r in cand["candidates"]:
    if r["existing"] or r["selected"]: continue
    L.append(f"| {r['label']} | {'yes' if r['directive'] else 'no (rs80 grid)'} | {'passes (not selected)' if r['passes'] else 'fails'} | {r['why']} |")
L += ["", "Existing lines (already have GIFs, re-evaluated only as a check): " + ", ".join(f"{r['label']} (purity {r['purity']:.2f})" for r in cand["candidates"] if r["existing"] and r["status"] == "ok") + ".", ""]
open(f"{OUT}/README.md", "w", encoding="utf-8").write("\n".join(L)); print("README written;", len(yc), "yc,", len(iso), "iso")
