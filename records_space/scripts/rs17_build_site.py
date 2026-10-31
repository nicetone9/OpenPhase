#!/usr/bin/env python3
"""rs17（本机，纯标准库 + cwebp）：搭 GitHub Pages 站点的暂存目录 site/，结构同仓库根目录：
  site/index.html                       首页（目录页）：主图、P 与 N+U 两张卡片、两张系统图（带可点热区）
  site/assets/                          主图、logo、两张系统图
  site/records_space/index.html         P：records 空间 explorer
  site/records_space/negatives/index.html   N+U：负例与未标注
热区坐标是图片原始像素；链接用 explorer 的地址栏参数（#na= #csat= #hl= #q= #lens=）。
用法：python3 -I scripts/rs17_build_site.py   （加 --debug 只是提醒：打开首页时在网址后加 ?debug 会显示所有热区）"""
import os
import shutil
import subprocess

ROOT = "/Users/lynn/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Projects/LLPS/condition_llps/records_space_2026-10-29"
BASE = "/Users/lynn/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Projects/LLPS/condition_llps"
THEME = f"{BASE}/output/imagegen/openphase-theme-2026-10-29"
SESS_IMG = "/private/tmp/claude-501/-Users-lynn-Library-CloudStorage-OneDrive-UniversityofPittsburgh-Projects-LLPS-condition-llps/fa0578ff-5717-42e6-8ecd-b4fd103048ad/images"
SITE = f"{ROOT}/site"
FIGS = f"{ROOT}/home_page/figs"
os.makedirs(FIGS, exist_ok=True)

# ---- 来源图：用户发来的两张系统图存进项目；主图 = v5，logo = 透明 wordmark（都在项目的 imagegen 目录里）
for src, name in ((f"{SESS_IMG}/2.webp", "fig-multi-protein-systems.webp"), (f"{SESS_IMG}/3.webp", "fig-protein1-systems.webp")):
    if os.path.exists(src):
        shutil.copy(src, f"{FIGS}/{name}")
assert os.path.exists(f"{FIGS}/fig-multi-protein-systems.webp") and os.path.exists(f"{FIGS}/fig-protein1-systems.webp")
os.makedirs(f"{SITE}/assets", exist_ok=True)
for name in ("fig-multi-protein-systems.webp", "fig-protein1-systems.webp"):
    shutil.copy(f"{FIGS}/{name}", f"{SITE}/assets/{name}")
subprocess.run(["cwebp", "-quiet", "-q", "88", f"{THEME}/openphase-system-conditions-v5.png", "-o", f"{SITE}/assets/openphase-main.webp"], check=True)
subprocess.run(["cwebp", "-quiet", "-q", "92", "-alpha_q", "100", f"{THEME}/openphase-wordmark-transparent.png", "-o", f"{SITE}/assets/openphase-wordmark.webp"], check=True)

# ---- 热区
WB, HB = 2000, 1793          # protein(1) 图
WA, HA = 2000, 1475          # 多蛋白系统图
P = "records_space/"
HOT_B = [   # (形状, 几何, 链接, 提示)
    ("rect", (52, 1158, 506, 1642), P + "#na=none&csat=hide", "protein(1): protein only, 7,490 records"),
    ("rect", (533, 1158, 987, 1642), P + "#na=RNA&csat=hide", "protein(1) + RNA, 2,511 records"),
    ("rect", (1013, 1158, 1467, 1642), P + "#na=DNA&csat=hide", "protein(1) + DNA, 662 records"),
    ("rect", (1493, 1158, 1947, 1642), P + "#csat=only", "protein(1), Csat pool, 776 records"),
    ("rect", (1335, 150, 1870, 700), P + "#protein=FUS", "FUS: 836 records, 46 papers"),
    ("circle", (605, 365, 172), P + "#hl=llpsdb&na=none&csat=hide", "LLPSDB v2, protein only"),
    ("circle", (868, 408, 75), P + "#hl=phasepdb3&na=none", "PhaSepDB 3 (pilot), protein only"),
    ("circle", (800, 570, 95), P + "#hl=llpsense&na=none", "LLPSense"),
    ("circle", (980, 745, 125), P + "#hl=llpsdb&na=RNA", "LLPSDB v2, + RNA"),
    ("circle", (1170, 745, 55), P + "#hl=phasepdb3&na=RNA", "PhaSepDB 3 (pilot), + RNA"),
    ("circle", (1150, 922, 115), P + "#hl=rnapsec", "RNAPSEC"),
    ("circle", (600, 862, 100), P + "#hl=llpsdb&na=DNA", "LLPSDB v2, + DNA"),
    ("circle", (710, 972, 52), P + "#hl=phasepdb3&na=DNA", "PhaSepDB 3 (pilot), + DNA"),
    ("circle", (1172, 290, 85), P + "#hl=ausserwoger&csat=only", "Ausserwöger 2025, Csat"),
    ("circle", (1195, 465, 68), P + "#hl=bremer&csat=only", "Bremer 2022, Csat"),
]
HOT_A = [
    ("rect", (30, 165, 520, 720), "protein(2): 953 systems, 4,406 records"),
    ("rect", (520, 165, 1000, 720), "protein(3): 114 systems, 547 records"),
    ("rect", (1000, 165, 1480, 720), "protein(≥4): 47 systems, 167 records"),
    ("rect", (1480, 165, 1965, 720), "protein(2) + RNA: 88 systems, 313 records"),
    ("rect", (30, 765, 520, 1320), "protein(2) + DNA: 24 systems, 38 records"),
    ("rect", (520, 765, 1000, 1320), "protein(3) + NA: 21 systems, 156 records"),
    ("rect", (1000, 765, 1480, 1320), "With complex: 15 systems, 57 records"),
]
CHIPS_B = [("Protein only", P + "#na=none&csat=hide"), ("+ RNA", P + "#na=RNA&csat=hide"), ("+ DNA", P + "#na=DNA&csat=hide"), ("Csat pool", P + "#csat=only"), ("FUS", P + "#protein=FUS"),
           ("LLPSDB v2", P + "#hl=llpsdb"), ("LLPSense", P + "#hl=llpsense"), ("RNAPSEC", P + "#hl=rnapsec"), ("PhaSepDB 3", P + "#hl=phasepdb3"),
           ("Bremer 2022", P + "#hl=bremer&csat=only"), ("Ausserwöger 2025", P + "#hl=ausserwoger&csat=only")]


def box(shape, g, W, H):
    if shape == "rect":
        x0, y0, x1, y1 = g
        return x0 / W * 100, y0 / H * 100, (x1 - x0) / W * 100, (y1 - y0) / H * 100
    cx, cy, r = g
    return (cx - r) / W * 100, (cy - r) / H * 100, 2 * r / W * 100, 2 * r / H * 100


def hs(shape, g, W, H, href, label):
    l, t, w, h = box(shape, g, W, H)
    st = f"left:{l:.2f}%;top:{t:.2f}%;width:{w:.2f}%;height:{h:.2f}%"
    cls = "hs round" if shape == "circle" else "hs"
    lab = label.replace('"', "&quot;")
    if href:
        return f'<a class="{cls}" style="{st}" href="{href}" data-label="{lab} → open" aria-label="{lab}"></a>'
    return f'<span class="{cls} off" style="{st}" data-label="{lab} · not built yet" aria-label="{lab}, not built yet"></span>'


hs_b = "\n      ".join(hs(s, g, WB, HB, h, lab) for s, g, h, lab in HOT_B)
hs_a = "\n      ".join(hs(s, g, WA, HA, None, lab) for s, g, lab in HOT_A)
chips = "".join(f'<a class="chip" href="{h}">{n}</a>' for n, h in CHIPS_B)
tpl = open(f"{ROOT}/home_page/home_template.html", encoding="utf-8").read()
for ph, val in (("{{HS_B}}", hs_b), ("{{HS_A}}", hs_a), ("{{CHIPS_B}}", chips), ("{{LOGO_W}}", "2172"), ("{{LOGO_H}}", "724")):
    assert tpl.count(ph) == 1, ph
    tpl = tpl.replace(ph, val)
open(f"{SITE}/index.html", "w", encoding="utf-8").write(tpl)

# ---- 两个页面
os.makedirs(f"{SITE}/records_space/negatives", exist_ok=True)
shutil.copy(f"{ROOT}/explorer/records_space_explorer.html", f"{SITE}/records_space/index.html")
shutil.copy(f"{ROOT}/pu_page/index.html", f"{SITE}/records_space/negatives/index.html")
for rel in ("index.html", "assets/openphase-main.webp", "assets/openphase-wordmark.webp", "assets/fig-protein1-systems.webp", "assets/fig-multi-protein-systems.webp", "records_space/index.html", "records_space/negatives/index.html"):
    print(f"{os.path.getsize(f'{SITE}/{rel}') / 1e3:9.0f} KB  {rel}")
