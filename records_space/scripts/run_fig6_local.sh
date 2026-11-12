#!/bin/bash
# FigR6 (this machine, figure rendering only): build a throw-away environment, render the figure, run the nature-figure audits.
# Rendering figures locally is allowed (11/03); clustering, embeddings and learning curves still run on Colab / Lane only.
# Needs: python3, network for pip. Output: $OUT/FigR6_coloring_gallery.{png,pdf,svg,tiff} + audits. Usage: scripts/run_fig6_local.sh [work_dir]
set -euo pipefail
ROOT="/Users/lynn/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Projects/LLPS/condition_llps"
RS="$ROOT/records_space_2026-10-29"
W="${1:-${TMPDIR:-/tmp}/fig6_work}"
mkdir -p "$W/skill_scripts" "$W/source_data" "$W/out"
[ -x "$W/venv/bin/python" ] || { python3 -m venv "$W/venv"; "$W/venv/bin/pip" install --quiet matplotlib numpy scipy pandas pillow pymupdf; }
cp "$HOME/.claude/skills/nature-figure/scripts/"*.py "$W/skill_scripts/"
cp "$ROOT/world_and_cluster_maps_2026-10-27/codex_task/work/common_style.py" "$W/"
cat "$RS/out/rs_plot_table.tsv" > "$W/source_data/rs_plot_table.tsv"; cat "$RS/out/rs_coords.tsv" > "$W/source_data/rs_coords.tsv"
cp "$RS/scripts/rs20_coloring_figure.py" "$W/figR6_v2.py"
cd "$W"
"$W/venv/bin/python" figR6_v2.py --lang en --out out
"$W/venv/bin/python" skill_scripts/audit_pdf_text.py out/FigR6_v2.pdf | grep -E "minimum found|below minimum|verdict"
"$W/venv/bin/python" skill_scripts/audit_figure_collisions.py out/FigR6_v2.pdf --json-out out/FigR6_v2.collision-audit.json --overlay-pdf out/FigR6_v2.collision-audit.pdf | grep -E "^\[|summary|verdict"
"$W/venv/bin/python" skill_scripts/validate_figure.py figR6_v2.py | grep -E "^\[(FAIL|WARN)\]|summary|verdict"
echo "figures and audits are in $W/out (copy FigR6_v2.* to $RS/figs/FigR6_coloring_gallery.* when satisfied)"
