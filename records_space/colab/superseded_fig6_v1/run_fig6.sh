#!/bin/zsh
# 在 Colab CPU 会话上画 rs18 的代表性染色图并拉回结果。用法：run_fig6.sh <会话名>   （环境变量 STOP=1：画完停掉会话；默认保留，方便改图重跑）
setopt nullglob
S=$1
T="$(cd "$(dirname "$0")/.." && pwd)"
if ! colab sessions 2>/dev/null | grep -q "\[$S\]"; then
  timeout 300 colab new -s $S 2>&1 | grep -v colab.dev | tail -1
  timeout 120 colab restart-kernel -s $S >/dev/null 2>&1
fi
echo "import os;[os.makedirs(d,exist_ok=True) for d in ('/content/rs6/data','/content/rs6/code','/content/rs6/out','/content/rs6/figs')]" > /tmp/mk6_$S.py; timeout 120 colab exec -s $S --timeout 60 -f /tmp/mk6_$S.py 2>&1 | tail -0
if ! colab ls -s $S /content/rs6/data 2>/dev/null | grep -q rs_coords.tsv; then
  timeout 300 colab upload -s $S "$T/out/rs_plot_table.tsv" /content/rs6/data/rs_plot_table.tsv 2>&1 | grep -v colab.dev | tail -0
  timeout 300 colab upload -s $S "$T/out/rs_coords.tsv" /content/rs6/data/rs_coords.tsv 2>&1 | grep -v colab.dev | tail -0
fi
for f in rs18_coloring_figure.py drv_fig6.py; do timeout 180 colab upload -s $S "$T/colab/$f" "/content/rs6/code/$f" 2>&1 | grep -v colab.dev | tail -0; done
echo "" > /tmp/empty6.txt; timeout 60 colab upload -s $S /tmp/empty6.txt /content/rs6/out/progress.log >/dev/null 2>&1
(timeout 900 colab exec -s $S --timeout 840 -f "$T/colab/drv_fig6.py" > /tmp/rs6_exec.log 2>&1 &)
for i in $(seq 1 120); do
  timeout 30 colab download -s $S /content/rs6/out/progress.log /tmp/rs6_progress.log >/dev/null 2>&1
  grep -qE "DRV_END" /tmp/rs6_progress.log 2>/dev/null && break
  sleep 5
done
tail -12 /tmp/rs6_progress.log
mkdir -p $T/figs
for e in png pdf svg; do timeout 180 colab download -s $S /content/rs6/figs/FigR6_coloring_gallery.$e "$T/figs/FigR6_coloring_gallery.$e" 2>&1 | grep -v colab.dev | tail -0; done
timeout 120 colab download -s $S /content/rs6/out/coloring_stats.csv "$T/out/coloring_stats.csv" 2>&1 | grep -v colab.dev | tail -0
[ "$STOP" = "1" ] && { timeout 120 colab stop -s $S 2>&1 | grep -v colab.dev | tail -1; }
echo FIG6_RUN_DONE
