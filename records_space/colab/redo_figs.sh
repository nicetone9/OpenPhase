#!/bin/zsh
# 新开一个 CPU 会话，只重渲静态图（rs04），拉回图与绘图表后停掉会话。用法：redo_figs.sh <会话名>
setopt nullglob
S=$1
T="$(cd "$(dirname "$0")/.." && pwd)"
timeout 240 colab new -s $S 2>&1 | grep -v colab.dev | tail -1
timeout 120 colab restart-kernel -s $S 2>&1 | grep -v colab.dev | tail -0
cat > /tmp/mk_$S.py <<'PY'
import os
for d in ('/content/rs/data', '/content/rs/code', '/content/rs/out', '/content/rs/figs'):
    os.makedirs(d, exist_ok=True)
print('dirs ok')
PY
timeout 120 colab exec -s $S --timeout 60 -f /tmp/mk_$S.py 2>&1 | grep -v colab.dev | tail -1
timeout 180 colab upload -s $S "$T/data/rs_records.tsv" /content/rs/data/rs_records.tsv 2>&1 | grep -v colab.dev | tail -0
for f in rslib.py rs04_figs.py; do timeout 180 colab upload -s $S "$T/colab/$f" "/content/rs/code/$f" 2>&1 | grep -v colab.dev | tail -0; done
for f in rs_coords.tsv missing_flags.tsv diagnostics.json diagnostics_extra.json; do timeout 180 colab upload -s $S "$T/out/$f" "/content/rs/out/$f" 2>&1 | grep -v colab.dev | tail -0; done
(timeout 600 colab exec -s $S --timeout 540 -f "$T/colab/drv_figs2.py" > /tmp/rs_redo2_exec.log 2>&1 &)
for i in $(seq 1 80); do
  timeout 30 colab download -s $S /content/rs/out/progress.log /tmp/rs_progress_$S.log >/dev/null 2>&1
  grep -q "FIGS3_FINISHED" /tmp/rs_progress_$S.log 2>/dev/null && break
  sleep 6
done
tail -3 /tmp/rs_progress_$S.log
for f in FigR1_sequence_views_umap FigR2_condition_views_umap FigR3_readout_provenance_missingness_umap FigR1_sequence_views_tsne FigR2_condition_views_tsne FigR3_readout_provenance_missingness_tsne FigR4_lens_comparison FigR5_diagnostics; do
  for e in png pdf; do timeout 120 colab download -s $S /content/rs/figs/$f.$e "$T/figs/$f.$e" 2>&1 | grep -v colab.dev | tail -0; done
done
timeout 120 colab download -s $S /content/rs/out/rs_plot_table.tsv "$T/out/rs_plot_table.tsv" 2>&1 | grep -v colab.dev | tail -0
timeout 120 colab stop -s $S 2>&1 | grep -v colab.dev | tail -1
echo REDO_FIGS_ALL_DONE
