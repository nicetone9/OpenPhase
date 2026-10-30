#!/bin/zsh
# 新开一个 CPU 会话，导出"试新实验"面板所需数据，拉回本机后停掉会话。用法：run_export.sh <会话名>
setopt nullglob
S=$1
T="$(cd "$(dirname "$0")/.." && pwd)"
TASK2="$T/../task2_single_component_condition"
timeout 240 colab new -s $S 2>&1 | grep -v colab.dev | tail -1
timeout 120 colab restart-kernel -s $S 2>&1 | grep -v colab.dev | tail -0
cat > /tmp/mk_$S.py <<'PY'
import os
for d in ('/content/rs/data', '/content/rs/code', '/content/rs/out', '/content/rs/export'):
    os.makedirs(d, exist_ok=True)
print('dirs ok')
PY
timeout 120 colab exec -s $S --timeout 60 -f /tmp/mk_$S.py 2>&1 | grep -v colab.dev | tail -1
for f in rs_records.tsv rs_sequences.tsv; do timeout 180 colab upload -s $S "$T/data/$f" "/content/rs/data/$f" 2>&1 | grep -v colab.dev | tail -0; done
timeout 180 colab upload -s $S "$TASK2/work/esmc_L18.npy" /content/rs/data/esmc_L18.npy 2>&1 | grep -v colab.dev | tail -0
timeout 180 colab upload -s $S "$T/explorer/examples/unseen_examples.fasta" /content/rs/data/unseen_examples.fasta 2>&1 | grep -v colab.dev | tail -0
timeout 180 colab upload -s $S "$T/out/rs02/esmc_L18_new.npy" /content/rs/out/esmc_L18_new.npy 2>&1 | grep -v colab.dev | tail -0
for f in rslib.py test_rslib.py rs02_embed_new_seqs.py rs09_export_explorer_data.py; do timeout 180 colab upload -s $S "$T/colab/$f" "/content/rs/code/$f" 2>&1 | grep -v colab.dev | tail -0; done
(timeout 1500 colab exec -s $S --timeout 1440 -f "$T/colab/drv_export.py" > /tmp/rs_export_exec.log 2>&1 &)
for i in $(seq 1 200); do
  timeout 30 colab download -s $S /content/rs/out/progress.log /tmp/rs_progress_$S.log >/dev/null 2>&1
  grep -qE "EXPORT_FINISHED" /tmp/rs_progress_$S.log 2>/dev/null && break
  sleep 8
done
tail -30 /tmp/rs_progress_$S.log | cut -c1-260
for f in params.json records.bin records_meta.json seqs.bin seqs_meta.json approx.json coverage.json examples.json EXPORT_DONE; do
  timeout 180 colab download -s $S /content/rs/export/$f "$T/export/$f" 2>&1 | grep -v colab.dev | tail -0
done
cp /tmp/rs_progress_$S.log "$T/export/export_log.txt"
timeout 120 colab stop -s $S 2>&1 | grep -v colab.dev | tail -1
echo RUN_EXPORT_ALL_DONE
