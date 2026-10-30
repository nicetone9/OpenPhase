#!/bin/zsh
# 在 Colab 会话里建目录、上传输入与代码、装依赖。用法：setup.sh <会话名>
# colab exec 只接受 -f 文件；上传失败的错误 URL 里带会话代理令牌，所以输出里滤掉含 colab.dev 的行。
setopt nullglob
S=$1
T="$(cd "$(dirname "$0")/.." && pwd)"
TASK2="$T/../task2_single_component_condition"
TMP=$(mktemp -d)
cat > $TMP/mk.py <<'PY'
import os
for d in ('/content/rs/data', '/content/rs/code', '/content/rs/out'):
    os.makedirs(d, exist_ok=True)
print('dirs ok')
PY
timeout 120 colab exec -s $S --timeout 60 -f $TMP/mk.py 2>&1 | tail -1
for f in rs_records.tsv rs_sequences.tsv rs_new_sequences.tsv; do
  timeout 180 colab upload -s $S "$T/data/$f" "/content/rs/data/$f" 2>&1 | grep -v colab.dev || true
done
timeout 180 colab upload -s $S "$TASK2/work/esmc_L18.npy" /content/rs/data/esmc_L18.npy 2>&1 | grep -v colab.dev || true
for f in rslib.py test_rslib.py rs02_embed_new_seqs.py rs03_run.py; do
  timeout 180 colab upload -s $S "$T/colab/$f" "/content/rs/code/$f" 2>&1 | grep -v colab.dev || true
done
echo "uploads done"
